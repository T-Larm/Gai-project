using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace GaiNpc
{
    /// <summary>
    /// Dialogue channel: POST /chat for full LLM conversation (three-layer
    /// persona + memory + rule guard). Open this UI when the player initiates
    /// talk or NpcBehaviorClient fires OnShouldTalk.
    /// </summary>
    public class NpcDialogueClient : MonoBehaviour
    {
        [Header("Server")]
        public string serverUrl = "http://127.0.0.1:8000";
        public string npcName = "Nicole";

        [Tooltip("Also synthesize reply audio (XTTS, seconds of extra latency)")]
        public bool speak = false;

        [Tooltip("AudioSource for the reply voice when speak=true")]
        public AudioSource voiceSource;

        [Header("Sentence Pipeline")]
        public bool useSentencePipeline = true;
        [Min(0.05f)] public float pipelinePollSeconds = 0.12f;

        public bool IsBusy { get; private set; }

        [Serializable]
        private class ChatRequest
        {
            public string npc;
            public string text;
            public bool speak;
        }

        [Serializable]
        private class ChatResponse
        {
            public string reply = "";
            public GuardInfo guard = null;
            public string audio_base64 = "";
        }

        [Serializable]
        private class GuardInfo
        {
            public string reason = "";
        }

        [Serializable]
        private class PipelineStartResponse
        {
            public string job_id = "";
        }

        [Serializable]
        private class PipelinePollResponse
        {
            public PipelineEvent[] events = Array.Empty<PipelineEvent>();
            public int next;
            public bool done;
            public string error = "";
        }

        [Serializable]
        private class PipelineEvent
        {
            public int seq;
            public string type = "";
            public int sentence_index;
            public string text = "";
            public string audio_base64 = "";
            public string reply = "";
            public string guard_reason = "";
            public string message = "";
        }

        private readonly Queue<AudioClip> pendingVoice = new Queue<AudioClip>();
        private Coroutine voicePlayback;

        /// <summary>
        /// Send one player line. onReply(replyText, guardReason) — guardReason
        /// is null for normal turns, "secret_low_trust" / "prompt_injection"
        /// when the rule guard constrained the reply (useful for UI effects,
        /// e.g. the NPC narrows their eyes).
        /// </summary>
        public void Send(
            string playerText,
            Action<string, string> onReply,
            Action<string> onSentence = null)
        {
            if (IsBusy || string.IsNullOrWhiteSpace(playerText)) return;
            if (useSentencePipeline)
                StartCoroutine(SendPipelineCoroutine(playerText, onReply, onSentence));
            else
                StartCoroutine(SendCoroutine(playerText, onReply));
        }

        private IEnumerator SendPipelineCoroutine(
            string playerText,
            Action<string, string> onReply,
            Action<string> onSentence)
        {
            IsBusy = true;
            EnsureVoiceSource();

            string body = JsonUtility.ToJson(new ChatRequest
            {
                npc = npcName,
                text = playerText,
                speak = speak
            });

            string jobId;
            using (var request = new UnityWebRequest(serverUrl + "/chat/pipeline", "POST"))
            {
                request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body));
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");
                yield return request.SendWebRequest();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    FailPipeline(request, "start");
                    IsBusy = false;
                    onReply?.Invoke(null, null);
                    yield break;
                }

                PipelineStartResponse start =
                    JsonUtility.FromJson<PipelineStartResponse>(request.downloadHandler.text);
                jobId = start != null ? start.job_id : "";
                if (string.IsNullOrEmpty(jobId))
                {
                    Debug.LogWarning("[NpcDialogueClient] Pipeline returned no job id.");
                    IsBusy = false;
                    onReply?.Invoke(null, null);
                    yield break;
                }
            }

            int cursor = 0;
            string finalReply = "";
            string guardReason = "";
            bool pipelineFailed = false;
            bool done = false;

            while (!done)
            {
                string pollUrl = serverUrl + "/chat/pipeline/" + jobId + "?after=" + cursor;
                using (var request = UnityWebRequest.Get(pollUrl))
                {
                    yield return request.SendWebRequest();
                    if (request.result != UnityWebRequest.Result.Success)
                    {
                        FailPipeline(request, "poll");
                        pipelineFailed = true;
                        break;
                    }

                    PipelinePollResponse poll =
                        JsonUtility.FromJson<PipelinePollResponse>(request.downloadHandler.text);
                    if (poll == null)
                    {
                        Debug.LogWarning("[NpcDialogueClient] Pipeline returned invalid JSON.");
                        pipelineFailed = true;
                        break;
                    }

                    if (poll.events != null)
                    {
                        foreach (PipelineEvent pipelineEvent in poll.events)
                        {
                            if (pipelineEvent.type == "sentence")
                            {
                                onSentence?.Invoke(pipelineEvent.text);
                            }
                            else if (pipelineEvent.type == "audio")
                            {
                                QueueVoice(pipelineEvent.audio_base64, pipelineEvent.sentence_index);
                            }
                            else if (pipelineEvent.type == "audio_error")
                            {
                                Debug.LogWarning(
                                    $"[NpcDialogueClient] XTTS sentence failed: {pipelineEvent.message}");
                            }
                            else if (pipelineEvent.type == "done")
                            {
                                finalReply = pipelineEvent.reply;
                                guardReason = pipelineEvent.guard_reason;
                            }
                            else if (pipelineEvent.type == "error")
                            {
                                Debug.LogWarning(
                                    $"[NpcDialogueClient] Pipeline failed: {pipelineEvent.message}");
                                pipelineFailed = true;
                            }
                        }
                    }

                    cursor = poll.next;
                    done = poll.done;
                    if (!string.IsNullOrEmpty(poll.error)) pipelineFailed = true;
                }

                if (!done && !pipelineFailed)
                    yield return new WaitForSecondsRealtime(Mathf.Max(0.05f, pipelinePollSeconds));
                if (pipelineFailed) break;
            }

            IsBusy = false;
            onReply?.Invoke(pipelineFailed ? null : finalReply, guardReason);
        }

        private IEnumerator SendCoroutine(string playerText, Action<string, string> onReply)
        {
            IsBusy = true;
            EnsureVoiceSource();

            string body = JsonUtility.ToJson(new ChatRequest
            {
                npc = npcName,
                text = playerText,
                speak = speak
            });

            using (var request = new UnityWebRequest(serverUrl + "/chat", "POST"))
            {
                request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body));
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");
                yield return request.SendWebRequest();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    string details = request.downloadHandler != null
                        ? request.downloadHandler.text
                        : "";
                    Debug.LogWarning(
                        $"[NpcDialogueClient] /chat failed ({request.responseCode}): " +
                        $"{request.error} {details}");
                    IsBusy = false;
                    onReply?.Invoke(null, null);
                    yield break;
                }

                var response = JsonUtility.FromJson<ChatResponse>(request.downloadHandler.text);
                if (response == null)
                {
                    Debug.LogWarning("[NpcDialogueClient] /chat returned invalid JSON.");
                    IsBusy = false;
                    onReply?.Invoke(null, null);
                    yield break;
                }
                string guardReason = null;
                if (response.guard != null && !string.IsNullOrEmpty(response.guard.reason))
                {
                    guardReason = response.guard.reason;
                }
                if (speak && voiceSource != null && !string.IsNullOrEmpty(response.audio_base64))
                {
                    var clip = WavUtility.FromBase64Wav(response.audio_base64, "reply");
                    if (clip != null)
                    {
                        voiceSource.clip = clip;
                        voiceSource.Play();
                    }
                }

                IsBusy = false;
                onReply?.Invoke(response.reply, guardReason);
            }
        }

        private void EnsureVoiceSource()
        {
            if (!speak || voiceSource != null) return;

            voiceSource = GetComponent<AudioSource>();
            if (voiceSource == null)
            {
                voiceSource = gameObject.AddComponent<AudioSource>();
                voiceSource.playOnAwake = false;
                voiceSource.spatialBlend = 1f;
                voiceSource.minDistance = 2f;
                voiceSource.maxDistance = 15f;
            }
        }

        private void QueueVoice(string audioBase64, int sentenceIndex)
        {
            if (!speak || voiceSource == null || string.IsNullOrEmpty(audioBase64)) return;

            AudioClip clip = WavUtility.FromBase64Wav(
                audioBase64, $"{npcName}-reply-{sentenceIndex}");
            if (clip == null) return;
            pendingVoice.Enqueue(clip);
            if (voicePlayback == null)
                voicePlayback = StartCoroutine(PlayVoiceQueue());
        }

        private IEnumerator PlayVoiceQueue()
        {
            while (pendingVoice.Count > 0)
            {
                AudioClip clip = pendingVoice.Dequeue();
                voiceSource.clip = clip;
                voiceSource.Play();
                yield return new WaitWhile(() => voiceSource != null && voiceSource.isPlaying);
                Destroy(clip);
            }
            voicePlayback = null;
        }

        private static void FailPipeline(UnityWebRequest request, string stage)
        {
            string details = request.downloadHandler != null
                ? request.downloadHandler.text
                : "";
            Debug.LogWarning(
                $"[NpcDialogueClient] Pipeline {stage} failed ({request.responseCode}): " +
                $"{request.error} {details}");
        }

    }
}
