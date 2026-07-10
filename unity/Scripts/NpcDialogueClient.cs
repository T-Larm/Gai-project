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
        public string npcName = "Aldric";

        [Tooltip("Also synthesize reply audio (XTTS, seconds of extra latency)")]
        public bool speak = false;

        [Tooltip("AudioSource for the reply voice when speak=true")]
        public AudioSource voiceSource;

        [Serializable]
        private class ChatResponse
        {
            public string npc;
            public string reply;
            public GuardInfo guard;
            public string audio_base64;
            public int sample_rate;
        }

        [Serializable]
        private class GuardInfo
        {
            public string reason;
        }

        /// <summary>
        /// Send one player line. onReply(replyText, guardReason) — guardReason
        /// is null for normal turns, "secret_low_trust" / "prompt_injection"
        /// when the rule guard constrained the reply (useful for UI effects,
        /// e.g. the NPC narrows their eyes).
        /// </summary>
        public void Send(string playerText, Action<string, string> onReply)
        {
            StartCoroutine(SendCoroutine(playerText, onReply));
        }

        private IEnumerator SendCoroutine(string playerText, Action<string, string> onReply)
        {
            string body = "{\"npc\":\"" + npcName + "\"," +
                          "\"text\":\"" + Escape(playerText) + "\"," +
                          "\"speak\":" + (speak ? "true" : "false") + "}";

            using (var request = new UnityWebRequest(serverUrl + "/chat", "POST"))
            {
                request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body));
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");
                yield return request.SendWebRequest();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogWarning($"[NpcDialogueClient] /chat failed: {request.error}");
                    onReply?.Invoke(null, null);
                    yield break;
                }

                var response = JsonUtility.FromJson<ChatResponse>(request.downloadHandler.text);
                string guardReason = null;
                if (response.guard != null && !string.IsNullOrEmpty(response.guard.reason))
                {
                    guardReason = response.guard.reason;
                }
                onReply?.Invoke(response.reply, guardReason);

                if (speak && voiceSource != null && !string.IsNullOrEmpty(response.audio_base64))
                {
                    var clip = WavUtility.FromBase64Wav(response.audio_base64, "reply");
                    if (clip != null)
                    {
                        voiceSource.clip = clip;
                        voiceSource.Play();
                    }
                }
            }
        }

        [Serializable]
        private class ChatStreamStart
        {
            public string npc;
            public string session_id;
        }

        [Serializable]
        private class StreamChunk
        {
            public int index;
            public string text;
            public string audio_base64;
            public int sample_rate;
        }

        [Serializable]
        private class ChatStreamPoll
        {
            public StreamChunk[] chunks;
            public bool done;
            public string error;
            public GuardInfo guard;
        }

        private readonly Queue<AudioClip> _clipQueue = new Queue<AudioClip>();
        private Coroutine _playbackLoop;

        /// <summary>
        /// Sentence-streamed variant of Send(): onSentence fires per sentence
        /// as it arrives (guardReason at most once, on the first guarded
        /// chunk), audio chunks play back-to-back from a queue, onComplete
        /// fires when the server finishes the reply.
        /// </summary>
        public void SendStreaming(string playerText,
                                  Action<string, string> onSentence,
                                  Action onComplete)
        {
            StartCoroutine(SendStreamingCoroutine(playerText, onSentence, onComplete));
        }

        private IEnumerator SendStreamingCoroutine(string playerText,
                                                   Action<string, string> onSentence,
                                                   Action onComplete)
        {
            string body = "{\"npc\":\"" + npcName + "\"," +
                          "\"text\":\"" + Escape(playerText) + "\"," +
                          "\"speak\":" + (speak ? "true" : "false") + "}";

            string sessionId = null;
            using (var request = new UnityWebRequest(serverUrl + "/chat_stream", "POST"))
            {
                request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body));
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");
                yield return request.SendWebRequest();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogWarning($"[NpcDialogueClient] /chat_stream failed: {request.error}");
                    onSentence?.Invoke(null, null);
                    onComplete?.Invoke();
                    yield break;
                }
                sessionId = JsonUtility.FromJson<ChatStreamStart>(
                    request.downloadHandler.text).session_id;
            }

            int after = -1;
            bool guardReported = false;
            while (true)
            {
                using (var poll = UnityWebRequest.Get(
                           serverUrl + "/chat_stream/" + sessionId + "?after=" + after))
                {
                    yield return poll.SendWebRequest();
                    if (poll.result != UnityWebRequest.Result.Success)
                    {
                        Debug.LogWarning($"[NpcDialogueClient] stream poll failed: {poll.error}");
                        break;
                    }

                    var state = JsonUtility.FromJson<ChatStreamPoll>(poll.downloadHandler.text);
                    if (state.chunks != null)
                    {
                        foreach (var chunk in state.chunks)
                        {
                            after = chunk.index;
                            string guardReason = null;
                            if (!guardReported && state.guard != null &&
                                !string.IsNullOrEmpty(state.guard.reason))
                            {
                                guardReason = state.guard.reason;
                                guardReported = true;
                            }
                            onSentence?.Invoke(chunk.text, guardReason);

                            if (speak && voiceSource != null &&
                                !string.IsNullOrEmpty(chunk.audio_base64))
                            {
                                var clip = WavUtility.FromBase64Wav(
                                    chunk.audio_base64, $"reply_{chunk.index}");
                                if (clip != null)
                                {
                                    _clipQueue.Enqueue(clip);
                                    if (_playbackLoop == null)
                                    {
                                        _playbackLoop = StartCoroutine(PlaybackLoop());
                                    }
                                }
                            }
                        }
                    }

                    if (!string.IsNullOrEmpty(state.error))
                    {
                        Debug.LogWarning($"[NpcDialogueClient] stream error: {state.error}");
                    }
                    if (state.done) break;
                }
                yield return new WaitForSeconds(0.25f);
            }
            onComplete?.Invoke();
        }

        /// <summary>Play queued sentence clips back-to-back on voiceSource.</summary>
        private IEnumerator PlaybackLoop()
        {
            while (_clipQueue.Count > 0)
            {
                var clip = _clipQueue.Dequeue();
                voiceSource.clip = clip;
                voiceSource.Play();
                // Poll instead of waiting clip.length: a new Play() call or a
                // closed dialogue can stop the source early.
                while (voiceSource.isPlaying)
                {
                    yield return null;
                }
            }
            _playbackLoop = null;
        }

        private static string Escape(string s)
        {
            return (s ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"")
                            .Replace("\n", "\\n").Replace("\r", "");
        }
    }
}
