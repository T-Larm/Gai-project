using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.Networking;

namespace GaiNpc
{
    /// <summary>
    /// Behavior channel: polls POST /act every tick with the NPC's current
    /// game state. POST /act returns the trained action immediately and starts
    /// bark/voice generation in the background. The client applies the action,
    /// then polls the bark job without blocking later policy ticks.
    ///
    /// Wire in the inspector:
    ///   OnActionChanged -> your animator/pathfinding ("drink", "flee", ...)
    ///   OnBark          -> speech-bubble text (and audio if speak=true)
    ///   OnShouldTalk    -> open the dialogue UI (NpcDialogueClient)
    /// </summary>
    public class NpcBehaviorClient : MonoBehaviour
    {
        [Header("Server")]
        public string serverUrl = "http://127.0.0.1:8000";
        public string npcName = "Nicole";

        [Header("Polling")]
        [Tooltip("Seconds between /act decision ticks")]
        public float tickSeconds = 5f;
        public bool requestBark = true;
        [Min(0.05f)] public float barkPollSeconds = 0.12f;
        [Tooltip("Also synthesize bark audio (XTTS, adds latency)")]
        public bool speak = false;

        [Header("Events")]
        public UnityEvent<string> OnActionChanged = new UnityEvent<string>();
        public UnityEvent<string> OnMoodChanged = new UnityEvent<string>();
        public UnityEvent<string> OnBark = new UnityEvent<string>();
        public UnityEvent OnShouldTalk = new UnityEvent();

        [Header("Optional")]
        [Tooltip("AudioSource for bark voice when speak=true")]
        public AudioSource voiceSource;

        /// <summary>Replace to feed the real scene state; defaults to a demo state.</summary>
        public Func<NpcGameState> StateProvider;

        public string CurrentAction { get; private set; } = "";
        public string CurrentMood { get; private set; } = "";

        private Coroutine tickLoop;

        [Serializable]
        private class ActResponse
        {
            public string action_id = "";
            public string mood = "";
            public bool should_talk;
            public string bark_job_id = "";
            public string bark = "";
            public string audio_base64 = "";
        }

        [Serializable]
        private class BarkJobResponse
        {
            public bool done;
            public string error = "";
            public string bark = "";
            public string audio_base64 = "";
        }

        private readonly HashSet<string> pendingBarkJobs = new HashSet<string>();

        private void OnEnable()
        {
            if (StateProvider == null) StateProvider = DefaultDemoState;
            if (tickLoop == null) tickLoop = StartCoroutine(TickLoop());
        }

        private void OnDisable()
        {
            StopAllCoroutines();
            tickLoop = null;
            pendingBarkJobs.Clear();
        }

        private IEnumerator TickLoop()
        {
            while (true)
            {
                yield return RequestAct();
                yield return new WaitForSeconds(Mathf.Max(0.25f, tickSeconds));
            }
        }

        private IEnumerator RequestAct()
        {
            NpcGameState state = StateProvider != null ? StateProvider() : null;
            if (state == null)
            {
                Debug.LogWarning("[NpcBehaviorClient] StateProvider returned null.", this);
                yield break;
            }

            string body = "{\"npc\":\"" + EscapeJson(npcName) + "\"," +
                          "\"game_state\":" + state.ToJson() + "," +
                          "\"bark\":" + (requestBark ? "true" : "false") + "," +
                          "\"speak\":" + (speak ? "true" : "false") + "}";

            using (var request = new UnityWebRequest(serverUrl + "/act", "POST"))
            {
                request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body));
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");
                yield return request.SendWebRequest();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    string details = request.downloadHandler != null ? request.downloadHandler.text : "";
                    Debug.LogWarning(
                        $"[NpcBehaviorClient] /act failed ({request.responseCode}): " +
                        $"{request.error} {details}", this);
                    yield break;
                }

                var response = JsonUtility.FromJson<ActResponse>(request.downloadHandler.text);
                HandleResponse(response);
            }
        }

        private void HandleResponse(ActResponse response)
        {
            if (response == null) return;

            if (response.action_id != CurrentAction)
            {
                CurrentAction = response.action_id;
                OnActionChanged?.Invoke(response.action_id);
            }
            if (!string.IsNullOrEmpty(response.mood) && response.mood != CurrentMood)
            {
                CurrentMood = response.mood;
                OnMoodChanged?.Invoke(response.mood);
            }
            if (!string.IsNullOrEmpty(response.bark))
            {
                OnBark?.Invoke(response.bark);
            }
            if (!string.IsNullOrEmpty(response.bark_job_id) &&
                pendingBarkJobs.Add(response.bark_job_id))
            {
                StartCoroutine(PollBark(response.bark_job_id));
            }
            if (response.should_talk)
            {
                OnShouldTalk?.Invoke();
            }
            PlayBarkAudio(response.audio_base64);
        }

        private IEnumerator PollBark(string jobId)
        {
            while (true)
            {
                using (var request = UnityWebRequest.Get(
                    serverUrl + "/act/bark/" + UnityWebRequest.EscapeURL(jobId)))
                {
                    yield return request.SendWebRequest();
                    if (request.result != UnityWebRequest.Result.Success)
                    {
                        string details = request.downloadHandler != null
                            ? request.downloadHandler.text
                            : "";
                        Debug.LogWarning(
                            $"[NpcBehaviorClient] bark poll failed ({request.responseCode}): " +
                            $"{request.error} {details}", this);
                        break;
                    }

                    BarkJobResponse response =
                        JsonUtility.FromJson<BarkJobResponse>(request.downloadHandler.text);
                    if (response == null)
                    {
                        Debug.LogWarning("[NpcBehaviorClient] Invalid bark job response.", this);
                        break;
                    }
                    if (!response.done)
                    {
                        yield return new WaitForSecondsRealtime(
                            Mathf.Max(0.05f, barkPollSeconds));
                        continue;
                    }
                    if (!string.IsNullOrEmpty(response.error))
                    {
                        Debug.LogWarning(
                            $"[NpcBehaviorClient] bark generation failed: {response.error}", this);
                        break;
                    }
                    if (!string.IsNullOrEmpty(response.bark))
                        OnBark?.Invoke(response.bark);
                    PlayBarkAudio(response.audio_base64);
                    break;
                }
            }
            pendingBarkJobs.Remove(jobId);
        }

        private void PlayBarkAudio(string audioBase64)
        {
            if (!speak || voiceSource == null || string.IsNullOrEmpty(audioBase64)) return;
            var clip = WavUtility.FromBase64Wav(audioBase64, "bark");
            if (clip == null) return;
            voiceSource.clip = clip;
            voiceSource.Play();
        }

        private static string EscapeJson(string value)
        {
            return (value ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"")
                .Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t");
        }

        /// <summary>Thirsty demo state so the wiring can be tested before any
        /// real scene sensing exists. Expected action: drink.</summary>
        private NpcGameState DefaultDemoState()
        {
            var state = new NpcGameState
            {
                occ = "Village Steward",
                arch = "Diplomatic",
                thi = 0.92f,
                schedAct = "work",
                hour = Mathf.Repeat(Time.time / 60f + 12f, 24f),
            };
            state.traits.Add("Composed");
            state.traits.Add("Protective");
            state.inventory.Add(new NpcGameState.Item { id = "water", n = 2 });
            return state;
        }
    }
}
