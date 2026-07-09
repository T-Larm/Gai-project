using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.Networking;

namespace GaiNpc
{
    /// <summary>
    /// Behavior channel: polls POST /act every tick with the NPC's current
    /// game state. The trained policy returns an action instantly; the bark
    /// line arrives in the same response (~1 s) — play the action animation
    /// immediately and show the bark whenever it lands.
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
        public string npcName = "Aldric";

        [Header("Polling")]
        [Tooltip("Seconds between /act decision ticks")]
        public float tickSeconds = 5f;
        public bool requestBark = true;
        [Tooltip("Also synthesize bark audio (XTTS, adds latency)")]
        public bool speak = false;

        [Header("Events")]
        public UnityEvent<string> OnActionChanged;
        public UnityEvent<string> OnMoodChanged;
        public UnityEvent<string> OnBark;
        public UnityEvent OnShouldTalk;

        [Header("Optional")]
        [Tooltip("AudioSource for bark voice when speak=true")]
        public AudioSource voiceSource;

        /// <summary>Replace to feed the real scene state; defaults to a demo state.</summary>
        public Func<NpcGameState> StateProvider;

        public string CurrentAction { get; private set; } = "";
        public string CurrentMood { get; private set; } = "";

        [Serializable]
        private class ActResponse
        {
            public string npc;
            public string action_id;
            public string mood;
            public bool should_talk;
            public string bark;
            public string audio_base64;
            public int sample_rate;
        }

        private void Start()
        {
            if (StateProvider == null)
            {
                StateProvider = DefaultDemoState;
            }
            StartCoroutine(TickLoop());
        }

        private IEnumerator TickLoop()
        {
            var wait = new WaitForSeconds(tickSeconds);
            while (true)
            {
                yield return RequestAct();
                yield return wait;
            }
        }

        private IEnumerator RequestAct()
        {
            string body = "{\"npc\":\"" + npcName + "\"," +
                          "\"game_state\":" + StateProvider().ToJson() + "," +
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
                    Debug.LogWarning($"[NpcBehaviorClient] /act failed: {request.error}");
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
            if (response.should_talk)
            {
                OnShouldTalk?.Invoke();
            }
            if (speak && voiceSource != null && !string.IsNullOrEmpty(response.audio_base64))
            {
                var clip = WavUtility.FromBase64Wav(response.audio_base64, "bark");
                if (clip != null)
                {
                    voiceSource.clip = clip;
                    voiceSource.Play();
                }
            }
        }

        /// <summary>Thirsty-at-the-forge demo state so the wiring can be tested
        /// before any real scene sensing exists. Expected action: drink.</summary>
        private NpcGameState DefaultDemoState()
        {
            var state = new NpcGameState
            {
                occ = "Blacksmith",
                arch = "Gruff",
                thi = 0.92f,
                schedAct = "work",
                hour = Mathf.Repeat(Time.time / 60f + 12f, 24f),
            };
            state.traits.Add("Gruff");
            state.traits.Add("Honest");
            state.inventory.Add(new NpcGameState.Item { id = "water", n = 2 });
            return state;
        }
    }
}
