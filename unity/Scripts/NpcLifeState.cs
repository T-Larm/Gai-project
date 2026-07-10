using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

namespace GaiNpc
{
    /// <summary>
    /// Per-NPC live simulation state, replacing NpcBehaviorClient's hardcoded
    /// thirsty-blacksmith demo state. Each NPC gets its own drifting needs
    /// (thirst/hunger/energy), a shared village clock, a Social percept when
    /// the player stands nearby, and identity (occupation/traits) fetched from
    /// GET /npc/{name} so Unity never duplicates the persona files.
    ///
    /// The behavior loop closes here: when the policy picks an action, the
    /// action's effect is applied back to the state (drink empties thirst and
    /// a water item, gather restocks, sleep restores energy), so the next tick
    /// faces a genuinely new situation instead of the same frozen snapshot.
    /// Added automatically by "GAI NPC → Setup Character NPCs".
    /// </summary>
    [RequireComponent(typeof(NpcBehaviorClient))]
    public class NpcLifeState : MonoBehaviour
    {
        [Header("Identity (fetched from /npc/{name}; fallbacks until then)")]
        public string occupation = "Villager";
        public string faction = "";
        public List<string> traits = new List<string>();

        [Header("Needs (0..1; drift up, actions push them back down)")]
        [Range(0f, 1f)] public float thirst;
        [Range(0f, 1f)] public float hunger;
        [Range(0f, 1f)] public float energy = 0.8f;
        [Tooltip("Per-second drift; per-NPC jitter is applied on top in Awake")]
        public float thirstPerSecond = 0.0040f;
        public float hungerPerSecond = 0.0025f;
        public float fatiguePerSecond = 0.0012f;

        [Header("Clock (shared by all NPCs; 1 game hour per realMinutesPerHour)")]
        public float realSecondsPerGameHour = 60f;

        [Header("Percepts")]
        [Tooltip("Player within this range shows up as a Social percept")]
        public float socialRange = 6f;

        public int waterItems = 2;
        public int foodItems = 2;
        public float happiness = 0.5f;

        private NpcBehaviorClient _client;
        private static float _clockStartHour = 10f;   // village morning
        private float _driftScale = 1f;

        void Awake()
        {
            _client = GetComponent<NpcBehaviorClient>();

            // Deterministic per-NPC variation (stable string hash — GetHashCode
            // isn't guaranteed stable), so the six NPCs desync from the first
            // tick: different starting needs, different drift speeds.
            int seed = 17;
            foreach (char c in _client.npcName) seed = seed * 31 + c;
            var rng = new System.Random(seed);
            thirst = (float)rng.NextDouble() * 0.6f;
            hunger = (float)rng.NextDouble() * 0.6f;
            energy = 0.6f + (float)rng.NextDouble() * 0.4f;
            _driftScale = 0.8f + (float)rng.NextDouble() * 0.4f;

            _client.StateProvider = BuildState;
            _client.OnActionChanged.AddListener(ApplyAction);
        }

        void Start()
        {
            StartCoroutine(FetchIdentity());
        }

        void Update()
        {
            float dt = Time.deltaTime * _driftScale;
            thirst = Mathf.Clamp01(thirst + thirstPerSecond * dt);
            hunger = Mathf.Clamp01(hunger + hungerPerSecond * dt);
            energy = Mathf.Clamp01(energy - fatiguePerSecond * dt);
            // mood drifts back to neutral between events
            happiness = Mathf.MoveTowards(happiness, 0.5f, 0.005f * Time.deltaTime);
        }

        public float GameHour =>
            Mathf.Repeat(_clockStartHour + Time.time / Mathf.Max(1f, realSecondsPerGameHour), 24f);

        /// <summary>Close the loop: the policy's chosen action changes the state.</summary>
        public void ApplyAction(string actionId)
        {
            switch (actionId)
            {
                case "drink":
                    if (waterItems > 0) waterItems--;
                    thirst = 0.05f;
                    break;
                case "eat":
                    if (foodItems > 0) foodItems--;
                    hunger = 0.05f;
                    break;
                case "sleep":
                case "rest":
                    energy = Mathf.Max(energy, 0.9f);
                    break;
                case "gather":
                    waterItems++;
                    foodItems++;
                    break;
                case "socialize":
                    happiness = Mathf.Clamp01(happiness + 0.25f);
                    break;
            }
        }

        private NpcGameState BuildState()
        {
            float hour = GameHour;
            var state = new NpcGameState
            {
                occ = occupation,
                arch = traits.Count > 0 ? traits[0] : "Villager",
                faction = faction,
                hun = hunger,
                thi = thirst,
                en = energy,
                hap = happiness,
                hour = hour,
                schedAct = ScheduleFor(hour),
            };
            foreach (var t in traits) state.traits.Add(t);
            if (waterItems > 0) state.inventory.Add(new NpcGameState.Item { id = "water", n = waterItems });
            if (foodItems > 0) state.inventory.Add(new NpcGameState.Item { id = "food", n = foodItems });

            var player = FindObjectOfType<CharacterController>();
            if (player != null &&
                Vector3.Distance(player.transform.position, transform.position) <= socialRange)
            {
                state.percepts.Add(new NpcGameState.Percept
                {
                    id = "player", tag = "Social", threat = 0f, sal = 0.9f,
                });
            }
            return state;
        }

        private static string ScheduleFor(float hour)
        {
            if (hour >= 22f || hour < 7f) return "sleep";
            if (hour >= 9f && hour < 17f) return "work";
            return "idle";
        }

        [Serializable]
        private class NpcInfo
        {
            public string occupation;
            public string faction;
            public List<string> personality_tags;
        }

        private IEnumerator FetchIdentity()
        {
            var url = _client.serverUrl + "/npc/" + UnityWebRequest.EscapeURL(_client.npcName);
            using (var request = UnityWebRequest.Get(url))
            {
                yield return request.SendWebRequest();
                if (request.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogWarning($"[NpcLifeState] /npc/{_client.npcName} failed: {request.error}; " +
                                     "using inspector fallbacks");
                    yield break;
                }
                var info = JsonUtility.FromJson<NpcInfo>(request.downloadHandler.text);
                if (info == null) yield break;
                if (!string.IsNullOrEmpty(info.occupation)) occupation = info.occupation;
                if (!string.IsNullOrEmpty(info.faction)) faction = info.faction;
                if (info.personality_tags != null && info.personality_tags.Count > 0)
                {
                    traits = info.personality_tags;
                }
            }
        }
    }
}
