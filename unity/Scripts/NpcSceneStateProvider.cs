using UnityEngine;

namespace GaiNpc
{
    /// <summary>
    /// Supplies POST /act with live scene values. Attach this beside
    /// NpcBehaviorClient and update the public fields from your game systems.
    /// </summary>
    [DefaultExecutionOrder(-100)]
    public class NpcSceneStateProvider : MonoBehaviour
    {
        [Header("References")]
        public NpcBehaviorClient client;
        public Transform player;

        [Header("Identity")]
        public string occupation = "Village Steward";
        public string archetype = "Diplomatic";
        public string faction = "";
        public string[] traits = { "Composed", "Practical", "Protective" };
        public string currentGoal = "work";

        [Header("Vitals (0-1 except HP)")]
        public float hp = 100f;
        public float hpMax = 120f;
        [Range(0f, 1f)] public float energy = 0.8f;
        [Range(0f, 1f)] public float hunger = 0.2f;
        [Range(0f, 1f)] public float thirst = 0.2f;
        [Range(0f, 1f)] public float stress = 0.2f;

        [Header("Emotion (0-1)")]
        [Range(0f, 1f)] public float happiness = 0.2f;
        [Range(0f, 1f)] public float fear = 0.1f;
        [Range(0f, 1f)] public float anger = 0.1f;
        public string mood = "Calm";

        [Header("Schedule")]
        [Range(0f, 24f)] public float hour = 12f;
        public int day = 1;
        public string scheduledAction = "work";
        public int workStart = 9;
        public int workEnd = 17;
        public int sleepAt = 22;
        public int wakeAt = 7;

        [Header("Scene Sensing")]
        public float playerSenseRadius = 6f;
        [Range(0f, 1f)] public float playerThreat = 0f;
        [Range(0f, 1f)] public float playerSalience = 0.8f;
        public bool interrupted;

        [Header("Inventory")]
        [Min(0)] public int waterCount = 1;
        [Min(0)] public int foodCount = 1;

        private void Awake()
        {
            if (client == null) client = GetComponent<NpcBehaviorClient>();
            if (client == null)
            {
                Debug.LogError("[NpcSceneStateProvider] NpcBehaviorClient is missing.", this);
                enabled = false;
                return;
            }

            client.StateProvider = BuildState;
        }

        public NpcGameState BuildState()
        {
            var state = new NpcGameState
            {
                occ = occupation,
                arch = archetype,
                faction = faction,
                goalsTop = currentGoal,
                interrupt = interrupted,
                hp = hp,
                hpMax = hpMax,
                en = energy,
                hun = hunger,
                thi = thirst,
                str = stress,
                hap = happiness,
                fear = fear,
                ang = anger,
                mood = mood,
                day = day,
                hour = hour,
                schedAct = scheduledAction,
                wkStart = workStart,
                wkEnd = workEnd,
                sleepAt = sleepAt,
                wakeAt = wakeAt
            };

            if (traits != null)
            {
                foreach (string trait in traits)
                {
                    if (!string.IsNullOrWhiteSpace(trait)) state.traits.Add(trait);
                }
            }

            if (player != null)
            {
                float distance = Vector3.Distance(transform.position, player.position);
                if (distance <= playerSenseRadius)
                {
                    state.percepts.Add(new NpcGameState.Percept
                    {
                        id = "player",
                        tag = playerThreat > 0.5f ? "Threat" : "Social",
                        threat = playerThreat,
                        sal = playerSalience
                    });
                }
            }

            if (waterCount > 0)
                state.inventory.Add(new NpcGameState.Item { id = "water", n = waterCount });
            if (foodCount > 0)
                state.inventory.Add(new NpcGameState.Item { id = "food", n = foodCount });

            return state;
        }
    }
}
