using UnityEngine;
using UnityEngine.UI;

namespace GaiNpc
{
    /// <summary>
    /// Connects NpcBehaviorClient events to a world-space speech bubble.
    /// </summary>
    public class BarkBubble : MonoBehaviour
    {
        public NpcBehaviorClient npc;
        public Text bubble;

        [Tooltip("Seconds the bark stays visible; 0 keeps it forever")]
        public float hideAfterSeconds = 6f;

        private void Awake()
        {
            if (npc == null) npc = GetComponent<NpcBehaviorClient>();
        }

        private void OnEnable()
        {
            if (npc == null) return;
            npc.OnActionChanged.AddListener(LogAction);
            npc.OnMoodChanged.AddListener(LogMood);
            npc.OnBark.AddListener(ShowBark);
            npc.OnShouldTalk.AddListener(LogShouldTalk);
        }

        private void OnDisable()
        {
            if (npc == null) return;
            npc.OnActionChanged.RemoveListener(LogAction);
            npc.OnMoodChanged.RemoveListener(LogMood);
            npc.OnBark.RemoveListener(ShowBark);
            npc.OnShouldTalk.RemoveListener(LogShouldTalk);
        }

        private void LogAction(string action) => Debug.Log($"[{npc.npcName}] action: {action}");
        private void LogMood(string mood) => Debug.Log($"[{npc.npcName}] mood: {mood}");
        private void LogShouldTalk() => Debug.Log($"[{npc.npcName}] wants to talk; open dialogue UI here");

        private void ShowBark(string line)
        {
            if (bubble == null) return;
            bubble.text = line;
            if (hideAfterSeconds > 0f)
            {
                CancelInvoke(nameof(ClearBubble));
                Invoke(nameof(ClearBubble), hideAfterSeconds);
            }
        }

        private void ClearBubble()
        {
            if (bubble != null) bubble.text = "";
        }
    }
}
