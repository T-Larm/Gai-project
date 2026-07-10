using UnityEngine;
using UnityEngine.UI;

namespace GaiNpc
{
    /// <summary>
    /// Minimal glue between NpcBehaviorClient events and a world-space
    /// speech-bubble Text. Drop on the NPC, assign the client and a Text,
    /// press Play — no other wiring needed (events are hooked in code).
    /// </summary>
    public class BarkBubble : MonoBehaviour
    {
        public NpcBehaviorClient npc;
        public Text bubble;

        [Tooltip("Seconds the bark stays visible; 0 keeps it forever")]
        public float hideAfterSeconds = 6f;

        void Start()
        {
            if (npc == null) npc = GetComponent<NpcBehaviorClient>();

            npc.OnActionChanged.AddListener(a => Debug.Log($"[{npc.npcName}] action: {a}"));
            npc.OnMoodChanged.AddListener(m => Debug.Log($"[{npc.npcName}] mood: {m}"));
            npc.OnBark.AddListener(ShowBark);
            npc.OnShouldTalk.AddListener(() => Debug.Log($"[{npc.npcName}] wants to talk — open dialogue UI here"));
        }

        void LateUpdate()
        {
            // Billboard: keep the bubble readable no matter how the NPC
            // (its parent) is rotated or where the camera moves.
            if (bubble == null || Camera.main == null) return;
            var canvasT = bubble.canvas.transform;
            canvasT.rotation = Quaternion.LookRotation(
                canvasT.position - Camera.main.transform.position);
        }

        void ShowBark(string line)
        {
            if (bubble == null) return;
            bubble.text = line;
            if (hideAfterSeconds > 0f)
            {
                CancelInvoke(nameof(ClearBubble));
                Invoke(nameof(ClearBubble), hideAfterSeconds);
            }
        }

        void ClearBubble()
        {
            bubble.text = "";
        }
    }
}
