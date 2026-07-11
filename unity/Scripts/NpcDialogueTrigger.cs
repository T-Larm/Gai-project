using System.Collections.Generic;
using UnityEngine;

namespace GaiNpc
{
    /// <summary>
    /// Attach to an NPC with a trigger Collider. When the Player enters the
    /// trigger, pressing E opens the shared NpcDialogueUI for this NPC.
    /// </summary>
    public class NpcDialogueTrigger : MonoBehaviour
    {
        public NpcDialogueClient dialogueClient;
        public NpcBehaviorClient behaviorClient;
        public NpcDialogueUI dialogueUI;
        public GameObject interactionPrompt;
        public KeyCode interactionKey = KeyCode.E;
        [Min(0.5f)] public float interactionDistance = 5f;

        private static readonly List<NpcDialogueTrigger> Instances =
            new List<NpcDialogueTrigger>();
        private static int handledInteractionFrame = -1;

        private Transform player;

        private void Awake()
        {
            if (dialogueClient == null) dialogueClient = GetComponentInParent<NpcDialogueClient>();
            if (behaviorClient == null) behaviorClient = GetComponentInParent<NpcBehaviorClient>();
            if (dialogueUI == null) dialogueUI = FindObjectOfType<NpcDialogueUI>();
            if (interactionPrompt != null) interactionPrompt.SetActive(false);
        }

        private void OnEnable()
        {
            if (!Instances.Contains(this)) Instances.Add(this);
            if (behaviorClient != null)
                behaviorClient.OnShouldTalk.AddListener(ShowPrompt);
        }

        private void OnDisable()
        {
            Instances.Remove(this);
            if (behaviorClient != null)
                behaviorClient.OnShouldTalk.RemoveListener(ShowPrompt);
            if (interactionPrompt != null) interactionPrompt.SetActive(false);
        }

        private void Update()
        {
            ResolvePlayer();
            if (player == null || dialogueUI == null) return;

            NpcDialogueTrigger best = FindBestCandidate(player);

            if (interactionPrompt != null)
                interactionPrompt.SetActive(best == this && !dialogueUI.IsOpen);

            if (dialogueUI.IsOpen || handledInteractionFrame == Time.frameCount ||
                !Input.GetKeyDown(interactionKey) || best == null)
                return;

            handledInteractionFrame = Time.frameCount;
            best.dialogueUI.Open(best.dialogueClient);
        }

        private void OnTriggerEnter(Collider other)
        {
            if (!IsPlayer(other)) return;
            player = other.transform.root;
            ShowPrompt();
        }

        private void OnTriggerExit(Collider other)
        {
            if (!IsPlayer(other)) return;
            if (interactionPrompt != null) interactionPrompt.SetActive(false);
        }

        private void ShowPrompt()
        {
            ResolvePlayer();
            if (interactionPrompt != null && player != null)
                interactionPrompt.SetActive(FindBestCandidate(player) == this);
        }

        private void ResolvePlayer()
        {
            if (player != null) return;

            NpcSceneStateProvider state = dialogueClient != null
                ? dialogueClient.GetComponent<NpcSceneStateProvider>()
                : null;
            if (state != null && state.player != null)
            {
                player = state.player;
                return;
            }

            GameObject taggedPlayer = GameObject.FindGameObjectWithTag("Player");
            if (taggedPlayer != null)
            {
                player = taggedPlayer.transform;
                return;
            }

            GameObject controller = GameObject.Find("Controller");
            if (controller != null) player = controller.transform;
        }

        private static NpcDialogueTrigger FindBestCandidate(Transform playerTransform)
        {
            NpcDialogueTrigger best = null;
            float bestScore = float.PositiveInfinity;
            Camera camera = Camera.main;

            foreach (NpcDialogueTrigger candidate in Instances)
            {
                if (candidate == null || !candidate.isActiveAndEnabled ||
                    candidate.dialogueClient == null || candidate.dialogueUI == null)
                    continue;

                Vector3 npcPosition = candidate.dialogueClient.transform.position;
                Vector3 planarOffset = npcPosition - playerTransform.position;
                planarOffset.y = 0f;
                float distance = planarOffset.magnitude;
                if (distance > candidate.interactionDistance) continue;

                float centerPenalty = 0f;
                if (camera != null)
                {
                    Vector3 viewport = camera.WorldToViewportPoint(
                        npcPosition + Vector3.up * 1.5f);
                    if (viewport.z <= 0f) continue;
                    float x = viewport.x - 0.5f;
                    float y = viewport.y - 0.5f;
                    centerPenalty = (x * x + y * y) * 12f;
                }

                float score = distance + centerPenalty;
                if (score >= bestScore) continue;
                bestScore = score;
                best = candidate;
            }

            return best;
        }

        private static bool IsPlayer(Collider other)
        {
            return other.CompareTag("Player") || other.transform.root.CompareTag("Player");
        }
    }
}
