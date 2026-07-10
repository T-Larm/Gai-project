using UnityEngine;

namespace GaiNpc
{
    /// <summary>
    /// Proximity interaction: when the player is within interactRadius,
    /// shows "[E] Talk to ..." and opens the DialogueUI on key press.
    /// Lives on the NPC next to NpcDialogueClient.
    /// </summary>
    public class NpcTalkTrigger : MonoBehaviour
    {
        public float interactRadius = 3f;
        public KeyCode key = KeyCode.E;

        private NpcDialogueClient _dialogue;
        private Transform _player;
        private bool _inRange;

        void Start()
        {
            _dialogue = GetComponent<NpcDialogueClient>();
            var cc = FindObjectOfType<CharacterController>();
            if (cc != null) _player = cc.transform;
        }

        void Update()
        {
            if (_player == null || _dialogue == null || DialogueUI.Instance == null) return;
            if (DialogueUI.Instance.IsOpen)
            {
                _inRange = false;
                return;
            }

            var nowInRange =
                Vector3.Distance(_player.position, transform.position) <= interactRadius;
            if (nowInRange && !_inRange)
            {
                DialogueUI.Instance.ShowHint($"[E]  Talk to {_dialogue.npcName}");
            }
            else if (!nowInRange && _inRange)
            {
                DialogueUI.Instance.HideHint();
            }
            _inRange = nowInRange;

            if (_inRange && Input.GetKeyDown(key))
            {
                DialogueUI.Instance.Open(_dialogue);
            }
        }
    }
}
