using System.Collections.Generic;
using UnityEngine;

namespace GaiNpc
{
    /// <summary>
    /// Proximity interaction: when the player is within interactRadius,
    /// shows "[E] Talk to ..." and opens the DialogueUI on key press.
    /// Lives on the NPC next to NpcDialogueClient.
    /// Only the trigger nearest to the player responds — NPCs stand close
    /// enough that interact radii overlap, and without this every trigger
    /// in range would open the dialogue, with the last Update() winning.
    /// </summary>
    public class NpcTalkTrigger : MonoBehaviour
    {
        public float interactRadius = 3f;
        public KeyCode key = KeyCode.E;

        private static readonly List<NpcTalkTrigger> All = new List<NpcTalkTrigger>();

        private NpcDialogueClient _dialogue;
        private Transform _player;
        private bool _active;

        void OnEnable()
        {
            All.Add(this);
        }

        void OnDisable()
        {
            All.Remove(this);
            if (_active && DialogueUI.Instance != null) DialogueUI.Instance.HideHint();
            _active = false;
        }

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
                _active = false;
                return;
            }

            var nowActive = IsNearestInRange();
            if (nowActive)
            {
                // Re-show every frame: another trigger's HideHint may have
                // cleared the text after our ShowHint ran this frame.
                DialogueUI.Instance.ShowHint($"[E]  Talk to {_dialogue.npcName}");
            }
            else if (_active)
            {
                DialogueUI.Instance.HideHint();
            }
            _active = nowActive;

            if (_active && Input.GetKeyDown(key))
            {
                DialogueUI.Instance.Open(_dialogue);
            }
        }

        private bool IsNearestInRange()
        {
            var myDist = Vector3.Distance(_player.position, transform.position);
            if (myDist > interactRadius) return false;

            foreach (var other in All)
            {
                if (other == this || other._player == null) continue;
                var d = Vector3.Distance(other._player.position, other.transform.position);
                if (d > other.interactRadius) continue;
                if (d < myDist) return false;
                // Distance tie: lowest instance ID wins so exactly one opens.
                if (d == myDist && other.GetInstanceID() < GetInstanceID()) return false;
            }
            return true;
        }
    }
}
