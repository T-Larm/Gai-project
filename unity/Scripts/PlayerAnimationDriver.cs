using UnityEngine;

namespace GaiNpc
{
    /// <summary>
    /// Drives the player body's walk animation from the CharacterController's
    /// actual horizontal velocity, so WASD movement plays walk/run instead of
    /// gliding, and turns the body to face the direction actually walked.
    /// The Suntail rig is a first-person controller: only the mouse rotates
    /// it, WASD strafes — without this the body stays glued to the camera
    /// direction and side-steps like a crab. Lives on the "Body" model under
    /// the Suntail Controller rig; added by "GAI NPC → Setup Walk Animations".
    /// </summary>
    [RequireComponent(typeof(Animator))]
    public class PlayerAnimationDriver : MonoBehaviour
    {
        private static readonly int SpeedParam = Animator.StringToHash("Speed");

        [Tooltip("How fast the body turns toward the walk direction (deg/s)")]
        public float turnSpeed = 720f;

        private Animator _animator;
        private CharacterController _controller;
        private bool _hasSpeedParam;

        void Awake()
        {
            _animator = GetComponent<Animator>();
            _animator.applyRootMotion = false;   // the rig moves the transform
            _controller = GetComponentInParent<CharacterController>();
            foreach (var p in _animator.parameters)
            {
                if (p.nameHash == SpeedParam) { _hasSpeedParam = true; break; }
            }
        }

        void Update()
        {
            if (_controller == null) return;
            var v = _controller.velocity;
            v.y = 0f;
            if (_hasSpeedParam) _animator.SetFloat(SpeedParam, v.magnitude);

            // Rotating this child transform never affects the parent rig's
            // CharacterController or camera; while idle the body just keeps
            // its last facing (and still follows mouse turns via the parent).
            if (v.sqrMagnitude > 0.04f)
            {
                transform.rotation = Quaternion.RotateTowards(
                    transform.rotation,
                    Quaternion.LookRotation(v),
                    turnSpeed * Time.deltaTime);
            }
        }
    }
}
