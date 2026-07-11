using UnityEngine;
using UnityEngine.AI;

namespace GaiNpc
{
    /// <summary>
    /// Turns the policy's action ids into actual movement on the baked NavMesh
    /// (menu "GAI NPC → Bake NavMesh" must have been run once for the scene).
    ///
    ///   walk_to / gather / work  -> wander to random points near the spawn spot
    ///   flee                     -> run directly away from the player, faster
    ///   socialize                -> walk toward the player, stop at chat range
    ///   everything else          -> stand still (eat/drink/sleep/heal/pray/attack)
    ///
    /// While any dialogue panel is open the whole village freezes (same rule as
    /// the /act pause in NpcBehaviorClient) and nearby NPCs turn to face the
    /// player. Added automatically by "GAI NPC → Setup Character NPCs".
    /// </summary>
    [RequireComponent(typeof(NpcBehaviorClient))]
    [RequireComponent(typeof(NavMeshAgent))]
    public class NpcLocomotion : MonoBehaviour
    {
        [Header("Speeds (m/s)")]
        public float walkSpeed = 1.2f;
        public float fleeSpeed = 3.0f;

        [Header("Wander (walk_to / gather / work)")]
        [Tooltip("Random destinations stay within this radius of the spawn point")]
        public float wanderRadius = 8f;
        [Tooltip("Seconds to stand at a destination before picking the next one")]
        public float pauseAtDestination = 2.5f;

        [Header("Flee")]
        public float fleeDistance = 10f;

        [Header("Approach (socialize)")]
        public float approachStopDistance = 2f;

        [Header("Facing")]
        [Tooltip("Turn toward the player during dialogue if within this range")]
        public float faceRange = 5f;

        private enum Mode { Idle, Wander, Flee, Approach }

        private static readonly int SpeedParam = Animator.StringToHash("Speed");

        private NpcBehaviorClient _client;
        private NavMeshAgent _agent;
        private Animator _animator;
        private bool _hasSpeedParam;
        private Transform _player;
        private Vector3 _home;
        private Mode _mode = Mode.Idle;
        private float _arrivedAt = -1f;
        private float _nextApproachRepath;

        void Awake()
        {
            _client = GetComponent<NpcBehaviorClient>();
            _agent = GetComponent<NavMeshAgent>();
            _client.OnActionChanged.AddListener(OnAction);

            // Locomotion clips carry root motion; the agent owns the transform,
            // so the animator must never also move it.
            _animator = GetComponentInChildren<Animator>();
            if (_animator != null)
            {
                _animator.applyRootMotion = false;
                foreach (var p in _animator.parameters)
                {
                    if (p.nameHash == SpeedParam) { _hasSpeedParam = true; break; }
                }
            }
        }

        void Start()
        {
            _home = transform.position;
            var pc = FindObjectOfType<CharacterController>();
            if (pc != null) _player = pc.transform;

            // If the spawn point missed the baked mesh (e.g. on a prop), snap
            // to the nearest walkable spot instead of erroring every frame.
            if (!_agent.isOnNavMesh &&
                NavMesh.SamplePosition(transform.position, out var hit, 5f, NavMesh.AllAreas))
            {
                _agent.Warp(hit.position);
                _home = hit.position;
            }
            if (!_agent.isOnNavMesh)
            {
                Debug.LogWarning($"[NpcLocomotion] {name} is not on the NavMesh — " +
                                 "did you run GAI NPC → Bake NavMesh?");
            }
        }

        private void OnAction(string actionId)
        {
            switch (actionId)
            {
                case "walk_to":
                case "gather":
                case "work":
                    SetMode(Mode.Wander);
                    break;
                case "flee":
                    SetMode(Mode.Flee);
                    break;
                case "socialize":
                    SetMode(Mode.Approach);
                    break;
                default:
                    // eat / drink / sleep / heal / pray / attack: stay put
                    SetMode(Mode.Idle);
                    break;
            }
        }

        private void SetMode(Mode mode)
        {
            _mode = mode;
            _arrivedAt = -1f;
            if (!_agent.isOnNavMesh) return;

            _agent.ResetPath();
            _agent.stoppingDistance = mode == Mode.Approach ? approachStopDistance : 0.3f;
            _agent.speed = mode == Mode.Flee ? fleeSpeed : walkSpeed;

            switch (mode)
            {
                case Mode.Wander: PickWanderPoint(); break;
                case Mode.Flee: PickFleePoint(); break;
            }
        }

        void Update()
        {
            // Feed actual movement speed to the animator (idle↔walk↔run blend).
            // Skipped when the controller has no Speed parameter (idle-only).
            if (_hasSpeedParam)
            {
                _animator.SetFloat(SpeedParam,
                    _agent.isOnNavMesh ? _agent.velocity.magnitude : 0f);
            }

            if (!_agent.isOnNavMesh) return;

            // Any open dialogue freezes village movement — same rule as the
            // /act pause — so NPCs don't wander off mid-conversation.
            bool talking = DialogueUI.Instance != null && DialogueUI.Instance.IsOpen;
            _agent.isStopped = talking;
            if (talking)
            {
                FacePlayerIfClose();
                return;
            }

            switch (_mode)
            {
                case Mode.Wander: TickWander(); break;
                case Mode.Flee: TickFlee(); break;
                case Mode.Approach: TickApproach(); break;
            }
        }

        private bool Arrived =>
            !_agent.pathPending && _agent.remainingDistance <= _agent.stoppingDistance + 0.1f;

        private void TickWander()
        {
            if (!Arrived) { _arrivedAt = -1f; return; }
            if (_arrivedAt < 0f) { _arrivedAt = Time.time; return; }
            if (Time.time - _arrivedAt >= pauseAtDestination)
            {
                _arrivedAt = -1f;
                PickWanderPoint();
            }
        }

        private void TickFlee()
        {
            if (!Arrived) return;
            // Far enough? Calm down; otherwise keep running.
            if (_player != null &&
                Vector3.Distance(_player.position, transform.position) < fleeDistance)
            {
                PickFleePoint();
            }
            else
            {
                _mode = Mode.Idle;
            }
        }

        private void TickApproach()
        {
            if (_player == null) { _mode = Mode.Idle; return; }
            if (Time.time >= _nextApproachRepath)
            {
                _nextApproachRepath = Time.time + 0.5f;
                _agent.SetDestination(_player.position);
            }
            if (Arrived) FacePlayerIfClose();
        }

        private void PickWanderPoint()
        {
            for (int i = 0; i < 8; i++)
            {
                var candidate = _home + new Vector3(
                    Random.Range(-wanderRadius, wanderRadius), 0f,
                    Random.Range(-wanderRadius, wanderRadius));
                if (NavMesh.SamplePosition(candidate, out var hit, 2f, NavMesh.AllAreas))
                {
                    _agent.SetDestination(hit.position);
                    return;
                }
            }
        }

        private void PickFleePoint()
        {
            Vector2 rnd = Random.insideUnitCircle;
            Vector3 away = _player != null
                ? transform.position - _player.position
                : new Vector3(rnd.x, 0f, rnd.y);
            away = new Vector3(away.x, 0f, away.z).normalized;
            if (away.sqrMagnitude < 0.01f) away = transform.forward;

            for (int i = 0; i < 8; i++)
            {
                // Fan out around the escape direction until a reachable spot is found.
                var dir = Quaternion.Euler(0f, Random.Range(-40f, 40f) * i / 2f, 0f) * away;
                var candidate = transform.position + dir * fleeDistance;
                if (NavMesh.SamplePosition(candidate, out var hit, 4f, NavMesh.AllAreas))
                {
                    _agent.SetDestination(hit.position);
                    return;
                }
            }
        }

        private void FacePlayerIfClose()
        {
            if (_player == null) return;
            var to = Vector3.ProjectOnPlane(_player.position - transform.position, Vector3.up);
            if (to.sqrMagnitude < 0.01f || to.magnitude > faceRange) return;
            transform.rotation = Quaternion.Slerp(
                transform.rotation, Quaternion.LookRotation(to), 5f * Time.deltaTime);
        }
    }
}
