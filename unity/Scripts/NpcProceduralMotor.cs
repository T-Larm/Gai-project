using UnityEngine;
using UnityEngine.AI;

namespace GaiNpc
{
    /// <summary>
    /// Lightweight movement and whole-body gesture fallback for models that
    /// do not yet have a usable Avatar or AnimatorController.
    /// </summary>
    [DisallowMultipleComponent]
    public class NpcProceduralMotor : MonoBehaviour
    {
        [Header("References")]
        public NpcBehaviorClient client;
        public NpcActionRouter actionRouter;
        public Transform player;
        public NpcDialogueUI dialogueUI;

        [Header("Movement")]
        [Min(0.1f)] public float moveSpeed = 0.9f;
        [Min(1f)] public float turnSpeed = 160f;
        [Min(0.5f)] public float roamRadius = 2.5f;
        [Min(0.1f)] public float arrivalDistance = 0.45f;
        [Min(0.2f)] public float playerStopDistance = 1.8f;
        public LayerMask groundMask = ~0;
        public LayerMask obstacleMask = ~0;
        [Min(0.05f)] public float obstacleRadius = 0.28f;
        [Min(0.1f)] public float obstacleProbeDistance = 0.65f;
        public bool snapToGround = true;
        public bool pauseWhileDialogueOpen = true;

        [Header("Procedural Gesture")]
        [Min(0.1f)] public float gestureFrequency = 2.2f;
        [Min(0f)] public float walkBobHeight = 0.045f;
        [Min(0f)] public float gestureTilt = 8f;

        public string CurrentAction => currentAction;
        public bool IsMoving => isMoving;

        private NavMeshAgent navMeshAgent;
        private Vector3 homePosition;
        private Vector3 motorPosition;
        private Vector3 roamTarget;
        private Vector3 initialEuler;
        private float currentYaw;
        private string currentAction = "work";
        private bool isMoving;
        private int roamStep;
        private Transform configuredDestination;

        private void Awake()
        {
            if (client == null) client = GetComponent<NpcBehaviorClient>();
            if (actionRouter == null) actionRouter = GetComponent<NpcActionRouter>();
            if (dialogueUI == null) dialogueUI = FindObjectOfType<NpcDialogueUI>();
            if (player == null)
            {
                NpcSceneStateProvider state = GetComponent<NpcSceneStateProvider>();
                if (state != null) player = state.player;
            }

            navMeshAgent = GetComponent<NavMeshAgent>();
            if (navMeshAgent != null && navMeshAgent.enabled && !navMeshAgent.isOnNavMesh)
                navMeshAgent.enabled = false;

            homePosition = transform.position;
            motorPosition = transform.position;
            initialEuler = transform.eulerAngles;
            currentYaw = initialEuler.y;
            ChooseRoamTarget();
        }

        private void OnEnable()
        {
            if (client != null) client.OnActionChanged.AddListener(HandleAction);
        }

        private void Start()
        {
            if (client != null && !string.IsNullOrWhiteSpace(client.CurrentAction))
                HandleAction(client.CurrentAction);
            else
                HandleAction(currentAction);
        }

        private void OnDisable()
        {
            if (client != null) client.OnActionChanged.RemoveListener(HandleAction);
            transform.position = motorPosition;
            transform.rotation = Quaternion.Euler(initialEuler.x, currentYaw, initialEuler.z);
        }

        private void Update()
        {
            bool paused = pauseWhileDialogueOpen && dialogueUI != null && dialogueUI.IsOpen;
            isMoving = !paused && UpdateMovement(Time.deltaTime);
            UpdateGesture(paused ? "idle" : currentAction);
        }

        public void HandleAction(string actionId)
        {
            currentAction = string.IsNullOrWhiteSpace(actionId)
                ? "work"
                : actionId.Trim().ToLowerInvariant();
            configuredDestination = FindConfiguredDestination(currentAction);

            if (IsRoamingAction(currentAction) && configuredDestination == null)
                ChooseRoamTarget();
        }

        public void ReturnHome()
        {
            configuredDestination = null;
            roamTarget = homePosition;
            currentAction = "walk_to";
        }

        private bool UpdateMovement(float deltaTime)
        {
            Vector3 destination;
            float stopDistance;

            if (configuredDestination != null)
            {
                destination = configuredDestination.position;
                stopDistance = arrivalDistance;
            }
            else if (currentAction == "flee" && player != null)
            {
                Vector3 away = motorPosition - player.position;
                away.y = 0f;
                if (away.sqrMagnitude < 0.01f) away = transform.right;
                destination = motorPosition + away.normalized * roamRadius;
                stopDistance = 0f;
            }
            else if ((currentAction == "attack" || currentAction == "socialize") && player != null)
            {
                destination = player.position;
                stopDistance = currentAction == "attack"
                    ? Mathf.Max(1.1f, playerStopDistance * 0.7f)
                    : playerStopDistance;
            }
            else if (IsRoamingAction(currentAction))
            {
                destination = roamTarget;
                stopDistance = arrivalDistance;
            }
            else
            {
                FacePlayerWhenRelevant(deltaTime);
                return false;
            }

            Vector3 offset = destination - motorPosition;
            offset.y = 0f;
            float distance = offset.magnitude;
            if (distance <= stopDistance)
            {
                if (IsRoamingAction(currentAction) && configuredDestination == null)
                    ChooseRoamTarget();
                FacePlayerWhenRelevant(deltaTime);
                return false;
            }

            Vector3 direction = offset / Mathf.Max(distance, 0.0001f);
            direction = AvoidObstacle(direction);
            if (direction.sqrMagnitude < 0.01f) return false;

            float desiredYaw = Mathf.Atan2(direction.x, direction.z) * Mathf.Rad2Deg;
            currentYaw = Mathf.MoveTowardsAngle(currentYaw, desiredYaw, turnSpeed * deltaTime);

            float step = Mathf.Min(moveSpeed * deltaTime, Mathf.Max(distance - stopDistance, 0f));
            Vector3 candidate = motorPosition + direction.normalized * step;
            SnapCandidateToGround(ref candidate);
            motorPosition = candidate;
            return step > 0f;
        }

        private Vector3 AvoidObstacle(Vector3 direction)
        {
            Vector3 origin = motorPosition + Vector3.up * 0.8f;
            float distance = Mathf.Max(obstacleProbeDistance, moveSpeed * Time.deltaTime + 0.1f);
            RaycastHit hit;
            if (!Physics.SphereCast(
                    origin,
                    obstacleRadius,
                    direction,
                    out hit,
                    distance,
                    obstacleMask,
                    QueryTriggerInteraction.Ignore))
                return direction;

            Vector3 slide = Vector3.ProjectOnPlane(direction, hit.normal);
            slide.y = 0f;
            if (slide.sqrMagnitude > 0.01f) return slide.normalized;

            float side = (GetInstanceID() & 1) == 0 ? 1f : -1f;
            return Quaternion.Euler(0f, 90f * side, 0f) * direction;
        }

        private void SnapCandidateToGround(ref Vector3 candidate)
        {
            if (!snapToGround) return;

            RaycastHit hit;
            Vector3 origin = candidate + Vector3.up * 1.5f;
            if (Physics.Raycast(
                    origin,
                    Vector3.down,
                    out hit,
                    3.5f,
                    groundMask,
                    QueryTriggerInteraction.Ignore))
            {
                candidate.y = hit.point.y;
            }
        }

        private void FacePlayerWhenRelevant(float deltaTime)
        {
            if (player == null || (currentAction != "socialize" && currentAction != "attack"))
                return;

            Vector3 direction = player.position - motorPosition;
            direction.y = 0f;
            if (direction.sqrMagnitude < 0.01f) return;
            float desiredYaw = Mathf.Atan2(direction.x, direction.z) * Mathf.Rad2Deg;
            currentYaw = Mathf.MoveTowardsAngle(currentYaw, desiredYaw, turnSpeed * deltaTime);
        }

        private void UpdateGesture(string action)
        {
            float phase = Time.time * gestureFrequency;
            float sine = Mathf.Sin(phase);
            float bob = 0f;
            float pitch = 0f;
            float roll = 0f;
            float lunge = 0f;

            if (isMoving)
            {
                bob = Mathf.Abs(sine) * walkBobHeight;
                roll = sine * gestureTilt * 0.25f;
            }

            switch (action)
            {
                case "drink":
                    pitch = -gestureTilt + sine * 2f;
                    bob = Mathf.Max(bob, Mathf.Abs(sine) * 0.025f);
                    break;
                case "eat":
                    pitch = gestureTilt * 0.65f + sine * 2f;
                    break;
                case "gather":
                    pitch = gestureTilt * 1.4f;
                    bob -= 0.08f;
                    break;
                case "heal":
                    roll += sine * gestureTilt * 0.55f;
                    break;
                case "pray":
                    pitch = gestureTilt * 1.8f + sine;
                    break;
                case "sleep":
                    pitch = gestureTilt * 2.8f;
                    roll = gestureTilt * 0.8f;
                    bob -= 0.18f;
                    break;
                case "socialize":
                    roll += sine * gestureTilt * 0.7f;
                    bob += Mathf.Abs(sine) * 0.02f;
                    break;
                case "attack":
                    pitch = -gestureTilt * 1.2f;
                    lunge = Mathf.Max(0f, sine) * 0.12f;
                    break;
                case "flee":
                    pitch = gestureTilt * 0.45f;
                    roll += sine * gestureTilt * 0.45f;
                    break;
                case "work":
                    roll += sine * gestureTilt * 0.35f;
                    break;
            }

            Quaternion yaw = Quaternion.Euler(0f, currentYaw, 0f);
            Vector3 forward = yaw * Vector3.forward;
            transform.position = motorPosition + Vector3.up * bob + forward * lunge;
            transform.rotation = Quaternion.Euler(
                initialEuler.x + pitch,
                currentYaw,
                initialEuler.z + roll);
        }

        private void ChooseRoamTarget()
        {
            roamStep++;
            float seed = Mathf.Abs(GetInstanceID() * 0.0137f) + roamStep * 2.39996f;
            float angle = Mathf.Repeat(seed, Mathf.PI * 2f);
            float radius = roamRadius * (0.55f + 0.4f * Mathf.Abs(Mathf.Sin(seed * 0.73f)));
            roamTarget = homePosition + new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle)) * radius;
            SnapCandidateToGround(ref roamTarget);
        }

        private Transform FindConfiguredDestination(string actionId)
        {
            if (actionRouter == null || actionRouter.routes == null) return null;
            foreach (NpcActionRouter.ActionRoute route in actionRouter.routes)
            {
                if (route == null || route.destination == null) continue;
                if (string.Equals(route.actionId, actionId, System.StringComparison.OrdinalIgnoreCase))
                    return route.destination;
            }
            return null;
        }

        private static bool IsRoamingAction(string action)
        {
            return action == "walk_to" || action == "gather" || action == "work";
        }
    }
}
