using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

namespace GaiNpc
{
    /// <summary>
    /// Routes policy action IDs to animation triggers and optional NavMesh
    /// destinations. Configure one route per action in the Inspector.
    /// </summary>
    public class NpcActionRouter : MonoBehaviour
    {
        [Serializable]
        public class ActionRoute
        {
            [Tooltip("Backend action: attack, drink, eat, flee, gather, heal, pray, sleep, socialize, walk_to, work")]
            public string actionId;
            public string animatorTrigger;
            public Transform destination;
            public bool navigateFirst;
            [Min(0.05f)] public float arrivalDistance = 1.2f;
        }

        [Header("References")]
        public NpcBehaviorClient client;
        public Animator animator;
        public NavMeshAgent agent;

        [Header("Action Routes")]
        public ActionRoute[] routes;

        private readonly Dictionary<string, ActionRoute> routeMap =
            new Dictionary<string, ActionRoute>(StringComparer.OrdinalIgnoreCase);

        private ActionRoute pendingRoute;

        private void Reset()
        {
            client = GetComponent<NpcBehaviorClient>();
            animator = GetComponentInChildren<Animator>();
            agent = GetComponent<NavMeshAgent>();

            string[] actions =
            {
                "attack", "drink", "eat", "flee", "gather", "heal",
                "pray", "sleep", "socialize", "walk_to", "work"
            };
            routes = new ActionRoute[actions.Length];
            for (int i = 0; i < actions.Length; i++)
            {
                routes[i] = new ActionRoute
                {
                    actionId = actions[i],
                    animatorTrigger = ToAnimatorName(actions[i]),
                    arrivalDistance = 1.2f
                };
            }
        }

        private void Awake()
        {
            if (client == null) client = GetComponent<NpcBehaviorClient>();
            if (animator == null) animator = GetComponentInChildren<Animator>();
            if (agent == null) agent = GetComponent<NavMeshAgent>();
            RebuildRouteMap();
        }

        private void OnEnable()
        {
            if (client != null) client.OnActionChanged.AddListener(HandleAction);
        }

        private void OnDisable()
        {
            if (client != null) client.OnActionChanged.RemoveListener(HandleAction);
        }

        private void Update()
        {
            if (pendingRoute == null || agent == null || agent.pathPending) return;
            if (agent.remainingDistance > Mathf.Max(agent.stoppingDistance, pendingRoute.arrivalDistance)) return;

            TriggerAnimation(pendingRoute.animatorTrigger);
            pendingRoute = null;
        }

        public void HandleAction(string actionId)
        {
            pendingRoute = null;
            if (string.IsNullOrWhiteSpace(actionId)) return;

            if (agent != null && agent.isOnNavMesh && agent.hasPath)
                agent.ResetPath();

            if (!routeMap.TryGetValue(actionId, out ActionRoute route))
            {
                Debug.LogWarning($"[NpcActionRouter] No route configured for '{actionId}'.", this);
                return;
            }

            bool canNavigate = route.navigateFirst && route.destination != null &&
                               agent != null && agent.isOnNavMesh;
            if (canNavigate)
            {
                agent.stoppingDistance = route.arrivalDistance;
                agent.SetDestination(route.destination.position);
                pendingRoute = route;
                return;
            }

            TriggerAnimation(route.animatorTrigger);
        }

        public void RebuildRouteMap()
        {
            routeMap.Clear();
            if (routes == null) return;

            foreach (ActionRoute route in routes)
            {
                if (route == null || string.IsNullOrWhiteSpace(route.actionId)) continue;
                routeMap[route.actionId.Trim()] = route;
            }
        }

        private void TriggerAnimation(string triggerName)
        {
            if (animator == null || string.IsNullOrWhiteSpace(triggerName)) return;
            if (animator.runtimeAnimatorController == null) return;
            if (!HasTrigger(triggerName))
            {
                Debug.LogWarning($"[NpcActionRouter] Animator trigger '{triggerName}' does not exist.", this);
                return;
            }
            animator.SetTrigger(triggerName);
        }

        private bool HasTrigger(string triggerName)
        {
            foreach (AnimatorControllerParameter parameter in animator.parameters)
            {
                if (parameter.type == AnimatorControllerParameterType.Trigger &&
                    parameter.name == triggerName) return true;
            }
            return false;
        }

        private static string ToAnimatorName(string actionId)
        {
            string result = "";
            bool upper = true;
            foreach (char c in actionId)
            {
                if (c == '_')
                {
                    upper = true;
                    continue;
                }
                result += upper ? char.ToUpperInvariant(c) : c;
                upper = false;
            }
            return result;
        }
    }
}
