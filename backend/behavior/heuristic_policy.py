"""Hand-written survival heuristic — the fair rule baseline for RQ1.

Written from gameplay common sense, NOT from the dataset generator's rule
(``_select_action_standard`` / ``pick_action_multifactor``): the generator's
rule is the labeling oracle, so comparing it against the trained policy would
be circular. This baseline gets the same raw simulator state as everyone else
and makes reasonable — but independently designed — decisions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from backend.behavior.native_features import inventory_flags


def heuristic_action_id(state: Mapping[str, Any]) -> str:
    vitals = _mapping(state.get("vitals"))
    inv = set(inventory_flags(state.get("inv")))
    percepts = [p for p in _list(state.get("percepts")) if isinstance(p, Mapping)]
    max_threat = max([_float(p.get("threat")) for p in percepts if p.get("tag") == "Threat"] or [0.0])
    strength = _float(vitals.get("str"))
    hp = _float(vitals.get("hp"), 1.0)
    hp_max = max(_float(vitals.get("hp_max"), 1.0), 1.0)

    if max_threat > max(strength, 0.45):
        return "flee"
    if max_threat > 0.0 and strength > max_threat:
        return "attack"
    if _float(vitals.get("thi")) >= 0.7 and "water" in inv:
        return "drink"
    if _float(vitals.get("hun")) >= 0.7 and "food" in inv:
        return "eat"
    if hp / hp_max < 0.45 and "medicine" in inv:
        return "heal"
    if _float(vitals.get("en"), 1.0) <= 0.25:
        return "sleep"
    if any(p.get("tag") == "Social" for p in percepts):
        return "socialize"
    scheduled = str(_mapping(state.get("sched")).get("act", "")).strip().lower()
    if scheduled == "work":
        return "work"
    return "walk_to"


class HeuristicSurvivalPolicy:
    """Policy-shaped wrapper so baselines and trained models share one interface."""

    def predict(self, state: Mapping[str, Any]) -> Dict[str, str]:
        return {"action_id": heuristic_action_id(state)}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
