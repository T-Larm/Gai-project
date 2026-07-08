"""Native feature extraction for the simulator behavior policy (方案 B).

Projects a raw Stateful-RPG simulator state onto the leakage-free feature
groups used by the supervised policy. Shared by the dataset converter
(evaluation/datasets/convert_stateful_rpg.py) and runtime inference, so
training and serving always see identical features.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

# Native action space of the dataset ("trade" has zero samples but stays in
# the vocabulary so policies expose the full simulator interface).
NATIVE_ACTIONS = (
    "eat", "drink", "sleep", "flee", "gather", "heal",
    "attack", "socialize", "trade", "work", "pray", "walk_to",
)


def extract_native_features(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Project a raw simulator state onto leakage-free native features."""
    vitals = _mapping(state.get("vitals"))
    emo = _mapping(state.get("emo"))
    b5 = _mapping(state.get("b5"))
    time = _mapping(state.get("time"))
    sched = _mapping(state.get("sched"))
    factions = _mapping(state.get("factions"))
    percepts = [p for p in _list(state.get("percepts")) if isinstance(p, Mapping)]
    memories = [m for m in _list(state.get("memories")) if isinstance(m, Mapping)]

    threat_percepts = [p for p in percepts if p.get("tag") == "Threat"]
    threats = [_float(p.get("threat")) for p in threat_percepts]
    neg_memories = [m for m in memories if _float(m.get("ew")) < 0]
    neg_ews = [abs(_float(m.get("ew"))) for m in neg_memories]
    faction_reps = [_float(v) for v in factions.values()]

    top_threat = max(threat_percepts, key=lambda p: _float(p.get("threat")), default=None)
    top_threat_id = _norm(top_threat.get("id")) if top_threat else "none"
    # Does any negative memory mention the perceived threat entity? Raw string
    # check on state fields — the NPC "remembers" this kind of enemy.
    threat_root = top_threat_id.split("_")[0] if top_threat else ""
    threat_in_neg_memory = 1.0 if threat_root and any(
        threat_root in str(m.get("desc", "")).lower() for m in neg_memories
    ) else 0.0

    categorical = {
        "occ": _norm(state.get("occ")),
        "arch": _norm(state.get("arch")),
        "faction": _norm(state.get("faction")),
        "sched_act": _norm(sched.get("act")),
        "goals_top": _norm(state.get("goals_top")),
        "top_threat_id": top_threat_id or "none",
    }
    multi = {
        "traits": sorted(_norm(t) for t in _list(state.get("traits")) if _norm(t)),
        "inv": inventory_flags(state.get("inv")),
    }
    continuous = {
        "hp": _float(vitals.get("hp")),
        "hp_max": _float(vitals.get("hp_max")),
        "en": _float(vitals.get("en")),
        "hun": _float(vitals.get("hun")),
        "thi": _float(vitals.get("thi")),
        "str": _float(vitals.get("str")),
        "b5_e": _float(b5.get("e")),
        "b5_a": _float(b5.get("a")),
        "b5_c": _float(b5.get("c")),
        "b5_n": _float(b5.get("n")),
        "b5_o": _float(b5.get("o")),
        "emo_hap": _float(emo.get("hap")),
        "emo_fear": _float(emo.get("fear")),
        "emo_ang": _float(emo.get("ang")),
        "time_day": _float(time.get("day")),
        "time_hr": _float(time.get("hr")),
        "sched_wk_start": _float(sched.get("wk_start")),
        "sched_wk_end": _float(sched.get("wk_end")),
        "sched_sleep": _float(sched.get("sleep")),
        "sched_wake": _float(sched.get("wake")),
        "max_threat": max(threats) if threats else 0.0,
        "n_threat_percepts": float(len(threats)),
        "has_social_percept": 1.0 if any(p.get("tag") == "Social" for p in percepts) else 0.0,
        "has_food_percept": 1.0 if any(p.get("tag") == "Food" for p in percepts) else 0.0,
        "n_memories": float(len(memories)),
        "n_neg_memories": float(len(neg_ews)),
        "max_neg_memory_ew": max(neg_ews) if neg_ews else 0.0,
        "threat_in_neg_memory": threat_in_neg_memory,
        "faction_rep_min": min(faction_reps) if faction_reps else 0.0,
        "faction_rep_max": max(faction_reps) if faction_reps else 0.0,
        "interrupt": 1.0 if state.get("interrupt") else 0.0,
    }
    return {"categorical": categorical, "multi": multi, "continuous": continuous}


def inventory_flags(inv: Any) -> List[str]:
    flags = []
    for item in _list(inv):
        if isinstance(item, Mapping) and _float(item.get("n")) > 0:
            flag = _norm(item.get("id"))
            if flag:
                flags.append(flag)
    return sorted(set(flags))


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
