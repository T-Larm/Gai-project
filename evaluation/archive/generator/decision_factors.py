# Copyright 2025-2026 Sadık Abdusselam Albayrak
# Licensed under the Apache License, Version 2.0
"""
Multi-factor NPC decision model for v1.6.0 dataset generator.

Replaces the old heuristic (if Brave: attack; else flee) with a weighted
score that weighs self_power vs perceived_threat, modulated by duty_pull.
Decision emerges from numbers, not trait pattern-matching.

Public API:
  self_power(state)              -> float [0, 1]
  perceived_threat(state, tid)   -> float [0, 1]
  duty_pull(state, tid)          -> float [0, 1]
  pick_action_multifactor(state) -> (action_id, factors_dict)
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Lookup tables
# ─────────────────────────────────────────────────────────────────────────────

_ROLE_COMBAT: dict[str, float] = {
    "Guard": 0.8, "Knight": 0.8, "Bandit": 0.8,
    "Warrior": 0.8, "Soldier": 0.8, "Goblin": 0.6, "Thief": 0.6,
    "Wizard": 0.5, "King": 0.5, "Blacksmith": 0.5,
    "Farmer": 0.4, "Merchant": 0.4, "Bard": 0.3,
    "Innkeeper": 0.3, "Herbalist": 0.2, "Priest": 0.2, "Scholar": 0.2,
}
_DEFAULT_COMBAT = 0.5

_WEAPONS = {"sword", "dagger", "bow", "axe", "spear", "crossbow", "knife", "mace"}

# Threat entity → inferred faction (None = natural threat, no faction)
_THREAT_FACTION: dict[str, str | None] = {
    "bandit_01": "Bandits",
    "soldier_enemy": "EnemyForce",
    "wolf": None,
    "ghost": None,
    "wild_boar": None,
}

# NPC faction → set of enemy factions that trigger loyalty bonus
_FACTION_ENEMIES: dict[str, set[str]] = {
    "CityWatch":    {"Bandits", "EnemyForce"},
    "Church":       {"Bandits"},
    "Farmers":      {"Bandits"},
    "MerchantGuild": {"Bandits"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Component functions
# ─────────────────────────────────────────────────────────────────────────────

def _role_combat_modifier(occ: str) -> float:
    return _ROLE_COMBAT.get(occ, _DEFAULT_COMBAT)


def _weapon_factor(inv: list[dict]) -> float:
    ids = {item["id"] for item in inv}
    return 1.0 if ids & _WEAPONS else 0.3


def _trait_courage_modifier(traits: list[str]) -> float:
    mod = 0.0
    if "Brave" in traits:
        mod += 0.2
    if "Aggressive" in traits or "Wrathful" in traits:
        mod += 0.3
    if "Cautious" in traits or "Anxious" in traits or "Coward" in traits:
        mod -= 0.2
    return max(-0.3, min(0.5, mod))


def self_power(state: dict) -> float:
    """Weighted combat readiness score [0, 1]."""
    v = state["vitals"]
    sp = (
        0.40 * (v["hp"] / v["hp_max"])
        + 0.20 * v["en"]
        + 0.15 * _role_combat_modifier(state["occ"])
        + 0.15 * _weapon_factor(state["inv"])
        + 0.10 * (0.5 + _trait_courage_modifier(state["traits"]))
    )
    return round(max(0.0, min(1.0, sp)), 3)


def _memory_bias(memories: list[dict], threat_id: str) -> float:
    """Negative past memories about this (or similar) entity increase perceived threat."""
    bias = 0.0
    entity_root = threat_id.split("_")[0]
    for m in memories:
        ew = m.get("ew", 0.0)
        if ew >= 0:
            continue
        if entity_root in m.get("desc", "").lower():
            bias += min(0.15, abs(ew) * 0.3)
        else:
            bias += 0.05  # any strong negative memory raises alertness
    return round(min(0.3, bias), 3)


def _crowd_factor(percepts: list[dict]) -> float:
    """Each additional threat percept beyond the first adds 0.1."""
    n_threats = sum(1 for p in percepts if p.get("tag") == "Threat")
    return round(min(0.3, max(0, n_threats - 1) * 0.1), 3)


def perceived_threat(state: dict, threat_id: str) -> float:
    """Subjective threat level [0, 1] — base + memory bias + crowd pressure."""
    threats = [p for p in state["percepts"] if p.get("tag") == "Threat"]
    if not threats:
        return 0.0
    top = max(threats, key=lambda p: p["threat"])
    pt = top["threat"] + _memory_bias(state["memories"], threat_id) + _crowd_factor(state["percepts"])
    return round(max(0.0, min(1.0, pt)), 3)


def _trait_duty(traits: list[str]) -> float:
    return min(0.9, sum(0.3 for t in ("Loyal", "Honorable", "Devout") if t in traits))


def _faction_loyalty_bonus(state: dict, threat_id: str) -> float:
    """+0.3 when threat entity belongs to a faction the NPC considers enemy."""
    threat_faction = _THREAT_FACTION.get(threat_id)
    if not threat_faction:
        return 0.0
    npc_faction = state.get("faction", "")
    enemies = _FACTION_ENEMIES.get(npc_faction, set())
    return 0.3 if threat_faction in enemies else 0.0


def duty_pull(state: dict, threat_id: str = "") -> float:
    """Moral/duty pressure to stand and fight [0, 1]."""
    dp = (
        _trait_duty(state["traits"])
        + _faction_loyalty_bonus(state, threat_id)
        + 0.2 * state["b5"].get("c", 0.5)
    )
    return round(max(0.0, min(1.0, dp)), 3)


# ─────────────────────────────────────────────────────────────────────────────
# Non-combat and defensive action selectors
# ─────────────────────────────────────────────────────────────────────────────

def _pick_noncombat(state: dict) -> str:
    v = state["vitals"]
    inv_ids = {i["id"] for i in state["inv"]}
    percepts = state["percepts"]
    emo = state["emo"]
    occ = state["occ"]
    sched = state["sched"]

    if v["hp"] < 25 and "medicine" in inv_ids:
        return "heal"
    if v["hun"] > 0.70:
        return "eat" if "food" in inv_ids else "gather"
    if v["thi"] > 0.70:
        return "drink" if "water" in inv_ids else "gather"
    if v["en"] < 0.25:
        return "sleep"

    social_p = next((p for p in percepts if p.get("tag") == "Social"), None)
    if social_p and emo["hap"] > 0.3:
        return "socialize"

    role_default = {"Priest": "pray", "Blacksmith": "work", "Farmer": "work", "Scholar": "work"}
    if occ in role_default and sched.get("act") == "work":
        return role_default[occ]

    return "walk_to"


def _pick_defensive(state: dict) -> str:
    inv_ids = {i["id"] for i in state["inv"]}
    if "medicine" in inv_ids and state["vitals"]["hp"] < 60:
        return "heal"
    return "gather"


# ─────────────────────────────────────────────────────────────────────────────
# Main decision entry point
# ─────────────────────────────────────────────────────────────────────────────

def pick_action_multifactor(state: dict) -> tuple[str, dict]:
    """
    Return (action_id, factors_dict) via 3-zone decision model.

    Zones:
      no_threat   — perceived_threat <= 0 (no threat percept at all)
      low_threat  — perceived_threat <= 0.30 (threat present but minor)
      dominant    — self_power > perceived_threat + 0.15
      duty_attack — weaker but duty_pull > 0.60 and faction under threat
      retreat     — self_power < perceived_threat - 0.15
      ambivalent  — balanced; take defensive action
    """
    sp = self_power(state)
    threats = [p for p in state["percepts"] if p.get("tag") == "Threat"]

    if not threats:
        action_id = _pick_noncombat(state)
        return action_id, {"self_power": sp, "perceived_threat": 0.0, "duty_pull": 0.0, "zone": "no_threat"}

    top_threat = max(threats, key=lambda p: p["threat"])
    threat_id = top_threat["id"]
    pt = perceived_threat(state, threat_id)
    dp = duty_pull(state, threat_id)
    faction_threatened = _faction_loyalty_bonus(state, threat_id) > 0

    if pt <= 0.30:
        action_id = _pick_noncombat(state)
        zone = "low_threat"
    elif sp > pt + 0.15:
        action_id = "attack"
        zone = "dominant"
    elif dp > 0.60 and faction_threatened:
        action_id = "attack"
        zone = "duty_attack"
    elif sp < pt - 0.15:
        action_id = "flee"
        zone = "retreat"
    else:
        action_id = _pick_defensive(state)
        zone = "ambivalent"

    return action_id, {
        "self_power": sp,
        "perceived_threat": pt,
        "duty_pull": dp,
        "zone": zone,
    }
