from backend.behavior.heuristic_policy import HeuristicSurvivalPolicy, heuristic_action_id


def _state(**overrides):
    state = {
        "vitals": {"hp": 100.0, "hp_max": 120.0, "en": 0.8, "hun": 0.2, "thi": 0.2, "str": 0.5},
        "inv": [{"id": "food", "n": 2}, {"id": "water", "n": 1}],
        "percepts": [],
        "sched": {"act": "idle"},
    }
    state.update(overrides)
    return state


def test_flees_when_threat_exceeds_strength():
    state = _state(percepts=[{"id": "wolf", "tag": "Threat", "threat": 0.8}])
    assert heuristic_action_id(state) == "flee"


def test_attacks_when_stronger_than_threat():
    state = _state(
        vitals={"hp": 110.0, "hp_max": 120.0, "en": 0.9, "hun": 0.2, "thi": 0.2, "str": 0.9},
        percepts=[{"id": "wolf", "tag": "Threat", "threat": 0.4}],
    )
    assert heuristic_action_id(state) == "attack"


def test_survival_needs_without_threat():
    assert heuristic_action_id(_state(vitals={"hp": 100, "hp_max": 120, "en": 0.8, "hun": 0.2, "thi": 0.9, "str": 0.5})) == "drink"
    assert heuristic_action_id(_state(vitals={"hp": 100, "hp_max": 120, "en": 0.8, "hun": 0.9, "thi": 0.2, "str": 0.5})) == "eat"
    assert heuristic_action_id(
        _state(
            vitals={"hp": 30, "hp_max": 120, "en": 0.8, "hun": 0.2, "thi": 0.2, "str": 0.5},
            inv=[{"id": "medicine", "n": 1}],
        )
    ) == "heal"
    assert heuristic_action_id(_state(vitals={"hp": 100, "hp_max": 120, "en": 0.1, "hun": 0.2, "thi": 0.2, "str": 0.5})) == "sleep"


def test_socializes_then_works_then_walks():
    assert heuristic_action_id(_state(percepts=[{"id": "guard_npc", "tag": "Social", "sal": 0.5}])) == "socialize"
    assert heuristic_action_id(_state(sched={"act": "work"})) == "work"
    assert heuristic_action_id(_state()) == "walk_to"


def test_policy_wrapper_predicts_from_raw_state():
    policy = HeuristicSurvivalPolicy()
    prediction = policy.predict(_state(percepts=[{"id": "wolf", "tag": "Threat", "threat": 0.9}]))
    assert prediction == {"action_id": "flee"}
