import json

from evaluation.datasets.convert_stateful_rpg import (
    convert_directory,
    convert_reasoner_file,
    dedupe_records,
    extract_native_features,
    load_decision_module,
    oracle_label,
    stratified_split,
)


def _chat(system: str, user: str, assistant: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{assistant}<|eot_id|>"
    )


def _state(**overrides):
    state = {
        "id": "npc_d301e377",
        "arch": "Aggressive",
        "occ": "King",
        "faction": "MerchantGuild",
        "b5": {"e": 0.88, "a": 0.16, "c": 0.38, "n": 0.73, "o": 0.32},
        "traits": ["Aggressive", "Wrathful"],
        "vitals": {"hp": 101.5, "hp_max": 120.0, "en": 0.36, "hun": 0.25, "thi": 0.86, "str": 0.65},
        "emo": {"hap": 0.0, "fear": 0.58, "ang": 0.38, "mood": "Fearful"},
        "inv": [{"id": "food", "n": 5}, {"id": "water", "n": 2}],
        "time": {"day": 10, "hr": 18.6},
        "pos": {"x": 19.3, "z": 9.1, "zone": "Barracks", "landmark": "ArmoryDoor"},
        "sched": {"act": "social", "wk_start": 9, "wk_end": 17, "sleep": 22, "wake": 7},
        "percepts": [],
        "memories": [],
        "beliefs": [],
        "factions": {"CityWatch": 0.1, "Bandits": -0.16},
        "goals_top": "FindWater",
        "interrupt": False,
    }
    state.update(overrides)
    return state


def test_load_decision_module_exposes_generator_rule():
    module = load_decision_module()
    action_id, factors = module.pick_action_multifactor(_state())
    assert action_id == "drink"
    assert factors["zone"] == "no_threat"


def test_oracle_label_uses_generator_ground_truth():
    # Default state has thi=0.86 > 0.85, so the survival override fires.
    label = oracle_label(_state())
    assert label["action_id"] == "drink"
    assert label["zone"] == "survival_override"
    assert set(label["factors"]) == {"self_power", "perceived_threat", "duty_pull"}

    # Below the override threshold the multifactor rule decides.
    mild_thirst = _state(
        vitals={"hp": 101.5, "hp_max": 120.0, "en": 0.36, "hun": 0.25, "thi": 0.75, "str": 0.65},
    )
    label = oracle_label(mild_thirst)
    assert label["action_id"] == "drink"
    assert label["zone"] == "no_threat"


def test_oracle_label_applies_survival_overrides_before_combat():
    # Generator's _select_action_standard: hun > 0.85 overrides threat logic.
    starving_under_threat = _state(
        vitals={"hp": 60.0, "hp_max": 120.0, "en": 0.65, "hun": 0.94, "thi": 0.54, "str": 0.43},
        percepts=[{"id": "wild_boar", "tag": "Threat", "sal": 0.65, "threat": 0.97}],
    )
    label = oracle_label(starving_under_threat)
    assert label["action_id"] == "eat"
    assert label["zone"] == "survival_override"

    thirsty_no_water = _state(
        vitals={"hp": 60.0, "hp_max": 120.0, "en": 0.65, "hun": 0.2, "thi": 0.9, "str": 0.43},
        inv=[],
    )
    assert oracle_label(thirsty_no_water)["action_id"] == "gather"

    critical_hp = _state(
        vitals={"hp": 15.0, "hp_max": 120.0, "en": 0.65, "hun": 0.2, "thi": 0.2, "str": 0.43},
        inv=[{"id": "medicine", "n": 1}],
    )
    assert oracle_label(critical_hp)["action_id"] == "heal"


def test_oracle_label_flees_when_overwhelmed():
    state = _state(
        occ="Scholar",
        traits=["Cautious"],
        vitals={"hp": 20.0, "hp_max": 120.0, "en": 0.1, "hun": 0.2, "thi": 0.2, "str": 0.1},
        inv=[],
        percepts=[{"id": "bandit_01", "tag": "Threat", "sal": 0.9, "threat": 0.9}],
        factions={},
        faction="",
    )
    label = oracle_label(state)
    assert label["action_id"] == "flee"
    assert label["zone"] == "retreat"


def test_native_features_have_expected_values():
    features = extract_native_features(_state())

    assert features["categorical"] == {
        "occ": "king",
        "arch": "aggressive",
        "faction": "merchantguild",
        "sched_act": "social",
        "goals_top": "findwater",
    }
    assert features["multi"]["traits"] == ["aggressive", "wrathful"]
    assert features["multi"]["inv"] == ["food", "water"]

    continuous = features["continuous"]
    assert continuous["hp"] == 101.5
    assert continuous["hp_max"] == 120.0
    assert continuous["thi"] == 0.86
    assert continuous["b5_c"] == 0.38
    assert continuous["emo_fear"] == 0.58
    assert continuous["time_hr"] == 18.6
    assert continuous["sched_sleep"] == 22
    assert continuous["max_threat"] == 0.0
    assert continuous["n_threat_percepts"] == 0
    assert continuous["has_social_percept"] == 0.0
    assert continuous["faction_rep_min"] == -0.16
    assert continuous["faction_rep_max"] == 0.1
    assert continuous["interrupt"] == 0.0


def test_native_features_summarize_percepts_and_memories():
    state = _state(
        percepts=[
            {"id": "wolf", "tag": "Threat", "sal": 0.79, "threat": 0.59},
            {"id": "bandit_01", "tag": "Threat", "sal": 0.5, "threat": 0.3},
            {"id": "guard_npc", "tag": "Social", "sal": 0.49},
            {"id": "herb_patch", "tag": "Food", "sal": 0.37},
        ],
        memories=[
            {"evt": "Memory", "desc": "iyi anı", "ew": 0.45, "dt": 100},
            {"evt": "Memory", "desc": "kötü anı", "ew": -0.73, "dt": 200},
        ],
    )
    continuous = extract_native_features(state)["continuous"]
    assert continuous["max_threat"] == 0.59
    assert continuous["n_threat_percepts"] == 2
    assert continuous["has_social_percept"] == 1.0
    assert continuous["has_food_percept"] == 1.0
    assert continuous["n_memories"] == 2
    assert continuous["n_neg_memories"] == 1
    assert continuous["max_neg_memory_ew"] == 0.73


def test_native_features_exclude_leakage_fields():
    features = extract_native_features(_state())
    flat_keys = (
        set(features["categorical"])
        | set(features["multi"])
        | set(features["continuous"])
    )
    for banned in ("player_intent", "mood", "self_power", "perceived_threat", "duty_pull", "zone"):
        assert banned not in flat_keys


def test_convert_reasoner_file_yields_v2_samples(tmp_path):
    path = tmp_path / "train_reasoner.jsonl"
    state = _state()
    path.write_text(
        json.dumps({"text": _chat("system", f"Intro text.\n\n{json.dumps(state)}", "Susuzluk dayanılmaz.")}),
        encoding="utf-8",
    )

    samples = list(convert_reasoner_file(path))

    assert len(samples) == 1
    sample = samples[0]
    assert sample["id"].startswith("srpg_")
    assert sample["source"]["file"] == "train_reasoner.jsonl"
    assert sample["label"]["action_id"] == "drink"
    assert sample["aux"]["mood"] == "fearful"
    assert sample["reasoning"] == "Susuzluk dayanılmaz."
    assert sample["source_state"] == state
    assert "player_intent" not in json.dumps(sample["features"])


def test_dedupe_records_drops_identical_states(tmp_path):
    path = tmp_path / "train_reasoner.jsonl"
    state = _state()
    row = json.dumps({"text": _chat("system", f"Intro {json.dumps(state)}", "Su içmeliyim.")})
    path.write_text(row + "\n" + row, encoding="utf-8")

    samples = list(convert_reasoner_file(path))
    unique, removed = dedupe_records(samples)

    assert len(samples) == 2
    assert len(unique) == 1
    assert removed == 1


def test_stratified_split_keeps_class_ratios():
    records = []
    for action in ("eat", "flee"):
        for index in range(10):
            records.append({"id": f"{action}_{index}", "label": {"action_id": action}})

    splits = stratified_split(records, seed=13)

    for split_name, expected in (("train", 8), ("valid", 1), ("test", 1)):
        for action in ("eat", "flee"):
            count = sum(1 for r in splits[split_name] if r["label"]["action_id"] == action)
            assert count == expected, f"{split_name}/{action}: {count} != {expected}"
    all_ids = [r["id"] for split in splits.values() for r in split]
    assert len(all_ids) == len(set(all_ids)) == 20


def test_stratified_split_puts_tiny_classes_in_train():
    records = [{"id": "only_1", "label": {"action_id": "pray"}}, {"id": "only_2", "label": {"action_id": "pray"}}]
    splits = stratified_split(records, seed=13)
    assert len(splits["train"]) == 2
    assert not splits["valid"] and not splits["test"]


def test_convert_directory_writes_splits_and_report(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "out"

    states = [
        _state(id=f"npc_{i}", vitals={"hp": 101.5, "hp_max": 120.0, "en": 0.36, "hun": 0.25, "thi": 0.86 - i * 0.001, "str": 0.65})
        for i in range(3)
    ]
    reasonings = [f"Su içmeliyim {i}." for i in range(3)]
    rows = [
        json.dumps({"text": _chat("system", f"Intro {json.dumps(state)}", reasoning)})
        for state, reasoning in zip(states, reasonings)
    ]
    # duplicate first row to exercise dedup
    (raw_dir / "train_reasoner.jsonl").write_text("\n".join(rows[:2] + [rows[0]]), encoding="utf-8")
    (raw_dir / "test_reasoner.jsonl").write_text(rows[2], encoding="utf-8")

    formatter_rows = [
        json.dumps({
            "text": _chat(
                "system",
                reasoning,
                json.dumps({
                    "reasoning": reasoning,
                    "selected_action": {"action_id": "drink", "target_id": None, "dialogue": None},
                    "emotion": "Fearful",
                }),
            )
        })
        for reasoning in reasonings[:2]
    ]
    (raw_dir / "train_formatter.jsonl").write_text("\n".join(formatter_rows), encoding="utf-8")

    report = convert_directory(raw_dir=raw_dir, out_dir=out_dir, seed=13)

    for split in ("train", "valid", "test"):
        assert (out_dir / f"{split}.jsonl").exists()
    assert (out_dir / "conversion_report.json").exists()

    assert report["dedup"]["removed"] == 1
    assert report["total_records"] == 3
    assert report["oracle_vs_formatter"]["matched"] == 2
    assert report["oracle_vs_formatter"]["rate"] == 1.0
    action_counts = report["action_counts"]
    assert action_counts == {"drink": 3}


def test_formatter_agreement_skips_colliding_reasoning_texts():
    from evaluation.datasets.convert_stateful_rpg import formatter_agreement

    records = [
        {"id": "a", "reasoning": "Su içmeliyim.", "label": {"action_id": "drink", "zone": "no_threat"}},
        {"id": "b", "reasoning": "Su içmeliyim.", "label": {"action_id": "flee", "zone": "retreat"}},
        {"id": "c", "reasoning": "Uyumalıyım.", "label": {"action_id": "sleep", "zone": "no_threat"}},
    ]
    index = {
        "Su içmeliyim.": {"selected_action": {"action_id": "drink"}},
        "Uyumalıyım.": {"selected_action": {"action_id": "sleep"}},
    }

    result = formatter_agreement(records, index)

    # The duplicated text maps two different states to one formatter label,
    # so it must not count toward the agreement rate.
    assert result["matched"] == 1
    assert result["rate"] == 1.0
    assert result["skipped_ambiguous"] == 2


