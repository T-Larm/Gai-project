import pytest

from backend.behavior.supervised_policy import (
    ACTION_HEAD,
    MOOD_HEAD,
    FeatureSpec,
    build_feature_spec,
    build_label_spec,
    encode_state_vector,
    labels_to_indices,
    macro_f1,
    require_torch,
)


def _features(occ="king", traits=("aggressive", "wrathful"), thi=0.86):
    return {
        "categorical": {
            "occ": occ,
            "arch": "aggressive",
            "faction": "merchantguild",
            "sched_act": "social",
            "goals_top": "findwater",
        },
        "multi": {
            "traits": sorted(traits),
            "inv": ["food", "water"],
        },
        "continuous": {
            "hp": 101.5,
            "hp_max": 120.0,
            "thi": thi,
            "b5_c": 0.38,
            "max_threat": 0.0,
        },
    }


def _sample(action_id="drink", mood="fearful", **feature_overrides):
    return {
        "features": _features(**feature_overrides),
        "label": {"action_id": action_id, "zone": "survival_override"},
        "aux": {"mood": mood},
    }


def test_feature_spec_encodes_v2_feature_groups():
    samples = [_sample(), _sample(action_id="eat", occ="guard", traits=("brave",))]
    spec = build_feature_spec(samples)

    restored = FeatureSpec.from_dict(spec.to_dict())
    vector = encode_state_vector(_sample()["features"], restored)

    assert len(vector) == restored.input_dim
    # Continuous features come first, z-scored with train statistics.
    for position, name in enumerate(restored.continuous):
        stats = restored.continuous_stats[name]
        raw = _sample()["features"]["continuous"][name]
        expected = (raw - stats["mean"]) / stats["std"] if stats["std"] else 0.0
        assert vector[position] == pytest.approx(expected)
    # Multi-hot: both traits of the first sample are active.
    trait_vocab = restored.multi["traits"]
    assert trait_vocab["aggressive"] >= 0 and trait_vocab["wrathful"] >= 0
    offset = len(restored.continuous) + sum(len(v) for v in restored.categorical.values())
    for name, vocab in restored.multi.items():
        if name == "traits":
            break
        offset += len(vocab)
    trait_slice = vector[offset:offset + len(trait_vocab)]
    assert sum(trait_slice) == 2.0


def test_continuous_stats_standardize_large_scale_features():
    samples = [_sample(thi=0.2), _sample(thi=0.8)]
    spec = build_feature_spec(samples)

    stats = spec.continuous_stats["thi"]
    assert stats["mean"] == pytest.approx(0.5)
    assert stats["std"] == pytest.approx(0.3)

    vector = encode_state_vector(_features(thi=0.8), spec)
    thi_position = spec.continuous.index("thi")
    assert vector[thi_position] == pytest.approx(1.0)

    # Constant features (zero std) encode as 0 instead of exploding.
    hp_position = spec.continuous.index("hp")
    assert vector[hp_position] == 0.0


def test_unknown_categorical_value_maps_to_unk():
    spec = build_feature_spec([_sample()])
    vector = encode_state_vector(_features(occ="never_seen_role"), spec)

    occ_vocab = spec.categorical["occ"]
    offset = len(spec.continuous)
    for name, vocab in spec.categorical.items():
        if name == "occ":
            occ_slice = vector[offset:offset + len(vocab)]
            assert occ_slice[occ_vocab["<UNK>"]] == 1.0
            break
        offset += len(vocab)


def test_label_spec_has_action_and_mood_heads():
    samples = [_sample("drink", "fearful"), _sample("eat", "calm")]
    spec = build_label_spec(samples)
    indices = labels_to_indices(_sample("drink", "fearful"), spec)

    assert set(spec.heads) == {ACTION_HEAD, MOOD_HEAD}
    assert spec.labels_for(ACTION_HEAD) == ["<UNK>", "drink", "eat"]
    assert indices[ACTION_HEAD] == spec.heads[ACTION_HEAD]["drink"]
    assert indices[MOOD_HEAD] == spec.heads[MOOD_HEAD]["fearful"]


def test_label_spec_can_exclude_mood_head():
    spec = build_label_spec([_sample()], include_mood=False)
    assert list(spec.heads) == [ACTION_HEAD]


def test_macro_f1_balances_rare_classes():
    # Class 0: P=2/3, R=1 -> F1=0.8. Class 1: never predicted -> F1=0. Macro = 0.4.
    gold = [0, 0, 1]
    predicted = [0, 0, 0]
    assert macro_f1(predicted, gold) == pytest.approx(0.4)
    assert macro_f1([1, 0, 1], [1, 0, 1]) == 1.0


def test_supervised_policy_predicts_from_v2_features(tmp_path):
    torch = pytest.importorskip("torch")
    from backend.behavior.supervised_policy import (
        SupervisedPolicy,
        make_model,
        save_checkpoint,
    )

    samples = [_sample("drink", "fearful"), _sample("eat", "calm", occ="guard")]
    feature_spec = build_feature_spec(samples)
    label_spec = build_label_spec(samples)
    model = make_model(
        input_dim=feature_spec.input_dim,
        head_sizes={head: len(vocab) for head, vocab in label_spec.heads.items()},
        hidden_dim=8,
    )
    save_checkpoint(tmp_path, model, feature_spec, label_spec, hidden_dim=8, metrics={})

    policy = SupervisedPolicy(tmp_path)
    prediction = policy.predict(_sample()["features"])

    assert prediction["action_id"] in label_spec.labels_for(ACTION_HEAD)
    assert prediction["mood"] in label_spec.labels_for(MOOD_HEAD)

    detailed = policy.predict_with_metadata(_sample()["features"])
    assert 0.0 <= detailed["predictions"][ACTION_HEAD]["confidence"] <= 1.0


def test_require_torch_error_is_actionable_when_missing():
    try:
        torch = require_torch()
    except RuntimeError as exc:
        assert "PyTorch is required" in str(exc)
    else:
        assert hasattr(torch, "tensor")
