import json

import pytest

from backend.behavior.schemas import StateFeatures
from backend.behavior.supervised_policy import (
    FeatureSpec,
    LabelSpec,
    build_feature_spec,
    build_label_spec,
    encode_state_vector,
    labels_to_indices,
    require_torch,
)


def _sample(action_id="drink"):
    state = StateFeatures(
        player_intent="unknown",
        quest_stage="none",
        trust_score=0.25,
        npc_emotion="neutral",
        relationship="known",
        memory_relevance=0.7,
        danger_level=0.1,
        distance_to_player=0.0,
        npc_role="blacksmith",
        persona_id="npc_1",
        location="forge",
    ).to_dict()
    return {
        "state": state,
        "action": {
            "dialogue_act": "end_conversation",
            "emotion": "neutral",
            "disclosure_level": "none",
            "gesture": "idle",
            "quest_update": "no_change",
        },
        "source_action": {"action_id": action_id},
    }


def test_feature_spec_round_trip_and_vector_length():
    spec = build_feature_spec([_sample("drink"), _sample("eat")])

    restored = FeatureSpec.from_dict(spec.to_dict())
    vector = encode_state_vector(_sample()["state"], restored)

    assert len(vector) == restored.input_dim
    assert vector[0] == 0.25
    assert restored.categorical["npc_role"]["blacksmith"] >= 0


def test_label_spec_maps_policy_and_source_action_heads():
    spec = build_label_spec([_sample("drink"), _sample("eat")])
    indices = labels_to_indices(_sample("drink"), spec)

    assert "dialogue_act" in indices
    assert "source_action_id" in indices
    assert spec.labels_for("source_action_id") == ["<UNK>", "drink", "eat"]


def test_label_spec_can_exclude_source_action_head():
    spec = build_label_spec([_sample("drink")], include_source_action=False)

    assert "source_action_id" not in spec.heads


def test_require_torch_error_is_actionable_when_missing():
    try:
        torch = require_torch()
    except RuntimeError as exc:
        assert "PyTorch is required" in str(exc)
    else:
        assert hasattr(torch, "tensor")
