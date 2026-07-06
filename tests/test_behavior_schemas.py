import pytest

from backend.behavior.schemas import (
    DialogueAct,
    DisclosureLevel,
    Gesture,
    MemoryWriteType,
    NpcEmotion,
    PlayerIntent,
    PolicyAction,
    QuestStage,
    QuestUpdate,
    RelationshipStatus,
    StateFeatures,
)


def test_state_features_json_round_trip_and_clamps_numbers():
    features = StateFeatures(
        player_intent="ask secret",
        quest_stage=QuestStage.NOT_STARTED,
        trust_score=2.5,
        npc_emotion="suspicious",
        relationship="known",
        memory_relevance=-1,
        danger_level=0.75,
        distance_to_player="1.8",
        forbidden_secret_asked=True,
        prompt_injection_detected=False,
        npc_role="blacksmith",
        persona_id="aldric-1",
        location="forge",
        inventory_flags=["iron_key"],
    )

    restored = StateFeatures.from_json(features.to_json())

    assert restored.player_intent is PlayerIntent.ASK_SECRET
    assert restored.quest_stage is QuestStage.NOT_STARTED
    assert restored.trust_score == 1.0
    assert restored.memory_relevance == 0.0
    assert restored.distance_to_player == 1.8
    assert restored.to_dict()["inventory_flags"] == ["iron_key"]


def test_policy_action_json_round_trip():
    action = PolicyAction(
        dialogue_act=DialogueAct.REFUSE,
        emotion=NpcEmotion.SUSPICIOUS,
        disclosure_level=DisclosureLevel.NONE,
        gesture=Gesture.STEP_BACK,
        quest_update=QuestUpdate.NO_CHANGE,
        memory_write_type=MemoryWriteType.SUSPICION,
    )

    restored = PolicyAction.from_json(action.to_json())

    assert restored == action
    assert restored.to_dict()["dialogue_act"] == "refuse"


def test_state_features_reject_unknown_enum_values():
    with pytest.raises(ValueError):
        StateFeatures(player_intent="dance")


def test_state_features_exposes_model_ready_feature_groups():
    features = StateFeatures(
        player_intent=PlayerIntent.ASK_HINT,
        quest_stage=QuestStage.STARTED,
        trust_score=0.4,
        npc_emotion=NpcEmotion.FRIENDLY,
        relationship=RelationshipStatus.ALLY,
        forbidden_secret_asked=True,
    )

    assert features.categorical_features["player_intent"] == "ask_hint"
    assert features.continuous_features["trust_score"] == 0.4
    assert features.continuous_features["forbidden_secret_asked"] == 1.0
