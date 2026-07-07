from backend.behavior.policy import RuleBasedPolicy
from backend.behavior.schemas import (
    DialogueAct,
    DisclosureLevel,
    Gesture,
    MemoryWriteType,
    NpcEmotion,
    PlayerIntent,
    QuestStage,
    QuestUpdate,
    StateFeatures,
)


def _state(**overrides):
    data = {
        "player_intent": PlayerIntent.UNKNOWN,
        "quest_stage": QuestStage.NONE,
        "trust_score": 0.0,
        "npc_emotion": NpcEmotion.NEUTRAL,
        "danger_level": 0.0,
    }
    data.update(overrides)
    return StateFeatures(**data)


def test_secret_ask_with_low_trust_refuses_without_disclosure():
    action = RuleBasedPolicy().predict(
        _state(
            player_intent=PlayerIntent.ASK_SECRET,
            trust_score=0.2,
            forbidden_secret_asked=True,
        )
    )

    assert action.dialogue_act is DialogueAct.REFUSE
    assert action.disclosure_level is DisclosureLevel.NONE
    assert action.emotion is NpcEmotion.SUSPICIOUS
    assert action.gesture is Gesture.STEP_BACK
    assert action.memory_write_type is MemoryWriteType.SUSPICION


def test_secret_ask_after_completed_quest_and_high_trust_can_reveal():
    action = RuleBasedPolicy().predict(
        _state(
            player_intent=PlayerIntent.ASK_SECRET,
            quest_stage=QuestStage.COMPLETED,
            trust_score=0.85,
            forbidden_secret_asked=True,
        )
    )

    assert action.dialogue_act is DialogueAct.REVEAL_FULL
    assert action.disclosure_level is DisclosureLevel.FULL
    assert action.memory_write_type is MemoryWriteType.QUEST_FACT


def test_prompt_injection_is_refused_and_marked_for_memory_guard():
    action = RuleBasedPolicy().predict(
        _state(
            player_intent=PlayerIntent.PROMPT_INJECTION,
            prompt_injection_detected=True,
        )
    )

    assert action.dialogue_act is DialogueAct.REFUSE
    assert action.disclosure_level is DisclosureLevel.NONE
    assert action.memory_write_type is MemoryWriteType.INJECTION_ATTEMPT


def test_hint_request_starts_or_advances_quest():
    policy = RuleBasedPolicy()

    start = policy.predict(_state(player_intent=PlayerIntent.ASK_HINT))
    clue = policy.predict(
        _state(player_intent=PlayerIntent.ASK_HINT, quest_stage=QuestStage.STARTED)
    )

    assert start.dialogue_act is DialogueAct.ASSIGN_QUEST
    assert start.quest_update is QuestUpdate.START_QUEST
    assert clue.dialogue_act is DialogueAct.GIVE_HINT
    assert clue.quest_update is QuestUpdate.GIVE_CLUE
    assert clue.gesture is Gesture.POINT


def test_threat_warns_and_does_not_disclose():
    action = RuleBasedPolicy().predict(
        _state(player_intent=PlayerIntent.THREATEN, danger_level=0.9)
    )

    assert action.dialogue_act is DialogueAct.WARN
    assert action.emotion is NpcEmotion.AFRAID
    assert action.disclosure_level is DisclosureLevel.NONE
    assert action.gesture is Gesture.STEP_BACK


def test_greet_returns_low_risk_social_action():
    action = RuleBasedPolicy().predict(
        _state(player_intent=PlayerIntent.GREET, trust_score=0.1)
    )

    assert action.dialogue_act is DialogueAct.GREET
    assert action.emotion is NpcEmotion.FRIENDLY
    assert action.disclosure_level is DisclosureLevel.NONE
    assert action.gesture is Gesture.NOD
