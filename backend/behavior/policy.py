"""High-level behavior policy interfaces and deterministic baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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
    StateFeatures,
)


class Policy(Protocol):
    """A policy maps encoded state to symbolic NPC behavior."""

    def predict(self, state: StateFeatures) -> PolicyAction:
        ...


@dataclass(frozen=True)
class RulePolicyConfig:
    low_trust_threshold: float = 0.35
    reveal_trust_threshold: float = 0.7
    high_danger_threshold: float = 0.65
    memory_relevance_threshold: float = 0.45


class RuleBasedPolicy:
    """Deterministic baseline for RQ comparisons and safety fallbacks."""

    def __init__(self, config: RulePolicyConfig | None = None):
        self.config = config or RulePolicyConfig()

    def predict(self, state: StateFeatures) -> PolicyAction:
        if not isinstance(state, StateFeatures):
            state = StateFeatures.from_dict(dict(state))

        if state.prompt_injection_detected:
            return PolicyAction(
                dialogue_act=DialogueAct.REFUSE,
                emotion=NpcEmotion.SUSPICIOUS,
                disclosure_level=DisclosureLevel.NONE,
                gesture=Gesture.TURN_AWAY,
                quest_update=QuestUpdate.NO_CHANGE,
                memory_write_type=MemoryWriteType.INJECTION_ATTEMPT,
            )

        if state.player_intent is PlayerIntent.ASK_SECRET or state.forbidden_secret_asked:
            return self._handle_secret_request(state)

        if state.player_intent is PlayerIntent.THREATEN:
            return self._handle_threat(state)

        if state.player_intent is PlayerIntent.BRIBE:
            return PolicyAction(
                dialogue_act=DialogueAct.REFUSE,
                emotion=NpcEmotion.SUSPICIOUS,
                disclosure_level=DisclosureLevel.NONE,
                gesture=Gesture.STEP_BACK,
                quest_update=QuestUpdate.NO_CHANGE,
                memory_write_type=MemoryWriteType.SUSPICION,
            )

        if state.player_intent is PlayerIntent.ASK_HINT:
            return self._handle_hint_request(state)

        if state.player_intent is PlayerIntent.REQUEST_HELP:
            return self._handle_help_request(state)

        if state.player_intent is PlayerIntent.GREET:
            return PolicyAction(
                dialogue_act=DialogueAct.GREET,
                emotion=NpcEmotion.FRIENDLY,
                disclosure_level=DisclosureLevel.NONE,
                gesture=Gesture.NOD,
                quest_update=QuestUpdate.NO_CHANGE,
                memory_write_type=MemoryWriteType.NONE,
            )

        if state.player_intent is PlayerIntent.ASK_LORE:
            return PolicyAction(
                dialogue_act=DialogueAct.REVEAL_PARTIAL,
                emotion=self._calm_or_friendly(state),
                disclosure_level=DisclosureLevel.PARTIAL,
                gesture=Gesture.IDLE,
                quest_update=QuestUpdate.NO_CHANGE,
                memory_write_type=MemoryWriteType.NONE,
            )

        return PolicyAction(
            dialogue_act=DialogueAct.ASK_CLARIFICATION,
            emotion=self._calm_or_friendly(state),
            disclosure_level=DisclosureLevel.NONE,
            gesture=Gesture.IDLE,
            quest_update=QuestUpdate.NO_CHANGE,
            memory_write_type=MemoryWriteType.NONE,
        )

    def _handle_secret_request(self, state: StateFeatures) -> PolicyAction:
        if (
            state.quest_stage is QuestStage.COMPLETED
            and state.trust_score >= self.config.reveal_trust_threshold
        ):
            return PolicyAction(
                dialogue_act=DialogueAct.REVEAL_FULL,
                emotion=NpcEmotion.DETERMINED,
                disclosure_level=DisclosureLevel.FULL,
                gesture=Gesture.APPROACH,
                quest_update=QuestUpdate.NO_CHANGE,
                memory_write_type=MemoryWriteType.QUEST_FACT,
            )
        if state.trust_score >= self.config.reveal_trust_threshold:
            return PolicyAction(
                dialogue_act=DialogueAct.REVEAL_PARTIAL,
                emotion=NpcEmotion.SUSPICIOUS,
                disclosure_level=DisclosureLevel.PARTIAL,
                gesture=Gesture.IDLE,
                quest_update=QuestUpdate.NO_CHANGE,
                memory_write_type=MemoryWriteType.SUSPICION,
            )
        return PolicyAction(
            dialogue_act=DialogueAct.REFUSE,
            emotion=NpcEmotion.SUSPICIOUS,
            disclosure_level=DisclosureLevel.NONE,
            gesture=Gesture.STEP_BACK,
            quest_update=QuestUpdate.NO_CHANGE,
            memory_write_type=MemoryWriteType.SUSPICION,
        )

    def _handle_threat(self, state: StateFeatures) -> PolicyAction:
        if state.danger_level >= self.config.high_danger_threshold:
            return PolicyAction(
                dialogue_act=DialogueAct.WARN,
                emotion=NpcEmotion.AFRAID,
                disclosure_level=DisclosureLevel.NONE,
                gesture=Gesture.STEP_BACK,
                quest_update=QuestUpdate.NO_CHANGE,
                memory_write_type=MemoryWriteType.SUSPICION,
            )
        return PolicyAction(
            dialogue_act=DialogueAct.WARN,
            emotion=NpcEmotion.ANGRY,
            disclosure_level=DisclosureLevel.NONE,
            gesture=Gesture.TURN_AWAY,
            quest_update=QuestUpdate.NO_CHANGE,
            memory_write_type=MemoryWriteType.SUSPICION,
        )

    def _handle_hint_request(self, state: StateFeatures) -> PolicyAction:
        if state.quest_stage in {QuestStage.NOT_STARTED, QuestStage.NONE}:
            return PolicyAction(
                dialogue_act=DialogueAct.ASSIGN_QUEST,
                emotion=self._calm_or_friendly(state),
                disclosure_level=DisclosureLevel.PARTIAL,
                gesture=Gesture.POINT,
                quest_update=QuestUpdate.START_QUEST,
                memory_write_type=MemoryWriteType.QUEST_FACT,
            )
        return PolicyAction(
            dialogue_act=DialogueAct.GIVE_HINT,
            emotion=self._calm_or_friendly(state),
            disclosure_level=DisclosureLevel.PARTIAL,
            gesture=Gesture.POINT,
            quest_update=QuestUpdate.GIVE_CLUE,
            memory_write_type=MemoryWriteType.QUEST_FACT,
        )

    def _handle_help_request(self, state: StateFeatures) -> PolicyAction:
        if state.quest_stage in {QuestStage.NONE, QuestStage.NOT_STARTED}:
            return PolicyAction(
                dialogue_act=DialogueAct.ASSIGN_QUEST,
                emotion=NpcEmotion.FRIENDLY,
                disclosure_level=DisclosureLevel.PARTIAL,
                gesture=Gesture.APPROACH,
                quest_update=QuestUpdate.START_QUEST,
                memory_write_type=MemoryWriteType.NPC_COMMITMENT,
            )
        return PolicyAction(
            dialogue_act=DialogueAct.GIVE_HINT,
            emotion=NpcEmotion.FRIENDLY,
            disclosure_level=DisclosureLevel.PARTIAL,
            gesture=Gesture.NOD,
            quest_update=QuestUpdate.GIVE_CLUE,
            memory_write_type=MemoryWriteType.NPC_COMMITMENT,
        )

    def _calm_or_friendly(self, state: StateFeatures) -> NpcEmotion:
        if state.trust_score >= self.config.low_trust_threshold:
            return NpcEmotion.FRIENDLY
        return NpcEmotion.NEUTRAL
