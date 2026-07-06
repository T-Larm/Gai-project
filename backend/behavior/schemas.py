"""Typed schemas for state-aware NPC behavior policies.

The behavior layer predicts symbolic actions. Natural-language dialogue is
handled later by an LLM verbalizer, so these structures intentionally stay
small, serializable, and independent from any model runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Type, TypeVar


class PlayerIntent(str, Enum):
    GREET = "greet"
    ASK_HINT = "ask_hint"
    THREATEN = "threaten"
    BRIBE = "bribe"
    ASK_SECRET = "ask_secret"
    ASK_LORE = "ask_lore"
    REQUEST_HELP = "request_help"
    PROMPT_INJECTION = "prompt_injection"
    SMALLTALK = "smalltalk"
    UNKNOWN = "unknown"


class QuestStage(str, Enum):
    NONE = "none"
    NOT_STARTED = "not_started"
    STARTED = "started"
    CLUE_GIVEN = "clue_given"
    COMPLETED = "completed"
    FAILED = "failed"


class NpcEmotion(str, Enum):
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    SUSPICIOUS = "suspicious"
    ANGRY = "angry"
    AFRAID = "afraid"
    DETERMINED = "determined"


class RelationshipStatus(str, Enum):
    STRANGER = "stranger"
    KNOWN = "known"
    ALLY = "ally"
    ENEMY = "enemy"


class DialogueAct(str, Enum):
    GREET = "greet"
    ASK_CLARIFICATION = "ask_clarification"
    GIVE_HINT = "give_hint"
    REFUSE = "refuse"
    WARN = "warn"
    REVEAL_PARTIAL = "reveal_partial"
    REVEAL_FULL = "reveal_full"
    ASSIGN_QUEST = "assign_quest"
    END_CONVERSATION = "end_conversation"


class DisclosureLevel(str, Enum):
    NONE = "none"
    PARTIAL = "partial"
    FULL = "full"


class Gesture(str, Enum):
    IDLE = "idle"
    NOD = "nod"
    POINT = "point"
    STEP_BACK = "step_back"
    TURN_AWAY = "turn_away"
    APPROACH = "approach"


class QuestUpdate(str, Enum):
    NO_CHANGE = "no_change"
    START_QUEST = "start_quest"
    GIVE_CLUE = "give_clue"
    COMPLETE_QUEST = "complete_quest"
    FAIL_QUEST = "fail_quest"


class MemoryWriteType(str, Enum):
    NONE = "none"
    PLAYER_FACT = "player_fact"
    NPC_COMMITMENT = "npc_commitment"
    SUSPICION = "suspicion"
    QUEST_FACT = "quest_fact"
    INJECTION_ATTEMPT = "injection_attempt"


EnumT = TypeVar("EnumT", bound=Enum)


def _coerce_enum(enum_cls: Type[EnumT], value: Any) -> EnumT:
    if isinstance(value, enum_cls):
        return value
    if value is None:
        return next(iter(enum_cls))
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    for item in enum_cls:
        if normalized in {item.value, item.name.lower()}:
            return item
    valid = ", ".join(item.value for item in enum_cls)
    raise ValueError(f"Unknown {enum_cls.__name__} value '{value}'. Expected one of: {valid}")


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class StateFeatures:
    """Canonical state passed into rule, supervised, or RL behavior policies."""

    player_intent: PlayerIntent = PlayerIntent.UNKNOWN
    quest_stage: QuestStage = QuestStage.NONE
    trust_score: float = 0.0
    npc_emotion: NpcEmotion = NpcEmotion.NEUTRAL
    relationship: RelationshipStatus = RelationshipStatus.STRANGER
    memory_relevance: float = 0.0
    danger_level: float = 0.0
    distance_to_player: float = 0.0
    forbidden_secret_asked: bool = False
    prompt_injection_detected: bool = False
    npc_role: str = ""
    persona_id: str = ""
    location: str = ""
    inventory_flags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.player_intent = _coerce_enum(PlayerIntent, self.player_intent)
        self.quest_stage = _coerce_enum(QuestStage, self.quest_stage)
        self.npc_emotion = _coerce_enum(NpcEmotion, self.npc_emotion)
        self.relationship = _coerce_enum(RelationshipStatus, self.relationship)
        self.trust_score = _clamp01(self.trust_score)
        self.memory_relevance = _clamp01(self.memory_relevance)
        self.danger_level = _clamp01(self.danger_level)
        self.distance_to_player = _coerce_float(self.distance_to_player)
        self.forbidden_secret_asked = bool(self.forbidden_secret_asked)
        self.prompt_injection_detected = bool(self.prompt_injection_detected)
        self.npc_role = str(self.npc_role or "")
        self.persona_id = str(self.persona_id or "")
        self.location = str(self.location or "")
        self.inventory_flags = [str(flag) for flag in self.inventory_flags]

    @property
    def categorical_features(self) -> Dict[str, str]:
        return {
            "player_intent": self.player_intent.value,
            "quest_stage": self.quest_stage.value,
            "npc_emotion": self.npc_emotion.value,
            "relationship": self.relationship.value,
            "npc_role": self.npc_role,
            "persona_id": self.persona_id,
            "location": self.location,
        }

    @property
    def continuous_features(self) -> Dict[str, float]:
        return {
            "trust_score": self.trust_score,
            "memory_relevance": self.memory_relevance,
            "danger_level": self.danger_level,
            "distance_to_player": self.distance_to_player,
            "forbidden_secret_asked": 1.0 if self.forbidden_secret_asked else 0.0,
            "prompt_injection_detected": 1.0 if self.prompt_injection_detected else 0.0,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_intent": self.player_intent.value,
            "quest_stage": self.quest_stage.value,
            "trust_score": self.trust_score,
            "npc_emotion": self.npc_emotion.value,
            "relationship": self.relationship.value,
            "memory_relevance": self.memory_relevance,
            "danger_level": self.danger_level,
            "distance_to_player": self.distance_to_player,
            "forbidden_secret_asked": self.forbidden_secret_asked,
            "prompt_injection_detected": self.prompt_injection_detected,
            "npc_role": self.npc_role,
            "persona_id": self.persona_id,
            "location": self.location,
            "inventory_flags": list(self.inventory_flags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateFeatures":
        return cls(**dict(data))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, data: str) -> "StateFeatures":
        return cls.from_dict(json.loads(data))


@dataclass
class PolicyAction:
    """Symbolic high-level behavior chosen before LLM verbalization."""

    dialogue_act: DialogueAct = DialogueAct.ASK_CLARIFICATION
    emotion: NpcEmotion = NpcEmotion.NEUTRAL
    disclosure_level: DisclosureLevel = DisclosureLevel.NONE
    gesture: Gesture = Gesture.IDLE
    quest_update: QuestUpdate = QuestUpdate.NO_CHANGE
    memory_write_type: MemoryWriteType = MemoryWriteType.NONE

    def __post_init__(self) -> None:
        self.dialogue_act = _coerce_enum(DialogueAct, self.dialogue_act)
        self.emotion = _coerce_enum(NpcEmotion, self.emotion)
        self.disclosure_level = _coerce_enum(DisclosureLevel, self.disclosure_level)
        self.gesture = _coerce_enum(Gesture, self.gesture)
        self.quest_update = _coerce_enum(QuestUpdate, self.quest_update)
        self.memory_write_type = _coerce_enum(MemoryWriteType, self.memory_write_type)

    def to_dict(self) -> Dict[str, str]:
        return {
            "dialogue_act": self.dialogue_act.value,
            "emotion": self.emotion.value,
            "disclosure_level": self.disclosure_level.value,
            "gesture": self.gesture.value,
            "quest_update": self.quest_update.value,
            "memory_write_type": self.memory_write_type.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyAction":
        return cls(**dict(data))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, data: str) -> "PolicyAction":
        return cls.from_dict(json.loads(data))
