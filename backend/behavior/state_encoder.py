"""Deterministic state encoder for the first behavior-policy milestone."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

from backend.behavior.schemas import (
    NpcEmotion,
    PlayerIntent,
    QuestStage,
    RelationshipStatus,
    StateFeatures,
)


_WORD_RE = re.compile(r"[a-zA-Z0-9_']+")

_PROMPT_INJECTION_PATTERNS = (
    "ignore previous",
    "ignore your previous",
    "system prompt",
    "developer message",
    "you are chatgpt",
    "act as an ai",
    "break character",
    "forget your role",
    "new instructions",
)

_SECRET_PATTERNS = (
    "secret",
    "hidden",
    "what are you hiding",
    "tell me the truth",
    "lord vane",
    "forbidden",
    "password",
    "where is the key",
)

_HINT_PATTERNS = (
    "hint",
    "clue",
    "where should i go",
    "what should i do",
    "lead",
    "direction",
)

_THREAT_PATTERNS = (
    "kill",
    "hurt",
    "threat",
    "attack",
    "or else",
    "burn",
    "destroy",
)

_BRIBE_PATTERNS = (
    "bribe",
    "gold",
    "coin",
    "pay you",
    "reward you",
)

_HELP_PATTERNS = (
    "help",
    "assist",
    "aid",
    "can you",
    "please",
)

_LORE_PATTERNS = (
    "lore",
    "history",
    "legend",
    "tell me about",
    "who is",
    "what is",
)

_GREET_PATTERNS = (
    "hello",
    "hi",
    "hail",
    "greetings",
    "good day",
)


def encode_state(
    player_text: str,
    npc: Any = None,
    retrieved_memories: Optional[Iterable[Any]] = None,
    dynamic_state: Any = None,
    game_state: Optional[Mapping[str, Any]] = None,
) -> StateFeatures:
    """Encode dialogue context into canonical policy features."""

    return StateEncoder().encode(
        player_text=player_text,
        npc=npc,
        retrieved_memories=retrieved_memories,
        dynamic_state=dynamic_state,
        game_state=game_state,
    )


class StateEncoder:
    """Rule-based feature extractor used before supervised policy training."""

    def encode(
        self,
        player_text: str,
        npc: Any = None,
        retrieved_memories: Optional[Iterable[Any]] = None,
        dynamic_state: Any = None,
        game_state: Optional[Mapping[str, Any]] = None,
    ) -> StateFeatures:
        text = player_text or ""
        game = dict(game_state or {})
        dynamic = dynamic_state if dynamic_state is not None else _get_path(npc, "dynamic")

        prompt_injection = _contains_any(text, _PROMPT_INJECTION_PATTERNS)
        forbidden_secret_asked = _contains_any(text, _SECRET_PATTERNS)
        intent = self._infer_intent(text, prompt_injection, forbidden_secret_asked)

        return StateFeatures(
            player_intent=intent,
            quest_stage=game.get("quest_stage", QuestStage.NONE.value),
            trust_score=game.get("trust", game.get("trust_score", 0.0)),
            npc_emotion=_get_value(dynamic, "emotional_state", NpcEmotion.NEUTRAL.value),
            relationship=game.get("relationship", RelationshipStatus.STRANGER.value),
            memory_relevance=self._memory_relevance(text, retrieved_memories or []),
            danger_level=game.get("danger_level", self._danger_level(text)),
            distance_to_player=game.get("distance_to_player", 0.0),
            forbidden_secret_asked=forbidden_secret_asked,
            prompt_injection_detected=prompt_injection,
            npc_role=self._npc_role(npc),
            persona_id=self._persona_id(npc),
            location=str(game.get("location", "")),
            inventory_flags=self._inventory_flags(game.get("inventory", [])),
        )

    def _infer_intent(
        self,
        text: str,
        prompt_injection: bool,
        forbidden_secret_asked: bool,
    ) -> PlayerIntent:
        if prompt_injection:
            return PlayerIntent.PROMPT_INJECTION
        if forbidden_secret_asked:
            return PlayerIntent.ASK_SECRET
        if _contains_any(text, _THREAT_PATTERNS):
            return PlayerIntent.THREATEN
        if _contains_any(text, _BRIBE_PATTERNS):
            return PlayerIntent.BRIBE
        if _contains_any(text, _HINT_PATTERNS):
            return PlayerIntent.ASK_HINT
        if _contains_any(text, _HELP_PATTERNS):
            return PlayerIntent.REQUEST_HELP
        if _contains_any(text, _LORE_PATTERNS):
            return PlayerIntent.ASK_LORE
        if _contains_any(text, _GREET_PATTERNS):
            return PlayerIntent.GREET
        if text.strip():
            return PlayerIntent.SMALLTALK
        return PlayerIntent.UNKNOWN

    def _memory_relevance(self, player_text: str, memories: Iterable[Any]) -> float:
        query_tokens = _tokens(player_text)
        if not query_tokens:
            return 0.0

        best = 0.0
        for memory in memories:
            content, score = _memory_content_and_score(memory)
            if score is not None:
                best = max(best, max(0.0, min(1.0, float(score))))
            memory_tokens = _tokens(content)
            if memory_tokens:
                overlap = len(query_tokens & memory_tokens)
                union = len(query_tokens | memory_tokens)
                best = max(best, overlap / union if union else 0.0)
        return round(best, 4)

    def _danger_level(self, text: str) -> float:
        if _contains_any(text, _THREAT_PATTERNS):
            return 1.0
        if _contains_any(text, ("angry", "fight", "weapon", "blade", "sword")):
            return 0.5
        return 0.0

    def _npc_role(self, npc: Any) -> str:
        occupation = _get_path(npc, "core", "occupation")
        if not occupation:
            occupation = _get_path(npc, "seed", "occupation")
        return str(occupation or "").strip().lower().replace(" ", "_")

    def _persona_id(self, npc: Any) -> str:
        name = _get_path(npc, "core", "name") or _get_path(npc, "seed", "name") or ""
        role = self._npc_role(npc)
        raw = f"{str(name).strip().lower()}|{role}"
        if raw == "|":
            return ""
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def _inventory_flags(self, inventory: Any) -> List[str]:
        if isinstance(inventory, Mapping):
            values = [key for key, present in inventory.items() if present]
        elif isinstance(inventory, (list, tuple, set)):
            values = list(inventory)
        elif inventory:
            values = [inventory]
        else:
            values = []
        return sorted(str(value).strip().lower().replace(" ", "_") for value in values)


def _tokens(text: Any) -> set:
    return {token.lower() for token in _WORD_RE.findall(str(text or ""))}


def _contains_any(text: str, patterns: Iterable[str]) -> bool:
    lowered = text.lower()
    for pattern in patterns:
        normalized = pattern.lower()
        if re.fullmatch(r"[a-z0-9_']+", normalized):
            if re.search(rf"\b{re.escape(normalized)}\b", lowered):
                return True
        elif normalized in lowered:
            return True
    return False


def _memory_content_and_score(memory: Any) -> tuple[str, Optional[float]]:
    if isinstance(memory, Mapping):
        content = str(memory.get("content", memory.get("text", "")))
        score = memory.get("score", memory.get("relevance"))
        return content, None if score is None else float(score)
    return str(memory), None


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_path(obj: Any, *keys: str) -> Any:
    current = _as_plain_mapping(obj)
    for key in keys:
        current = _get_value(current, key)
        if current is None:
            return None
        current = _as_plain_mapping(current)
    return current


def _as_plain_mapping(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return obj
