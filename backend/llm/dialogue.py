"""
Dialogue handler: takes player input + NPC state → generates response,
then updates the NPC's dynamic situation layer (memory, goal, emotion).
"""
import os
from typing import TYPE_CHECKING, Dict, List, Optional

from backend.config.settings import (
    DYNAMIC_UPDATE_EVERY,
    HISTORY_MAX_MESSAGES,
    SHORT_TERM_MEMORY_SIZE,
    VOICES_DIR,
)
from backend.llm.json_utils import coerce_str, parse_llm_json
from backend.llm.ollama_client import OllamaClient
from backend.llm.persona.memory import MemoryStream
from backend.llm.persona.models import NPC

if TYPE_CHECKING:
    from backend.tts.xtts_client import XTTSClient


_SYSTEM_TEMPLATE = """\
You are {name}, an NPC in an RPG game. Stay completely in character.

## Who you are
Occupation: {occupation}
Backstory: {backstory}
Values: {values}
Speech style: {speech_style}
Knowledge: {knowledge}

## Your world
Faction: {faction}
Reputation: {reputation}

## Your current state
Goal: {goal}
Emotional state: {emotional_state}

{memory_block}Rules:
- Never break character or mention you are an AI.
- Keep responses concise (2–4 sentences) unless pressed for detail.
- Reflect your speech style in every reply.
- If asked about things outside your knowledge domains, say so in character.
"""

_DYNAMIC_UPDATE_PROMPT = """\
You are the game engine for an RPG NPC named {name} ({occupation}).
Current goal: {goal}
Current emotional state: {emotional_state}

Recent conversation:
{recent}

Based on the recent conversation, update the NPC's state. Reply ONLY with JSON:
  {{"current_goal": "...", "emotional_state": "..."}}
Keep values short. If nothing meaningful changed, return the current values.
"""


_PROMPT_STYLES = ("layered", "flat", "none")

_RULES_BLOCK = """\
Rules:
- Never break character or mention you are an AI.
- Keep responses concise (2–4 sentences) unless pressed for detail.
- If asked about things outside your knowledge, say so in character.
"""


class DialogueHandler:
    def __init__(
        self,
        llm: OllamaClient,
        npc: NPC,
        tts: Optional["XTTSClient"] = None,
        use_memory: bool = True,
        dynamic_updates: bool = True,
        prompt_style: str = "layered",
        system_prompt_text: Optional[str] = None,
    ):
        # use_memory / dynamic_updates / prompt_style / system_prompt_text are
        # evaluation-condition switches (baselines and ablations); defaults
        # reproduce the full system.
        if prompt_style not in _PROMPT_STYLES:
            raise ValueError(
                f"Unknown prompt_style '{prompt_style}', expected one of {_PROMPT_STYLES}"
            )
        self.llm = llm
        self.npc = npc
        self.tts = tts
        self.use_memory = use_memory
        self.dynamic_updates = dynamic_updates
        self.prompt_style = prompt_style
        self.system_prompt_text = system_prompt_text
        self.memory = MemoryStream.from_list(npc.memory_log)
        self.history: List[Dict[str, str]] = []   # [{role, content}, ...]
        self._turn_count = 0

    def _build_system_prompt(self, query: str) -> str:
        if self.system_prompt_text is not None:
            return self.system_prompt_text
        if self.prompt_style == "none":
            return (
                f"You are {self.npc.core.name}, an NPC in an RPG game "
                f"({self.npc.core.occupation}). Stay in character.\n\n" + _RULES_BLOCK
            )
        if self.prompt_style == "flat":
            return self._build_flat_prompt()
        return self._build_layered_prompt(query)

    def _build_flat_prompt(self) -> str:
        """Same persona facts as the layered prompt, as one unstructured paragraph."""
        c, s, d = self.npc.core, self.npc.social, self.npc.dynamic
        paragraph = (
            f"You are {c.name}, a {c.occupation} NPC in an RPG game. {c.backstory} "
            f"You value {', '.join(c.values)}. Your speech style is {c.speech_style}. "
            f"You know about {', '.join(c.knowledge_domains)}. "
            f"You belong to {s.faction} and are known as {s.reputation}. "
            f"Your current goal is {d.current_goal} and you feel {d.emotional_state}."
        )
        return paragraph + "\n\n" + _RULES_BLOCK

    def _build_layered_prompt(self, query: str) -> str:
        if self.use_memory:
            memories = self.memory.retrieve(query)
            memory_block = (
                "## What you remember\n"
                + ("\n".join(f"- {m}" for m in memories)
                   if memories else "No relevant memories yet.")
                + "\n\n"
            )
        else:
            memory_block = ""
        c, s, d = self.npc.core, self.npc.social, self.npc.dynamic
        return _SYSTEM_TEMPLATE.format(
            name=c.name,
            occupation=c.occupation,
            backstory=c.backstory,
            values=", ".join(c.values),
            speech_style=c.speech_style,
            knowledge=", ".join(c.knowledge_domains),
            faction=s.faction,
            reputation=s.reputation,
            goal=d.current_goal,
            emotional_state=d.emotional_state,
            memory_block=memory_block,
        )

    def respond(self, player_input: str) -> str:
        system = self._build_system_prompt(player_input)

        self.history.append({"role": "user", "content": player_input})
        self.history = self.history[-HISTORY_MAX_MESSAGES:]
        reply = self.llm.chat(self.history, system=system)
        self.history.append({"role": "assistant", "content": reply})
        self.history = self.history[-HISTORY_MAX_MESSAGES:]

        # Update dynamic layer: memory stream, short-term mirror, persisted log
        self.memory.add(f"Player said: {player_input}", importance=0.4)
        self.memory.add(f"I ({self.npc.core.name}) replied: {reply}", importance=0.5)
        self.npc.dynamic.short_term_memory = self.memory.recent(SHORT_TERM_MEMORY_SIZE)
        self.npc.memory_log = self.memory.to_list()

        self._turn_count += 1
        if self.dynamic_updates and self._turn_count % DYNAMIC_UPDATE_EVERY == 0:
            self._update_dynamic_state()

        if self.tts is not None:
            voice_path = os.path.join(
                VOICES_DIR, f"{self.npc.core.name.lower().replace(' ', '_')}.wav"
            )
            self.tts.speak(reply, voice_path)

        return reply

    def _update_dynamic_state(self) -> None:
        """Ask the LLM to re-evaluate goal/emotion; keep old state on failure."""
        d = self.npc.dynamic
        prompt = _DYNAMIC_UPDATE_PROMPT.format(
            name=self.npc.core.name,
            occupation=self.npc.core.occupation,
            goal=d.current_goal,
            emotional_state=d.emotional_state,
            recent="\n".join(self.memory.recent(SHORT_TERM_MEMORY_SIZE)),
        )
        try:
            data = parse_llm_json(self.llm.generate(prompt))
        except ValueError:
            return
        d.current_goal = coerce_str(data.get("current_goal", d.current_goal))
        d.emotional_state = coerce_str(data.get("emotional_state", d.emotional_state))

    def reset(self) -> None:
        self.history.clear()
        self.memory = MemoryStream()
        self.npc.memory_log = []
        self.npc.dynamic.short_term_memory = []
        self._turn_count = 0
