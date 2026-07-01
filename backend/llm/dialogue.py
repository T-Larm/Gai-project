"""
Dialogue handler: takes player input + NPC state → generates response,
then updates the NPC's dynamic situation layer.
"""
from typing import List, Dict

from backend.llm.ollama_client import OllamaClient
from backend.llm.persona.memory import MemoryStream
from backend.llm.persona.models import NPC


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

## What you remember
{memories}

Rules:
- Never break character or mention you are an AI.
- Keep responses concise (2–4 sentences) unless pressed for detail.
- Reflect your speech style in every reply.
- If asked about things outside your knowledge domains, say so in character.
"""


class DialogueHandler:
    def __init__(self, llm: OllamaClient, npc: NPC):
        self.llm = llm
        self.npc = npc
        self.memory = MemoryStream()
        self.history: List[Dict[str, str]] = []   # [{role, content}, ...]

    def _build_system_prompt(self, query: str) -> str:
        memories = self.memory.retrieve(query)
        memory_block = (
            "\n".join(f"- {m}" for m in memories)
            if memories
            else "No relevant memories yet."
        )
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
            memories=memory_block,
        )

    def respond(self, player_input: str) -> str:
        system = self._build_system_prompt(player_input)

        self.history.append({"role": "user", "content": player_input})
        reply = self.llm.chat(self.history, system=system)
        self.history.append({"role": "assistant", "content": reply})

        # Update dynamic layer: store what the NPC just said
        self.memory.add(f"Player said: {player_input}", importance=0.4)
        self.memory.add(f"I ({self.npc.core.name}) replied: {reply}", importance=0.5)
        self.npc.dynamic.short_term_memory = self.memory.retrieve("", top_k=5)

        return reply

    def reset(self) -> None:
        self.history.clear()
        self.memory = MemoryStream()
        self.npc.dynamic.short_term_memory = []
