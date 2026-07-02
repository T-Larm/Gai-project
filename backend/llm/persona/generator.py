"""
Offline step: generate the three-layer NPC persona from a minimal seed.
Run once before the game starts; result is saved to data/personas/.
"""
import json
import os
from pathlib import Path

from backend.llm.json_utils import (
    coerce_str as _to_str,
    coerce_str_list as _to_str_list,
    parse_llm_json,
)
from backend.llm.ollama_client import OllamaClient
from backend.llm.persona.models import (
    CorePersona, DynamicSituation, NPC, PersonaSeed, SocialPersona,
)
from backend.config.settings import PERSONAS_DIR


_CORE_PROMPT = """\
You are a game designer creating an NPC for an RPG game.
Given the seed below, generate a detailed Core Persona as JSON with these exact keys:
  backstory, values (list), speech_style, knowledge_domains (list)

Seed:
  Name: {name}
  Occupation: {occupation}
  Personality: {personality}

Reply ONLY with valid JSON. No markdown fences, no extra text.
"""

_SOCIAL_PROMPT = """\
You are a game designer. Given the NPC info below, generate a Social Persona as JSON with:
  faction, reputation, relationships (object mapping each name to a rich description)

NPC: {name}, {occupation}
Known relationships: {relationships}

Reply ONLY with valid JSON. No markdown fences, no extra text.
"""

_DYNAMIC_PROMPT = """\
You are a game designer. Given the NPC info below, generate an initial Dynamic Situation as JSON with:
  current_goal, emotional_state

NPC: {name}, {occupation}
Backstory summary: {backstory_snippet}

Reply ONLY with valid JSON. No markdown fences, no extra text.
"""


class PersonaGenerator:
    def __init__(self, llm: OllamaClient):
        self.llm = llm

    def _parse_json(self, raw: str) -> dict:
        return parse_llm_json(raw, llm=self.llm)

    def generate(self, seed: PersonaSeed) -> NPC:
        print(f"[Persona] Generating persona for '{seed.name}'...")

        # --- Core Persona ---
        core_raw = self.llm.generate(
            _CORE_PROMPT.format(
                name=seed.name,
                occupation=seed.occupation,
                personality=", ".join(seed.personality_tags),
            )
        )
        core_data = self._parse_json(core_raw)
        core = CorePersona(
            name=seed.name,
            occupation=seed.occupation,
            backstory=_to_str(core_data.get("backstory", "")),
            values=_to_str_list(core_data.get("values", [])),
            speech_style=_to_str(core_data.get("speech_style", "neutral")),
            knowledge_domains=_to_str_list(core_data.get("knowledge_domains", [])),
        )

        # --- Social Persona ---
        social_raw = self.llm.generate(
            _SOCIAL_PROMPT.format(
                name=seed.name,
                occupation=seed.occupation,
                relationships=json.dumps(seed.relationships),
            )
        )
        social_data = self._parse_json(social_raw)
        social = SocialPersona(
            relationships=social_data.get("relationships", seed.relationships),
            faction=_to_str(social_data.get("faction", "Independent")),
            reputation=_to_str(social_data.get("reputation", "Unknown")),
        )

        # --- Dynamic Situation ---
        dynamic_raw = self.llm.generate(
            _DYNAMIC_PROMPT.format(
                name=seed.name,
                occupation=seed.occupation,
                backstory_snippet=core.backstory[:200],
            )
        )
        dynamic_data = self._parse_json(dynamic_raw)
        dynamic = DynamicSituation(
            current_goal=_to_str(dynamic_data.get("current_goal", "Attend to daily tasks")),
            emotional_state=_to_str(dynamic_data.get("emotional_state", "neutral")),
        )

        return NPC(seed=seed, core=core, social=social, dynamic=dynamic)

    def save(self, npc: NPC, directory: str = PERSONAS_DIR) -> str:
        Path(directory).mkdir(parents=True, exist_ok=True)
        path = os.path.join(directory, f"{npc.core.name.lower().replace(' ', '_')}.json")
        data = {
            "seed": {
                "name": npc.seed.name,
                "occupation": npc.seed.occupation,
                "personality_tags": npc.seed.personality_tags,
                "relationships": npc.seed.relationships,
                "extra": npc.seed.extra,
            },
            "core": {
                "name": npc.core.name,
                "occupation": npc.core.occupation,
                "backstory": npc.core.backstory,
                "values": npc.core.values,
                "speech_style": npc.core.speech_style,
                "knowledge_domains": npc.core.knowledge_domains,
            },
            "social": {
                "relationships": npc.social.relationships,
                "faction": npc.social.faction,
                "reputation": npc.social.reputation,
            },
            "dynamic": {
                "current_goal": npc.dynamic.current_goal,
                "emotional_state": npc.dynamic.emotional_state,
                "short_term_memory": npc.dynamic.short_term_memory,
            },
            "memory_log": npc.memory_log,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[Persona] Saved to {path}")
        return path

    @staticmethod
    def load(path: str) -> NPC:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        seed = PersonaSeed(**data["seed"])
        c = data["core"]
        core = CorePersona(
            name=c["name"],
            occupation=c["occupation"],
            backstory=_to_str(c.get("backstory", "")),
            values=_to_str_list(c.get("values", [])),
            speech_style=_to_str(c.get("speech_style", "neutral")),
            knowledge_domains=_to_str_list(c.get("knowledge_domains", [])),
        )
        s = data["social"]
        social = SocialPersona(
            relationships=s.get("relationships", {}),
            faction=_to_str(s.get("faction", "Independent")),
            reputation=_to_str(s.get("reputation", "Unknown")),
        )
        d = data["dynamic"]
        dynamic = DynamicSituation(
            current_goal=_to_str(d.get("current_goal", "Attend to daily tasks")),
            emotional_state=_to_str(d.get("emotional_state", "neutral")),
            short_term_memory=d.get("short_term_memory", []),
        )
        return NPC(
            seed=seed, core=core, social=social, dynamic=dynamic,
            memory_log=data.get("memory_log", []),
        )
