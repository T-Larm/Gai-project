from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class PersonaSeed:
    name: str
    occupation: str
    personality_tags: List[str]
    relationships: Dict[str, str]   # other_npc_name -> relationship description
    extra: Dict[str, str] = field(default_factory=dict)  # optional free-form extras
    gender: str = "unspecified"     # "male" / "female"; used to pick a matching placeholder voice


@dataclass
class CorePersona:
    """Stable identity — generated once, rarely changes."""
    name: str
    occupation: str
    backstory: str
    values: List[str]
    speech_style: str               # e.g. "gruff and laconic", "scholarly and verbose"
    knowledge_domains: List[str]    # topics the NPC knows about


@dataclass
class SocialPersona:
    """The NPC's relational world — who they know and how."""
    relationships: Dict[str, str]   # other_npc_name -> rich relationship description
    faction: str
    reputation: str


@dataclass
class DynamicSituation:
    """Runtime state — updated each conversation turn."""
    current_goal: str
    emotional_state: str            # e.g. "neutral", "suspicious", "friendly"
    short_term_memory: List[str] = field(default_factory=list)  # recent statements


@dataclass
class NPC:
    seed: PersonaSeed
    core: CorePersona
    social: SocialPersona
    dynamic: DynamicSituation
    # Serialized MemoryStream entries; persisted with the persona so the NPC
    # remembers past sessions. Kept as plain dicts to keep this module light.
    memory_log: List[Dict] = field(default_factory=list)
