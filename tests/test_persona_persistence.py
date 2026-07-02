"""PersonaGenerator.save/load must round-trip the NPC including its memory log."""
from backend.llm.persona.generator import PersonaGenerator
from backend.llm.persona.models import (
    CorePersona, DynamicSituation, NPC, PersonaSeed, SocialPersona,
)


def _make_npc() -> NPC:
    return NPC(
        seed=PersonaSeed(
            name="Aldric", occupation="blacksmith",
            personality_tags=["gruff"], relationships={"Mira": "friend"},
        ),
        core=CorePersona(
            name="Aldric", occupation="blacksmith", backstory="Forged in war.",
            values=["honesty"], speech_style="gruff", knowledge_domains=["smithing"],
        ),
        social=SocialPersona(
            relationships={"Mira": "friend"}, faction="Guild", reputation="Solid",
        ),
        dynamic=DynamicSituation(
            current_goal="Sell swords", emotional_state="neutral",
            short_term_memory=["Player said: hello"],
        ),
        memory_log=[
            {"content": "Player said: hello", "timestamp": 1000.0, "importance": 0.4},
            {"content": "I (Aldric) replied: hail", "timestamp": 1001.0, "importance": 0.5},
        ],
    )


def test_save_and_load_round_trips_memory_log(tmp_path):
    npc = _make_npc()
    gen = PersonaGenerator(llm=None)

    path = gen.save(npc, directory=str(tmp_path))
    loaded = PersonaGenerator.load(path)

    assert loaded.memory_log == npc.memory_log
    assert loaded.dynamic.short_term_memory == npc.dynamic.short_term_memory
    assert loaded.core.backstory == "Forged in war."


def test_load_tolerates_persona_files_without_memory_log(tmp_path):
    npc = _make_npc()
    npc.memory_log = []
    gen = PersonaGenerator(llm=None)
    path = gen.save(npc, directory=str(tmp_path))

    loaded = PersonaGenerator.load(path)

    assert loaded.memory_log == []
