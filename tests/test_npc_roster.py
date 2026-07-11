import json
from pathlib import Path

from scripts.generate_roster_personas import build_persona


def test_seed_roster_has_demo_scale_and_unique_names():
    seeds = json.loads(Path("data/seeds/example_seeds.json").read_text(encoding="utf-8"))
    names = [seed["name"] for seed in seeds]

    assert len(seeds) == 6
    assert len(names) == len(set(names))
    assert set(names) == {"Asuna", "Lanyan", "Loen", "Frederica", "Nicole", "Sanji"}


def test_deterministic_persona_matches_loader_schema():
    seed = {
        "name": "Testa",
        "occupation": "Archivist",
        "personality_tags": ["meticulous", "secretive"],
        "relationships": {"Aldric": "keeps his contracts"},
        "extra": {"location": "archive", "secret": "hid a ledger"},
    }

    persona = build_persona(seed)

    assert persona["core"]["name"] == "Testa"
    assert isinstance(persona["core"]["speech_style"], str)
    assert isinstance(persona["dynamic"]["current_goal"], str)
    assert persona["memory_log"] == []


def test_roster_persona_files_match_persona_schema():
    persona_files = sorted(Path("data/personas").glob("*.json"))

    assert len(persona_files) == 6
    for path in persona_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["seed"]["name"]
        assert data["core"]["name"]
        assert data["core"]["occupation"]
        assert data["core"]["backstory"]
        assert data["social"]["relationships"] is not None
        assert data["dynamic"]["current_goal"]
        assert isinstance(data.get("memory_log", []), list)
