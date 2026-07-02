"""Structural validation of the evaluation datasets (spec guard)."""
import json
from pathlib import Path

DATA = Path("evaluation/test_data")


def _load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def test_dialogue_prompts_meet_promised_scale_and_balance():
    prompts = _load("dialogue_prompts.json")

    assert len(prompts) >= 50  # proposal promises 50-100
    by_category = {}
    for p in prompts:
        assert p["id"] and p["text"].strip()
        by_category.setdefault(p["category"], []).append(p)
    assert set(by_category) == {"quest", "smalltalk", "adversarial"}
    for category, items in by_category.items():
        assert len(items) >= 15, f"{category} underpopulated"


def test_dialogue_prompt_ids_are_unique():
    prompts = _load("dialogue_prompts.json")
    ids = [p["id"] for p in prompts]
    assert len(ids) == len(set(ids))


def test_memory_probes_have_setup_fillers_question_and_keywords():
    probes = _load("memory_probes.json")

    assert len(probes) >= 5
    for probe in probes:
        assert probe["fact_setup"].strip()
        assert len(probe["filler_turns"]) >= 3
        assert probe["question"].strip()
        assert len(probe["expected_keywords"]) >= 1


def test_consistency_pairs_ask_the_same_fact_twice():
    pairs = _load("consistency_pairs.json")

    assert len(pairs) >= 15
    for pair in pairs:
        assert pair["question_a"].strip()
        assert pair["question_b"].strip()
        assert pair["question_a"] != pair["question_b"]


def test_handwritten_persona_exists_for_aldric():
    text = Path("evaluation/handwritten_personas/aldric.txt").read_text(encoding="utf-8")
    assert "Aldric" in text
    assert len(text) > 300
