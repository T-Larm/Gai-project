"""Evaluation scripts: condition registry, dialogue runner, judge, latency."""
import pytest
import torch

import backend.llm.persona.memory as memory_module
from backend.llm.persona.models import (
    CorePersona, DynamicSituation, NPC, PersonaSeed, SocialPersona,
)
from evaluation.judge_consistency import parse_verdict, summarize_rates
from evaluation.measure_latency import summarize_latency, time_calls
from evaluation.run_dialogues import (
    CONDITIONS, make_handler, run_memory_probes, run_prompts,
)


class _ConstantEmbedder:
    def encode(self, texts, convert_to_tensor=True):
        if isinstance(texts, str):
            return torch.ones(1)
        return torch.ones(len(texts), 1)


class _EchoLLM:
    """Replies with a fixed string; records nothing heavy."""

    def __init__(self, reply="I remember Renn and poor Kira well."):
        self.reply = reply

    def chat(self, messages, system=""):
        return self.reply

    def generate(self, prompt, system=""):
        return '{"current_goal": "x", "emotional_state": "y"}'


def _make_npc() -> NPC:
    return NPC(
        seed=PersonaSeed(name="Aldric", occupation="Blacksmith",
                         personality_tags=["gruff"], relationships={}),
        core=CorePersona(name="Aldric", occupation="Blacksmith", backstory="Smith.",
                         values=["honesty"], speech_style="gruff",
                         knowledge_domains=["smithing"]),
        social=SocialPersona(relationships={}, faction="Guild", reputation="Solid"),
        dynamic=DynamicSituation(current_goal="Sell swords", emotional_state="neutral"),
    )


@pytest.fixture(autouse=True)
def constant_embedder(monkeypatch):
    monkeypatch.setattr(memory_module, "_get_embedder", lambda: _ConstantEmbedder())


def test_condition_registry_covers_planned_baselines():
    assert set(CONDITIONS) == {"full", "no_memory", "flat", "none", "handwritten"}
    assert CONDITIONS["full"]["use_memory"] is True
    assert CONDITIONS["no_memory"]["use_memory"] is False
    assert CONDITIONS["flat"]["prompt_style"] == "flat"
    assert CONDITIONS["none"]["prompt_style"] == "none"


def test_make_handler_applies_condition():
    handler = make_handler(_EchoLLM(), _make_npc(), "no_memory")

    assert handler.use_memory is False
    assert handler.prompt_style == "layered"


def test_make_handler_handwritten_requires_persona_text(tmp_path):
    text_file = tmp_path / "aldric.txt"
    text_file.write_text("You are Aldric, a smith.", encoding="utf-8")

    handler = make_handler(
        _EchoLLM(), _make_npc(), "handwritten", handwritten_dir=str(tmp_path)
    )

    assert handler.system_prompt_text == "You are Aldric, a smith."


def test_run_prompts_produces_one_record_per_prompt_with_latency():
    handler = make_handler(_EchoLLM(), _make_npc(), "full")
    prompts = [
        {"id": "q01", "category": "quest", "text": "Hello?"},
        {"id": "s01", "category": "smalltalk", "text": "Nice day."},
    ]

    records = run_prompts(handler, prompts, npc_name="Aldric", condition="full")

    assert len(records) == 2
    assert records[0]["id"] == "q01"
    assert records[0]["condition"] == "full"
    assert records[0]["reply"]
    assert records[0]["latency_s"] >= 0
    assert records[1]["turn_index"] == 2


def test_run_memory_probes_marks_keyword_hits():
    probes = [{
        "id": "m01",
        "fact_setup": "My name is Renn and my sister Kira is sick.",
        "filler_turns": ["How are you?", "Nice forge.", "Busy day?"],
        "question": "Do you remember my name?",
        "expected_keywords": ["Renn", "Kira"],
    }]

    hit_records = run_memory_probes(
        lambda: make_handler(_EchoLLM("Of course — Renn, and your sister Kira."),
                             _make_npc(), "full"),
        probes, npc_name="Aldric", condition="full",
    )
    miss_records = run_memory_probes(
        lambda: make_handler(_EchoLLM("I have no idea who you are."),
                             _make_npc(), "full"),
        probes, npc_name="Aldric", condition="full",
    )

    assert hit_records[0]["hit"] is True
    assert miss_records[0]["hit"] is False


def test_parse_verdict_is_robust_to_judge_phrasing():
    assert parse_verdict("YES") is True
    assert parse_verdict("yes, the reply fits the persona.") is True
    assert parse_verdict(" No.") is False
    assert parse_verdict("NO — it mentions being an AI.") is False
    assert parse_verdict("Maybe? Hard to tell.") is None


def test_summarize_rates_groups_by_condition_and_category():
    judged = [
        {"condition": "full", "category": "quest", "in_character": True},
        {"condition": "full", "category": "quest", "in_character": False},
        {"condition": "none", "category": "quest", "in_character": False},
        {"condition": "full", "category": "adversarial", "in_character": True},
    ]

    summary = summarize_rates(judged)

    assert summary[("full", "quest")] == pytest.approx(0.5)
    assert summary[("none", "quest")] == pytest.approx(0.0)
    assert summary[("full", "adversarial")] == pytest.approx(1.0)


def test_time_calls_and_summarize_latency():
    times = time_calls(lambda: None, n=5)
    assert len(times) == 5

    stats = summarize_latency({"llm": [1.0, 2.0, 3.0]})
    assert stats["llm"]["mean"] == pytest.approx(2.0)
    assert stats["llm"]["n"] == 3
    assert stats["llm"]["std"] >= 0
