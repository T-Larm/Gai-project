"""Experiment-condition switches on DialogueHandler (evaluation baselines)."""
import pytest
import torch

import backend.llm.persona.memory as memory_module
from backend.config.settings import DYNAMIC_UPDATE_EVERY
from backend.llm.dialogue import DialogueHandler
from backend.llm.persona.models import (
    CorePersona, DynamicSituation, NPC, PersonaSeed, SocialPersona,
)


class _ConstantEmbedder:
    def encode(self, texts, convert_to_tensor=True):
        if isinstance(texts, str):
            return torch.ones(1)
        return torch.ones(len(texts), 1)


class _FakeLLM:
    def __init__(self):
        self.systems = []
        self.generate_calls = []

    def chat(self, messages, system=""):
        self.systems.append(system)
        return "Aye."

    def generate(self, prompt, system=""):
        self.generate_calls.append(prompt)
        return '{"current_goal": "x", "emotional_state": "y"}'


def _make_npc() -> NPC:
    return NPC(
        seed=PersonaSeed(name="Aldric", occupation="Blacksmith",
                         personality_tags=["gruff"], relationships={}),
        core=CorePersona(name="Aldric", occupation="Blacksmith",
                         backstory="Forged in the war of the northern marches.",
                         values=["honesty"], speech_style="gruff",
                         knowledge_domains=["smithing"]),
        social=SocialPersona(relationships={}, faction="Guild", reputation="Solid"),
        dynamic=DynamicSituation(current_goal="Sell swords", emotional_state="neutral"),
    )


@pytest.fixture(autouse=True)
def constant_embedder(monkeypatch):
    monkeypatch.setattr(memory_module, "_get_embedder", lambda: _ConstantEmbedder())


def test_default_condition_is_layered_with_memory_and_dynamic():
    llm = _FakeLLM()
    handler = DialogueHandler(llm, _make_npc())

    handler.respond("hello")

    assert "## Who you are" in llm.systems[0]
    assert "## What you remember" in llm.systems[0]


def test_no_memory_condition_omits_memory_block_but_keeps_persona():
    llm = _FakeLLM()
    handler = DialogueHandler(llm, _make_npc(), use_memory=False)

    handler.respond("hello")
    handler.respond("hello again")

    for system in llm.systems:
        assert "## What you remember" not in system
        assert "## Who you are" in system


def test_dynamic_updates_can_be_disabled():
    llm = _FakeLLM()
    npc = _make_npc()
    handler = DialogueHandler(llm, npc, dynamic_updates=False)

    for _ in range(DYNAMIC_UPDATE_EVERY):
        handler.respond("hello")

    assert llm.generate_calls == []
    assert npc.dynamic.current_goal == "Sell swords"


def test_flat_prompt_style_has_same_facts_without_section_structure():
    llm = _FakeLLM()
    handler = DialogueHandler(llm, _make_npc(), prompt_style="flat")

    handler.respond("hello")

    system = llm.systems[0]
    assert "## Who you are" not in system
    assert "##" not in system
    assert "Aldric" in system
    assert "Forged in the war of the northern marches." in system
    assert "gruff" in system


def test_none_prompt_style_gives_only_name_and_role():
    llm = _FakeLLM()
    handler = DialogueHandler(llm, _make_npc(), prompt_style="none")

    handler.respond("hello")

    system = llm.systems[0]
    assert "Aldric" in system
    assert "Forged in the war of the northern marches." not in system
    assert "gruff" not in system


def test_system_prompt_text_overrides_everything():
    llm = _FakeLLM()
    handler = DialogueHandler(
        llm, _make_npc(),
        system_prompt_text="You are Aldric, a grumpy but kind-hearted smith.",
    )

    handler.respond("hello")

    assert llm.systems[0] == "You are Aldric, a grumpy but kind-hearted smith."


def test_unknown_prompt_style_raises():
    with pytest.raises(ValueError):
        DialogueHandler(_FakeLLM(), _make_npc(), prompt_style="fancy")
