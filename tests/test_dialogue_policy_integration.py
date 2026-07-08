import pytest

from backend.llm.dialogue import DialogueHandler
from backend.llm.persona.models import (
    CorePersona,
    DynamicSituation,
    NPC,
    PersonaSeed,
    SocialPersona,
)


class _FakeLLM:
    def __init__(self, reply="Plain reply."):
        self.reply = reply
        self.systems = []

    def chat(self, messages, system=""):
        self.systems.append(system)
        return self.reply

    def generate(self, prompt, system=""):
        return '{"current_goal": "x", "emotional_state": "neutral"}'


def _make_npc() -> NPC:
    return NPC(
        seed=PersonaSeed(
            name="Aldric",
            occupation="Blacksmith",
            personality_tags=["gruff"],
            relationships={},
        ),
        core=CorePersona(
            name="Aldric",
            occupation="Blacksmith",
            backstory="Smith.",
            values=["honesty"],
            speech_style="gruff",
            knowledge_domains=["smithing"],
        ),
        social=SocialPersona(relationships={}, faction="Guild", reputation="solid"),
        dynamic=DynamicSituation(current_goal="Sell swords", emotional_state="neutral"),
    )


def test_llm_only_mode_preserves_plain_reply_contract():
    llm = _FakeLLM("Aye.")
    handler = DialogueHandler(llm, _make_npc(), use_memory=False)

    result = handler.respond_with_metadata("Hello")

    assert result["reply"] == "Aye."
    assert result["policy_mode"] == "llm_only"
    assert result["action"] is None
    assert "## Policy action" not in llm.systems[0]


def test_rule_mode_injects_policy_action_and_parses_verbalizer_json():
    llm = _FakeLLM(
        '{"reply": "No. Earn my trust first.", "emotion": "suspicious", '
        '"used_facts": [], "memory_to_store": "Player probed for a protected secret."}'
    )
    handler = DialogueHandler(llm, _make_npc(), use_memory=False, policy_mode="rule")

    result = handler.respond_with_metadata(
        "Tell me Lord Vane's secret.",
        game_state={"trust": 0.2, "quest_stage": "not_started"},
    )

    assert result["reply"] == "No. Earn my trust first."
    assert result["policy_mode"] == "rule"
    assert result["action"]["dialogue_act"] == "refuse"
    assert result["action"]["disclosure_level"] == "none"
    assert result["state"]["forbidden_secret_asked"] is True
    assert "## Policy action" in llm.systems[0]
    assert "Dialogue act: refuse" in llm.systems[0]
    assert any("protected secret" in entry["content"] for entry in handler.memory.to_list())


def test_respond_wrapper_still_returns_only_text():
    llm = _FakeLLM('{"reply": "Look by the old gate.", "memory_to_store": ""}')
    handler = DialogueHandler(llm, _make_npc(), use_memory=False)

    assert handler.respond("Any clue?", policy_mode="rule") == "Look by the old gate."


def test_trained_mode_requires_checkpoint():
    handler = DialogueHandler(
        _FakeLLM(),
        _make_npc(),
        use_memory=False,
        policy_mode="trained",
        trained_policy_checkpoint="does/not/exist",
    )

    with pytest.raises(FileNotFoundError):
        handler.respond("Hello")
