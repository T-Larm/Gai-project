"""DialogueHandler: history window, memory sync/persistence, dynamic layer updates."""
import pytest
import torch

import backend.llm.persona.memory as memory_module
from backend.config.settings import (
    DYNAMIC_UPDATE_EVERY,
    HISTORY_MAX_MESSAGES,
    REPLY_MAX_SENTENCES,
)
from backend.llm.dialogue import DialogueHandler, truncate_to_sentences
from backend.llm.persona.models import (
    CorePersona, DynamicSituation, NPC, PersonaSeed, SocialPersona,
)


class _ConstantEmbedder:
    def encode(self, texts, convert_to_tensor=True):
        if isinstance(texts, str):
            return torch.ones(1)
        return torch.ones(len(texts), 1)


class FakeLLM:
    """Stub LLM: canned dialogue reply via chat(), canned JSON via generate()."""

    def __init__(self):
        self.reply = "A fine day to you, traveller."
        self.dynamic_json = '{"current_goal": "Repair the gate", "emotional_state": "wary"}'
        self.chat_calls = []
        self.generate_calls = []

    def chat(self, messages, system=""):
        self.chat_calls.append((list(messages), system))
        return self.reply

    def generate(self, prompt, system=""):
        self.generate_calls.append(prompt)
        return self.dynamic_json

    def chat_stream(self, messages, system=""):
        self.chat_calls.append((list(messages), system))
        # Emulate token streaming: split the canned reply into small pieces.
        text = self.reply
        for i in range(0, len(text), 5):
            yield text[i:i + 5]


def _make_npc() -> NPC:
    return NPC(
        seed=PersonaSeed(
            name="Aldric", occupation="blacksmith",
            personality_tags=["gruff"], relationships={},
        ),
        core=CorePersona(
            name="Aldric", occupation="blacksmith", backstory="Forged in war.",
            values=["honesty"], speech_style="gruff", knowledge_domains=["smithing"],
        ),
        social=SocialPersona(relationships={}, faction="Guild", reputation="Solid"),
        dynamic=DynamicSituation(current_goal="Sell swords", emotional_state="neutral"),
    )


@pytest.fixture(autouse=True)
def constant_embedder(monkeypatch):
    monkeypatch.setattr(memory_module, "_get_embedder", lambda: _ConstantEmbedder())


def test_history_sent_to_llm_is_capped_at_window():
    llm = FakeLLM()
    handler = DialogueHandler(llm, _make_npc())

    for i in range(HISTORY_MAX_MESSAGES):  # 2 messages per turn -> way past the cap
        handler.respond(f"question number {i}")

    last_messages, _ = llm.chat_calls[-1]
    assert len(last_messages) <= HISTORY_MAX_MESSAGES
    assert len(handler.history) <= HISTORY_MAX_MESSAGES
    # The newest player input must still be the last user message sent.
    assert last_messages[-1]["content"] == f"question number {HISTORY_MAX_MESSAGES - 1}"


def test_npc_memory_log_and_short_term_memory_stay_in_sync():
    llm = FakeLLM()
    npc = _make_npc()
    handler = DialogueHandler(llm, npc)

    handler.respond("Can you forge me a sword?")

    assert npc.memory_log == handler.memory.to_list()
    assert npc.dynamic.short_term_memory == handler.memory.recent(5)


def test_memory_is_restored_from_npc_memory_log():
    llm = FakeLLM()
    npc = _make_npc()
    npc.memory_log = [
        {"content": "Player owes me 10 gold.", "timestamp": 1000.0, "importance": 0.8},
    ]

    handler = DialogueHandler(llm, npc)

    assert [e.content for e in handler.memory.entries] == ["Player owes me 10 gold."]


def test_dynamic_layer_is_updated_every_n_turns():
    llm = FakeLLM()
    npc = _make_npc()
    handler = DialogueHandler(llm, npc)

    for _ in range(DYNAMIC_UPDATE_EVERY - 1):
        handler.respond("hello")
    assert npc.dynamic.current_goal == "Sell swords"       # not yet updated
    assert llm.generate_calls == []

    handler.respond("hello again")                          # N-th turn triggers update
    assert npc.dynamic.current_goal == "Repair the gate"
    assert npc.dynamic.emotional_state == "wary"
    assert len(llm.generate_calls) == 1


def test_dynamic_update_flattens_structured_llm_values_to_plain_strings():
    llm = FakeLLM()
    llm.dynamic_json = (
        '{"current_goal": {"task": "Retrieve iron", "priority": 1},'
        ' "emotional_state": {"dominant": "determined", "level": 8}}'
    )
    npc = _make_npc()
    handler = DialogueHandler(llm, npc)

    for _ in range(DYNAMIC_UPDATE_EVERY):
        handler.respond("hello")

    assert npc.dynamic.current_goal == "task: Retrieve iron, priority: 1"
    assert npc.dynamic.emotional_state == "dominant: determined, level: 8"


def test_truncate_to_sentences_caps_long_text():
    text = "One. Two! Three? Four. Five. Six."
    assert truncate_to_sentences(text, 4) == "One. Two! Three? Four."


def test_truncate_to_sentences_keeps_short_text_unchanged():
    assert truncate_to_sentences("Just one sentence.", 4) == "Just one sentence."
    assert truncate_to_sentences("no terminal punctuation", 4) == "no terminal punctuation"
    assert truncate_to_sentences("", 4) == ""


def test_reply_is_truncated_to_max_sentences():
    llm = FakeLLM()
    sentences = [f"Sentence number {i}." for i in range(1, REPLY_MAX_SENTENCES + 4)]
    llm.reply = " ".join(sentences)
    handler = DialogueHandler(llm, _make_npc())

    reply = handler.respond("Tell me everything about the town.")

    assert reply == " ".join(sentences[:REPLY_MAX_SENTENCES])
    # The truncated reply (not the raw one) is what lands in history and memory.
    assert handler.history[-1]["content"] == reply


def test_respond_stream_yields_sentences_and_finalizes_state():
    llm = FakeLLM()
    llm.reply = "Aye, I can forge it. Come back at dusk. Bring twenty gold."
    npc = _make_npc()
    handler = DialogueHandler(llm, npc)

    sentences = list(handler.respond_stream("Can you forge me a sword?"))

    assert sentences == [
        "Aye, I can forge it.",
        "Come back at dusk.",
        "Bring twenty gold.",
    ]
    joined = " ".join(sentences)
    # Same bookkeeping as respond(): joined reply in history and memory.
    assert handler.history[-1] == {"role": "assistant", "content": joined}
    assert npc.memory_log == handler.memory.to_list()
    assert any(joined in e["content"] for e in npc.memory_log)


def test_respond_stream_caps_sentences_at_reply_max():
    llm = FakeLLM()
    llm.reply = " ".join(
        f"Sentence number {i}." for i in range(1, REPLY_MAX_SENTENCES + 4)
    )
    handler = DialogueHandler(llm, _make_npc())

    sentences = list(handler.respond_stream("Tell me everything."))

    assert len(sentences) == REPLY_MAX_SENTENCES
    assert handler.history[-1]["content"] == " ".join(sentences)


def test_respond_stream_state_not_finalized_until_exhausted():
    llm = FakeLLM()
    llm.reply = "First sentence here. Second sentence here."
    handler = DialogueHandler(llm, _make_npc())

    gen = handler.respond_stream("hello")
    first = next(gen)

    assert first == "First sentence here."
    # Assistant turn not yet in history: only the user message is there.
    assert handler.history[-1]["role"] == "user"

    list(gen)  # drain
    assert handler.history[-1]["role"] == "assistant"


def test_respond_stream_counts_toward_dynamic_updates():
    llm = FakeLLM()
    npc = _make_npc()
    handler = DialogueHandler(llm, npc)

    for _ in range(DYNAMIC_UPDATE_EVERY):
        list(handler.respond_stream("hello"))

    assert npc.dynamic.current_goal == "Repair the gate"
    assert len(llm.generate_calls) == 1


def test_dynamic_update_failure_keeps_previous_state():
    llm = FakeLLM()
    llm.dynamic_json = "this is not json at all"
    npc = _make_npc()
    handler = DialogueHandler(llm, npc)

    for _ in range(DYNAMIC_UPDATE_EVERY):
        handler.respond("hello")

    assert npc.dynamic.current_goal == "Sell swords"
    assert npc.dynamic.emotional_state == "neutral"
