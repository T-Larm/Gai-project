import os

from backend.config.settings import VOICES_DIR
from backend.llm.dialogue import DialogueHandler
from backend.llm.persona.models import (
    CorePersona, DynamicSituation, NPC, PersonaSeed, SocialPersona,
)


class _FakeLLM:
    def chat(self, messages, system=""):
        return "A fine day to you too."


class _FakeTTS:
    def __init__(self):
        self.calls = []

    def speak(self, text, speaker_wav):
        self.calls.append((text, speaker_wav))


def _make_npc(name: str) -> NPC:
    seed = PersonaSeed(name=name, occupation="Blacksmith", personality_tags=["gruff"], relationships={})
    core = CorePersona(
        name=name, occupation="Blacksmith", backstory="A sturdy smith.",
        values=["honesty"], speech_style="gruff", knowledge_domains=["smithing"],
    )
    social = SocialPersona(relationships={}, faction="Town", reputation="respected")
    dynamic = DynamicSituation(current_goal="work", emotional_state="neutral")
    return NPC(seed=seed, core=core, social=social, dynamic=dynamic)


def test_respond_calls_tts_speak_with_reply_and_voice_path_when_tts_provided():
    npc = _make_npc("Aldric")
    tts = _FakeTTS()
    handler = DialogueHandler(_FakeLLM(), npc, tts=tts)

    reply = handler.respond("Hello there")

    assert tts.calls == [(reply, os.path.join(VOICES_DIR, "aldric.wav"))]


def test_respond_resolves_voice_path_with_underscored_name_for_multiword_npc():
    npc = _make_npc("Lord Vane")
    tts = _FakeTTS()
    handler = DialogueHandler(_FakeLLM(), npc, tts=tts)

    handler.respond("Greetings")

    assert tts.calls[0][1] == os.path.join(VOICES_DIR, "lord_vane.wav")


def test_respond_does_not_call_tts_when_none():
    npc = _make_npc("Aldric")
    handler = DialogueHandler(_FakeLLM(), npc)  # tts defaults to None

    reply = handler.respond("Hello there")

    assert isinstance(reply, str) and reply
