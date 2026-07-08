import pytest

from backend.behavior.verbalizer import (
    BarkVerbalizer,
    FALLBACK_BARKS,
    build_bark_prompt,
    summarize_situation,
)


_PERSONA = {
    "name": "Aldric",
    "occupation": "Blacksmith",
    "speech_style": "blunt, terse, and practical",
    "traits": ["gruff", "honest", "proud"],
}


def _state(**overrides):
    state = {
        "vitals": {"hp": 100.0, "hp_max": 120.0, "en": 0.8, "hun": 0.2, "thi": 0.9, "str": 0.5},
        "emo": {"hap": 0.1, "fear": 0.1, "ang": 0.2, "mood": "Calm"},
        "percepts": [],
        "sched": {"act": "work"},
    }
    state.update(overrides)
    return state


def test_summarize_situation_mentions_dominant_need():
    summary = summarize_situation(_state())
    assert "thirst" in summary.lower()


def test_summarize_situation_mentions_threat():
    state = _state(percepts=[{"id": "wolf", "tag": "Threat", "threat": 0.8}])
    summary = summarize_situation(state)
    assert "wolf" in summary.lower()
    assert "threat" in summary.lower()


def test_build_bark_prompt_contains_persona_action_and_situation():
    prompt = build_bark_prompt(_PERSONA, _state(), "drink")
    assert "Aldric" in prompt
    assert "Blacksmith" in prompt
    assert "blunt" in prompt
    assert "drink" in prompt
    assert "thirst" in prompt.lower()


class _StubLLM:
    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.calls = []

    def generate(self, prompt, system=""):
        self.calls.append((prompt, system))
        if self.error:
            raise self.error
        return self.reply


def test_bark_returns_cleaned_single_line():
    llm = _StubLLM(reply='"Throat\'s dry as forge ash."\nSecond line to ignore.')
    verbalizer = BarkVerbalizer(llm)

    line = verbalizer.bark(_PERSONA, _state(), "drink")

    assert line == "Throat's dry as forge ash."


def test_bark_falls_back_on_llm_failure():
    verbalizer = BarkVerbalizer(_StubLLM(error=RuntimeError("ollama down")))
    line = verbalizer.bark(_PERSONA, _state(), "drink")
    assert line == FALLBACK_BARKS["drink"]


def test_bark_falls_back_on_empty_or_overlong_reply():
    assert BarkVerbalizer(_StubLLM(reply="")).bark(_PERSONA, _state(), "flee") == FALLBACK_BARKS["flee"]
    assert (
        BarkVerbalizer(_StubLLM(reply="word " * 100)).bark(_PERSONA, _state(), "flee")
        == FALLBACK_BARKS["flee"]
    )


def test_fallback_barks_cover_all_native_actions():
    from backend.behavior.native_features import NATIVE_ACTIONS

    assert set(FALLBACK_BARKS) == set(NATIVE_ACTIONS)
    assert all(isinstance(line, str) and line for line in FALLBACK_BARKS.values())


def test_unknown_action_uses_generic_fallback_on_failure():
    verbalizer = BarkVerbalizer(_StubLLM(error=RuntimeError("down")))
    line = verbalizer.bark(_PERSONA, _state(), "unknown_action")
    assert isinstance(line, str) and line
