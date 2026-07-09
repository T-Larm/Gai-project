"""LLM verbalizer: turn a policy-selected action into an in-character bark.

方案 B division of labor: the trained policy decides WHAT the NPC does
(fast, reliable); the LLM only decides HOW it sounds. A bark is one short
first-person line that makes the behavior legible to the player ("Throat's
dry as forge ash." while walking to the well).

Robustness: any LLM failure, empty reply, or overlong reply falls back to a
deterministic per-action template, so gameplay never blocks on the LLM.
"""

from __future__ import annotations

from typing import Any, List, Mapping

from backend.behavior.native_features import NATIVE_ACTIONS


MAX_BARK_CHARS = 160

FALLBACK_BARKS: Mapping[str, str] = {
    "eat": "I need to eat something.",
    "drink": "I need water, now.",
    "sleep": "I can't keep my eyes open. Time to rest.",
    "flee": "I can't win this — run!",
    "gather": "Better stock up while I can.",
    "heal": "Patch myself up before it gets worse.",
    "attack": "You picked the wrong fight!",
    "socialize": "Hey there — got a minute?",
    "trade": "Let's talk business.",
    "work": "Back to work. It won't do itself.",
    "pray": "May the gods watch over me.",
    "walk_to": "I should get moving.",
}
_GENERIC_FALLBACK = "Hm. Best get on with it."

_SYSTEM_PROMPT = (
    "You write a single short line of NPC dialogue for a medieval RPG. "
    "Stay strictly in character. Output ONLY the spoken line: first person, "
    "at most 15 words, no quotation marks, no stage directions, no explanations."
)


def summarize_situation(state: Mapping[str, Any]) -> str:
    """Deterministic one-line summary of what is driving the NPC right now."""
    vitals = _mapping(state.get("vitals"))
    emo = _mapping(state.get("emo"))
    percepts = [p for p in _list(state.get("percepts")) if isinstance(p, Mapping)]

    facts: List[str] = []

    threats = [p for p in percepts if p.get("tag") == "Threat"]
    if threats:
        top = max(threats, key=lambda p: _float(p.get("threat")))
        facts.append(f"a threat is near: {top.get('id', 'unknown')} (threat {_float(top.get('threat')):.2f})")

    hp = _float(vitals.get("hp"), 100.0)
    hp_max = max(_float(vitals.get("hp_max"), 100.0), 1.0)
    if hp / hp_max < 0.3:
        facts.append("badly wounded")
    if _float(vitals.get("thi")) >= 0.7:
        facts.append("suffering from thirst")
    if _float(vitals.get("hun")) >= 0.7:
        facts.append("very hungry")
    if _float(vitals.get("en"), 1.0) <= 0.25:
        facts.append("exhausted")

    social = next((p for p in percepts if p.get("tag") == "Social"), None)
    if social:
        facts.append(f"{social.get('id', 'someone')} is nearby")

    mood = str(emo.get("mood", "")).strip()
    if mood:
        facts.append(f"mood: {mood.lower()}")

    return "; ".join(facts) if facts else "an ordinary moment, nothing pressing"


def build_bark_prompt(persona: Mapping[str, Any], state: Mapping[str, Any], action: str) -> str:
    name = str(persona.get("name", "The NPC"))
    occupation = str(persona.get("occupation", ""))
    speech_style = str(persona.get("speech_style", ""))
    traits = ", ".join(str(t) for t in _list(persona.get("traits")) or _list(persona.get("personality_tags")))

    lines = [f"Character: {name}, a {occupation}." if occupation else f"Character: {name}."]
    if traits:
        lines.append(f"Personality: {traits}.")
    if speech_style:
        lines.append(f"Speech style: {speech_style}.")
    lines.append(f"Situation: {summarize_situation(state)}.")
    lines.append(f"{name} has just decided to: {action}.")
    lines.append(f"Write the single line {name} says out loud (or mutters) right now.")
    return "\n".join(lines)


class BarkVerbalizer:
    """Generate one in-character line for a policy-selected action."""

    def __init__(self, llm: Any):
        self.llm = llm

    def bark(self, persona: Mapping[str, Any], state: Mapping[str, Any], action: str) -> str:
        try:
            raw = self.llm.generate(build_bark_prompt(persona, state, action), system=_SYSTEM_PROMPT)
        except Exception:
            return self._fallback(action)
        line = _clean_line(raw)
        if not line or len(line) > MAX_BARK_CHARS:
            return self._fallback(action)
        return line

    def _fallback(self, action: str) -> str:
        return FALLBACK_BARKS.get(action, _GENERIC_FALLBACK)


def _clean_line(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip()
    return first_line.strip('"').strip("'").strip("*").strip()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# Ensure the fallback table stays in sync with the action space.
assert set(FALLBACK_BARKS) == set(NATIVE_ACTIONS), "FALLBACK_BARKS must cover every native action"
