"""Bark persona-consistency evaluation (LLM-as-judge, with baselines).

Answers the teacher's "beyond consistency metrics — compare with a baseline"
requirement for the bark channel: are generated barks in character and
consistent with the chosen action?

Conditions:
- ours:       BarkVerbalizer with the full persona (the system)
- no_persona: same generator, persona stripped to "An NPC" (ablation)
- template:   the deterministic per-action fallback lines (lower bound)

Each bark is judged twice by the judge model with binary questions
(persona fit, action fit). Judge and generator are both llama3 — the
self-preference bias caveat from the dialogue evaluation applies here too.

Example:
    python -m evaluation.eval_barks --out data/behavior_policy/eval/bark_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from backend.behavior.native_features import NATIVE_ACTIONS
from backend.behavior.verbalizer import FALLBACK_BARKS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_BASE_VITALS = {"hp": 100.0, "hp_max": 120.0, "en": 0.8, "hun": 0.2, "thi": 0.2, "str": 0.5}


def _state(situation_vitals: Optional[Dict[str, float]] = None, **extra) -> Dict[str, Any]:
    vitals = dict(_BASE_VITALS)
    vitals.update(situation_vitals or {})
    state = {"vitals": vitals, "emo": {"hap": 0.2, "fear": 0.1, "ang": 0.1, "mood": "Calm"},
             "percepts": [], "sched": {"act": "idle"}}
    state.update(extra)
    return state


# One representative situation per native action.
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "eat": {"state": _state({"hun": 0.9}), "situation": "very hungry"},
    "drink": {"state": _state({"thi": 0.92}), "situation": "suffering from thirst"},
    "sleep": {"state": _state({"en": 0.1}), "situation": "exhausted"},
    "flee": {
        "state": _state({"hp": 35.0, "str": 0.2},
                        percepts=[{"id": "wolf", "tag": "Threat", "threat": 0.85}],
                        emo={"hap": 0.0, "fear": 0.8, "ang": 0.1, "mood": "Fearful"}),
        "situation": "a dangerous wolf is near and they are too weak to fight",
    },
    "gather": {"state": _state({"hun": 0.75}, inv=[]), "situation": "hungry with no food, must find supplies"},
    "heal": {"state": _state({"hp": 25.0}), "situation": "badly wounded"},
    "attack": {
        "state": _state({"str": 0.9},
                        percepts=[{"id": "bandit_01", "tag": "Threat", "threat": 0.4}],
                        emo={"hap": 0.0, "fear": 0.1, "ang": 0.7, "mood": "Angry"}),
        "situation": "a weak bandit threatens them and they are strong enough to win",
    },
    "socialize": {
        "state": _state({}, percepts=[{"id": "mira", "tag": "Social", "sal": 0.8}],
                        emo={"hap": 0.7, "fear": 0.0, "ang": 0.0, "mood": "Happy"}),
        "situation": "in a good mood with a friend nearby",
    },
    "trade": {
        "state": _state({}, percepts=[{"id": "merchant_npc", "tag": "Social", "sal": 0.6}]),
        "situation": "a merchant is nearby and they want to strike a deal",
    },
    "work": {"state": _state({}, sched={"act": "work"}), "situation": "it is their working hours"},
    "pray": {"state": _state({}, emo={"hap": 0.05, "fear": 0.3, "ang": 0.0, "mood": "Anxious"}),
             "situation": "feeling anxious and seeking comfort in faith"},
    "walk_to": {"state": _state({}), "situation": "an ordinary moment, moving on to the next place"},
}

_JUDGE_SYSTEM = (
    "You are a strict evaluator of RPG NPC dialogue. Answer with a single word: YES or NO."
)


def build_judge_prompt(
    kind: str,
    persona: Mapping[str, Any],
    line: str,
    action: str,
    situation: str,
) -> str:
    traits = ", ".join(str(t) for t in persona.get("traits", []))
    card = (
        f"Character: {persona.get('name', 'An NPC')}, a {persona.get('occupation', 'villager')}. "
        f"Personality: {traits or 'unspecified'}. Speech style: {persona.get('speech_style', 'unspecified')}."
    )
    if kind == "persona":
        question = (
            "Does this line sound like something this specific character would say, "
            "in their voice and personality? Answer YES or NO."
        )
    elif kind == "action":
        question = (
            f"The character just decided to '{action}' because they are {situation}. "
            "Is the spoken line consistent with that decision and situation? Answer YES or NO."
        )
    else:
        raise ValueError(f"Unknown judge kind '{kind}'")
    return f"{card}\n\nSpoken line: \"{line}\"\n\n{question}"


def parse_verdict(raw: str) -> Optional[bool]:
    text = (raw or "").strip().lower()
    if text.startswith("yes"):
        return True
    if text.startswith("no"):
        return False
    return None


def run_bark_eval(
    personas: Mapping[str, Mapping[str, Any]],
    scenarios: Mapping[str, Mapping[str, Any]],
    conditions: Iterable[str],
    generate_line: Callable[[str, Mapping[str, Any], Mapping[str, Any], str], str],
    judge_llm: Any,
) -> Dict[str, Any]:
    transcripts = []
    results: Dict[str, Any] = {}

    for condition in conditions:
        persona_verdicts = []
        action_verdicts = []
        for persona_key, persona in personas.items():
            for action, scenario in scenarios.items():
                line = generate_line(condition, persona, scenario["state"], action)
                verdicts: Dict[str, Optional[bool]] = {}
                for kind, bucket in (("persona", persona_verdicts), ("action", action_verdicts)):
                    prompt = build_judge_prompt(kind, persona, line, action, scenario["situation"])
                    verdict = parse_verdict(judge_llm.generate(prompt, system=_JUDGE_SYSTEM))
                    verdicts[kind] = verdict
                    if verdict is not None:
                        bucket.append(verdict)
                transcripts.append({
                    "condition": condition, "persona": persona_key, "action": action,
                    "line": line,
                    "persona_fit": verdicts["persona"], "action_fit": verdicts["action"],
                })
        results[condition] = {
            "persona_fit_rate": _rate(persona_verdicts),
            "action_fit_rate": _rate(action_verdicts),
            "n_judged_persona": len(persona_verdicts),
            "n_judged_action": len(action_verdicts),
        }

    results["transcripts"] = transcripts
    return results


def _rate(verdicts) -> float:
    return round(sum(verdicts) / len(verdicts), 4) if verdicts else 0.0


def _load_personas(names: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    personas = {}
    for name in names:
        data = json.loads(
            (PROJECT_ROOT / "data" / "personas" / f"{name}.json").read_text(encoding="utf-8")
        )
        personas[name] = {
            "name": data["core"]["name"],
            "occupation": data["core"]["occupation"],
            "speech_style": data["core"]["speech_style"],
            "traits": data["seed"]["personality_tags"],
        }
    return personas


def _real_generate_line(llm):
    from backend.behavior.verbalizer import BarkVerbalizer

    verbalizer = BarkVerbalizer(llm)

    def generate(condition, persona, state, action):
        if condition == "template":
            return FALLBACK_BARKS[action]
        if condition == "no_persona":
            return verbalizer.bark({"name": "An NPC"}, state, action)
        return verbalizer.bark(persona, state, action)

    return generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate bark persona consistency (LLM-as-judge)")
    parser.add_argument("--personas", nargs="*", default=["nicole", "asuna", "lanyan"])
    parser.add_argument("--judge-model", default=None, help="Defaults to settings.OLLAMA_MODEL")
    parser.add_argument("--out", default="data/behavior_policy/eval/bark_eval.json")
    args = parser.parse_args()

    from backend.llm.ollama_client import OllamaClient

    llm = OllamaClient(args.judge_model) if args.judge_model else OllamaClient()
    results = run_bark_eval(
        personas=_load_personas(args.personas),
        scenarios=SCENARIOS,
        conditions=("ours", "no_persona", "template"),
        generate_line=_real_generate_line(llm),
        judge_llm=llm,
    )
    results["personas"] = list(args.personas)
    results["n_scenarios"] = len(SCENARIOS)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {c: results[c] for c in ("ours", "no_persona", "template")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
