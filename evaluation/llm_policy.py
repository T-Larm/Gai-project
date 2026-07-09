"""LLM-as-policy baseline: an instruction-tuned LLM picks the NPC action.

Third RQ1 condition next to the hand-written heuristic and the trained MLP.
The LLM receives the raw simulator state as JSON and must answer with one of
the native actions. Invalid or unparseable answers fall back to ``walk_to``
and are counted in ``predict.stats`` so the invalid rate can be reported.

Example:
    python -m evaluation.eval_policies --llm-model llama3:latest --max-records 200
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Mapping, Optional

from backend.behavior.native_features import NATIVE_ACTIONS


SYSTEM_PROMPT = (
    "You are the decision engine of an NPC in a medieval life simulation. "
    "You receive the NPC's current state as JSON (vitals are 0-1 unless a "
    "max is given; percepts list what the NPC currently notices). "
    "Choose exactly ONE action the NPC should take right now.\n"
    f"Valid actions: {', '.join(NATIVE_ACTIONS)}\n"
    'Answer with ONLY this JSON, nothing else: {"action_id": "<action>"}'
)

_EXAMPLES = (
    (
        '{"occ":"Farmer","vitals":{"hp":90,"hp_max":120,"en":0.8,"hun":0.2,"thi":0.9},'
        '"inv":[{"id":"water","n":2}],"percepts":[]}',
        '{"action_id": "drink"}',
    ),
    (
        '{"occ":"Scholar","vitals":{"hp":30,"hp_max":120,"en":0.5,"hun":0.3,"thi":0.2},'
        '"inv":[],"percepts":[{"id":"bandit_01","tag":"Threat","threat":0.9}]}',
        '{"action_id": "flee"}',
    ),
)


def build_prompt(state: Mapping[str, Any]) -> str:
    examples = "\n\n".join(
        f"State: {state_json}\nAnswer: {answer}" for state_json, answer in _EXAMPLES
    )
    return (
        f"Valid actions: {', '.join(NATIVE_ACTIONS)}\n\n"
        f"{examples}\n\n"
        f"State: {json.dumps(state, ensure_ascii=False)}\n"
        "Answer:"
    )


def parse_action(raw: str) -> Optional[str]:
    """Extract a valid action_id from an LLM reply, or None."""
    text = (raw or "").strip()
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            action = str(parsed.get("action_id", "")).strip().lower()
            return action if action in NATIVE_ACTIONS else None
        except json.JSONDecodeError:
            pass
    lowered = text.lower()
    for action in NATIVE_ACTIONS:
        if re.search(rf"\b{re.escape(action)}\b", lowered):
            return action
    return None


def llm_predict_fn(
    model: str,
    client: Any = None,
    fallback: str = "walk_to",
) -> Callable[[Mapping[str, Any]], str]:
    if client is None:
        from backend.llm.ollama_client import OllamaClient

        client = OllamaClient(model)

    stats: Dict[str, int] = {"total": 0, "invalid": 0}

    def predict(record: Mapping[str, Any]) -> str:
        stats["total"] += 1
        raw = client.generate(build_prompt(record["source_state"]), system=SYSTEM_PROMPT)
        action = parse_action(raw)
        if action is None:
            stats["invalid"] += 1
            return fallback
        return action

    predict.stats = stats
    return predict
