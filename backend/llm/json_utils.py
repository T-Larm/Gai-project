"""Robust parsing of JSON returned by an LLM."""
import json
import re
from typing import Optional


def parse_llm_json(raw: str, llm=None) -> dict:
    """Parse `raw` into a dict, tolerating markdown fences and surrounding prose.

    If parsing fails and `llm` is given, ask it once to repair the output.
    Raises ValueError (including the raw text) when nothing works.
    """
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.MULTILINE).strip()
    cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()

    parsed = _try_parse(cleaned)
    if parsed is not None:
        return parsed

    if llm is not None:
        fix_prompt = (
            "The following text should be valid JSON but is malformed. "
            "Return ONLY the corrected JSON, no explanation:\n\n" + cleaned
        )
        parsed = _try_parse(llm.generate(fix_prompt).strip())
        if parsed is not None:
            return parsed

    raise ValueError(f"Could not parse LLM output as JSON:\n{raw}")


def coerce_str(value) -> str:
    """Coerce any LLM output value to a plain string."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items())
    if isinstance(value, list):
        return ", ".join(str(i) for i in value)
    return str(value)


def coerce_str_list(value) -> list:
    """Coerce any LLM output value to a list of strings."""
    if isinstance(value, list):
        return [coerce_str(i) for i in value]
    if isinstance(value, str):
        return [value]
    return [coerce_str(value)]


def _try_parse(text: str) -> Optional[dict]:
    """Parse directly, then fall back to the first {...} block. None on failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None
