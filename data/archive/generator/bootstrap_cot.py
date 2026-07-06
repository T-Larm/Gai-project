# Copyright 2025-2026 Sadık Abdusselam Albayrak
# Licensed under the Apache License, Version 2.0
"""
Gemma 4 E4B bootstrap CoT generator (v1.6.0).

Calls a local Ollama instance running gemma4:e4b to produce 3-5 sentence
Turkish chain-of-thought reasoning for each NPC decision. Results are cached
on disk (SHA-keyed) so incremental re-runs skip already-generated entries.

Falls back to None on any failure; the generator's template CoT
(generate_cot_reasoning) is the caller's responsibility as fallback.

Public API:
  generate_via_gemma(state, action_id, factors, persona) -> str | None
"""

from __future__ import annotations
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

_CACHE_DIR = Path(__file__).parent / ".cot_cache"
_OLLAMA_URL = "http://localhost:11434/api/generate"
_MODEL = "gemma4:e4b"
_TIMEOUT_SEC = 30

# Structured prompt: feeds the decision factors explicitly so Gemma reasons
# from numbers rather than free-associating on the NPC name.
_PROMPT_TEMPLATE = """\
Sen bir ortaçağ simülasyonundaki NPC'nin iç sesisin. \
Aşağıdaki bilgilere göre NPC'nin kararını açıklayan 3-5 cümlelik Türkçe iç monolog yaz.

Persona:
{persona}

Karar analizi:
  - Öz güç (self_power): {self_power:.3f}
  - Algılanan tehdit (perceived_threat): {perceived_threat:.3f}
  - Görev çekimi (duty_pull): {duty_pull:.3f}
  - Karar bölgesi: {zone}
  - Seçilen eylem: {action_id}

Kurallar:
- ASLA JSON yazma, ASLA "action_id" kelimesini kullanma
- Birinci şahıs iç monolog (Türkçe), nokta ile bitir
- Kararın mantığını sayısal faktörlere dayandır (güç, tehdit, görev)
- 3-5 cümle, kısa ve özlü
"""


def _cache_key(state: dict, action_id: str) -> str:
    """SHA256 of state content (excluding random UUID) + action — 16 hex chars."""
    state_for_hash = {k: v for k, v in state.items() if k != "id"}
    payload = json.dumps({"s": state_for_hash, "a": action_id}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def generate_via_gemma(
    state: dict,
    action_id: str,
    factors: dict,
    persona: str,
) -> str | None:
    """
    Generate a 3-5 sentence Turkish CoT via local Gemma 3 4B (Ollama).

    Returns None on any failure (Ollama not running, model not loaded,
    empty/too-short response, network error). The caller must provide
    a fallback.

    Caches results in .cot_cache/<sha>.txt so repeated generator runs
    with the same seed skip already-bootstrapped examples.
    """
    _CACHE_DIR.mkdir(exist_ok=True)

    key = _cache_key(state, action_id)
    cache_file = _CACHE_DIR / f"{key}.txt"
    if cache_file.exists():
        cached = cache_file.read_text(encoding="utf-8").strip()
        if cached:
            return cached

    prompt = _PROMPT_TEMPLATE.format(
        persona=persona,
        self_power=factors.get("self_power", 0.5),
        perceived_threat=factors.get("perceived_threat", 0.0),
        duty_pull=factors.get("duty_pull", 0.0),
        zone=factors.get("zone", "unknown"),
        action_id=action_id,
    )

    body = json.dumps(
        {
            "model": _MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.75, "num_predict": 200, "stop": ["\n\n"]},
        },
        ensure_ascii=False,
    ).encode()

    req = urllib.request.Request(
        _OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read())
            cot = data.get("response", "").strip()
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
        print(f"  [CoT] Gemma unavailable ({type(e).__name__}), using template", file=sys.stderr)
        return None

    if not cot or len(cot) < 30:
        return None

    # Reject if the model leaked JSON despite instructions
    if cot.lstrip().startswith("{"):
        return None

    cache_file.write_text(cot, encoding="utf-8")
    return cot
