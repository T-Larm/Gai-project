"""Generate deterministic baseline personas for every seed in the NPC roster.

This is intentionally LLM-free. It gives the Unity/demo pipeline enough NPCs to
load immediately, while the richer PersonaGenerator can still overwrite these
files later when Ollama is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from backend.config.settings import PERSONAS_DIR


def build_persona(seed: Dict[str, Any]) -> Dict[str, Any]:
    name = seed["name"]
    occupation = seed["occupation"]
    tags = [str(tag) for tag in seed.get("personality_tags", [])]
    relationships = dict(seed.get("relationships", {}))
    extra = dict(seed.get("extra", {}))
    location = extra.get("location", "the town")
    setting = extra.get("setting", "Suntail Village")
    conflict = extra.get("conflict", "the village's unresolved troubles")
    secret = extra.get("secret", "keeps a private worry hidden from strangers")

    values = _values_from_tags(tags)
    speech_style = _speech_style_from_tags(tags)
    knowledge = _knowledge_from_occupation(occupation)
    faction = _faction_from_occupation(occupation)
    dominant_emotion = _initial_emotion(tags)

    return {
        "seed": {
            "name": name,
            "occupation": occupation,
            "personality_tags": tags,
            "relationships": relationships,
            "extra": extra,
        },
        "core": {
            "name": name,
            "occupation": occupation,
            "backstory": (
                f"{name} is the {occupation.lower()} of {setting}, usually found at {location}. "
                f"Known as {', '.join(tags[:3]) if tags else 'practical'}, {name} has become "
                f"tangled in {conflict}. "
                f"Privately, {name} carries a dangerous secret: {secret}."
            ),
            "values": values,
            "speech_style": speech_style,
            "knowledge_domains": knowledge,
        },
        "social": {
            "relationships": relationships,
            "faction": faction,
            "reputation": _reputation_from_tags(tags),
        },
        "dynamic": {
            "current_goal": _goal_from_occupation(occupation, secret),
            "emotional_state": dominant_emotion,
            "short_term_memory": [],
        },
        "memory_log": [],
    }


def generate_roster(
    seed_path: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> List[Path]:
    seeds = json.loads(seed_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for seed in seeds:
        path = out_dir / f"{seed['name'].lower().replace(' ', '_')}.json"
        if path.exists() and not overwrite:
            continue
        path.write_text(
            json.dumps(build_persona(seed), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written.append(path)
    return written


def _values_from_tags(tags: List[str]) -> List[str]:
    values = []
    mapping = {
        "honest": "honesty",
        "protective": "protection of the town",
        "compassionate": "mercy",
        "devout": "faith",
        "loyal": "loyalty",
        "practical": "practical survival",
        "meticulous": "truth in records",
        "idealistic": "justice",
    }
    for tag in tags:
        value = mapping.get(tag.lower())
        if value and value not in values:
            values.append(value)
    for fallback in ("survival", "reputation", "local stability"):
        if len(values) >= 4:
            break
        if fallback not in values:
            values.append(fallback)
    return values[:4]


def _speech_style_from_tags(tags: List[str]) -> str:
    lowered = {tag.lower() for tag in tags}
    if "gruff" in lowered or "blunt" in lowered:
        return "blunt, terse, and practical"
    if "bookish" in lowered or "meticulous" in lowered:
        return "precise, careful, and quietly witty"
    if "warm" in lowered:
        return "warm, conversational, and observant"
    if "sarcastic" in lowered:
        return "quick, guarded, and sarcastic"
    if "charming" in lowered:
        return "polished, flattering, and evasive"
    return "grounded, direct, and in character"


def _knowledge_from_occupation(occupation: str) -> List[str]:
    base = {
        "blacksmith": ["metalwork", "weapons", "ore quality"],
        "healer": ["medicine", "herbs", "injuries"],
        "corrupt nobleman": ["politics", "taxes", "noble houses"],
        "town guard captain": ["law", "patrols", "local threats"],
        "innkeeper": ["travelers", "rumors", "town gossip"],
        "ranger": ["woods", "tracks", "ruins"],
        "merchant": ["trade", "prices", "shipping routes"],
        "archivist": ["records", "history", "legal claims"],
        "miner": ["tunnels", "ore", "mine hazards"],
        "street informant": ["rumors", "passwords", "alleys"],
        "priest": ["faith", "confession", "temple records"],
        "forge apprentice": ["forge work", "ingots", "market errands"],
        "wandering swordswoman": ["swordsmanship", "patrol routes", "local threats"],
        "herbalist": ["medicinal herbs", "remedies", "forest plants"],
        "ranger": ["tracking", "woodland survival", "monster signs"],
        "traveling minstrel": ["music", "regional folklore", "coded messages"],
        "village steward": ["village affairs", "trade agreements", "local disputes"],
        "cook": ["cooking", "ingredients", "tavern supplies"],
    }
    return base.get(occupation.lower(), ["local life", "town politics", "daily work"])


def _faction_from_occupation(occupation: str) -> str:
    mapping = {
        "corrupt nobleman": "Vane Estate",
        "town guard captain": "Town Guard",
        "healer": "Herbalist Guild",
        "merchant": "Market Consortium",
        "priest": "Temple of the First Bell",
        "archivist": "Civic Archive",
        "wandering swordswoman": "Suntail Watch",
        "herbalist": "Suntail Apothecary",
        "ranger": "Suntail Wardens",
        "village steward": "Suntail Council",
        "cook": "Lantern Kitchen",
    }
    return mapping.get(occupation.lower(), "Oakmere Commoners")


def _reputation_from_tags(tags: List[str]) -> str:
    if any(tag.lower() in {"manipulative", "greedy", "paranoid"} for tag in tags):
        return "useful but difficult to trust"
    if any(tag.lower() in {"compassionate", "warm", "gentle"} for tag in tags):
        return "trusted by ordinary townsfolk"
    if any(tag.lower() in {"disciplined", "meticulous", "hardworking"} for tag in tags):
        return "reliable and exacting"
    return "known by most locals"


def _initial_emotion(tags: List[str]) -> str:
    lowered = {tag.lower() for tag in tags}
    if lowered & {"paranoid", "cautious", "skeptical", "secretive"}:
        return "suspicious"
    if lowered & {"compassionate", "warm", "gentle", "eager"}:
        return "friendly"
    if lowered & {"manipulative", "greedy", "impatient"}:
        return "determined"
    return "neutral"


def _goal_from_occupation(occupation: str, secret: str) -> str:
    return (
        f"Continue daily work as {occupation.lower()} while protecting this secret: {secret}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic NPC roster personas")
    parser.add_argument("--seed", default="data/seeds/example_seeds.json")
    parser.add_argument("--out-dir", default=PERSONAS_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    written = generate_roster(
        seed_path=Path(args.seed),
        out_dir=Path(args.out_dir),
        overwrite=args.overwrite,
    )
    print(f"Wrote {len(written)} persona files to {args.out_dir}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
