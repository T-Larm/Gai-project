"""Convert Stateful RPG NPC chat records into behavior-policy samples (v2).

v2 targets the dataset's native 11-action space instead of projecting actions
onto dialogue acts. Ground-truth labels are recomputed exactly with the
deterministic rule shipped inside the dataset's own generator
(``data/archive/generator/decision_factors.py``), so every state gets a precise
label — no more heuristic guessing for unmatched test rows.

Leakage rules (RQ1 honesty):
- No ``player_intent`` feature (v1 derived it from the action label).
- No oracle intermediates (``self_power``/``perceived_threat``/``duty_pull``/
  ``zone``) in features; they are kept only as label metadata for analysis.
- ``emo.mood`` is excluded from features and kept as the auxiliary emotion
  label (predicted from the numeric ``hap``/``fear``/``ang`` inputs).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from backend.behavior.native_features import (
    NATIVE_ACTIONS,
    extract_native_features,
    inventory_flags as _inventory_flags,
)
from evaluation.datasets.inspect_stateful_rpg import (
    extract_json_objects,
    iter_jsonl,
    parse_chat_messages,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "archive" / "data"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "behavior_policy" / "stateful_rpg_v2"
DECISION_MODULE_PATH = PROJECT_ROOT / "data" / "archive" / "generator" / "decision_factors.py"

VALID_ACTIONS = NATIVE_ACTIONS

SPLIT_RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}

_decision_module: Optional[ModuleType] = None


def load_decision_module() -> ModuleType:
    """Import the dataset's own decision rule from data/archive/generator."""
    global _decision_module
    if _decision_module is None:
        spec = importlib.util.spec_from_file_location("srpg_decision_factors", DECISION_MODULE_PATH)
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"Cannot load decision module from {DECISION_MODULE_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _decision_module = module
    return _decision_module


def oracle_label(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Recompute the deterministic ground-truth action for a simulator state.

    Mirrors the generator's ``_select_action_standard`` (npc_sim_generator_v2.py):
    critical survival needs override combat logic, everything else goes through
    ``pick_action_multifactor``. The generator additionally replaced ~15% of
    labels with stochastic persona deviations (D1-D7); those are irreproducible
    from the state alone, so v2 relabels them with this deterministic rule.
    """
    state = dict(state)
    action_id, factors = load_decision_module().pick_action_multifactor(state)
    zone = factors.get("zone", "")

    override = _survival_override(state)
    if override is not None:
        action_id = override
        zone = "survival_override"

    return {
        "action_id": action_id,
        "zone": zone,
        "factors": {
            "self_power": factors.get("self_power", 0.0),
            "perceived_threat": factors.get("perceived_threat", 0.0),
            "duty_pull": factors.get("duty_pull", 0.0),
        },
    }


def _survival_override(state: Mapping[str, Any]) -> Optional[str]:
    """Generator's pre-combat survival checks, in original priority order."""
    vitals = _mapping(state.get("vitals"))
    inv_ids = set(_inventory_flags(state.get("inv")))
    if _float(vitals.get("hp"), 100.0) < 20 and "medicine" in inv_ids:
        return "heal"
    if _float(vitals.get("hun")) > 0.85:
        return "eat" if "food" in inv_ids else "gather"
    if _float(vitals.get("thi")) > 0.85:
        return "drink" if "water" in inv_ids else "gather"
    return None


def convert_reasoner_file(
    path: Path,
    max_records: Optional[int] = None,
) -> Iterable[Dict[str, Any]]:
    """Yield v2 samples from one reasoner JSONL file."""
    for row_index, record in enumerate(iter_jsonl(path, max_records=max_records)):
        messages = parse_chat_messages(record.get("text", ""))
        user = _message_content(messages, "user")
        reasoning = _message_content(messages, "assistant")
        state_objects = extract_json_objects(user)
        if not state_objects or not reasoning:
            continue
        state = state_objects[-1]
        try:
            label = oracle_label(state)
        except (KeyError, TypeError):
            continue
        mood = _norm(_mapping(state.get("emo")).get("mood"))

        yield {
            "id": f"srpg_{_state_digest(state)}",
            "source": {
                "dataset": "stateful_rpg_npc",
                "file": path.name,
                "row_index": row_index,
            },
            "features": extract_native_features(state),
            "label": label,
            "aux": {"mood": mood},
            "reasoning": reasoning,
            "source_state": state,
        }


def dedupe_records(records: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Drop records whose state hash was already seen. Returns (unique, removed)."""
    unique: List[Dict[str, Any]] = []
    seen: set = set()
    removed = 0
    for record in records:
        if record["id"] in seen:
            removed += 1
            continue
        seen.add(record["id"])
        unique.append(record)
    return unique, removed


def stratified_split(
    records: List[Dict[str, Any]],
    seed: int = 13,
    ratios: Mapping[str, float] = SPLIT_RATIOS,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split per action class so rare actions keep their share in every split.

    Classes with fewer than 3 samples go entirely to train (they cannot be
    meaningfully evaluated and would otherwise vanish from training).
    """
    rng = random.Random(seed)
    by_action: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        by_action.setdefault(record["label"]["action_id"], []).append(record)

    splits: Dict[str, List[Dict[str, Any]]] = {name: [] for name in ratios}
    for action in sorted(by_action):
        group = by_action[action]
        rng.shuffle(group)
        if len(group) < 3:
            splits["train"].extend(group)
            continue
        n_test = max(1, int(round(len(group) * ratios["test"])))
        n_valid = max(1, int(round(len(group) * ratios["valid"])))
        splits["test"].extend(group[:n_test])
        splits["valid"].extend(group[n_test:n_test + n_valid])
        splits["train"].extend(group[n_test + n_valid:])

    for name in splits:
        rng.shuffle(splits[name])
    return splits


def build_formatter_index(formatter_path: Path) -> Dict[str, Dict[str, Any]]:
    """Map normalized reasoning text to the formatter's action JSON."""
    labels: Dict[str, Dict[str, Any]] = {}
    for record in iter_jsonl(formatter_path):
        messages = parse_chat_messages(record.get("text", ""))
        user = _message_content(messages, "user")
        assistant = _message_content(messages, "assistant")
        if not user or not assistant:
            continue
        objects = extract_json_objects(assistant)
        if not objects:
            continue
        labels[_normalize_reasoning(user)] = objects[0]
    return labels


def formatter_agreement(
    records: List[Dict[str, Any]],
    formatter_index: Mapping[str, Dict[str, Any]],
    max_disagreements: int = 20,
) -> Dict[str, Any]:
    """Sanity check: oracle labels vs the dataset's own formatter labels.

    Reasoning texts are template-generated and can collide across different
    states; a collided text cannot be matched back to its own formatter label,
    so those records are skipped instead of counted as disagreements.
    """
    text_counts = Counter(_normalize_reasoning(record["reasoning"]) for record in records)
    matched = 0
    agreed = 0
    skipped_ambiguous = 0
    disagreements: List[Dict[str, Any]] = []
    for record in records:
        key = _normalize_reasoning(record["reasoning"])
        formatter = formatter_index.get(key)
        if not formatter:
            continue
        if text_counts[key] > 1:
            skipped_ambiguous += 1
            continue
        selected = formatter.get("selected_action")
        formatter_action = str(_mapping(selected).get("action_id", "")).strip().lower()
        if not formatter_action:
            continue
        matched += 1
        oracle_action = record["label"]["action_id"]
        if formatter_action == oracle_action:
            agreed += 1
        elif len(disagreements) < max_disagreements:
            disagreements.append({
                "id": record["id"],
                "oracle": oracle_action,
                "formatter": formatter_action,
                "zone": record["label"]["zone"],
            })
    return {
        "matched": matched,
        "agreed": agreed,
        "rate": round(agreed / matched, 4) if matched else None,
        "skipped_ambiguous": skipped_ambiguous,
        "disagreements_sample": disagreements,
    }


def convert_directory(
    raw_dir: Path,
    out_dir: Path,
    seed: int = 13,
    max_records: Optional[int] = None,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    source_files = ["train_reasoner.jsonl", "test_reasoner.jsonl"]
    for name in source_files:
        path = raw_dir / name
        if path.exists():
            records.extend(convert_reasoner_file(path, max_records=max_records))

    parsed = len(records)
    records, removed = dedupe_records(records)

    formatter_path = raw_dir / "train_formatter.jsonl"
    formatter_index = build_formatter_index(formatter_path) if formatter_path.exists() else {}
    agreement = formatter_agreement(records, formatter_index)

    splits = stratified_split(records, seed=seed)
    for name, split_records in splits.items():
        _write_jsonl(split_records, out_dir / f"{name}.jsonl")

    report = {
        "version": 2,
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "seed": seed,
        "source_files": source_files,
        "label_source": (
            "deterministic generator rule: survival overrides + pick_action_multifactor "
            "(mirrors _select_action_standard in data/archive/generator/npc_sim_generator_v2.py)"
        ),
        "parsed_records": parsed,
        "dedup": {"removed": removed},
        "total_records": len(records),
        "action_counts": _counts(records, lambda r: r["label"]["action_id"]),
        "zone_counts": _counts(records, lambda r: r["label"]["zone"]),
        "mood_counts": _counts(records, lambda r: r["aux"]["mood"]),
        "oracle_vs_formatter": agreement,
        "splits": {
            name: {
                "records": len(split_records),
                "action_counts": _counts(split_records, lambda r: r["label"]["action_id"]),
            }
            for name, split_records in splits.items()
        },
        "notes": [
            "label.action_id is recomputed with the generator's deterministic rule "
            "(survival overrides + multifactor); every state gets an exact label.",
            "the original dataset replaced ~15% of labels with stochastic persona deviations "
            "(D1-D7 in npc_sim_generator_v2.py); these are irreproducible from the state alone "
            "and are relabeled deterministically here — expect oracle_vs_formatter.rate ~0.85-0.9.",
            "label.zone/factors are oracle metadata for analysis only — never model inputs.",
            "features exclude player_intent, emo.mood, and all oracle intermediates (leakage).",
            "aux.mood is the auxiliary emotion label, predicted from numeric emo features.",
            "trade never occurs in the data; the effective action space has 11 classes.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def heuristic_action_id(state: Mapping[str, Any]) -> str:
    """Independent survival heuristic — the fair rule baseline for RQ1.

    Written from gameplay common sense, NOT from the generator's rule (which
    is the labeling oracle and would make the comparison circular).
    """
    vitals = _mapping(state.get("vitals"))
    inv = set(_inventory_flags(state.get("inv")))
    percepts = [p for p in _list(state.get("percepts")) if isinstance(p, Mapping)]
    max_threat = max([_float(p.get("threat")) for p in percepts if p.get("tag") == "Threat"] or [0.0])
    strength = _float(vitals.get("str"))
    hp = _float(vitals.get("hp"), 1.0)
    hp_max = max(_float(vitals.get("hp_max"), 1.0), 1.0)

    if max_threat > max(strength, 0.45):
        return "flee"
    if max_threat > 0.0 and strength > max_threat:
        return "attack"
    if _float(vitals.get("thi")) >= 0.7 and "water" in inv:
        return "drink"
    if _float(vitals.get("hun")) >= 0.7 and "food" in inv:
        return "eat"
    if hp / hp_max < 0.45 and "medicine" in inv:
        return "heal"
    if _float(vitals.get("en"), 1.0) <= 0.25:
        return "sleep"
    if any(p.get("tag") == "Social" for p in percepts):
        return "socialize"
    scheduled = _norm(_mapping(state.get("sched")).get("act"))
    if scheduled == "work":
        return "work"
    return "walk_to"


def _counts(records: List[Dict[str, Any]], key) -> Dict[str, int]:
    return dict(Counter(key(record) for record in records).most_common())


def _state_digest(state: Mapping[str, Any]) -> str:
    canonical = json.dumps(state, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def _message_content(messages: List[Dict[str, str]], role: str) -> str:
    for message in messages:
        if message["role"] == role:
            return message["content"]
    return ""


def _normalize_reasoning(text: str) -> str:
    return " ".join((text or "").split())


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Stateful RPG NPC data (v2, native actions)")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-records", type=int, default=0,
                        help="Records per reasoner file; 0 converts all")
    args = parser.parse_args()

    max_records = None if args.max_records == 0 else args.max_records
    report = convert_directory(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        seed=args.seed,
        max_records=max_records,
    )
    summary = {key: report[key] for key in ("total_records", "action_counts", "oracle_vs_formatter")}
    summary["splits"] = {name: info["records"] for name, info in report["splits"].items()}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
