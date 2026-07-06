"""Convert Stateful RPG NPC chat records into canonical policy samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from backend.behavior.schemas import (
    DialogueAct,
    DisclosureLevel,
    Gesture,
    MemoryWriteType,
    NpcEmotion,
    PlayerIntent,
    PolicyAction,
    QuestStage,
    QuestUpdate,
    RelationshipStatus,
    StateFeatures,
)
from evaluation.datasets.inspect_stateful_rpg import (
    extract_json_objects,
    iter_jsonl,
    parse_chat_messages,
)


ACTION_TO_DIALOGUE = {
    "attack": DialogueAct.WARN,
    "drink": DialogueAct.END_CONVERSATION,
    "eat": DialogueAct.END_CONVERSATION,
    "flee": DialogueAct.WARN,
    "gather": DialogueAct.GIVE_HINT,
    "heal": DialogueAct.END_CONVERSATION,
    "pray": DialogueAct.END_CONVERSATION,
    "sleep": DialogueAct.END_CONVERSATION,
    "socialize": DialogueAct.GREET,
    "trade": DialogueAct.ASK_CLARIFICATION,
    "walk_to": DialogueAct.GIVE_HINT,
    "work": DialogueAct.END_CONVERSATION,
}

ACTION_TO_GESTURE = {
    "attack": Gesture.APPROACH,
    "drink": Gesture.IDLE,
    "eat": Gesture.IDLE,
    "flee": Gesture.STEP_BACK,
    "gather": Gesture.POINT,
    "heal": Gesture.IDLE,
    "pray": Gesture.NOD,
    "sleep": Gesture.IDLE,
    "socialize": Gesture.NOD,
    "trade": Gesture.APPROACH,
    "walk_to": Gesture.POINT,
    "work": Gesture.IDLE,
}

GOAL_TO_ACTION = {
    "findfood": "eat",
    "findwater": "drink",
    "rest": "sleep",
    "heal": "heal",
    "work": "work",
    "socialize": "socialize",
    "trade": "trade",
    "pray": "pray",
}

MOOD_TO_EMOTION = {
    "angry": NpcEmotion.ANGRY,
    "wrathful": NpcEmotion.ANGRY,
    "fearful": NpcEmotion.AFRAID,
    "anxious": NpcEmotion.AFRAID,
    "happy": NpcEmotion.FRIENDLY,
    "calm": NpcEmotion.NEUTRAL,
    "neutral": NpcEmotion.NEUTRAL,
    "determined": NpcEmotion.DETERMINED,
}


def build_formatter_index(formatter_path: Path) -> Dict[str, Dict[str, Any]]:
    """Map reasoning text to formatter action JSON."""
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


def convert_directory(
    raw_dir: Path,
    out_dir: Path,
    valid_ratio: float = 0.1,
    seed: int = 13,
    max_records: Optional[int] = None,
) -> Dict[str, Any]:
    formatter_index = build_formatter_index(raw_dir / "train_formatter.jsonl")
    train_records = list(
        convert_reasoner_file(
            raw_dir / "train_reasoner.jsonl",
            formatter_index=formatter_index,
            source_split="train",
            max_records=max_records,
        )
    )
    test_records = list(
        convert_reasoner_file(
            raw_dir / "test_reasoner.jsonl",
            formatter_index=formatter_index,
            source_split="test",
            max_records=max_records,
        )
    )

    rng = random.Random(seed)
    rng.shuffle(train_records)
    valid_size = int(round(len(train_records) * valid_ratio))
    valid_records = train_records[:valid_size]
    train_records = train_records[valid_size:]

    for split, records in {
        "train": train_records,
        "valid": valid_records,
        "test": test_records,
    }.items():
        _write_jsonl(records, out_dir / f"{split}.jsonl")

    report = build_conversion_report(
        formatter_labels=len(formatter_index),
        train=train_records,
        valid=valid_records,
        test=test_records,
        valid_ratio=valid_ratio,
        seed=seed,
        raw_dir=raw_dir,
        out_dir=out_dir,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def convert_reasoner_file(
    path: Path,
    formatter_index: Mapping[str, Dict[str, Any]],
    source_split: str,
    max_records: Optional[int] = None,
) -> Iterable[Dict[str, Any]]:
    for row_index, record in enumerate(iter_jsonl(path, max_records=max_records)):
        messages = parse_chat_messages(record.get("text", ""))
        user = _message_content(messages, "user")
        reasoning = _message_content(messages, "assistant")
        state_objects = extract_json_objects(user)
        if not state_objects or not reasoning:
            continue
        source_state = state_objects[-1]
        label, label_source = _label_for_reasoning(reasoning, source_state, formatter_index)
        action_id = _action_id(label)
        emotion = _emotion_from_label_or_state(label, source_state)

        state_features = map_state_features(source_state, action_id, emotion)
        policy_action = map_policy_action(action_id, emotion)
        sample_id = _sample_id(path.name, row_index, source_state, reasoning)

        yield {
            "id": sample_id,
            "source": {
                "dataset": "stateful_rpg_npc",
                "file": path.name,
                "row_index": row_index,
                "split": source_split,
                "label_source": label_source,
            },
            "state": state_features.to_dict(),
            "action": policy_action.to_dict(),
            "source_action": {
                "action_id": action_id,
                "target_id": _selected_action(label).get("target_id"),
                "dialogue": _selected_action(label).get("dialogue"),
                "emotion": str(label.get("emotion", "")),
            },
            "reasoning": reasoning,
            "source_state": source_state,
        }


def map_state_features(
    state: Mapping[str, Any],
    action_id: str,
    emotion: NpcEmotion,
) -> StateFeatures:
    vitals = _mapping(state.get("vitals"))
    pos = _mapping(state.get("pos"))
    factions = _mapping(state.get("factions"))
    percepts = state.get("percepts") if isinstance(state.get("percepts"), list) else []
    memories = state.get("memories") if isinstance(state.get("memories"), list) else []

    max_threat = max([_float(item.get("threat"), 0.0) for item in percepts if isinstance(item, Mapping)] or [0.0])
    relationship = _relationship_from_factions(factions)
    intent = _intent_from_action(action_id)
    inventory_flags = _inventory_flags(state.get("inv"))

    return StateFeatures(
        player_intent=intent,
        quest_stage=QuestStage.NONE,
        trust_score=_trust_from_factions(factions),
        npc_emotion=emotion,
        relationship=relationship,
        memory_relevance=_memory_relevance(memories),
        danger_level=max(max_threat, _float(vitals.get("str"), 0.0) if action_id == "attack" else 0.0),
        distance_to_player=0.0,
        forbidden_secret_asked=False,
        prompt_injection_detected=False,
        npc_role=str(state.get("occ", "")).strip().lower().replace(" ", "_"),
        persona_id=str(state.get("id", "")),
        location=str(pos.get("zone", "")),
        inventory_flags=inventory_flags,
    )


def map_policy_action(action_id: str, emotion: NpcEmotion) -> PolicyAction:
    return PolicyAction(
        dialogue_act=ACTION_TO_DIALOGUE.get(action_id, DialogueAct.ASK_CLARIFICATION),
        emotion=emotion,
        disclosure_level=DisclosureLevel.NONE,
        gesture=ACTION_TO_GESTURE.get(action_id, Gesture.IDLE),
        quest_update=QuestUpdate.NO_CHANGE,
        memory_write_type=MemoryWriteType.NONE,
    )


def build_conversion_report(
    formatter_labels: int,
    train: List[Dict[str, Any]],
    valid: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    valid_ratio: float,
    seed: int,
    raw_dir: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    all_records = train + valid + test
    return {
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "formatter_labels": formatter_labels,
        "valid_ratio": valid_ratio,
        "seed": seed,
        "splits": {
            "train": _split_stats(train),
            "valid": _split_stats(valid),
            "test": _split_stats(test),
        },
        "total_records": len(all_records),
        "notes": [
            "source_action.action_id is the primary supervised label from the dataset.",
            "action is a lossy dialogue-policy projection kept for compatibility with PolicyAction.",
            "label_source=formatter_match means reasoning text matched train_formatter exactly.",
            "label_source=heuristic means action was inferred from vitals, threat, schedule, or goal.",
        ],
    }


def _split_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    action_counts = Counter(record["source_action"]["action_id"] for record in records)
    label_sources = Counter(record["source"]["label_source"] for record in records)
    emotions = Counter(record["action"]["emotion"] for record in records)
    return {
        "records": len(records),
        "action_counts": dict(action_counts.most_common()),
        "label_sources": dict(label_sources.most_common()),
        "emotion_counts": dict(emotions.most_common()),
    }


def _label_for_reasoning(
    reasoning: str,
    source_state: Mapping[str, Any],
    formatter_index: Mapping[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], str]:
    matched = formatter_index.get(_normalize_reasoning(reasoning))
    if matched:
        return matched, "formatter_match"
    return heuristic_label(source_state), "heuristic"


def heuristic_label(state: Mapping[str, Any]) -> Dict[str, Any]:
    action_id = heuristic_action_id(state)
    mood = str(_mapping(state.get("emo")).get("mood", "Neutral"))
    return {
        "reasoning": "",
        "selected_action": {"action_id": action_id, "target_id": None, "dialogue": None},
        "emotion": mood,
    }


def heuristic_action_id(state: Mapping[str, Any]) -> str:
    vitals = _mapping(state.get("vitals"))
    inv = set(_inventory_flags(state.get("inv")))
    percepts = state.get("percepts") if isinstance(state.get("percepts"), list) else []
    max_threat = max([_float(item.get("threat"), 0.0) for item in percepts if isinstance(item, Mapping)] or [0.0])
    strength = _float(vitals.get("str"), 0.0)
    hp = _float(vitals.get("hp"), 1.0)
    hp_max = max(_float(vitals.get("hp_max"), 1.0), 1.0)

    if max_threat > max(strength, 0.45):
        return "flee"
    if _float(vitals.get("thi"), 0.0) >= 0.7 and "water" in inv:
        return "drink"
    if _float(vitals.get("hun"), 0.0) >= 0.7 and "food" in inv:
        return "eat"
    if hp / hp_max < 0.45 and "medicine" in inv:
        return "heal"
    if _float(vitals.get("en"), 1.0) <= 0.25:
        return "sleep"
    goal = str(state.get("goals_top", "") or "").lower()
    if goal in GOAL_TO_ACTION:
        return GOAL_TO_ACTION[goal]
    sched = _mapping(state.get("sched"))
    scheduled = str(sched.get("act", "")).lower()
    if scheduled in ACTION_TO_DIALOGUE:
        return scheduled
    return "work"


def _emotion_from_label_or_state(label: Mapping[str, Any], state: Mapping[str, Any]) -> NpcEmotion:
    raw = str(label.get("emotion") or _mapping(state.get("emo")).get("mood") or "neutral")
    normalized = raw.strip().lower()
    return MOOD_TO_EMOTION.get(normalized, NpcEmotion.NEUTRAL)


def _intent_from_action(action_id: str) -> PlayerIntent:
    if action_id in {"socialize", "trade"}:
        return PlayerIntent.SMALLTALK
    if action_id in {"flee", "attack"}:
        return PlayerIntent.THREATEN
    if action_id in {"gather", "walk_to"}:
        return PlayerIntent.ASK_HINT
    if action_id in {"heal"}:
        return PlayerIntent.REQUEST_HELP
    return PlayerIntent.UNKNOWN


def _relationship_from_factions(factions: Mapping[str, Any]) -> RelationshipStatus:
    if not factions:
        return RelationshipStatus.STRANGER
    best = max(_float(value, 0.0) for value in factions.values())
    worst = min(_float(value, 0.0) for value in factions.values())
    if best >= 0.5:
        return RelationshipStatus.ALLY
    if worst <= -0.5:
        return RelationshipStatus.ENEMY
    return RelationshipStatus.KNOWN


def _trust_from_factions(factions: Mapping[str, Any]) -> float:
    if not factions:
        return 0.0
    values = [_float(value, 0.0) for value in factions.values()]
    average = sum(values) / len(values)
    return round(max(0.0, min(1.0, (average + 1.0) / 2.0)), 4)


def _memory_relevance(memories: List[Any]) -> float:
    weights = [
        abs(_float(item.get("ew"), 0.0))
        for item in memories
        if isinstance(item, Mapping)
    ]
    if not weights:
        return 0.0
    return round(max(0.0, min(1.0, max(weights))), 4)


def _inventory_flags(inv: Any) -> List[str]:
    if not isinstance(inv, list):
        return []
    flags = []
    for item in inv:
        if isinstance(item, Mapping) and _float(item.get("n"), 0.0) > 0:
            flags.append(str(item.get("id", "")).strip().lower().replace(" ", "_"))
    return sorted(flag for flag in flags if flag)


def _selected_action(label: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = label.get("selected_action")
    return selected if isinstance(selected, Mapping) else {}


def _action_id(label: Mapping[str, Any]) -> str:
    action_id = str(_selected_action(label).get("action_id", "")).strip().lower()
    return action_id if action_id in ACTION_TO_DIALOGUE else "work"


def _message_content(messages: List[Dict[str, str]], role: str) -> str:
    for message in messages:
        if message["role"] == role:
            return message["content"]
    return ""


def _normalize_reasoning(text: str) -> str:
    return " ".join((text or "").split())


def _sample_id(file_name: str, row_index: int, state: Mapping[str, Any], reasoning: str) -> str:
    digest = hashlib.sha1(
        f"{file_name}|{row_index}|{state.get('id', '')}|{reasoning}".encode("utf-8")
    ).hexdigest()[:12]
    return f"srpg_{digest}"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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
    parser = argparse.ArgumentParser(description="Convert Stateful RPG NPC data")
    parser.add_argument("--raw-dir", default="../archive/data")
    parser.add_argument("--out-dir", default="data/behavior_policy/stateful_rpg")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-records", type=int, default=0,
                        help="Records per reasoner file; 0 converts all")
    args = parser.parse_args()

    max_records = None if args.max_records == 0 else args.max_records
    report = convert_directory(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        max_records=max_records,
    )
    print(json.dumps(report["splits"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
