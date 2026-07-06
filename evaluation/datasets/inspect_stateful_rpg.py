"""Inspect the Stateful RPG NPC JSONL files before conversion.

The Kaggle archive stores Llama-style chat transcripts in a single top-level
``text`` field. This script reports both that raw JSONL schema and the embedded
state/action JSON objects inside the chat messages.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


CHAT_RE = re.compile(
    r"<\|start_header_id\|>(?P<role>[^<]+)<\|end_header_id\|>\n\n"
    r"(?P<content>.*?)(?=<\|eot_id\|>)",
    re.S,
)


def iter_jsonl(path: Path, max_records: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_records is not None and index >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_chat_messages(text: str) -> List[Dict[str, str]]:
    return [
        {"role": match.group("role").strip(), "content": match.group("content").strip()}
        for match in CHAT_RE.finditer(text or "")
    ]


def extract_json_objects(text: str) -> List[Any]:
    """Extract balanced JSON objects from mixed natural-language text."""
    objects = []
    index = 0
    text = text or ""
    while index < len(text):
        start = text.find("{", index)
        if start == -1:
            break
        depth = 0
        in_string = False
        escape = False
        found_end = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:idx + 1]
                    try:
                        objects.append(json.loads(candidate))
                    except json.JSONDecodeError:
                        pass
                    index = idx + 1
                    found_end = True
                    break
        if not found_end:
            index = start + 1
    return objects


def flatten_schema(value: Any, prefix: str = "") -> Dict[str, str]:
    if isinstance(value, Mapping):
        fields: Dict[str, str] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            fields.update(flatten_schema(item, path))
        return fields
    if isinstance(value, list):
        if not value:
            return {prefix: "list[empty]"}
        child_types = sorted({_type_name(item) for item in value})
        fields = {prefix: f"list[{ '|'.join(child_types) }]"}
        if isinstance(value[0], Mapping):
            fields.update(flatten_schema(value[0], f"{prefix}[]"))
        return fields
    return {prefix: _type_name(value)}


def inspect_file(path: Path, max_records: Optional[int] = None) -> Dict[str, Any]:
    top_level_presence: Counter[str] = Counter()
    top_level_types: Dict[str, Counter[str]] = defaultdict(Counter)
    role_counts: Counter[str] = Counter()
    message_count_histogram: Counter[int] = Counter()
    embedded_presence: Dict[str, Counter[str]] = defaultdict(Counter)
    embedded_types: Dict[str, Dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    action_counts: Counter[str] = Counter()
    emotion_counts: Counter[str] = Counter()
    valid_actions: set[str] = set()
    sample_record: Optional[Dict[str, Any]] = None
    sample_messages: List[Dict[str, str]] = []
    sample_embedded: Dict[str, Any] = {}
    records_seen = 0
    malformed_records = 0

    for record in iter_jsonl(path, max_records=max_records):
        records_seen += 1
        if sample_record is None:
            sample_record = record
        for key, value in record.items():
            top_level_presence[key] += 1
            top_level_types[key][_type_name(value)] += 1

        messages = parse_chat_messages(str(record.get("text", "")))
        if not messages:
            malformed_records += 1
        if not sample_messages and messages:
            sample_messages = [
                {"role": msg["role"], "content_preview": _preview(msg["content"])}
                for msg in messages
            ]
        message_count_histogram[len(messages)] += 1

        for message in messages:
            role = message["role"]
            role_counts[role] += 1
            for embedded in extract_json_objects(message["content"]):
                bucket = f"{role}_json"
                if bucket not in sample_embedded:
                    sample_embedded[bucket] = embedded
                for field, type_name in flatten_schema(embedded).items():
                    embedded_presence[bucket][field] += 1
                    embedded_types[bucket][field][type_name] += 1
                if role != "system":
                    _collect_action_stats(embedded, action_counts, emotion_counts, valid_actions)

    top_level_fields = sorted(top_level_presence)
    return {
        "file": str(path),
        "records_seen": records_seen,
        "malformed_records": malformed_records,
        "top_level": {
            "fields": top_level_fields,
            "missing_rate": {
                field: _missing_rate(top_level_presence[field], records_seen)
                for field in top_level_fields
            },
            "types": _counter_map(top_level_types),
        },
        "chat": {
            "role_counts": dict(role_counts),
            "message_count_histogram": {str(k): v for k, v in sorted(message_count_histogram.items())},
        },
        "embedded_json": {
            bucket: {
                "fields": sorted(fields),
                "missing_rate": {
                    field: _missing_rate(count, records_seen)
                    for field, count in sorted(fields.items())
                },
                "types": _counter_map(embedded_types[bucket]),
            }
            for bucket, fields in sorted(embedded_presence.items())
        },
        "action_id_counts": dict(action_counts.most_common()),
        "emotion_counts": dict(emotion_counts.most_common(20)),
        "valid_actions": sorted(valid_actions),
        "sample": {
            "record": sample_record,
            "messages": sample_messages,
            "embedded_json": sample_embedded,
        },
    }


def inspect_directory(raw_dir: Path, max_records: Optional[int] = None) -> Dict[str, Any]:
    files = sorted(raw_dir.glob("*.jsonl"))
    summaries = [inspect_file(path, max_records=max_records) for path in files]
    return {
        "raw_dir": str(raw_dir),
        "max_records_per_file": max_records,
        "files": summaries,
        "conclusion": build_conclusion(summaries),
    }


def build_conclusion(files: List[Dict[str, Any]]) -> Dict[str, Any]:
    top_fields = sorted({field for item in files for field in item["top_level"]["fields"]})
    embedded_buckets = sorted({bucket for item in files for bucket in item["embedded_json"]})
    has_state_json = any("user_json" in item["embedded_json"] for item in files)
    has_action_json = any("assistant_json" in item["embedded_json"] for item in files)
    action_ids = sorted({action for item in files for action in item["action_id_counts"]})
    return {
        "top_level_fields": top_fields,
        "embedded_json_buckets": embedded_buckets,
        "has_state_json": has_state_json,
        "has_action_json": has_action_json,
        "action_ids": action_ids,
        "needs_conversion": top_fields == ["text"],
        "notes": [
            "Top-level JSONL rows are chat-template records, not canonical state/action samples.",
            "Reasoner files contain NPC state JSON in the user message.",
            "Formatter files contain selected_action/emotion JSON in the assistant message.",
            "Conversion must parse chat messages and map simulation fields to StateFeatures/PolicyAction.",
        ],
    }


def write_json_report(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown_report(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Stateful RPG NPC Dataset Schema Report", ""]
    lines.append(f"Raw directory: `{report['raw_dir']}`")
    lines.append(f"Max records per file: `{report['max_records_per_file']}`")
    lines.append("")
    conclusion = report["conclusion"]
    lines.append("## Conclusion")
    lines.append("")
    lines.append(f"- Top-level fields: `{', '.join(conclusion['top_level_fields'])}`")
    lines.append(f"- Embedded JSON buckets: `{', '.join(conclusion['embedded_json_buckets'])}`")
    lines.append(f"- Needs conversion: `{conclusion['needs_conversion']}`")
    lines.append(f"- Action ids found: `{', '.join(conclusion['action_ids'])}`")
    for note in conclusion["notes"]:
        lines.append(f"- {note}")
    lines.append("")

    for file_report in report["files"]:
        lines.append(f"## {Path(file_report['file']).name}")
        lines.append("")
        lines.append(f"- Records inspected: `{file_report['records_seen']}`")
        lines.append(f"- Malformed chat records: `{file_report['malformed_records']}`")
        lines.append(f"- Top-level fields: `{', '.join(file_report['top_level']['fields'])}`")
        lines.append(f"- Chat roles: `{json.dumps(file_report['chat']['role_counts'], ensure_ascii=False)}`")
        if file_report["valid_actions"]:
            lines.append(f"- Valid actions: `{', '.join(file_report['valid_actions'])}`")
        if file_report["action_id_counts"]:
            top_actions = list(file_report["action_id_counts"].items())[:10]
            lines.append(f"- Top action labels: `{json.dumps(dict(top_actions), ensure_ascii=False)}`")
        if file_report["emotion_counts"]:
            lines.append(f"- Top emotions: `{json.dumps(file_report['emotion_counts'], ensure_ascii=False)}`")
        lines.append("")
        lines.append("### Embedded JSON Fields")
        lines.append("")
        for bucket, info in file_report["embedded_json"].items():
            lines.append(f"**{bucket}**")
            for field in info["fields"]:
                type_counts = info["types"].get(field, {})
                missing = info["missing_rate"].get(field, 0.0)
                lines.append(f"- `{field}`: types `{type_counts}`, missing `{missing}`")
            lines.append("")
        lines.append("### Sample")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(file_report["sample"], ensure_ascii=False, indent=2)[:4000])
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _collect_action_stats(
    value: Any,
    action_counts: Counter[str],
    emotion_counts: Counter[str],
    valid_actions: set[str],
) -> None:
    if not isinstance(value, Mapping):
        return
    selected = value.get("selected_action")
    if isinstance(selected, Mapping) and selected.get("action_id"):
        action_counts[str(selected["action_id"])] += 1
    if value.get("emotion"):
        emotion_counts[str(value["emotion"])] += 1
    actions = value.get("valid_actions")
    if isinstance(actions, list):
        valid_actions.update(str(action) for action in actions)


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, Mapping):
        return "dict"
    return type(value).__name__


def _counter_map(data: Mapping[str, Counter[str]]) -> Dict[str, Dict[str, int]]:
    return {key: dict(counter) for key, counter in sorted(data.items())}


def _missing_rate(present: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((total - present) / total, 4)


def _preview(text: str, limit: int = 320) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Stateful RPG NPC JSONL schema")
    parser.add_argument("--raw-dir", default="../archive/data", help="Directory with raw .jsonl files")
    parser.add_argument("--max-records", type=int, default=1000,
                        help="Records to inspect per file; use 0 for all records")
    parser.add_argument("--out-json", default="evaluation/datasets/schema_report.json")
    parser.add_argument("--out-md", default="evaluation/datasets/schema_report.md")
    args = parser.parse_args()

    max_records = None if args.max_records == 0 else args.max_records
    report = inspect_directory(Path(args.raw_dir), max_records=max_records)
    write_json_report(report, Path(args.out_json))
    write_markdown_report(report, Path(args.out_md))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
