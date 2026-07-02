"""LLM-as-judge: prompt-to-line persona consistency (binary, TIFA-inspired QA).

Reads transcript JSONL from run_dialogues.py, asks a judge LLM one binary
question per reply, writes judged JSONL and prints per-condition rates.

Usage (from project root, Ollama running):
    python -m evaluation.judge_consistency evaluation/results/aldric_full.jsonl
    python -m evaluation.judge_consistency results/*.jsonl --judge-model llama3:latest
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

from backend.llm.ollama_client import OllamaClient

SEEDS_PATH = os.path.join("data", "seeds", "example_seeds.json")

_JUDGE_TEMPLATE = """\
You are evaluating whether a game NPC's reply stays in character.

The NPC's character sheet (ground truth):
  Name: {name}
  Occupation: {occupation}
  Personality: {personality}
  Relationships: {relationships}

Player said: {player}
NPC replied: {reply}

Question: Is this reply consistent with the character sheet — the right
occupation, personality and world knowledge, with no breaking character
(e.g. mentioning being an AI or modern technology)?

Answer with exactly one word: YES or NO.
"""


def parse_verdict(raw: str) -> Optional[bool]:
    """Map judge output to True/False; None if unparseable."""
    token = raw.strip().split()[0].strip(".,:;—-").lower() if raw.strip() else ""
    if token == "yes":
        return True
    if token == "no":
        return False
    return None


def seed_summary(npc_name: str, seeds_path: str = SEEDS_PATH) -> dict:
    with open(seeds_path, encoding="utf-8") as f:
        seeds = json.load(f)
    for seed in seeds:
        if seed["name"].lower() == npc_name.lower():
            return seed
    raise ValueError(f"No seed found for NPC '{npc_name}'")


def judge_record(judge_llm, seed: dict, record: dict) -> Optional[bool]:
    prompt = _JUDGE_TEMPLATE.format(
        name=seed["name"],
        occupation=seed["occupation"],
        personality=", ".join(seed["personality_tags"]),
        relationships=json.dumps(seed["relationships"]),
        player=record["player"],
        reply=record["reply"],
    )
    return parse_verdict(judge_llm.generate(prompt))


def summarize_rates(judged_records) -> dict:
    """(condition, category) -> in-character rate over parseable verdicts."""
    grouped = defaultdict(list)
    for record in judged_records:
        if record.get("in_character") is not None:
            grouped[(record["condition"], record["category"])].append(
                record["in_character"]
            )
    return {key: sum(vals) / len(vals) for key, vals in grouped.items()}


def main():
    parser = argparse.ArgumentParser(description="Judge persona consistency")
    parser.add_argument("transcripts", nargs="+", help="JSONL files from run_dialogues")
    parser.add_argument("--judge-model", default="llama3:latest")
    args = parser.parse_args()

    judge_llm = OllamaClient(model=args.judge_model)
    judged = []
    for path in args.transcripts:
        seed_cache = {}
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        for record in records:
            if record.get("suite") != "prompts":
                continue
            npc = record["npc"]
            if npc not in seed_cache:
                seed_cache[npc] = seed_summary(npc)
            record["in_character"] = judge_record(judge_llm, seed_cache[npc], record)
            record["judge_model"] = args.judge_model
            judged.append(record)

        out = Path(path).with_suffix(".judged.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for record in judged:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[Judge] {path} -> {out}")

    unparsed = sum(1 for r in judged if r["in_character"] is None)
    print(f"\n[Judge] {len(judged)} replies judged, {unparsed} unparseable verdicts")
    print(f"{'condition':<14}{'category':<14}{'in-character rate':>18}")
    for (condition, category), rate in sorted(summarize_rates(judged).items()):
        print(f"{condition:<14}{category:<14}{rate:>17.1%}")


if __name__ == "__main__":
    main()
