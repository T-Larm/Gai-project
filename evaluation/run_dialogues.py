"""Run evaluation dialogues under each experimental condition.

Produces JSONL transcripts consumed by judge_consistency.py.

Usage (from project root, Ollama running):
    python -m evaluation.run_dialogues --npc aldric --condition full --suite all
    python -m evaluation.run_dialogues --npc aldric --condition no_memory --suite memory
"""
import argparse
import json
import os
import time
from pathlib import Path

from backend.config.settings import PERSONAS_DIR
from backend.llm.dialogue import DialogueHandler
from backend.llm.ollama_client import OllamaClient
from backend.llm.persona.generator import PersonaGenerator

TEST_DATA_DIR = os.path.join("evaluation", "test_data")
HANDWRITTEN_DIR = os.path.join("evaluation", "handwritten_personas")
RESULTS_DIR = os.path.join("evaluation", "results")

# Experimental conditions: baselines (RQ1/RQ2) and ablations (RQ3).
CONDITIONS = {
    "full":        dict(prompt_style="layered", use_memory=True,  dynamic_updates=True),
    "no_memory":   dict(prompt_style="layered", use_memory=False, dynamic_updates=True),
    "flat":        dict(prompt_style="flat",    use_memory=False, dynamic_updates=False),
    "none":        dict(prompt_style="none",    use_memory=False, dynamic_updates=False),
    "handwritten": dict(use_memory=False, dynamic_updates=False),  # + system_prompt_text
}


def make_handler(llm, npc, condition: str,
                 handwritten_dir: str = HANDWRITTEN_DIR) -> DialogueHandler:
    kwargs = dict(CONDITIONS[condition])
    if condition == "handwritten":
        path = Path(handwritten_dir) / f"{npc.core.name.lower().replace(' ', '_')}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"No handwritten persona for {npc.core.name} at {path}"
            )
        kwargs["system_prompt_text"] = path.read_text(encoding="utf-8").strip()
    return DialogueHandler(llm, npc, **kwargs)


def _timed_respond(handler: DialogueHandler, text: str):
    start = time.perf_counter()
    reply = handler.respond(text)
    return reply, time.perf_counter() - start


def run_prompts(handler: DialogueHandler, prompts, npc_name: str, condition: str):
    """Sequential conversation over the prompt suite; one record per prompt."""
    records = []
    for turn_index, prompt in enumerate(prompts, start=1):
        reply, latency = _timed_respond(handler, prompt["text"])
        records.append({
            "suite": "prompts",
            "npc": npc_name,
            "condition": condition,
            "id": prompt["id"],
            "category": prompt["category"],
            "turn_index": turn_index,
            "player": prompt["text"],
            "reply": reply,
            "latency_s": round(latency, 3),
        })
    return records


def run_memory_probes(handler_factory, probes, npc_name: str, condition: str):
    """Fresh session per probe: fact -> fillers -> recall question -> keyword hit."""
    records = []
    for probe in probes:
        handler = handler_factory()
        handler.respond(probe["fact_setup"])
        for filler in probe["filler_turns"]:
            handler.respond(filler)
        reply, latency = _timed_respond(handler, probe["question"])
        reply_lower = reply.lower()
        hit = all(k.lower() in reply_lower for k in probe["expected_keywords"])
        records.append({
            "suite": "memory",
            "npc": npc_name,
            "condition": condition,
            "id": probe["id"],
            "player": probe["question"],
            "reply": reply,
            "expected_keywords": probe["expected_keywords"],
            "hit": hit,
            "latency_s": round(latency, 3),
        })
    return records


def run_consistency_pairs(handler: DialogueHandler, pairs, npc_name: str, condition: str):
    """Ask every question_a first, then every question_b — the gap between the
    two phrasings of the same fact is len(pairs) turns."""
    records = []
    replies_a = {}
    for pair in pairs:
        reply, _ = _timed_respond(handler, pair["question_a"])
        replies_a[pair["id"]] = reply
    for pair in pairs:
        reply_b, _ = _timed_respond(handler, pair["question_b"])
        records.append({
            "suite": "pairs",
            "npc": npc_name,
            "condition": condition,
            "id": pair["id"],
            "question_a": pair["question_a"],
            "reply_a": replies_a[pair["id"]],
            "question_b": pair["question_b"],
            "reply_b": reply_b,
        })
    return records


def _load_json(name: str):
    with open(os.path.join(TEST_DATA_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def _write_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run evaluation dialogues")
    parser.add_argument("--npc", required=True, help="NPC name (persona must exist)")
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--suite", default="all",
                        choices=["prompts", "memory", "pairs", "all"])
    parser.add_argument("--out", default=RESULTS_DIR)
    args = parser.parse_args()

    llm = OllamaClient()
    persona_path = os.path.join(PERSONAS_DIR, f"{args.npc.lower()}.json")

    def fresh_handler():
        npc = PersonaGenerator.load(persona_path)  # fresh copy, nothing persisted
        npc.memory_log = []
        return make_handler(llm, npc, args.condition)

    out = Path(args.out) / f"{args.npc.lower()}_{args.condition}.jsonl"
    print(f"[Eval] NPC={args.npc} condition={args.condition} -> {out}")

    if args.suite in ("prompts", "all"):
        records = run_prompts(fresh_handler(), _load_json("dialogue_prompts.json"),
                              args.npc, args.condition)
        _write_jsonl(records, out)
        print(f"[Eval] prompts: {len(records)} records")
    if args.suite in ("memory", "all"):
        records = run_memory_probes(fresh_handler, _load_json("memory_probes.json"),
                                    args.npc, args.condition)
        _write_jsonl(records, out)
        hits = sum(r["hit"] for r in records)
        print(f"[Eval] memory: {hits}/{len(records)} recalled")
    if args.suite in ("pairs", "all"):
        records = run_consistency_pairs(fresh_handler(),
                                        _load_json("consistency_pairs.json"),
                                        args.npc, args.condition)
        _write_jsonl(records, out)
        print(f"[Eval] pairs: {len(records)} records")


if __name__ == "__main__":
    main()
