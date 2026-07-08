"""Evaluate behavior policies on the v2 test split (RQ1 comparison).

Policies compared on identical states:
- heuristic: hand-written survival baseline (backend/behavior/heuristic_policy)
- trained:   supervised MLP checkpoint (pass --checkpoint)
- llm:       LLM-as-policy via Ollama (pass --llm-model; see llm_policy module)

Example:
    python -m evaluation.eval_policies \
        --data-dir data/behavior_policy/stateful_rpg_v2 \
        --checkpoint data/behavior_policy/checkpoints/stateful_rpg_v2_mlp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping

from backend.behavior.heuristic_policy import heuristic_action_id
from backend.behavior.supervised_policy import confusion_matrix, macro_f1


PredictFn = Callable[[Mapping[str, Any]], str]


def load_split(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_policy(records: List[Mapping[str, Any]], predict_fn: PredictFn) -> Dict[str, Any]:
    gold = [str(record["label"]["action_id"]) for record in records]
    predicted = [str(predict_fn(record)) for record in records]

    classes = sorted(set(gold))
    class_index = {label: index for index, label in enumerate(classes)}
    gold_idx = [class_index[label] for label in gold]
    pred_idx = [class_index.get(label, -1) for label in predicted]

    per_class_f1 = {}
    for label in classes:
        cls = class_index[label]
        tp = sum(1 for p, g in zip(pred_idx, gold_idx) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(pred_idx, gold_idx) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(pred_idx, gold_idx) if p != cls and g == cls)
        denominator = 2 * tp + fp + fn
        per_class_f1[label] = round((2 * tp / denominator) if denominator else 0.0, 4)

    correct = sum(1 for p, g in zip(predicted, gold) if p == g)
    return {
        "n": len(records),
        "accuracy": round(correct / len(records), 4) if records else 0.0,
        "macro_f1": round(macro_f1(pred_idx, gold_idx), 4),
        "per_class_f1": per_class_f1,
        "confusion": confusion_matrix(predicted, gold),
    }


def heuristic_predict_fn(record: Mapping[str, Any]) -> str:
    return heuristic_action_id(record["source_state"])


def trained_predict_fn(checkpoint_dir: Path) -> PredictFn:
    from backend.behavior.supervised_policy import ACTION_HEAD, SupervisedPolicy

    policy = SupervisedPolicy(checkpoint_dir)

    def predict(record: Mapping[str, Any]) -> str:
        return policy.predict(record["features"])[ACTION_HEAD]

    return predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate behavior policies on the v2 test split")
    parser.add_argument("--data-dir", default="data/behavior_policy/stateful_rpg_v2")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--checkpoint", default=None,
                        help="Trained MLP checkpoint dir; omit to skip the trained policy")
    parser.add_argument("--llm-model", default=None,
                        help="Ollama model name for LLM-as-policy; omit to skip")
    parser.add_argument("--max-records", type=int, default=0,
                        help="Evaluate at most N records (0 = all); useful for the slow LLM policy")
    parser.add_argument("--out", default=None, help="Write full results JSON here")
    args = parser.parse_args()

    records = load_split(Path(args.data_dir) / f"{args.split}.jsonl")
    if args.max_records:
        records = records[: args.max_records]

    results: Dict[str, Any] = {"data_dir": args.data_dir, "split": args.split, "n": len(records)}

    results["heuristic"] = evaluate_policy(records, heuristic_predict_fn)

    if args.checkpoint:
        results["trained"] = evaluate_policy(records, trained_predict_fn(Path(args.checkpoint)))

    if args.llm_model:
        from evaluation.llm_policy import llm_predict_fn

        results["llm"] = evaluate_policy(records, llm_predict_fn(args.llm_model))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        name: {"accuracy": info["accuracy"], "macro_f1": info["macro_f1"]}
        for name, info in results.items()
        if isinstance(info, dict) and "accuracy" in info
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
