"""Train a supervised multi-head NPC behavior policy.

Example:
    python -m evaluation.train_policy \
        --data-dir data/behavior_policy/stateful_rpg \
        --out-dir data/behavior_policy/checkpoints/stateful_rpg_mlp
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from backend.behavior.supervised_policy import (
    ACTION_HEADS,
    SOURCE_ACTION_HEAD,
    accuracy_for_logits,
    build_feature_spec,
    build_label_spec,
    confusion_matrix,
    label_value,
    load_jsonl,
    make_model,
    require_torch,
    save_checkpoint,
    tensors_from_samples,
)


def resolve_device(torch: Any, device: str, allow_cpu: bool = False) -> Any:
    requested = device.lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but no CUDA GPU is available. "
                "Use the school A40 environment, or pass --allow-cpu --device cpu only for debugging."
            )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        return torch.device(requested)
    if requested == "cpu" and not allow_cpu:
        raise RuntimeError(
            "CPU training is disabled for this project step. "
            "Use --device cuda on the A40 machine, or add --allow-cpu only for local smoke tests."
        )
    return torch.device(requested)


def train_policy(
    data_dir: Path,
    out_dir: Path,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    seed: int = 13,
    include_source_action: bool = True,
    device: str = "cuda",
    allow_cpu: bool = False,
    amp: bool = True,
    max_vocab_size: int = 512,
) -> Dict[str, Any]:
    torch = require_torch()
    train_device = resolve_device(torch, device=device, allow_cpu=allow_cpu)
    random.seed(seed)
    torch.manual_seed(seed)
    if train_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    train_samples = load_jsonl(data_dir / "train.jsonl")
    valid_samples = load_jsonl(data_dir / "valid.jsonl")
    test_samples = load_jsonl(data_dir / "test.jsonl")

    feature_spec = build_feature_spec(train_samples, max_vocab_size=max_vocab_size)
    label_spec = build_label_spec(train_samples, include_source_action=include_source_action)
    model = make_model(
        input_dim=feature_spec.input_dim,
        head_sizes={head: len(vocab) for head, vocab in label_spec.heads.items()},
        hidden_dim=hidden_dim,
    ).to(train_device)

    train_x, train_y = tensors_from_samples(train_samples, feature_spec, label_spec)
    valid_x, valid_y = tensors_from_samples(valid_samples, feature_spec, label_spec)
    test_x, test_y = tensors_from_samples(test_samples, feature_spec, label_spec)
    train_x = train_x.to(train_device)
    valid_x = valid_x.to(train_device)
    test_x = test_x.to(train_device)
    train_y = {head: values.to(train_device) for head, values in train_y.items()}
    valid_y = {head: values.to(train_device) for head, values in valid_y.items()}
    test_y = {head: values.to(train_device) for head, values in test_y.items()}

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    use_amp = bool(amp and train_device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(train_x.shape[0], device=train_device)
        total_loss = 0.0
        total_items = 0
        for start in range(0, train_x.shape[0], batch_size):
            batch_idx = order[start:start + batch_size]
            batch_x = train_x[batch_idx]
            batch_y = {head: values[batch_idx] for head, values in train_y.items()}
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(batch_x)
                loss = sum(loss_fn(logits[head], batch_y[head]) for head in label_spec.heads)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * batch_x.shape[0]
            total_items += int(batch_x.shape[0])

        valid_metrics = evaluate_tensors(model, valid_x, valid_y)
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(total_items, 1),
            "valid_accuracy": valid_metrics["accuracy"],
        })

    valid_metrics = evaluate_tensors(model, valid_x, valid_y)
    test_metrics = evaluate_tensors(model, test_x, test_y)
    metrics = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "hidden_dim": hidden_dim,
        "seed": seed,
        "device": str(train_device),
        "amp": use_amp,
        "max_vocab_size": max_vocab_size,
        "input_dim": feature_spec.input_dim,
        "heads": list(label_spec.heads),
        "train_records": len(train_samples),
        "valid_records": len(valid_samples),
        "test_records": len(test_samples),
        "history": history,
        "valid": valid_metrics,
        "test": test_metrics,
    }
    save_checkpoint(out_dir, model, feature_spec, label_spec, hidden_dim, metrics)
    write_metrics(out_dir, metrics)
    write_confusion_matrices(out_dir, model, test_samples, test_x, test_y, label_spec)
    return metrics


def evaluate_tensors(model: Any, features: Any, targets: Mapping[str, Any]) -> Dict[str, Any]:
    torch = require_torch()
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = sum(loss_fn(logits[head], targets[head]) for head in logits)
        accuracies = accuracy_for_logits(logits, targets)
    return {"loss": float(loss.item()), "accuracy": accuracies}


def write_metrics(out_dir: Path, metrics: Mapping[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_confusion_matrices(
    out_dir: Path,
    model: Any,
    samples: Sequence[Mapping[str, Any]],
    features: Any,
    targets: Mapping[str, Any],
    label_spec: Any,
) -> None:
    torch = require_torch()
    model.eval()
    with torch.no_grad():
        logits = model(features)
    matrices = {}
    for head, values in logits.items():
        labels = label_spec.labels_for(head)
        pred_indices = values.argmax(dim=1).tolist()
        gold_indices = targets[head].tolist()
        predicted = [labels[index] for index in pred_indices]
        gold = [labels[index] for index in gold_indices]
        matrices[head] = confusion_matrix(predicted, gold)
    (out_dir / "confusion_matrices.json").write_text(
        json.dumps(matrices, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train supervised behavior policy")
    parser.add_argument("--data-dir", default="data/behavior_policy/stateful_rpg")
    parser.add_argument("--out-dir", default="data/behavior_policy/checkpoints/stateful_rpg_mlp")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda",
                        help="Training device. Default is cuda and CPU fallback is disabled.")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Allow CPU only for local smoke tests; not for final training.")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable CUDA mixed precision.")
    parser.add_argument("--max-vocab-size", type=int, default=512,
                        help="Max values kept per categorical feature; rare values map to <UNK>.")
    parser.add_argument("--no-source-action-head", action="store_true",
                        help="Train only PolicyAction heads, excluding source_action_id")
    args = parser.parse_args()

    metrics = train_policy(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        include_source_action=not args.no_source_action_head,
        device=args.device,
        allow_cpu=args.allow_cpu,
        amp=not args.no_amp,
        max_vocab_size=args.max_vocab_size,
    )
    summary = {
        "valid_accuracy": metrics["valid"]["accuracy"],
        "test_accuracy": metrics["test"]["accuracy"],
        "out_dir": args.out_dir,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
