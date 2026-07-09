"""Supervised behavior-policy inference and training utilities (方案 B).

Consumes the v2 dataset schema (data/behavior_policy/stateful_rpg_v2): each
sample carries ``features`` split into categorical / multi / continuous
groups, a native ``label.action_id`` (11 effective classes) and an auxiliary
``aux.mood`` emotion label.

This module is importable without PyTorch. Functions that build, train, or load
the neural model call ``require_torch()`` lazily so lightweight dataset tests can
run in minimal environments.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from backend.behavior.native_features import extract_native_features


ACTION_HEAD = "action_id"
MOOD_HEAD = "mood"
UNKNOWN_TOKEN = "<UNK>"


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for supervised policy training/inference. "
            "Install the project requirements or run with an environment that has torch."
        ) from exc
    return torch


@dataclass
class FeatureSpec:
    """Vectorization plan: continuous slots, one-hot and multi-hot vocabularies."""

    categorical: Dict[str, Dict[str, int]]
    multi: Dict[str, Dict[str, int]] = field(default_factory=dict)
    continuous: List[str] = field(default_factory=list)
    continuous_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    max_vocab_size: int = 512

    def to_dict(self) -> Dict[str, Any]:
        return {
            "categorical": self.categorical,
            "multi": self.multi,
            "continuous": self.continuous,
            "continuous_stats": self.continuous_stats,
            "max_vocab_size": self.max_vocab_size,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureSpec":
        return cls(
            categorical=_vocab_map(data.get("categorical", {})),
            multi=_vocab_map(data.get("multi", {})),
            continuous=[str(name) for name in data.get("continuous", [])],
            continuous_stats={
                str(name): {"mean": float(stats["mean"]), "std": float(stats["std"])}
                for name, stats in data.get("continuous_stats", {}).items()
            },
            max_vocab_size=int(data.get("max_vocab_size", 512)),
        )

    @property
    def input_dim(self) -> int:
        return (
            len(self.continuous)
            + sum(len(vocab) for vocab in self.categorical.values())
            + sum(len(vocab) for vocab in self.multi.values())
        )


@dataclass
class LabelSpec:
    heads: Dict[str, Dict[str, int]]

    def to_dict(self) -> Dict[str, Any]:
        return {"heads": self.heads}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LabelSpec":
        return cls(heads=_vocab_map(data.get("heads", {})))

    def labels_for(self, head: str) -> List[str]:
        vocab = self.heads[head]
        labels = [None] * len(vocab)
        for label, index in vocab.items():
            labels[index] = label
        return [str(label) for label in labels]


def build_feature_spec(
    samples: Sequence[Mapping[str, Any]],
    max_vocab_size: int = 512,
) -> FeatureSpec:
    categorical_counts: Dict[str, Counter] = {}
    multi_counts: Dict[str, Counter] = {}
    continuous_names: set = set()
    for sample in samples:
        features = _features_of(sample)
        for name, value in features.get("categorical", {}).items():
            categorical_counts.setdefault(name, Counter())[str(value)] += 1
        for name, values in features.get("multi", {}).items():
            counter = multi_counts.setdefault(name, Counter())
            for value in values:
                counter[str(value)] += 1
        continuous_names.update(features.get("continuous", {}))

    continuous = sorted(continuous_names)
    stats: Dict[str, Dict[str, float]] = {}
    for name in continuous:
        values = [
            _safe_float(_features_of(sample).get("continuous", {}).get(name, 0.0))
            for sample in samples
        ]
        mean = sum(values) / len(values) if values else 0.0
        variance = sum((v - mean) ** 2 for v in values) / len(values) if values else 0.0
        stats[name] = {"mean": mean, "std": variance ** 0.5}

    return FeatureSpec(
        categorical={
            name: _build_vocab(counts, max_vocab_size)
            for name, counts in sorted(categorical_counts.items())
        },
        multi={
            name: _build_vocab(counts, max_vocab_size, include_unknown=False)
            for name, counts in sorted(multi_counts.items())
        },
        continuous=continuous,
        continuous_stats=stats,
        max_vocab_size=max_vocab_size,
    )


def build_label_spec(
    samples: Sequence[Mapping[str, Any]],
    include_mood: bool = True,
) -> LabelSpec:
    head_names = [ACTION_HEAD] + ([MOOD_HEAD] if include_mood else [])
    heads: Dict[str, Dict[str, int]] = {}
    for head in head_names:
        labels = {UNKNOWN_TOKEN}
        for sample in samples:
            label = label_value(sample, head)
            if label:
                labels.add(label)
        heads[head] = {value: index for index, value in enumerate(sorted(labels))}
    return LabelSpec(heads=heads)


def encode_state_vector(features: Mapping[str, Any], spec: FeatureSpec) -> List[float]:
    """Vectorize one v2 ``features`` dict: continuous, then one-hot, then multi-hot."""
    continuous = features.get("continuous", {})
    categorical = features.get("categorical", {})
    multi = features.get("multi", {})

    vector: List[float] = []
    for name in spec.continuous:
        value = _safe_float(continuous.get(name, 0.0))
        stats = spec.continuous_stats.get(name)
        if stats:
            std = stats.get("std", 0.0)
            value = (value - stats.get("mean", 0.0)) / std if std > 1e-12 else 0.0
        vector.append(value)
    for name, vocab in spec.categorical.items():
        value = str(categorical.get(name, UNKNOWN_TOKEN))
        index = vocab.get(value, vocab.get(UNKNOWN_TOKEN, 0))
        vector.extend(1.0 if i == index else 0.0 for i in range(len(vocab)))
    for name, vocab in spec.multi.items():
        active = {vocab[str(value)] for value in multi.get(name, []) if str(value) in vocab}
        vector.extend(1.0 if i in active else 0.0 for i in range(len(vocab)))
    return vector


def labels_to_indices(sample: Mapping[str, Any], label_spec: LabelSpec) -> Dict[str, int]:
    indices = {}
    for head, vocab in label_spec.heads.items():
        label = label_value(sample, head)
        if label not in vocab:
            label = UNKNOWN_TOKEN
        indices[head] = vocab[label]
    return indices


def label_value(sample: Mapping[str, Any], head: str) -> str:
    if head == ACTION_HEAD:
        return str(sample.get("label", {}).get("action_id", ""))
    if head == MOOD_HEAD:
        return str(sample.get("aux", {}).get("mood", ""))
    raise ValueError(f"Unknown label head '{head}'")


def macro_f1(predicted: Sequence[int], gold: Sequence[int]) -> float:
    """Macro-averaged F1 over the classes present in gold (rare classes count equally)."""
    classes = sorted(set(gold))
    if not classes:
        return 0.0
    scores = []
    for cls in classes:
        tp = sum(1 for p, g in zip(predicted, gold) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(predicted, gold) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(predicted, gold) if p != cls and g == cls)
        denominator = 2 * tp + fp + fn
        scores.append((2 * tp / denominator) if denominator else 0.0)
    return sum(scores) / len(scores)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_model(input_dim: int, head_sizes: Mapping[str, int], hidden_dim: int = 128):
    torch = require_torch()
    nn = torch.nn

    class MultiHeadPolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.heads = nn.ModuleDict({
                name: nn.Linear(hidden_dim, size) for name, size in head_sizes.items()
            })

        def forward(self, inputs):
            encoded = self.encoder(inputs)
            return {name: head(encoded) for name, head in self.heads.items()}

    return MultiHeadPolicyNet()


def tensors_from_samples(
    samples: Sequence[Mapping[str, Any]],
    feature_spec: FeatureSpec,
    label_spec: LabelSpec,
):
    torch = require_torch()
    features = [encode_state_vector(_features_of(sample), feature_spec) for sample in samples]
    targets = {head: [] for head in label_spec.heads}
    for sample in samples:
        labels = labels_to_indices(sample, label_spec)
        for head, index in labels.items():
            targets[head].append(index)
    return (
        torch.tensor(features, dtype=torch.float32),
        {head: torch.tensor(values, dtype=torch.long) for head, values in targets.items()},
    )


def save_checkpoint(
    out_dir: Path,
    model: Any,
    feature_spec: FeatureSpec,
    label_spec: LabelSpec,
    hidden_dim: int,
    metrics: Mapping[str, Any],
) -> None:
    torch = require_torch()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    metadata = {
        "feature_spec": feature_spec.to_dict(),
        "label_spec": label_spec.to_dict(),
        "hidden_dim": hidden_dim,
        "input_dim": feature_spec.input_dim,
        "metrics": dict(metrics),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class SupervisedPolicy:
    """Load a trained checkpoint and predict native action + mood."""

    def __init__(self, checkpoint_dir: Path):
        torch = require_torch()
        self.torch = torch
        metadata_path = checkpoint_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.feature_spec = FeatureSpec.from_dict(metadata["feature_spec"])
        self.label_spec = LabelSpec.from_dict(metadata["label_spec"])
        self.model = make_model(
            input_dim=self.feature_spec.input_dim,
            head_sizes={head: len(vocab) for head, vocab in self.label_spec.heads.items()},
            hidden_dim=int(metadata.get("hidden_dim", 128)),
        )
        state = torch.load(checkpoint_dir / "model.pt", map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, state: Mapping[str, Any]) -> Dict[str, str]:
        logits = self._logits(state)
        prediction = {}
        for head in self.label_spec.heads:
            index = int(logits[head].argmax(dim=1).item())
            prediction[head] = self.label_spec.labels_for(head)[index]
        return prediction

    def predict_with_metadata(self, state: Mapping[str, Any]) -> Dict[str, Any]:
        logits = self._logits(state)
        predictions: Dict[str, Dict[str, Any]] = {}
        for head, values in logits.items():
            probs = self.torch.softmax(values, dim=1)[0]
            index = int(probs.argmax().item())
            predictions[head] = {
                "label": self.label_spec.labels_for(head)[index],
                "confidence": float(probs[index].item()),
            }
        return {
            "action": {head: info["label"] for head, info in predictions.items()},
            "predictions": predictions,
        }

    def _logits(self, state: Mapping[str, Any]):
        features = _features_of({"features": state}) if _is_features_dict(state) else extract_native_features(state)
        vector = encode_state_vector(features, self.feature_spec)
        with self.torch.no_grad():
            inputs = self.torch.tensor([vector], dtype=self.torch.float32)
            return self.model(inputs)


def accuracy_for_logits(logits: Mapping[str, Any], targets: Mapping[str, Any]) -> Dict[str, float]:
    accuracies = {}
    for head, values in logits.items():
        predicted = values.argmax(dim=1)
        correct = (predicted == targets[head]).float().mean().item()
        accuracies[head] = float(correct)
    return accuracies


def macro_f1_for_logits(logits: Mapping[str, Any], targets: Mapping[str, Any]) -> Dict[str, float]:
    scores = {}
    for head, values in logits.items():
        predicted = values.argmax(dim=1).tolist()
        gold = targets[head].tolist()
        scores[head] = macro_f1(predicted, gold)
    return scores


def confusion_matrix(predicted: Sequence[str], gold: Sequence[str]) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = {}
    for pred, actual in zip(predicted, gold):
        matrix.setdefault(actual, {})
        matrix[actual][pred] = matrix[actual].get(pred, 0) + 1
    return matrix


def _is_features_dict(state: Mapping[str, Any]) -> bool:
    return any(group in state for group in ("categorical", "multi", "continuous"))


def _features_of(sample: Mapping[str, Any]) -> Mapping[str, Any]:
    features = sample.get("features", {})
    return features if isinstance(features, Mapping) else {}


def _build_vocab(counts: Counter, max_vocab_size: int, include_unknown: bool = True) -> Dict[str, int]:
    values = [
        value for value, _count in counts.most_common(max(max_vocab_size - 1, 0))
        if value != UNKNOWN_TOKEN
    ]
    values = sorted(values)
    if include_unknown:
        values.insert(0, UNKNOWN_TOKEN)
    return {value: index for index, value in enumerate(values)}


def _vocab_map(data: Mapping[str, Any]) -> Dict[str, Dict[str, int]]:
    return {
        str(name): {str(key): int(value) for key, value in vocab.items()}
        for name, vocab in data.items()
    }


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number or number in (float("inf"), float("-inf")):
        return 0.0
    return number
