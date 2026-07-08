"""Supervised behavior-policy inference and training utilities.

This module is importable without PyTorch. Functions that build, train, or load
the neural model call ``require_torch()`` lazily so lightweight dataset tests can
run in minimal environments.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from backend.behavior.schemas import PolicyAction, StateFeatures


CONTINUOUS_FEATURES = [
    "trust_score",
    "memory_relevance",
    "danger_level",
    "distance_to_player",
    "forbidden_secret_asked",
    "prompt_injection_detected",
]

CATEGORICAL_FEATURES = [
    "player_intent",
    "quest_stage",
    "npc_emotion",
    "relationship",
    "npc_role",
    "persona_id",
    "location",
]

ACTION_HEADS = [
    "dialogue_act",
    "emotion",
    "disclosure_level",
    "gesture",
    "quest_update",
]

SOURCE_ACTION_HEAD = "source_action_id"
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
    categorical: Dict[str, Dict[str, int]]
    continuous: List[str]
    max_vocab_size: int = 512

    def to_dict(self) -> Dict[str, Any]:
        return {
            "categorical": self.categorical,
            "continuous": self.continuous,
            "max_vocab_size": self.max_vocab_size,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureSpec":
        return cls(
            categorical={
                str(name): {str(key): int(value) for key, value in vocab.items()}
                for name, vocab in data.get("categorical", {}).items()
            },
            continuous=[str(name) for name in data.get("continuous", CONTINUOUS_FEATURES)],
            max_vocab_size=int(data.get("max_vocab_size", 512)),
        )

    @property
    def input_dim(self) -> int:
        return len(self.continuous) + sum(len(vocab) for vocab in self.categorical.values())


@dataclass
class LabelSpec:
    heads: Dict[str, Dict[str, int]]

    def to_dict(self) -> Dict[str, Any]:
        return {"heads": self.heads}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LabelSpec":
        return cls(
            heads={
                str(name): {str(key): int(value) for key, value in vocab.items()}
                for name, vocab in data.get("heads", {}).items()
            }
        )

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
    vocabs: Dict[str, Dict[str, int]] = {}
    for name in CATEGORICAL_FEATURES:
        counts = Counter(_state_value(sample, name, UNKNOWN_TOKEN) for sample in samples)
        values = [UNKNOWN_TOKEN]
        for value, _count in counts.most_common(max(max_vocab_size - 1, 0)):
            if value != UNKNOWN_TOKEN:
                values.append(value)
        values = sorted(set(values))
        if UNKNOWN_TOKEN in values:
            values.remove(UNKNOWN_TOKEN)
        values.insert(0, UNKNOWN_TOKEN)
        vocabs[name] = {value: index for index, value in enumerate(values)}
    return FeatureSpec(
        categorical=vocabs,
        continuous=list(CONTINUOUS_FEATURES),
        max_vocab_size=max_vocab_size,
    )


def build_label_spec(
    samples: Sequence[Mapping[str, Any]],
    include_source_action: bool = True,
) -> LabelSpec:
    head_names = list(ACTION_HEADS)
    if include_source_action:
        head_names.append(SOURCE_ACTION_HEAD)

    heads: Dict[str, Dict[str, int]] = {}
    for head in head_names:
        labels = {UNKNOWN_TOKEN}
        for sample in samples:
            label = label_value(sample, head)
            if label:
                labels.add(label)
        heads[head] = {value: index for index, value in enumerate(sorted(labels))}
    return LabelSpec(heads=heads)


def encode_state_vector(state: Mapping[str, Any], spec: FeatureSpec) -> List[float]:
    vector: List[float] = []
    for name in spec.continuous:
        vector.append(_safe_float(state.get(name, 0.0)))
    for name, vocab in spec.categorical.items():
        value = str(state.get(name, UNKNOWN_TOKEN))
        index = vocab.get(value, vocab[UNKNOWN_TOKEN])
        vector.extend(1.0 if i == index else 0.0 for i in range(len(vocab)))
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
    if head == SOURCE_ACTION_HEAD:
        return str(sample.get("source_action", {}).get("action_id", ""))
    return str(sample.get("action", {}).get(head, ""))


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
    features = [encode_state_vector(sample["state"], feature_spec) for sample in samples]
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
    """Load a trained checkpoint and predict a PolicyAction."""

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

    def predict(self, state: StateFeatures | Mapping[str, Any]) -> PolicyAction:
        # 把连续特征和分类特征转变成模型能够认识的形式（神经网路输入向量），然后再根据这些值来预测输出一个PolicyAction
        state_dict = state.to_dict() if isinstance(state, StateFeatures) else dict(state)
        vector = encode_state_vector(state_dict, self.feature_spec)
        #关闭梯度，这不是训练，是推理，可以省内存，加速推理
        with self.torch.no_grad():
            inputs = self.torch.tensor([vector], dtype=self.torch.float32)
            logits = self.model(inputs)
        labels = {}
        for head in ACTION_HEADS:
            if head not in logits:
                continue
            index = int(logits[head].argmax(dim=1).item())
            labels[head] = self.label_spec.labels_for(head)[index]
        return PolicyAction.from_dict(labels)

    def predict_with_metadata(self, state: StateFeatures | Mapping[str, Any]) -> Dict[str, Any]:
        state_dict = state.to_dict() if isinstance(state, StateFeatures) else dict(state)
        vector = encode_state_vector(state_dict, self.feature_spec)
        with self.torch.no_grad():
            inputs = self.torch.tensor([vector], dtype=self.torch.float32)
            logits = self.model(inputs)
        predictions: Dict[str, Dict[str, Any]] = {}
        for head, values in logits.items():
            probs = self.torch.softmax(values, dim=1)[0]
            index = int(probs.argmax().item())
            predictions[head] = {
                "label": self.label_spec.labels_for(head)[index],
                "confidence": float(probs[index].item()),
            }
        return {
            "action": self.predict(state_dict).to_dict(),
            "predictions": predictions,
        }


def accuracy_for_logits(logits: Mapping[str, Any], targets: Mapping[str, Any]) -> Dict[str, float]:
    accuracies = {}
    for head, values in logits.items():
        predicted = values.argmax(dim=1)
        correct = (predicted == targets[head]).float().mean().item()
        accuracies[head] = float(correct)
    return accuracies


def confusion_matrix(predicted: Sequence[str], gold: Sequence[str]) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = {}
    for pred, actual in zip(predicted, gold):
        matrix.setdefault(actual, {})
        matrix[actual][pred] = matrix[actual].get(pred, 0) + 1
    return matrix


def _state_value(sample: Mapping[str, Any], name: str, default: str) -> str:
    return str(sample.get("state", {}).get(name, default))


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    if math.isnan(number) or math.isinf(number):
        return 0.0
    return number
