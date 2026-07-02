"""Phase 2: retrieve() must rank by semantic relevance + recency decay + importance."""
import time

import torch

import backend.llm.persona.memory as memory_module
from backend.llm.persona.memory import MemoryStream


class _ConstantEmbedder:
    """Stub embedder returning identical vectors for any input, so cosine
    similarity is always 1.0 and only recency/importance affect ranking."""

    def encode(self, texts, convert_to_tensor=True):
        if isinstance(texts, str):
            return torch.ones(1)
        return torch.ones(len(texts), 1)


def _use_constant_embedder(monkeypatch):
    monkeypatch.setattr(memory_module, "_get_embedder", lambda: _ConstantEmbedder())


def test_retrieve_ranks_semantically_relevant_memory_above_irrelevant_one():
    stream = MemoryStream()
    stream.add("The blacksmith forges swords from folded steel.", importance=0.5)
    stream.add("The weather today is sunny with a light breeze.", importance=0.5)
    # Tie-break recency so only semantic relevance differs.
    now = time.time()
    for entry in stream.entries:
        entry.timestamp = now

    results = stream.retrieve("Tell me about forging swords", top_k=2)

    assert results[0] == "The blacksmith forges swords from folded steel."


def test_retrieve_ranks_recent_memory_above_older_one_when_content_ties(monkeypatch):
    _use_constant_embedder(monkeypatch)
    stream = MemoryStream()
    stream.add("Memory A", importance=0.5)
    stream.add("Memory B", importance=0.5)
    stream.entries[0].timestamp = time.time() - 10_000  # old
    stream.entries[1].timestamp = time.time()           # recent

    results = stream.retrieve("hello", top_k=2)

    assert results == ["Memory B", "Memory A"]


def test_retrieve_ranks_higher_importance_above_lower_when_content_and_time_tie(monkeypatch):
    _use_constant_embedder(monkeypatch)
    stream = MemoryStream()
    # Higher-importance entry added *first* (older insertion order), so a
    # recency-only ranking would wrongly put the low-importance one on top.
    stream.add("Memory A", importance=0.9)
    stream.add("Memory B", importance=0.1)
    now = time.time()
    for entry in stream.entries:
        entry.timestamp = now

    results = stream.retrieve("quest", top_k=2)

    assert results == ["Memory A", "Memory B"]


def test_retrieve_respects_top_k(monkeypatch):
    _use_constant_embedder(monkeypatch)
    stream = MemoryStream()
    for i in range(10):
        stream.add(f"memory number {i}", importance=0.5)

    results = stream.retrieve("memory", top_k=3)

    assert len(results) == 3


def test_retrieve_on_empty_stream_returns_empty_list(monkeypatch):
    _use_constant_embedder(monkeypatch)
    stream = MemoryStream()

    assert stream.retrieve("anything") == []


class _CountingEmbedder(_ConstantEmbedder):
    """Records every text encoded, to verify entries are embedded only once."""

    def __init__(self):
        self.encoded_texts = []

    def encode(self, texts, convert_to_tensor=True):
        if isinstance(texts, str):
            self.encoded_texts.append(texts)
        else:
            self.encoded_texts.extend(texts)
        return super().encode(texts, convert_to_tensor=convert_to_tensor)


def test_retrieve_embeds_each_entry_only_once_across_calls(monkeypatch):
    embedder = _CountingEmbedder()
    monkeypatch.setattr(memory_module, "_get_embedder", lambda: embedder)
    stream = MemoryStream()
    stream.add("Memory A")
    stream.add("Memory B")

    stream.retrieve("first query")
    stream.add("Memory C")
    stream.retrieve("second query")

    entry_encodes = [t for t in embedder.encoded_texts if t.startswith("Memory")]
    assert sorted(entry_encodes) == ["Memory A", "Memory B", "Memory C"]


def test_recent_returns_last_n_contents_without_touching_embedder(monkeypatch):
    def _fail():
        raise AssertionError("recent() must not use the embedder")

    monkeypatch.setattr(memory_module, "_get_embedder", _fail)
    stream = MemoryStream()
    for i in range(5):
        stream.add(f"memory {i}")

    assert stream.recent(3) == ["memory 2", "memory 3", "memory 4"]


def test_to_list_from_list_round_trip():
    stream = MemoryStream()
    stream.add("The gate guard owes me a favour.", importance=0.7)
    stream.add("Iron prices doubled this week.", importance=0.3)

    restored = MemoryStream.from_list(stream.to_list())

    assert [(e.content, e.importance) for e in restored.entries] == [
        ("The gate guard owes me a favour.", 0.7),
        ("Iron prices doubled this week.", 0.3),
    ]
