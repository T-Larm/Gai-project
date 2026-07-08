"""
Memory stream with three-factor retrieval ranking:
semantic relevance + recency decay + importance.
"""
import math
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from backend.config.settings import (
    EMBEDDING_MODEL,
    IMPORTANCE_WEIGHTS,
    MEMORY_MAX_SIZE,
    MEMORY_RECENCY_HALFLIFE_SEC,
    MEMORY_TOP_K,
)

_embedder: Optional[Any] = None


def _get_embedder() -> Any:
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "sentence-transformers is required for semantic memory retrieval. "
                "Install project requirements or disable memory retrieval for light tests."
            ) from exc
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


@dataclass
class MemoryEntry:
    content: str
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5   # 0–1, higher = more important
    embedding: Optional[Any] = field(default=None, repr=False, compare=False)


class MemoryStream:
    def __init__(self, max_size: int = MEMORY_MAX_SIZE):
        self.entries: List[MemoryEntry] = []
        self.max_size = max_size

    def add(self, content: str, importance: float = 0.5) -> None:
        self.entries.append(MemoryEntry(content=content, importance=importance))
        if len(self.entries) > self.max_size:
            self.entries.pop(0)

    def retrieve(self, query: str, top_k: int = MEMORY_TOP_K) -> List[str]:
        """Rank entries by semantic relevance + recency decay + importance,
        and return the content of the top_k highest-scoring entries."""
        if not self.entries:
            return []

        embedder = _get_embedder()
        try:
            import torch
            from sentence_transformers import util
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "torch and sentence-transformers are required for semantic memory retrieval. "
                "Install project requirements or disable memory retrieval."
            ) from exc
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        missing = [e for e in self.entries if e.embedding is None]
        if missing:
            new_embeddings = embedder.encode(
                [e.content for e in missing], convert_to_tensor=True
            )
            for entry, emb in zip(missing, new_embeddings):
                entry.embedding = emb
        content_embeddings = torch.stack([e.embedding for e in self.entries])
        similarities = util.cos_sim(query_embedding, content_embeddings)[0]

        now = time.time()
        decay_rate = math.log(2) / MEMORY_RECENCY_HALFLIFE_SEC
        weights = IMPORTANCE_WEIGHTS

        scored = []
        for entry, similarity in zip(self.entries, similarities):
            recency = math.exp(-decay_rate * (now - entry.timestamp))
            score = (
                weights["semantic"] * float(similarity)
                + weights["recency"] * recency
                + weights["importance"] * entry.importance
            )
            scored.append((score, entry))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry.content for _, entry in scored[:top_k]]

    def recent(self, n: int) -> List[str]:
        """Return the content of the n most recently added entries, oldest first."""
        return [entry.content for entry in self.entries[-n:]]

    def to_list(self) -> List[dict]:
        return [
            {"content": e.content, "timestamp": e.timestamp, "importance": e.importance}
            for e in self.entries
        ]

    @classmethod
    def from_list(cls, data: List[dict]) -> "MemoryStream":
        stream = cls()
        for item in data:
            stream.entries.append(
                MemoryEntry(
                    content=item["content"],
                    timestamp=item.get("timestamp", time.time()),
                    importance=item.get("importance", 0.5),
                )
            )
        return stream
