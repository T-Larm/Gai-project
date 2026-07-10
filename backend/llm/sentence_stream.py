"""
Incremental sentence splitter for streamed LLM tokens.

Feeds sentence-level chunks to XTTS while the LLM is still generating.
Sentence boundary = terminal punctuation [.!?…] followed by whitespace
(same convention as dialogue.truncate_to_sentences). Ultra-short fragments
("Ah.") are merged into the following sentence so XTTS calls stay worth
their fixed overhead.
"""
import re
from typing import Iterable, Iterator

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")

# Fragments shorter than this ride along with the next sentence.
MIN_SENTENCE_CHARS = 8


def stream_sentences(tokens: Iterable[str], max_sentences: int) -> Iterator[str]:
    """Yield complete sentences from a token stream, at most `max_sentences`.

    Returning early (sentence cap) stops pulling from `tokens`, which closes
    the upstream generator — for OllamaClient.chat_stream this aborts the
    LLM generation.
    """
    buffer = ""
    pending = ""   # short fragment waiting to merge with the next sentence
    emitted = 0

    for token in tokens:
        buffer += token
        parts = _SENTENCE_BOUNDARY.split(buffer)
        buffer = parts.pop()   # last part has no boundary after it yet
        for part in parts:
            sentence = f"{pending} {part}".strip() if pending else part.strip()
            if len(sentence) < MIN_SENTENCE_CHARS:
                pending = sentence
                continue
            pending = ""
            yield sentence
            emitted += 1
            if emitted >= max_sentences:
                return

    tail = f"{pending} {buffer}".strip() if pending else buffer.strip()
    if tail and emitted < max_sentences:
        yield tail
