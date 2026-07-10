"""Incremental sentence splitter for streamed LLM tokens."""
from backend.llm.sentence_stream import stream_sentences


def test_splits_sentences_across_token_boundaries():
    tokens = ["Good morn", "ing, trav", "eller. What br", "ings you here? "]
    assert list(stream_sentences(tokens, max_sentences=4)) == [
        "Good morning, traveller.",
        "What brings you here?",
    ]


def test_flushes_trailing_partial_sentence_at_end_of_stream():
    tokens = ["First one. And then it just trails off"]
    assert list(stream_sentences(tokens, max_sentences=4)) == [
        "First one.",
        "And then it just trails off",
    ]


def test_stops_at_max_sentences_and_closes_upstream():
    consumed = []

    def tokens():
        for t in ["First sentence. ", "Second sentence. ", "Third sentence. ",
                  "Fourth sentence. ", "Fifth sentence. "]:
            consumed.append(t)
            yield t

    result = list(stream_sentences(tokens(), max_sentences=2))

    assert result == ["First sentence.", "Second sentence."]
    # Upstream must NOT be drained to the end: the whole point is aborting
    # Ollama generation early. Yielding "Two." needs at most 3 pulls.
    assert len(consumed) <= 3


def test_merges_short_fragment_into_next_sentence():
    tokens = ["Ah. Well met indeed, stranger. "]
    assert list(stream_sentences(tokens, max_sentences=4)) == [
        "Ah. Well met indeed, stranger.",
    ]


def test_empty_stream_yields_nothing():
    assert list(stream_sentences([], max_sentences=4)) == []


def test_handles_exclamation_question_and_ellipsis():
    tokens = ["By the forge! Really… now? Yes."]
    assert list(stream_sentences(tokens, max_sentences=4)) == [
        "By the forge!",
        "Really… now?",
        "Yes.",
    ]
