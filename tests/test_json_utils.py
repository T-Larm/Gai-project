"""parse_llm_json must extract JSON from messy LLM output or fail loudly."""
import pytest

from backend.llm.json_utils import parse_llm_json


class _FixerLLM:
    """Stub LLM used only for the fix-malformed-JSON fallback."""

    def __init__(self, fixed_output: str):
        self.fixed_output = fixed_output
        self.prompts = []

    def generate(self, prompt: str, system: str = "") -> str:
        self.prompts.append(prompt)
        return self.fixed_output


def test_parses_clean_json():
    assert parse_llm_json('{"faction": "Guild"}') == {"faction": "Guild"}


def test_strips_markdown_fences():
    raw = '```json\n{"faction": "Guild"}\n```'
    assert parse_llm_json(raw) == {"faction": "Guild"}


def test_extracts_json_block_from_surrounding_prose():
    raw = 'Sure! Here is the JSON:\n{"faction": "Guild"}\nHope that helps.'
    assert parse_llm_json(raw) == {"faction": "Guild"}


def test_asks_llm_to_fix_malformed_json():
    llm = _FixerLLM('{"faction": "Guild"}')

    result = parse_llm_json('{"faction": "Guild",}', llm=llm)

    assert result == {"faction": "Guild"}
    assert len(llm.prompts) == 1


def test_raises_value_error_including_raw_text_when_llm_fix_fails():
    llm = _FixerLLM("sorry, I cannot help with that")

    with pytest.raises(ValueError) as exc_info:
        parse_llm_json("totally not json", llm=llm)

    assert "totally not json" in str(exc_info.value)


def test_raises_value_error_when_unparseable_and_no_llm():
    with pytest.raises(ValueError) as exc_info:
        parse_llm_json("totally not json")

    assert "totally not json" in str(exc_info.value)
