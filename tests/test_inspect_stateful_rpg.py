import json

from evaluation.datasets.inspect_stateful_rpg import (
    extract_json_objects,
    inspect_file,
    parse_chat_messages,
)


def _chat(system: str, user: str, assistant: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{assistant}<|eot_id|>"
    )


def test_parse_chat_messages_reads_llama_template_roles():
    messages = parse_chat_messages(_chat("sys", "hello", "reply"))

    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert messages[1]["content"] == "hello"


def test_extract_json_objects_handles_nested_objects():
    objects = extract_json_objects('prefix {"a":{"b":1},"c":[2]} suffix')

    assert objects == [{"a": {"b": 1}, "c": [2]}]


def test_inspect_file_reports_top_level_and_embedded_json(tmp_path):
    path = tmp_path / "sample.jsonl"
    rows = [
        {
            "text": _chat(
                "system",
                'Intro {"id":"npc_1","occ":"King","vitals":{"thi":0.9},'
                '"valid_actions":["drink","sleep"]}',
                "Susuzluk var.",
            )
        },
        {
            "text": _chat(
                "system",
                "Reasoning text.",
                '{"reasoning":"Reasoning text.","selected_action":{"action_id":"drink",'
                '"target_id":null,"dialogue":null},"emotion":"Calm"}',
            )
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    report = inspect_file(path)

    assert report["records_seen"] == 2
    assert report["top_level"]["fields"] == ["text"]
    assert "user_json" in report["embedded_json"]
    assert "assistant_json" in report["embedded_json"]
    assert "vitals.thi" in report["embedded_json"]["user_json"]["fields"]
    assert report["valid_actions"] == ["drink", "sleep"]
    assert report["action_id_counts"] == {"drink": 1}
