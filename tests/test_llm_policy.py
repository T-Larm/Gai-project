import json

from evaluation.llm_policy import build_prompt, llm_predict_fn, parse_action


_STATE = {
    "occ": "King",
    "vitals": {"hp": 100, "hp_max": 120, "thi": 0.9},
    "inv": [{"id": "water", "n": 1}],
    "percepts": [],
}


def test_build_prompt_contains_actions_and_state():
    prompt = build_prompt(_STATE)
    assert '"thi": 0.9' in prompt or '"thi":0.9' in prompt
    assert "walk_to" in prompt
    assert "action_id" in prompt


def test_parse_action_accepts_json_and_fenced_json():
    assert parse_action('{"action_id": "drink"}') == "drink"
    assert parse_action('```json\n{"action_id": "flee"}\n```') == "flee"


def test_parse_action_falls_back_to_action_token_scan():
    assert parse_action("I think the NPC should sleep now.") == "sleep"


def test_parse_action_rejects_garbage():
    assert parse_action("no idea") is None
    assert parse_action('{"action_id": "dance"}') is None


class _StubClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, prompt, system=""):
        self.calls.append((prompt, system))
        return self.replies.pop(0)


def test_llm_predict_fn_predicts_and_counts_invalid():
    client = _StubClient(['{"action_id": "drink"}', "gibberish"])
    predict = llm_predict_fn("llama3", client=client)

    record = {"source_state": _STATE}
    assert predict(record) == "drink"
    assert predict(record) == "walk_to"  # fallback
    assert predict.stats == {"total": 2, "invalid": 1}
    assert json.dumps(_STATE, ensure_ascii=False) in client.calls[0][0]
