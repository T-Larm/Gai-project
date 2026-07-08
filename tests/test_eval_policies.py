import json

import pytest

from evaluation.eval_policies import evaluate_policy, heuristic_predict_fn, load_split


def _record(action_id, thi=0.9):
    return {
        "label": {"action_id": action_id},
        "features": {"categorical": {}, "multi": {}, "continuous": {"thi": thi}},
        "source_state": {
            "vitals": {"hp": 100, "hp_max": 120, "en": 0.8, "hun": 0.2, "thi": thi, "str": 0.5},
            "inv": [{"id": "water", "n": 1}],
            "percepts": [],
            "sched": {"act": "idle"},
        },
    }


def test_evaluate_policy_reports_accuracy_and_macro_f1():
    records = [_record("drink"), _record("drink"), _record("walk_to")]
    result = evaluate_policy(records, lambda record: "drink")

    assert result["n"] == 3
    assert result["accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    # drink: P=2/3, R=1 -> 0.8; walk_to: never predicted -> 0.
    assert result["macro_f1"] == pytest.approx(0.4)
    assert result["per_class_f1"]["drink"] == pytest.approx(0.8)
    assert result["confusion"]["walk_to"]["drink"] == 1


def test_heuristic_predict_fn_uses_raw_state():
    record = _record("drink", thi=0.9)
    assert heuristic_predict_fn(record) == "drink"


def test_load_split_reads_jsonl(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        "\n".join(json.dumps(_record("drink")) for _ in range(2)),
        encoding="utf-8",
    )
    assert len(load_split(path)) == 2
