import json

from evaluation.datasets.convert_stateful_rpg import (
    build_formatter_index,
    convert_reasoner_file,
    heuristic_action_id,
    map_policy_action,
    map_state_features,
)
from backend.behavior.schemas import DialogueAct, Gesture, NpcEmotion, PlayerIntent


def _chat(system: str, user: str, assistant: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{assistant}<|eot_id|>"
    )


def test_build_formatter_index_reads_reasoning_to_action(tmp_path):
    path = tmp_path / "train_formatter.jsonl"
    path.write_text(
        json.dumps({
            "text": _chat(
                "system",
                "I need water.",
                '{"reasoning":"I need water.","selected_action":{"action_id":"drink",'
                '"target_id":null,"dialogue":null},"emotion":"Calm"}',
            )
        }),
        encoding="utf-8",
    )

    labels = build_formatter_index(path)

    assert labels["I need water."]["selected_action"]["action_id"] == "drink"


def test_convert_reasoner_file_prefers_formatter_match(tmp_path):
    path = tmp_path / "train_reasoner.jsonl"
    state = {
        "id": "npc_1",
        "occ": "King",
        "vitals": {"thi": 0.91, "hun": 0.1, "str": 0.2, "hp": 100, "hp_max": 100},
        "emo": {"mood": "Calm"},
        "inv": [{"id": "water", "n": 1}],
        "pos": {"zone": "Temple"},
        "factions": {"CityWatch": 0.2},
        "memories": [{"ew": -0.8, "desc": "old fear"}],
        "percepts": [],
    }
    path.write_text(
        json.dumps({"text": _chat("system", f"Intro {json.dumps(state)}", "I need water.")}),
        encoding="utf-8",
    )
    formatter = {
        "I need water.": {
            "selected_action": {"action_id": "drink", "target_id": None, "dialogue": None},
            "emotion": "Calm",
        }
    }

    sample = list(convert_reasoner_file(path, formatter, "train"))[0]

    assert sample["source"]["label_source"] == "formatter_match"
    assert sample["source_action"]["action_id"] == "drink"
    assert sample["state"]["npc_role"] == "king"
    assert sample["state"]["memory_relevance"] == 0.8
    assert sample["action"]["dialogue_act"] == "end_conversation"


def test_heuristic_action_prefers_flee_when_threat_exceeds_strength():
    state = {
        "vitals": {"str": 0.2, "thi": 0.95},
        "inv": [{"id": "water", "n": 1}],
        "percepts": [{"threat": 0.8}],
    }

    assert heuristic_action_id(state) == "flee"


def test_mapping_functions_project_simulation_state_to_policy_schema():
    state = {
        "id": "npc_2",
        "occ": "Guard",
        "vitals": {"str": 0.7},
        "pos": {"zone": "Barracks"},
        "factions": {"CityWatch": 0.8},
        "memories": [],
        "percepts": [{"threat": 0.2}],
        "inv": [{"id": "food", "n": 2}],
    }

    features = map_state_features(state, "attack", NpcEmotion.ANGRY)
    action = map_policy_action("attack", NpcEmotion.ANGRY)

    assert features.player_intent is PlayerIntent.THREATEN
    assert features.relationship.value == "ally"
    assert features.inventory_flags == ["food"]
    assert action.dialogue_act is DialogueAct.WARN
    assert action.gesture is Gesture.APPROACH
