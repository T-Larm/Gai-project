from backend.behavior.native_features import NATIVE_ACTIONS
from evaluation.eval_barks import (
    SCENARIOS,
    build_judge_prompt,
    parse_verdict,
    run_bark_eval,
)


def test_scenarios_cover_every_native_action():
    assert set(SCENARIOS) == set(NATIVE_ACTIONS)
    for action, scenario in SCENARIOS.items():
        assert "vitals" in scenario["state"], action
        assert scenario["situation"], action


def test_parse_verdict_reads_yes_no_robustly():
    assert parse_verdict("YES") is True
    assert parse_verdict("yes, it fits the character.") is True
    assert parse_verdict("NO.") is False
    assert parse_verdict("No - too formal for a blacksmith.") is False
    assert parse_verdict("maybe?") is None


def test_build_judge_prompt_mentions_persona_line_and_question():
    persona = {"name": "Aldric", "occupation": "Blacksmith",
               "speech_style": "blunt", "traits": ["gruff"]}
    prompt = build_judge_prompt(
        kind="persona",
        persona=persona,
        line="Throat's dry as forge ash.",
        action="drink",
        situation="suffering from thirst",
    )
    assert "Aldric" in prompt and "Blacksmith" in prompt
    assert "Throat's dry as forge ash." in prompt
    assert "YES or NO" in prompt

    action_prompt = build_judge_prompt(
        kind="action",
        persona=persona,
        line="Throat's dry as forge ash.",
        action="drink",
        situation="suffering from thirst",
    )
    assert "drink" in action_prompt


class _ScriptedJudgeLLM:
    """Generator says a fixed line; judge says YES for ours, NO for others."""

    def generate(self, prompt, system=""):
        if "YES or NO" in prompt:
            return "YES" if "in-character line" in prompt else "NO"
        return "An in-character line."


def test_run_bark_eval_computes_fit_rates():
    personas = {"aldric": {"name": "Aldric", "occupation": "Blacksmith",
                           "speech_style": "blunt", "traits": ["gruff"]}}
    scenarios = {"drink": {"state": {"vitals": {"thi": 0.9}}, "situation": "thirsty"}}

    def generate_line(condition, persona, state, action):
        return "An in-character line." if condition == "ours" else "Generic filler."

    result = run_bark_eval(
        personas=personas,
        scenarios=scenarios,
        conditions=("ours", "template"),
        generate_line=generate_line,
        judge_llm=_ScriptedJudgeLLM(),
    )

    assert result["ours"]["persona_fit_rate"] == 1.0
    assert result["ours"]["action_fit_rate"] == 1.0
    assert result["template"]["persona_fit_rate"] == 0.0
    assert len(result["transcripts"]) == 2  # one bark per condition
    assert result["transcripts"][0]["line"]
