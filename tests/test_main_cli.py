from backend.main import _build_arg_parser


def test_speak_flag_defaults_to_false():
    args = _build_arg_parser().parse_args(["--npc", "aldric", "--text"])
    assert args.speak is False


def test_speak_flag_can_be_enabled():
    args = _build_arg_parser().parse_args(["--npc", "aldric", "--text", "--speak"])
    assert args.speak is True


def test_condition_flags_default_to_full_system():
    args = _build_arg_parser().parse_args(["--npc", "aldric", "--text"])
    assert args.no_memory is False
    assert args.prompt_style == "layered"


def test_condition_flags_can_select_ablation():
    args = _build_arg_parser().parse_args(
        ["--npc", "aldric", "--text", "--no-memory", "--prompt-style", "flat"]
    )
    assert args.no_memory is True
    assert args.prompt_style == "flat"


def test_prompt_style_rejects_unknown_value():
    import pytest
    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args(
            ["--npc", "aldric", "--text", "--prompt-style", "fancy"]
        )
