from backend.main import _build_arg_parser


def test_speak_flag_defaults_to_false():
    args = _build_arg_parser().parse_args(["--npc", "aldric", "--text"])
    assert args.speak is False


def test_speak_flag_can_be_enabled():
    args = _build_arg_parser().parse_args(["--npc", "aldric", "--text", "--speak"])
    assert args.speak is True
