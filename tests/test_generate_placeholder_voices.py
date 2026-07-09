import os
from types import SimpleNamespace

from scripts.generate_placeholder_voices import (
    _synthesize_in_subprocess,
    pick_voice_id,
    voice_line_for,
    voice_output_path,
)


def test_voice_output_path_lowercases_simple_name():
    path = voice_output_path("Aldric", voices_dir="data/voices")
    assert path == os.path.join("data/voices", "aldric.wav")


def test_voice_output_path_replaces_spaces_with_underscores():
    path = voice_output_path("Lord Vane", voices_dir="data/voices")
    assert path == os.path.join("data/voices", "lord_vane.wav")


def test_voice_line_for_includes_name_and_lowercased_occupation():
    line = voice_line_for("Aldric", "Blacksmith")
    assert "Aldric" in line
    assert "blacksmith" in line.lower()


def _voice(id_, gender, languages):
    return SimpleNamespace(id=id_, gender=gender, languages=languages)


def test_pick_voice_id_prefers_matching_gender_and_english():
    voices = [
        _voice("zh-female", "Female", ["zh-CN"]),
        _voice("en-female", "Female", ["en-US"]),
        _voice("en-male", "Male", ["en-US"]),
    ]
    assert pick_voice_id(voices, "female") == "en-female"
    assert pick_voice_id(voices, "male") == "en-male"


def test_pick_voice_id_falls_back_to_any_matching_gender_if_no_english():
    voices = [
        _voice("it-female", "Female", ["it-IT"]),
        _voice("en-male", "Male", ["en-US"]),
    ]
    assert pick_voice_id(voices, "female") == "it-female"


def test_pick_voice_id_falls_back_to_first_voice_if_gender_unavailable():
    voices = [_voice("en-male", "Male", ["en-US"])]
    assert pick_voice_id(voices, "female") == "en-male"


def test_synthesize_in_subprocess_passes_voice_id_and_out_path_then_cleans_up_temp_file(
    monkeypatch, tmp_path
):
    recorded = {}
    written_text_paths = []

    def fake_run(args, check, timeout):
        recorded["args"] = args
        recorded["check"] = check
        recorded["timeout"] = timeout
        text_path = args[-1]
        written_text_paths.append(text_path)
        assert os.path.exists(text_path)

    monkeypatch.setattr(
        "scripts.generate_placeholder_voices.subprocess.run", fake_run
    )

    out_path = str(tmp_path / "aldric.wav")
    _synthesize_in_subprocess("Hello, I am Aldric.", "voice-id-123", out_path)

    args = recorded["args"]
    assert args[-3] == "voice-id-123"
    assert args[-2] == out_path
    assert recorded["check"] is True
    assert recorded["timeout"] == 30
    # The temp text file must be cleaned up after the subprocess call.
    assert not os.path.exists(written_text_paths[0])
