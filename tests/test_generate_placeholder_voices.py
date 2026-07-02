import os

from scripts.generate_placeholder_voices import voice_line_for, voice_output_path


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
