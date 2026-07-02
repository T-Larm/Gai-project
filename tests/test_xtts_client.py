import pytest

import backend.tts.xtts_client as xtts_module
from backend.tts.xtts_client import XTTSClient


class _FakeSynthesizer:
    output_sample_rate = 24000


class _FakeTTSModel:
    def __init__(self):
        self.calls = []
        self.synthesizer = _FakeSynthesizer()

    def tts(self, text, speaker_wav, language):
        self.calls.append((text, speaker_wav, language))
        return [0.0, 0.1, 0.2]


def test_speak_raises_when_speaker_wav_missing(tmp_path):
    client = XTTSClient()
    missing_path = str(tmp_path / "nope.wav")

    with pytest.raises(FileNotFoundError) as exc_info:
        client.speak("hello", missing_path)

    assert "nope.wav" in str(exc_info.value)


def test_speak_synthesizes_and_plays_audio(tmp_path, monkeypatch):
    speaker_wav = tmp_path / "aldric.wav"
    speaker_wav.write_bytes(b"fake wav data")

    fake_model = _FakeTTSModel()
    monkeypatch.setattr(xtts_module, "_get_tts_model", lambda: fake_model)

    played = {}

    def fake_play(data, samplerate):
        played["data"] = data
        played["samplerate"] = samplerate

    monkeypatch.setattr(xtts_module.sd, "play", fake_play)
    monkeypatch.setattr(xtts_module.sd, "wait", lambda: None)

    client = XTTSClient(language="en")
    client.speak("Good day, traveler.", str(speaker_wav))

    assert fake_model.calls == [("Good day, traveler.", str(speaker_wav), "en")]
    assert played["data"] == [0.0, 0.1, 0.2]
    assert played["samplerate"] == 24000
