import pytest

import backend.tts.xtts_client as xtts_module
from backend.tts.xtts_client import XTTSClient


class _FakeXtts:
    """Stands in for the low-level Xtts model behind synthesizer.tts_model."""

    def __init__(self):
        self.latent_calls = []
        self.inference_calls = []

    def get_conditioning_latents(self, audio_path):
        self.latent_calls.append(list(audio_path))
        return ("gpt_latent", "speaker_embedding")

    def inference(self, text, language, gpt_cond_latent, speaker_embedding, **kwargs):
        self.inference_calls.append(
            (text, language, gpt_cond_latent, speaker_embedding, kwargs)
        )
        return {"wav": [0.0, 0.1, 0.2]}


class _FakeSynthesizer:
    output_sample_rate = 24000

    def __init__(self):
        self.tts_model = _FakeXtts()


class _FakeTTSModel:
    def __init__(self):
        self.synthesizer = _FakeSynthesizer()


class _RecordingTTSClass:
    """Stands in for TTS.api.TTS; records the constructor args it was called with."""

    last_kwargs = None

    def __new__(cls, model_name, **kwargs):
        cls.last_kwargs = kwargs
        return _FakeTTSModel()


@pytest.fixture(autouse=True)
def clean_latent_cache(monkeypatch):
    monkeypatch.setattr(xtts_module, "_latent_cache", {})


@pytest.mark.parametrize("cuda_available", [True, False])
def test_get_tts_model_passes_gpu_flag_from_cuda_availability(monkeypatch, cuda_available):
    monkeypatch.setattr(xtts_module, "_tts_model", None)
    monkeypatch.setattr(xtts_module, "TTS", _RecordingTTSClass)
    monkeypatch.setattr(xtts_module.torch.cuda, "is_available", lambda: cuda_available)

    xtts_module._get_tts_model()

    assert _RecordingTTSClass.last_kwargs == {"gpu": cuda_available}


def test_speak_raises_when_speaker_wav_missing(tmp_path):
    client = XTTSClient()
    missing_path = str(tmp_path / "nope.wav")

    with pytest.raises(FileNotFoundError) as exc_info:
        client.speak("hello", missing_path)

    assert "nope.wav" in str(exc_info.value)


def test_synthesize_returns_waveform_and_sample_rate_without_playing(tmp_path, monkeypatch):
    speaker_wav = tmp_path / "aldric.wav"
    speaker_wav.write_bytes(b"fake wav data")

    fake_model = _FakeTTSModel()
    monkeypatch.setattr(xtts_module, "_get_tts_model", lambda: fake_model)

    def _fail_play(*args, **kwargs):
        raise AssertionError("synthesize() must not play audio")

    monkeypatch.setattr(xtts_module.sd, "play", _fail_play)

    client = XTTSClient(language="en")
    waveform, sample_rate = client.synthesize("Good day.", str(speaker_wav))

    assert waveform == [0.0, 0.1, 0.2]
    assert sample_rate == 24000
    xtts = fake_model.synthesizer.tts_model
    assert len(xtts.inference_calls) == 1
    text, language, gpt_cond_latent, speaker_embedding, _ = xtts.inference_calls[0]
    assert (text, language) == ("Good day.", "en")
    assert (gpt_cond_latent, speaker_embedding) == ("gpt_latent", "speaker_embedding")


def test_speaker_latents_computed_once_per_voice(tmp_path, monkeypatch):
    speaker_wav = tmp_path / "aldric.wav"
    speaker_wav.write_bytes(b"fake wav data")

    fake_model = _FakeTTSModel()
    monkeypatch.setattr(xtts_module, "_get_tts_model", lambda: fake_model)

    client = XTTSClient(language="en")
    client.synthesize("First line.", str(speaker_wav))
    client.synthesize("Second line.", str(speaker_wav))

    xtts = fake_model.synthesizer.tts_model
    assert xtts.latent_calls == [[str(speaker_wav)]]
    assert len(xtts.inference_calls) == 2


def test_speaker_latents_cached_per_distinct_voice(tmp_path, monkeypatch):
    wav_a = tmp_path / "aldric.wav"
    wav_b = tmp_path / "asuna.wav"
    wav_a.write_bytes(b"fake wav data")
    wav_b.write_bytes(b"fake wav data")

    fake_model = _FakeTTSModel()
    monkeypatch.setattr(xtts_module, "_get_tts_model", lambda: fake_model)

    client = XTTSClient(language="en")
    client.synthesize("Hello.", str(wav_a))
    client.synthesize("Hello.", str(wav_b))
    client.synthesize("Again.", str(wav_a))

    xtts = fake_model.synthesizer.tts_model
    assert xtts.latent_calls == [[str(wav_a)], [str(wav_b)]]


def test_synthesize_raises_when_speaker_wav_missing(tmp_path):
    client = XTTSClient()

    with pytest.raises(FileNotFoundError):
        client.synthesize("hello", str(tmp_path / "missing.wav"))


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

    assert played["data"] == [0.0, 0.1, 0.2]
    assert played["samplerate"] == 24000
