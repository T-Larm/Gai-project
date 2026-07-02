"""WAV encode/decode helpers: float32 waveform <-> 16-bit PCM WAV bytes."""
import numpy as np

from backend.audio_utils import wav_bytes_to_float32, waveform_to_wav_bytes


def test_waveform_round_trips_through_wav_bytes():
    waveform = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)

    data = waveform_to_wav_bytes(waveform, sample_rate=16000)
    decoded, sample_rate = wav_bytes_to_float32(data)

    assert sample_rate == 16000
    assert decoded.dtype == np.float32
    np.testing.assert_allclose(decoded, waveform, atol=1e-3)


def test_waveform_to_wav_bytes_accepts_plain_python_list():
    data = waveform_to_wav_bytes([0.0, 0.1, -0.1], sample_rate=24000)

    decoded, sample_rate = wav_bytes_to_float32(data)

    assert sample_rate == 24000
    assert len(decoded) == 3


def test_waveform_to_wav_bytes_clips_out_of_range_samples():
    data = waveform_to_wav_bytes([2.0, -2.0], sample_rate=16000)

    decoded, _ = wav_bytes_to_float32(data)

    assert decoded.max() <= 1.0
    assert decoded.min() >= -1.0
