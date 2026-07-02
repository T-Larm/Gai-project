"""WAV encode/decode helpers (16-bit PCM mono), used by the HTTP server."""
import io
import wave

import numpy as np


def waveform_to_wav_bytes(waveform, sample_rate: int) -> bytes:
    """Encode a float32 waveform in [-1, 1] as mono 16-bit PCM WAV bytes."""
    samples = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def wav_bytes_to_float32(data: bytes):
    """Decode mono 16-bit PCM WAV bytes to (float32 waveform in [-1, 1], sample_rate)."""
    with wave.open(io.BytesIO(data), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit PCM WAV, got {wav.getsampwidth() * 8}-bit")
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
        pcm = np.frombuffer(frames, dtype=np.int16)
        if wav.getnchannels() > 1:
            pcm = pcm.reshape(-1, wav.getnchannels()).mean(axis=1).astype(np.int16)
    return pcm.astype(np.float32) / 32768.0, sample_rate
