"""
Coqui XTTS v2 wrapper: synthesizes NPC replies in a cloned voice and plays them back.
"""
import os

# XTTS v2 is under the CPML (non-commercial); agree up front so the first
# model load doesn't block on an interactive prompt. University course project.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

from typing import Optional

import sounddevice as sd
from TTS.api import TTS

from backend.config.settings import TTS_LANGUAGE, TTS_MODEL

_tts_model: Optional[TTS] = None


def _get_tts_model() -> TTS:
    global _tts_model
    if _tts_model is None:
        _tts_model = TTS(TTS_MODEL)
    return _tts_model


class XTTSClient:
    def __init__(self, language: str = TTS_LANGUAGE):
        self.language = language

    def synthesize(self, text: str, speaker_wav: str):
        """Synthesize `text` in the reference voice; return (waveform, sample_rate)."""
        if not os.path.exists(speaker_wav):
            raise FileNotFoundError(
                f"No reference voice found at '{speaker_wav}'. "
                "Run: python -m scripts.generate_placeholder_voices"
            )

        model = _get_tts_model()
        waveform = model.tts(text=text, speaker_wav=speaker_wav, language=self.language)
        return waveform, model.synthesizer.output_sample_rate

    def speak(self, text: str, speaker_wav: str) -> None:
        waveform, sample_rate = self.synthesize(text, speaker_wav)
        sd.play(waveform, samplerate=sample_rate)
        sd.wait()
