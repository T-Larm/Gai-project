"""
Coqui XTTS v2 wrapper: synthesizes NPC replies in a cloned voice and plays them back.
"""
import os

# XTTS v2 is under the CPML (non-commercial); agree up front so the first
# model load doesn't block on an interactive prompt. University course project.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

from typing import Dict, Optional, Tuple

import sounddevice as sd
import torch
from TTS.api import TTS

from backend.config.settings import TTS_LANGUAGE, TTS_MODEL

_tts_model: Optional[TTS] = None

# Speaker conditioning latents per reference-wav path. Computing them costs
# 1-2 s per call and the reference voice never changes at runtime.
_latent_cache: Dict[str, Tuple] = {}


def _get_tts_model() -> TTS:
    # TTS() defaults to gpu=False; without this, synthesis silently runs on
    # CPU (several seconds per line) even on a machine with a CUDA GPU.
    global _tts_model
    if _tts_model is None:
        _tts_model = TTS(TTS_MODEL, gpu=torch.cuda.is_available())
    return _tts_model


def _get_speaker_latents(model: TTS, speaker_wav: str) -> Tuple:
    if speaker_wav not in _latent_cache:
        xtts = model.synthesizer.tts_model
        _latent_cache[speaker_wav] = xtts.get_conditioning_latents(
            audio_path=[speaker_wav]
        )
    return _latent_cache[speaker_wav]


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
        gpt_cond_latent, speaker_embedding = _get_speaker_latents(model, speaker_wav)
        out = model.synthesizer.tts_model.inference(
            text,
            self.language,
            gpt_cond_latent,
            speaker_embedding,
            enable_text_splitting=True,
        )
        return out["wav"], model.synthesizer.output_sample_rate

    def speak(self, text: str, speaker_wav: str) -> None:
        waveform, sample_rate = self.synthesize(text, speaker_wav)
        sd.play(waveform, samplerate=sample_rate)
        sd.wait()
