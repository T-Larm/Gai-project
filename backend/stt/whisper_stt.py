import numpy as np
import whisper
import sounddevice as sd

from backend.config.settings import WHISPER_MODEL, SAMPLE_RATE, RECORD_DURATION


class WhisperSTT:
    def __init__(self, model_size: str = WHISPER_MODEL):
        print(f"[STT] Loading Whisper model '{model_size}'...")
        self.model = whisper.load_model(model_size)
        self.sample_rate = SAMPLE_RATE

    def transcribe(self, audio: np.ndarray) -> str:
        result = self.model.transcribe(audio, fp16=False, language="en")
        return result["text"].strip()

    def record_and_transcribe(self, duration: float = RECORD_DURATION) -> str:
        print(f"[STT] Recording {duration}s — speak now...")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        print("[STT] Processing...")
        return self.transcribe(audio.flatten())
