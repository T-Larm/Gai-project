"""
FastAPI bridge for Unity (Phase 4).

Endpoints:
    GET  /health          — liveness check
    GET  /npc/{name}      — persona summary + current dynamic state
    POST /chat            — {npc, text, speak?} -> {npc, reply[, audio_base64, sample_rate]}
    POST /transcribe      — WAV upload (16 kHz mono preferred) -> {text}

Run from the project root:
    uvicorn backend.server:app --host 127.0.0.1 --port 8000

Heavy models (Whisper, XTTS) are lazy-loaded on first use, so /chat in
text-only mode stays light. NPC state (memory + dynamic layer) is persisted
back to data/personas/ after every turn.
"""
import base64
import os
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from backend.audio_utils import wav_bytes_to_float32, waveform_to_wav_bytes
from backend.config.settings import PERSONAS_DIR, VOICES_DIR, WHISPER_MODEL
from backend.llm.dialogue import DialogueHandler
from backend.llm.ollama_client import OllamaClient
from backend.llm.persona.generator import PersonaGenerator

app = FastAPI(title="GAI NPC Dialogue Server")

_handlers: Dict[str, DialogueHandler] = {}
_llm = None
_tts = None
_stt = None


def _get_llm() -> OllamaClient:
    global _llm
    if _llm is None:
        _llm = OllamaClient()
    return _llm


def _get_tts():
    global _tts
    if _tts is None:
        from backend.tts.xtts_client import XTTSClient
        _tts = XTTSClient()
    return _tts


def _get_stt():
    global _stt
    if _stt is None:
        import whisper
        _stt = whisper.load_model(WHISPER_MODEL)
    return _stt


def _npc_key(name: str) -> str:
    return name.lower().replace(" ", "_")


def _get_handler(npc_name: str) -> DialogueHandler:
    key = _npc_key(npc_name)
    if key not in _handlers:
        path = os.path.join(PERSONAS_DIR, f"{key}.json")
        if not os.path.exists(path):
            raise HTTPException(
                status_code=404,
                detail=f"No persona named '{npc_name}'. Generate it first via the CLI.",
            )
        npc = PersonaGenerator.load(path)
        _handlers[key] = DialogueHandler(_get_llm(), npc)
    return _handlers[key]


class ChatRequest(BaseModel):
    npc: str
    text: str
    speak: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/npc/{name}")
def npc_info(name: str):
    handler = _get_handler(name)
    npc = handler.npc
    return {
        "name": npc.core.name,
        "occupation": npc.core.occupation,
        "speech_style": npc.core.speech_style,
        "faction": npc.social.faction,
        "current_goal": npc.dynamic.current_goal,
        "emotional_state": npc.dynamic.emotional_state,
        "memory_entries": len(handler.memory.entries),
    }


@app.post("/chat")
def chat(request: ChatRequest):
    handler = _get_handler(request.npc)
    reply = handler.respond(request.text)
    PersonaGenerator(_get_llm()).save(handler.npc, directory=PERSONAS_DIR)

    response = {"npc": handler.npc.core.name, "reply": reply}
    if request.speak:
        voice_path = os.path.join(VOICES_DIR, f"{_npc_key(request.npc)}.wav")
        waveform, sample_rate = _get_tts().synthesize(reply, voice_path)
        response["audio_base64"] = base64.b64encode(
            waveform_to_wav_bytes(waveform, sample_rate)
        ).decode("ascii")
        response["sample_rate"] = sample_rate
    return response


@app.post("/transcribe")
async def transcribe(file: UploadFile):
    data = await file.read()
    try:
        waveform, sample_rate = wav_bytes_to_float32(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid WAV upload: {exc}")

    if sample_rate != 16000:
        # Whisper expects 16 kHz input; linear resample is fine for speech
        target_length = int(len(waveform) * 16000 / sample_rate)
        waveform = np.interp(
            np.linspace(0.0, len(waveform), target_length, endpoint=False),
            np.arange(len(waveform)),
            waveform,
        ).astype(np.float32)

    result = _get_stt().transcribe(waveform, fp16=False, language="en")
    return {"text": result["text"].strip()}
