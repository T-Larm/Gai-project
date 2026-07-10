"""
FastAPI bridge for Unity (Phase 4).

Endpoints:
    GET  /health          — liveness check
    GET  /npc/{name}      — persona summary + current dynamic state
    POST /chat            — {npc, text, speak?} -> {npc, reply[, audio_base64, sample_rate]}
    POST /act             — {npc, game_state, bark?, speak?} -> policy action + in-character bark
    POST /transcribe      — WAV upload (16 kHz mono preferred) -> {text}

Run from the project root:
    uvicorn backend.server:app --host 127.0.0.1 --port 8000

Heavy models (Whisper, XTTS, policy checkpoint) are lazy-loaded on first use,
so /chat in text-only mode stays light. NPC state (memory + dynamic layer) is
persisted back to data/personas/ after every turn.
"""
import base64
import os
import time
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from backend.audio_utils import wav_bytes_to_float32, waveform_to_wav_bytes
from backend.config.settings import (
    PERSONAS_DIR,
    POLICY_CHECKPOINT_DIR,
    VOICES_DIR,
    WHISPER_MODEL,
)
from backend.llm.dialogue import DialogueHandler
from backend.llm.ollama_client import OllamaClient
from backend.llm.persona.generator import PersonaGenerator
from backend.streaming import StreamSessionManager

app = FastAPI(title="GAI NPC Dialogue Server")

_handlers: Dict[str, DialogueHandler] = {}
_llm = None
_tts = None
_stt = None
_policy = None
_verbalizer = None
_stream_sessions = StreamSessionManager()


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
        # CPU on purpose: the base model transcribes fast enough there, and
        # VRAM is the scarce resource (llama3 + XTTS + Unity share 8 GB).
        _stt = whisper.load_model(WHISPER_MODEL, device="cpu")
    return _stt


def _load_policy():
    global _policy
    if _policy is None:
        from pathlib import Path

        from backend.behavior.supervised_policy import SupervisedPolicy

        checkpoint = Path(POLICY_CHECKPOINT_DIR)
        if not (checkpoint / "metadata.json").exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"No trained policy checkpoint at {checkpoint}. "
                    "Train one with: python -m evaluation.train_policy"
                ),
            )
        _policy = SupervisedPolicy(checkpoint)
    return _policy


def _get_policy():
    return _load_policy()


def _get_verbalizer():
    global _verbalizer
    if _verbalizer is None:
        from backend.behavior.verbalizer import BarkVerbalizer

        _verbalizer = BarkVerbalizer(_get_llm())
    return _verbalizer


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
        from backend.behavior.dialogue_guard import DialogueGuard, secret_topics_from_text

        secret = (npc.seed.extra or {}).get("secret", "")
        guard = DialogueGuard(secret_topics=secret_topics_from_text(secret))
        _handlers[key] = DialogueHandler(_get_llm(), npc, guard=guard)
    return _handlers[key]


class ChatRequest(BaseModel):
    npc: str
    text: str
    speak: bool = False
    game_state: Optional[Dict[str, Any]] = None
    policy_mode: str = "llm_only"


class ChatStreamRequest(BaseModel):
    npc: str
    text: str
    speak: bool = True


class ActRequest(BaseModel):
    npc: str
    game_state: Dict[str, Any]
    bark: bool = True
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
        "personality_tags": npc.seed.personality_tags,
        "current_goal": npc.dynamic.current_goal,
        "emotional_state": npc.dynamic.emotional_state,
        "memory_entries": len(handler.memory.entries),
    }


@app.post("/chat")
def chat(request: ChatRequest):
    handler = _get_handler(request.npc)
    try:
        result = handler.respond_with_metadata(
            request.text,
            game_state=request.game_state,
            policy_mode=request.policy_mode,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    reply = result["reply"]
    PersonaGenerator(_get_llm()).save(handler.npc, directory=PERSONAS_DIR)

    response = {"npc": handler.npc.core.name, "reply": reply}
    if handler.last_guard is not None:
        response["guard"] = {"reason": handler.last_guard.reason}
    if request.speak:
        voice_path = os.path.join(VOICES_DIR, f"{_npc_key(request.npc)}.wav")
        waveform, sample_rate = _get_tts().synthesize(reply, voice_path)
        response["audio_base64"] = base64.b64encode(
            waveform_to_wav_bytes(waveform, sample_rate)
        ).decode("ascii")
        response["sample_rate"] = sample_rate
    return response


@app.post("/chat_stream")
def chat_stream(request: ChatStreamRequest):
    """Sentence-streamed variant of /chat (llm_only mode).

    Returns a session id immediately; a worker thread streams LLM tokens,
    splits them into sentences and synthesizes each one. Unity polls
    GET /chat_stream/{id}. Blocking /chat is unchanged.
    """
    handler = _get_handler(request.npc)   # 404 before the worker starts
    voice_path = os.path.join(VOICES_DIR, f"{_npc_key(request.npc)}.wav")
    speak = request.speak
    text = request.text

    def worker(session):
        started = time.perf_counter()
        for index, sentence in enumerate(handler.respond_stream(text)):
            chunk: Dict[str, Any] = {"index": index, "text": sentence}
            if speak:
                waveform, sample_rate = _get_tts().synthesize(sentence, voice_path)
                chunk["audio_base64"] = base64.b64encode(
                    waveform_to_wav_bytes(waveform, sample_rate)
                ).decode("ascii")
                chunk["sample_rate"] = sample_rate
            chunk["t_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
            with session.lock:
                if index == 0 and handler.last_guard is not None:
                    session.meta["guard"] = {"reason": handler.last_guard.reason}
                session.chunks.append(chunk)
        PersonaGenerator(_get_llm()).save(handler.npc, directory=PERSONAS_DIR)

    session_id = _stream_sessions.start(worker)
    return {"npc": handler.npc.core.name, "session_id": session_id}


@app.get("/chat_stream/{session_id}")
def chat_stream_poll(session_id: str, after: int = -1):
    state = _stream_sessions.poll(session_id, after)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown or expired stream session")
    return state


@app.post("/act")
def act(request: ActRequest):
    """Policy picks the NPC's next action; the LLM only verbalizes it (方案 B)."""
    handler = _get_handler(request.npc)
    npc = handler.npc

    start = time.perf_counter()
    prediction = _get_policy().predict(request.game_state)
    policy_ms = (time.perf_counter() - start) * 1000.0

    action_id = prediction.get("action_id", "")
    response: Dict[str, Any] = {
        "npc": npc.core.name,
        "action_id": action_id,
        "mood": prediction.get("mood"),
        # socialize (or the player speaking) should route to the full
        # dialogue system via POST /chat
        "should_talk": action_id == "socialize",
        "latency_ms": {"policy": round(policy_ms, 2)},
    }

    if request.bark:
        persona = {
            "name": npc.core.name,
            "occupation": npc.core.occupation,
            "speech_style": npc.core.speech_style,
            "traits": npc.seed.personality_tags,
        }
        start = time.perf_counter()
        line = _get_verbalizer().bark(persona, request.game_state, action_id)
        response["bark"] = line
        response["latency_ms"]["bark"] = round((time.perf_counter() - start) * 1000.0, 2)

        if request.speak:
            voice_path = os.path.join(VOICES_DIR, f"{_npc_key(request.npc)}.wav")
            waveform, sample_rate = _get_tts().synthesize(line, voice_path)
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
