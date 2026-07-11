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
import queue
import re
import threading
import time
import uuid
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

app = FastAPI(title="GAI NPC Dialogue Server")

_handlers: Dict[str, DialogueHandler] = {}
_llm = None
_tts = None
_stt = None
_policy = None
_verbalizer = None
_pipeline_jobs: Dict[str, "_PipelineJob"] = {}
_pipeline_jobs_lock = threading.Lock()
_tts_lock = threading.Lock()

_SENTENCE_RE = re.compile(r"^(.+?[.!?\u3002\uff01\uff1f]+(?:[\"'\u201d\u2019)]*)?)(?:\s+|$)", re.DOTALL)


class _PipelineJob:
    def __init__(self, npc: str):
        self.npc = npc
        self.events = []
        self.done = False
        self.error = ""
        self.created_at = time.time()
        self.lock = threading.Lock()

    def append(self, event_type: str, **payload) -> None:
        with self.lock:
            event = {"seq": len(self.events), "type": event_type}
            event.update(payload)
            self.events.append(event)

    def snapshot(self, after: int) -> Dict[str, Any]:
        with self.lock:
            cursor = max(0, min(after, len(self.events)))
            return {
                "events": list(self.events[cursor:]),
                "next": len(self.events),
                "done": self.done,
                "error": self.error,
            }


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


class ActRequest(BaseModel):
    npc: str
    game_state: Dict[str, Any]
    bark: bool = True
    speak: bool = False


def _extract_sentences(buffer: str, force: bool = False):
    sentences = []
    remainder = buffer
    while remainder:
        match = _SENTENCE_RE.match(remainder)
        if match is None:
            break
        sentence = match.group(1).strip()
        if sentence:
            sentences.append(sentence)
        remainder = remainder[match.end():].lstrip()
    if force and remainder.strip():
        sentences.append(remainder.strip())
        remainder = ""
    return sentences, remainder


def _append_pipeline_sentence(job: _PipelineJob, work_queue, index: int, text: str) -> None:
    job.append("sentence", sentence_index=index, text=text)
    work_queue.put((index, text))


def _run_pipeline_tts(job: _PipelineJob, request: ChatRequest, work_queue) -> None:
    voice_path = os.path.join(VOICES_DIR, f"{_npc_key(request.npc)}.wav")
    while True:
        item = work_queue.get()
        if item is None:
            return
        sentence_index, sentence = item
        if not request.speak:
            continue
        try:
            with _tts_lock:
                waveform, sample_rate = _get_tts().synthesize(sentence, voice_path)
            job.append(
                "audio",
                sentence_index=sentence_index,
                audio_base64=base64.b64encode(
                    waveform_to_wav_bytes(waveform, sample_rate)
                ).decode("ascii"),
                sample_rate=sample_rate,
            )
        except Exception as exc:
            job.append(
                "audio_error",
                sentence_index=sentence_index,
                message=str(exc),
            )


def _run_chat_pipeline(job: _PipelineJob, request: ChatRequest) -> None:
    handler = _get_handler(request.npc)
    work_queue = queue.Queue()
    tts_thread = threading.Thread(
        target=_run_pipeline_tts,
        args=(job, request, work_queue),
        daemon=True,
        name=f"xtts-{_npc_key(request.npc)}",
    )
    tts_thread.start()

    buffer = ""
    sentence_index = 0

    def on_token(chunk: str) -> None:
        nonlocal buffer, sentence_index
        buffer += chunk
        sentences, buffer = _extract_sentences(buffer)
        for sentence in sentences:
            _append_pipeline_sentence(job, work_queue, sentence_index, sentence)
            sentence_index += 1

    try:
        result = handler.respond_stream_with_metadata(
            request.text,
            on_token=on_token,
            game_state=request.game_state,
            policy_mode=request.policy_mode,
        )
        sentences, buffer = _extract_sentences(buffer, force=True)
        for sentence in sentences:
            _append_pipeline_sentence(job, work_queue, sentence_index, sentence)
            sentence_index += 1

        PersonaGenerator(_get_llm()).save(handler.npc, directory=PERSONAS_DIR)
        work_queue.put(None)
        tts_thread.join()

        guard_reason = ""
        if handler.last_guard is not None:
            guard_reason = handler.last_guard.reason
        job.append(
            "done",
            reply=result["reply"],
            guard_reason=guard_reason,
        )
    except Exception as exc:
        work_queue.put(None)
        tts_thread.join()
        job.error = str(exc)
        job.append("error", message=str(exc))
    finally:
        with job.lock:
            job.done = True


def _remove_old_pipeline_jobs(max_age_seconds: float = 1800.0) -> None:
    cutoff = time.time() - max_age_seconds
    with _pipeline_jobs_lock:
        expired = [
            job_id
            for job_id, job in _pipeline_jobs.items()
            if job.done and job.created_at < cutoff
        ]
        for job_id in expired:
            del _pipeline_jobs[job_id]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/tts/warmup")
def warmup_tts(npc: str = "sanji"):
    """Load XTTS in this server process and verify the selected accelerator."""
    import torch

    voice_path = os.path.join(VOICES_DIR, f"{_npc_key(npc)}.wav")
    start = time.perf_counter()
    _waveform, sample_rate = _get_tts().synthesize("Ready.", voice_path)
    elapsed = time.perf_counter() - start
    return {
        "status": "ready",
        "cuda": torch.cuda.is_available(),
        "device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "sample_rate": sample_rate,
        "warmup_seconds": round(elapsed, 2),
    }


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


@app.post("/chat/pipeline")
def start_chat_pipeline(request: ChatRequest):
    """Start Ollama sentence streaming and per-sentence XTTS synthesis."""
    _get_handler(request.npc)
    _remove_old_pipeline_jobs()
    job_id = uuid.uuid4().hex
    job = _PipelineJob(request.npc)
    with _pipeline_jobs_lock:
        _pipeline_jobs[job_id] = job

    threading.Thread(
        target=_run_chat_pipeline,
        args=(job, request),
        daemon=True,
        name=f"chat-{_npc_key(request.npc)}-{job_id[:8]}",
    ).start()
    return {"job_id": job_id, "npc": request.npc}


@app.get("/chat/pipeline/{job_id}")
def poll_chat_pipeline(job_id: str, after: int = 0):
    with _pipeline_jobs_lock:
        job = _pipeline_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired pipeline job.")
    return job.snapshot(after)


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
