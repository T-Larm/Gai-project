"""FastAPI server: /health, /npc/{name}, /chat (optional speak), /transcribe."""
import base64
import json
import os

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

import backend.llm.persona.memory as memory_module
import backend.server as server_module
from backend.audio_utils import wav_bytes_to_float32, waveform_to_wav_bytes
from backend.llm.persona.generator import PersonaGenerator
from backend.llm.persona.models import (
    CorePersona, DynamicSituation, NPC, PersonaSeed, SocialPersona,
)


class _ConstantEmbedder:
    def encode(self, texts, convert_to_tensor=True):
        if isinstance(texts, str):
            return torch.ones(1)
        return torch.ones(len(texts), 1)


class _FakeLLM:
    def chat(self, messages, system=""):
        return "Hail, traveller."

    def generate(self, prompt, system=""):
        return '{"current_goal": "work", "emotional_state": "neutral"}'


class _FakeTTS:
    def synthesize(self, text, speaker_wav):
        return [0.0, 0.1, -0.1], 24000


class _FakeWhisper:
    def transcribe(self, audio, **kwargs):
        return {"text": " hello there "}


def _write_persona(directory: str) -> None:
    npc = NPC(
        seed=PersonaSeed(name="Aldric", occupation="Blacksmith",
                         personality_tags=["gruff"], relationships={}),
        core=CorePersona(name="Aldric", occupation="Blacksmith", backstory="Smith.",
                         values=["honesty"], speech_style="gruff",
                         knowledge_domains=["smithing"]),
        social=SocialPersona(relationships={}, faction="Town", reputation="solid"),
        dynamic=DynamicSituation(current_goal="Sell swords", emotional_state="neutral"),
    )
    PersonaGenerator(llm=None).save(npc, directory=directory)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_module, "_get_embedder", lambda: _ConstantEmbedder())
    monkeypatch.setattr(server_module, "PERSONAS_DIR", str(tmp_path))
    monkeypatch.setattr(server_module, "VOICES_DIR", str(tmp_path))
    monkeypatch.setattr(server_module, "_get_llm", lambda: _FakeLLM())
    monkeypatch.setattr(server_module, "_get_tts", lambda: _FakeTTS())
    monkeypatch.setattr(server_module, "_get_stt", lambda: _FakeWhisper())
    server_module._handlers.clear()
    _write_persona(str(tmp_path))
    (tmp_path / "aldric.wav").write_bytes(b"fake reference wav")
    return TestClient(server_module.app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_npc_info_returns_persona_summary(client):
    response = client.get("/npc/Aldric")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Aldric"
    assert body["occupation"] == "Blacksmith"
    assert body["current_goal"] == "Sell swords"
    assert body["emotional_state"] == "neutral"


def test_npc_info_404_for_unknown_npc(client):
    assert client.get("/npc/nobody").status_code == 404


def test_chat_returns_reply_and_persists_memory(client, tmp_path):
    response = client.post("/chat", json={"npc": "Aldric", "text": "Hello!"})

    assert response.status_code == 200
    body = response.json()
    assert body["npc"] == "Aldric"
    assert body["reply"] == "Hail, traveller."
    assert "audio_base64" not in body

    saved = json.loads((tmp_path / "aldric.json").read_text(encoding="utf-8"))
    assert len(saved["memory_log"]) == 2  # player line + NPC reply


def test_chat_with_speak_returns_playable_wav(client):
    response = client.post(
        "/chat", json={"npc": "Aldric", "text": "Hello!", "speak": True}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sample_rate"] == 24000
    waveform, sample_rate = wav_bytes_to_float32(base64.b64decode(body["audio_base64"]))
    assert sample_rate == 24000
    assert len(waveform) == 3


def test_chat_404_for_unknown_npc(client):
    response = client.post("/chat", json={"npc": "nobody", "text": "Hi"})
    assert response.status_code == 404


def test_transcribe_accepts_wav_upload(client):
    wav = waveform_to_wav_bytes(np.zeros(1600, dtype=np.float32), sample_rate=16000)

    response = client.post(
        "/transcribe", files={"file": ("input.wav", wav, "audio/wav")}
    )

    assert response.status_code == 200
    assert response.json() == {"text": "hello there"}
