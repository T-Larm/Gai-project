# Generative NPC Interaction System

A multimodal generative AI framework for RPG/VR NPCs: players **speak naturally** to NPCs whose personalities are **automatically constructed from minimal seeds** — no hand-written character prompts — and who **answer back in their own cloned voice**, remember past conversations, and evolve their goals and emotions as you talk.

Course project for *Generative Artificial Intelligence for Graphics and Multimedia* (Politecnico di Torino).
**Authors:** DENG Lan · ZHAN Xinwei

## Pipeline

```
 seed (a few lines of JSON)
        │  offline: 3 LLM calls
        ▼
 three-layer persona ────────────────────────────────┐
 (core / social / dynamic)                           │
                                                     ▼
 player speech ─► Whisper STT ─► llama3 (Ollama) ─► reply text ─► XTTS v2 ─► cloned voice
                                     ▲    │                        (zero-shot)
                    memory retrieval ┘    │ memory write /
                    (top-5, 3-factor)     │ dynamic-layer update every 4 turns
                                          ▼
                              persisted to disk (NPC remembers across sessions)
```

The research focus is **persona scalability**: can automatically generated structured personas match hand-authored ones in consistency and perceived quality? (See `docs/progress-log.md` for the full research-question framing, RQ1–RQ5.)

## What is ours vs. off-the-shelf

| Component | Off-the-shelf | Built by us |
|---|---|---|
| STT | Whisper (base) | recording + resampling integration |
| LLM inference | llama3 via Ollama (fully local) | **three-layer persona schema, seed→persona auto-generation, prompt assembly, dynamic state updates** |
| Memory | sentence-transformers (embeddings only) | **3-factor retrieval (semantic·recency·importance), embedding cache, cross-session persistence** |
| TTS | Coqui XTTS v2 (zero-shot cloning) | voice-identity convention, synthesis/streaming wrapper |
| Server | FastAPI (framework) | all endpoints (Unity bridge) |

## Setup

Requirements: Python 3.10, [Ollama](https://ollama.com) with `llama3:latest` pulled, a microphone (optional), Windows-tested.

```bash
pip install -r requirements.txt          # torch pinned to 2.8.0 — see comments inside
ollama pull llama3
python -m scripts.generate_placeholder_voices   # placeholder reference voices
```

## Usage

```bash
# Generate a persona from a seed (once per NPC)
python -m backend.main --seed data/seeds/example_seeds.json --name Aldric

# Talk — text mode
python -m backend.main --npc aldric --text

# Talk — with spoken replies (XTTS v2; first run downloads ~2 GB)
python -m backend.main --npc aldric --text --speak

# Talk — microphone input
python -m backend.main --npc aldric

# HTTP server for Unity (endpoints: /health, /npc/{name}, /chat, /transcribe)
uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

## Tests & evaluation

```bash
python -m pytest        # 69 tests, no Ollama/XTTS needed (heavy models stubbed)
```

The evaluation harness (baselines, ablations, LLM-as-judge, latency) lives in `evaluation/` — see [`evaluation/README.md`](evaluation/README.md). Example:

```bash
python -m evaluation.run_dialogues --npc aldric --condition full --suite all
python -m evaluation.judge_consistency evaluation/results/aldric_*.jsonl
python -m evaluation.measure_latency --n 30 --components llm
```

## Repository layout

```
backend/          STT · LLM dialogue · persona generation · memory · TTS · FastAPI server
data/             seeds, generated personas, reference voices
evaluation/       experiment conditions, datasets, runner / judge / latency scripts
scripts/          placeholder voice generation
tests/            pytest suite
unity/            Unity client (in progress)
docs/             design & progress documentation (Chinese)
```

Design deep-dive: [`docs/npc-design.md`](docs/npc-design.md) · Progress log: [`docs/progress-log.md`](docs/progress-log.md)

## Status

| Phase | Scope | Status |
|---|---|---|
| 1 | STT + LLM dialogue + auto persona generation | ✅ |
| 2 | Semantic memory retrieval (3-factor) | ✅ |
| 3 | TTS voice cloning (XTTS v2) | ✅ |
| 4a | FastAPI bridge for Unity | ✅ |
| 4b | Unity scene + client + lip-sync | 🔧 in progress |
| Eval | Full runs (5 conditions) + user study | 🔧 harness ready |

## References

1. Park et al., *Generative Agents: Interactive Simulacra of Human Behavior*, UIST 2023.
2. Abdulhai et al., *Consistently Simulating Human Personas with Multi-Turn Reinforcement Learning*, arXiv:2511.00222, 2025.
