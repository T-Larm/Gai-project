# Generative NPC Interaction System

A multimodal generative AI framework for RPG/VR NPCs with a two-channel architecture: a **trained behavior policy decides what the NPC does** (eat, flee, work, socialize — in 0.6 ms), while the **LLM only decides how it sounds** — one-line in-character barks, and full dialogue when the player speaks. Personas are **automatically constructed from minimal seeds** — no hand-written character prompts — and NPCs **answer in their own cloned voice**, remember past conversations, and evolve their goals and emotions as you talk.

Course project for *Generative Artificial Intelligence for Graphics and Multimedia* (Politecnico di Torino).
**Authors:** DENG Lan · ZHAN Xinwei

## Pipeline

```
 ── Behavior channel (autonomous NPC life) ─────────────────────────────
 game state ─► trained policy (MLP, 0.6 ms) ─► action + mood
 (JSON)              │                            │ NPC acts immediately
                     ▼                            ▼
              LLM bark verbalizer (~1.2 s, async) ─► one in-character line ─► XTTS v2
              (falls back to templates if the LLM stalls — gameplay never blocks)

 ── Dialogue channel (player interaction; action=socialize routes here) ─
 seed (a few lines of JSON) ──offline, 3 LLM calls──► three-layer persona
                                                          │
 player speech ─► Whisper STT ─► llama3 (Ollama) ─► reply ─► XTTS v2 cloned voice
                                     ▲    │
                    memory retrieval ┘    │ memory write / dynamic update every 4 turns
                                          ▼
                              persisted to disk (NPC remembers across sessions)
```

Two research threads: **(1) behavior**: a supervised policy beats both a hand-written heuristic and LLM-as-policy at state-aware action selection (91.0% vs 51.6% vs 16.0% accuracy); **(2) persona scalability**: can automatically generated structured personas match hand-authored ones in consistency and perceived quality? (See `docs/npc-design.md` for the full architecture and `docs/progress-log.md` for the research-question framing.)

## What is ours vs. off-the-shelf

| Component | Off-the-shelf | Built by us |
|---|---|---|
| Behavior policy | PyTorch (framework), Kaggle Stateful-RPG dataset | **leak-free dataset conversion with oracle relabeling, native feature extraction, multi-head MLP + training, heuristic & LLM-as-policy baselines, 3-way evaluation** |
| Bark verbalizer | llama3 via Ollama | **situation summarization, persona-conditioned prompting, robust fallback templates** |
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

# HTTP server for Unity (endpoints: /health, /npc/{name}, /act, /chat, /transcribe)
uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

Behavior policy (train / evaluate):

```bash
# Convert the raw dataset (labels recomputed with the generator's deterministic rule)
python -m evaluation.datasets.convert_stateful_rpg

# Train the MLP (CUDA by default; --allow-cpu for local runs)
python -m evaluation.train_policy --device cpu --allow-cpu --epochs 80 --hidden-dim 256

# Compare trained vs heuristic vs LLM-as-policy on the test split
python -m evaluation.eval_policies --checkpoint data/behavior_policy/checkpoints/stateful_rpg_v2_mlp_h256 --llm-model llama3:latest
```

## Tests & evaluation

```bash
python -m pytest        # 143 tests, no Ollama/XTTS/GPU needed (heavy models stubbed)
```

The evaluation harness (baselines, ablations, LLM-as-judge, latency) lives in `evaluation/` — see [`evaluation/README.md`](evaluation/README.md). Example:

```bash
python -m evaluation.run_dialogues --npc aldric --condition full --suite all
python -m evaluation.judge_consistency evaluation/results/aldric_*.jsonl
python -m evaluation.measure_latency --n 30 --components llm
```

## Repository layout

```
backend/          behavior policy · verbalizer · STT · LLM dialogue · persona · memory · TTS · FastAPI
data/             seeds, personas, voices, converted policy dataset + checkpoints + eval results
evaluation/       dataset conversion, policy training/eval, dialogue runner / judge / latency scripts
scripts/          placeholder voice generation
tests/            pytest suite (143)
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
| B | Behavior policy: dataset v2 + training + 3-way eval + bark verbalizer + `/act` | ✅ |
| 4b | Unity scene + client + lip-sync | 🔧 in progress |
| Eval | Dialogue-side runs + user study | 🔧 harness ready |

## References

1. Park et al., *Generative Agents: Interactive Simulacra of Human Behavior*, UIST 2023.
2. Abdulhai et al., *Consistently Simulating Human Personas with Multi-Turn Reinforcement Learning*, arXiv:2511.00222, 2025.
3. Albayrak, *RPG Dataset (Llama-3)* — Stateful RPG NPC simulation data + generator, [Kaggle](https://www.kaggle.com/datasets/abdusselamalbayrak/rpg-dataset-llama-3/data), Apache-2.0.
