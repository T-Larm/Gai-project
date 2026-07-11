# Generative NPC Interaction System

A local-first multimodal generative AI system for RPG and VR NPCs. A trained behavior policy decides **what an NPC does**, while an LLM decides **how the NPC expresses it**. NPCs can act autonomously, talk in character, remember conversations across sessions, update their goals and emotions, and answer with a cloned voice.

Course project for *Generative Artificial Intelligence for Graphics and Multimedia* at Politecnico di Torino.

**Authors:** DENG Lan · ZHAN Xinwei

## Highlights

- Two-channel architecture: low-latency autonomous behavior plus full player dialogue.
- Supervised multi-head MLP for action and mood selection, with heuristic and LLM-policy baselines.
- Three-layer personas generated automatically from small JSON seeds.
- Persistent semantic memory using relevance, recency, and importance.
- Local inference with Ollama, Whisper STT, and Coqui XTTS v2 voice cloning.
- FastAPI bridge and Unity client scripts for behavior, dialogue, streaming speech, movement, and UI.
- Six ready-to-use NPCs: Asuna, Lanyan, Loen, Frederica, Nicole, and Sanji.

## Architecture

```text
Behavior channel
game state (JSON) -> trained MLP policy -> action + mood -> Unity acts immediately
                                      \-> async LLM bark -> optional XTTS speech

Dialogue channel
player text/speech + game state -> trained MLP -> action + mood
                                      \-> persona + memory + dialogue guard
                                           -> Ollama LLM -> reply -> optional XTTS speech
                                                            \-> persisted memory/state
```

The behavior channel does not wait for the LLM before returning an action. Bark generation is asynchronous and has template fallbacks. During player-initiated dialogue, Unity sends the same native game state with `policy_mode=trained`; the resulting `action_id` and `mood` constrain the LLM reply. The dialogue channel supports both a standard response and a sentence-level pipeline so Unity can begin presenting and synthesizing a reply before the complete response is ready.

The behavior evaluation compares the trained policy with a hand-written heuristic and an LLM-as-policy baseline. See [NPC design](docs/npc-design.md), [report facts](docs/report_facts.md), and the [evaluation guide](evaluation/README.md) for details and recorded results.

## Technology stack

| Area | Technology |
| --- | --- |
| Behavior policy | PyTorch multi-head MLP |
| Dialogue and persona generation | Ollama (`llama3:latest`) |
| Memory embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Speech-to-text | OpenAI Whisper (`base`) |
| Text-to-speech | Coqui XTTS v2 |
| API | FastAPI + Uvicorn |
| Client | Unity C# scripts using `UnityWebRequest` |

## Requirements

- Python 3.10 (the dependency set is tested on Windows)
- [Ollama](https://ollama.com/) with `llama3:latest`
- A microphone for voice input (optional)
- An NVIDIA GPU is recommended for XTTS and policy training; text-only dialogue can run without one

## Installation

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull llama3
```

Generate placeholder reference voices if `data/voices/` is empty:

```bash
python -m scripts.generate_placeholder_voices
```

> XTTS v2 downloads roughly 2 GB on first use. Ollama, XTTS, Whisper, and the behavior checkpoint are loaded lazily where possible.

## Quick start

Start Ollama, then launch the backend from the repository root:

```bash
uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

Verify the service:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

### Command-line dialogue

The repository includes six generated personas. Names are case-insensitive when used with `--npc`:

```bash
# Text dialogue
python -m backend.main --npc nicole --text

# Text input with cloned-voice replies
python -m backend.main --npc nicole --text --speak

# Microphone input
python -m backend.main --npc nicole
```

Generate or overwrite personas from the roster seed file:

```bash
python -m scripts.generate_roster_personas --overwrite

# Generate one persona through the interactive backend entry point
python -m backend.main --seed data/seeds/example_seeds.json --name Nicole --text
```

## Active NPC roster

| Persona | Persona file | Unity model |
| --- | --- | --- |
| Asuna | `data/personas/asuna.json` | `assets/asuna/source/q.fbx` |
| Lanyan | `data/personas/lanyan.json` | `assets/Lanyan_Unity/Lanyan.fbx` |
| Loen | `data/personas/loen.json` | `assets/Loen/Loen.fbx` |
| Frederica | `data/personas/frederica.json` | `assets/miyamoto-frederica/source/Breathing Idle.fbx` |
| Nicole | `data/personas/nicole.json` | `assets/Nicole_Unity/Nicole.fbx` |
| Sanji | `data/personas/sanji.json` | `assets/sanji-anime-character/source/crowds_30039.fbx` |

Use the persona name in both `NpcBehaviorClient.npcName` and `NpcDialogueClient.npcName`. See [the roster notes](data/personas/README.md) for Unity object naming details.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness and runtime status |
| `POST` | `/tts/warmup` | Load and warm up XTTS |
| `GET` | `/npc/{name}` | Persona summary and dynamic state |
| `POST` | `/act` | Immediately select an action/mood and optionally start a background bark job |
| `GET` | `/act/bark/{job_id}` | Poll the asynchronous bark and optional audio result |
| `POST` | `/chat` | Complete dialogue response with optional audio |
| `POST` | `/chat/pipeline` | Start sentence-level dialogue/audio generation |
| `GET` | `/chat/pipeline/{job_id}` | Poll sentence and audio events |
| `POST` | `/transcribe` | Transcribe an uploaded WAV file |

Interactive request schemas are available at `/docs` after the server starts.

## Unity integration

Copy `unity/Scripts/` into a Unity project's `Assets/Scripts/` directory. The included components cover:

- periodic game-state submission and autonomous NPC actions;
- action routing and procedural movement;
- proximity-triggered dialogue UI;
- sentence-pipeline polling and WAV playback;
- bark bubbles and scene-state collection.

`NpcDialogueClient` uses `NpcSceneStateProvider` by default and requests the trained policy for every player-initiated turn. Disable `useTrainedPolicy` in the Inspector only when an LLM-only comparison is required.

Editor helpers under `unity/Editor/` automate scene setup and run readiness checks. For a local backend, keep the backend URL at `http://127.0.0.1:8000`. More detail is available in [the Unity integration guide](unity/README.md).

## Behavior policy training and evaluation

```bash
# Convert the Stateful RPG dataset and recompute labels with the oracle rule
python -m evaluation.datasets.convert_stateful_rpg

# Train locally on CPU (CUDA is the default)
python -m evaluation.train_policy --device cpu --allow-cpu --epochs 80 --hidden-dim 256

# Compare trained, heuristic, and LLM policies
python -m evaluation.eval_policies \
  --checkpoint data/behavior_policy/checkpoints2/stateful_rpg_v2_mlp_h512 \
  --llm-model llama3:latest
```

Dialogue-side evaluation examples:

```bash
python -m evaluation.run_dialogues --npc nicole --condition full --suite all
python -m evaluation.judge_consistency evaluation/results/nicole_*.jsonl
python -m evaluation.measure_latency --n 30 --components llm
python -m evaluation.eval_guard
python -m evaluation.eval_barks
```

## Tests

Heavy models are stubbed in the automated tests, so Ollama, XTTS, and a GPU are not required:

```bash
pip install pytest
python -m pytest
```

## Repository layout

```text
backend/       FastAPI, behavior policy, dialogue, persona, memory, STT, and TTS
data/          seeds, personas, voices, datasets, checkpoints, and evaluation output
evaluation/    data conversion, training, baselines, dialogue evaluation, and latency tools
scripts/       persona and placeholder-voice generation
tests/         unit and integration tests
unity/         Unity runtime scripts, editor setup helpers, and integration notes
docs/          architecture, research notes, results, and progress documentation
assets/        Unity-ready character and player assets
```

## Project status

| Area | Status |
| --- | --- |
| Persona generation, dialogue, memory, and dialogue guard | Complete |
| Whisper STT and XTTS voice cloning | Complete |
| Trained behavior policy, bark verbalizer, and evaluation harness | Complete |
| FastAPI and Unity bridge | Complete |
| Six-NPC Unity scene integration | In active development/testing |

## References

1. Park et al., *Generative Agents: Interactive Simulacra of Human Behavior*, UIST 2023.
2. Abdulhai et al., *Consistently Simulating Human Personas with Multi-Turn Reinforcement Learning*, arXiv:2511.00222, 2025.
3. Albayrak, *RPG Dataset (Llama-3)* — Stateful RPG NPC simulation data and generator, [Kaggle](https://www.kaggle.com/datasets/abdusselamalbayrak/rpg-dataset-llama-3/data), Apache-2.0.
