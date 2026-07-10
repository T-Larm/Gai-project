from pathlib import Path

WHISPER_MODEL = "base"       # tiny / base / small / medium / large
OLLAMA_MODEL  = "llama3:latest"  # available: llama3:latest, gemma3:4b
OLLAMA_NUM_GPU_LAYERS = 20  # llama3 8B has 33 layers; capping the GPU share frees
                            # ~2 GB VRAM so XTTS stays GPU-resident while Unity runs
                            # (full-GPU llama3 + XTTS + Unity oversubscribes the 8 GB
                            # card and TTS degrades from seconds to minutes).
                            # -1 = let Ollama fit as many layers as it can.

SAMPLE_RATE      = 16000
RECORD_DURATION  = 5.0      # seconds per voice input

MEMORY_MAX_SIZE  = 200      # max entries in a single NPC's memory stream
MEMORY_TOP_K     = 5        # how many memories to inject into each prompt

HISTORY_MAX_MESSAGES = 16   # sliding window: max messages (8 exchanges) sent to the LLM;
                            # older turns are only reachable via memory retrieval
REPLY_MAX_SENTENCES = 4     # hard cap on reply length; the LLM often ignores the
                            # "2-4 sentences" prompt rule and TTS time scales with text
DYNAMIC_UPDATE_EVERY = 4    # re-evaluate goal/emotional_state every N player turns
SHORT_TERM_MEMORY_SIZE = 5  # recent statements mirrored into npc.dynamic

EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

MEMORY_RECENCY_HALFLIFE_SEC = 300   # recency score halves every 5 minutes

IMPORTANCE_WEIGHTS = {
    "semantic":   0.4,
    "recency":    0.4,
    "importance": 0.2,
}

TTS_MODEL     = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_LANGUAGE  = "en"

# Anchor data paths at the project root so the app works from any CWD
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR     = str(_PROJECT_ROOT / "data")
PERSONAS_DIR = str(_PROJECT_ROOT / "data" / "personas")
SEEDS_DIR    = str(_PROJECT_ROOT / "data" / "seeds")
VOICES_DIR   = str(_PROJECT_ROOT / "data" / "voices")

# Trained behavior policy served by POST /act
POLICY_CHECKPOINT_DIR = str(
    _PROJECT_ROOT / "data" / "behavior_policy" / "checkpoints" / "stateful_rpg_v2_mlp_h256"
)
