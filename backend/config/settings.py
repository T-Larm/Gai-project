WHISPER_MODEL = "base"       # tiny / base / small / medium / large
OLLAMA_MODEL  = "llama3:latest"  # available: llama3:latest, gemma3:4b

SAMPLE_RATE      = 16000
RECORD_DURATION  = 5.0      # seconds per voice input

MEMORY_MAX_SIZE  = 200      # max entries in a single NPC's memory stream
MEMORY_TOP_K     = 5        # how many memories to inject into each prompt

EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

MEMORY_RECENCY_HALFLIFE_SEC = 300   # recency score halves every 5 minutes

IMPORTANCE_WEIGHTS = {
    "semantic":   0.4,
    "recency":    0.4,
    "importance": 0.2,
}

DATA_DIR     = "data"
PERSONAS_DIR = "data/personas"
SEEDS_DIR    = "data/seeds"
