"""One-off: generate placeholder reference voices for each NPC seed.

Uses pyttsx3 (offline Windows SAPI voice) — no network access, no model
download. Replace the files in data/voices/ with real recorded or sourced
clips later; no code elsewhere needs to change.

Usage (run from the project root, as a module so `backend` is importable):
    python -m scripts.generate_placeholder_voices [--seeds data/seeds/example_seeds.json]
"""
import argparse
import json
import os
from pathlib import Path

from backend.config.settings import SEEDS_DIR, VOICES_DIR


def voice_output_path(npc_name: str, voices_dir: str = VOICES_DIR) -> str:
    filename = npc_name.lower().replace(" ", "_") + ".wav"
    return os.path.join(voices_dir, filename)


def voice_line_for(npc_name: str, occupation: str) -> str:
    return f"Hello, I am {npc_name}, the {occupation.lower()}."


def generate_all(seeds_path: str, voices_dir: str = VOICES_DIR) -> list:
    import pyttsx3

    with open(seeds_path, encoding="utf-8") as f:
        seeds = json.load(f)

    Path(voices_dir).mkdir(parents=True, exist_ok=True)
    engine = pyttsx3.init()
    written = []
    for seed in seeds:
        line = voice_line_for(seed["name"], seed["occupation"])
        path = voice_output_path(seed["name"], voices_dir)
        engine.save_to_file(line, path)
        written.append(path)
    engine.runAndWait()
    return written


def main():
    parser = argparse.ArgumentParser(description="Generate placeholder NPC reference voices")
    parser.add_argument("--seeds", default=os.path.join(SEEDS_DIR, "example_seeds.json"))
    args = parser.parse_args()

    written = generate_all(args.seeds)
    for path in written:
        print(f"[Voices] Wrote {path}")


if __name__ == "__main__":
    main()
