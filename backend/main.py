"""
Phase 1 CLI entry point.
Usage:
    python -m backend.main --npc aldric          # load saved persona
    python -m backend.main --seed data/seeds/example_seeds.json --name Aldric  # generate + save
    python -m backend.main --npc aldric --text   # text mode (no microphone)
"""
import argparse
import json
import os
import sys

from backend.config.settings import PERSONAS_DIR, SEEDS_DIR
from backend.llm.dialogue import DialogueHandler
from backend.llm.ollama_client import OllamaClient
from backend.llm.persona.generator import PersonaGenerator
from backend.llm.persona.models import PersonaSeed


def load_or_generate_npc(args, llm: OllamaClient):
    gen = PersonaGenerator(llm)

    if args.npc:
        path = os.path.join(PERSONAS_DIR, f"{args.npc.lower()}.json")
        if not os.path.exists(path):
            sys.exit(f"[Error] Persona file not found: {path}\nRun with --seed first.")
        print(f"[Main] Loading persona from {path}")
        return gen.load(path)

    if args.seed and args.name:
        with open(args.seed, encoding="utf-8") as f:
            seeds_data = json.load(f)
        match = next((s for s in seeds_data if s["name"].lower() == args.name.lower()), None)
        if not match:
            sys.exit(f"[Error] No seed named '{args.name}' in {args.seed}")
        seed = PersonaSeed(**match)
        npc = gen.generate(seed)
        gen.save(npc)
        return npc

    sys.exit("[Error] Provide either --npc <name> or --seed <file> --name <name>")


def run_cli(handler: DialogueHandler, text_mode: bool) -> None:
    npc_name = handler.npc.core.name
    print(f"\n{'='*50}")
    print(f"  Talking to: {npc_name}  ({handler.npc.core.occupation})")
    print(f"  Type 'quit' to exit | 'reset' to restart conversation")
    if not text_mode:
        print(f"  Press Enter to start recording each turn")
    print(f"{'='*50}\n")

    if text_mode:
        _text_loop(handler)
    else:
        _voice_loop(handler)


def _text_loop(handler: DialogueHandler) -> None:
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Goodbye]")
            break

        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            handler.reset()
            print("[Conversation reset]\n")
            continue
        if not user_input:
            continue

        reply = handler.respond(user_input)
        print(f"\n{handler.npc.core.name}: {reply}\n")


def _voice_loop(handler: DialogueHandler) -> None:
    from backend.stt.whisper_stt import WhisperSTT
    stt = WhisperSTT()

    while True:
        try:
            cmd = input("\nPress Enter to speak (or type 'quit'/'reset'): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[Goodbye]")
            break

        if cmd == "quit":
            break
        if cmd == "reset":
            handler.reset()
            print("[Conversation reset]")
            continue

        user_input = stt.record_and_transcribe()
        if not user_input:
            print("[STT] No speech detected, try again.")
            continue

        print(f"You (transcribed): {user_input}")
        reply = handler.respond(user_input)
        print(f"\n{handler.npc.core.name}: {reply}")


def main():
    parser = argparse.ArgumentParser(description="GAI NPC Dialogue System — Phase 1 CLI")
    parser.add_argument("--npc",  help="NPC name to load from data/personas/")
    parser.add_argument("--seed", help="Path to seeds JSON file")
    parser.add_argument("--name", help="NPC name within the seed file")
    parser.add_argument("--text", action="store_true", help="Use text input instead of microphone")
    args = parser.parse_args()

    llm = OllamaClient()
    npc = load_or_generate_npc(args, llm)

    print(f"\n[Main] NPC ready: {npc.core.name} | {npc.core.occupation}")
    print(f"       Speech style: {npc.core.speech_style}")

    handler = DialogueHandler(llm, npc)
    run_cli(handler, text_mode=args.text)


if __name__ == "__main__":
    main()
