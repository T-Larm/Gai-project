"""Per-component latency measurement (RQ4).

Usage (from project root; Ollama running, models downloaded):
    python -m evaluation.measure_latency --n 30 --components llm
    python -m evaluation.measure_latency --n 10 --components stt,llm,tts
"""
import argparse
import statistics
import time


def time_calls(fn, n: int):
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return times


def summarize_latency(component_times: dict) -> dict:
    summary = {}
    for component, times in component_times.items():
        summary[component] = {
            "n": len(times),
            "mean": statistics.mean(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min": min(times),
            "max": max(times),
        }
    return summary


def _build_llm_call():
    from backend.config.settings import PERSONAS_DIR
    from backend.llm.dialogue import DialogueHandler
    from backend.llm.ollama_client import OllamaClient
    from backend.llm.persona.generator import PersonaGenerator
    import os

    npc = PersonaGenerator.load(os.path.join(PERSONAS_DIR, "aldric.json"))
    npc.memory_log = []
    handler = DialogueHandler(OllamaClient(), npc, dynamic_updates=False)
    return lambda: handler.respond("Tell me about your work.")


def _build_stt_call():
    import whisper
    from backend.audio_utils import wav_bytes_to_float32
    from backend.config.settings import VOICES_DIR, WHISPER_MODEL
    import os

    # CPU to match production (backend/server.py pins whisper to CPU so it
    # never competes with XTTS/llama3 for the 8 GB of VRAM).
    model = whisper.load_model(WHISPER_MODEL, device="cpu")
    with open(os.path.join(VOICES_DIR, "aldric.wav"), "rb") as f:
        waveform, _ = wav_bytes_to_float32(f.read())
    return lambda: model.transcribe(waveform, fp16=False, language="en")


def _build_tts_call():
    from backend.config.settings import VOICES_DIR
    from backend.tts.xtts_client import XTTSClient
    import os

    client = XTTSClient()
    speaker = os.path.join(VOICES_DIR, "aldric.wav")
    return lambda: client.synthesize("A fine blade takes a week of honest work.", speaker)


_BUILDERS = {"llm": _build_llm_call, "stt": _build_stt_call, "tts": _build_tts_call}


def main():
    parser = argparse.ArgumentParser(description="Measure per-component latency")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--components", default="llm",
                        help="comma-separated subset of: stt,llm,tts")
    args = parser.parse_args()

    component_times = {}
    for name in args.components.split(","):
        name = name.strip()
        print(f"[Latency] {name}: preparing (model load excluded from timing)...")
        call = _BUILDERS[name]()
        call()  # warm-up, excluded
        print(f"[Latency] {name}: timing {args.n} calls...")
        component_times[name] = time_calls(call, args.n)

    print(f"\n{'component':<12}{'n':>4}{'mean (s)':>12}{'std':>10}{'min':>10}{'max':>10}")
    for name, stats in summarize_latency(component_times).items():
        print(f"{name:<12}{stats['n']:>4}{stats['mean']:>12.2f}"
              f"{stats['std']:>10.2f}{stats['min']:>10.2f}{stats['max']:>10.2f}")


if __name__ == "__main__":
    main()
