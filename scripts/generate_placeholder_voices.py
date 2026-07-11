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
import subprocess
import sys
import tempfile
from pathlib import Path

from backend.config.settings import SEEDS_DIR, VOICES_DIR


def voice_output_path(npc_name: str, voices_dir: str = VOICES_DIR) -> str:
    filename = npc_name.lower().replace(" ", "_") + ".wav"
    return os.path.join(voices_dir, filename)


def voice_line_for(npc_name: str, occupation: str) -> str:
    return (
        f"Hello, I am {npc_name}, the {occupation.lower()} of Suntail Village. "
        "The morning is quiet, but there is always work to do. "
        "Travelers bring news from the mountain road, and I listen carefully before I answer."
    )


def pick_voice_id(voices, gender: str, preferred_lang: str = "en") -> str:
    """Pick an installed SAPI voice id matching `gender`, preferring one whose
    language contains `preferred_lang`. Falls back to any voice of that
    gender, then to the first available voice, if no exact match exists."""
    wanted = (gender or "").lower()
    fallback = None
    for voice in voices:
        if (getattr(voice, "gender", None) or "").lower() != wanted:
            continue
        languages = [str(lang).lower() for lang in (getattr(voice, "languages", None) or [])]
        if any(preferred_lang in lang for lang in languages):
            return voice.id
        if fallback is None:
            fallback = voice.id
    if fallback is not None:
        return fallback
    return voices[0].id


_WORKER_SCRIPT = """
import sys
import pyttsx3

voice_id, out_path, text_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(text_path, encoding="utf-8") as f:
    line = f.read()

engine = pyttsx3.init()
engine.setProperty("voice", voice_id)
engine.save_to_file(line, out_path)
engine.runAndWait()
"""


def _synthesize_in_subprocess(line: str, voice_id: str, out_path: str) -> None:
    """Run one pyttsx3 save_to_file() in its own short-lived process.

    pyttsx3's SAPI5 driver on Windows reliably hangs forever on the *second*
    init()+runAndWait() inside a single process (reproduced even reusing the
    same voice id, so it isn't about switching voices) — isolating each line
    in a fresh subprocess sidesteps the deadlock entirely.
    """
    fd, text_path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(line)
        subprocess.run(
            [sys.executable, "-c", _WORKER_SCRIPT, voice_id, out_path, text_path],
            check=True,
            timeout=30,
        )
    finally:
        os.remove(text_path)


_SYSTEM_SPEECH_SCRIPT = r"""
param(
    [string]$Voice,
    [string]$OutPath,
    [string]$Line
)
Add-Type -AssemblyName System.Speech
$engine = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $engine.SelectVoice($Voice)
    $engine.SetOutputToWaveFile($OutPath)
    $engine.Speak($Line)
}
finally {
    $engine.Dispose()
}
"""


def _generate_with_windows_system_speech(seeds, voices_dir: str) -> list:
    """Fallback for Windows machines where pyttsx3's SAPI COM driver is absent."""
    if os.name != "nt":
        raise RuntimeError("Windows System.Speech fallback is only available on Windows")

    female_voices = ["Microsoft Zira Desktop", "Microsoft Hazel Desktop", "Microsoft Huihui Desktop"]
    male_voices = ["Microsoft David Desktop"]
    Path(voices_dir).mkdir(parents=True, exist_ok=True)

    fd, script_path = tempfile.mkstemp(suffix=".ps1")
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig") as script_file:
            script_file.write(_SYSTEM_SPEECH_SCRIPT)

        written = []
        female_index = 0
        male_index = 0
        for seed in seeds:
            if str(seed.get("gender", "male")).lower() == "female":
                voice = female_voices[female_index % len(female_voices)]
                female_index += 1
            else:
                voice = male_voices[male_index % len(male_voices)]
                male_index += 1

            path = voice_output_path(seed["name"], voices_dir)
            subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", script_path,
                    "-Voice", voice,
                    "-OutPath", os.path.abspath(path),
                    "-Line", voice_line_for(seed["name"], seed["occupation"]),
                ],
                check=True,
                timeout=30,
            )
            written.append(path)
        return written
    finally:
        os.remove(script_path)


def generate_all(seeds_path: str, voices_dir: str = VOICES_DIR) -> list:
    import pyttsx3

    with open(seeds_path, encoding="utf-8") as f:
        seeds = json.load(f)

    Path(voices_dir).mkdir(parents=True, exist_ok=True)

    try:
        probe_engine = pyttsx3.init()
    except Exception as exc:
        print(f"[Voices] pyttsx3 unavailable ({exc}); using Windows System.Speech.")
        return _generate_with_windows_system_speech(seeds, voices_dir)
    voices = probe_engine.getProperty("voices")
    probe_engine.stop()

    written = []
    for seed in seeds:
        voice_id = pick_voice_id(voices, seed.get("gender", "male"))
        line = voice_line_for(seed["name"], seed["occupation"])
        path = voice_output_path(seed["name"], voices_dir)
        _synthesize_in_subprocess(line, voice_id, path)
        written.append(path)
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
