# Phase 3: TTS Integration Design (Coqui XTTS v2)

## Goal

Extend the existing text-only NPC dialogue pipeline (Phase 1: STT+LLM, Phase 2: semantic memory retrieval) with speech output: every NPC reply is synthesized in the NPC's cloned voice and played back, completing the voice-in → LLM → voice-out loop.

## Scope

Full integration (not just a standalone TTS module): TTS is wired into `DialogueHandler` and exposed via a CLI flag in `backend/main.py`.

## Components

```
backend/tts/xtts_client.py               XTTSClient — lazy-loads Coqui XTTS v2, synthesizes + plays audio (blocking)
data/voices/{npc_name_lower}.wav         reference clips for zero-shot voice cloning (placeholder for now)
scripts/generate_placeholder_voices.py   one-off script (pyttsx3, offline Windows SAPI) creating placeholder wavs per NPC
```

### XTTSClient

- Lazily loads `TTS("tts_models/multilingual/multi-dataset/xtts_v2")` on first use (mirrors the lazy `_get_embedder()` pattern in `backend/llm/persona/memory.py`).
- Sets `COQUI_TOS_AGREED=1` before importing/loading the model, since XTTS v2 is distributed under the non-commercial Coqui Public Model License and normally prompts for interactive agreement on first load. Acceptable here: this is a non-commercial university course project.
- `speak(text: str, speaker_wav: str) -> None`: calls `tts.tts(text=text, speaker_wav=speaker_wav, language="en")` to get a waveform array directly (no temp files), then plays it via `sounddevice.play()` + `sd.wait()` (same blocking pattern already used for recording in `backend/stt/whisper_stt.py`).
- If `speaker_wav` path does not exist, raises a clear error telling the user to run `scripts/generate_placeholder_voices.py` first.

### Voice reference resolution

Convention-based: `data/voices/{npc.core.name.lower()}.wav`. No changes to `PersonaSeed`, `CorePersona`, or `generator.py`. Chosen over an explicit `voice_reference_path` field on `CorePersona` to avoid touching the persona schema for no current benefit (YAGNI) — can be added later if per-NPC path flexibility becomes necessary.

### Placeholder voice generation

`scripts/generate_placeholder_voices.py` uses `pyttsx3` (offline Windows SAPI voice, no model download, no network) to synthesize one short reference line per NPC (using `npc.core.name` / `npc.core.speech_style` to pick a line) into `data/voices/{name}.wav`. Run manually once per NPC seed set; not part of the app startup path. Real recorded/sourced voice clips can replace these files later without any code changes.

### Wiring into DialogueHandler

```python
class DialogueHandler:
    def __init__(self, llm: OllamaClient, npc: NPC, tts: Optional[XTTSClient] = None):
        ...
        self.tts = tts

    def respond(self, player_input: str) -> str:
        ...
        reply = self.llm.chat(...)
        ...
        if self.tts is not None:
            speaker_wav = os.path.join(VOICES_DIR, f"{self.npc.core.name.lower()}.wav")
            self.tts.speak(reply, speaker_wav)
        return reply
```

`tts` defaults to `None` — fully backward compatible. Existing `test_dialogue.py` and Phase 1/2 automated tests are unaffected.

### CLI wiring (`backend/main.py`)

New `--speak` flag (default off). When passed, instantiates `XTTSClient()` and passes it into `DialogueHandler`. Kept opt-in because the XTTS v2 model is a large (~2GB) first-time download and this machine has no GPU (`torch.cuda.is_available() == False`), so CPU-only synthesis is slow — default-off preserves fast text-only iteration for LLM/persona debugging.

## Settings additions (`backend/config/settings.py`)

```python
TTS_MODEL     = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_LANGUAGE  = "en"
VOICES_DIR    = "data/voices"
```

## Dependencies

**Revised after hands-on verification** (see below): the originally planned `TTS` package does not work in this environment.

- `coqui-tts` (PyPI) — **not** the original `TTS` package. The original `coqui-ai/TTS` (last released 2023, PyPI name `TTS`) is abandoned: it hard-pins `numpy==1.22.0`, which breaks `sentence-transformers`/`transformers` (Phase 2 dependency), and even after forcing a newer numpy, its XTTS code fails with `ImportError: cannot import name 'BeamSearchScorer' from transformers` against any modern `transformers`. `coqui-tts` is the actively maintained fork (same import name `TTS`, same `TTS.api.TTS` interface, same model IDs) built specifically to fix this. Verified: importable and runs real XTTS v2 synthesis alongside the existing `sentence-transformers` Phase 2 stack with no regressions (`pytest tests/test_memory.py` still passes).
- `torch==2.8.0` / `torchaudio==2.8.0` — pinned down from whatever `coqui-tts` would otherwise pull in. From torch 2.9 onward, `torchaudio`'s audio loading requires `torchcodec`, which dynamically loads system FFmpeg shared libraries at runtime (`libtorchcodec_coreN.dll`) — these are not present on this Windows machine and installing them requires a manual system-wide FFmpeg install outside pip. Pinning to 2.8.0 avoids the `torchcodec` dependency entirely (falls back to the legacy `torchaudio` backend). Verified: XTTS v2 synthesis works and produces a waveform (list of floats) with this pin; `pytest tests/test_memory.py` still passes (sentence-transformers has no torch version floor above 1.11).
- `pyttsx3` — placeholder voice generation only (offline Windows SAPI, no model weights, no network).
- Known installer quirk (already resolved on this machine, noted for reproducibility): `coqui-tts` depends on `coqpit-config`, which installs into the same `coqpit` import path as the old, unrelated `coqpit` package. If both end up installed, imports break with `cannot import name 'Coqpit' from 'coqpit'`. Fix: `pip uninstall coqpit` then `pip install --force-reinstall --no-deps coqpit-config`.

Verified end-to-end on this machine: `TTS(TTS_MODEL).tts(text=..., speaker_wav=..., language="en")` returns a `list[float]` waveform, and `model.synthesizer.output_sample_rate` is `24000` for XTTS v2 — confirming the `XTTSClient.speak()` design above is accurate, not speculative.

## Error handling

- Missing reference wav for an NPC → explicit, actionable error message (run the placeholder script).
- XTTS license prompt → bypassed via `COQUI_TOS_AGREED=1` (documented, non-commercial use).
- Audio playback errors (e.g. no output device) are allowed to propagate — not swallowed.

## Testing strategy

Real XTTS v2 synthesis is too slow/heavy (large model, CPU-only, non-deterministic audio) for automated pytest runs. Split:

1. **Automated (pytest)**: verify wiring only — `DialogueHandler.respond()` calls `tts.speak(reply_text, expected_speaker_wav_path)` when a `tts` is provided, and does not call it when `tts=None`. Uses a lightweight fake TTS stub (records calls, does no real synthesis), following the same isolation pattern as the `_ConstantEmbedder` stub in `tests/test_memory.py`.
2. **Manual smoke test**: `python -m backend.main --npc aldric --text --speak` — real XTTS synthesis + real playback, judged by ear. Not part of the automated suite; audio quality isn't something an assertion can verify.

## Out of scope (future phases)

- Real recorded/sourced reference voice clips (VCTK or similar) — placeholder audio is a stand-in.
- Lip-sync (OVRLipSync, Phase 4).
- Unity integration of audio playback.
- Latency optimization / GPU inference.
