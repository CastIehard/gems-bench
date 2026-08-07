"""Reference driver — connect your own voice agent to GEMS-Bench.

Everything except talking to your system is done for you: reading the manifest,
feeding the question audio at real time, writing the answer recording and the run
index in the layout `judge.py` expects.

To use it, implement the three methods of `VoiceSystem` for your own agent and
pass your class to `run()`. Nothing else needs to change.

    class MyAgent(VoiceSystem):
        def start_turn(self, item): ...        # open a session / new turn
        def send_audio(self, chunk): ...       # push PCM16 audio at mic pace
        def end_turn(self): ...                # -> the spoken answer as PCM16

Then:

    python example_driver.py --mode my_system

Run it unchanged to check the plumbing without touching a model: the built-in
`SilentSystem` returns silence, so every item scores incorrect but the whole
chain — audio, records, index, scoring — is exercised for free.

Audio is fed in real time on purpose. A voice agent handed the whole utterance at
once has no listening window to work in, which is half of what this benchmark
measures. Do not "optimize" the pacing away.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import load_config  # noqa: E402

_CONFIG = load_config()
_PATHS = _CONFIG["_paths"]
_RT = _CONFIG["realtime"]

GEMS_DIR = _PATHS["gems_dir"]
MANIFEST_PATH = _PATHS["manifest"]
RESULTS_DIR = _PATHS["results_dir"]
BENCHMARK_RUN_PATH = _PATHS["benchmark_run"]
RUNNER_OUTPUT_DIR = _PATHS["runner_output_dir"]

AUDIO_SOURCE = _CONFIG["audio"]["source"]
SAMPLE_RATE_HZ = _CONFIG["tts"]["sample_rate"]
BYTES_PER_SAMPLE = _CONFIG["tts"]["bytes_per_sample"]
CHUNK_SAMPLES = int(_RT["chunk_samples"])
CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE
CHUNK_SECONDS = CHUNK_SAMPLES / SAMPLE_RATE_HZ
LEAD_IN_SILENCE_S = float(_RT["lead_in_silence_s"])
PAUSE_S = float(_RT["pause_s"])


class VoiceSystem:
    """The integration point. Implement these three methods for your agent."""

    def start_turn(self, item: dict) -> None:
        """Called before any audio of one question.

        `item` is the manifest entry, so you have the id and the category. Do not
        read `item["prompt"]`, `item["spoken"]` or `item["answer"]` here — the
        agent is supposed to hear the question, not be handed its text or answer.
        """
        raise NotImplementedError

    def send_audio(self, chunk: bytes) -> None:
        """One chunk of the user's speech, mono PCM16 little-endian.

        Called repeatedly at real time. Hand it straight to your agent's audio
        input; do not buffer the whole utterance and submit it at the end.
        """
        raise NotImplementedError

    def end_turn(self) -> bytes:
        """The user stopped speaking. Return the spoken answer as mono PCM16.

        Block until your agent has finished answering. Return empty bytes if it
        produced no answer — that scores as incorrect, which is the honest
        outcome, rather than aborting the run.
        """
        raise NotImplementedError


class SilentSystem(VoiceSystem):
    """Answers nothing. For testing the harness without a model or credentials."""

    def start_turn(self, item: dict) -> None:
        pass

    def send_audio(self, chunk: bytes) -> None:
        pass

    def end_turn(self) -> bytes:
        return b"\x00" * (SAMPLE_RATE_HZ * BYTES_PER_SAMPLE)  # 1 s of silence


def read_pcm16(path: Path) -> bytes:
    """Raw PCM16 frames of a WAV, asserting the format the benchmark ships."""
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != SAMPLE_RATE_HZ or wav.getnchannels() != 1:
            raise SystemExit(
                f"{path}: expected {SAMPLE_RATE_HZ} Hz mono, got "
                f"{wav.getframerate()} Hz / {wav.getnchannels()} channels"
            )
        return wav.readframes(wav.getnframes())


def write_pcm16(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(BYTES_PER_SAMPLE)
        wav.setframerate(SAMPLE_RATE_HZ)
        wav.writeframes(pcm)


def stream_at_mic_pace(system: VoiceSystem, pcm: bytes) -> None:
    """Feed `pcm` to the system in chunks, paced like a live microphone."""
    lead_in = b"\x00" * int(LEAD_IN_SILENCE_S * SAMPLE_RATE_HZ) * BYTES_PER_SAMPLE
    started = time.monotonic()
    sent = 0
    for offset in range(0, len(lead_in + pcm), CHUNK_BYTES):
        system.send_audio((lead_in + pcm)[offset : offset + CHUNK_BYTES])
        sent += 1
        # sleep until this chunk's playback time has actually elapsed, so a slow
        # system falls behind rather than the audio racing ahead of real time
        behind = started + sent * CHUNK_SECONDS - time.monotonic()
        if behind > 0:
            time.sleep(behind)


def ask(system: VoiceSystem, item: dict, run_id: str) -> dict:
    """Run one question end to end and write its answer recording."""
    audio_file = item["audio_files"][AUDIO_SOURCE]
    if not audio_file:
        raise SystemExit(
            f"{item['id']} has no {AUDIO_SOURCE} audio — set audio.source in config.yaml"
        )
    session_id = str(uuid.uuid4())
    started = time.monotonic()

    system.start_turn(item)
    stream_at_mic_pace(system, read_pcm16(GEMS_DIR / audio_file))
    answer_pcm = system.end_turn()

    # judge.py grades this file, and finds it by (run_id, session_id)
    write_pcm16(
        RUNNER_OUTPUT_DIR / run_id / "records" / session_id / "assistant_audio.wav",
        answer_pcm,
    )
    return {
        "session_id": session_id,
        "run_id": run_id,
        "status": "ok" if answer_pcm else "no_answer",
        "wall_time_s": round(time.monotonic() - started, 2),
    }


def run(
    system: VoiceSystem,
    mode: str,
    questions: list[str] | None = None,
    run_id: str | None = None,
) -> dict:
    """Run the benchmark and write results/benchmark_run.json for judge.py."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if questions:
        wanted = set(questions)
        manifest = [it for it in manifest if it["id"] in wanted]
        unknown = wanted - {it["id"] for it in manifest}
        if unknown:
            raise SystemExit(f"unknown question ids: {sorted(unknown)}")
    if not manifest:
        raise SystemExit("no questions selected")

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M-") + uuid.uuid4().hex[:8]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    index = (
        json.loads(BENCHMARK_RUN_PATH.read_text(encoding="utf-8"))
        if BENCHMARK_RUN_PATH.exists()
        else {}
    )
    # One entry per mode. Re-running the same mode overwrites its questions, so
    # archive results/benchmark_run.json before running a second configuration.
    mode_runs = index.setdefault(mode, {})

    try:
        for position, item in enumerate(manifest, start=1):
            print(
                f"[{position}/{len(manifest)}] {item['id']} "
                f"({item['audio_durations_s'][AUDIO_SOURCE]}s {AUDIO_SOURCE})",
                flush=True,
            )
            result = ask(system, item, run_id)
            result["mode"] = mode
            mode_runs[item["id"]] = result
            print(f"    {result['status']} in {result['wall_time_s']}s", flush=True)
            if position < len(manifest):
                time.sleep(PAUSE_S)
    finally:
        # written even on interrupt, so a partial run is still scorable
        BENCHMARK_RUN_PATH.write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nrun index -> {BENCHMARK_RUN_PATH}")
        print(f"records   -> {RUNNER_OUTPUT_DIR / run_id}")
        print("next      -> python judge.py")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode",
        required=True,
        help="label for this configuration; becomes the key in benchmark_run.json",
    )
    parser.add_argument(
        "--questions", default=None, help="comma-separated item ids (default: all)"
    )
    parser.add_argument("--run-id", default=None, help="default: generated")
    args = parser.parse_args()

    # Replace SilentSystem() with your own VoiceSystem implementation.
    run(
        SilentSystem(),
        mode=args.mode,
        questions=args.questions.split(",") if args.questions else None,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
