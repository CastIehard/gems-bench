"""Record one canonical human WAV for every GEMS-Bench question.

Starts a local browser UI that shows `spoken_question` as the script to read
ALOUD VERBATIM — same wording as the TTS channel, so the two audio sources are
directly comparable (no free paraphrasing). Captures a selected microphone and
writes 24 kHz mono PCM16 WAV files to audio_real/. Existing files are skipped
unless rerun=True.

Two speakers share the set 50/50, stratified per category (see
`recording.speakers` in config.yaml). The recorder serves only the queue of
`recording.currently_recording`, so each speaker sits down once and works
through their own half; flip that config key when the first speaker is done.

Usage:
    python src/record_audio.py
    python src/record_audio.py --rerun
    python src/record_audio.py --no-open-browser
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import threading
import traceback
import wave
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import verify_recording
from common import read_json, read_jsonl, write_json
from config import load_config

PAGE_PATH = Path(__file__).with_name("record_audio.html")


def assign_speakers(cfg: dict, items: list[dict[str, Any]]) -> bool:
    """Deal every question to exactly one speaker, stratified per category.

    Deterministic from `seed`: per category the items are ordered by item_id,
    shuffled with a category-salted seed, then dealt round-robin over
    `recording.speakers`. Each speaker therefore gets the same count PER
    CATEGORY, which keeps speaker orthogonal to category (see config.yaml).
    Idempotent — re-running never reshuffles an already-assigned set. Returns
    True if any item changed, so the caller knows to persist.
    """
    speakers = list(cfg["recording"]["speakers"])
    by_category: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_category.setdefault(item["category"], []).append(item)

    changed = False
    for category, group in sorted(by_category.items()):
        order = sorted(group, key=lambda item: item["item_id"])
        random.Random(f"{cfg['seed']}:{category}").shuffle(order)
        for index, item in enumerate(order):
            speaker = speakers[index % len(speakers)]
            if item.get("speaker") != speaker:
                item["speaker"] = speaker
                changed = True
    return changed


class RecorderState:
    def __init__(self, cfg: dict, *, rerun: bool, rerun_failed: bool = False) -> None:
        self.cfg = cfg
        self.questions_path = cfg["_paths"]["questions"]
        if not self.questions_path.exists():
            raise SystemExit(f"{self.questions_path} missing - run qa_checks.py first")

        self.items: list[dict[str, Any]] = read_json(self.questions_path)

        # ── which half of the set this session records ──
        self.speakers = list(cfg["recording"]["speakers"])
        self.speaker = str(cfg["recording"]["currently_recording"])
        if self.speaker not in self.speakers:
            raise SystemExit(
                f"recording.currently_recording={self.speaker!r} is not one of "
                f"recording.speakers={self.speakers}"
            )

        self.output_dir = cfg["_paths"]["audio_dirs"]["real"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.relative_dir = str(cfg["audio"]["directories"]["real"])
        self.sample_rate = int(cfg["tts"]["sample_rate"])
        self.bytes_per_sample = int(cfg["tts"]["bytes_per_sample"])
        self._lock = threading.Lock()

        # ── recording gate (AGB check) ──
        gate_cfg = cfg["recording"]["gate"]
        self.gate_enabled = gate_cfg["enabled"]
        self.requeue_failed = gate_cfg["requeue_failed"]
        self._verify_pool = ThreadPoolExecutor(max_workers=1)  # STT is serial anyway
        self._verify_path = cfg["_paths"]["recording_verification"]
        self._verdicts: dict[str, dict] = (
            read_json(self._verify_path) if self._verify_path.exists() else {}
        )
        corpus_path = cfg["_paths"]["corpus"]
        self._doc_ids = (
            {d["doc_id"] for d in read_jsonl(corpus_path)}
            if corpus_path.exists()
            else set()
        )

        changed = assign_speakers(cfg, self.items)
        self.mine = {
            item["item_id"] for item in self.items if item["speaker"] == self.speaker
        }
        pending: list[dict[str, Any]] = []
        for item in self.items:
            target = self._target(item["item_id"])
            if target.is_file():
                changed |= self._set_real_path(item)
            if item["item_id"] not in self.mine:
                continue  # the other speaker's half — not this session's queue
            failed = not self._verdicts.get(item["item_id"], {}).get("passed", True)
            need = rerun or not target.is_file() or (rerun_failed and failed)
            if need:
                pending.append(item)
        if changed:
            write_json(self.questions_path, self.items)

        # Interleave categories (blinding hygiene): questions are generated
        # per-category, so an unshuffled queue records all serial items in a
        # block, then all combined, etc. — session warm-up/fatigue would align
        # with category. A seeded shuffle interleaves them; deterministic so a
        # resumed session (Ctrl+C) rebuilds the same order minus completed items.
        random.Random(cfg["seed"]).shuffle(pending)

        self.pending = pending
        self.position = 0

    def _target(self, item_id: str) -> Path:
        return self.output_dir / f"{item_id}.wav"

    def _relative_path(self, item_id: str) -> str:
        return f"{self.relative_dir}/{item_id}.wav"

    def _set_real_path(self, item: dict[str, Any]) -> bool:
        audio_files = item.setdefault("audio_files", {})
        expected = self._relative_path(item["item_id"])
        if audio_files.get("real") == expected:
            return False
        audio_files["real"] = expected
        return True

    def _current(self) -> dict[str, Any] | None:
        if self.position >= len(self.pending):
            return None
        return self.pending[self.position]

    def _item_payload(self, item: dict[str, Any]) -> dict[str, Any]:
        """The recording card: `text` is spoken_question, read ALOUD VERBATIM
        (no paraphrasing) — TTS and the human speaker use identical wording so
        the two audio channels are directly comparable. `entities`/`quantities`
        are shown only as a pronunciation aid (the invented names/portions that
        appear inside `text`), not as a free-wording constraint."""
        skeleton = item.get("skeleton", {})
        prior = self._verdicts.get(item["item_id"], {})
        retake = prior.get("passed") is False
        # NOTE: category is deliberately NOT sent to the speaker (blinding). If the
        # speaker sees "combined"/"serial" they could unconsciously pace the "money"
        # categories more generously → inflate the between-category contrast that is
        # the headline.
        return {
            "id": item["item_id"],
            "text": item.get("spoken_question") or item.get("question_text", ""),
            "entities": skeleton.get("entities", []),
            "quantities": skeleton.get("quantities", []),
            "retake": retake,
            "prior_failed_checks": prior.get("failed_checks", []) if retake else [],
        }

    def _verification_summary(self) -> dict[str, Any]:
        """Scoped to the current speaker's half — the person at the microphone
        should see their own progress, not the other speaker's counts."""
        mine = {vid: v for vid, v in self._verdicts.items() if vid in self.mine}
        return {
            "passed": sum(1 for v in mine.values() if v.get("passed")),
            "failed": sum(1 for v in mine.values() if v.get("passed") is False),
            "failed_ids": [vid for vid, v in mine.items() if v.get("passed") is False],
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self.snapshot_unlocked()

    def snapshot_unlocked(self) -> dict[str, Any]:
        current = self._current()
        base = {
            "gate_enabled": self.gate_enabled,
            "verification": self._verification_summary(),
            "pending": len(self.pending) - self.position,
            "speaker": self.speaker,
            "speaker_total": len(self.mine),
        }
        if current is None:
            return {"complete": True, **base}
        return {"complete": False, "item": self._item_payload(current), **base}

    def save(self, item_id: str, payload: bytes) -> dict[str, Any]:
        with self._lock:
            current = self._current()
            if current is None:
                raise ValueError("recording set is already complete")
            if item_id != current["item_id"]:
                raise ValueError(
                    f"expected recording for {current['item_id']}, got {item_id}"
                )
            self._validate_wav(payload)

            target = self._target(item_id)
            temporary = target.with_suffix(".wav.tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, target)

            self._set_real_path(current)
            write_json(self.questions_path, self.items)
            self.position += 1
            if self.gate_enabled:
                # Verify OFF the recording thread: Whisper + closed-book gate are
                # slow, so the speaker advances immediately; a failed item is
                # appended back to the queue for a re-take when its verdict lands.
                self._verify_pool.submit(self._verify_async, current)
            return self.snapshot_unlocked()

    def _verify_async(self, item: dict[str, Any]) -> None:
        try:
            verdict = verify_recording.verify_item(
                self.cfg,
                item,
                doc_ids=self._doc_ids,
                cache={},  # force fresh STT of the just-saved WAV
                rerun=True,
            )
        except Exception as exc:  # noqa: BLE001 — a flaky gate must not kill recording
            print(f"  verify [{item['item_id']}] error: {exc}")
            traceback.print_exc()
            verdict = {
                "item_id": item["item_id"],
                "passed": None,  # unknown → do NOT requeue, keep the recording
                "error": str(exc),
            }
        with self._lock:
            self._verdicts[item["item_id"]] = verdict
            write_json(self._verify_path, self._verdicts)
            status = (
                "ok"
                if verdict.get("passed")
                else ("error" if verdict.get("passed") is None else "FAIL")
            )
            print(
                f"  verify [{item['item_id']}] {status} {verdict.get('failed_checks', '')}"
            )
            if verdict.get("passed") is False and self.requeue_failed:
                self.pending.append(item)  # re-take at the end of the queue

    def _validate_wav(self, payload: bytes) -> None:
        try:
            with wave.open(io.BytesIO(payload), "rb") as recording:
                properties = (
                    recording.getframerate(),
                    recording.getnchannels(),
                    recording.getsampwidth(),
                    recording.getcomptype(),
                )
                expected = (self.sample_rate, 1, self.bytes_per_sample, "NONE")
                if properties != expected:
                    raise ValueError(
                        "expected 24 kHz mono PCM16 WAV; "
                        f"received rate={properties[0]}, channels={properties[1]}, "
                        f"sample_width={properties[2]}, compression={properties[3]}"
                    )
                if recording.getnframes() == 0:
                    raise ValueError("recording contains no audio frames")
        except (EOFError, wave.Error) as exc:
            raise ValueError(f"invalid WAV upload: {exc}") from exc


def _handler_class(state: RecorderState, page: bytes, app_config: dict[str, Any]):
    config_json = json.dumps(app_config, ensure_ascii=False).encode("utf-8")

    class RecorderHandler(BaseHTTPRequestHandler):
        server_version = "GEMSRecorder/1.0"

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(page, "text/html; charset=utf-8")
                return
            if self.path == "/api/config":
                self._send(config_json, "application/json; charset=utf-8")
                return
            if self.path == "/api/state":
                self._send_json(state.snapshot())
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/api/recording":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            item_id = self.headers.get("X-GEMS-Item-ID", "")
            try:
                content_length = int(self.headers.get("Content-Length", ""))
                if content_length <= 0:
                    raise ValueError("empty recording upload")
                payload = self.rfile.read(content_length)
                if len(payload) != content_length:
                    raise ValueError("incomplete recording upload")
                self._send_json(state.save(item_id, payload))
            except (TypeError, ValueError) as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        def _send_json(
            self, value: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self._send(payload, "application/json; charset=utf-8", status)

        def _send(
            self,
            payload: bytes,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            if self.path != "/api/state":
                super().log_message(format, *args)

    return RecorderHandler


def run(
    cfg: dict,
    *,
    rerun: bool = False,
    rerun_failed: bool = False,
    host: str | None = None,
    port: int | None = None,
    open_browser: bool | None = None,
) -> None:
    recorder_cfg = cfg["audio"]["recorder"]
    host = host or str(recorder_cfg["host"])
    port = port if port is not None else int(recorder_cfg["port"])
    if open_browser is None:
        open_browser = bool(recorder_cfg["open_browser"])

    state = RecorderState(cfg, rerun=rerun, rerun_failed=rerun_failed)
    page = PAGE_PATH.read_bytes()
    app_config = {
        "sampleRate": state.sample_rate,
        "echoCancellation": bool(recorder_cfg["echo_cancellation"]),
        "noiseSuppression": bool(recorder_cfg["noise_suppression"]),
        "autoGainControl": bool(recorder_cfg["auto_gain_control"]),
    }
    server = ThreadingHTTPServer((host, port), _handler_class(state, page, app_config))
    url = f"http://{host}:{port}"
    print(f"Recorder: {url}")
    print(f"Speaker: {state.speaker} (of {', '.join(state.speakers)})")
    print(
        f"Pending: {len(state.pending)} / {len(state.mine)} questions "
        f"for this speaker ({len(state.items)} total)"
    )
    print("Stop with Ctrl+C when recording is complete.")
    if open_browser:
        threading.Timer(0.25, webbrowser.open, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nRecorder stopped — waiting for pending verifications…")
    finally:
        server.server_close()
        state._verify_pool.shutdown(wait=True)
        summary = state._verification_summary()
        print(
            f"Verification: {summary['passed']} passed, {summary['failed']} failed"
            + (
                f" — re-record: {','.join(summary['failed_ids'])}"
                if summary["failed_ids"]
                else ""
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="overwrite every existing real WAV of recording.currently_recording",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="re-record only items whose recording gate verdict failed",
    )
    parser.add_argument("--host", help="override audio.recorder.host")
    parser.add_argument("--port", type=int, help="override audio.recorder.port")
    parser.add_argument(
        "--no-open-browser", action="store_true", help="print URL without opening it"
    )
    args = parser.parse_args()
    run(
        load_config(),
        rerun=args.rerun,
        rerun_failed=args.rerun_failed,
        host=args.host,
        port=args.port,
        open_browser=False if args.no_open_browser else None,
    )


if __name__ == "__main__":
    main()
