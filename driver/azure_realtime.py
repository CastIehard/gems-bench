"""Plain Azure OpenAI Realtime baseline for GEMS-Bench.

One realtime session per question, wired the way Azure intends it: the question
audio is streamed in at microphone pace, server VAD decides on its own when the
turn is over, the model calls the two GEMS tools as often as it wants and speaks
one answer. Nothing sits between the model and the tools.

    python driver/azure_realtime.py --mode baseline_synthetic
    python driver/azure_realtime.py --mode baseline_real --questions one_hop_0000

This is a HOST-side driver, not part of the benchmark: everything specific to the
provider lives in driver/azure_realtime.yaml, so another provider is another
driver plus another settings file and the benchmark's config.yaml never changes.
From config.yaml it reads only how the audio is fed (`realtime`) and which
channel to use (`audio.source`) — the stimulus, identical for every system.

Per question it writes what judge.py reads: the answer recording, plus
transcript/timing/raw-event logs so the latency and tool-call metrics are
populated instead of missing.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import orjson
import websockets
import yaml
from dotenv import load_dotenv

GEMS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GEMS_DIR))
import example_driver as ed  # noqa: E402
from example_driver import VoiceSystem  # noqa: E402
from gems_tools import build_tool_catalog  # noqa: E402
from src.config import load_config  # noqa: E402

SETTINGS_PATH = Path(__file__).resolve().with_suffix(".yaml")

_CFG = load_config()
_RT = _CFG["realtime"]
_AR = yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8"))

load_dotenv(GEMS_DIR / ".env.local")

SILENCE_CHUNK = b"\x00" * ed.CHUNK_BYTES
TIMEOUT_S = float(_RT["timeout_s"])
AUDIO_QUIET_DONE_S = float(_RT["audio_quiet_done_s"])
RECOVERABLE = frozenset(_RT["recoverable_error_codes"])
CONNECT_TIMEOUT_S = float(_AR["connect_timeout_s"])
TOOL_DELAY_S = float(_AR["tool_api_delay_ms"]) / 1000.0


def websocket_url(endpoint: str, deployment: str) -> str:
    """The Azure Realtime (v1 GA) websocket URL for a deployment."""
    e = endpoint.strip().rstrip("/")
    if e.startswith("https://"):
        e = f"wss://{e[len('https://'):]}"
    elif e.startswith("http://"):
        e = f"ws://{e[len('http://'):]}"
    elif not e.startswith(("wss://", "ws://")):
        e = f"wss://{e.lstrip('/')}"

    if e.endswith("/openai/v1"):
        base = f"{e}/realtime"
    elif e.endswith("/openai/v1/realtime"):
        base = e
    elif e.endswith("/openai"):
        base = f"{e}/v1/realtime"
    else:
        base = f"{e}/openai/v1/realtime"
    return f"{base}?{urlencode({'model': deployment})}"


def _credentials() -> tuple[str, str]:
    endpoint = (os.environ.get(_AR["endpoint_var"]) or "").strip()
    key = (os.environ.get(_AR["key_var"]) or "").strip()
    if not endpoint or not key:
        raise SystemExit(
            f"{_AR['endpoint_var']} / {_AR['key_var']} missing — add them to "
            f"{GEMS_DIR / '.env.local'} (names come from {SETTINGS_PATH.name})"
        )
    return endpoint, key


def realtime_tools(catalog: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The GEMS tool catalog in the schema the Realtime API expects."""
    return [
        {
            "type": "function",
            "name": name,
            "description": tool["description"],
            "parameters": tool["parameters"],
        }
        for name, tool in catalog.items()
    ]


class _LoopThread:
    """A background asyncio loop, so the synchronous VoiceSystem calls can drive
    an async websocket without every method becoming a coroutine."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def call(self, coro: Any, timeout: float | None = None) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)


class AzureRealtimeSystem(VoiceSystem):
    """A plain Azure Realtime session, one per question."""

    def __init__(self) -> None:
        self.endpoint, self.api_key = _credentials()
        self.url = websocket_url(self.endpoint, _AR["deployment"])
        prompts = yaml.safe_load(
            _CFG["_paths"]["prompts"].read_text(encoding="utf-8")
        )
        self.instructions = prompts["system_prompt"]
        self.catalog = build_tool_catalog()
        self.tools = realtime_tools(self.catalog)
        self._loop = _LoopThread()
        self._ws: Any = None
        self._reader: asyncio.Task[None] | None = None
        self._reset()

    # ── per-turn state ──────────────────────────────────────────────────────
    def _reset(self) -> None:
        self.audio = bytearray()
        self.transcripts: list[str] = []
        self.timing: list[dict[str, Any]] = []
        self.raw: list[dict[str, Any]] = []
        self.status = "ok"
        self.error: str | None = None
        self._speech_active = False
        self._eos_ts: float | None = None
        self._first_audio_ts: float | None = None
        self._ttft: float | None = None
        self._last_audio_wall: float | None = None
        self._done: asyncio.Event | None = None
        # True while the question audio is still being fed. A response that
        # finishes in this window answers a question the user has not finished
        # asking, so it is discarded and the turn keeps going.
        self._streaming = True
        self._premature_audio = b""
        self._premature_count = 0

    def _log_timing(self, event: str, ts: float, **payload: Any) -> None:
        self.timing.append(
            {
                "event_type": "timing_event",
                "timestamp_monotonic": ts,
                "payload": {"event": event, **payload},
            }
        )

    def _log_raw(self, direction: str, event: dict[str, Any], ts: float) -> None:
        # Audio deltas are logged without their base64 payload: the answer audio
        # is already on disk as a WAV, and keeping it here twice would make the
        # log bigger than the recording.
        if event.get("type") in {"response.output_audio.delta", "input_audio_buffer.append"}:
            event = {**event, "delta": None, "audio": None}
        self.raw.append(
            {
                "timestamp_monotonic": ts,
                "payload": {"direction": direction, "event": event},
            }
        )

    # ── VoiceSystem contract ────────────────────────────────────────────────
    def start_turn(self, item: dict) -> None:
        self._reset()
        self._loop.call(self._open(), timeout=CONNECT_TIMEOUT_S + 10)

    def send_audio(self, chunk: bytes) -> None:
        self._loop.call(self._append_audio(chunk), timeout=30)

    def end_turn(self) -> bytes:
        self._loop.call(self._await_answer(), timeout=TIMEOUT_S + 60)
        return bytes(self.audio)

    def write_records(self, record_dir: Path) -> None:
        def dump(name: str, rows: list[dict[str, Any]]) -> None:
            if not rows:
                return
            (record_dir / name).write_bytes(
                b"".join(orjson.dumps(row) + b"\n" for row in rows)
            )

        transcript = " ".join(t for t in self.transcripts if t).strip()
        dump(
            "transcript.jsonl",
            [
                {
                    "payload": {
                        "speaker": "assistant",
                        "transcript": transcript,
                        "ttft_seconds": self._ttft,
                    }
                }
            ]
            if transcript
            else [],
        )
        dump("timing.jsonl", self.timing)
        dump("raw_realtime_events.jsonl", self.raw)

    def close(self) -> None:
        self._loop.stop()

    # ── session ─────────────────────────────────────────────────────────────
    def _session_payload(self) -> dict[str, Any]:
        audio_format = {"type": "audio/pcm", "rate": ed.SAMPLE_RATE_HZ}
        return {
            "type": "realtime",
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": audio_format,
                    "turn_detection": dict(_AR["turn_detection"]),
                },
                "output": {"format": audio_format, "voice": _AR["voice"]},
            },
            "instructions": self.instructions,
            "tools": self.tools,
            "tool_choice": "auto",
        }

    async def _open(self) -> None:
        self._done = asyncio.Event()
        self._ready = asyncio.Event()
        self._ws = await asyncio.wait_for(
            websockets.connect(
                self.url,
                additional_headers={"api-key": self.api_key},
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
                max_size=2**24,
            ),
            timeout=CONNECT_TIMEOUT_S,
        )
        self._reader = asyncio.create_task(self._read_loop())
        await self._send({"type": "session.update", "session": self._session_payload()})
        # Wait for the ack: audio appended before the session settles would be
        # judged by the default VAD, not the configured one.
        await asyncio.wait_for(self._ready.wait(), timeout=CONNECT_TIMEOUT_S)

    async def _send(self, event: dict[str, Any]) -> None:
        self._log_raw("out", event, time.monotonic())
        await self._ws.send(orjson.dumps(event).decode("utf-8"))

    async def _append_audio(self, chunk: bytes) -> None:
        await self._ws.send(
            orjson.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
            ).decode("utf-8")
        )

    async def _await_answer(self) -> None:
        """Keep the microphone open with silence until the answer is complete.

        The turn is never committed by hand — server VAD ends it, which is the
        whole point of the plain configuration. Silence keeps flowing so the
        session behaves like a live call rather than a closed pipe.
        """
        loop = asyncio.get_running_loop()
        # Flipped here rather than from the calling thread: this coroutine and the
        # event handlers share one loop, so no response can slip through half-way.
        self._streaming = False
        deadline = loop.time() + TIMEOUT_S
        next_t = loop.time()
        try:
            assert self._done is not None
            while not self._done.is_set():
                if loop.time() > deadline:
                    self.status = "timeout"
                    break
                await self._append_audio(SILENCE_CHUNK)
                next_t += ed.CHUNK_SECONDS
                delay = next_t - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                if self._audio_went_quiet():
                    break
        finally:
            await self._close()
        if not self.audio and self._premature_audio:
            # No answer came after the question ended. Better the early one than
            # nothing, but it is recorded so the item can be told apart later.
            self.audio = bytearray(self._premature_audio)
            self.status = "premature_answer_only"
            self._log_timing("answer_from_premature_response", time.monotonic())

    def _audio_went_quiet(self) -> bool:
        """True once answer audio arrived and then stopped for long enough.

        A fallback for a session that never sends its closing `response.done`;
        without it a single stalled turn would burn the whole per-question cap.
        """
        if not self.audio or self._last_audio_wall is None:
            return False
        return (time.monotonic() - self._last_audio_wall) >= AUDIO_QUIET_DONE_S

    async def _close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._reader = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass
            self._ws = None

    # ── events ──────────────────────────────────────────────────────────────
    async def _read_loop(self) -> None:
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    continue
                await self._on_event(orjson.loads(message), time.monotonic())
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "failed"
            if self._done is not None:
                self._done.set()

    async def _on_event(self, event: dict[str, Any], ts: float) -> None:
        etype = event.get("type") or ""
        self._log_raw("in", event, ts)

        if etype in {"session.created", "session.updated"}:
            self._ready.set()
            return

        if etype == "input_audio_buffer.speech_started":
            self._speech_active = True
            self._log_timing("input_speech_started", ts)
            return

        if etype == "input_audio_buffer.speech_stopped":
            self._speech_active = False
            self._eos_ts = ts
            self._log_timing("input_speech_stopped", ts)
            return

        if etype == "response.output_audio.delta":
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                self.audio.extend(base64.b64decode(delta))
                self._last_audio_wall = time.monotonic()
                if self._first_audio_ts is None:
                    self._first_audio_ts = ts
                    self._log_timing("assistant_first_audio", ts)
                    if self._eos_ts is not None:
                        self._ttft = round(ts - self._eos_ts, 3)
            return

        if etype == "response.output_audio_transcript.done":
            text = (event.get("transcript") or "").strip()
            if text:
                self.transcripts.append(text)
            return

        if etype == "response.function_call_arguments.done":
            # Logged at the moment the model asked, which is what the tool-split
            # metric needs; the call itself runs when the response closes.
            self._log_timing(
                "speech_tool_call",
                ts,
                tool_name=event.get("name"),
                user_speech_active=self._speech_active,
            )
            return

        if etype == "response.done":
            await self._on_response_done(event)
            return

        if etype == "error":
            err = event.get("error") or {}
            code = err.get("code") or err.get("type")
            self.error = orjson.dumps(err).decode("utf-8")
            if code not in RECOVERABLE and not self.audio:
                self.status = "failed"
                if self._done is not None:
                    self._done.set()
            return

    async def _on_response_done(self, event: dict[str, Any]) -> None:
        """Run any tools the finished response asked for, else end the turn.

        Tools are executed here rather than on `function_call_arguments.done` so
        a response that requests several of them yields one batch of outputs and
        exactly one follow-up request, instead of one response per call.
        """
        response = event.get("response") or {}
        calls = [
            item
            for item in (response.get("output") or [])
            if item.get("type") == "function_call"
        ]

        if not calls:
            if self._streaming:
                self._discard_premature(response)
                return
            if response.get("status") in {"failed", "cancelled"}:
                self.error = orjson.dumps(response.get("status_details") or {}).decode(
                    "utf-8"
                )
                self.status = "failed"
            if self._done is not None:
                self._done.set()
            return

        for call in calls:
            result = await self._run_tool(call)
            await self._send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call.get("call_id"),
                        "output": orjson.dumps(result).decode("utf-8"),
                    },
                }
            )
        await self._send({"type": "response.create"})

    def _discard_premature(self, response: dict[str, Any]) -> None:
        """Throw away an answer the model gave before the question was over.

        Server VAD ends the turn on any pause long enough to look like one, so a
        question with a real pause in it can be answered halfway through. That
        answer is to a truncated question, and keeping it would score the model
        on a question nobody asked. The audio is stashed only as a last resort
        for the case where no further answer ever arrives, and everything else
        is reset so the next response starts from a clean slate.
        """
        self._premature_count += 1
        if self.audio and not self._premature_audio:
            self._premature_audio = bytes(self.audio)
        self._log_timing(
            "premature_response_discarded",
            time.monotonic(),
            status=response.get("status"),
            audio_bytes=len(self.audio),
            occurrence=self._premature_count,
        )
        self.audio = bytearray()
        self.transcripts = []
        self._first_audio_ts = None
        self._ttft = None
        self._last_audio_wall = None

    async def _run_tool(self, call: dict[str, Any]) -> dict[str, Any]:
        name = call.get("name") or ""
        tool = self.catalog.get(name)
        if tool is None:
            return {"status": "error", "message": f"unknown tool: {name}"}
        try:
            arguments = orjson.loads(call.get("arguments") or "{}")
        except ValueError:
            return {"status": "error", "message": "arguments were not valid JSON"}
        if TOOL_DELAY_S and tool.get("api_delay"):
            await asyncio.sleep(TOOL_DELAY_S)
        try:
            return await asyncio.to_thread(tool["handler"], arguments)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "message": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode",
        required=True,
        help="label for this run; becomes the key in benchmark_run.json",
    )
    parser.add_argument(
        "--questions", default=None, help="comma-separated item ids (default: all)"
    )
    parser.add_argument("--run-id", default=None, help="default: generated")
    parser.add_argument(
        "--audio",
        default=None,
        choices=sorted(_CFG["audio"]["directories"]),
        help="audio channel for this run (default: audio.source from config.yaml)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="connect, update the session, print the tool schema, exit",
    )
    args = parser.parse_args()

    # Overriding here rather than editing config.yaml: the channel is a property
    # of THIS run, and both channels are listed for every manifest item anyway.
    if args.audio:
        ed.AUDIO_SOURCE = args.audio

    system = AzureRealtimeSystem()
    print(f"deployment : {_AR['deployment']}")
    print(f"endpoint   : {system.endpoint}")
    print(f"audio      : {ed.AUDIO_SOURCE}")
    print(f"tools      : {', '.join(t['name'] for t in system.tools)}")
    print(f"tool delay : {_AR['tool_api_delay_ms']} ms")

    try:
        if args.check:
            system.start_turn({"id": "connection_check"})
            print("\nsession accepted — credentials, deployment and schema are fine")
            system._loop.call(system._close())
            return
        ed.run(
            system,
            mode=args.mode,
            questions=args.questions.split(",") if args.questions else None,
            run_id=args.run_id,
        )
    finally:
        system.close()


if __name__ == "__main__":
    main()
