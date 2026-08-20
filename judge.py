"""Post-hoc GEMS-Bench evaluation: DETERMINISTIC scoring + latency/tool metrics.

Reads results/benchmark_run.json (written by your driver — see example_driver.py),
pulls each session's records from the output directory of the system under test,
and grades the spoken answer against the GEMS ground truth WITHOUT an LLM judge
(src/scoring.py — deterministic scorability is the point). Writes
results/results.json.

The graded answer comes from transcribing the recorded answer audio
(assistant_audio.wav) with the vendored open-source STT (Whisper large-v3 via
src/oss_stt.py), so scoring needs no cloud service.

Usage:
    python judge.py [--modes my_system,baseline]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import oss_stt  # noqa: E402
from src.config import load_config  # noqa: E402
from src.scoring import normalize_name, score_item, score_list_f1  # noqa: E402

_CONFIG = load_config()
_PATHS = _CONFIG["_paths"]
BENCHMARK_RUN_PATH = _PATHS["benchmark_run"]
MANIFEST_PATH = _PATHS["manifest"]
RESULTS_PATH = _PATHS["results"]
RESULTS_DIR = _PATHS["results_dir"]
RUNNER_OUTPUT_DIR = _PATHS["runner_output_dir"]

STT_CACHE_PATH = RESULTS_DIR / "stt_cache.json"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line]


def session_dir(run: dict) -> Path:
    return RUNNER_OUTPUT_DIR / run["run_id"] / "records" / run["session_id"]


def whisper_answer(record_dir: Path, run: dict, cache: dict) -> str | None:
    """Transcribe the recorded assistant answer audio with OSS Whisper large-v3.

    Cached by run_id/session_id — Whisper is slow and answer audio never changes.
    Returns None if the audio is missing (caller falls back to any realtime
    transcript).
    """
    key = f"{run['run_id']}/{run['session_id']}"
    if key in cache:
        return cache[key]
    wav = record_dir / "assistant_audio.wav"
    if not wav.is_file():
        return None
    stt = _CONFIG["stt"]
    text = oss_stt.transcribe_wav(
        wav,
        model=stt["model"],
        language=stt["language"],
        long_form=stt["long_form"],
    )
    cache[key] = text
    return text


def extract_answer_and_ttft(transcript_events: list[dict]) -> tuple[str, float | None]:
    answers: list[str] = []
    ttft: float | None = None
    for event in transcript_events:
        payload = event.get("payload", {})
        if payload.get("speaker") != "assistant":
            continue
        text = (payload.get("transcript") or "").strip()
        if text:
            answers.append(text)
        if ttft is None and payload.get("ttft_seconds") is not None:
            ttft = float(payload["ttft_seconds"])
    return " ".join(answers), ttft


def extract_timing_metrics(timing_events: list[dict]) -> dict:
    """TTFA + tool-call split (during vs after listening) from timing.jsonl."""
    listening_intervals: list[list[float | None]] = []
    first_audio_ts: float | None = None
    last_stop_before_audio: float | None = None
    tool_events: list[dict] = []

    for event in timing_events:
        ts = event.get("timestamp_monotonic")
        payload = event.get("payload", {})
        if event.get("event_type") == "timing_event":
            name = payload.get("event")
            if name == "input_speech_started":
                listening_intervals.append([ts, None])
            elif name == "input_speech_stopped":
                if listening_intervals and listening_intervals[-1][1] is None:
                    listening_intervals[-1][1] = ts
                if first_audio_ts is None:
                    last_stop_before_audio = ts
            elif name == "assistant_first_audio" and first_audio_ts is None:
                first_audio_ts = ts
            elif name == "speech_tool_call":
                tool_events.append(
                    {
                        "timestamp_monotonic": ts,
                        "tool_name": payload.get("tool_name"),
                        "call_id": payload.get("call_id"),
                        "source": "interaction_layer",
                        "user_speech_active": payload.get("user_speech_active"),
                    }
                )
        elif event.get("event_type") == "background_event":
            if payload.get("kind") == "tool_call":
                tool_events.append(
                    {
                        "timestamp_monotonic": ts,
                        "tool_name": payload.get("tool_name"),
                        "call_id": payload.get("call_id"),
                        "source": payload.get("source") or "background",
                        "user_speech_active": None,
                    }
                )

    # Chunked calls are logged twice: once as the canonical timing event and
    # once as a UI background event. Keep one record per call_id, preferring the
    # timing event because it carries the explicit speech-active flag. Legacy
    # events without a call_id remain distinct.
    deduplicated: dict[str, dict] = {}
    unkeyed: list[dict] = []
    for tool in tool_events:
        call_id = tool.get("call_id")
        if not call_id:
            unkeyed.append(tool)
            continue
        existing = deduplicated.get(str(call_id))
        if existing is None or (
            existing["user_speech_active"] is None
            and tool["user_speech_active"] is not None
        ):
            deduplicated[str(call_id)] = tool
    tool_events = [*deduplicated.values(), *unkeyed]

    def during_listening(tool: dict) -> bool:
        if tool["user_speech_active"] is not None:
            return bool(tool["user_speech_active"])
        ts = tool["timestamp_monotonic"]
        if ts is None:
            return False
        for start, stop in listening_intervals:
            if start is not None and ts >= start and (stop is None or ts < stop):
                return True
        return False

    for tool in tool_events:
        tool["during_listening"] = during_listening(tool)

    ttfa = None
    if first_audio_ts is not None and last_stop_before_audio is not None:
        ttfa = first_audio_ts - last_stop_before_audio

    return {
        "ttfa_s": round(ttfa, 3) if ttfa is not None else None,
        # end-of-speech timestamp (monotonic): the last input_speech_stopped
        # before the assistant started answering. The deadline for
        # gold-doc-recall @ EOS is measured against this.
        "eos_ts": last_stop_before_audio,
        "tools_during_listening": sum(
            1 for tool in tool_events if tool["during_listening"]
        ),
        "tools_after_listening": sum(
            1 for tool in tool_events if not tool["during_listening"]
        ),
        "tool_events": [
            {
                "tool_name": tool["tool_name"],
                "source": tool["source"],
                "during_listening": tool["during_listening"],
            }
            for tool in tool_events
        ],
    }


def extract_retrieved_docs(
    raw_events: list[dict], timing_events: list[dict]
) -> dict[str, float | None]:
    """Distinct doc_ids the search tool RETURNED this session → earliest
    retrieval timestamp (monotonic).

    Two sources, because a system may retrieve from more than one place:

    * the model's own calls, logged as a `function_call_output` item inside a
      `conversation.item.create` event in raw_realtime_events.jsonl;
    * calls made by background workers the system runs beside the model, logged
      as a `background_event` with `kind == "subagent_tool_result"` in
      timing.jsonl. Those never travel over the realtime protocol, so reading
      only the first source scored a delegating system on the fraction of its
      retrieval that happened to run in the model itself.

    Document metrics are about the SYSTEM's retrieval and count both. The
    tool-CALL split in `extract_timing_metrics` is about what the model itself
    did and counts only `kind == "tool_call"` — the two answer different
    questions and must not be merged.

    Only `search_database` returns doc_ids (the structured `product_lookup` has
    none), so this measures prose-retrieval precision/recall. The timestamp lets
    us bin a gold doc as retrieved before/after end-of-speech (deadline).
    """
    first_ts: dict[str, float | None] = {}

    def note(result: object, ts: float | None) -> None:
        if not isinstance(result, dict):
            return
        for r in result.get("results", []) or []:
            if not isinstance(r, dict):
                continue
            did = r.get("doc_id")
            if not did:
                continue
            prev = first_ts.get(did)
            if did not in first_ts or (ts is not None and (prev is None or ts < prev)):
                first_ts[did] = ts

    for e in raw_events:
        inner = e.get("payload", e)
        ev = inner.get("event") or {}
        if ev.get("type") != "conversation.item.create":
            continue
        item = ev.get("item") or {}
        if item.get("type") != "function_call_output":
            continue
        out = item.get("output")
        if not out:
            continue
        try:
            parsed = out if isinstance(out, dict) else orjson.loads(out)
        except (TypeError, ValueError):
            continue
        note(parsed, e.get("timestamp_monotonic"))

    for e in timing_events:
        if e.get("event_type") != "background_event":
            continue
        payload = e.get("payload") or {}
        if payload.get("kind") != "subagent_tool_result":
            continue
        note(payload.get("result"), e.get("timestamp_monotonic"))

    return first_ts


def doc_retrieval_metrics(
    retrieved_ts: dict[str, float | None],
    gold_docs: list[str],
    eos_ts: float | None,
) -> dict:
    """Retrieval precision/recall over prose docs, + gold-doc-recall @ EOS.

    precision = required docs retrieved / all docs retrieved (the "how many of
    the docs it pulled were actually needed" metric). recall = gold docs found /
    gold docs. recall_eos = gold docs found BEFORE end-of-speech / gold docs —
    how much of the needed evidence was already in hand when the user stopped
    speaking."""
    retrieved = set(retrieved_ts)
    gold = set(gold_docs)
    hit = retrieved & gold
    precision = len(hit) / len(retrieved) if retrieved else None
    recall = len(hit) / len(gold) if gold else None
    recall_eos = None
    if gold and eos_ts is not None:
        gold_before = {
            d
            for d in hit
            if retrieved_ts.get(d) is not None and retrieved_ts[d] <= eos_ts
        }
        recall_eos = len(gold_before) / len(gold)
    return {
        "docs_retrieved": len(retrieved),
        "docs_gold": len(gold),
        "docs_gold_hit": len(hit),
        "doc_precision": round(precision, 3) if precision is not None else None,
        "doc_recall": round(recall, 3) if recall is not None else None,
        "gold_doc_recall_eos": round(recall_eos, 3) if recall_eos is not None else None,
    }


def _parse_spoken_list(answer: str) -> list[str]:
    """Split a spoken answer into candidate list elements (for the F1 diagnostic).

    Best-effort only: gold product names themselves contain commas
    ("Duschgel …, 250 ml"), so a comma split cannot recover exact elements —
    the headline list verdict uses full-text containment instead (see
    score_answer). This split just feeds the graded set-F1.
    """
    text = answer.split(":", 1)[1] if ":" in answer else answer
    parts = re.split(r"\bund\b|\bsowie\b|;", text)
    return [p.strip(" .,;") for p in parts if p.strip(" .,;")]


def _list_exact_spoken(answer: str, gold: list[str], universe: list[str]) -> bool:
    """Exact set on a spoken answer, without parsing commas (product names
    contain commas, so a split can't recover elements).

    Correct iff EVERY gold product is named AND NO other catalog product is —
    i.e. all gold present and no extras. "Extra" = a non-gold catalog descriptor
    whose normalized form appears in the answer and isn't merely a substring of a
    gold name already matched. Order-free, normalized (STT-robust).
    """
    na = normalize_name(answer)
    gold_norm = {normalize_name(g) for g in gold if normalize_name(g)}
    if not all(g in na for g in gold_norm):
        return False  # a gold element is missing
    for name in universe:
        nn = normalize_name(name)
        if not nn or nn in gold_norm:
            continue
        if any(nn in g for g in gold_norm):
            continue  # substring of a gold name we already matched — not a real extra
        if nn in na:
            return False  # named a product that isn't in gold → extra
    return True


def score_answer(
    answer: str, entry: dict, tolerances: dict, universe: list[str]
) -> dict:
    """Deterministic verdict for one item. Returns {decision, [list_f1]}."""
    if not answer.strip():
        return {"decision": "incorrect", "rationale": "No assistant answer recorded."}
    atype = entry.get("answer_type", "name")
    if atype == "list":
        gold = entry["answer"]
        exact = _list_exact_spoken(answer, gold, universe)
        return {
            "decision": "correct" if exact else "incorrect",
            "list_f1": score_list_f1(_parse_spoken_list(answer), gold),
        }
    # score_item expects the generator's item keys (gold_answer/number_kind);
    # the manifest renames gold_answer -> answer for the driver, so bridge here.
    item = {
        "answer_type": atype,
        "gold_answer": entry["answer"],
        "accepted_answers": entry.get("accepted_answers", []),
        "number_kind": entry.get("number_kind"),
    }
    correct = score_item(answer, item, tolerances)
    return {"decision": "correct" if correct else "incorrect"}


def run(modes: list[str] | None = None) -> dict:
    """Score all recorded runs and extract metrics. Returns results dict."""
    if not BENCHMARK_RUN_PATH.exists():
        raise SystemExit(
            f"{BENCHMARK_RUN_PATH} not found — run your driver first "
            f"(see example_driver.py)"
        )
    benchmark_run = orjson.loads(BENCHMARK_RUN_PATH.read_bytes())
    manifest = {
        str(entry["id"]): entry for entry in orjson.loads(MANIFEST_PATH.read_bytes())
    }
    tolerances = _CONFIG["scoring"]["number_tolerance"]
    # full product-name universe (for list exact-set: detect extras) — benchmark data.
    # Required: without it, list scoring can't detect extra products and turns
    # silently lenient, so a missing graph is a dataset error, not a soft-skip.
    graph_path = _PATHS["graph"]
    product_universe = [
        p["descriptor"]
        for p in orjson.loads(graph_path.read_bytes())["nodes"]["produkt"]
    ]

    mode_filter = set(modes) if modes else None
    results: dict = (
        orjson.loads(RESULTS_PATH.read_bytes()) if RESULTS_PATH.exists() else {}
    )
    stt_cache: dict = (
        orjson.loads(STT_CACHE_PATH.read_bytes()) if STT_CACHE_PATH.exists() else {}
    )

    for mode, runs in benchmark_run.items():
        if mode_filter and mode not in mode_filter:
            continue
        mode_results = results.setdefault(mode, {})
        for question_id, run_entry in runs.items():
            entry = manifest.get(question_id)
            if entry is None:
                print(f"[{mode}/{question_id}] not in manifest, skipping")
                continue
            if run_entry.get("status") != "ok":
                # category and speaker come along even though there is nothing to
                # grade: without them the item counts in the total but drops out
                # of every per-category and per-speaker breakdown, so the parts
                # stop summing to the whole.
                mode_results[question_id] = {
                    "status": run_entry.get("status"),
                    "decision": "incorrect",
                    "rationale": f"Run status: {run_entry.get('status')}",
                    "category": entry.get("category"),
                    "speaker": entry.get("speaker"),
                    "audio_source": run_entry.get("audio_source"),
                    "runtime_mode": run_entry.get("mode"),
                    "benchmark": run_entry.get("benchmark"),
                    "model_backend": run_entry.get("model_backend"),
                    "deployment": run_entry.get("deployment"),
                    "voice": run_entry.get("voice"),
                    "turn_detection": run_entry.get("turn_detection"),
                    "question": entry.get("prompt"),
                    "ground_truth": entry.get("answer"),
                }
                continue

            record_dir = session_dir(run_entry)
            realtime_answer, ttft = extract_answer_and_ttft(
                read_jsonl(record_dir / "transcript.jsonl")
            )
            # Grade the Whisper pass over the answer audio, so a system that
            # emits no transcript of its own is graded the same way. Fall back to
            # the system's transcript only if the audio is missing.
            stt_answer = whisper_answer(record_dir, run_entry, stt_cache)
            answer = stt_answer if stt_answer is not None else realtime_answer
            answer_source = "whisper" if stt_answer is not None else "realtime_fallback"
            timing_events = read_jsonl(record_dir / "timing.jsonl")
            metrics = extract_timing_metrics(timing_events)
            retrieved_ts = extract_retrieved_docs(
                read_jsonl(record_dir / "raw_realtime_events.jsonl"),
                timing_events,
            )
            doc_metrics = doc_retrieval_metrics(
                retrieved_ts, entry.get("gold_documents", []), metrics.get("eos_ts")
            )
            verdict = score_answer(answer, entry, tolerances, product_universe)
            mode_results[question_id] = {
                "status": "ok",
                "question": entry["prompt"],
                "ground_truth": entry["answer"],
                "answer_type": entry.get("answer_type"),
                "category": entry.get("category"),
                "speaker": entry.get("speaker"),
                "audio_source": run_entry.get("audio_source"),
                "runtime_mode": run_entry.get("mode"),
                "benchmark": run_entry.get("benchmark"),
                "model_backend": run_entry.get("model_backend"),
                "deployment": run_entry.get("deployment"),
                "voice": run_entry.get("voice"),
                "turn_detection": run_entry.get("turn_detection"),
                "model_answer": answer,
                "answer_source": answer_source,
                # verbosity covariate: lets analyze.py check whether a mode's
                # accuracy correlates with how MUCH it says (a verbose mode gets
                # more chances for the scorer to find the gold token in passing).
                # Scoring is unchanged; this only makes the (non-)bias auditable.
                "answer_word_count": len(answer.split()),
                "realtime_transcript": realtime_answer,
                **verdict,
                "ttft_s": ttft,
                **metrics,
                **doc_metrics,
                "session_id": run_entry["session_id"],
                "run_id": run_entry["run_id"],
            }
            f1 = verdict.get("list_f1", {}).get("f1")
            print(
                f"[{mode}/{question_id}] {verdict['decision']}"
                f"{f' f1={f1}' if f1 is not None else ''} "
                f"(ttft={ttft}, ttfa={metrics['ttfa_s']}, "
                f"tools during/after={metrics['tools_during_listening']}"
                f"/{metrics['tools_after_listening']}, "
                f"doc_prec={doc_metrics['doc_precision']}, "
                f"doc_recall={doc_metrics['doc_recall']})"
            )

    RESULTS_DIR.mkdir(exist_ok=True)
    RESULTS_PATH.write_bytes(orjson.dumps(results, option=orjson.OPT_INDENT_2))
    STT_CACHE_PATH.write_bytes(orjson.dumps(stt_cache, option=orjson.OPT_INDENT_2))
    print(f"\nWrote {RESULTS_PATH}")
    for mode, mode_results in results.items():
        judged = [r for r in mode_results.values() if r.get("decision")]
        correct = sum(1 for r in judged if r["decision"] == "correct")
        print(f"  {mode}: accuracy {correct}/{len(judged)}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes", default=None, help="Comma-separated mode filter (default: all)"
    )
    args = parser.parse_args()
    run(modes=args.modes.split(",") if args.modes else None)


if __name__ == "__main__":
    main()
