"""Stage 9 — verify HUMAN recordings read the question faithfully.

Humans read `spoken_question` VERBATIM — the same wording the TTS path voices, so
the synthetic and human audio channels differ only in voice. Verbatim does not
mean trustworthy: a misread, a swallowed product name, or an unintelligible take
would silently break the deterministic gold, so every recording is checked here:

  1. entity_presence — every skeleton entity appears in the transcript (token
     coverage, umlaut-folded, STT-tolerant). The core fidelity check: it catches a
     dropped or misread entity, which leaves the item underspecified.
  2. gold_uniqueness — the gold answer + gold documents still resolve (structural
     re-assert; the recording never changes the gold, this just guards it).
  3. window (soft, warning only) — the skeleton entities are named within the
     first `window_front_fraction` of the transcript, i.e. front-loaded, so an
     agent that retrieves while listening has a window to work in.

Transcription uses the local Whisper build (src/oss_stt.py) — no cloud, no
credentials. Verdicts + transcripts cache to
stage_8_recording_verification.json (Whisper is slow; re-runs reuse the cache).
Failed items get re-recorded (recorder auto-requeue, or `record_audio.py
--rerun-failed`).

Usage:
    python src/verify_recording.py                  # verify all recorded items
    python src/verify_recording.py --items id1,id2   # only these items
    python src/verify_recording.py --rerun           # ignore cached transcripts
"""

from __future__ import annotations

import argparse

import oss_stt
import scoring
from common import (
    distinctive_tokens,
    entity_present,
    read_json,
    read_jsonl,
    write_json,
)
from config import load_config


def _entity_first_word_index(
    entity: str, words_norm: list[str], ratio: float
) -> int | None:
    """Earliest word index in the transcript where a distinctive entity token
    appears (for the front-loading window check). None if not found."""
    toks = [scoring.normalize_name(t) for t in distinctive_tokens(entity)]
    toks = [t for t in toks if t]
    for i, w in enumerate(words_norm):
        for t in toks:
            if t in w or scoring._fuzzy_contains(t, w, ratio):
                return i
    return None


def _structural_ok(item: dict, doc_ids: set[str]) -> bool:
    """Gold answer present + all gold documents resolve (judge-free uniqueness)."""
    if item.get("gold_answer") in ("", None, []):
        return False
    return all(d in doc_ids for d in item.get("gold_documents", []))


def verify_item(cfg, item, *, doc_ids: set[str], cache: dict, rerun: bool) -> dict:
    """Run all checks for one recorded item. Returns a verdict dict."""
    gate_cfg = cfg["recording"]["gate"]
    ratio = gate_cfg["entity_fuzzy_ratio"]
    min_cov = gate_cfg["entity_min_coverage"]
    tok_cov = gate_cfg["entity_token_coverage"]
    front = gate_cfg["window_front_fraction"]
    item_id = item["item_id"]

    # --- transcript (cached; Whisper is slow) ---
    audio_rel = item.get("audio_files", {}).get("real")
    if not audio_rel:
        return {"item_id": item_id, "passed": False, "reason": "no real recording"}
    audio_path = cfg["_paths"]["gems_dir"] / audio_rel
    if not audio_path.is_file():
        return {"item_id": item_id, "passed": False, "reason": f"missing {audio_rel}"}

    cached = cache.get(item_id)
    if cached and not rerun and cached.get("transcript"):
        transcript = cached["transcript"]
    else:
        stt = cfg["stt"]
        transcript = oss_stt.transcribe_wav(
            audio_path,
            model=stt["model"],
            language=gate_cfg["stt_language"],
            long_form=stt["long_form"],
        )

    transcript_norm = scoring.normalize_name(transcript)
    words = transcript.split()
    words_norm = [scoring.normalize_name(w) for w in words]

    checks: dict = {}

    # --- 1. entity presence ---
    entities = item.get("skeleton", {}).get("entities", [])
    present = [
        e for e in entities if entity_present(e, transcript_norm, ratio, tok_cov)
    ]
    missing = [e for e in entities if e not in present]
    coverage = len(present) / len(entities) if entities else 1.0
    checks["entity_presence"] = {
        "coverage": round(coverage, 3),
        "missing": missing,
        "passed": coverage >= min_cov,
    }

    # --- 2. gold uniqueness (structural) ---
    checks["gold_uniqueness"] = {"passed": _structural_ok(item, doc_ids)}

    # --- 3. window (soft) ---
    if entities and words:
        idxs = [_entity_first_word_index(e, words_norm, ratio) for e in entities]
        idxs = [i for i in idxs if i is not None]
        last_entity_pos = (max(idxs) / len(words)) if idxs else 1.0
        checks["window"] = {
            "last_entity_fraction": round(last_entity_pos, 3),
            "front_loaded": last_entity_pos <= front,
            "passed": True,  # soft: warning only
        }

    passed = all(c.get("passed", True) for c in checks.values())
    reasons = [k for k, c in checks.items() if not c.get("passed", True)]
    verdict = {
        "item_id": item_id,
        "category": item.get("category"),
        "passed": passed,
        "failed_checks": reasons,
        "transcript": transcript,
        "checks": checks,
    }

    # An audited manual pass overrides a failing automatic check, but never hides
    # it: the original verdict stays visible under `manual_override` so a reviewer
    # can see which items were accepted by ear and on what grounds.
    override = gate_cfg["manual_pass"].get(item_id)
    if override and not passed:
        verdict["passed"] = True
        verdict["failed_checks"] = []
        verdict["manual_override"] = {
            "reason": override,
            "overridden_checks": reasons,
        }
    return verdict


def run(
    cfg: dict,
    *,
    item_ids: list[str] | None = None,
    rerun: bool = False,
) -> dict:
    questions_path = cfg["_paths"]["questions"]
    if not questions_path.exists():
        raise SystemExit(f"{questions_path} missing — run qa_checks.py first")
    items = read_json(questions_path)
    if item_ids:
        wanted = set(item_ids)
        items = [it for it in items if it["item_id"] in wanted]

    # Recording runs in sessions, so this is normally called with the set only
    # partly recorded. Items with no WAV yet are NOT failures — skip them, or a
    # mid-session verify would stamp every unrecorded item as failed and the
    # recorder's progress summary would count it.
    gems_dir = cfg["_paths"]["gems_dir"]
    items = [
        it
        for it in items
        if (rel := it.get("audio_files", {}).get("real")) and (gems_dir / rel).is_file()
    ]

    corpus_path = cfg["_paths"]["corpus"]
    doc_ids = (
        {d["doc_id"] for d in read_jsonl(corpus_path)}
        if corpus_path.exists()
        else set()
    )

    out_path = cfg["_paths"]["recording_verification"]
    cache = read_json(out_path) if out_path.exists() else {}

    verdicts: dict = dict(cache)
    n_pass = n_fail = 0
    for item in items:
        verdict = verify_item(cfg, item, doc_ids=doc_ids, cache=cache, rerun=rerun)
        verdicts[item["item_id"]] = verdict
        write_json(out_path, verdicts)  # checkpoint each item (STT is expensive)
        status = (
            "ok"
            if verdict["passed"]
            else f"FAIL {verdict.get('failed_checks') or verdict.get('reason')}"
        )
        print(f"[{item['item_id']}] {status}")
        n_pass += int(verdict["passed"])
        n_fail += int(not verdict["passed"])

    print(
        f"\nverified {n_pass + n_fail}: {n_pass} passed, {n_fail} failed -> {out_path}"
    )
    if n_fail:
        failed = [v["item_id"] for v in verdicts.values() if not v["passed"]]
        print(f"  re-record: {','.join(failed)}")
    return verdicts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items", default=None, help="comma-separated item ids (default: all)"
    )
    parser.add_argument(
        "--rerun", action="store_true", help="ignore cached transcripts"
    )
    args = parser.parse_args()
    run(
        load_config(),
        item_ids=args.items.split(",") if args.items else None,
        rerun=args.rerun,
    )


if __name__ == "__main__":
    main()
