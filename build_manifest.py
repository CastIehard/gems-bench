"""Build manifest.json for the run layer from the final stage_7_questions.json.

Emits the keys the run layer and the scorer consume. Both audio channels are
listed per item; the consumer selects one at run time via config `audio.source`,
so changing channel needs no rebuild:
    id, prompt, audio_files, audio_durations_s                (run layer)
  answer, accepted_answers, answer_type, number_kind, category, gold_documents
                                             (deterministic scorer)

Usage:
    python build_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import load_config  # noqa: E402


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def write_provenance(cfg: dict) -> Path:
    """Stamp what produced this benchmark so the committed stages can be
    verified as mutually consistent. GEMS is ARTIFACT-reproducible, not
    generation-reproducible (hosted OSS endpoints are not weight-pinned) —
    this records the exact inputs behind the frozen files.

    Writes data/provenance.json: config hash, git commit, seed, generation +
    gate model ids, and a sha256 per stage artifact."""
    paths = cfg["_paths"]
    gems_dir = paths["gems_dir"]
    stage_files = {
        name: paths[name]
        for name in (
            "products",
            "names",
            "graph",
            "corpus",
            "questions_raw",
            "questions_spoken",
            "questions",
        )
        if name in paths
    }
    prov = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(paths["repo_root"]),
        "config_sha256": _sha256(gems_dir / "config.yaml"),
        "seed": cfg["seed"],
        "gen_model": cfg["llm"]["model"],
        "gate_models": cfg["qa"]["closed_book_gate"]["models"],
        "stage_sha256": {name: _sha256(p) for name, p in stage_files.items()},
    }
    out_path = paths["data_dir"] / "provenance.json"
    out_path.write_text(
        json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


def _duration_s(gems_dir: Path, audio_file: str) -> float | None:
    path = gems_dir / audio_file
    if not path.is_file():
        return None
    with wave.open(str(path), "rb") as wav:
        return round(wav.getnframes() / wav.getframerate(), 3)


def run(cfg: dict | None = None) -> list[dict]:
    cfg = cfg or load_config()
    paths = cfg["_paths"]
    channels = list(cfg["audio"]["directories"])
    gems_dir = paths["gems_dir"]
    manifest_path = paths["manifest"]
    questions_path = paths["questions"]
    if not questions_path.exists():
        raise SystemExit(
            f"{questions_path} missing — run the generator (stage 6) first"
        )
    items = json.loads(questions_path.read_text(encoding="utf-8"))
    manifest: list[dict] = []
    missing = {channel: 0 for channel in channels}
    for it in items:
        # BOTH channels are listed. The consumer picks one at run time from
        # `audio.source`, so switching channels needs no rebuild and the manifest
        # never hides the fact that the other channel exists.
        available = it.get("audio_files", {})
        audio_files = {c: available.get(c) for c in channels}
        durations = {
            c: _duration_s(gems_dir, rel) if rel else None
            for c, rel in audio_files.items()
        }
        for c, d in durations.items():
            if d is None:
                missing[c] += 1
        manifest.append(
            {
                "id": it["item_id"],
                "category": it["category"],
                # which of the two humans read this item (None until the
                # recorder has assigned speakers — see recording.speakers)
                "speaker": it.get("speaker"),
                "prompt": it["question_text"],
                "spoken": it.get("spoken_question", it["question_text"]),
                "audio_files": audio_files,
                "audio_durations_s": durations,
                "answer": it["gold_answer"],
                "accepted_answers": it.get("accepted_answers", []),
                "answer_type": it.get("answer_type", "name"),
                "number_kind": it.get("number_kind"),
                "gold_documents": it.get("gold_documents", []),
            }
        )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    have = ", ".join(f"{c}={len(manifest) - n}/{len(manifest)}" for c, n in missing.items())
    print(f"manifest: {len(manifest)} items, audio {have} -> {manifest_path}")
    for channel, n in missing.items():
        if n:
            print(f"  WARNING: {n} items have no {channel} audio yet")
    prov_path = write_provenance(cfg)
    print(f"provenance -> {prov_path}")
    return manifest


if __name__ == "__main__":
    run()
