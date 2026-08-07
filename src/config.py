"""Load config.yaml and resolve data paths relative to the gems-bench dir."""

from __future__ import annotations

from pathlib import Path

import yaml

GEMS_DIR = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = "config.yaml") -> dict:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = GEMS_DIR / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # Resolve paths relative to gems-bench/ and expose absolute helpers.
    paths = cfg["paths"]

    def resolve(name: str) -> Path:
        return (GEMS_DIR / paths[name]).resolve()

    data_dir = resolve("data_dir")
    raw_dir = resolve("raw_dir")
    audio_dirs = {
        source: (GEMS_DIR / directory).resolve()
        for source, directory in cfg["audio"]["directories"].items()
    }
    audio_source = cfg["audio"]["source"]
    if audio_source not in audio_dirs:
        choices = ", ".join(sorted(audio_dirs))
        raise ValueError(f"audio.source must be one of: {choices}")
    cfg["_paths"] = {
        "gems_dir": resolve("gems_dir"),
        "repo_root": resolve("repo_root"),
        "data_dir": data_dir,
        "raw_dir": raw_dir,  # stage 0 (fetch): raw API cache, one file per query
        "products": data_dir / "stage_1_products.json",
        "names": data_dir / "stage_2_names.json",
        "graph": resolve("graph"),
        "corpus": resolve("corpus"),
        "corpus_cache": data_dir / "stage_4_corpus_cache.json",
        "questions_raw": data_dir / "stage_5_questions_raw.json",
        # stage 6: raw questions + spoken_question (rewrite runs BEFORE the gate)
        "questions_spoken": data_dir / "stage_6_questions_spoken.json",
        "questions": resolve("questions"),  # stage 7: gate-passed final set
        # stage 7 in-progress checkpoint (item_id -> hits|null); lets a crashed
        # gate run resume instead of re-paying for already-graded items
        "gate_checkpoint": data_dir / "stage_7_gate_checkpoint.json",
        # stage 8: human-recording gate verdicts (item_id -> checks|transcript);
        # also serves as the recording-STT cache (Whisper is slow, re-runs reuse it)
        "recording_verification": data_dir / "stage_8_recording_verification.json",
        "audio_dirs": audio_dirs,
        "audio_dir": audio_dirs[audio_source],
        "manifest": resolve("manifest"),
        # prompts.yaml: system prompt + tool descriptions the model under test reads
        "prompts": resolve("prompts"),
        "results_dir": resolve("results_dir"),
        "benchmark_run": resolve("benchmark_run"),
        "results": resolve("results"),
        "plots_dir": resolve("plots_dir"),
        "runner_output_dir": resolve("runner_output_dir"),
    }
    return cfg
