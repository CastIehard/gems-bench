"""Stage 8 — synthesize each question to speech (Audio8-TTS-Preview-0.6b, OSS TTS).

IN : data/stage_7_questions.json  (uses `spoken_question`, fallback `question_text`)
OUT: audio_synthetic/*.wav  +  data/stage_7_questions.json

Synthesizes every item to a WAV under audio_synthetic/, records the
relative path in each item's `audio_files.synthetic`. Model loads once;
existing WAVs skipped unless --rerun. No reference-audio conditioning —
each item gets the model's own default voice.

torch/transformers are imported lazily so importing this module (e.g. in
the notebook) never fails on machines without them. Install:
`uv pip install "torch>=2.5.0" "torchaudio>=2.5.0" "transformers>=4.57.0,<5" "soundfile>=0.12" "safetensors>=0.4"`.
transformers MUST stay below 5 — 5.x produces invalid all-zero codes for
this custom-code model (see https://github.com/Audio8-AI/Audio8_TTS
troubleshooting section).

If the model doesn't reach end-of-speech within `max_new_tokens` (an
occasional failure mode of this preview checkpoint), the item is retried
once with `retry_max_new_tokens`.

EVERY generated WAV is verified by ASR round-trip (`tts.verify`), because this
checkpoint fails in two ways a token budget cannot catch: premature
end-of-speech, where it reports `finished` after only a fraction of the text,
and babble, where the audio runs a plausible length but its content does not
match the text. Both produce a file that looks healthy on disk; only the
round-trip catches them. A rejected draw is re-sampled, and if every attempt
fails NO file is written — a missing WAV is retried next run, a truncated one
would silently corrupt the benchmark.

Failure risk grows with text length, and re-sampling or lowering temperature
cannot rescue a draw past a capability limit. Text longer than
`long_form.max_chars` is therefore synthesized SEGMENT BY SEGMENT and
concatenated with a short silence: each segment is short enough to synthesize
reliably and is verified on its own, so a bad draw costs one short
re-synthesis instead of the whole utterance. Segmentation is lossless — the
segments rejoin to exactly the original text.

Usage:
    python src/oss_tts.py [--rerun] [--limit N]
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path

from tqdm.auto import tqdm

from common import distinctive_tokens, read_json, write_json
from config import load_config
from scoring import normalize_name

_MODEL = None
_PROCESSOR = None


def _get_model(model_id: str):
    """Load the TTS model once (cached across calls)."""
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        import torch
        import transformers
        from transformers import AutoModel, AutoProcessor

        if int(transformers.__version__.split(".")[0]) >= 5:
            raise SystemExit(
                f"transformers {transformers.__version__} unsupported — need <5 "
                '(pip install "transformers>=4.57.0,<5") — 5.x produces invalid '
                "all-zero codes for this model"
            )
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"loading TTS model {model_id} on {device} ...")
        _PROCESSOR = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        _MODEL = (
            AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=dtype)
            .eval()
            .to(device)
        )
    return _MODEL, _PROCESSOR


class SynthesisRejected(Exception):
    """Every attempt for one item failed ASR verification — no WAV was written."""


def word_coverage(expected: str, transcript: str) -> float:
    """Fraction of the expected distinctive words recoverable from an ASR
    transcript of the generated audio.

    Word-based, not duration-based: duration alone cannot see the babble failure
    mode, where the audio runs its expected length but says something else.

    Matching is substring containment against the transcript stripped down to its
    letters, because German compounding is not a miss: the recognizer writes
    "Aftershave-Stick mit Kirschblütenextrakt" for "After Shave Stick mit
    Kirschblüten-Extrakt", and exact token equality would score that as a miss."""
    flat = re.sub(r"[^a-z]", "", normalize_name(transcript))
    want = [w for w in (normalize_name(t) for t in distinctive_tokens(expected)) if w]
    if not want:
        return 1.0
    return sum(1 for w in want if w in flat) / len(want)


def segments(text: str, max_chars: int) -> list[str]:
    """Cut `text` into pieces short enough for the checkpoint to render.

    Splits at the separators a human reading the same enumeration would pause at:
    "; " between products first, then ", ", then word boundaries as a last resort.
    Delimiters stay attached to the piece before them, so the segments rejoin to
    exactly the input — segmentation changes where the pauses fall, never the
    words."""
    if len(text) <= max_chars:
        return [text]
    units = [u for u in re.split(r"(?<=; )|(?<=, )", text) if u]
    packed: list[str] = []
    for unit in units:
        while len(unit) > max_chars:  # single product longer than a segment
            cut = unit.rfind(" ", 0, max_chars)
            cut = cut + 1 if cut > 0 else max_chars
            packed.append(unit[:cut])
            unit = unit[cut:]
        if packed and len(packed[-1]) + len(unit) <= max_chars:
            packed[-1] += unit
        else:
            packed.append(unit)
    assert "".join(packed) == text, "segmentation must not alter the text"
    return packed


def _generate_waveform(cfg: dict, text: str):
    """One synthesis draw. Returns (waveform, sample_rate). Escalates the token
    budget once if the model does not reach end-of-speech."""
    import torch

    t = cfg["tts"]
    model, processor = _get_model(t["model_id"])
    inputs = processor(text=[text], return_tensors="pt")
    inputs = {name: value.to(model.device) for name, value in inputs.items()}

    finished = False
    for max_new_tokens in (t["max_new_tokens"], t["retry_max_new_tokens"]):
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=t["temperature"],
                top_p=t["top_p"],
                top_k=t["top_k"],
                return_dict_in_generate=True,
            )
        finished = bool(output.finished[0])
        if finished:
            break
    if not finished:
        print(
            f"  WARNING: no end-of-speech within {t['retry_max_new_tokens']} tokens "
            f"for {text[:40]!r}"
        )

    waveforms, lengths = model.decode_audio(output.codes)
    return (
        waveforms[0, : lengths[0]].float().cpu().numpy(),
        int(model.config.codec_sample_rate),
    )


def _write_wav(waveform, sample_rate: int, out_path) -> None:
    import soundfile as sf

    # format explicit: draft path is "<name>.wav.tmp", so the ".tmp" extension
    # gives soundfile no format to infer.
    sf.write(out_path, waveform, samplerate=sample_rate, subtype="PCM_16", format="WAV")


def _generate_wav(cfg: dict, text: str, out_path) -> None:
    """One single-pass synthesis attempt written to `out_path`."""
    waveform, sample_rate = _generate_waveform(cfg, text)
    _write_wav(waveform, sample_rate, out_path)


def _verified_segment(cfg: dict, text: str, scratch):
    """Synthesize one segment, re-sampling until its own ASR round-trip passes.

    Verifying per segment rather than only the joined result means a bad draw costs
    one short re-synthesis instead of the whole utterance."""
    import oss_stt

    verify = cfg["tts"]["verify"]
    best = 0.0
    for attempt in range(1, verify["max_attempts"] + 1):
        waveform, sample_rate = _generate_waveform(cfg, text)
        if not verify["enabled"]:
            return waveform, sample_rate
        _write_wav(waveform, sample_rate, scratch)
        stt = cfg["stt"]
        coverage = word_coverage(
            text,
            oss_stt.transcribe_wav(
                scratch,
                model=stt["model"],
                language=verify["stt_language"],
                long_form=stt["long_form"],
            ),
        )
        best = max(best, coverage)
        if coverage >= verify["min_word_coverage"]:
            return waveform, sample_rate
        print(
            f"    segment attempt {attempt}/{verify['max_attempts']} "
            f"{coverage:.0%}: {text[:48]!r}"
        )
    raise SynthesisRejected(f"segment never passed ({best:.0%} best): {text[:60]!r}")


def _generate_segmented_wav(cfg: dict, text: str, out_path) -> None:
    """Synthesize a long text segment by segment and concatenate with silence."""
    import numpy as np

    long_form = cfg["tts"]["long_form"]
    parts = segments(text, long_form["max_chars"])
    waveforms, sample_rate = [], None
    # The per-segment scratch file lives OUTSIDE the audio directory so a killed
    # run cannot leave a half-question WAV where the next run would pick it up
    # as the finished item.
    with tempfile.TemporaryDirectory() as tmp_dir:
        scratch = Path(tmp_dir) / "segment.wav"
        for part in parts:
            waveform, sample_rate = _verified_segment(cfg, part, scratch)
            waveforms.append(waveform)
    gap = np.zeros(int(sample_rate * long_form["gap_s"]), dtype=waveforms[0].dtype)
    joined = waveforms[0]
    for waveform in waveforms[1:]:
        joined = np.concatenate([joined, gap, waveform])
    _write_wav(joined, sample_rate, out_path)


def synthesize(cfg: dict, text: str, out_path) -> float:
    """Generate one verified WAV for `text`. Returns the accepted ASR coverage.

    Re-samples while the ASR round-trip says the audio does not carry the text.
    Raises SynthesisRejected (leaving no file behind) if every attempt fails."""
    verify = cfg["tts"]["verify"]
    # Above long_form.max_chars a single pass fails by capability, not by luck,
    # so those are built from verified segments instead of re-rolled whole.
    generate = (
        _generate_segmented_wav
        if len(text) > cfg["tts"]["long_form"]["max_chars"]
        else _generate_wav
    )
    if not verify["enabled"]:
        generate(cfg, text, out_path)
        return 1.0

    import oss_stt

    minimum = verify["min_word_coverage"]
    attempts = verify["max_attempts"]
    best = 0.0
    # Build into a sidecar and move it into place only once it has passed: writing
    # out_path first and verifying afterwards means an interrupted run leaves an
    # UNVERIFIED file that the next run happily treats as finished. The ".tmp"
    # name cannot be mistaken for the item's audio by the caller's exact-name check
    # nor by an `audio_synthetic/*.wav` listing, and it sits in the same directory
    # so the move is atomic.
    draft = out_path.with_name(out_path.name + ".tmp")
    try:
        for attempt in range(1, attempts + 1):
            generate(cfg, text, draft)
            stt = cfg["stt"]
            transcript = oss_stt.transcribe_wav(
                draft,
                model=stt["model"],
                language=verify["stt_language"],
                long_form=stt["long_form"],
            )
            coverage = word_coverage(text, transcript)
            best = max(best, coverage)
            if coverage >= minimum:
                os.replace(draft, out_path)
                return coverage
            print(
                f"  REJECTED {out_path.name} attempt {attempt}/{attempts}: ASR "
                f"coverage {coverage:.0%} < {minimum:.0%} — re-sampling"
            )
    finally:
        draft.unlink(missing_ok=True)
    # Leave nothing behind: a missing WAV is retried on the next run, a rejected
    # one that stayed on disk would be mistaken for a finished item.
    out_path.unlink(missing_ok=True)
    raise SynthesisRejected(
        f"{out_path.name}: {attempts} attempts, best ASR coverage {best:.0%} "
        f"< {minimum:.0%} — no file written"
    )


def run(cfg: dict, *, rerun: bool = False, limit: int | None = None) -> list[dict]:
    questions_path = cfg["_paths"]["questions"]
    if not questions_path.exists():
        raise SystemExit(f"{questions_path} missing — run qa_checks.py first")
    items = read_json(questions_path)
    source = "synthetic"
    audio_dir = cfg["_paths"]["audio_dirs"][source]
    relative_dir = cfg["audio"]["directories"][source]
    audio_dir.mkdir(parents=True, exist_ok=True)
    field = cfg["tts"]["use_field"]
    fmt = cfg["tts"]["audio_format"]

    targets = items if limit is None else items[:limit]
    done = 0
    rejected: list[str] = []
    for it in tqdm(targets, desc="synthesize TTS", unit="q"):
        audio_files = it.setdefault("audio_files", {})
        # Match the output path EXACTLY — a glob would also match sidecar files
        # such as a leftover `<id>.segment.wav` and could point the item at a
        # fragment of its own question.
        out_path = audio_dir / f"{it['item_id']}.{fmt}"
        if out_path.is_file() and not rerun:
            audio_files[source] = f"{relative_dir}/{out_path.name}"
            continue
        text = it.get(field) or it["question_text"]
        try:
            coverage = synthesize(cfg, text, out_path)
        except SynthesisRejected as exc:
            # One stubborn item must not discard the rest of a long local run; the
            # failure is still fatal at the end, and the missing file makes the next
            # run retry exactly this item.
            print(f"  FAILED {exc}")
            rejected.append(it["item_id"])
            audio_files.pop(source, None)
            continue
        audio_files[source] = f"{relative_dir}/{out_path.name}"
        done += 1
        print(
            f"  [{done}] {it['item_id']} -> {audio_files[source]} (ASR {coverage:.0%})"
        )

    write_json(cfg["_paths"]["questions"], items)
    cached = len(targets) - done - len(rejected)
    print(f"TTS: {done} synthesized, {cached} cached -> {audio_dir}")
    if rejected:
        raise SystemExit(
            f"TTS: {len(rejected)} items failed ASR verification and have NO audio: "
            f"{', '.join(rejected)} — re-run to retry them"
        )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun", action="store_true", help="re-synthesize existing WAVs"
    )
    parser.add_argument("--limit", type=int, help="only first N items (debug)")
    args = parser.parse_args()
    run(load_config(), rerun=args.rerun, limit=args.limit)


if __name__ == "__main__":
    main()
