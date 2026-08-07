"""Open-source speech-to-text for offline scoring.

Runs Whisper large-v3 on Apple Silicon via ``mlx-whisper`` (Metal GPU, no cloud,
no proprietary model). Used to transcribe the assistant's answer audio
(``assistant_audio.wav``, 24 kHz mono PCM16) to German text for offline scoring,
to verify the human recordings, and to round-trip-check every synthesized WAV.

Every offline scoring path uses this STT, so grading a run needs no cloud service
and no credentials.

Long audio is cut at real pauses before decoding rather than handed to
Whisper as one long-form window: its 30 s window can collapse on monotone
enumerations, emitting one opening sentence and dropping the rest. The
segmentation knobs live in config.yaml under ``stt.long_form``.

Every setting is a required argument read from ``cfg["stt"]`` by the caller — the
model name and language included. There are deliberately no module-level defaults,
so a missing config key fails loudly instead of falling back to a hidden value.

The model is downloaded from the Hugging Face hub on first use and cached; MLX is
imported on first transcription only, so importing this module stays cheap.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np

# Whisper's own front end resamples to 16 kHz; the silence scan and the piece
# boundaries below are computed in that same rate. Protocol-fixed by the model.
_WHISPER_RATE_HZ = 16000


def _decode(audio: Any, model: str, language: str) -> str:
    import mlx_whisper  # lazy: avoids MLX import at module load

    result = mlx_whisper.transcribe(audio, path_or_hf_repo=model, language=language)
    return str(result.get("text", "")).strip()


def _read_wav_16k(path: str | Path) -> np.ndarray:
    """Mono float32 samples at Whisper's 16 kHz, from any PCM16 WAV."""
    with wave.open(str(path), "rb") as source:
        rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return _to_16k(samples, rate)


def _to_16k(samples: np.ndarray, rate: int) -> np.ndarray:
    if rate == _WHISPER_RATE_HZ or samples.size == 0:
        return samples
    target = int(round(samples.size / rate * _WHISPER_RATE_HZ))
    if target <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.interp(
        np.linspace(0.0, samples.size, target, endpoint=False),
        np.arange(samples.size),
        samples,
    ).astype(np.float32)


def _pause_centres(
    samples: np.ndarray, *, drop_db: float, window_ms: int
) -> list[tuple[int, float]]:
    """(centre sample index, duration in seconds) for every silence in `samples`.

    Silence is relative to the clip's own peak: the human takes peak around
    -14 dBFS and the TTS output around -23, so an absolute floor would call one
    source's speech the other's silence.
    """
    window = max(1, int(_WHISPER_RATE_HZ * window_ms / 1000))
    if samples.size < window * 2:
        return []
    usable = samples[: samples.size - samples.size % window].reshape(-1, window)
    rms = np.sqrt((usable.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    db = 20.0 * np.log10(rms)
    quiet = db < (db.max() - drop_db)

    centres: list[tuple[int, float]] = []
    run = 0
    for index, is_quiet in enumerate(quiet):
        if is_quiet:
            run += 1
            continue
        if run:
            centre = int((index - run / 2.0) * window)
            centres.append((centre, run * window / _WHISPER_RATE_HZ))
            run = 0
    if run:
        centre = int((len(quiet) - run / 2.0) * window)
        centres.append((centre, run * window / _WHISPER_RATE_HZ))
    return centres


def _split_on_pauses(samples: np.ndarray, long_form: dict) -> list[np.ndarray]:
    """Cut `samples` into <= max_s pieces, always inside a pause.

    Walks forward greedily: from the current start, take the LAST legal pause that
    still fits inside max_s, so pieces stay as long as the cap allows and every
    boundary lands in silence rather than mid-word. If a stretch has no qualifying
    pause the piece is cut at max_s anyway — a hard cut costs one word boundary,
    while an over-long window costs half the transcript.
    """
    max_samples = int(long_form["max_s"] * _WHISPER_RATE_HZ)
    if samples.size <= max_samples:
        return [samples]

    min_gap = long_form["min_gap_s"]
    min_piece = int(long_form["min_piece_s"] * _WHISPER_RATE_HZ)
    pauses = [
        centre
        for centre, length in _pause_centres(
            samples,
            drop_db=long_form["silence_drop_db"],
            window_ms=long_form["window_ms"],
        )
        if length >= min_gap
    ]

    pieces: list[np.ndarray] = []
    start = 0
    while start < samples.size:
        limit = start + max_samples
        if limit >= samples.size:
            pieces.append(samples[start:])
            break
        fits = [p for p in pauses if start + min_piece < p <= limit]
        end = fits[-1] if fits else limit
        pieces.append(samples[start:end])
        start = end
    return pieces


def _transcribe_samples(
    samples: np.ndarray, *, model: str, language: str, long_form: dict
) -> str:
    pieces = _split_on_pauses(samples, long_form)
    if len(pieces) == 1:
        return _decode(pieces[0], model, language)
    parts = [_decode(piece, model, language) for piece in pieces]
    return " ".join(part for part in parts if part)


def transcribe_wav(
    path: str | Path, *, model: str, language: str, long_form: dict
) -> str:
    """Transcribe a WAV file to text. Slowness is fine (offline scoring)."""
    return _transcribe_samples(
        _read_wav_16k(path), model=model, language=language, long_form=long_form
    )


def transcribe_pcm16(
    data: bytes,
    sample_rate: int,
    *,
    model: str,
    language: str,
    long_form: dict,
) -> str:
    """Transcribe raw mono PCM16 audio bytes to text."""
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return _transcribe_samples(
        _to_16k(samples, sample_rate),
        model=model,
        language=language,
        long_form=long_form,
    )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import load_config

    target = sys.argv[1] if len(sys.argv) > 1 else None
    if not target:
        raise SystemExit("usage: python -m src.oss_stt <wav_path>")
    stt = load_config()["stt"]
    print(
        transcribe_wav(
            target,
            model=stt["model"],
            language=stt["language"],
            long_form=stt["long_form"],
        )
    )
