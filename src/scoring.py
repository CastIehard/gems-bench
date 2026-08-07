"""Deterministic scorers — no LLM judge.

- name:   normalized exact-match against gold + accepted_answers
- number: rounded, correct within a per-attribute tolerance band
- list:   exact set (all gold elements, no extras), order-free
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

# config.yaml is the single source of truth (no magic numbers). Dual-context
# import: this module is loaded both as `scoring` (src on path) and `src.scoring`
# (judge, root on path).
try:
    from config import load_config
except ImportError:  # pragma: no cover
    from src.config import load_config

# Name-match knobs (config.yaml scoring.name). Grading runs on an STT transcript
# of the answer audio, which mis-spells invented compound names ("Logistik"->
# "Lobistik", "&"->"und"):
#   fuzzy_ratio     min SequenceMatcher ratio for a fuzzy hit on invented names
#   fuzzy_min_len   below this, no fuzzy (a near-match could be a different short entity)
#   contain_min_len below this, no substring containment ("berg" ⊂ "annaberg")
_NAME_CFG = load_config()["scoring"]["name"]
_FUZZY_NAME_RATIO = _NAME_CFG["fuzzy_ratio"]
_FUZZY_MIN_LEN = _NAME_CFG["fuzzy_min_len"]
_CONTAIN_MIN_LEN = _NAME_CFG["contain_min_len"]


_UMLAUT_FOLD = str.maketrans({"ä": "a", "ö": "o", "ü": "u", "ß": "ss"})


def normalize_name(s: str) -> str:
    """STT-robust match key: fold German umlauts/ß and keep only [a-z0-9].

    Grading runs on a Whisper pass over the answer audio, so a system that
    returns audio only is graded like any other. STT renders the
    invented compound names with inconsistent spacing, hyphens and punctuation
    ("Stahlstrom Logistik" -> "Stahlstromlogistik", "Nordtor" -> "Nord-Tor"),
    which are the SAME entity. Dropping everything but alphanumerics (plus umlaut
    folding) makes the deterministic match insensitive to that variance without
    resorting to an LLM judge."""
    s = unicodedata.normalize("NFC", s or "").lower()
    s = s.translate(_UMLAUT_FOLD)
    return re.sub(r"[^a-z0-9]+", "", s)


def _fuzzy_contains(gold: str, pred: str, ratio: float = _FUZZY_NAME_RATIO) -> bool:
    """True if `gold` fuzzily appears in `pred` (best same-length window ratio).

    Tolerates single-token STT errors on invented names. Skipped for short names
    (< _FUZZY_MIN_LEN) where a near-match could be a different short entity.
    """
    n = len(gold)
    if n < _FUZZY_MIN_LEN or n > len(pred):
        return False
    matcher = SequenceMatcher(None, gold, "")
    for i in range(len(pred) - n + 1):
        matcher.set_seq2(pred[i : i + n])
        if matcher.ratio() >= ratio:
            return True
    return False


def score_name(prediction: str, gold: str, accepted: list[str] | None = None) -> bool:
    pred = normalize_name(prediction)
    golds = {normalize_name(g) for g in ([gold, *(accepted or [])]) if g}
    if pred in golds:
        return True

    # containment both ways (spoken answers embed the name in a sentence), then
    # a fuzzy fallback for STT mis-spellings of invented compound names. Each
    # containment branch is length-guarded so a short token can't match a
    # different entity that merely contains it.
    def _contains(g: str) -> bool:
        if not g:
            return False
        if len(g) >= _CONTAIN_MIN_LEN and g in pred:
            return True
        if len(pred) >= _CONTAIN_MIN_LEN and pred in g:
            return True
        return _fuzzy_contains(g, pred)

    return any(_contains(g) for g in golds)


_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def extract_number(s: str) -> float | None:
    m = _NUM_RE.search((s or "").replace("\xa0", " "))
    if not m:
        return None
    return float(m.group(0).replace(",", "."))


def extract_numbers(s: str) -> list[float]:
    """All numbers in a string (German comma decimals normalized)."""
    return [
        float(m.replace(",", "."))
        for m in _NUM_RE.findall((s or "").replace("\xa0", " "))
    ]


def score_number(
    prediction, gold: float, tolerance: float = 0.0, *, final_only: bool = False
) -> bool:
    """Correct if the gold value matches a number in the answer within tolerance.

    Spoken answers state intermediate values before the final result ("2,19 …
    8,76 … zusammen 15,63 Euro"). For tight tolerances (e.g. 0.01 for prices) an
    accidental intermediate collision is effectively impossible, so we accept the
    gold matching ANY spoken number. For a WIDE tolerance (kcal ±5) an operand
    (a per-100g value or gram amount) can fall within ±5 of the total by chance —
    so with `final_only` we match ONLY the last number in the answer (where the
    stated total lands), killing that false positive.
    """
    if isinstance(prediction, str):
        candidates = extract_numbers(prediction)
    else:
        candidates = [prediction] if prediction is not None else []
    if not candidates:
        return False
    if final_only:
        candidates = candidates[-1:]
    return any(abs(float(p) - float(gold)) <= tolerance for p in candidates)


def score_list(prediction: list[str], gold: list[str]) -> bool:
    """Exact set match: every gold element present, no extras (order-free)."""
    pred_norm = {normalize_name(x) for x in prediction}
    gold_norm = {normalize_name(x) for x in gold}
    return pred_norm == gold_norm


def score_list_f1(prediction: list[str], gold: list[str]) -> dict[str, float]:
    """Overlap-based set metrics for list answers (partial credit).

    Exact-set stays the headline pass/fail; this gives the graded signal that
    matters for spoken answers, where naming every element verbatim is brutal.
    Matching uses containment either way (a spoken element embeds the name).
    """
    gold_norm = [normalize_name(x) for x in gold if normalize_name(x)]
    pred_norm = [normalize_name(x) for x in prediction if normalize_name(x)]
    if not gold_norm:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def _hits(a: list[str], b: list[str]) -> int:
        return sum(any(x in y or y in x for y in b) for x in a)

    recall = _hits(gold_norm, pred_norm) / len(gold_norm)
    precision = _hits(pred_norm, gold_norm) / len(pred_norm) if pred_norm else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def score_item(prediction, item: dict, tolerances: dict) -> bool:
    """Dispatch on the item's answer_type. tolerances = cfg['scoring']['number_tolerance']."""
    atype = item.get("answer_type", "name")
    if atype == "name":
        return score_name(
            str(prediction), item["gold_answer"], item.get("accepted_answers")
        )
    if atype == "number":
        kind = item.get("number_kind", "default")
        tol = tolerances.get(kind, tolerances["default"])
        # kcal uses a wide ±5 band → match only the final stated total, not any
        # intermediate operand that could land within ±5 by chance.
        return score_number(
            prediction, float(item["gold_answer"]), tol, final_only=(kind == "kcal")
        )
    if atype == "list":
        pred = prediction if isinstance(prediction, list) else [prediction]
        return score_list(pred, item["gold_answer"])
    raise ValueError(f"unknown answer_type {atype!r}")
