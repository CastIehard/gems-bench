"""Deterministic pre-audio validation gate for the question set.

IN : data/stage_7_questions.json (+ data/stage_4_corpus.jsonl for the oracles)
OUT: exit 0 + "VALIDATION PASSED" when clean, else exit 1 + a defect list.

Run this BEFORE spending time on TTS (stage 8) or human recording (stage 9).
Every text-level defect class below — verb drift, missing front-loading, dropped
numbers, wrong argmax/sum answers, answer leaks, duplicates — is catchable here
with no LLM and no audio. Audio only adds pronunciation/ASR risk, which oss_tts
(ASR self-gate) and verify_recording already cover. If this gate passes, a rerun
of the audio stages cannot surface a new *content* bug.

    python src/validate.py            # validate the final set (stage 7)

Checks (ERROR = blocks the gate, WARN = human should look):
  structural   counts match config; unique ids; required fields; docs resolve   ERROR
  numbers      every digit in question_text survives in spoken_question         ERROR
  verb         no "herstellt"/"produziert" for supply chains                     ERROR
  leak         supplier/region answer not spoken aloud (chain categories)        ERROR
  frontload    early names the product first; serial does not                    ERROR/WARN
  answer       gold_answer re-derived from the corpus (argmax/sum/terminal hop)  ERROR
  duplicates   no two items share product-set + answer                           WARN
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict

from common import distinctive_tokens, normalize_ws, read_json, read_jsonl
from config import load_config

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_STOCK_RE = re.compile(r"(\d+)\s+Stück\b[^.]*?\bauf Lager")
_EURO_RE = re.compile(r"(\d+,\d+)\s*Euro")
_CHAIN = {"one_hop", "serial", "early"}
# relation/ask words that open a serial-style question; a product named BEFORE
# any of them is genuinely front-loaded (the early lever), regardless of phrasing
_ANCHOR_WORDS = ("lieferant", "lager", "region", "welche", "liegt", "marke")


def _load_corpus(cfg: dict) -> dict[str, str]:
    return {o["doc_id"]: o["text"] for o in read_jsonl(cfg["_paths"]["corpus"])}


def _first_token_pos(entity: str, text_norm: str) -> int | None:
    """Char index of the earliest distinctive product token in text_norm, or None."""
    positions = [
        text_norm.find(t) for t in (normalize_ws(x) for x in distinctive_tokens(entity))
    ]
    hits = [p for p in positions if p >= 0]
    return min(hits) if hits else None


def run(cfg: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warns: list[str] = []
    items = read_json(cfg["_paths"]["questions"])
    corpus = _load_corpus(cfg)

    # ── structural ────────────────────────────────────────────────────────
    dist = cfg["questions"]["distribution"]
    want_total = cfg["questions"]["n_total"]
    if len(items) != want_total:
        errors.append(f"[structural] {len(items)} items, config wants {want_total}")
    got = Counter(it["category"] for it in items)
    for cat, n in dist.items():
        if got[cat] != n:
            errors.append(f"[structural] {cat}: {got[cat]} items, want {n}")
    ids = [it["item_id"] for it in items]
    for iid, c in Counter(ids).items():
        if c > 1:
            errors.append(f"[structural] duplicate item_id {iid} (x{c})")

    for it in items:
        iid = it["item_id"]
        for f in ("question_text", "spoken_question", "gold_answer"):
            if not it.get(f):
                errors.append(f"[structural] {iid}: empty '{f}'")
        # numeric categories (combined) score off gold_answer, no accepted_answers
        if it.get("answer_type") != "number" and not it.get("accepted_answers"):
            errors.append(f"[structural] {iid}: no accepted_answers")
        if not it.get("gold_documents"):
            errors.append(f"[structural] {iid}: no gold_documents")
        for did in it.get("gold_documents", []):
            if did not in corpus:
                errors.append(f"[structural] {iid}: gold_document {did} not in corpus")

    # ── per-item text + oracle checks ─────────────────────────────────────
    for it in items:
        iid, cat = it["item_id"], it["category"]
        spoken = it.get("spoken_question", "")
        sp_norm = normalize_ws(spoken)

        # numbers survive
        for num in _NUM_RE.findall(it["question_text"]):
            if normalize_ws(num) not in sp_norm:
                errors.append(f"[numbers] {iid}: '{num}' dropped from spoken_question")

        # verb drift
        if cat in _CHAIN and re.search(r"herstell|produzier", spoken, re.IGNORECASE):
            errors.append(f"[verb] {iid}: manufacturing verb in supply-chain question")

        # answer leak (only where the answer is NOT a listed candidate)
        if cat in _CHAIN:
            for ans in it["accepted_answers"]:
                if normalize_ws(ans) in sp_norm:
                    errors.append(f"[leak] {iid}: answer '{ans}' spoken aloud")

        # front-loading: is the product named before any relation/ask word?
        if cat in ("early", "serial") and it.get("skeleton", {}).get("entities"):
            ent = it["skeleton"]["entities"][0]
            e_pos = _first_token_pos(ent, sp_norm)
            anchors = [sp_norm.find(w) for w in _ANCHOR_WORDS]
            a_pos = min([p for p in anchors if p >= 0], default=-1)
            if e_pos is None or a_pos < 0:
                warns.append(f"[frontload] {iid}: could not locate entity/ask anchor")
            elif cat == "early" and e_pos > a_pos:
                errors.append(f"[frontload] {iid}: early does NOT front-load product")
            elif cat == "serial" and e_pos < a_pos:
                warns.append(
                    f"[frontload] {iid}: serial front-loads product (blurs contrast)"
                )

        # answer oracle
        _check_answer(it, corpus, errors)

    # ── duplicates ────────────────────────────────────────────────────────
    seen: dict[tuple, str] = {}
    for it in items:
        key = (
            it["category"],
            tuple(sorted(it.get("skeleton", {}).get("entities", []))),
            it["gold_answer"],
        )
        if key in seen:
            warns.append(
                f"[duplicates] {it['item_id']} == {seen[key]} (same product-set+answer)"
            )
        else:
            seen[key] = it["item_id"]

    return errors, warns


def _check_answer(it: dict, corpus: dict[str, str], errors: list[str]) -> None:
    iid, cat = it["item_id"], it["category"]
    docs = [corpus[d] for d in it["gold_documents"] if d in corpus]
    gold = it["gold_answer"]

    if cat == "one_hop":
        if normalize_ws(gold) not in normalize_ws(" ".join(docs)):
            errors.append(f"[answer] {iid}: supplier '{gold}' not in gold_document")

    elif cat in ("serial", "early"):
        # the region is stated only by the terminal "liegt in" hop (last doc)
        if not docs or normalize_ws(gold) not in normalize_ws(docs[-1]):
            errors.append(
                f"[answer] {iid}: region '{gold}' not in terminal gold_document"
            )

    elif cat == "select":
        stocks = {d: _STOCK_RE.search(t) for d, t in zip(it["gold_documents"], docs)}
        vals = {d: int(m.group(1)) for d, m in stocks.items() if m}
        if len(vals) != len(docs):
            errors.append(f"[answer] {iid}: could not parse Lagerbestand in every doc")
            return
        max_doc = max(vals, key=vals.get)
        ans_key = normalize_ws(it["accepted_answers"][0].split(",")[0])
        if ans_key not in normalize_ws(corpus[max_doc]):
            errors.append(
                f"[answer] {iid}: gold '{it['accepted_answers'][0]}' is not the argmax "
                f"(max stock {vals[max_doc]} in {max_doc})"
            )

    elif cat == "combined":
        # each product doc states exactly one Euro amount (its Einkaufspreis)
        prices = []
        for t in docs:
            euros = _EURO_RE.findall(t)
            if len(euros) == 1:
                prices.append(euros[0])
        if len(prices) != len(docs):
            errors.append(
                f"[answer] {iid}: could not parse a unique Einkaufspreis in every doc"
            )
            return
        total = round(sum(float(p.replace(",", ".")) for p in prices), 2)
        try:
            want = round(float(str(gold).replace(",", ".")), 2)
        except ValueError:
            errors.append(f"[answer] {iid}: gold_answer '{gold}' is not numeric")
            return
        if abs(total - want) > 0.01:
            errors.append(f"[answer] {iid}: sum {total} != gold {want}")


def main() -> None:
    errors, warns = run(load_config())
    for w in warns:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    print(f"\n{len(errors)} errors, {len(warns)} warnings across the question set.")
    if errors:
        print("VALIDATION FAILED — fix before recording audio.")
        sys.exit(1)
    print("VALIDATION PASSED — safe to run stage 8 (TTS) / stage 9 (recording).")


if __name__ == "__main__":
    main()
