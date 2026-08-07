"""Stage 6 — rewrite each raw question into spoken-natural German.

IN : data/stage_5_questions_raw.json
OUT: data/stage_6_questions_spoken.json  (adds `spoken_question` per item)

Runs BEFORE the closed-book gate (stage 7) on purpose: the gate must see the
spoken form (spoken_question) — the exact text that gets read aloud — so any
answer-leak introduced by the rewrite is caught, not the written form.

spoken_question is read WORD-FOR-WORD by both audio channels (TTS synthesis
AND the human speaker, who reads it verbatim rather than paraphrasing from the
skeleton) — so it must preserve every number and entity, not just sound
natural. Each rewrite is verified (every number in question_text + every
skeleton entity must survive in the output); on failure it retries
(spoken.max_retries) and finally falls back to the plain question_text — same
safety net as emit_corpus's corpus weave.

Example:
    "Von welcher Firma wird das Produkt Nachtcreme Vitamin A, 50 ml geliefert?"
 -> "Von welcher Firma wird nochmal diese Nachtcreme mit Vitamin A geliefert,
     ich meine die mit 50 Millilitern?"

Usage:
    python src/spoken_rewrite.py [--rerun]
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm

from common import (
    det_rng,
    entity_present,
    llm_chat,
    normalize_name,
    normalize_ws,
    read_json,
    write_json,
)
from config import load_config

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _verify(cfg: dict, item: dict, out: str) -> bool:
    """Every number in the raw question must survive as an exact substring (no
    tolerance — a dropped/altered digit is a fact change); every skeleton
    entity must still be named (token-coverage/fuzzy, like recording.gate — a
    natural rewrite may restructure how a product name and its own package
    size relate grammatically). Deliberately NOT the gold_answer — the question
    must never state its own answer, so requiring it would validate a leak."""
    numbers = _NUM_RE.findall(item["question_text"])
    out_norm = normalize_ws(out)
    if not all(normalize_ws(n) in out_norm for n in numbers):
        return False
    # The supply relation is "liefert/beliefert". A rewrite that turns it into
    # "herstellt/produziert" changes the graph relation (manufactures ≠ supplies)
    # and slips past the number/entity checks — reject it for the chain categories.
    if item["category"] in ("one_hop", "serial", "early") and re.search(
        r"herstell|produzier", out, re.IGNORECASE
    ):
        return False
    sp = cfg["spoken"]
    out_name_norm = normalize_name(out)
    entities = item.get("skeleton", {}).get("entities", [])
    return all(
        entity_present(
            e, out_name_norm, sp["entity_fuzzy_ratio"], sp["entity_token_coverage"]
        )
        for e in entities
    )


def rewrite_one(cfg: dict, item: dict, style: str) -> tuple[str, dict]:
    """Rewrite one question; verify + retry + fall back to question_text.

    Returns (spoken_question, meta) where meta = {retries, error, fallback}."""
    text = item["question_text"]
    # style directive appended so stateless calls diverge (no fixed "Ähm" opener)
    system = cfg["spoken"]["prompt"] + "\nStil für DIESE Frage: " + style
    max_tries = max(1, cfg["spoken"]["max_retries"])
    for attempt in range(max_tries):
        try:
            out = llm_chat(
                cfg,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": text},
                ],
            ).strip()
        except Exception as exc:  # noqa: BLE001 — keep written text as fallback
            print(
                f"  spoken rewrite error ({exc.__class__.__name__}) — using question_text"
            )
            return text, {"retries": attempt, "error": True, "fallback": True}
        if _verify(cfg, item, out):
            return out, {"retries": attempt, "error": False, "fallback": False}
    # model wrote, but no attempt kept every number/entity → fall back to the
    # written question (verbatim-complete by construction).
    return text, {"retries": max_tries - 1, "error": False, "fallback": True}


def run(cfg: dict, *, rerun: bool = False) -> list[dict]:
    in_path = cfg["_paths"]["questions_raw"]
    out_path = cfg["_paths"]["questions_spoken"]
    if not in_path.exists():
        raise SystemExit(f"{in_path} missing — run gen_questions.py first")
    items = read_json(in_path)

    # Resume: reuse spoken_question already written by a prior (possibly
    # interrupted) run instead of re-calling the LLM for it.
    if out_path.exists() and not rerun:
        prior = {it["item_id"]: it.get("spoken_question") for it in read_json(out_path)}
        for it in items:
            sq = prior.get(it["item_id"])
            if sq:
                it["spoken_question"] = sq

    todo = [i for i, it in enumerate(items) if not it.get("spoken_question")]
    if not todo:
        print(f"cached ({len(items)} items) -> {out_path}")
        return items

    styles = cfg["spoken"]["styles"]
    # the "name the thing first" directive: `early` MUST front-load its entity
    # (its whole reason to exist), `serial` MUST NOT (keeps the early/serial
    # contrast clean). Located by keyword so it survives style-list reordering.
    front_idx = next((i for i, s in enumerate(styles) if "zuerst" in s.lower()), None)

    def _resolve(i):
        it = items[i]
        r = det_rng(cfg["seed"], "spoken_style", it["item_id"])
        if it["category"] == "early" and front_idx is not None:
            style = styles[front_idx]
        else:
            pool = styles
            if it["category"] == "serial" and front_idx is not None:
                pool = [s for j, s in enumerate(styles) if j != front_idx]
            style = pool[r.randrange(len(pool))]
        text, meta = rewrite_one(cfg, it, style)
        return i, text, meta

    n_reruns = n_errors = n_fallback = 0
    fallbacks: list[str] = []
    max_workers = cfg["spoken"]["max_workers"]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_resolve, i): i for i in todo}
        for fut in tqdm(
            as_completed(futures), total=len(todo), desc="spoken rewrite", unit="q"
        ):
            i, text, meta = fut.result()
            items[i]["spoken_question"] = text
            n_reruns += meta["retries"]
            n_errors += 1 if meta["error"] else 0
            if meta.get("fallback"):
                n_fallback += 1
                fallbacks.append(items[i]["item_id"])
            # Checkpoint after every completion so a crash/quit mid-run loses
            # at most the items in flight — rerunning resumes via the prior
            # spoken_question values instead of re-rewriting everything.
            write_json(out_path, items)

    print(
        f"spoken rewrite: {len(todo)}/{len(items)} rewritten | "
        f"reruns={n_reruns} errors={n_errors} fallback={n_fallback} -> {out_path}"
    )
    if fallbacks:
        print(f"  fallback (verify failed / error): {', '.join(fallbacks)}")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true", help="re-rewrite existing")
    args = parser.parse_args()
    run(load_config(), rerun=args.rerun)


if __name__ == "__main__":
    main()
