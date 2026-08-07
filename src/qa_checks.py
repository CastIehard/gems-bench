"""Stage 7 — QA the (already spoken-rewritten) items.

IN : data/stage_6_questions_spoken.json + data/stage_4_corpus.jsonl
OUT: data/stage_7_questions.json  (filtered, final)

- uniqueness: gold answer present + gold docs resolve (structural)
- closed-book gate: ask an ENSEMBLE of models the SPOKEN question WITHOUT corpus
  (qa.closed_book_gate.models, each n_votes tries); drop items ANY model can guess
  (proves closed-world unanswerability on the exact text that gets asked). Gating
  the spoken form — not the written one — catches answer-leaks the rewrite adds.
- anti-rate: report answer-space size per type

Usage:
    python src/qa_checks.py [--no-gate] [--rerun]   # --no-gate skips LLM
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm

import scoring
from common import llm_chat, read_json, read_jsonl, write_json
from config import load_config

CLOSED_BOOK_SYS = (
    "Beantworte die Frage NUR aus deinem Wissen, ohne Datenbank. "
    "Wenn du es nicht weißt, rate trotzdem eine konkrete Antwort. Antworte kurz."
)


def _structural_ok(item, doc_ids) -> bool:
    if item["gold_answer"] in ("", None, []):
        return False
    return all(d in doc_ids for d in item["gold_documents"])


def _name_collision_ids(items, graph_path) -> set:
    """Ids of `name` items whose gold answer is ambiguous under the scorer.

    A gold name that is a sub-string of (or contains) a DIFFERENT entity's name
    could be matched to the wrong entity by score_name's containment branch. We
    drop such items so every name answer resolves to exactly one entity.
    """
    if not graph_path.exists():
        return set()
    graph = read_json(graph_path)
    universe = [
        scoring.normalize_name(n["name"])
        for k in ("marke", "lieferant", "lager", "region", "einkaeufer", "team")
        for n in graph["nodes"][k]
    ]
    bad = set()
    for item in items:
        if item.get("answer_type") != "name":
            continue
        g = scoring.normalize_name(item["gold_answer"])
        if not g:
            continue
        for other in universe:
            if other == g:
                continue
            if g in other or other in g:
                bad.add(item["item_id"])
                break
    return bad


def _closed_book_hits(
    cfg, item, models, n_votes, tol, temperature, *, fail_fast
) -> dict | None:
    """Poll each gate model n_votes times on the SPOKEN question, no corpus.

    Returns {model: hits_without_corpus} so the per-item verdict is auditable.
    Lists are never guessable → returns {} (no model polled).

    fail_fast=True (default): an LLM error propagates and stops the run with
    the full exception (model, item, root cause) — debugging visibility over
    silent continuation. fail_fast=False: log and keep the item with gate not
    applied (None) so a flaky endpoint doesn't waste a whole paid run.
    """
    if item["answer_type"] == "list":
        return {}
    question = item.get("spoken_question") or item["question_text"]
    results: dict[str, int] = {}
    for model in models:
        hits = 0
        for _ in range(n_votes):
            if fail_fast:
                pred = llm_chat(
                    cfg,
                    [
                        {"role": "system", "content": CLOSED_BOOK_SYS},
                        {"role": "user", "content": question},
                    ],
                    model=model,
                    temperature=temperature,
                )
            else:
                try:
                    pred = llm_chat(
                        cfg,
                        [
                            {"role": "system", "content": CLOSED_BOOK_SYS},
                            {"role": "user", "content": question},
                        ],
                        model=model,
                        temperature=temperature,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  gate LLM error [{item['item_id']}/{model}] "
                        f"{exc.__class__.__name__}: {exc} — skipping gate for this item"
                    )
                    return None
            if scoring.score_item(pred, item, tol):
                hits += 1
        results[model] = hits
    return results


def run(cfg: dict, *, gate: bool = True, rerun: bool = False) -> list[dict]:
    out_path = cfg["_paths"]["questions"]
    if out_path.exists() and not rerun:
        kept = read_json(out_path)
        print(f"cached ({len(kept)} items) -> {out_path}")
        return kept

    spoken_path = cfg["_paths"]["questions_spoken"]
    if not spoken_path.exists():
        raise SystemExit(f"{spoken_path} missing — run spoken_rewrite.py first")
    corpus_path = cfg["_paths"]["corpus"]
    if not corpus_path.exists():
        raise SystemExit(f"{corpus_path} missing — run emit_corpus.py first")
    items = read_json(spoken_path)
    doc_ids = {d["doc_id"] for d in read_jsonl(corpus_path)}
    tol = cfg["scoring"]["number_tolerance"]
    gate_cfg = cfg["qa"]["closed_book_gate"]
    models = gate_cfg["models"]
    n_votes = gate_cfg["n_votes"]
    gate_temp = gate_cfg["temperature"]
    ambiguous_names = (
        _name_collision_ids(items, cfg["_paths"]["graph"])
        if cfg["qa"]["uniqueness_check"]
        else set()
    )

    # structural/ambiguous checks are local (no network) — cheap, stays sequential
    survivors, dropped_struct, dropped_ambiguous = [], 0, 0
    for item in items:
        if cfg["qa"]["uniqueness_check"] and not _structural_ok(item, doc_ids):
            dropped_struct += 1
            continue
        if item["item_id"] in ambiguous_names:
            dropped_ambiguous += 1
            continue
        survivors.append(item)

    # closed-book gate is network I/O per item (up to len(models) calls) — the
    # slow part, and the one that can hit a flaky endpoint. Items gate
    # concurrently across threads (llm_chat is a stateless httpx.post per
    # call — thread-safe, no shared state). Results are checkpointed to disk
    # AS THEY COMPLETE (not just at the end) so a crash/timeout mid-run loses
    # at most the few items in flight, not the whole (paid) run — rerunning
    # resumes from the checkpoint instead of re-grading everything.
    kept, dropped_gate = [], 0
    if gate:
        max_workers = gate_cfg["max_workers"]
        fail_fast = gate_cfg["fail_fast"]
        checkpoint_path = cfg["_paths"]["gate_checkpoint"]
        checkpoint: dict = (
            read_json(checkpoint_path) if checkpoint_path.exists() else {}
        )
        if checkpoint:
            print(
                f"  gate checkpoint found: {len(checkpoint)}/{len(survivors)} "
                f"items already graded, resuming"
            )

        def _gate_one(item):
            return item["item_id"], _closed_book_hits(
                cfg, item, models, n_votes, tol, gate_temp, fail_fast=fail_fast
            )

        todo = [it for it in survivors if it["item_id"] not in checkpoint]
        if todo:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_gate_one, item): item for item in todo}
                pending = set(futures)
                try:
                    for fut in tqdm(
                        as_completed(futures),
                        total=len(todo),
                        desc="qa gate",
                        unit="q",
                    ):
                        pending.discard(fut)
                        item_id, hits = fut.result()
                        checkpoint[item_id] = hits
                        write_json(checkpoint_path, checkpoint)
                finally:
                    # fail_fast crash: other workers' items are still in flight
                    # (already paid for) — wait for and save them too before
                    # the exception propagates, instead of discarding them.
                    for fut in pending:
                        try:
                            item_id, hits = fut.result()
                        except Exception:  # noqa: BLE001 — that item failed too
                            continue
                        checkpoint[item_id] = hits
                    write_json(checkpoint_path, checkpoint)

        exempt_categories = set(gate_cfg["exempt_categories"])
        for item in survivors:
            hits = checkpoint[item["item_id"]]
            if hits is not None:  # None = LLM error → keep item, gate not applied
                item["closed_book_hits"] = hits
                guessed = any(h > 0 for h in hits.values())
                if guessed and item["category"] in exempt_categories:
                    item["closed_book_exempt"] = True
                elif guessed and gate_cfg["drop_if_correct"]:
                    dropped_gate += 1
                    continue
            kept.append(item)
    else:
        kept = survivors

    write_json(cfg["_paths"]["questions"], kept)
    if gate:
        checkpoint_path.unlink(
            missing_ok=True
        )  # final output written — no longer needed

    # anti-rate report
    by_type = Counter(i["answer_type"] for i in kept)
    name_space = len({i["gold_answer"] for i in kept if i["answer_type"] == "name"})
    print(
        f"kept {len(kept)}/{len(items)} "
        f"(dropped {dropped_struct} structural, {dropped_ambiguous} ambiguous-name, "
        f"{dropped_gate} closed-book-guessable)"
    )
    print(f"  answer types: {dict(by_type)}; distinct name answers: {name_space}")
    print(f"-> {cfg['_paths']['questions']}")
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-gate", action="store_true", help="skip closed-book LLM gate"
    )
    parser.add_argument(
        "--rerun", action="store_true", help="ignore cached stage_7_questions.json"
    )
    args = parser.parse_args()
    run(load_config(), gate=not args.no_gate, rerun=args.rerun)


if __name__ == "__main__":
    main()
