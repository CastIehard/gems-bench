"""Stage 4 — emit the fact corpus as natural German prose.

IN : data/stage_3_graph.json
OUT: data/stage_4_corpus.jsonl  (+ data/stage_4_corpus_cache.json, LLM-weave cache)


Facts are bundled per entity (product / brand / supplier / warehouse / buyer).
Per profile we randomize the fact order (deterministic) and ask the LLM to
weave a natural paragraph. Each fact carries answer-bearing `tokens` (entity
names, prices, numbers) that MUST appear verbatim; a check verifies every
required token (plus the entity anchor) survived (retry on failure, else fall
back to a plain join). The LLM is free to phrase the connective text around
them — so docs read naturally, not as an identical template per product.

Each doc records which (edge_type, subject_id) facts it covers so gen_questions
can resolve gold_documents. Woven text is cached by fact-hash so re-runs don't
re-call the LLM.

Usage:
    python src/emit_corpus.py [--no-llm]
"""

from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm

from common import (
    det_rng,
    llm_chat,
    read_json,
    read_jsonl,
    text_has_all,
    write_json,
    write_jsonl,
)
from config import load_config


def _fmt_preis(v: float, decimals: int) -> str:
    return f"{v:.{decimals}f}".replace(".", ",")


def _profiles(graph: dict, preis_decimals: int) -> list[dict]:
    """Per-entity fact bundles. Each fact = {edge_type, subject_id, sentence, tokens}.
    `sentence` is the semantic ground truth; `tokens` are the verbatim strings
    the woven text must contain. `anchor` is the entity name that must appear so
    the doc stays retrievable."""
    N, E = graph["nodes"], graph["edges"]
    name = {
        n["id"]: n["name"]
        for k in ("marke", "lieferant", "lager", "region", "einkaeufer", "team")
        for n in N[k]
    }
    desc = {p["id"]: p["descriptor"] for p in N["produkt"]}
    profiles: list[dict] = []

    for p in N["produkt"]:
        pid = p["id"]
        d = desc[pid]
        a = p["attrs"]
        # Split each product into two shorter docs instead of one stuffed 6-fact
        # paragraph: a commercial profile (brand/supplier/buyer/price) and a
        # logistics-nutrition profile (kcal/stock). gold_documents resolves by
        # (edge_type, subject_id) -> doc_id (gen_questions), so the split is
        # transparent to scoring and sharpens gold precision. entity_id carries a
        # sub-suffix so the deterministic fact-order shuffle diverges per doc.
        commercial = [
            {
                "edge_type": "hat_marke",
                "subject_id": pid,
                "sentence": f"Das Produkt {d} gehört zur Marke {name[E['hat_marke'][pid]]}.",
                "tokens": [name[E["hat_marke"][pid]]],
            },
            {
                "edge_type": "liefert",
                "subject_id": pid,
                "sentence": f"Das Produkt {d} wird von der {name[E['liefert'][pid]]} geliefert.",
                "tokens": [name[E["liefert"][pid]]],
            },
            {
                "edge_type": "eingekauft_von",
                "subject_id": pid,
                "sentence": f"Das Produkt {d} wird von {name[E['eingekauft_von'][pid]]} eingekauft.",
                "tokens": [name[E["eingekauft_von"][pid]]],
            },
        ]
        if a.get("einkaufspreis") is not None:
            commercial.append(
                {
                    "edge_type": "einkaufspreis",
                    "subject_id": pid,
                    "sentence": f"Der Einkaufspreis von {d} beträgt {_fmt_preis(a['einkaufspreis'], preis_decimals)} Euro.",
                    "tokens": [
                        f"{_fmt_preis(a['einkaufspreis'], preis_decimals)} Euro"
                    ],
                }
            )
        logistik = []
        if a.get("kcal_100g") is not None:
            logistik.append(
                {
                    "edge_type": "kcal_100g",
                    "subject_id": pid,
                    "sentence": f"{d} hat {a['kcal_100g']} Kilokalorien pro 100 Gramm.",
                    "tokens": [f"{a['kcal_100g']} Kilokalorien"],
                }
            )
        if a.get("lagerbestand") is not None:
            logistik.append(
                {
                    "edge_type": "lagerbestand",
                    "subject_id": pid,
                    "sentence": f"Vom Produkt {d} sind aktuell {a['lagerbestand']} Stück auf Lager.",
                    "tokens": [f"{a['lagerbestand']} Stück"],
                }
            )
        profiles.append(
            {
                "entity_type": "produkt",
                "entity_id": f"{pid}#kommerz",
                "anchor": d,
                "facts": commercial,
            }
        )
        if logistik:
            profiles.append(
                {
                    "entity_type": "produkt",
                    "entity_id": f"{pid}#logistik",
                    "anchor": d,
                    "facts": logistik,
                }
            )

    # supplier-summary ("sortiment") docs: one per supplier, listing ALL its
    # products verbatim, as extra supplier evidence. edge_type "sortiment"
    # keeps it distinct from the per-product ("liefert", pid) facts (no
    # gold-doc-id collision).
    products_by_supplier: dict[str, list[str]] = {}
    for pid, lid in E["liefert"].items():
        products_by_supplier.setdefault(lid, []).append(pid)
    for lid, pids in products_by_supplier.items():
        prod_names = [desc[p] for p in pids]
        profiles.append(
            {
                "entity_type": "lieferant_sortiment",
                "entity_id": f"{lid}#sortiment",
                "anchor": name[lid],
                "facts": [
                    {
                        "edge_type": "sortiment",
                        "subject_id": lid,
                        "sentence": (
                            f"Die {name[lid]} liefert folgende Produkte: "
                            + "; ".join(prod_names)
                            + "."
                        ),
                        "tokens": prod_names,
                    }
                ],
            }
        )

    for mid, lid in E["bezieht_von"].items():
        profiles.append(
            {
                "entity_type": "marke",
                "entity_id": mid,
                "anchor": name[mid],
                "facts": [
                    {
                        "edge_type": "bezieht_von",
                        "subject_id": mid,
                        "sentence": f"Die Marke {name[mid]} bezieht ihre Waren von der {name[lid]}.",
                        "tokens": [name[mid], name[lid]],
                    }
                ],
            }
        )
    for lid, wid in E["lagert_in"].items():
        profiles.append(
            {
                "entity_type": "lieferant",
                "entity_id": lid,
                "anchor": name[lid],
                "facts": [
                    {
                        "edge_type": "lagert_in",
                        "subject_id": lid,
                        "sentence": f"Die {name[lid]} lagert ihre Waren hauptsächlich im {name[wid]}.",
                        "tokens": [name[lid], name[wid]],
                    }
                ],
            }
        )
    for wid, rid in E["liegt_in"].items():
        profiles.append(
            {
                "entity_type": "lager",
                "entity_id": wid,
                "anchor": name[wid],
                "facts": [
                    {
                        "edge_type": "liegt_in",
                        "subject_id": wid,
                        "sentence": f"Das {name[wid]} liegt in der Region {name[rid]}.",
                        "tokens": [name[wid], name[rid]],
                    }
                ],
            }
        )
    for eid, tid in E["gehoert_zu"].items():
        profiles.append(
            {
                "entity_type": "einkaeufer",
                "entity_id": eid,
                "anchor": name[eid],
                "facts": [
                    {
                        "edge_type": "gehoert_zu",
                        "subject_id": eid,
                        "sentence": f"{name[eid]} gehört zum {name[tid]}.",
                        "tokens": [name[eid], name[tid]],
                    }
                ],
            }
        )
    return profiles


def _weave(cfg, anchor, sentences, tokens, cache):
    """Weave facts into natural prose. Let the model write freely; loosely verify
    every answer-bearing token + anchor survived (case/whitespace-insensitive).

    Returns (text, meta). On LLM error or after all attempts fail the loose check,
    falls back to the plain sentence join (always token-complete) so no doc is
    ever missing — a missing doc would break gold_documents resolution. meta =
    {retries, error, fallback} for the run-level report. Text is cached by hash."""
    required = [anchor, *tokens]
    # Plain-join mode short-circuits before the cache lookup: plain text is
    # deterministic and free, and a stale LLM-woven cache entry (same fact hash)
    # must not win when the caller asked for plain.
    if not cfg["corpus"]["llm_weave"] or not sentences:
        return " ".join(sentences), {"retries": 0, "error": False, "fallback": False}

    key = hashlib.sha256(("||".join(sentences)).encode()).hexdigest()
    if key in cache:
        return cache[key], {"retries": 0, "error": False, "fallback": False}

    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    user = (
        numbered
        + "\n\nDiese Angaben muessen im Text WORTWOERTLICH vorkommen: "
        + "; ".join(required)
    )
    max_tries = max(1, cfg["corpus"]["max_retries"])
    for attempt in range(max_tries):
        try:
            out = llm_chat(
                cfg,
                [
                    {"role": "system", "content": cfg["corpus"]["prompt"]},
                    {"role": "user", "content": user},
                ],
            ).strip()
        except Exception as exc:  # noqa: BLE001
            print(f"  weave LLM error ({exc.__class__.__name__}) — plain-join fallback")
            plain = " ".join(sentences)
            cache[key] = plain
            return plain, {"retries": attempt, "error": True, "fallback": True}
        if text_has_all(out, required):
            cache[key] = out
            return out, {"retries": attempt, "error": False, "fallback": False}
    # model wrote, but no attempt kept all required tokens → plain-join fallback
    # (token-complete by construction; keeps the doc in the corpus so
    # gold_documents never dangles).
    plain = " ".join(sentences)
    cache[key] = plain
    return plain, {"retries": max_tries - 1, "error": False, "fallback": True}


def run(cfg: dict, *, llm: bool | None = None, rerun: bool = False) -> list[dict]:
    out_path = cfg["_paths"]["corpus"]
    if out_path.exists() and not rerun:
        docs = read_jsonl(out_path)
        print(f"cached ({len(docs)} docs) -> {out_path}")
        return docs

    if llm is not None:
        cfg = {**cfg, "corpus": {**cfg["corpus"], "llm_weave": llm}}
    graph_path = cfg["_paths"]["graph"]
    if not graph_path.exists():
        raise SystemExit(f"{graph_path} missing — run build_graph.py first")
    graph = read_json(graph_path)
    seed = cfg["seed"]
    preis_decimals = cfg["schema"]["attributes"]["einkaufspreis_decimals"]
    cache_path = cfg["_paths"]["corpus_cache"]
    cache = read_json(cache_path) if cache_path.exists() else {}

    # Per-profile fact order is deterministic (seed + entity_id) and cheap — do
    # it upfront, sequentially, so doc order/doc_id stays reproducible regardless
    # of which docs later get parallelized onto worker threads.
    profiles = _profiles(graph, preis_decimals)
    jobs = []
    for prof in profiles:
        facts = list(prof["facts"])
        det_rng(seed, "corpus_order", prof["entity_id"]).shuffle(facts)
        jobs.append(
            {
                "facts": facts,
                "sentences": [f["sentence"] for f in facts],
                "tokens": [t for f in facts for t in f["tokens"]],
            }
        )

    # Only a genuine LLM cache-miss is network I/O worth a thread — cache hits
    # and plain-join (llm_weave off) resolve instantly inline, same "todo" split
    # as the qa closed-book gate (qa_checks.py).
    def _cache_key(job):
        if not cfg["corpus"]["llm_weave"] or not job["sentences"]:
            return None
        return hashlib.sha256(("||".join(job["sentences"])).encode()).hexdigest()

    def _resolve(i):
        job = jobs[i]
        return i, *_weave(
            cfg, profiles[i]["anchor"], job["sentences"], job["tokens"], cache
        )

    results: list[tuple[str, dict] | None] = [None] * len(jobs)
    todo = []
    for i, job in enumerate(jobs):
        key = _cache_key(job)
        if key is None or key in cache:
            _, text, meta = _resolve(i)
            results[i] = (text, meta)
        else:
            todo.append(i)

    max_workers = cfg["corpus"]["max_workers"]
    if todo:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_resolve, i): i for i in todo}
            for fut in tqdm(
                as_completed(futures),
                total=len(todo),
                desc="weave corpus (LLM)",
                unit="doc",
            ):
                i, text, meta = fut.result()
                results[i] = (text, meta)
                # Checkpoint after every completion (not just at the end) so a
                # crash/quit mid-run loses at most the doc in flight per worker —
                # rerunning resumes via cache hits instead of re-weaving everything.
                write_json(cache_path, cache)

    docs: list[dict] = []
    n_reruns = n_errors = n_fallback = 0
    fallbacks: list[str] = []
    for i, prof in enumerate(profiles):
        text, meta = results[i]
        facts = jobs[i]["facts"]
        n_reruns += meta["retries"]
        n_errors += 1 if meta["error"] else 0
        if meta.get("fallback"):
            n_fallback += 1
            fallbacks.append(prof["entity_id"])
        docs.append(
            {
                "doc_id": f"doc_{i:04d}",
                "entity_type": prof["entity_type"],
                "entity_id": prof["entity_id"],
                "facts": [[f["edge_type"], f["subject_id"]] for f in facts],
                "text": text,
            }
        )

    write_jsonl(cfg["_paths"]["corpus"], docs)
    write_json(cache_path, cache)
    mode = "LLM-woven" if cfg["corpus"]["llm_weave"] else "plain"
    print(
        f"corpus: {len(docs)} docs ({mode}) | "
        f"reruns={n_reruns} errors={n_errors} plain-join-fallback={n_fallback} "
        f"-> {cfg['_paths']['corpus']}"
    )
    if fallbacks:
        print(f"  plain-join fallback (weave failed verify): {', '.join(fallbacks)}")
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-llm", action="store_true", help="plain join, no LLM weave"
    )
    parser.add_argument(
        "--rerun", action="store_true", help="ignore cached stage_4_corpus.jsonl"
    )
    args = parser.parse_args()
    run(load_config(), llm=False if args.no_llm else None, rerun=args.rerun)


if __name__ == "__main__":
    main()
