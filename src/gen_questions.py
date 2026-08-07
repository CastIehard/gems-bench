"""Stage 5 — generate questions by walking the graph; derive gold answer + docs.

IN : data/stage_3_graph.json + data/stage_4_corpus.jsonl
OUT: data/stage_5_questions_raw.json

One generator per category (one_hop, serial, early, select, combined).
Hop/branch counts drawn ~Gauss(mu, sigma) from config. Questions
are emitted as raw structured templates (verbatim entity names preserved as
retrieval anchors).

Usage:
    python src/gen_questions.py [--rerun]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import det_rng, read_json, read_jsonl, write_json
from config import load_config

# repo root on path so the answerability gate reuses the EXACT product_lookup
# the model under test calls.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gems_tools import ProductCatalog  # noqa: E402


# ── index over graph + corpus ────────────────────────────────────────────────
class World:
    def __init__(self, cfg):
        self.cfg = cfg
        self.graph = read_json(cfg["_paths"]["graph"])
        self.docs = read_jsonl(cfg["_paths"]["corpus"])
        N, E = self.graph["nodes"], self.graph["edges"]
        self.E = E
        self.name = {
            n["id"]: n["name"]
            for k in ("marke", "lieferant", "lager", "region", "einkaeufer", "team")
            for n in N[k]
        }
        self.prod = {p["id"]: p for p in N["produkt"]}
        self.desc = {p["id"]: p["descriptor"] for p in N["produkt"]}
        # doc lookups — each doc lists the (edge_type, subject_id) facts it covers
        self.edge_doc = {}
        self.attr_doc = {}
        for d in self.docs:
            for edge_type, subject_id in d["facts"]:
                if edge_type in ("einkaufspreis", "kcal_100g", "lagerbestand"):
                    self.attr_doc[(edge_type, subject_id)] = d["doc_id"]
                else:
                    self.edge_doc[(edge_type, subject_id)] = d["doc_id"]
        # reverse: products per supplier (via liefert)
        self.products_by_supplier = {}
        for pid, lid in E["liefert"].items():
            self.products_by_supplier.setdefault(lid, []).append(pid)

        # override status: is the product's actual supplier ≠ its brand's supplier?
        # one_hop draws non-override (naive brand path works); serial/early
        # depth-4 draw override (naive path is a trap / no shortcut).
        self.is_override = {
            pid: E["liefert"][pid] != E["bezieht_von"][E["hat_marke"][pid]]
            for pid in self.prod
        }

        # answerability gate: only reference products that resolve UNIQUELY to
        # themselves via the SAME product_lookup the model calls (full descriptor
        # -> exactly this product). Excludes rare token-superset collisions so no
        # question can be unanswerable/ambiguous by construction.
        catalog = ProductCatalog(cfg["_paths"]["graph"])
        self.resolvable = set()
        for pid, d in self.desc.items():
            res = catalog.lookup(d)
            prod = res.get("product") if res.get("status") == "success" else None
            if prod and str(prod.get("dan")) == str(self.prod[pid].get("dan")):
                self.resolvable.add(pid)

    def pool(self, *, override: bool | None = None) -> list[str]:
        """Resolvable product ids, optionally filtered by override status."""
        return [
            pid
            for pid in self.prod
            if pid in self.resolvable
            and (override is None or self.is_override[pid] == override)
        ]

    def edoc(self, etype, sid):
        return self.edge_doc[(etype, sid)]

    def adoc(self, atype, pid):
        return self.attr_doc[(atype, pid)]


def _gauss_int(rng, mu, sigma, lo, hi):
    return max(lo, min(hi, round(rng.gauss(mu, sigma))))


def _pick(pool, r, used):
    """Draw one product id from pool not yet used by any single-anchor category,
    and mark it used. Enforces GLOBAL (cross-item, cross-category) uniqueness of
    the anchor product: without it two items can draw the same product and become
    the identical question with the identical gold. Returns None when the pool is
    exhausted (caller stops emitting)."""
    avail = [pid for pid in pool if pid not in used]
    if not avail:
        return None
    pid = avail[r.randrange(len(avail))]
    used.add(pid)
    return pid


def _short(w, desc: str) -> str:
    """Short product form for aggregation lists (keeps audio/questions concise):
    text before the first comma or parenthesis. Toggle via config."""
    if not w.cfg["questions"]["short_descriptors"]:
        return desc
    for sep in (" (", "(", ", ", ","):
        if sep in desc:
            desc = desc.split(sep, 1)[0]
    return desc.strip()


def _agg_k(w, r, n_available):
    """Products per fan-out question. 6–10 (config) → a real ≥30 s listening
    window from 6–10 full product names AND enough independent sub-lookups that
    parallel/early retrieval can fill the window. Every product is a real,
    independently-retrieved leg of the question (same work for all modes)."""
    a = w.cfg["questions"]["aggregation_size"]
    lo, hi = a["min"], min(a["max"], n_available)
    return max(lo, r.randint(lo, hi)) if hi >= lo else n_available


def _sample_unique_max(r, pool, k, keyfn, *, tries=12):
    """Sample k items whose max under keyfn is UNIQUE (no tie for first place).

    select's gold is the argmax product; a tie would make the answer ambiguous
    (two valid golds) and break judge-free scoring. Resample until the top value
    is strictly greater than the runner-up, or give up (caller skips the item)."""
    for _ in range(tries):
        chosen = r.sample(pool, k)
        vals = sorted((keyfn(p) for p in chosen), reverse=True)
        if len(vals) >= 2 and vals[0] > vals[1]:
            return chosen
    return None


# ── per-category generators (return list of items) ───────────────────────────
def gen_one_hop(w, n, seed, used):
    """Single edge: who supplies product X? Non-override products only (naive
    brand-supplier path is correct here — the trap lives in `override`)."""
    pids = w.pool(override=False)
    items = []
    for i in range(n):
        r = det_rng(seed, "one_hop", i)
        pid = _pick(pids, r, used)
        if pid is None:
            break
        sup = w.E["liefert"][pid]
        items.append(
            _item(
                f"one_hop_{i:04d}",
                "one_hop",
                f"Von welcher Firma wird das Produkt {w.desc[pid]} geliefert?",
                "name",
                w.name[sup],
                [w.edoc("liefert", pid)],
                [f"Wer liefert {w.desc[pid]}?"],
                {"type": "single", "depth": 1},
                seed,
                skeleton={
                    "entities": [w.desc[pid]],
                    "ask": "Von welcher Firma wird dieses Produkt geliefert?",
                },
            )
        )
    return items


def gen_serial(w, n, seed, used, category="serial", front_load=False):
    """Depth 3–4 supply chain: region of the warehouse of the supplier of X.

    Override products only: at depth 4 the chain runs through the BRAND's
    supplier, so a product whose own supplier differs from its brand's supplier
    can't be shortcut via `product_lookup` — the deep hops stay real.

    front_load=True (the `early` category) puts the product entity FIRST in the
    question so an early-retrieval agent can start the hop chain while still
    listening; the default serial phrasing names the product last."""
    pids = w.pool(override=True)
    items = []
    for i in range(n):
        r = det_rng(seed, category, i)
        pid = _pick(pids, r, used)
        if pid is None:
            break
        hops = w.cfg["questions"]["hops"]
        depth = _gauss_int(
            r,
            hops["mu"],
            hops["sigma"],
            hops["depth_min"],
            hops["depth_max"],
        )
        sup = w.E["liefert"][pid]
        lager = w.E["lagert_in"][sup]
        region = w.E["liegt_in"][lager]
        if depth >= 4:
            marke = w.E["hat_marke"][pid]
            bsup = w.E["bezieht_von"][marke]
            blager = w.E["lagert_in"][bsup]
            bregion = w.E["liegt_in"][blager]
            if front_load:
                q = (
                    f"Beim Produkt {w.desc[pid]} — in welcher Region liegt das "
                    f"Lager des Lieferanten seiner Marke?"
                )
            else:
                q = (
                    f"In welcher Region liegt das Lager des Lieferanten der Marke "
                    f"des Produkts {w.desc[pid]}?"
                )
            docs = [
                w.edoc("hat_marke", pid),
                w.edoc("bezieht_von", marke),
                w.edoc("lagert_in", bsup),
                w.edoc("liegt_in", blager),
            ]
            gold = w.name[bregion]
            subq = [
                f"Welche Marke hat {w.desc[pid]}?",
                f"Welcher Lieferant beliefert die Marke?",
                "In welchem Lager lagert der Lieferant?",
                "In welcher Region liegt das Lager?",
            ]
            # Flat step-list instead of a nested genitive chain — the difficulty
            # is the multi-hop traversal, not parsing "die Region des Lagers des
            # Lieferanten der Marke des …". A human speaker reads these steps.
            ask = (
                "Finde die Marke dieses Produkts, dann den Lieferanten dieser "
                "Marke, dann das Lager dieses Lieferanten — und nenne die Region, "
                "in der dieses Lager liegt."
            )
        else:
            if front_load:
                q = (
                    f"Beim Produkt {w.desc[pid]} — in welcher Region liegt das "
                    f"Lager seines Lieferanten?"
                )
            else:
                q = (
                    f"In welcher Region liegt das Lager des Lieferanten "
                    f"des Produkts {w.desc[pid]}?"
                )
            docs = [
                w.edoc("liefert", pid),
                w.edoc("lagert_in", sup),
                w.edoc("liegt_in", lager),
            ]
            gold = w.name[region]
            subq = [
                f"Wer liefert {w.desc[pid]}?",
                "In welchem Lager lagert der Lieferant?",
                "In welcher Region liegt das Lager?",
            ]
            ask = (
                "Finde den Lieferanten dieses Produkts, dann das Lager dieses "
                "Lieferanten — und nenne die Region, in der dieses Lager liegt."
            )
        items.append(
            _item(
                f"{category}_{i:04d}",
                category,
                q,
                "name",
                gold,
                docs,
                subq,
                {"type": "serial", "depth": depth},
                seed,
                skeleton={"entities": [w.desc[pid]], "ask": ask},
            )
        )
    return items


def gen_early(w, n, seed, used):
    """Same chain as serial, but the product entity is FRONT-LOADED (named first)
    so the audio exposes it early — the listening-window lever this category exists
    to test. Only the question phrasing differs from serial; gold/docs/chain are
    identical."""
    return gen_serial(w, n, seed, used, category="early", front_load=True)


def gen_select(w, n, seed, used):
    """Parallel select (money category): of 6–10 INDEPENDENT products named up
    front, which has the highest stock (Lagerbestand)?

    Genuinely parallel: k independent Hop-1 `product_lookup`s (one per product)
    + a cheap late argmax. Entities are front-loaded so an early/parallel agent
    can fan out N sub-lookups DURING the ~30 s of listening, while a serial
    post-turn baseline must chain them after end-of-speech. Answer = the
    winning product's descriptor (name-type; unique max enforced so the gold
    is unambiguous)."""
    pool = [
        p
        for p in w.prod.values()
        if p["id"] in w.resolvable and p["attrs"].get("lagerbestand") is not None
    ]
    items = []
    for i in range(n):
        r = det_rng(seed, "select", i)
        if len(pool) < 2:
            break
        k = _agg_k(w, r, len(pool))
        chosen = _sample_unique_max(r, pool, k, lambda p: p["attrs"]["lagerbestand"])
        if chosen is None:
            continue  # couldn't break a tie — skip (gate tolerates < n items)
        winner = max(chosen, key=lambda p: p["attrs"]["lagerbestand"])
        names = "; ".join(w.desc[p["id"]] for p in chosen)
        docs = [w.adoc("lagerbestand", p["id"]) for p in chosen]
        # entities FIRST (front-loaded), the argmax op stated last
        q = (
            f"Es geht um folgende Produkte: {names}. "
            f"Welches davon hat den höchsten Lagerbestand?"
        )
        items.append(
            _item(
                f"select_{i:04d}",
                "select",
                q,
                "name",
                w.desc[winner["id"]],
                docs,
                [
                    f"Wie hoch ist der Lagerbestand von {w.desc[p['id']]}?"
                    for p in chosen
                ],
                {"type": "parallel", "branches": k},
                seed,
                skeleton={
                    "entities": [w.desc[p["id"]] for p in chosen],
                    "ask": "Welches dieser Produkte hat den höchsten Lagerbestand?",
                },
            )
        )
    return items


def gen_combined(w, n, seed, used):
    """Flagship aggregation: sum the purchase prices of N products (fan-out +
    arithmetic). Robust — every product has an einkaufspreis."""
    priced = [
        p
        for p in w.prod.values()
        if p["attrs"].get("einkaufspreis") is not None and p["id"] in w.resolvable
    ]
    items = []
    for i in range(n):
        r = det_rng(seed, "combined", i)
        if len(priced) < 2:
            break
        k = _agg_k(w, r, len(priced))
        chosen = r.sample(priced, k)
        total = round(
            sum(p["attrs"]["einkaufspreis"] for p in chosen),
            w.cfg["schema"]["attributes"]["einkaufspreis_decimals"],
        )
        # semicolons delimit items — full descriptors contain commas, so a comma
        # join would blur where one product ends and the next begins (TTS reads ; as a pause).
        names = "; ".join(_short(w, p["descriptor"]) for p in chosen)
        docs = [w.adoc("einkaufspreis", p["id"]) for p in chosen]
        # entities FIRST (front-loaded), the aggregation op stated last
        q = (
            f"Es geht um folgende Produkte: {names}. "
            f"Wie hoch ist die Summe ihrer Einkaufspreise?"
        )
        items.append(
            _item(
                f"combined_{i:04d}",
                "combined",
                q,
                "number",
                total,
                docs,
                [f"Was kostet {p['descriptor']} im Einkauf?" for p in chosen],
                {"type": "parallel", "branches": k},
                seed,
                number_kind="preis",
                skeleton={
                    "entities": [_short(w, p["descriptor"]) for p in chosen],
                    "ask": "Wie hoch ist die Summe der Einkaufspreise dieser Produkte?",
                },
            )
        )
    return items


def _item(
    item_id,
    category,
    text,
    answer_type,
    gold,
    docs,
    subq,
    dep,
    seed,
    number_kind=None,
    skeleton=None,
):
    docs = list(dict.fromkeys(docs))  # dedupe, preserve order
    item = {
        "item_id": item_id,
        "category": category,
        "raw_question": text,
        "question_text": text,
        "answer_type": answer_type,
        "gold_answer": gold,
        "accepted_answers": [] if answer_type != "name" else [gold],
        "gold_documents": docs,
        "gold_subquestions": subq,
        "dependency_graph": dep,
        "seed": seed,
    }
    if number_kind:
        item["number_kind"] = number_kind
    # Skeleton = what a recording is verified against: the exact entities that
    # must be recoverable from the transcript and the target relation being asked.
    # The speaker reads spoken_question verbatim; the skeleton is the machine-side
    # fidelity contract, and its entity list doubles as the recorder's
    # pronunciation aid. src/verify_recording.py checks the recording against it.
    # Default (single-entity categories): the one product + the question itself.
    item["skeleton"] = skeleton or {
        "entities": dep.get("entities", []),
        "ask": text,
    }
    return item


GENERATORS = {
    "one_hop": gen_one_hop,
    "serial": gen_serial,
    "early": gen_early,
    "select": gen_select,
    "combined": gen_combined,
}


def run(cfg: dict, *, rerun: bool = False) -> list[dict]:
    out_path = cfg["_paths"]["questions_raw"]
    if out_path.exists() and not rerun:
        items = read_json(out_path)
        print(f"cached ({len(items)} items) -> {out_path}")
        return items

    if not cfg["_paths"]["graph"].exists():
        raise SystemExit(f"{cfg['_paths']['graph']} missing — run build_graph.py first")
    if not cfg["_paths"]["corpus"].exists():
        raise SystemExit(
            f"{cfg['_paths']['corpus']} missing — run emit_corpus.py first"
        )
    w = World(cfg)
    n_prod = len(w.prod)
    n_resolv = len(w.resolvable)
    n_over = sum(1 for pid in w.resolvable if w.is_override[pid])
    print(
        f"  answerability: {n_resolv}/{n_prod} products uniquely resolvable via "
        f"product_lookup ({n_prod - n_resolv} excluded from questions); "
        f"of resolvable: {n_over} override / {n_resolv - n_over} non-override"
    )
    seed = cfg["seed"]
    items: list[dict] = []
    used: set[str] = set()  # anchor products already used by single-anchor categories
    for cat, count in cfg["questions"]["distribution"].items():
        gen = GENERATORS[cat]
        cat_items = gen(w, count, seed, used)
        items.extend(cat_items)
        print(f"  {cat}: {len(cat_items)} items")
    write_json(cfg["_paths"]["questions_raw"], items)
    print(f"generated {len(items)} raw questions -> {cfg['_paths']['questions_raw']}")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun", action="store_true", help="ignore cached questions_raw"
    )
    args = parser.parse_args()
    run(
        load_config(),
        rerun=args.rerun,
    )


if __name__ == "__main__":
    main()
