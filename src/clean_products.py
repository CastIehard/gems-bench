"""Stage 1 — clean raw products into leaf skeletons for the graph.

IN : data/raw/*.json               (via raw_cache.load_leaves — no network)
OUT: data/stage_1_products.json

- scrub the real brand out of the title -> generic descriptor (anti-leak)
- parse fill quantity (amount + unit) from the title
- parse the real price (only used later as a perturbation base)
- tag coarse category + is_food (kcal eligibility) from the query provenance

Usage:
    python src/clean_products.py [--rerun]
"""

from __future__ import annotations

import re

from tqdm.auto import tqdm

import raw_cache
from common import FUELLMENGE_RE as _FUELLMENGE_RE
from common import det_rng, read_json, write_json
from config import load_config

_UNIT_CANON = {
    "ml": "ml",
    "l": "l",
    "g": "g",
    "kg": "kg",
    "st": "St",
    "stück": "St",
    "stk": "St",
}


def parse_fuellmenge(title: str) -> dict | None:
    """Return {value, unit, grams?} parsed from the title, or None."""
    m = _FUELLMENGE_RE.search(title)
    if not m:
        return None
    value = float(m.group(1).replace(",", "."))
    unit = _UNIT_CANON[m.group(2).lower()]
    grams = None
    if unit == "g":
        grams = value
    elif unit == "kg":
        grams = value * 1000
    elif unit == "ml":
        grams = value  # approx 1g/ml, fine for synthetic aggregation
    elif unit == "l":
        grams = value * 1000
    return {"value": value, "unit": unit, "grams": grams}


def scrub_brand(title: str, brand: str | None) -> str:
    """Remove brand tokens from the title (case-insensitive). Safety net —
    most titles are already brand-free, but some embed it."""
    if not brand:
        return title.strip()
    cleaned = title
    # whole brand string, then individual tokens (handles "head&shoulders")
    for tok in [brand, *re.split(r"[\s&/-]+", brand)]:
        tok = tok.strip()
        if len(tok) < 2:
            continue
        cleaned = re.sub(re.escape(tok), "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,-–—")
    return cleaned or title.strip()


def _base_key(descriptor: str) -> str:
    """Descriptor with size/amount stripped → key to collapse size-variants
    ('... 120 ml' vs '... 150 ml' → same product)."""
    base = _FUELLMENGE_RE.sub("", descriptor)
    base = re.sub(r"[,;]\s*$", "", base)
    return re.sub(r"\s+", " ", base).strip().lower().strip(" ,-–—")


def classify_food(descriptor: str, food_markers, non_food_markers) -> bool:
    d = descriptor.lower()
    if any(m in d for m in non_food_markers):
        return False
    return any(m in d for m in food_markers)


def _build_candidates(
    leaves: list[dict], food_markers: list[str], non_food_markers: list[str]
) -> tuple[list[dict], int, int]:
    candidates: list[dict] = []
    seen_keys: set[tuple[str, int | str | None]] = set()
    n_too_short = 0
    n_duplicates = 0
    for leaf in tqdm(leaves, desc="clean products", unit="prod"):
        descriptor = scrub_brand(leaf["title"], leaf.get("brand"))
        if len(descriptor) < 3:
            n_too_short += 1
            continue
        base = _base_key(descriptor)
        key = (base, leaf.get("gtin"))
        if key in seen_keys:  # collapse repeated product hits across queries
            n_duplicates += 1
            continue
        seen_keys.add(key)
        candidates.append(
            {
                "dan": leaf["dan"],
                "descriptor": descriptor,
                "is_food": classify_food(descriptor, food_markers, non_food_markers),
                "fuellmenge": parse_fuellmenge(leaf["title"]),
            }
        )
    return candidates, n_too_short, n_duplicates


def _print_summary(
    *,
    total_products: int,
    n_too_short: int,
    n_duplicates: int,
    n_candidates: int,
    products: list[dict],
    n_target: int,
) -> tuple[int, int]:
    n_food = sum(1 for c in products if c["is_food"])
    n_menge = sum(1 for c in products if c["fuellmenge"])
    print(
        "clean_products summary:\n"
        f"  total products: {total_products}\n"
        f"  skipped too short: {n_too_short}\n"
        f"  duplicates removed: {n_duplicates}\n"
        f"  after deduplication: {n_candidates}\n"
        f"  used products: {len(products)} / {n_target}\n"
        f"  used stats: {n_food} food, {n_menge} with fill quantity"
    )
    return n_food, n_menge


def run(cfg: dict, *, rerun: bool = False) -> list[dict]:
    out_path = cfg["_paths"]["products"]
    ds = cfg["data_source"]
    leaves = raw_cache.load_leaves(cfg)
    if not leaves:
        raise SystemExit(
            "no cached raw products in data/raw/ — run get_products.py first"
        )
    n_target = ds["n_products_target"]
    food_m = [m.lower() for m in ds["food_markers"]]
    nonfood_m = [m.lower() for m in ds["non_food_markers"]]

    # deterministic shuffle so the target-sized sample is category-diverse
    # (not just the first query's products)
    det_rng(cfg["seed"], "product_sample").shuffle(leaves)

    candidates, n_too_short, n_duplicates = _build_candidates(leaves, food_m, nonfood_m)

    if out_path.exists() and not rerun:
        cleaned = read_json(out_path)
        _print_summary(
            total_products=len(leaves),
            n_too_short=n_too_short,
            n_duplicates=n_duplicates,
            n_candidates=len(candidates),
            products=cleaned,
            n_target=n_target,
        )
        print(f"cached ({len(cleaned)} products) -> {out_path}")
        return cleaned

    cleaned = candidates[:n_target]
    write_json(cfg["_paths"]["products"], cleaned)
    n_food, n_menge = _print_summary(
        total_products=len(leaves),
        n_too_short=n_too_short,
        n_duplicates=n_duplicates,
        n_candidates=len(candidates),
        products=cleaned,
        n_target=n_target,
    )
    print(
        f"cleaned {len(cleaned)} products "
        f"({n_food} food, {n_menge} with fill quantity) "
        f"-> {cfg['_paths']['products']}"
    )
    return cleaned


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun", action="store_true", help="ignore cached stage_1_products.json"
    )
    args = parser.parse_args()
    run(load_config(), rerun=args.rerun)


if __name__ == "__main__":
    main()
