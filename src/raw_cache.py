"""Shared access to the raw DM product cache (data/raw/{query}.json).

This is stage 0's (get_products) output location. The fetcher writes here; any
downstream reader (clean_products, gen_names) reads via these helpers — so no
pipeline stage has to import another stage. Pure file/parse helpers: NEVER
touches the network.
"""

from __future__ import annotations

from common import read_json


def query_terms(cfg: dict) -> list[str]:
    """Resolve the configured queries from config."""
    return list(cfg["data_source"]["queries"])


def cache_path(cfg: dict, query: str):
    """Path to one query's raw cache file (creates data/raw/ if needed)."""
    raw_dir = cfg["_paths"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe = query.replace("/", "_").replace(" ", "_")
    return raw_dir / f"{safe}.json"


def extract_leaf(product: dict) -> dict | None:
    """Keep only the fields we need. brandName kept for later scrubbing."""
    dan = product.get("dan")
    title = product.get("title")
    if not dan or not title:
        return None
    # category from tileData/trackingData is unreliable per-product; the
    # category facet is search-level. We tag with the query term downstream.
    return {
        "dan": dan,
        "gtin": product.get("gtin"),
        "title": title,
        "brand": product.get("brandName"),
    }


def load_leaves(cfg: dict) -> list[dict]:
    """Read every query cache that exists on disk, dedupe by dan, return leaf
    skeletons. NO network calls — missing queries are skipped and reported."""
    queries = query_terms(cfg)
    seen: dict[int, dict] = {}
    missing = []
    for query in queries:
        path = cache_path(cfg, query)
        if not path.exists():
            missing.append(query)
            continue
        for product in read_json(path):
            leaf = extract_leaf(product)
            if leaf is None:
                continue
            leaf.setdefault("query", query)
            seen.setdefault(leaf["dan"], leaf)
    if missing:
        print(
            f"  {len(missing)}/{len(queries)} queries not cached yet (skipped): "
            f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
        )
    return list(seen.values())
