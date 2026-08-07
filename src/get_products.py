"""Stage 0 (fetch) — pull products from the public DM search API into the raw cache.

Public endpoint (no auth): product-search.services.dmtech.com/de/search/static
Returns 302 -> /crawl (httpx follows redirects). Each query is cached
independently in data/raw/{query}.json (never overwritten unless rerun=True).

IN : config queries + network
OUT: data/raw/{query}.json  (one raw API dump per query)

Downstream stages read the cache via `raw_cache`, never this module — so a
later stage can never trigger a fetch.

Usage:
    python src/get_products.py                 # all queries from config
    python src/get_products.py --query shampoo # single query, cache only
"""

from __future__ import annotations

import argparse
import time

import httpx
from tqdm.auto import tqdm

import raw_cache
from common import read_json, write_json
from config import load_config


def search(cfg: dict, query: str) -> list[dict]:
    """One search call. Returns raw `products`."""
    ds = cfg["data_source"]
    resp = httpx.get(
        ds["search_url"],
        params={"query": query},
        headers={
            "User-Agent": ds["user_agent"],
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
            "Referer": "https://www.dm.de/",
            "Origin": "https://www.dm.de",
        },
        timeout=ds["timeout_s"],
        follow_redirects=True,
    )
    if resp.status_code == 429:
        retry_after = resp.headers.get("retry-after")
        suffix = f" retry-after={retry_after}s" if retry_after else ""
        print(
            f"429 for query '{query}' — skipping; increase throttle_s in config.{suffix}"
        )
        return []
    resp.raise_for_status()
    return resp.json().get("products", [])


def fetch_query(cfg: dict, query: str, *, rerun: bool = False) -> list[dict]:
    """Fetch one query, caching raw JSON. Never re-fetches unless rerun=True."""
    path = raw_cache.cache_path(cfg, query)
    if path.exists() and not rerun:
        return read_json(path)
    products = search(cfg, query)
    write_json(path, products)
    return products


def run(cfg: dict, *, rerun: bool = False) -> list[dict]:
    """Fetch all configured queries into the raw cache, return leaf skeletons.

    Already-cached queries are skipped (no network call, no throttle);
    rerun=True re-fetches everything. Returns the deduped leaves for preview —
    the durable artifact is the per-query files in data/raw/."""
    ds = cfg["data_source"]
    throttle = ds["throttle_s"]
    queries = raw_cache.query_terms(cfg)
    for i, query in enumerate(tqdm(queries, desc="fetch products", unit="query")):
        path = raw_cache.cache_path(cfg, query)
        if i > 0 and not (path.exists() and not rerun):
            time.sleep(throttle)  # only throttle real network calls
        fetch_query(cfg, query, rerun=rerun)
    leaves = raw_cache.load_leaves(cfg)
    print(f"collected {len(leaves)} unique products across {len(queries)} queries")
    return leaves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="fetch a single query and cache it")
    parser.add_argument("--rerun", action="store_true", help="ignore cache, re-fetch")
    args = parser.parse_args()
    cfg = load_config()
    if args.query:
        products = fetch_query(cfg, args.query, rerun=args.rerun)
        print(
            f"{args.query}: {len(products)} products -> "
            f"{raw_cache.cache_path(cfg, args.query)}"
        )
    else:
        run(cfg, rerun=args.rerun)


if __name__ == "__main__":
    main()
