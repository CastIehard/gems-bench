"""Stage 2 — generate invented name pools via the OSS LLM (`llm.model`).

IN : data/raw/*.json           (via raw_cache.load_leaves — real brands, anti-leak)
OUT: data/stage_2_names.json

Deterministically cached — only regenerates when missing or --rerun. Marken are
checked against real brand names so no real brand leaks.

Usage:
    python src/gen_names.py [--rerun]
"""

from __future__ import annotations

import argparse

import orjson
from tqdm.auto import tqdm

import raw_cache
from common import llm_chat, read_json, write_json
from config import load_config


def _real_brands(cfg: dict) -> set[str]:
    """Collect the real brand names from cached raw products (anti-leak set)."""
    return {
        leaf["brand"].strip().lower()
        for leaf in raw_cache.load_leaves(cfg)
        if leaf.get("brand")
    }


def _generate(cfg: dict, kind: str, n: int) -> list[str]:
    prompt = cfg["names"]["prompts"][kind].format(n=n)
    content = llm_chat(
        cfg,
        [
            {
                "role": "system",
                "content": "Du gibst ausschließlich gültiges JSON zurück.",
            },
            {
                "role": "user",
                "content": prompt + '\n\nAntwortformat: {"names": ["...", "..."]}',
            },
        ],
        json_mode=True,
    )
    data = orjson.loads(content)
    names = data.get("names", data if isinstance(data, list) else [])
    # dedupe preserving order
    seen, out = set(), []
    for name in names:
        key = str(name).strip()
        if key and key.lower() not in seen:
            seen.add(key.lower())
            out.append(key)
    return out


def run(cfg: dict, *, rerun: bool = False) -> dict:
    path = cfg["_paths"]["names"]
    if path.exists() and not rerun:
        pools = read_json(path)
        print(
            f"cached names ({', '.join(f'{k}:{len(v)}' for k, v in pools.items())}) -> {path}"
        )
        return pools

    real = _real_brands(cfg)
    pools: dict[str, list[str]] = {}
    for kind, n in tqdm(cfg["names"]["counts"].items(), desc="gen names", unit="pool"):
        names = _generate(cfg, kind, n)
        if kind == "marken":
            names = [x for x in names if x.lower() not in real]
        pools[kind] = names
        print(f"  {kind}: {len(names)} generated")
    write_json(path, pools)
    print(f"-> {path}")
    return pools


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    run(load_config(), rerun=args.rerun)


if __name__ == "__main__":
    main()
