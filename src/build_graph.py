"""Stage 3 — build the closed-world knowledge graph, deterministic from seed.

IN : data/stage_1_products.json + data/stage_2_names.json
OUT: data/stage_3_graph.json

Assigns invented attributes to each product, wires every hub edge per the
hub_sharing config, and applies the marke_override_rate override edges
(a fraction of products get an actual supplier ≠ their brand's supplier).
Running twice with the same seed produces a byte-identical stage_3_graph.json.

Usage:
    python src/build_graph.py [--rerun]
"""

from __future__ import annotations

from tqdm.auto import tqdm

from common import det_rng, read_json, write_json
from config import load_config


def _pick(rng, items):
    return items[rng.randrange(len(items))]


def run(cfg: dict, *, rerun: bool = False) -> dict:
    out_path = cfg["_paths"]["graph"]
    if out_path.exists() and not rerun:
        graph = read_json(out_path)
        print(f"cached ({len(graph['nodes']['produkt'])} product) -> {out_path}")
        return graph

    seed = cfg["seed"]
    products_path = cfg["_paths"]["products"]
    if not products_path.exists():
        raise SystemExit(f"{products_path} missing — run clean_products.py first")
    products = read_json(products_path)

    names_path = cfg["_paths"]["names"]
    if not names_path.exists():
        raise SystemExit(f"{names_path} missing — run gen_names.py first")
    names = read_json(names_path)
    attr = cfg["schema"]["attributes"]
    override_rate = cfg["schema"]["marke_override_rate"]

    # ── node pools with stable ids ──
    def nodes(kind, prefix):
        return [
            {"id": f"{prefix}{i:03d}", "name": nm} for i, nm in enumerate(names[kind])
        ]

    marken = nodes("marken", "M")
    lieferanten = nodes("lieferanten", "L")
    lager = nodes("lager", "W")
    regionen = nodes("regionen", "R")
    einkaeufer = nodes("einkaeufer", "E")
    teams = nodes("teams", "T")

    # ── hub→hub edges (deterministic per source id) ──
    marke_lieferant = {
        m["id"]: _pick(det_rng(seed, "bezieht_von", m["id"]), lieferanten)["id"]
        for m in marken
    }
    lieferant_lager = {
        l["id"]: _pick(det_rng(seed, "lagert_in", l["id"]), lager)["id"]
        for l in lieferanten
    }
    lager_region = {
        w["id"]: _pick(det_rng(seed, "liegt_in", w["id"]), regionen)["id"]
        for w in lager
    }
    einkaeufer_team = {
        e["id"]: _pick(det_rng(seed, "gehoert_zu", e["id"]), teams)["id"]
        for e in einkaeufer
    }

    # ── product nodes + product-level edges + invented attributes ──
    produkt_nodes, hat_marke, liefert, eingekauft_von = [], {}, {}, {}
    for p in tqdm(products, desc="graph: product", unit="prod"):
        dan = p["dan"]
        pid = f"P{dan}"
        r_marke = det_rng(seed, "hat_marke", dan)
        r_eink = det_rng(seed, "eingekauft_von", dan)
        r_attr = det_rng(seed, "attr", dan)

        marke_id = _pick(r_marke, marken)["id"]
        hat_marke[pid] = marke_id
        eingekauft_von[pid] = _pick(r_eink, einkaeufer)["id"]

        # liefert: default = brand's supplier; override_rate → a DIFFERENT
        # supplier (the override trap). A pick equal to the default is no override,
        # so sample from the others; with <2 lieferanten no override is possible.
        default_sup = marke_lieferant[marke_id]
        r_over = det_rng(seed, "override", dan)
        if r_over.random() < override_rate and len(lieferanten) > 1:
            alts = [l["id"] for l in lieferanten if l["id"] != default_sup]
            sup = _pick(r_over, alts)
        else:
            sup = default_sup
        liefert[pid] = sup

        # invented numeric attributes — drawn independently of any real value
        is_food = p.get("is_food", False)
        price_cfg = attr["einkaufspreis_eur"]
        einkaufspreis = round(
            r_attr.uniform(price_cfg["min"], price_cfg["max"]),
            attr["einkaufspreis_decimals"],
        )
        kcal = (
            r_attr.randint(attr["kcal_100g"]["min"], attr["kcal_100g"]["max"])
            if is_food
            else None
        )
        lagerbestand = r_attr.randint(
            attr["lagerbestand"]["min"], attr["lagerbestand"]["max"]
        )

        produkt_nodes.append(
            {
                "id": pid,
                "dan": dan,
                "descriptor": p["descriptor"],
                "is_food": is_food,
                "fuellmenge": p.get("fuellmenge"),
                "attrs": {
                    "einkaufspreis": einkaufspreis,
                    "kcal_100g": kcal,
                    "lagerbestand": lagerbestand,
                },
            }
        )

    # ── cross-sell (symmetric, degree from config) ──
    cs_cfg = cfg["schema"]["hub_sharing"]["cross_sell_grad"]
    all_ids = [p["id"] for p in produkt_nodes]
    cross_sell: dict[str, list[str]] = {pid: [] for pid in all_ids}
    for p in tqdm(produkt_nodes, desc="graph: cross_sell", unit="prod"):
        pid = p["id"]
        pool = [q for q in all_ids if q != pid]
        if not pool:
            continue
        r = det_rng(seed, "cross_sell", pid)
        k = min(r.randint(cs_cfg["min"], cs_cfg["max"]), len(pool))
        for partner in r.sample(pool, k):
            if partner not in cross_sell[pid]:
                cross_sell[pid].append(partner)
            if pid not in cross_sell[partner]:  # keep symmetric
                cross_sell[partner].append(pid)

    graph = {
        "nodes": {
            "produkt": produkt_nodes,
            "marke": marken,
            "lieferant": lieferanten,
            "lager": lager,
            "region": regionen,
            "einkaeufer": einkaeufer,
            "team": teams,
        },
        "edges": {
            "hat_marke": hat_marke,
            "bezieht_von": marke_lieferant,
            "liefert": liefert,
            "lagert_in": lieferant_lager,
            "liegt_in": lager_region,
            "eingekauft_von": eingekauft_von,
            "gehoert_zu": einkaeufer_team,
            "cross_sell": cross_sell,
        },
    }
    write_json(cfg["_paths"]["graph"], graph)

    n_over = sum(
        1 for pid, s in liefert.items() if s != marke_lieferant[hat_marke[pid]]
    )
    print(
        f"graph: {len(produkt_nodes)} product, {len(marken)} brand, "
        f"{len(lieferanten)} supplier, {len(lager)} warehouse, {len(regionen)} region, "
        f"{len(einkaeufer)} buyer, {len(teams)} team; "
        f"{n_over} supplier-overrides -> {cfg['_paths']['graph']}"
    )
    return graph


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun", action="store_true", help="ignore cached stage_3_graph.json"
    )
    args = parser.parse_args()
    run(load_config(), rerun=args.rerun)


if __name__ == "__main__":
    main()
