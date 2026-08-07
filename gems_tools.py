"""The two tools the model under test may call — the benchmark world.

Runs locally: no network, no credentials. Owns the GEMS fact corpus and offers
`search_database` (BM25 retrieval over German prose, optional regex pre-filter)
and `product_lookup` (structured product master data over stage_3_graph.json, a
stand-in for an enterprise product-catalog/PIM API). Your system imports this
module and merges `build_tool_catalog(...)` into its own tool registry.

`product_lookup` returns ONLY a product's direct (hop-1) fields and exactly one
product per call, so a single call never yields a whole supply chain or an
aggregate. See README "product_lookup — scope & rationale" and the
ProductCatalog docstring.

Catalog entry shape (framework-agnostic):
    {name: {"description": str, "parameters": json_schema, "handler": callable,
            "api_delay": bool}}
The handler takes an arguments dict and returns a JSON-serializable dict.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import load_config  # noqa: E402

_TOKEN_PATTERN = re.compile(r"[a-zäöüß0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class GemsCorpus:
    """BM25 index over the flat GEMS fact corpus (stage_4_corpus.jsonl).

    Each line is {doc_id, entity_type, facts, text}; retrieval runs over `text`.
    """

    def __init__(self, corpus_path: Path) -> None:
        self.doc_ids: list[str] = []
        self.texts: list[str] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_frequencies: dict[str, int] = {}
        self._average_length = 0.0
        for line in corpus_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            text = doc.get("text", "")
            tokens = _tokenize(text)
            self.doc_ids.append(doc.get("doc_id", f"doc_{len(self.doc_ids):04d}"))
            self.texts.append(text)
            self._doc_tokens.append(tokens)
            for term in set(tokens):
                self._doc_frequencies[term] = self._doc_frequencies.get(term, 0) + 1
        if self._doc_tokens:
            self._average_length = sum(map(len, self._doc_tokens)) / len(
                self._doc_tokens
            )

    def _bm25(self, index: int, query_terms: list[str], k1: float, b: float) -> float:
        tokens = self._doc_tokens[index]
        length = len(tokens)
        total_docs = len(self._doc_tokens)
        score = 0.0
        for term in query_terms:
            doc_frequency = self._doc_frequencies.get(term)
            if not doc_frequency:
                continue
            term_frequency = tokens.count(term)
            if not term_frequency:
                continue
            idf = math.log(
                (total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5) + 1.0
            )
            score += idf * (
                term_frequency
                * (k1 + 1)
                / (term_frequency + k1 * (1 - b + b * length / self._average_length))
            )
        return score

    def top_k(
        self,
        query: str,
        regex: str | None = None,
        *,
        k: int,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[dict[str, Any]]:
        """Return up to k docs ranked by BM25, optionally pre-filtered by regex.

        regex (case-insensitive) filters the candidate set first; BM25 then
        ranks the survivors. Raises re.error on an invalid pattern (caller
        turns it into an error response).
        """
        if not self._doc_tokens:
            return []
        candidates = range(len(self._doc_tokens))
        if regex:
            pattern = re.compile(regex, re.IGNORECASE)
            candidates = [i for i in candidates if pattern.search(self.texts[i])]
            if not candidates:
                return []
        query_terms = _tokenize(query)
        scored = [(self._bm25(i, query_terms, k1, b), i) for i in candidates]
        # keep positive-scoring docs; if a regex filter is active but the query
        # matches nothing, fall back to regex order so a filter alone still works
        positive = [(s, i) for s, i in scored if s > 0]
        ranked = (
            positive if positive else ([(0.0, i) for i in candidates] if regex else [])
        )
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {
                "doc_id": self.doc_ids[i],
                "text": self.texts[i],
                "score": round(score, 3),
            }
            for score, i in ranked[:k]
        ]


class ProductCatalog:
    """Structured product master data over stage_3_graph.json.

    Backs the `product_lookup` tool — a stand-in for the product-catalog / PIM
    API a real enterprise assistant would call instead of parsing prose.

    Scope is one product per call and only that product's own direct (hop-1)
    fields: brand name, delivering supplier name, purchasing buyer name,
    purchase price, kcal per 100 g, stock level. The deeper chain
    (brand→sources-from-supplier, supplier→warehouse, warehouse→region,
    buyer→team) is reachable only through the prose corpus
    (`search_database`), and aggregation questions need one call per product
    plus the arithmetic. See README "product_lookup — scope & rationale".
    """

    def __init__(self, graph_path: Path) -> None:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        nodes, edges = graph["nodes"], graph["edges"]
        name = {
            n["id"]: n["name"]
            for k in ("marke", "lieferant", "lager", "region", "einkaeufer", "team")
            for n in nodes[k]
        }
        self._products: list[dict[str, Any]] = []
        self._by_dan: dict[str, dict[str, Any]] = {}
        self._tokens: list[list[str]] = []
        for p in nodes["produkt"]:
            pid = p["id"]
            attrs = p.get("attrs", {})
            record = {
                "descriptor": p["descriptor"],
                "dan": p.get("dan"),
                # hop-1 catalog fields only — names, never the deeper chain
                "marke": name.get(edges["hat_marke"].get(pid)),
                "lieferant": name.get(edges["liefert"].get(pid)),
                "eingekauft_von": name.get(edges["eingekauft_von"].get(pid)),
                "einkaufspreis_eur": attrs.get("einkaufspreis"),
                "kcal_100g": attrs.get("kcal_100g"),
                "lagerbestand": attrs.get("lagerbestand"),
            }
            record = {k: v for k, v in record.items() if v is not None}
            self._products.append(record)
            self._tokens.append(_tokenize(p["descriptor"]))
            if p.get("dan") is not None:
                self._by_dan[str(p["dan"])] = record

    def lookup(self, query: str) -> dict[str, Any]:
        """Resolve a product name (or DAN) to its direct fields.

        Confident single match → {"status":"success","product":{...}}. When the
        query is ambiguous (several products match comparably), returns the
        candidate descriptors so the caller can disambiguate instead of guessing.
        """
        q = query.strip()
        if q in self._by_dan:
            return {"status": "success", "product": self._by_dan[q]}
        digits = re.sub(r"\D", "", q)
        if digits and digits in self._by_dan:
            return {"status": "success", "product": self._by_dan[digits]}

        q_tokens = set(_tokenize(q))
        if not q_tokens:
            return {"status": "error", "message": "query is required"}
        scored: list[tuple[float, int]] = []
        for i, tokens in enumerate(self._tokens):
            doc_tokens = set(tokens)
            if not doc_tokens:
                continue
            overlap = len(q_tokens & doc_tokens)
            if not overlap:
                continue
            # coverage of the product's own name tokens — rewards a query that
            # names the product fully over one that only clips a shared word
            scored.append((overlap / len(doc_tokens), i))
        if not scored:
            return {"status": "success", "product": None}
        scored.sort(key=lambda pair: pair[0], reverse=True)
        best = scored[0][0]
        top = [i for s, i in scored if s >= best - 1e-9]
        if len(top) > 1:
            return {
                "status": "ambiguous",
                "candidates": [self._products[i]["descriptor"] for i in top[:8]],
            }
        return {"status": "success", "product": self._products[top[0]]}


def build_tool_catalog(**_ignored: Any) -> dict[str, dict[str, Any]]:
    """Return the tool catalog (search_database + product_lookup).

    Data locations come from `config.yaml` (`paths.corpus`, `paths.graph`); the
    German tool descriptions, parameter schemas and response hints come from
    `prompts.yaml` (`paths.prompts`), so every system under test sees identical
    tool text. The corpus and product catalog are loaded lazily on the first
    tool call, so your system can start before generation has run.

    Any keyword arguments are accepted and ignored, so a host that passes its
    own tool-registry settings to every builder can call this unchanged.
    """
    config = load_config()
    prompts = yaml.safe_load(config["_paths"]["prompts"].read_text(encoding="utf-8"))
    tool_prompts = prompts["tools"]
    notes = prompts["responses"]
    retrieval_config = config["retrieval"]
    top_k = int(retrieval_config["top_k"])
    corpus_path = config["_paths"]["corpus"]
    graph_path = config["_paths"]["graph"]
    corpus: GemsCorpus | None = None
    catalog: ProductCatalog | None = None

    product_lookup_description = tool_prompts["product_lookup"]["description"]
    product_lookup_parameters = tool_prompts["product_lookup"]["parameters"]

    def product_lookup(arguments: dict[str, Any]) -> dict[str, Any]:
        nonlocal catalog
        product = str(arguments.get("product") or "").strip()
        if not product:
            return {"status": "error", "message": "product is required"}
        if not graph_path.is_file():
            return {
                "status": "error",
                "message": f"catalog not found at {graph_path} — run the generator first",
            }
        if catalog is None:
            catalog = ProductCatalog(graph_path)
        result = catalog.lookup(product)
        # The disambiguation hints are model-facing text and live in prompts.yaml,
        # so ProductCatalog stays a pure data class.
        if result["status"] == "ambiguous":
            result["note"] = notes["product_ambiguous"]
        elif result["status"] == "success" and result["product"] is None:
            result["note"] = notes["product_not_found"]
        return result

    search_tool_description = tool_prompts["search_database"]["description"].format(
        top_k=top_k
    )
    search_tool_parameters = tool_prompts["search_database"]["parameters"]

    def search_database(arguments: dict[str, Any]) -> dict[str, Any]:
        nonlocal corpus
        query = str(arguments.get("query") or "").strip()
        regex = arguments.get("regex")
        regex = str(regex).strip() if regex else None
        if not query and not regex:
            return {"status": "error", "message": "query is required"}
        if not corpus_path.is_file():
            return {
                "status": "error",
                "message": f"corpus not found at {corpus_path} — run the generator first",
            }
        if corpus is None:
            corpus = GemsCorpus(corpus_path)
        try:
            results = corpus.top_k(query, regex, k=top_k)
        except re.error as exc:
            return {"status": "error", "message": f"invalid regex: {exc}"}
        if not results:
            return {
                "status": "success",
                "results": [],
                "count": 0,
                "note": notes["no_document"],
            }
        return {"status": "success", "results": results, "count": len(results)}

    return {
        "product_lookup": {
            "description": product_lookup_description,
            "parameters": product_lookup_parameters,
            "handler": product_lookup,
            "api_delay": True,
        },
        "search_database": {
            "description": search_tool_description,
            "parameters": search_tool_parameters,
            "handler": search_database,
            "api_delay": True,
        },
    }
