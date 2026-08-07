"""Shared helpers for GEMS-Bench generator scripts.

Loads .env.local from the repository root at import time, exposes the single OSS
LLM (llm.model), deterministic RNG keyed off the global seed, and orjson I/O
helpers.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import httpx
import orjson
from dotenv import load_dotenv

from scoring import _fuzzy_contains, normalize_name

GEMS_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = GEMS_DIR / ".env.local"

# Load before any script reads os.environ.
load_dotenv(ENV_PATH)


# ── I/O ──────────────────────────────────────────────────────────────────────
def read_json(path: str | Path) -> Any:
    return orjson.loads(Path(path).read_bytes())


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(
        orjson.dumps(obj, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS)
    )


def read_jsonl(path: str | Path) -> list[Any]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [orjson.loads(line) for line in lines if line.strip()]


def write_jsonl(path: str | Path, rows: list[Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        for row in rows:
            fh.write(orjson.dumps(row))
            fh.write(b"\n")


# ── Verbatim-preservation check (LLM rewrite stages: corpus weave, spoken) ──
def normalize_ws(s: str) -> str:
    """Loose match key: strip non-breaking spaces, collapse whitespace, lowercase.
    Lets minor formatting diffs (casing, doubled/nbsp spaces) pass the token check
    instead of triggering needless reruns/fallbacks."""
    return re.sub(r"\s+", " ", (s or "").replace("\xa0", " ")).lower().strip()


def text_has_all(text: str, required: list[str]) -> bool:
    """True if every required string survives (case/whitespace-insensitive
    substring) in text — the shared verbatim gate for any LLM rewrite step that
    must not drop/alter facts (corpus weave, spoken-question rewrite)."""
    t = normalize_ws(text)
    return all(normalize_ws(r) in t for r in required)


# Fill quantity in a product title: amount + unit, e.g. "500 ml", "400 g",
# "1,5 kg", "20 St". Used by clean_products (parse + size-variant dedup key).
FUELLMENGE_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(ml|l|g|kg|st|stück|stk)\b", re.IGNORECASE
)


# Units/measure tokens carry no identifying signal and get spoken/written
# inconsistently ("ml" -> "Milliliter", "150 ml" <-> "hundertfünfzig
# Milliliter"), so they're dropped before matching an entity against a text —
# the distinctive product tokens are what must actually appear.
_UNIT_TOKENS = {
    "ml",
    "milliliter",
    "g",
    "gramm",
    "kg",
    "kilogramm",
    "l",
    "liter",
    "st",
    "stk",
    "stueck",
    "stück",
    "prozent",
    "cm",
    "mm",
    "mg",
}


def distinctive_tokens(entity: str) -> list[str]:
    """Identifying tokens of an entity: alphabetic, length >= 4, not a unit.

    Product descriptors carry their signal in the content words ("Rasierschaum",
    "Lychee"), not the numbers/units — those render inconsistently across
    TTS/STT/rewrite. Matching on these tokens is robust to phrasing drift."""
    words = re.findall(r"[a-zäöüß]+", entity.lower())
    return [w for w in words if len(w) >= 4 and w not in _UNIT_TOKENS]


def entity_present(entity: str, text_norm: str, ratio: float, coverage: float) -> bool:
    """True if `entity` is named in text_norm (token-coverage, fuzzy-tolerant).

    An entity counts as present when at least `coverage` of its distinctive
    tokens appear (exact substring, or fuzzy for STT/rewrite drift). Falls back
    to whole-string containment when an entity has no distinctive tokens (e.g.
    an all-numeric/short descriptor). `text_norm` must already be
    normalize_name()-normalized."""
    toks = distinctive_tokens(entity)
    if not toks:
        return normalize_name(entity) in text_norm
    hits = 0
    for tok in toks:
        tn = normalize_name(tok)
        if tn and (tn in text_norm or _fuzzy_contains(tn, text_norm, ratio)):
            hits += 1
    return (hits / len(toks)) >= coverage


# ── Deterministic RNG ────────────────────────────────────────────────────────
def det_rng(seed: int, *keys: Any) -> random.Random:
    """A reproducible random.Random keyed off (global_seed, *keys).

    Same seed + same keys → same stream, on any machine, across runs. Use for
    every invented attribute / hub assignment so the whole benchmark is
    byte-reproducible (no Math.random-style nondeterminism).
    """
    material = "|".join([str(seed), *(str(k) for k in keys)])
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


# ── LLM (single OSS model, OpenAI-v1 compatible) ─────────────────────────────
def _llm_creds(cfg: dict) -> tuple[str, str, str]:
    llm = cfg["llm"]
    endpoint = (os.environ.get(llm["endpoint_var"]) or "").strip()
    api_key = (os.environ.get(llm["key_var"]) or "").strip()
    if not endpoint or not api_key:
        raise SystemExit(
            f"{llm['endpoint_var']} / {llm['key_var']} missing (looked in {ENV_PATH})"
        )
    return endpoint, api_key, llm["model"]


# Transient — worth a retry (connection blips, endpoint overload/queueing).
_RETRYABLE_TRANSPORT_ERRORS = (
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
)
# Transient server-side — worth a retry. NOT retried: 4xx auth/bad-request
# (wrong model name, bad key) — those are permanent and should surface fast.
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def llm_chat(
    cfg: dict,
    messages: list[dict],
    *,
    model: str | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
) -> str:
    """Call an OSS chat model on AZURE_ENDPOINT; returns the message content.

    All models live behind the same endpoint/key (from cfg["llm"]); `model`
    overrides only the model name — used by the closed-book gate to poll several
    models (Kimi, DeepSeek, gpt-oss) through one endpoint. None → cfg default.

    Retries transient network/server errors (llm.retry.max_attempts, linear
    backoff of llm.retry.backoff_s * attempt) before raising — the endpoint
    queues/drops connections under concurrent load, which isn't a real failure.
    """
    endpoint, api_key, default_model = _llm_creds(cfg)
    body: dict[str, Any] = {
        "model": model or default_model,
        "messages": messages,
        "temperature": (
            cfg["llm"]["temperature"] if temperature is None else temperature
        ),
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if cfg["llm"]["reasoning_effort"]:
        body["reasoning_effort"] = cfg["llm"]["reasoning_effort"]

    retry_cfg = cfg["llm"]["retry"]
    max_attempts = retry_cfg["max_attempts"]
    backoff_s = retry_cfg["backoff_s"]
    label = model or default_model

    for attempt in range(1, max_attempts + 1):
        try:
            response = httpx.post(
                endpoint,
                headers={"api-key": api_key},
                json=body,
                timeout=cfg["llm"]["timeout_s"],
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except _RETRYABLE_TRANSPORT_ERRORS as exc:
            if attempt >= max_attempts:
                raise
            wait = backoff_s * attempt
            print(
                f"  llm_chat[{label}] retry {attempt}/{max_attempts} "
                f"after {exc.__class__.__name__} — backing off {wait:.0f}s"
            )
            time.sleep(wait)
        except httpx.HTTPStatusError as exc:
            # raise_for_status() alone drops the response body; the body
            # usually has the actual reason (rate limit, content filter, bad
            # model name).
            detailed = httpx.HTTPStatusError(
                f"{exc} | body: {response.text[:2000]}",
                request=exc.request,
                response=exc.response,
            )
            if (
                exc.response.status_code not in _RETRYABLE_STATUS
                or attempt >= max_attempts
            ):
                raise detailed from None
            wait = backoff_s * attempt
            print(
                f"  llm_chat[{label}] retry {attempt}/{max_attempts} "
                f"after HTTP {exc.response.status_code} — backing off {wait:.0f}s"
            )
            time.sleep(wait)
    raise RuntimeError("unreachable")  # loop always returns or raises
