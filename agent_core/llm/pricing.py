"""Cloud LLM price discovery & storage (plan ARCH item: pricing).

Price discovery strategy
------------------------
The authoritative OpenCode Go price table lives on the public docs page
``https://opencode.ai/docs/en/go/`` (the "prices per 1M tokens" table under the
*Usage limits* section).  That page is the source of truth for the keyed
``opencode-go/<id>`` model prices.

Because the web client is SSRF-restricted to query-only DuckDuckGo (decision
#006 -- protects local services), we cannot issue an arbitrary GET to that
page from the agent runtime.  We therefore:

1. Ships an **authoritative baseline** table below, transcribed from the
   opencode.ai docs price table and keyed to the ``opencode-go/<id>`` ids that
   ``OpencodeProvider.list_models`` actually produces.  These are
   ``verified: True, source: "opencode-docs"``.
2. Optionally **refresh from the web** via ``fetch_cloud_prices``:
     * primary: an allowlisted single-URL fetch of the docs page
       (``_fetch_opencode_docs``, host allowlist = ``opencode.ai`` only) which
       parses the price table -> ``verified: True``;
     * fallback: the existing query-only ``search_ddg`` snippet extractor for
       any cloud model not covered by the docs table -> ``verified: False``.
   Every number extracted from the web is treated as UNTRUSTED (decision
   #004/#005); the baseline stands when parsing fails.
3. Stores refreshed values in a local JSON cache (``pricing_cache.json``) that
   overrides the baseline on key conflict, so a successful web refresh wins.

Free/zen and local (LM Studio / llama.cpp) tiers are cost ``0.0`` by
definition, so the "cheapest failover" strategy stays correct even before any
price is fetched.

Cadence: a 24h TTL (``_PRICE_TTL_SECONDS``) means each cloud model is only
re-fetched at most once per day; ``fetch_cloud_prices(fetch=True, force=True)``
ignores the TTL for an explicit on-demand refresh (e.g. ``model prices
--refresh``).
"""
from __future__ import annotations

import json
import re
import time as _time
from pathlib import Path
from typing import Any

from agent_core.constants import _ZEN_TIER_PREFIXES

#: Local cache for discovered/refreshed prices (overrides BASELINE_PRICES).
_PRICING_CACHE_PATH = Path(__file__).parent / "pricing_cache.json"

#: Refresh TTL -- at most one web refresh per model per day (decision: daily).
_PRICE_TTL_SECONDS = 24 * 3600

#: Only host we are allowed to fetch directly (decision #006: no arbitrary
#: URL fetching; the docs price table is the single allowlisted source).
_DOCS_HOST = "opencode.ai"
_DOCS_URL = "https://opencode.ai/docs/en/go/"

#: Baseline prices -- AUTHORITATIVE, transcribed from the opencode.ai docs
#: "prices per 1M tokens" table.  Keyed by ``opencode-go/<id>`` (the id
#: ``OpencodeProvider.list_models`` returns).  Values are USD per 1M tokens.
#: For tiered rows (token-count or peak/off-peak) we store the common-case
#: LOWER tier (<= threshold / off-peak); the higher tier is noted in comments.
#: Free/zen/local tiers are 0.0 by definition.
BASELINE_PRICES: dict[str, dict[str, Any]] = {
    # Keyless opencode-zen FREE tier -- free by definition.
    "opencode-zen/hy3-free": {
        "prompt_per_1m": 0.0, "completion_per_1m": 0.0,
        "verified": True, "source": "free-tier",
    },
    # --- OpenCode Go keyed tier (opencode.ai docs, prices per 1M tokens) ---
    "opencode-go/grok-4.6": {  # <=200K ; >200K is $4/$12
        "prompt_per_1m": 2.00, "completion_per_1m": 6.00,
        "cached_read_per_1m": 0.50,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/gpt-5.6-luna": {  # <=272K ; >272K is $0.40/$1.80
        "prompt_per_1m": 0.20, "completion_per_1m": 1.20,
        "cached_read_per_1m": 0.02, "cached_write_per_1m": 0.25,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/glm-5.3-flash": {
        "prompt_per_1m": 0.15, "completion_per_1m": 0.50,
        "cached_read_per_1m": 0.03,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/glm-5.3": {
        "prompt_per_1m": 1.40, "completion_per_1m": 4.40,
        "cached_read_per_1m": 0.26,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/glm-5.2": {
        "prompt_per_1m": 1.40, "completion_per_1m": 4.40,
        "cached_read_per_1m": 0.26,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/glm-5.1": {
        "prompt_per_1m": 1.40, "completion_per_1m": 4.40,
        "cached_read_per_1m": 0.26,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/kimi-k3": {
        "prompt_per_1m": 3.00, "completion_per_1m": 15.00,
        "cached_read_per_1m": 0.30,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/kimi-k2.7-code": {
        "prompt_per_1m": 0.95, "completion_per_1m": 4.00,
        "cached_read_per_1m": 0.19,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/kimi-k2.6": {
        "prompt_per_1m": 0.95, "completion_per_1m": 4.00,
        "cached_read_per_1m": 0.16,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/longcat-2.0": {
        "prompt_per_1m": 0.30, "completion_per_1m": 1.20,
        "cached_read_per_1m": 0.006,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/mimo-v2.5": {
        "prompt_per_1m": 0.14, "completion_per_1m": 0.28,
        "cached_read_per_1m": 0.0028,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/mimo-v2.5-pro": {
        "prompt_per_1m": 0.435, "completion_per_1m": 0.87,
        "cached_read_per_1m": 0.003625,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/minimax-m3": {
        "prompt_per_1m": 0.30, "completion_per_1m": 1.20,
        "cached_read_per_1m": 0.06,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/minimax-m2.7": {
        "prompt_per_1m": 0.30, "completion_per_1m": 1.20,
        "cached_read_per_1m": 0.06, "cached_write_per_1m": 0.375,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/minimax-m2.5": {
        "prompt_per_1m": 0.30, "completion_per_1m": 1.20,
        "cached_read_per_1m": 0.06, "cached_write_per_1m": 0.375,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/muse-spark-1.3-contributor": {
        "prompt_per_1m": 0.10, "completion_per_1m": 0.20,
        "cached_read_per_1m": 0.002,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/muse-spark-1.2-contributor": {
        "prompt_per_1m": 0.10, "completion_per_1m": 0.20,
        "cached_read_per_1m": 0.002,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/qwen3.8-max": {
        "prompt_per_1m": 2.00, "completion_per_1m": 6.00,
        "cached_read_per_1m": 0.25, "cached_write_per_1m": 2.50,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/qwen3.8-flash": {
        "prompt_per_1m": 0.15, "completion_per_1m": 0.47,
        "cached_read_per_1m": 0.016, "cached_write_per_1m": 0.20,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/qwen3.7-max": {
        "prompt_per_1m": 2.50, "completion_per_1m": 7.50,
        "cached_read_per_1m": 0.50, "cached_write_per_1m": 3.125,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/qwen3.7-plus": {  # <=256K ; >256K is $1.20/$4.80
        "prompt_per_1m": 0.40, "completion_per_1m": 1.60,
        "cached_read_per_1m": 0.04, "cached_write_per_1m": 0.50,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/qwen3.6-plus": {  # <=256K ; >256K is $2.00/$6.00
        "prompt_per_1m": 0.50, "completion_per_1m": 3.00,
        "cached_read_per_1m": 0.05, "cached_write_per_1m": 0.625,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/deepseek-v4-pro": {  # off-peak ; peak is $1.32/$3.96
        "prompt_per_1m": 0.66, "completion_per_1m": 1.98,
        "cached_read_per_1m": 0.022,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/deepseek-v4-flash": {  # off-peak ; peak is $0.44/$1.32
        "prompt_per_1m": 0.22, "completion_per_1m": 0.66,
        "cached_read_per_1m": 0.007,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/deepseek-v4-flash-vision-exp": {  # off-peak ; peak is $0.44/$1.32
        "prompt_per_1m": 0.22, "completion_per_1m": 0.66,
        "cached_read_per_1m": 0.007,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/hy4-preview": {
        "prompt_per_1m": 0.834, "completion_per_1m": 2.501,
        "cached_read_per_1m": 0.042,
        "verified": True, "source": "opencode-docs",
    },
    "opencode-go/hy3": {
        "prompt_per_1m": 0.14, "completion_per_1m": 0.58,
        "cached_read_per_1m": 0.035,
        "verified": True, "source": "opencode-docs",
    },
}

#: Matches a ``$<number>`` amount anywhere in a snippet.
_AMOUNT_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)")

#: Map human model names (as they appear in the docs price table) to the
#: ``opencode-go/<id>`` routing id.  Built from the docs "Endpoints" table.
_DOCS_NAME_TO_ID: dict[str, str] = {
    "Grok 4.6": "opencode-go/grok-4.6",
    "GPT 5.6 Luna": "opencode-go/gpt-5.6-luna",
    "GLM-5.3-Flash": "opencode-go/glm-5.3-flash",
    "GLM-5.3": "opencode-go/glm-5.3",
    "GLM-5.2": "opencode-go/glm-5.2",
    "GLM-5.1": "opencode-go/glm-5.1",
    "Kimi K3": "opencode-go/kimi-k3",
    "Kimi K2.7 Code": "opencode-go/kimi-k2.7-code",
    "Kimi K2.6": "opencode-go/kimi-k2.6",
    "LongCat-2.0": "opencode-go/longcat-2.0",
    "MiMo V2.5": "opencode-go/mimo-v2.5",
    "MiMo V2.5 Pro": "opencode-go/mimo-v2.5-pro",
    "MiniMax M3": "opencode-go/minimax-m3",
    "MiniMax M2.7": "opencode-go/minimax-m2.7",
    "MiniMax M2.5": "opencode-go/minimax-m2.5",
    "Muse Spark 1.3 Contributor": "opencode-go/muse-spark-1.3-contributor",
    "Muse Spark 1.2 Contributor": "opencode-go/muse-spark-1.2-contributor",
    "Qwen3.8 Max": "opencode-go/qwen3.8-max",
    "Qwen3.8 Flash": "opencode-go/qwen3.8-flash",
    "Qwen3.7 Max": "opencode-go/qwen3.7-max",
    "Qwen3.7 Plus": "opencode-go/qwen3.7-plus",
    "Qwen3.6 Plus": "opencode-go/qwen3.6-plus",
    "DeepSeek V4 Pro": "opencode-go/deepseek-v4-pro",
    "DeepSeek V4 Flash": "opencode-go/deepseek-v4-flash",
    "DeepSeek V4 Flash Vision Exp": "opencode-go/deepseek-v4-flash-vision-exp",
    "Hy4 preview": "opencode-go/hy4-preview",
    "Hy3": "opencode-go/hy3",
}


def _is_free_tier(model_id: str) -> bool:
    """True for keyless free-tier model ids (``-free`` / ``:free`` / zen prefix)."""
    m = (model_id or "").lower()
    if m.endswith("-free") or ":free" in m:
        return True
    return any(m.startswith(p) for p in _ZEN_TIER_PREFIXES)


def _is_local_provider(provider: str) -> bool:
    """True for locally-served (free) providers."""
    return provider in ("lmstudio", "llama")


def _extract_prices_from_snippet(snippet: str) -> tuple[float, float] | None:
    """Pull (prompt, completion) per-1M prices from an UNTRUSTED snippet.

    Returns ``None`` when the snippet doesn't mention tokens.  When a single
    amount is present it is used for both in/out; when two are present the
    first is treated as input/prompt and the second as output/completion.
    """
    if not snippet or "token" not in snippet.lower():
        return None
    amounts = [float(a) for a in _AMOUNT_RE.findall(snippet)]
    if not amounts:
        return None
    if len(amounts) >= 2:
        return amounts[0], amounts[1]
    return amounts[0], amounts[0]


def load_prices() -> dict[str, dict[str, Any]]:
    """Return the effective price table: BASELINE merged with the local cache.

    The cache wins on key conflict so refreshed web values override baselines.
    """
    table: dict[str, dict[str, Any]] = {k: dict(v) for k, v in BASELINE_PRICES.items()}
    try:
        if _PRICING_CACHE_PATH.exists():
            cached = json.loads(_PRICING_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(cached, dict):
                for key, value in cached.items():
                    if isinstance(value, dict):
                        table[key] = value
    except Exception:
        pass
    return table


def save_prices(table: dict[str, dict[str, Any]]) -> None:
    """Atomically write the price table to the local cache."""
    try:
        tmp = _PRICING_CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(table, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(_PRICING_CACHE_PATH)
    except Exception:
        pass


def cost_per_token(model_id: str, provider: str = "") -> float:
    """Blended per-token cost (USD). Free/zen/local tiers return ``0.0``."""
    if _is_free_tier(model_id) or _is_local_provider(provider):
        return 0.0
    entry = load_prices().get(model_id)
    if not entry:
        return 0.0
    per_1m = (float(entry.get("prompt_per_1m") or 0.0) + float(entry.get("completion_per_1m") or 0.0)) / 2.0
    return per_1m / 1_000_000.0


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model_id: str,
    provider: str = "",
) -> float:
    """Estimated USD cost for a call. Free/zen/local tiers return ``0.0``."""
    if _is_free_tier(model_id) or _is_local_provider(provider):
        return 0.0
    entry = load_prices().get(model_id)
    if not entry:
        return 0.0
    p = float(entry.get("prompt_per_1m") or 0.0)
    c = float(entry.get("completion_per_1m") or 0.0)
    return (prompt_tokens * p + completion_tokens * c) / 1_000_000.0


def cheapest_opencode_go_model() -> str | None:
    """Return the cheapest paid ``opencode-go/<id>`` model from the price table.

    Used by the ``model --cheapest`` / ``model cheapest`` auto-selection so the
    agent can route to the lowest-cost cloud model without the user naming it.
    Returns ``None`` when no priced opencode-go model is available (e.g. the
    price table is empty and offline).  Comparison uses :func:`cost_per_token`
    (blended in/out per-token USD), so a model with a lower blended cost wins
    even if it is not the cheapest on the input side alone.  Ties are broken by
    the stable key order of :func:`load_prices` (deterministic, not network-order).

    Models requiring data-policy opt-in (``-contributor`` suffix) are excluded
    because they 403 without explicit workspace configuration.
    """
    cloud = {
        k: v for k, v in load_prices().items()
        if k.startswith("opencode-go/")
        and not _is_free_tier(k)
        and "-contributor" not in k
    }
    if not cloud:
        return None
    return min(
        cloud,
        key=lambda k: (cost_per_token(k, "opencode"), k),
    )


def discover_cloud_models() -> list[str]:
    """Authoritative cloud model ids via the OpenCode ``/models`` API.

    Enumerates both the keyless opencode-zen FREE tier and the keyed
    opencode-go tier.  Returns de-duplicated ``provider/model`` id strings.
    Network errors yield an empty list for that seed (never raises).
    """
    from .opencode_provider import OpencodeProvider

    out: list[str] = []
    for seed in ("opencode-zen/hy3-free", "opencode-go/deepseek-v4-flash"):
        try:
            provider = OpencodeProvider(model_name=seed, api_key="", read_store=False)
            out.extend(provider.list_models())
        except Exception:
            continue
    seen: set[str] = set()
    unique: list[str] = []
    for model in out:
        if model not in seen:
            seen.add(model)
            unique.append(model)
    return unique


def _strip_html_to_rows(html: str) -> list[list[str]]:
    """Turn an HTML table (or tag soup) into a list of cell-row lists.

    Robust to the docs page's markup: every ``<tr>`` becomes one row, every
    ``<t[dh]>`` one cell.  Cell text is tag-stripped and whitespace-collapsed.
    """
    rows: list[list[str]] = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.DOTALL | re.IGNORECASE)
        row = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", c)).strip() for c in cells]
        rows.append(row)
    return rows


def _base_docs_name(name: str) -> str:
    """Strip a parenthetical tier suffix so a row name maps to a table key.

    ``"Grok 4.6 (≤ 200K tokens)"`` -> ``"Grok 4.6"``; ``"DeepSeek V4 Pro
    (Off-Peak)"`` -> ``"DeepSeek V4 Pro"``.  Non-suffixed names pass through.
    """
    return re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()


def _money(cell: str) -> float | None:
    """Parse a price cell (``$2.00``, ``-``, ``$0.50``) into a float or None."""
    cell = cell.strip().lstrip("$").strip()
    if cell in ("-", "", "—", "–"):
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", cell)
    return float(m.group(1)) if m else None


def parse_opencode_docs_prices(html: str) -> dict[str, dict[str, Any]]:
    """Parse the opencode.ai docs "prices per 1M tokens" table.

    Returns a price dict keyed by ``opencode-go/<id>`` (matching
    ``OpencodeProvider.list_models`` ids).  Tiered rows (token-count or
    peak/off-peak) share a base name; only the common-case LOWER tier is kept.
    Unknown model names are skipped.  Never raises -- returns ``{}`` on parse
    failure.
    """
    out: dict[str, dict[str, Any]] = {}
    try:
        rows = _strip_html_to_rows(html)
    except Exception:
        return out

    for row in rows:
        # Need at least: Name, Input, Output, Cached Read, Cached Write, Usage.
        if len(row) < 6:
            continue
        base = _base_docs_name(row[0].strip())
        if base not in _DOCS_NAME_TO_ID:
            continue
        model_id = _DOCS_NAME_TO_ID[base]
        name = row[0]
        # Keep the lower tier for tiered models (order-independent): a row is
        # "lower" when it carries a lower-bound / off-peak marker, otherwise it
        # is the higher tier and we keep whatever was stored first.
        is_lower = ("≤" in name) or ("Off-Peak" in name) or ("off-peak" in name)
        if model_id in out and not is_lower:
            continue
        try:
            prompt = _money(row[1])
            completion = _money(row[2])
            if prompt is None or completion is None:
                continue
            entry: dict[str, Any] = {
                "prompt_per_1m": prompt,
                "completion_per_1m": completion,
                "verified": True,
                "source": "opencode-docs",
            }
            cached_read = _money(row[3])
            if cached_read is not None:
                entry["cached_read_per_1m"] = cached_read
            cached_write = _money(row[4]) if len(row) > 4 else None
            if cached_write is not None:
                entry["cached_write_per_1m"] = cached_write
            out[model_id] = entry
        except (ValueError, IndexError):
            continue
    return out


def _fetch_opencode_docs(url: str = _DOCS_URL) -> str | None:
    """Allowlisted single-URL GET of the docs page (decision #006).

    Only ``opencode.ai`` is permitted; any other host returns ``None`` so we
    never perform arbitrary URL fetching.  Returns the raw HTML or ``None``.
    """
    import urllib.request
    from urllib.parse import urlparse

    try:
        host = urlparse(url).netloc.lower().lstrip("www.")
        if host != _DOCS_HOST:
            return None
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def fetch_cloud_prices(
    fetch: bool = False,
    force: bool = False,
    ttl_seconds: int = _PRICE_TTL_SECONDS,
    _http_get: Any = None,
) -> dict[str, dict[str, Any]]:
    """Refresh cloud prices from the web (opencode.ai docs, then DDG fallback).

    Cadence: each model is re-fetched at most once per ``ttl_seconds`` (default
    24h) unless ``force=True`` (on-demand refresh).  The network is NEVER
    touched when ``fetch`` is ``False`` (the default), so tests/CI stay offline.

    Sources, in order:
      1. ``_fetch_opencode_docs`` (allowlisted host) -> ``parse_opencode_docs_prices``
         -> ``verified: True, source: "opencode-docs"``.
      2. DDG ``search_ddg`` snippet extraction for any cloud model not covered
         by the docs table -> ``verified: False`` (UNTRUSTED, decision #004/#005).

    Returns the merged table (baseline + cache + any fresh web values).
    ``_http_get`` is an injection point for tests (replaces ``_fetch_opencode_docs``).
    """
    table = load_prices()
    if not fetch:
        return table

    now = _time.time()
    fetcher = _http_get if callable(_http_get) else _fetch_opencode_docs

    # --- Primary: opencode.ai docs price table ---------------------------
    html = fetcher(_DOCS_URL)
    if html:
        for model_id, entry in parse_opencode_docs_prices(html).items():
            entry = dict(entry)
            entry["_fetched_at"] = now
            table[model_id] = entry

    # --- Fallback: DDG snippets for models still missing -----------------
    from agent_core.tools.web_search import search_ddg

    for model_id in discover_cloud_models():
        if _is_free_tier(model_id):
            continue
        existing = table.get(model_id, {})
        if existing.get("verified") and existing.get("source") == "opencode-docs":
            continue  # already authoritative from the docs parse
        fetched_at = float(existing.get("_fetched_at", 0) or 0)
        if not force and fetched_at and (now - fetched_at) < ttl_seconds:
            continue
        short = model_id.split("/", 1)[-1]
        query = f"opencode go {short} price per 1M tokens"
        try:
            results = search_ddg(query, max_results=3)
        except Exception:
            results = []
        for result in results:
            prices = _extract_prices_from_snippet(result.get("snippet", ""))
            if prices:
                table[model_id] = {
                    "prompt_per_1m": prices[0],
                    "completion_per_1m": prices[1],
                    "verified": False,
                    "source": "web-search-untrusted",
                    "_fetched_at": now,
                }
                break

    save_prices(table)
    return table
