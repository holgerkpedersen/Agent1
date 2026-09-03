"""Regression tests for cloud LLM price discovery (opencode.ai docs).

These tests are fully offline: the docs HTML is loaded from a committed
fixture (``tests/fixtures/opencode_go_docs.html``) and the live fetcher is
never invoked.  They lock in the authoritative price table transcribed from
the opencode.ai *Usage limits* section so a future docs change (or a bad
edit) is caught.

Decision #006 (no arbitrary URL fetching) is also asserted: the allowlisted
fetcher refuses any host other than ``opencode.ai``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent_core.llm.pricing as pricing

_FIXTURE = Path(__file__).parent / "fixtures" / "opencode_go_docs.html"


def _load_fixture() -> str:
    return _FIXTURE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Parsing the authoritative docs price table
# ---------------------------------------------------------------------------

def test_parse_docs_prices_fixture_count() -> None:
    """The committed fixture yields every keyed opencode-go model (27)."""
    parsed = pricing.parse_opencode_docs_prices(_load_fixture())
    assert len(parsed) == 27
    # Every parsed id must be an opencode-go routing id.
    assert all(k.startswith("opencode-go/") for k in parsed)


def test_parse_docs_prices_spot_values() -> None:
    """Spot-check a few rows against the opencode.ai docs table."""
    parsed = pricing.parse_opencode_docs_prices(_load_fixture())

    grok = parsed["opencode-go/grok-4.6"]
    assert grok["prompt_per_1m"] == 2.00
    assert grok["completion_per_1m"] == 6.00
    assert grok["cached_read_per_1m"] == 0.50
    assert grok["verified"] is True
    assert grok["source"] == "opencode-docs"

    luna = parsed["opencode-go/gpt-5.6-luna"]
    assert luna["prompt_per_1m"] == 0.20
    assert luna["completion_per_1m"] == 1.20
    assert luna["cached_read_per_1m"] == 0.02
    assert luna["cached_write_per_1m"] == 0.25

    mimo = parsed["opencode-go/mimo-v2.5"]
    assert mimo["prompt_per_1m"] == 0.14
    assert mimo["completion_per_1m"] == 0.28
    assert mimo["cached_read_per_1m"] == 0.0028

    hy3 = parsed["opencode-go/hy3"]
    assert hy3["prompt_per_1m"] == 0.14
    assert hy3["completion_per_1m"] == 0.58
    assert hy3["cached_read_per_1m"] == 0.035


def test_parse_docs_prices_keeps_lower_tier() -> None:
    """Tiered rows keep the common-case LOWER tier (<= threshold / off-peak)."""
    parsed = pricing.parse_opencode_docs_prices(_load_fixture())

    # Grok 4.6 >200K is $4/$12; the parsed value must be the <=200K $2/$6 row.
    assert parsed["opencode-go/grok-4.6"]["prompt_per_1m"] == 2.00
    # DeepSeek V4 Pro peak is $1.32/$3.96; parsed must be off-peak $0.66/$1.98.
    ds = parsed["opencode-go/deepseek-v4-pro"]
    assert ds["prompt_per_1m"] == 0.66
    assert ds["completion_per_1m"] == 1.98


def test_parse_docs_prices_malformed_html_safe() -> None:
    """Parser never raises -- garbage input yields an empty dict."""
    assert pricing.parse_opencode_docs_prices("") == {}
    assert pricing.parse_opencode_docs_prices("<tr><td>not a price</td></tr>") == {}
    assert pricing.parse_opencode_docs_prices(None) == {}  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SSRF guard (decision #006): only opencode.ai may be fetched
# ---------------------------------------------------------------------------

def test_fetch_fetcher_refuses_non_allowlisted_host(monkeypatch) -> None:
    """The allowlisted fetcher returns None for any non-opencode.ai host."""
    captured: dict[str, object] = {}

    def _fake_get(req, *a, **k):  # pragma: no cover - injected
        captured["req"] = req
        return None

    monkeypatch.setattr("urllib.request.urlopen", _fake_get)
    # A clearly disallowed host must be rejected before any network call.
    result = pricing._fetch_opencode_docs("https://evil.example.com/x")
    assert result is None
    assert "req" not in captured  # urlopen was never called


def test_fetch_fetcher_allows_opencode_host(monkeypatch) -> None:
    """opencode.ai is the single allowlisted host and is fetched."""
    class _Resp:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return b"<html>docs</html>"
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
    result = pricing._fetch_opencode_docs(pricing._DOCS_URL)
    assert result == "<html>docs</html>"


# ---------------------------------------------------------------------------
# fetch_cloud_prices: offline by default, web refresh on demand
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point the pricing cache at a temp file so tests never touch the repo."""
    cache = tmp_path / "pricing_cache.json"
    monkeypatch.setattr(pricing, "_PRICING_CACHE_PATH", cache)
    return cache


def test_fetch_offline_by_default(isolated_cache) -> None:
    """fetch=False (default) never touches the network and returns baseline."""
    table = pricing.fetch_cloud_prices(fetch=False)
    assert "opencode-go/grok-4.6" in table
    # No cache file should be written when offline.
    assert not isolated_cache.exists()


def test_fetch_refresh_stores_docs_prices(isolated_cache) -> None:
    """fetch=True with an injected docs fetcher writes the cache (verified)."""
    html = _load_fixture()

    def _fake_docs_get(url):
        return html

    table = pricing.fetch_cloud_prices(
        fetch=True, force=True, _http_get=_fake_docs_get
    )
    # Cache now reflects the docs parse.
    assert isolated_cache.exists()
    cached = json.loads(isolated_cache.read_text(encoding="utf-8"))
    assert cached["opencode-go/grok-4.6"]["source"] == "opencode-docs"
    assert cached["opencode-go/grok-4.6"]["verified"] is True
    # Returned table is the merged baseline + cache.
    assert table["opencode-go/grok-4.6"]["prompt_per_1m"] == 2.00


def test_docs_overrides_stale_baseline(isolated_cache, monkeypatch) -> None:
    """A web-refreshed docs price must win over a stale baseline on conflict."""
    monkeypatch.setattr(
        pricing,
        "BASELINE_PRICES",
        {
            "opencode-go/grok-4.6": {
                "prompt_per_1m": 99.0, "completion_per_1m": 99.0,
                "verified": False, "source": "stale-baseline",
            }
        },
    )
    html = _load_fixture()

    def _fake_docs_get(url):
        return html

    table = pricing.fetch_cloud_prices(
        fetch=True, force=True, _http_get=_fake_docs_get
    )
    # Docs value (2.00) overrides the stale 99.0 baseline.
    assert table["opencode-go/grok-4.6"]["prompt_per_1m"] == 2.00
    assert table["opencode-go/grok-4.6"]["source"] == "opencode-docs"


def test_load_prices_merges_cache_over_baseline(isolated_cache) -> None:
    """load_prices merges the cache on top of BASELINE_PRICES."""
    cache = {"opencode-go/hy3": {"prompt_per_1m": 0.5, "completion_per_1m": 0.5,
                                 "verified": True, "source": "opencode-docs"}}
    isolated_cache.write_text(json.dumps(cache), encoding="utf-8")
    table = pricing.load_prices()
    assert table["opencode-go/hy3"]["prompt_per_1m"] == 0.5
    # Baseline entries still present.
    assert "opencode-go/grok-4.6" in table


# ---------------------------------------------------------------------------
# Free-tier / cost semantics
# ---------------------------------------------------------------------------

def test_is_free_tier_matches() -> None:
    """Free-tier detection covers -free, :free and zen prefixes."""
    assert pricing._is_free_tier("opencode-zen/hy3-free")
    assert pricing._is_free_tier("opencode-go/deepseek-v4-flash:free")
    assert pricing._is_free_tier("opencode-zen/anything")
    assert not pricing._is_free_tier("opencode-go/grok-4.6")


def test_cost_per_token_free_is_zero() -> None:
    """Free/zen/local tiers always cost 0.0 (cheapest-failover correctness)."""
    assert pricing.cost_per_token("opencode-zen/hy3-free", "opencode") == 0.0
    assert pricing.cost_per_token("anything", "lmstudio") == 0.0
    assert pricing.cost_per_token("anything", "llama") == 0.0


def test_estimate_cost_paid_uses_docs_price() -> None:
    """A paid model's cost uses the authoritative docs price."""
    # grok-4.6: $2 in / $6 out per 1M.  1M in + 1M out = 8.0 USD.
    cost = pricing.estimate_cost(1_000_000, 1_000_000, "opencode-go/grok-4.6", "opencode")
    assert cost == pytest.approx(8.0)
