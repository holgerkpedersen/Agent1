"""Regression tests: Histogram view renders a real binned distribution chart.

Ships three layers of guards:
1. API contract - ``GET /api/histograms`` exposes raw per-metric samples
   (``MetricsCollector.all_histogram_samples``) so the UI can bin them.
2. Frontend scaffolding - index.html must contain the chart container,
   metric picker, p50/p95 stat chips and the CSS classes the chart draws.
3. Chart math - ``percentile()`` extracted from index.html and executed
   under node against a known vector (skipped if node is unavailable).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import typing
import urllib.request

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(REPO, "static", "index.html")


def _seeded_collector():
    from agent_core.monitoring.metrics_collector import MetricsCollector

    c = MetricsCollector()
    for v in (0.11, 0.12, 0.13, 0.14, 0.15):
        c.record_histogram("command.elapsed.seconds", v)
    c.record_histogram("tool.elapsed.seconds", 2.5)
    return c


@pytest.fixture(scope="module")
def base_url() -> typing.Iterator[str]:
    from agent_core.monitoring import DashboardAPIServer

    holder = DashboardAPIServer(_seeded_collector(), port=0)
    httpd = holder.start()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        shutdown = threading.Thread(target=httpd.shutdown, daemon=True)
        shutdown.start()
        shutdown.join(timeout=5)
        httpd.server_close()


def test_api_exposes_raw_histogram_samples(base_url: str) -> None:
    with urllib.request.urlopen(base_url + "/api/histograms", timeout=10) as r:
        payload = json.loads(r.read())
    assert set(payload["samples"]) == {
        "command.elapsed.seconds", "tool.elapsed.seconds"
    }
    assert payload["samples"]["command.elapsed.seconds"] == [0.11, 0.12, 0.13, 0.14, 0.15]
    # legacy summary fields stay intact: without ?name= it summarizes nothing,
    # with an explicit name it returns that metric's stats.
    assert payload["histogram"]["count"] == 0.0
    q = urllib.request.urlopen(
        base_url + "/api/histograms?name=command.elapsed.seconds", timeout=10)
    with q:
        named = json.loads(q.read())
    assert named["histogram"]["count"] == 5.0
    assert abs(named["histogram"]["mean"] - 0.13) < 1e-9


def test_all_histogram_samples_returns_copies() -> None:
    c = _seeded_collector()
    snapshot = c.all_histogram_samples()
    snapshot["command.elapsed.seconds"].append(999.0)
    assert 999.0 not in c.all_histogram_samples()["command.elapsed.seconds"], \
        "caller mutated collector state -> API would leak mutable internals"


def test_index_has_chart_scaffolding_and_styles() -> None:
    html = open(INDEX, encoding="utf-8").read()
    for needle in (
        'id="hgram-chart"',            # bar container
        'id="hgram-select"',           # metric picker
        'id="hstat-p50"', 'id="hstat-p95"',  # tail-latency chips
        ".hgram-bar {",                # bar styling hook
        "renderChart(",                # renderer wired up
        "api/histograms",              # new endpoint consumed
    ):
        assert needle in html, f"histogram chart lost its scaffolding: {needle}"
    # bars are percentage-height columns inside a fixed-height flex row;
    # without align-items:stretch they collapse to 0px (the original bug).
    assert re.search(r"\.hgram-chart\s*\{[^}]*align-items:\s*stretch", html), \
        ".hgram-chart needs align-items:stretch or every bar collapses to 0 height"


def test_percentile_math_in_page_matches_reference() -> None:
    """Extract percentile() from the page and execute it under node."""
    if shutil.which("node") is None:
        pytest.skip("node not available")
    html = open(INDEX, encoding="utf-8").read()
    m = re.search(r"function percentile\(sortedAsc, q\) \{.*?\n    \}", html, re.S)
    assert m, "percentile() not found in index.html"
    js = m.group(0) + """
      const s = [0.11,0.12,0.13,0.14,0.15,0.16,0.18,0.22,0.31,1.4];
      console.log(JSON.stringify([percentile(s,0.5), percentile(s,0.95)]));
    """
    r = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    p50, p95 = json.loads(r.stdout.strip())
    # reference values computed independently (linear interpolation)
    assert abs(p50 - 0.155) < 1e-9
    assert abs(p95 - 0.9095) < 1e-9
    assert p95 > p50, "p95 must sit above p50 when a tail exists"
