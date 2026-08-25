"""Regression tests: Histogram view renders a real binned distribution chart.

Ships three layers of guards:
1. API contract - ``GET /api/histograms`` exposes raw per-metric samples
   (``MetricsCollector.all_histogram_samples``) so the UI can bin them.
2. Frontend scaffolding - index.html must contain the chart container,
   metric picker, p50/p95 stat chips and the CSS classes the chart draws.
3. Chart math - ``percentile()`` extracted from index.html and executed
   under node against a known vector (skipped if node is unavailable).
4. Axis + view modes - index.html must expose X/Y value indicators and the
   Bars / Cumulative % / Bin table toggles so the histogram is readable.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import typing
import importlib
import importlib.util
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
    # The fixed 240px height lives on .hgram-chart-inner (rendered by JS),
    # NOT on the outer #hgram-chart wrapper - otherwise the X-axis tick row
    # overflows onto the stat chips below. Both must hold.
    assert re.search(r"\.hgram-chart-inner\s*\{[^}]*align-items:\s*stretch", html), \
        ".hgram-chart-inner needs align-items:stretch or every bar collapses to 0 height"
    assert re.search(r"\.hgram-chart-inner\s*\{[^}]*height:\s*240px", html), \
        ".hgram-chart-inner must keep the fixed 240px bar height"
    assert re.search(r"\.hgram-chart\s*\{[^}]*display:\s*block", html), \
        "outer #hgram-chart must be a plain block wrapper so the X-axis cannot overflow onto the stat chips"


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


def test_histogram_view_has_axis_indicators_and_view_modes() -> None:
    """The histogram must show real value indicators on both axes and offer
    more than one way to read the same data."""
    html = open(INDEX, encoding="utf-8").read()
    # X/Y axis scaffolding + the three view-mode toggles all present.
    for needle in (
        'class="hgram-yaxis"',            # Y-axis value ladder (count)
        'class="hgram-yaxis-title"',      # rotated Y-axis label
        'class="hgram-tick"',             # X-axis tick row
        'hgram-view-bars',                # Bars mode
        'hgram-view-cum',                 # Cumulative % mode
        'hgram-view-table',               # Bin table mode
        'bindHgramViewToggle(',           # toggle wired to re-render
        'window.__hgramSamples',          # samples cached for the toggle
    ):
        assert needle in html, f"histogram axis/view-mode support missing: {needle}"
    # Regression guard: .hgram-plot must grow to fill the card width, otherwise
    # the flexed bars + X-axis ticks collapse into a ~70px strip and the tick
    # labels overlap each other.
    assert re.search(r"\.hgram-plot\s*\{[^}]*flex:\s*1\s*1\s*0", html, re.S), \
        "histogram plot must be flex:1 1 0 so the X-axis does not squash/overlap"


def test_render_chart_emits_axis_labels_and_all_three_views() -> None:
    """Drive renderChart under node for the three view modes and assert the
    X-axis (time, seconds) and Y-axis (count) indicators are present."""
    if shutil.which("node") is None:
        pytest.skip("node not available")
    html = open(INDEX, encoding="utf-8").read()
    # renderChart + bindHgramViewToggle live in the page; pull both out.
    m = re.search(r"function renderChart\(vals\) \{.*?\n    \}", html, re.S)
    assert m, "renderChart() not found in index.html"
    render = m.group(0)
    b = re.search(r"function bindHgramViewToggle\(\) \{.*?\n    \}", html, re.S)
    assert b, "bindHgramViewToggle() not found in index.html"
    p = re.search(r"function percentile\(sortedAsc, q\) \{.*?\n    \}", html, re.S)
    assert p, "percentile() not found in index.html"
    js = render + "\n" + p.group(0) + "\n" + b.group(0) + r"""
      const vals = [0.12,0.13,0.14,0.15,0.18,0.22,0.31,0.40,1.20,2.50];
      const out = {};
      // Stable fake elements so renderChart writes into the same objects we read.
      const els = {
        'hgram-chart': { innerHTML: '', style: {}, textContent: '',
                         addEventListener: function(){}, querySelector: function(){return null;} },
        'hgram-caption': { innerHTML: '', style: {}, textContent: '',
                           addEventListener: function(){}, querySelector: function(){return null;} },
      };
      function fakeEl() {
        return { innerHTML: '', style: {}, textContent: '',
                 addEventListener: function(){}, querySelector: function(){return null;} };
      }
      function setView(v) {
        global.document.querySelector = function(sel){
          if (sel.indexOf('hgram-view') !== -1) return { value: v };
          return null;
        };
      }
      global.document = {
        getElementById: function(id){ return els[id] || fakeEl(); },
        querySelectorAll: function(){ return []; }
      };
      global.window = {};

      setView('bars');
      renderChart(vals);
      out.bars = els['hgram-chart'].innerHTML;
      out.caption = els['hgram-caption'].textContent;

      setView('cumulative');
      renderChart(vals);
      out.cum = els['hgram-chart'].innerHTML;
      out.cumCaption = els['hgram-caption'].textContent;

      setView('table');
      renderChart(vals);
      out.table = els['hgram-chart'].innerHTML;

      console.log(JSON.stringify(out));
    """
    r = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip())
    # Bars view: Y-axis ladder (count) + X-axis ticks in seconds.
    assert "hgram-yaxis" in out["bars"], "bars view missing Y-axis value ladder"
    assert "hgram-tick" in out["bars"], "bars view missing X-axis tick row"
    assert out["caption"].startswith("Distribution of"), out["caption"]
    # Cumulative view: 100% / 50% / 0% ladder + 'done' tooltips.
    assert "100%" in out["cum"] and "0%" in out["cum"], "cumulative Y-axis not 0-100%"
    assert "done" in out["cum"], "cumulative bars missing %-done tooltip"
    assert "Cumulative" in out["cumCaption"], out["cumCaption"]
    # Table view: a real <table> with a header row. Because the table carries
    # its own column headers (# / time bin / count / share), the chart's Y
    # value ladder, rotated "commands / bin" label, X-axis tick row and
    # "time (s)" axis label must NOT be drawn around it.
    assert "<table" in out["table"], "table view missing <table>"
    assert "share" in out["table"], "table view missing share column"
    for forbidden in ("hgram-yaxis", "hgram-yaxis-title", "hgram-tick", "hgram-axis-label"):
        assert forbidden not in out["table"], \
            f"table view must not draw chart axis scaffolding ({forbidden})"


@pytest.mark.skipif(
    shutil.which("node") is None or importlib.util.find_spec("playwright") is None,
    reason="needs node + playwright (python) to drive the real page",
)
def test_axis_indicators_only_with_data_and_never_overlap_stats(base_url: str) -> None:
    """Regression for two histogram layout bugs:

    * The X-axis tick row must sit *above* the SAMPLES/MEAN/P50/P95/MIN/MAX
      stat chips and never overlap them. The outer ``#hgram-chart`` wrapper
      must NOT impose a fixed height on the plot (which forced the tick row
      to overflow onto the chips).
    * Axis indicators (Y ladder + X ticks) must only be drawn when there is
      an actual histogram; the empty state shows only the placeholder text.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1200})
        page.goto(base_url + "/")
        page.wait_for_timeout(300)

        def render(vals):
            page.evaluate(
                """(v) => {
                    const panel = document.querySelector('[data-view-panel="histogram"]');
                    if (panel) panel.hidden = false;
                    const sel = document.getElementById('hgram-select');
                    sel.innerHTML = '<option value="command.elapsed.seconds">command.elapsed.seconds (10)</option>';
                    sel.value = 'command.elapsed.seconds';
                    window.__hgramSamples = {'command.elapsed.seconds': v};
                    document.getElementById('hgram-view-bars').checked = true;
                    renderChart(v);
                }""",
                vals,
            )

        # --- empty state: no axis indicators, just the placeholder ----------
        render([])
        assert page.query_selector(".hgram-tick") is None, \
            "axis ticks must not render when there is no histogram"
        assert page.query_selector(".hgram-yaxis") is None, \
            "Y-axis must not render when there is no histogram"
        assert "No histogram data" in (page.query_selector(".hgram-chart").inner_text() or "")

        # --- with data: ticks exist and stay clear of the stat chips --------
        render([0.12, 0.13, 0.14, 0.15, 0.18, 0.22, 0.31, 0.40, 1.20, 2.50])
        tick = page.query_selector(".hgram-tick").bounding_box()
        stats = page.query_selector(".hgram-stats").bounding_box()
        assert tick is not None and stats is not None
        assert tick["y"] + tick["height"] <= stats["y"] + 1, \
            "X-axis tick row overlaps the SAMPLES/MEAN/P50/P95/MIN/MAX chips"

        # chips themselves must not overlap each other (horizontal flex row)
        boxes = [d.bounding_box() for d in page.query_selector_all(".hgram-stats > div")]
        assert len(boxes) == 6, f"expected 6 stat chips, got {len(boxes)}"
        for a, b in zip(boxes, boxes[1:]):
            assert a["y"] + a["height"] <= b["y"] + 1 or a["x"] + a["width"] <= b["x"] + 1, \
                "stat chips overlap each other"

        # --- table view: real table, but NO chart axis scaffolding ----------
        # The table already carries its own column headers (# / time bin /
        # count / share), so the Y value ladder, rotated "commands / bin"
        # label, X-axis tick row and "time (s)" axis label must not appear.
        page.evaluate(
            """() => {
                document.getElementById('hgram-view-table').checked = true;
                const sel = document.getElementById('hgram-select');
                renderChart((window.__hgramSamples[sel.value] || []).map(Number));
            }"""
        )
        chart = page.query_selector(".hgram-chart")
        assert chart.query_selector("table") is not None, "table view missing <table>"
        for forbidden in (".hgram-yaxis", ".hgram-yaxis-title", ".hgram-tick", ".hgram-axis-label"):
            assert chart.query_selector(forbidden) is None, \
                f"table view must not draw chart axis scaffolding ({forbidden})"
        browser.close()
