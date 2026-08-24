"""Regression tests for dashboard view-panel markup integrity.

Bug being guarded against (fixed 2026-08-24): the closing ``</div>`` of the
third Dashboard-view row was missing, which pushed all five dedicated view
panels (commands/log/alerts/gauges/histogram) INSIDE that dashboard row.
Clicking a sidebar item then hid the panels' own ancestor, so e.g. the
Command Summary view rendered as a blank page (title changed, nothing else).
HTML parsers auto-close such tags, so plain tag-balance checks did NOT flag
it -- the nesting depth is the reliable signal.
"""
from __future__ import annotations

import pathlib
from html.parser import HTMLParser

INDEX = pathlib.Path(__file__).resolve().parent.parent / "static" / "index.html"

VOID_TAGS = {"meta", "link", "br", "hr", "img", "input", "source", "wbr"}

#: Every view name that must have its own top-level panel row.
EXPECTED_VIEWS = {"dashboard", "commands", "log", "alerts", "gauges", "histogram"}


class PanelNestingParser(HTMLParser):
    """Collect (line, ancestor-stack, panel-name) for each data-view-panel."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.panels: list[tuple[int, tuple[str, ...], str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in VOID_TAGS:
            return
        names = dict(attrs)
        if "data-view-panel" in names:
            self.panels.append(
                (self.getpos()[0], tuple(self.stack), names["data-view-panel"])
            )
        self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID_TAGS:
            return
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        elif tag in self.stack:
            # tolerate implicit closes like browsers do, but keep stack sane
            idx = len(self.stack) - 1 - self.stack[::-1].index(tag)
            del self.stack[idx:]

    @property
    def ancestors(self) -> set[tuple[str, ...]]:
        return {anc for _, anc, _ in self.panels}


def _parse_index() -> PanelNestingParser:
    parser = PanelNestingParser()
    parser.feed(INDEX.read_text(encoding="utf-8"))
    return parser


def test_every_view_has_a_panel() -> None:
    parser = _parse_index()
    views = {name for _, _, name in parser.panels}
    missing = EXPECTED_VIEWS - views
    assert not missing, f"data-view-panel missing for views: {sorted(missing)}"


def test_view_panels_are_siblings_not_nested() -> None:
    """All [data-view-panel] rows must share ONE identical ancestor chain.

    A panel nested inside another panel means openView() hides its own
    container -> zero-height blank view (the original bug).
    """
    parser = _parse_index()
    chains = parser.ancestors
    assert len(chains) == 1, (
        "data-view-panel elements are NOT siblings; ancestor chains differ: "
        f"{sorted(chains)}"
    )


def test_no_panel_is_descendant_of_another_panel() -> None:
    """Panels must sit directly under .dashboard-main-body (html>body>main>div).

    Anything deeper means a panel got swallowed by another panel's row --
    exactly how the missing </div> bug manifested.
    """
    parser = _parse_index()
    expected_chain = ("html", "body", "main", "div")
    for line, ancestors, name in parser.panels:
        assert ancestors == expected_chain, (
            f"panel '{name}' at line {line} has unexpected ancestors "
            f"{ancestors}; expected direct child of .dashboard-main-body "
            f"({expected_chain})"
        )

