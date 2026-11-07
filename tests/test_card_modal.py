"""Regression tests: /CSS pin for the card-integrated modal.

The style of ``tests/test_dashboard_assets.py`` is reused verbatim: a CSS
selector asserted against the shipped stylesheet, and a DOM element asserted
against the served template -- both are *pins*, so a rename in either file
fails loudly instead of silently un-styling the page.

1. The Kanban checkout's ``static/style.css`` must carry a
   ``.card-integrated`` rule (the feature is implemented there).  Drop the
   rule and the dialog renders as a detached box.
2. ``index.html`` must keep the badge element(s) the card header pins its
   counters to (``id="badge-..."``). The JS writes textContent into those ids
   by name, so renaming them in the template leaves the badges on "-".

Note: the ``.card-integrated`` rule lives in the Kanban checkout's stylesheet;
all other pins stay against Agent1's own assets.  CI checks out only Agent1, so
the Kanban pin skips itself when that checkout is not next to this repo -- the
same gate ``test_kanban_roundtrip.py`` / ``test_linked_cards.py`` use.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

#: Repo root derived from this file's location -- a hardcoded path breaks CI
#: the moment the checkout lands anywhere else.
REPO = Path(__file__).resolve().parent.parent


def _kanban_root() -> Path:
    """Locate the Kanban checkout (env override, then sibling directory)."""
    env = os.environ.get("KANBAN_REPO_ROOT")
    if env:
        return Path(env)
    return REPO.parent / "Kanban"


#: The CSS pin below is cross-repo.  CI checks out Agent1 alone, so when no
#: Kanban checkout sits next to it the pin skips (mirrors test_linked_cards).
KANBAN_ROOT = _kanban_root()
CSS = KANBAN_ROOT / "src" / "agent1" / "static" / "style.css"
TEMPLATE = REPO / "static" / "index.html"

if not CSS.is_file():
    pytest.skip(
        f"Kanban checkout not found at {KANBAN_ROOT} (cross-repo pin; set "
        "KANBAN_REPO_ROOT to point at it)",
        allow_module_level=True,
    )


def _read(path: Path) -> str:
    assert path.is_file(), f"missing asset: {path}"
    return path.read_text(encoding="utf-8", errors="replace")


def test_card_integrated_rule_exists_in_style_css() -> None:
    """``.card-integrated`` must be a real selector, not just a stray token."""
    css = _read(CSS)
    rule = re.search(r"(?m)^\s*([^{}]*\.card-integrated[^{}]*)\{", css)
    assert rule, (
        "Kanban style.css has no .card-integrated rule -> the modal card is not "
        "stitched into its container and renders detached"
    )
    selector = rule.group(1)
    assert ".card-integrated" in selector
    # A rule with an empty body is as good as absent.
    body = css[rule.end() : css.index("}", rule.end())]
    assert body.strip(), f"{selector.strip()} has an empty body -> no-op rule"


def test_badge_element_exists_in_template() -> None:
    """The card header badges are addressed by id from JS -- keep them present."""
    html = _read(TEMPLATE)
    badges = re.findall(
        r'<(\w+)\b[^>]*\bclass="[^"]*\bbadge\b[^"]*"[^>]*\bid="([^"]+)"', html
    )
    assert badges, (
        "index.html has no class='badge' element with an id -> the JS "
        "getElementById('badge-...') lookups resolve to null"
    )
    ids = [b[1] for b in badges]
    assert len(set(ids)) == len(ids), f"duplicate badge ids: {ids}"

    # Every badge id referenced from script must exist in the markup.
    referenced = set(re.findall(r"""getElementById\(\s*['"](badge-[^'"]+)['"]""", html))
    missing = sorted(referenced - set(ids))
    assert not missing, f"JS references badge ids absent from the template: {missing}"


def test_badges_live_inside_the_card_modal_markup() -> None:
    """Badges must sit inside a card/modal container, not float in the page."""
    html = _read(TEMPLATE)
    for tag, badge_id in re.findall(
        r'<(\w+)\b[^>]*\bclass="[^"]*\bbadge\b[^"]*"[^>]*\bid="([^"]+)"', html
    ):
        start = html.index(f'id="{badge_id}"')
        window = html[max(0, start - 1200) : start]
        assert re.search(r'class="[^"]*\b(card|modal)\b', window), (
            f"badge #{badge_id} is not nested under a card/modal container"
        )


@pytest.mark.parametrize("rel", ["static/ttheme/css/style.css", "static/index.html"])
def test_assets_are_served(rel: str) -> None:
    """Both pins point at files the dashboard actually serves."""
    path = REPO / rel
    assert path.is_file() and path.stat().st_size > 1_000, (
        f"missing/suspicious asset: {rel}"
    )
