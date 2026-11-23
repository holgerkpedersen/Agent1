"""Regression tests for the keyboard-shortcuts overlay on the /chat page.

A shortcuts edit broke the page in two ways at once:

- the ``.shortcuts`` rule set an explicit ``display`` value that outranks the
  ``[hidden]`` attribute's user-agent rule, so the overlay painted over the
  page on EVERY load (a hard refresh, Ctrl-F5, was enough to reproduce it);
- the main script's opening ``<script>`` tag was dropped, so the browser
  rendered the whole JS source as visible text under the Send button and none
  of the handlers ever ran - including the overlay's own Close button.

These tests pin both invariants at the source level, so either regression
fails CI without a browser.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "static"


def _page() -> str:
    return (STATIC / "chat.html").read_text(encoding="utf-8")


def _css() -> str:
    """Every inline <style> block, concatenated (the page has more than one)."""
    return "\n".join(re.findall(r"<style>(.*?)</style>", _page(), flags=re.S))


def test_shortcuts_overlay_is_hidden_until_opened() -> None:
    """A native <dialog> must not paint before showModal() opens it.

    An explicit display value (e.g. ``display: flex``) beats the ``[hidden]``
    attribute, so the stylesheet needs a gate keyed on the dialog's own
    ``open`` state - not a ``[hidden]`` rule, which the flex rule overrides.
    """
    m = re.search(r"\.shortcuts:not\(\[open\]\)\s*\{([^}]*)\}", _css())
    assert m, "the .shortcuts overlay needs a :not([open]) display gate"
    assert re.search(r"display\s*:\s*none", m.group(1)), (
        "the :not([open]) gate must set display: none"
    )


def test_script_blocks_are_balanced() -> None:
    """Every <script> must be closed and every </script> must have an open.

    Regression: the shortcuts edit dropped the main script's opening tag,
    leaving an unpaired ``</script>`` and rendering the JS source as page
    text.
    """
    html = _page()
    opens = len(re.findall(r"<script\b", html))
    closes = len(re.findall(r"</script>", html))
    assert opens == closes, (
        f"unbalanced script tags: {opens} opens vs {closes} closes"
    )
    assert opens >= 2, "expect at least the theme inline script and the main script"


def test_main_script_sits_inside_an_open_script_tag() -> None:
    """The main JS must sit INSIDE a <script> element, not in body text.

    If the opening tag is missing (or an earlier close wins), the last
    ``<script`` before the code must come after the last ``</script>``.
    """
    html = _page()
    strict = html.index('"use strict"')
    last_open = html.rfind("<script", 0, strict)
    last_close = html.rfind("</script>", 0, strict)
    assert last_open != -1 and last_open > last_close, (
        "the main script's <script> tag is missing or already closed - "
        "the JS would render as visible text"
    )


def test_shortcuts_dialog_element_is_closed() -> None:
    """The shortcuts <dialog> must exist and be balanced (comments excluded)."""
    code = re.sub(r"<!--.*?-->", "", _page(), flags=re.S)
    opens = len(re.findall(r"<dialog\b", code))
    closes = len(re.findall(r"</dialog>", code))
    assert opens == closes, "unbalanced <dialog> elements"
    assert opens >= 1, "the shortcuts dialog element is missing"
    assert 'id="shortcutsDialog"' in code
