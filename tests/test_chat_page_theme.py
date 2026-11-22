"""Regression tests for the /chat page's shell parity with the dashboard.

``static/chat.html`` is served VERBATIM (``dashboard_api._send_chat_page``), so
it never receives the ``partials/_head.html`` / ``_sidebar.html`` includes that
``index.html`` is assembled from.  That isolation is why the page had drifted:

* it read ``localStorage["theme"]`` on load, but shipped no ``[data-theme-
  toggle]`` button - a user could not change the theme from this page at all,
  while ``index.html`` and ``autonomous.html`` both offer it;
* nothing checked that its inline script even parsed, so a syntax error in the
  page would only ever show up in a browser console.

These tests pin the shared contract rather than the implementation: the page
may own its markup, but it must speak the same language as the dashboard.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
PAGE = STATIC / "chat.html"
APP_JS = STATIC / "ttheme" / "js" / "app.js"
STYLE_CSS = STATIC / "ttheme" / "css" / "style.css"

NODE = shutil.which("node")
requires_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

#: The inline ``<script>`` blocks that carry the page's own behaviour. The
#: ``src=`` tags are skipped: they are libraries, not this page's code.
_SCRIPT = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.S)


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _inline_scripts() -> str:
    """Every inline script block, concatenated, in document order."""
    return "\n".join(_SCRIPT.findall(_page()))


def test_chat_page_ships_a_theme_toggle() -> None:
    """The page must let the user change the theme, like its sibling pages.

    Regression: the page depended on a stored theme but had no control to set
    one, so a light-mode user was stuck on whatever another page had written.
    """
    html = _page()
    assert "data-theme-toggle" in html, (
        "chat.html must carry the [data-theme-toggle] button that style.css "
        "draws the theme glyph for"
    )
    assert 'id="themeToggle"' in html
    # The handler has to be wired to the button, not merely declared.
    assert re.search(r'\$\("themeToggle"\)\.addEventListener\("click"', html)


def test_chat_page_uses_the_same_theme_storage_key_as_the_dashboard() -> None:
    """One key, two surfaces: a theme chosen here must apply on the dashboard.

    ``static/ttheme/js/app.js`` writes ``localStorage["theme"]``; the page's
    inline head script reads the same key before first paint.
    """
    html = _page()
    assert re.search(r'(?:const|let|var)\s+THEME_KEY\s*=\s*"theme"', html), (
        "the page must name the shared storage key"
    )
    assert 'localStorage.setItem(THEME_KEY' in html
    assert 'localStorage.getItem("theme")' in html.replace("'", '"')
    # And the dashboard's own writer must agree on that key.
    assert 'localStorage.setItem("theme"' in APP_JS.read_text(encoding="utf-8")


def test_chat_page_theme_aria_label_matches_the_css_glyph_contract() -> None:
    """style.css selects the glyph on ``[data-theme-toggle][aria-label=dark]``.

    So the attribute values are a CSS contract: "dark"/"light" only, and the
    label must describe the CURRENT theme (the glyph shown is the alternative).
    """
    css = STYLE_CSS.read_text(encoding="utf-8")
    assert "[data-theme-toggle][aria-label=dark]::after" in css, (
        "style.css changed its glyph contract - update chat.html with it"
    )
    html = _page()
    assert re.search(r'aria-label="dark"', html), "initial label must match data-theme"
    # Both labels must be produced by the apply function, never free text.
    assert 'setAttribute("aria-label"' in html
    assert '"dark" : "light"' in html or '"light" : "dark"' in html


def test_chat_page_never_writes_a_glyph_into_the_toggle_button() -> None:
    """The glyph is CSS ::after - text inside the button would offset it.

    ``app.js`` documents the same rule for the navbar button.
    """
    html = _page()
    assert "deliberately does NOT set innerText" in APP_JS.read_text(encoding="utf-8")
    # The toggle button is empty in markup.
    match = re.search(r'<button[^>]*id="themeToggle".*?</button>', html, re.S)
    assert match is not None, "theme toggle button not found"
    inner = match.group(0).split(">", 1)[1].rsplit("</button>", 1)[0]
    assert inner.strip() == "", "the toggle glyph comes from CSS, not markup"


@requires_node
def test_chat_page_inline_script_parses(tmp_path: Path) -> None:
    """A syntax error in the page must fail here, not in a browser console.

    The page has no bundler and no build step, so nothing else parses it. This
    is the only automated guard that the JS the browser will run is valid.
    """
    scripts = _inline_scripts()
    assert scripts.strip(), "chat.html has no inline script"
    target = tmp_path / "chat_page.js"
    # node --check parses without executing, so DOM globals are irrelevant.
    target.write_text(scripts, encoding="utf-8")
    result = subprocess.run(
        [NODE, "--check", str(target)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"inline script does not parse:\n{result.stderr}"


@requires_node
def test_chat_page_theme_functions_behave_as_documented(tmp_path: Path) -> None:
    """Exercise the real applyTheme/toggleTheme source under node.

    The functions are extracted from the page (not copied) so this test cannot
    drift from what ships.
    """
    html = _page()
    needed = ("currentTheme", "applyTheme", "toggleTheme")
    chunks = []
    for name in needed:
        start = html.index("function " + name + "(")
        brace = html.index("{", start)
        depth = 0
        for index in range(brace, len(html)):
            char = html[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    chunks.append(html[start : index + 1])
                    break
        else:  # pragma: no cover - guards the extraction contract
            raise AssertionError(f"unbalanced braces in {name}()")

    driver = """
const THEME_KEY = "theme";
const store = {};
const localStorage = {
  setItem: (k, v) => { store[k] = v; },
};
let attr = { "data-theme": "dark" };
let label = null;
const document = {
  documentElement: {
    getAttribute: (n) => attr[n] || null,
    setAttribute: (n, v) => { attr[n] = v; },
  },
};
const $ = (id) => ({
  setAttribute: (n, v) => {
    if (n === "aria-label" && id === "themeToggle") label = v;
  },
  getAttribute: (n) => attr[n],
});
\nthe_functions\n
const out = {};
out.initial = currentTheme();
out.labelled = applyTheme("dark") + ":" + label;
applyTheme("light");
out.light = attr["data-theme"] + ":" + label;
attr["data-theme"] = "light";
toggleTheme();
out.afterToggle = attr["data-theme"] + ":" + store[THEME_KEY] + ":" + label;
\ntoggleTheme();
out.backAgain = attr["data-theme"] + ":" + store[THEME_KEY];
out.garbage = applyTheme("purple") + ":" + attr["data-theme"];
process.stdout.write(JSON.stringify(out));
""".replace("the_functions", "\n".join(chunks))

    script = tmp_path / "theme_driver.js"
    script.write_text(driver, encoding="utf-8")
    result = subprocess.run(
        [NODE, str(script)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    import json

    out = json.loads(result.stdout)
    # Dark is the shipped default, and the label names the CURRENT theme.
    assert out["initial"] == "dark"
    assert out["labelled"] == "dark:dark"
    assert out["light"] == "light:light"
    # Toggling persists to the shared key and flips the attribute.
    assert out["afterToggle"] == "dark:dark:dark"
    assert out["backAgain"] == "light:light"
    # An unknown stored value must not produce a third theme.
    assert out["garbage"] == "dark:dark"
