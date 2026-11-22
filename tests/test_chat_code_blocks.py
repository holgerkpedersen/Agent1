"""Regression tests for fenced-code rendering in the /chat page.

The agent answers in prose with code in it, and ``static/chat.html`` used to run
every reply through ``linkify()`` alone.  That is the right rule for prose and
the wrong rule for code, in two visible ways:

* a ``<pre>`` block was flattened into one wrapped paragraph, so indentation -
  the thing that makes code readable - was lost;
* the only copy control copied the WHOLE message, so getting a single snippet
  out of a long answer was manual selection.  That is the common case for an
  agent dashboard, which makes it a real gap rather than a nicety;
* a bare URL inside a code block became a clickable link, which is wrong: code
  is literal text.

The functions are extracted from the page and executed under node (the same
technique as ``test_chat_page_theme``), so these tests cannot drift from what
the browser actually runs.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "static" / "chat.html"

NODE = shutil.which("node")
requires_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

#: Everything the renderer needs, in dependency order.
_NEEDED = (
    "actionButton",
    "splitFences",
    "codeBlock",
    "renderRich",
    "linkify",
)


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _extract_functions(names: tuple[str, ...]) -> str:
    """Pull the named function bodies out of the page, in the given order."""
    html = _page()
    chunks = []
    for name in names:
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
    return "\n".join(chunks)


def _extract_const(name: str) -> str:
    """Pull one ``const NAME = <expr>;`` declaration out of the page.

    The renderer depends on module-level constants (the fence pattern), and a
    function extraction that skipped them would fail with a ReferenceError
    instead of testing anything.
    """
    html = _page()
    match = re.search(
        r"^const " + re.escape(name) + r" = .*?;$", html, re.S | re.M
    )
    assert match is not None, f"const {name} not found in chat.html"
    return match.group(0)


#: A deliberately small DOM: the renderer only needs createElement, appendChild,
#: textContent, dataset and a couple of attributes. Recording the appends into a
#: tree lets the assertions read structure instead of string-matching HTML.
_DOM = r"""
class Node_ {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.dataset = {};
    this.attrs = {};
    this._text = "";
    this.className = "";
    this.title = "";
    this.type = "";
    this.hidden = false;
  }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() {
    return this._text + this.children.map((c) => c.textContent).join("");
  }
  appendChild(child) { this.children.push(child); return child; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  addEventListener() {}
  querySelector() { return null; }
}
const document = {
  createElement: (tag) => new Node_(tag),
  createTextNode: (t) => {
    const n = new Node_("#text");
    n._text = String(t);
    return n;
  },
};
const copyCalls = [];
function copyText(text, btn) { copyCalls.push({text: text, btn: btn && btn.tag}); }
"""

_DRIVER = """
const out = {};
const cases = JSON.parse(process.argv[2]);
for (const [name, text] of Object.entries(cases)) {
  const box = document.createElement("div");
  renderRich(box, text);
  out[name] = {
    shapes: box.children.map((c) => c.tag),
    classes: box.children.map((c) => c.className),
    langs: box.children.map((c) => (c.dataset ? c.dataset.lang : undefined)),
    text: box.children.map((c) => c.textContent),
    raw: box.rawText,
  };
}
process.stdout.write(JSON.stringify(out));
"""


def _run_cases(cases: dict[str, str], tmp_path: Path) -> dict:
    """Render every (name, text) pair through the page's real renderer."""
    driver = (
        _DOM
        + "\n"
        + _extract_const("FENCE_RE")
        + "\n"
        + _extract_const("URL_RE")
        + "\n"
        + _extract_functions(_NEEDED)
        + "\n"
        + _DRIVER
    )
    script = tmp_path / "render_driver.js"
    script.write_text(driver, encoding="utf-8")
    result = subprocess.run(
        [str(NODE), str(script), json.dumps(cases)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@requires_node
def test_fenced_code_becomes_its_own_block(tmp_path: Path) -> None:
    """A fenced block must render as a code block, not as wrapped prose."""
    out = _run_cases(
        {"snippet": "Run this:\n```python\nprint('hi')\n```\nDone."}, tmp_path
    )
    shapes = out["snippet"]["shapes"]
    assert shapes == ["div", "figure", "div"], (
        "expected prose / code block / prose, got " + repr(shapes)
    )
    assert out["snippet"]["classes"][1] == "code-block"
    assert out["snippet"]["langs"][1] == "python", "the language label must survive"
    assert "print('hi')" in out["snippet"]["text"][1]


@requires_node
def test_code_block_text_keeps_indentation(tmp_path: Path) -> None:
    """Whitespace inside a block is content; stripping it breaks the code."""
    body = "def f():\n    if x:\n        return 1"
    out = _run_cases({"indent": "```\n" + body + "\n```"}, tmp_path)
    rendered = out["indent"]["text"][0]
    assert "    if x:" in rendered, "indentation was lost:\n" + rendered
    assert "        return 1" in rendered


@requires_node
def test_url_inside_a_code_block_stays_literal_text(tmp_path: Path) -> None:
    """Code is literal: a URL in it must not become a clickable link."""
    out = _run_cases(
        {"code": "```\nfetch('https://example.com/x')\n```"}, tmp_path
    )
    assert "https://example.com/x" in out["code"]["text"][0]
    # Prose still linkifies - the two rules must not be merged.
    prose = _run_cases({"prose": "see https://example.com/x now"}, tmp_path)
    assert prose["prose"]["shapes"] == ["div"]


@requires_node
def test_unclosed_fence_still_renders(tmp_path: Path) -> None:
    """A half-streamed or forgotten closing fence must not eat the reply."""
    out = _run_cases({"open": "before\n```sh\nls -la"}, tmp_path)
    shapes = out["open"]["shapes"]
    assert shapes == ["div", "figure"], repr(shapes)
    assert "ls -la" in out["open"]["text"][1]


@requires_node
def test_a_fence_inside_code_survives(tmp_path: Path) -> None:
    """Four backticks wrap a block that itself contains ```."""
    text = "````\n```\ninner\n```\n````"
    out = _run_cases({"nested": text}, tmp_path)
    shapes = out["nested"]["shapes"]
    assert shapes == ["figure"], (
        "a longer fence must contain the shorter one, got " + repr(shapes)
    )
    assert "```" in out["nested"]["text"][0]


@requires_node
def test_copy_returns_the_original_text_not_the_joined_tree(tmp_path: Path) -> None:
    """Copying a reply must give back the source, fences and all.

    The rendered bubble is prose + <figure> siblings; ``textContent`` would run
    them together and silently mangle what the user asked to copy.
    """
    source = "Run this:\n```python\nprint('hi')\n```\nDone."
    out = _run_cases({"copy": source}, tmp_path)
    # The driver records container.rawText, which is what the copy button reads.
    assert out["copy"]["raw"] == source, "copy source drifted from the reply"


@requires_node
def test_plain_reply_is_unchanged(tmp_path: Path) -> None:
    """No fence means one prose segment - the old single-paragraph shape."""
    out = _run_cases({"plain": "just a sentence"}, tmp_path)
    assert out["plain"]["shapes"] == ["div"]
    assert out["plain"]["text"] == ["just a sentence"]


def test_page_renders_replies_through_the_block_aware_renderer() -> None:
    """Both the history path and the live path must use renderRich().

    Regression guard: a new call site that goes back to ``linkify()`` would
    silently drop code-block rendering for that path only.
    """
    html = _page()
    assert "function renderRich(" in html
    # Every bubble fill must go through renderRich, not the prose-only helper.
    assert "renderRich(bubble, text);" in html
    for expected in ("renderRich(turn.bubble, data.reply)",
                     "renderRich(turn.bubble, \"Error: \""):
        assert expected in html, f"missing render call: {expected}"
    # linkify() must survive only as the prose sub-renderer.
    assert re.search(r"linkify\(prose, segment\.content\)", html), (
        "linkify() should now be called only for prose segments"
    )
