"""Behaviour tests for the /chat slash menu, run against the page's REAL JS.

The other chat tests assert that strings are present in ``static/chat.html``.
That cannot catch the bug these tests were written for: the menu existed, the
catalog was complete, and 27 of the 39 REPL commands were still invisible
because ``MAX_ROWS = 12`` truncated the alphabetical list before ``model``
(index 19) was reached.

So the functions are EXTRACTED from the page and executed by node here. The
command list is the live registry, so the fixtures cannot drift from the REPL
either. Node is optional tooling: the tests skip when it is absent.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest

STATIC = Path(__file__).resolve().parent.parent / "static"
PAGE = STATIC / "chat.html"

NODE = shutil.which("node")
requires_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

#: Functions copied out of the page. They are dependency-free by design, which
#: is what makes them extractable - keep it that way.
_NEEDED = (
    "fuzzyMatch",
    "hitPositions",
    "filterCommands",
    "stripUsageMeta",
    "commandFill",
)


def _read_page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _extract_function(source: str, name: str) -> str:
    """Return the source of top-level ``function <name>(...) {...}``.

    Brace counting is safe for these four functions: none of them contains a
    brace inside a string or a comment.
    """
    try:
        start = source.index("function " + name + "(")
    except ValueError:  # pragma: no cover - guards the extraction contract
        raise AssertionError(f"{name}() is gone from chat.html")
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unbalanced braces in {name}()")  # pragma: no cover


def _max_rows(source: str) -> int:
    match = re.search(r"const MAX_ROWS\s*=\s*(\d+)\s*;", source)
    assert match is not None, "MAX_ROWS is no longer a plain integer constant"
    return int(match.group(1))


def _live_commands() -> List[Dict[str, Any]]:
    """The catalog the dashboard actually serves, from the live registry."""
    from agent_core.cli.interactive import _build_registry
    from agent_core.commands.catalog import command_catalog

    return command_catalog(_build_registry())


@pytest.fixture(scope="module")
def menu(tmp_path_factory) -> Dict[str, Any]:
    """Run the page's menu functions under node and return their results."""
    if NODE is None:  # pragma: no cover - requires_node guards the callers
        pytest.skip("node is not installed")
    source = _read_page()
    functions = "\n\n".join(_extract_function(source, name) for name in _NEEDED)
    commands = _live_commands()
    max_rows = _max_rows(source)

    driver = f"""
const COMMANDS = PAYLOAD.commands;
const MAX_ROWS = PAYLOAD.maxRows;
const name = (c) => c.name;
const byName = (n) => COMMANDS.filter((c) => c.name === n)[0];

{functions}

const result = {{
  bare: filterCommands("").map(name),
  mdl: filterCommands("mdl").map(name),
  re: filterCommands("re").map(name),
  kg: filterCommands("kg").map(name),
  res: filterCommands("res").map(name),
  none: filterCommands("zzzq").map(name),
  fill: {{
    model: commandFill(byName("model")),
    read: commandFill(byName("read")),
    kg: commandFill(byName("kg")),
    help: commandFill(byName("help")),
  }},
  hits: hitPositions("model", "mdl"),
}};
process.stdout.write(JSON.stringify(result));
"""
    script = tmp_path_factory.mktemp("slash") / "driver.js"
    payload = tmp_path_factory.mktemp("slash") / "payload.json"
    script.write_text(driver.replace("PAYLOAD", "__PAYLOAD__"), encoding="utf-8")
    payload.write_text(
        json.dumps({"commands": commands, "maxRows": max_rows}), encoding="utf-8"
    )
    # The driver reads the payload from argv[2] to keep the JSON out of source.
    script.write_text(
        driver.replace(
            "PAYLOAD",
            "JSON.parse(require('fs').readFileSync(process.argv[2],'utf8'))",
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [NODE, str(script), str(payload)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    data = json.loads(completed.stdout)
    data["commands"] = commands
    return data


@requires_node
def test_bare_slash_lists_every_repl_command(menu) -> None:
    """A command nobody can see does not exist: no truncation on bare "/".

    Regression: MAX_ROWS was 12, so 27 of 39 commands were unreachable and
    ``model`` - the command this menu was asked about - never appeared.
    """
    shown = menu["bare"]
    registered = [c["name"] for c in menu["commands"]]
    assert "model" in shown, "the model command must be visible from a bare slash"
    assert shown == registered[: len(shown)]
    assert len(shown) == len(registered), (
        f"{len(registered) - len(shown)} commands are hidden by MAX_ROWS"
    )


@requires_node
def test_menu_cap_can_hold_the_whole_registry(menu) -> None:
    """MAX_ROWS must be at least the registry size, or the list is truncated."""
    print("commands=", len(menu["commands"]))
    assert len(menu["bare"]) == len(menu["commands"])


@requires_node
def test_fuzzy_subsequence_finds_model(menu) -> None:
    """The user asked for fuzzy search: "mdl" must reach "model"."""
    assert menu["mdl"][0] == "model", menu["mdl"]


@requires_node
def test_fuzzy_still_prefers_the_obvious_prefix_hits(menu) -> None:
    """Fuzzy must not scramble ranking: "re" keeps "read" near the front."""
    assert "read" in menu["re"][:3], menu["re"]
    assert "reconstruct" in menu["re"]


@requires_node
def test_filter_matches_on_the_command_name_only(menu) -> None:
    """"kg" selects kg itself, not every summary that mentions a knowledge graph."""
    assert menu["kg"][0] == "kg", menu["kg"]
    assert menu["none"] == []


@requires_node
def test_autofill_inserts_a_runnable_usage_line(menu) -> None:
    """Choosing a command fills a usage line the user can actually run.

    Regression: ``/model``'s usage IS the synopsis
    ``model [pick|list|load|...]``, and it was pasted verbatim into the
    composer. Meta-syntax (``[optional]``, ``<required>``, ``...``) is now
    stripped, because it describes a form rather than naming a command.
    """
    assert menu["fill"]["model"].startswith("/model ")
    assert menu["fill"]["read"] == "/read "
    # A usage offering alternatives autofills the first form only, arguments
    # and all: "/kg add" is a real verb, "/kg" alone is not.
    assert menu["fill"]["kg"].startswith("/kg add")
    # A command whose usage is just its name still gets a trailing space.
    assert menu["fill"]["help"].startswith("/help")


@requires_node
def test_autofill_never_produces_a_double_slash_or_empty_string(menu) -> None:
    for name, fill in menu["fill"].items():
        assert fill.startswith("/"), f"{name} autofill lost its slash: {fill!r}"
        assert not fill.startswith("//"), f"{name} autofill doubled the slash"
        assert fill.strip() != "/", f"{name} autofill is bare slash"


@requires_node
def test_highlight_marks_the_matched_letters(menu) -> None:
    """The typed letters are underlined in the menu row ("model" -> m0 d2 l4)."""
    assert menu["hits"] == [0, 2, 4], menu["hits"]


#: These stay static checks: they pin the XSS rule on the newly added code.
def test_highlighting_stays_textcontent_only() -> None:
    """The new highlighter must not become an innerHTML sink."""
    code = _read_page()
    body = code.split("function appendHighlighted")[1].split("function renderSlash")[0]
    assert "textContent" in body
    assert "innerHTML" not in body


def test_menu_still_never_sends_a_command() -> None:
    """Completing a command writes text; it must not POST a turn."""
    code = _read_page()
    choose = code.split("function chooseCommand")[1].split("function commandFill")[0]
    assert "postJSON" not in choose
    assert "fetch(" not in choose


def test_reset_warns_that_it_clears_agent_memory() -> None:
    """Reset is not "clear transcript": it must confirm before wiping memory.

    /api/chat/reset calls the REPL's clear_history, which empties the knowledge
    graph, working memory, the semantic index and the persisted memory files.
    """
    code = _read_page()
    body = code.split("async function reset()")[1]
    body = body.split("async function loadHistory")[0]
    assert "confirm(" in body, "reset must ask before destroying agent memory"
    assert "clear_history" in body
    assert "knowledge graph" in body.lower()
