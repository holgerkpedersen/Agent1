"""Regression tests: the /chat page's model picker actually picks a model.

Reported bug: "/model is not making it possible to pick a model from the found
models. Nothing happens."

Two independent defects caused it, and each gets a test here:

1. SERVER: ``/api/chat/models/select`` was listed in ``_CHAT_POST_PATHS`` but
   ``_chat_post`` had no branch for it, so the request fell through to
   ``service.send("")`` and answered ``{"ok": false, "error": "message is
   required"}``. ``select_model`` was never called. ``test_select_endpoint_*``
   pin the routing.

2. PAGE: ``static/chat.html`` never fetched ``/api/chat/models`` at all (the
   only occurrence of "catalog" in the file was a comment). "/model" was sent
   to the LLM as a prompt, and the badge merely typed "/model " into the
   composer. ``test_page_*`` extract the page's REAL JS and run it under node,
   the same technique tests/test_chat_slash_menu.py uses, so they pin the
   behaviour and not a string in the markup.

Node is optional tooling: the JS tests skip when it is absent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

import pytest

from agent_core.monitoring.chat_api import ChatService
from agent_core.monitoring.dashboard_api import DashboardAPIHandler

STATIC = Path(__file__).resolve().parent.parent / "static"
PAGE = STATIC / "chat.html"

NODE = shutil.which("node")
requires_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

#: Page functions executed by the node driver. They are deliberately
#: dependency-free (no DOM, no network) so they can be extracted verbatim.
_NEEDED = (
    "wantsModelList",
    "catalogOptions",
    "catalogCurrent",
    "stripUsageMeta",
    "commandFill",
    "runsOnSelect",
)


def _live_commands() -> List[Dict[str, Any]]:
    """The catalog the dashboard actually serves, from the live registry.

    The composer fill is derived from these usage lines, so taking them from
    the real registry is what stops the fixture drifting from the REPL.
    """
    from agent_core.cli.interactive import _build_registry
    from agent_core.commands.catalog import command_catalog

    return command_catalog(_build_registry())

CATALOG: Dict[str, Any] = {
    "current": "m-a",
    "active_provider": "lmstudio",
    "model": "m-a",
    "count": 3,
    "groups": [
        {
            "provider": "lmstudio",
            "title": "LM Studio",
            "active": True,
            "entries": [
                {
                    "value": "m-a",
                    "label": "M A",
                    "provider": "lmstudio",
                    "current": True,
                    "loaded": True,
                    "detail": "9B",
                },
                {
                    "value": "m-b",
                    "label": "M B",
                    "provider": "lmstudio",
                    "current": False,
                    "loaded": False,
                    "detail": "",
                },
            ],
        },
        {
            "provider": "openrouter",
            "title": "OpenRouter",
            "hint": "set OPENROUTER_API_KEY",
            "entries": [
                {
                    "value": "or/c",
                    "label": "C",
                    "provider": "openrouter",
                    "current": False,
                    "loaded": False,
                    "detail": "",
                }
            ],
        },
    ],
}


class FakeClient:
    """Stand-in for LLMClient so no test touches a provider."""

    model_name = "m-a"

    def __init__(self) -> None:
        self.calls: List[Any] = []

    async def chat(self, messages, tools=None, **kwargs):
        self.calls.append(list(messages))
        return "hello"


class RecordingService(ChatService):
    """The real ChatService with the switch recorded, not performed."""

    def __init__(self) -> None:
        super().__init__(client_factory=FakeClient)
        self.selects: List[tuple] = []

    def models(self, fresh: bool = False) -> Dict[str, Any]:
        return dict(CATALOG)

    def select_model(self, model: str, provider=None) -> Dict[str, Any]:
        self.selects.append((model, provider))
        return {"ok": True, "model": model, "provider": provider, "changed": True}


@pytest.fixture()
def server():
    """A live dashboard server wired to the recording service."""
    service = RecordingService()
    orig = DashboardAPIHandler.__dict__["_chat"]
    DashboardAPIHandler._chat = service
    httpd = ThreadingHTTPServer(("localhost", 0), DashboardAPIHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://localhost:{httpd.server_address[1]}", service
    finally:
        httpd.shutdown()
        httpd.server_close()
        DashboardAPIHandler._chat = orig


def post_json(url: str, body: Dict[str, Any]) -> Dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST"
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Origin", "http://localhost")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def get_json(url: str) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=15) as resp:
        return json.loads(resp.read())


# -- 1. the server route -------------------------------------------------


def test_select_endpoint_switches_the_model(server) -> None:
    """Regression: the pick POSTed, answered "message is required", did nothing."""
    base, service = server
    body = post_json(
        base + "/api/chat/models/select", {"model": "m-b", "provider": "lmstudio"}
    )
    assert body.get("ok") is True, body
    assert body.get("model") == "m-b"
    assert service.selects == [("m-b", "lmstudio")], "select_model was not called"


def test_select_endpoint_accepts_the_entry_value_key(server) -> None:
    """The catalog rows carry ``value``; the picker may send either key."""
    base, service = server
    body = post_json(base + "/api/chat/models/select", {"value": "or/c"})
    assert body.get("ok") is True, body
    assert service.selects == [("or/c", None)], service.selects


def test_select_endpoint_rejects_an_empty_model(server) -> None:
    """An empty name must not reach select_model (it would clear the model)."""
    base, service = server
    body = post_json(base + "/api/chat/models/select", {"model": "   "})
    assert body.get("ok") is False
    assert "model" in (body.get("error") or "").lower()
    assert service.selects == [], service.selects


def test_models_endpoint_serves_the_catalog_the_picker_renders(server) -> None:
    """The picker's data source: groups/entries with value, label, current."""
    base, _ = server
    body = get_json(base + "/api/chat/models")
    assert body["count"] == 3, body
    first = body["groups"][0]["entries"][0]
    assert first["value"] == "m-a"
    assert first["current"] is True


# -- 2. the page's own JS ------------------------------------------------


def _read_page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _extract_function(source: str, name: str) -> str:
    """Return the source of top-level ``function <name>(...) {...}``."""
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


@pytest.fixture(scope="module")
def page(tmp_path_factory) -> Dict[str, Any]:
    """Run the page's picker functions under node and return their results."""
    if NODE is None:  # pragma: no cover - requires_node guards the callers
        pytest.skip("node is not installed")
    source = _read_page()
    functions = "\n\n".join(_extract_function(source, name) for name in _NEEDED)
    commands = _live_commands()

    driver = """
const CATALOG = JSON.parse(
  require("fs").readFileSync(process.argv[2], "utf8")
);
const COMMANDS = JSON.parse(
  require("fs").readFileSync(process.argv[3], "utf8")
);
__FUNCTIONS__

const options = catalogOptions(CATALOG);
const byName = (n) => COMMANDS.filter((c) => c.name === n)[0];
const result = {
  wants: {
    bare: wantsModelList("/model"),
    list: wantsModelList("/model list"),
    ls: wantsModelList("/model  LS"),
    models: wantsModelList("/models"),
    load: wantsModelList("/model load m-a"),
    provider: wantsModelList("/model provider openrouter"),
    none: wantsModelList("explain the repo"),
    slashOther: wantsModelList("/kg show"),
  },
  // Every command's composer fill: what Enter inserts before it can be sent.
  fills: COMMANDS.map((c) => [c.name, commandFill(c)]),
  select: COMMANDS.filter((c) => runsOnSelect(c)).map((c) => c.name),
  options: options,
  current: catalogCurrent(CATALOG),
  emptyCurrent: catalogCurrent({groups: []}),
  emptyOptions: catalogOptions(null),
  hintOnly: catalogOptions({
    current: "",
    groups: [{provider: "p", title: "P", hint: "no server", entries: []}],
  }),
};
process.stdout.write(JSON.stringify(result));
"""
    script = tmp_path_factory.mktemp("picker") / "driver.js"
    payload = tmp_path_factory.mktemp("picker") / "payload.json"
    script.write_text(driver.replace("__FUNCTIONS__", functions), encoding="utf-8")
    payload.write_text(json.dumps(CATALOG), encoding="utf-8")
    commands_file = tmp_path_factory.mktemp("picker") / "commands.json"
    commands_file.write_text(json.dumps(commands), encoding="utf-8")
    completed = subprocess.run(
        [NODE, str(script), str(payload), str(commands_file)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


@requires_node
def test_page_routes_a_bare_model_command_to_the_picker(page) -> None:
    """Regression: "/model" was sent to the LLM, which then invented a list."""
    wants = page["wants"]
    assert wants["bare"] is True
    assert wants["list"] is True
    assert wants["ls"] is True
    assert wants["models"] is True


@requires_node
def test_page_leaves_model_mutations_to_the_repl(page) -> None:
    """"/model load x" CHANGES something: the REPL path must still run it."""
    wants = page["wants"]
    assert wants["load"] is False
    assert wants["provider"] is False
    assert wants["none"] is False
    assert wants["slashOther"] is False


@requires_node
def test_page_flattens_every_group_into_pickable_rows(page) -> None:
    """The reported symptom: models were "found" but could not be picked."""
    options = page["options"]
    assert [o["value"] for o in options] == ["m-a", "m-b", "or/c"]
    assert options[0]["current"] is True
    assert options[0]["groupTitle"] == "LM Studio"
    assert options[1]["current"] is False
    # A per-entry provider wins; the group provider is the fallback.
    assert options[2]["provider"] == "openrouter"
    assert options[2]["detail"] == ""


@requires_node
def test_page_reports_the_current_model_from_the_catalog(page) -> None:
    assert page["current"] == "m-a"
    assert page["emptyCurrent"] == ""


@requires_node
def test_page_survives_an_empty_or_missing_catalog(page) -> None:
    """An unreachable provider is an empty list, never a crash."""
    assert page["emptyOptions"] == []
    assert page["hintOnly"] == []


# -- 2b. the reported bug: Enter on "/model" ------------------------------


@requires_node
def test_page_never_pastes_a_synopsis_bracket_into_the_composer(page) -> None:
    """Regression: Enter on "/model" filled the box with "/model [pick|...".

    ``commandFill`` copied the registry's usage line verbatim, and
    ``/model``'s usage IS the bracketed synopsis
    ``model [pick|list|load|unload|reload|provider|profile|jev]``.  The
    brackets and pipes then sat in the composer, where they are neither a
    command nor a question: pressing Enter again sent that literal text.
    """
    fills = dict(page["fills"])
    assert fills, "the node driver returned no composer fills"
    for name, fill in fills.items():
        with_subtest = (name, fill)
        assert "[" not in fill and "]" not in fill, (
            f"composer fill for /{name} contains a bracket: {fill!r} "
            f"{with_subtest}"
        )
        assert "|" not in fill, f"composer fill for /{name} has a pipe: {fill!r}"
        assert fill.startswith("/"), f"/{name} fill lost its slash: {fill!r}"


@requires_node
def test_page_puts_a_pickable_bare_command_in_the_composer(page) -> None:
    """/model must complete to "/model", not to its argument synopsis.

    The user reported "/model -> /model [pick|list...]" and "nothing happens".
    A command that already does what the user asked (showing the model list)
    is left bare so Enter runs it.
    """
    fills = dict(page["fills"])
    assert fills["model"].strip() == "/model"
    assert fills["models"].strip() == "/models"
    assert fills["memory"].strip() == "/memory"


@requires_node
def test_page_commands_with_arguments_keep_the_pasteable_form(page) -> None:
    """An argument-taking command fills its first REAL form, never a bracket.

    ``/read`` keeps its bare name, and ``/kg`` fills the first verb of its
    ``kg add|show|remove ...`` synopsis because "/kg" alone is not a command.
    """
    fills = dict(page["fills"])
    assert fills["read"].strip() == "/read"
    assert fills["kg"].strip() == "/kg add"


@requires_node
def test_page_marks_model_and_models_as_run_on_select(page) -> None:
    """/model and /models give the picker its answer, so Enter may execute them."""
    assert set(page["select"]) == {"model", "models"}, page["select"]


@requires_node
def test_enter_on_model_reaches_send_with_the_command_in_the_composer(tmp_path) -> None:
    """Regression: Enter left the composer EMPTY, so send() returned early.

    ``send()`` starts with ``if (!text) return;``.  Running the command from the
    menu must therefore put it in the composer and only then call send - the
    order this driver records.  If the picker never opens after this, the
    failure is in the network path, not in the keyboard path.
    """
    source = _read_page()
    functions = _extract_function(source, "runCommandFromMenu")

    driver = """
let SENT = null;
let SEND_CALLS = 0;
let CLOSED = false;
const input = { value: "", selectionStart: 0, style: {} };
const $ = (id) => (id === "chatInput" ? input : null);
const autosize = () => {};
const closeSlash = () => { CLOSED = true; };
function send() { SEND_CALLS += 1; SENT = input.value; }

__FUNCTIONS__

// What pressing Enter on the highlighted "/model" row does.
runCommandFromMenu({ name: "model", usage: "model [pick|list|load]" });
process.stdout.write(
  JSON.stringify({ sent: SENT, calls: SEND_CALLS, closed: CLOSED, box: input.value })
);
"""
    script = tmp_path / "enter.js"
    script.write_text(driver.replace("__FUNCTIONS__", functions), encoding="utf-8")
    completed = subprocess.run(
        [NODE, str(script)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["calls"] == 1, result
    assert result["sent"] == "/model", (
        f"send() saw {result['sent']!r}: the composer was cleared too early"
    )
    assert result["closed"] is True, "the menu must close when the command runs"


@requires_node
def test_no_page_fill_repeats_the_command_name(page) -> None:
    """A bare command must not expand to "/memory [--json] ..." either."""
    for name, fill in page["fills"]:
        head = fill.strip().split(" ")[0].lstrip("/").lower()
        assert head == name.lower(), f"/{name} filled in as {fill!r}"


# -- 3. wiring, checked on the markup itself -----------------------------


def test_page_declares_the_picker_endpoints_and_panel() -> None:
    html = _read_page()
    assert "/api/chat/models" in html, "the page must fetch the catalog"
    assert "/api/chat/models/select" in html, "the page must send the pick"
    assert 'id="modelPanel"' in html, "the picker needs a panel to render into"


def test_page_never_writes_model_names_with_innerHTML() -> None:
    """Model names come from a provider: textContent only, never a sink."""
    html = _read_page()
    start = html.index("/* ---- model picker")
    end = html.index("async function reset()")
    panel = html[start:end]
    assert "innerHTML" not in panel, "the picker must not build markup from models"
