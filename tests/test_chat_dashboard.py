"""Tests for the alternate dashboard chat: /chat page + /api/chat/* endpoints.

Safety behaviours pinned here:
- the page is served and the endpoints round-trip a message to the model;
- cross-origin and non-loopback-Host POSTs are rejected (CSRF guard) BEFORE
  any model call is attempted;
- oversized bodies are rejected;
- a metrics-only dashboard (no ChatService wired) answers 503-style
  ``{"ok": false}`` instead of crashing;
- the page never routes model output through innerHTML (XSS);
- the sidebar links to the chat page.

The LLM client is injected, so these tests never touch a provider.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from agent_core.monitoring.chat_api import ChatService
from agent_core.monitoring.dashboard_api import DashboardAPIHandler

STATIC = Path(__file__).resolve().parent.parent / "static"


class FakeClient:
    """Stand-in for LLMClient: records calls, echoes a deterministic reply."""

    model_name = "fake/model-1"

    def __init__(
        self,
        reply: str = "hello from the model",
        raises: Exception | None = None,
    ):
        self.reply = reply
        self.raises = raises
        self.calls: list[list[dict[str, str]]] = []

    async def chat(self, messages, tools=None, **kwargs):
        self.calls.append(list(messages))
        if self.raises is not None:
            raise self.raises
        return self.reply


@pytest.fixture()
def server():
    """A live dashboard server with a chat service wired to a fake client."""
    client = FakeClient()
    service = ChatService(client_factory=lambda: client)
    orig = DashboardAPIHandler.__dict__["_chat"]
    DashboardAPIHandler._chat = service
    httpd = ThreadingHTTPServer(("localhost", 0), DashboardAPIHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://localhost:{httpd.server_address[1]}"
    yield base, client, service
    httpd.shutdown()
    httpd.server_close()
    DashboardAPIHandler._chat = orig


@pytest.fixture()
def bare_server():
    """A dashboard server with NO chat service (metrics-only mode)."""
    orig = DashboardAPIHandler.__dict__["_chat"]
    DashboardAPIHandler._chat = None
    httpd = ThreadingHTTPServer(("localhost", 0), DashboardAPIHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://localhost:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    DashboardAPIHandler._chat = orig


def get(url: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.status, resp.read()


def post_json(
    url: str,
    body: dict,
    origin: str | None = "http://localhost",
    host: str | None = None,
) -> tuple[int, bytes]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST"
    )
    req.add_header("Content-Type", "application/json")
    if origin is not None:
        req.add_header("Origin", origin)
    if host is not None:
        req.add_header("Host", host)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, resp.read()


# -- page ---------------------------------------------------------------


@pytest.mark.parametrize("path", ["/chat", "/chat.html"])
def test_chat_page_is_served(server, path: str) -> None:
    base, _, _ = server
    status, body = get(base + path)
    assert status == 200
    html = body.decode("utf-8")
    assert 'id="chatInput"' in html
    assert 'id="messages"' in html
    assert "/api/chat/send" in html


@pytest.mark.parametrize("path", ["/chat", "/chat.html"])
def test_chat_page_is_served_uncached(server, path: str) -> None:
    """The page must never be served from the browser cache.

    Regression: the page is a single local file that gets edited in place.  With
    no cache header the browser kept rendering the OLD copy after a fix, so the
    repaired page still showed the symptoms the user reported (the shortcuts
    dialog painted over the page on Ctrl-F5, and the whole inline script visible
    as text under the Send button, leaving the Close button dead).  ``no-store``
    guarantees the running page is the file on disk.
    """
    base, _, _ = server
    with urllib.request.urlopen(base + path, timeout=10) as resp:
        assert resp.status == 200
        cache_control = resp.headers.get("Cache-Control", "")
    assert "no-store" in cache_control.lower(), (
        f"/chat served without Cache-Control: no-store (got {cache_control!r}); "
        "a browser may render a stale page after an edit"
    )


def test_chat_page_never_uses_innerhtml_for_model_output() -> None:
    """Model output is untrusted: every DOM sink must be textContent."""
    html = (STATIC / "chat.html").read_text(encoding="utf-8")
    # Strip HTML + JS comments first so prose ABOUT innerHTML cannot satisfy
    # (or trip) the guard - only real assignments count.
    code = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)
    assert not re.search(r"\.innerHTML\s*(\+?=)", code), (
        "chat.html assigns into innerHTML (XSS sink for model output)"
    )
    assert not re.search(r"\.(outerHTML|insertAdjacentHTML)\s*(\+?=)?\s*\(", code)
    assert "textContent" in code, "chat.html must render via textContent"


def test_sidebar_links_to_chat_page() -> None:
    sidebar = (Path(__file__).resolve().parent.parent / "partials" / "_sidebar.html")
    text = sidebar.read_text(encoding="utf-8")
    assert 'href="chat.html"' in text
    assert 'data-view="chat"' in text


# -- happy path ---------------------------------------------------------


def test_send_round_trips_and_grows_history(server) -> None:
    base, client, service = server
    status, body = post_json(base + "/api/chat/send", {"message": "hi there"})
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is True
    assert data["reply"] == "hello from the model"
    assert data["model"] == "fake/model-1"
    # The model saw the system prompt + the user turn.
    assert client.calls[0][0]["role"] == "system"
    assert client.calls[0][-1] == {"role": "user", "content": "hi there"}
    assert [m["role"] for m in service.history()] == ["user", "assistant"]


def test_history_endpoint_returns_stored_turns(server) -> None:
    base, _, _ = server
    post_json(base + "/api/chat/send", {"message": "one"})
    status, body = get(base + "/api/chat/history")
    assert status == 200
    data = json.loads(body)
    assert data["model"] == "fake/model-1"
    assert [m["content"] for m in data["messages"]] == [
        "one",
        "hello from the model",
    ]


def test_second_message_carries_the_first_as_context(server) -> None:
    base, client, _ = server
    post_json(base + "/api/chat/send", {"message": "first"})
    post_json(base + "/api/chat/send", {"message": "second"})
    sent = client.calls[-1]
    assert {"role": "user", "content": "first"} in sent
    assert {"role": "assistant", "content": "hello from the model"} in sent
    assert sent[-1] == {"role": "user", "content": "second"}


def test_reset_clears_history(server) -> None:
    base, _, service = server
    post_json(base + "/api/chat/send", {"message": "forget me"})
    assert service.history()
    status, body = post_json(base + "/api/chat/reset", {})
    assert status == 200
    assert json.loads(body)["ok"] is True
    assert service.history() == []


def test_empty_message_is_rejected_without_calling_the_model(server) -> None:
    base, client, _ = server
    status, body = post_json(base + "/api/chat/send", {"message": "   "})
    assert status == 200
    assert json.loads(body)["ok"] is False
    assert client.calls == []


def test_provider_exception_is_reported_not_raised(server) -> None:
    base, client, service = server
    client.raises = RuntimeError("backend down")
    status, body = post_json(base + "/api/chat/send", {"message": "ping"})
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is False
    assert "backend down" in data["error"]
    assert service.history() == [], "a failed turn must not be stored"


def test_in_band_provider_error_is_not_stored_as_an_answer(server) -> None:
    base, client, service = server
    client.reply = "[Error: no model loaded]"
    status, body = post_json(base + "/api/chat/send", {"message": "ping"})
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is False
    assert service.history() == []


# -- safety guards ------------------------------------------------------


def test_cross_origin_post_is_rejected(server) -> None:
    base, client, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        post_json(base + "/api/chat/send", {"message": "csrf"}, origin="http://evil.com")
    assert exc.value.code == 403
    assert client.calls == [], "guarded request must not reach the model"


def test_non_loopback_host_is_rejected(server) -> None:
    base, client, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        post_json(base + "/api/chat/send", {"message": "rebind"}, host="evil.com")
    assert exc.value.code == 403
    assert client.calls == []


def test_oversized_body_is_rejected(server) -> None:
    base, client, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        post_json(base + "/api/chat/send", {"message": "x" * (80 * 1024)})
    assert exc.value.code == 413
    assert client.calls == []


def test_unknown_api_path_is_404(server) -> None:
    base, _, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        post_json(base + "/api/chat/../mcp/call", {"name": "x"})
    assert exc.value.code == 404


# -- unconfigured -------------------------------------------------------


def test_unconfigured_chat_answers_error_not_crash(bare_server) -> None:
    status, body = post_json(bare_server + "/api/chat/send", {"message": "hi"})
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is False
    assert "not configured" in data["error"]


def test_unconfigured_history_is_empty(bare_server) -> None:
    status, body = get(bare_server + "/api/chat/history")
    assert status == 200
    data = json.loads(body)
    assert data["messages"] == []
    assert "error" in data


# -- service unit behaviour ---------------------------------------------


def test_history_is_capped() -> None:
    service = ChatService(
        client_factory=lambda: FakeClient(reply="ack"), max_history=4
    )
    for i in range(5):
        assert service.send(f"msg {i}")["ok"] is True
    history = service.history()
    assert len(history) == 4
    assert history[-1]["content"] == "ack"


def test_client_is_built_lazily_only_on_first_send() -> None:
    built: list[int] = []

    def factory():
        built.append(1)
        return FakeClient()

    service = ChatService(client_factory=factory)
    assert built == [], "client must not be built at construction"
    assert service.model_name == "unloaded"
    service.send("hello")
    assert built == [1]
    service.send("again")
    assert built == [1], "client must be reused, not rebuilt per message"


def test_long_message_is_rejected() -> None:
    from agent_core.monitoring.chat_api import MAX_MESSAGE_CHARS

    service = ChatService(client_factory=lambda: FakeClient())
    result = service.send("x" * (MAX_MESSAGE_CHARS + 1))
    assert result["ok"] is False
    assert "too long" in result["error"]


# -- AgentTurnRunner: the dashboard chat runs the REPL pipeline ----------
#
# The point of these tests is DRY: the dashboard must NOT own a second chat
# implementation.  A turn has to go through ``Agent.chat_nlp`` — the same
# function the terminal calls — so tools, the tool loop and persistence come
# from one place.  A fake agent pins that contract: if someone rewires the
# runner back to a bare LLM call, ``turns`` stays empty and these fail.


class FakeTurnAgent:
    """Stand-in for ``Agent``: records turns, exposes the REPL outcome slots."""

    model_name = "fake/agent-model"

    def __init__(self, answer: str = "ran the tool", error: str | None = None):
        self.answer = answer
        self.error = error
        self.turns: list[str] = []
        self.clears = 0
        self._chat_history: list[dict] = []
        self._last_turn_answer = ""
        self._last_turn_error = None

    async def chat_nlp(self, user_input, images=None):
        self.turns.append(user_input)
        self._chat_history.append({"role": "user", "content": user_input})
        self._chat_history.append({"role": "tool", "content": "raw tool output"})
        self._chat_history.append({"role": "assistant", "content": self.answer})
        self._last_turn_answer = self.answer
        self._last_turn_error = self.error

    def clear_history(self) -> None:
        self.clears += 1
        self._chat_history.clear()


def test_runner_sends_through_chat_nlp_not_a_bare_llm_call() -> None:
    from agent_core.monitoring.chat_api import AgentTurnRunner

    agent = FakeTurnAgent()
    result = AgentTurnRunner(agent=agent).send("read the notes")
    assert result["ok"] is True
    assert result["reply"] == "ran the tool"
    assert result["tools"] is True, "the UI must be told tools are available"
    assert agent.turns == ["read the notes"], "turn must run Agent.chat_nlp"


def test_runner_is_tool_capable_and_advertises_a_model() -> None:
    from agent_core.monitoring.chat_api import AgentTurnRunner

    runner = AgentTurnRunner(agent=FakeTurnAgent())
    assert runner.tool_capable is True
    assert runner.model_name == "fake/agent-model"


def test_runner_reports_turn_error_instead_of_an_answer() -> None:
    from agent_core.monitoring.chat_api import AgentTurnRunner

    agent = FakeTurnAgent(error="provider exploded")
    result = AgentTurnRunner(agent=agent).send("hi")
    assert result["ok"] is False
    assert "provider exploded" in result["error"]


def test_runner_reports_a_turn_that_produced_no_answer() -> None:
    from agent_core.monitoring.chat_api import AgentTurnRunner

    result = AgentTurnRunner(agent=FakeTurnAgent(answer="")).send("hi")
    assert result["ok"] is False
    assert "no answer" in result["error"]


def test_runner_rejects_blank_and_oversized_messages_without_a_turn() -> None:
    from agent_core.monitoring.chat_api import MAX_MESSAGE_CHARS, AgentTurnRunner

    agent = FakeTurnAgent()
    runner = AgentTurnRunner(agent=agent)
    assert runner.send("   ")["ok"] is False
    assert runner.send("x" * (MAX_MESSAGE_CHARS + 1))["ok"] is False
    assert agent.turns == [], "a rejected message must not reach the pipeline"


def test_runner_history_exposes_only_the_visible_transcript() -> None:
    from agent_core.monitoring.chat_api import AgentTurnRunner

    agent = FakeTurnAgent()
    runner = AgentTurnRunner(agent=agent)
    runner.send("hello")
    history = runner.history()
    # The ``tool`` row is internal plumbing, not conversation.
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[-1]["content"] == "ran the tool"


def test_runner_history_flattens_multimodal_blocks() -> None:
    """A pasted image turn must not render a Python repr in the transcript."""
    from agent_core.monitoring.chat_api import AgentTurnRunner

    agent = FakeTurnAgent()
    agent._chat_history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
            ],
        }
    ]
    assert AgentTurnRunner(agent=agent).history() == [
        {"role": "user", "content": "what is this?"}
    ]


def test_runner_reset_uses_the_agents_own_clear_history() -> None:
    from agent_core.monitoring.chat_api import AgentTurnRunner

    agent = FakeTurnAgent()
    runner = AgentTurnRunner(agent=agent)
    runner.send("forget me")
    runner.reset()
    assert agent.clears == 1, "reset must reuse Agent.clear_history (DRY)"
    assert runner.history() == []


def test_runner_model_name_does_not_build_the_agent() -> None:
    from agent_core.monitoring.chat_api import AgentTurnRunner

    built: list[int] = []

    def factory():
        built.append(1)
        return FakeTurnAgent()

    runner = AgentTurnRunner(agent_factory=factory)
    assert runner.model_name == "unloaded"
    assert built == [], "reading a badge must not boot an agent"
    runner.send("hi")
    assert built == [1]


def test_runner_reuses_a_live_repl_agent_instead_of_forking_it() -> None:
    """Terminal + browser must share ONE conversation, not two histories."""
    import agent as agent_module

    from agent_core.monitoring.chat_api import AgentTurnRunner

    shared = FakeTurnAgent(answer="from the terminal session")
    built: list[int] = []

    def factory():
        built.append(1)
        return FakeTurnAgent(answer="from a second agent")

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(agent_module, "_shared_agent", shared, raising=False)
        runner = AgentTurnRunner(agent_factory=factory)
        result = runner.send("hi from the browser")
        assert result["reply"] == "from the terminal session"
        assert shared.turns == ["hi from the browser"]
        assert built == [], "never build a second Agent beside a live REPL"


def test_runner_serialises_concurrent_sends() -> None:
    from agent_core.monitoring.chat_api import AgentTurnRunner

    agent = FakeTurnAgent()
    runner = AgentTurnRunner(agent=agent)
    results: list[dict] = []

    def send(text: str) -> None:
        results.append(runner.send(text))

    threads = [threading.Thread(target=send, args=(f"m{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [r["ok"] for r in results] == [True] * 4
    assert sorted(agent.turns) == ["m0", "m1", "m2", "m3"]


class _StopTurn(Exception):
    """Halts ``chat_nlp`` right after its busy flag is set (no LLM needed)."""


def test_chat_nlp_refuses_a_turn_while_one_is_running(tmp_path) -> None:
    """Two surfaces can drive ONE agent: a racing turn must be refused.

    Without this guard the dashboard chat and the terminal REPL could interleave
    ``_chat_history`` and the tool loop.  The flag is exactly what a live turn on
    the other surface sets, so setting it pins the real decorated ``chat_nlp``.
    """
    import asyncio

    import agent as agent_mod

    bot = agent_mod.Agent(workspace=str(tmp_path))
    reached: list[object] = []
    bot._ensure_original_goal = lambda pending_user_input=None: reached.append(1)
    bot._turn_in_progress = True  # the other surface is mid-turn

    with pytest.raises(RuntimeError, match="already running"):
        asyncio.run(bot.chat_nlp("second surface"))

    assert reached == [], "a refused turn must not touch the pipeline"
    assert bot._turn_in_progress is True, "the running turn keeps ownership"


def test_chat_nlp_sets_and_clears_the_busy_flag_around_a_turn(tmp_path) -> None:
    """The flag must cover the whole turn and clear even when it raises.

    A crashed turn that left the flag set would wedge the agent for every later
    turn — including the other surface — so the finally-clear is load-bearing.
    """
    import asyncio

    import agent as agent_mod

    bot = agent_mod.Agent(workspace=str(tmp_path))
    seen: dict[str, object] = {}

    def first_phase(pending_user_input=None):
        seen["during"] = bot._turn_in_progress
        seen["input"] = pending_user_input
        raise _StopTurn

    bot._ensure_original_goal = first_phase

    with pytest.raises(_StopTurn):
        asyncio.run(bot.chat_nlp("first surface"))
    assert seen["input"] == "first surface"
    assert seen["during"] is True, "flag must be set for the whole turn"
    assert bot._turn_in_progress is False, "flag must clear on the way out"

    # Not wedged: the next turn reaches the pipeline again instead of being
    # refused by a stale flag.
    with pytest.raises(_StopTurn):
        asyncio.run(bot.chat_nlp("second surface"))
    assert seen["input"] == "second surface"


def test_repl_publishes_its_agent_for_the_dashboard() -> None:
    import agent as agent_module

    from agent_core.cli.interactive import publish_shared_agent

    marker = object()
    with pytest.MonkeyPatch.context() as mp:
        mp.delattr(agent_module, "_shared_agent", raising=False)
        publish_shared_agent(marker)
        assert agent_module._shared_agent is marker


def test_dashboard_wires_the_tool_capable_runner() -> None:
    """The shipped wiring must be the runner, never the LLM-only service."""
    import agent_dashboard
    from agent_core.monitoring import MetricsCollector
    from agent_core.monitoring.chat_api import AgentTurnRunner

    orig = DashboardAPIHandler.__dict__["_chat"]
    try:
        agent_dashboard._build_dashboard(MetricsCollector(), 0)
        wired = DashboardAPIHandler._chat
        assert isinstance(wired, AgentTurnRunner)
        assert wired.tool_capable is True
        assert wired.model_name == "unloaded", "boot must not load an LLM"
    finally:
        DashboardAPIHandler._chat = orig


def test_chat_page_does_not_claim_it_cannot_use_tools() -> None:
    """The page copy must match what the runner actually does."""
    html = (STATIC / "chat.html").read_text(encoding="utf-8")
    assert "no tool access" not in html.lower()
    assert "same turn pipeline as the repl" in html.lower()


# -- slash-command menu (opencode-style "/" autocomplete) ----------------
#
# The page shows the REPL command list in a popup above the composer, the way
# opencode opens its slash autocomplete on "/".  What these tests pin:
# - the catalog IS the REPL registry (no second, drifting list);
# - the endpoint is a read-only GET that degrades instead of 500-ing;
# - the menu COMPLETES text and never executes anything itself;
# - it is built with textContent only (the XSS rule applies to it too).


def _repl_registry():
    from agent_core.cli.interactive import _build_registry

    return _build_registry()


def test_command_catalog_mirrors_the_repl_registry_exactly() -> None:
    """One source of truth: the menu cannot drift from the REPL."""
    from agent_core.commands.catalog import command_catalog

    registry = _repl_registry()
    catalog = command_catalog(registry)
    assert {entry["name"] for entry in catalog} == registry.names()
    assert len(catalog) == len(registry.names()), "no duplicate/no missing entries"
    assert [e["name"] for e in catalog] == sorted(e["name"] for e in catalog)


def test_command_catalog_splits_usage_from_summary() -> None:
    """``help_text`` is "<usage> - <summary>"; the menu needs both parts."""
    from agent_core.commands.catalog import split_synopsis

    assert split_synopsis("read <path>        - Read a file") == (
        "read <path>",
        "Read a file",
    )
    # Em dash separators are used by kg / kanban / multillm.
    assert split_synopsis("kg add|show|remove ... \u2014 edit the graph") == (
        "kg add|show|remove ...",
        "edit the graph",
    )
    # No separator: the line is both, and nothing is invented.
    assert split_synopsis("Full pipeline") == ("Full pipeline", "Full pipeline")
    assert split_synopsis("") == ("", "")


def test_command_catalog_flags_commands_that_need_a_terminal() -> None:
    """A stdin-prompting command is marked, not silently offered as usable."""
    from agent_core.commands.catalog import command_catalog

    by_name = {e["name"]: e for e in command_catalog(_repl_registry())}
    # These call read_choice/read_input/read_stdin (see commands/*.py).
    for name in ("fix", "optimize", "write", "workflow"):
        assert by_name[name]["needs_input"] is True, f"{name} prompts on stdin"
    # Read-only/argument-only commands must stay usable from the browser.
    for name in ("read", "search", "git", "help", "model"):
        assert by_name[name]["needs_input"] is False, f"{name} needs no stdin"


def test_catalog_entries_carry_renderable_fields() -> None:
    from agent_core.commands.catalog import command_catalog

    for entry in command_catalog(_repl_registry()):
        assert set(entry) == {"name", "usage", "summary", "detail", "needs_input"}
        assert entry["name"]
        assert entry["summary"], f"{entry['name']} has no summary to show"


def test_commands_endpoint_lists_the_repl_commands(server) -> None:
    """GET /api/chat/commands is how the page learns the list."""
    base, _client, _service = server
    status, body = get(base + "/api/chat/commands")
    assert status == 200
    data = json.loads(body)
    names = {c["name"] for c in data["commands"]}
    assert names == _repl_registry().names()
    assert data["count"] == len(names)


def test_commands_endpoint_is_served_without_a_chat_service(bare_server) -> None:
    """A metrics-only dashboard must still answer, with an empty list."""
    status, body = get(bare_server + "/api/chat/commands")
    assert status == 200
    data = json.loads(body)
    assert data["commands"] == []
    assert "error" in data


def test_agent_runner_exposes_the_registry_without_building_the_agent() -> None:
    """Listing commands must not boot an LLM client."""
    from agent_core.monitoring.chat_api import AgentTurnRunner

    built: list[int] = []

    def _factory():
        built.append(1)
        raise AssertionError("commands() must not build the agent")

    runner = AgentTurnRunner(agent_factory=_factory)
    names = {c["name"] for c in runner.commands()}
    assert "read" in names and "kanban" in names
    assert built == []


def test_chat_page_ships_the_slash_menu() -> None:
    """The page must contain the popup, the button and the fetch of the list."""
    html = (STATIC / "chat.html").read_text(encoding="utf-8")
    assert 'id="slashMenu"' in html
    assert 'id="slashBtn"' in html
    assert "/api/chat/commands" in html
    assert "slashQuery" in html
    # Keyboard contract: arrows move, Tab/Enter complete, Escape closes.
    for key in ("ArrowDown", "ArrowUp", "Tab", "Escape"):
        assert key in html, f"slash menu must handle {key}"


def test_chat_page_slash_menu_never_uses_chat_send_directly() -> None:
    """Completing a command WRITES text; it never fires a turn.

    A command needs arguments, and firing an argument-less ``/write`` from a
    menu click would be a surprise side effect.  Exactly one call site may
    POST a message, and it is the user's Enter/click on Send.
    """
    html = (STATIC / "chat.html").read_text(encoding="utf-8")
    code = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)
    assert code.count("postJSON(api.send") == 1
    # The menu builder must not reach for the network at all.
    choose = code.split("function chooseCommand")[1].split("function syncSlash")[0]
    assert "postJSON" not in choose
    assert "fetch(" not in choose


def test_chat_page_slash_menu_renders_via_textcontent() -> None:
    """The existing XSS rule must cover the menu code too."""
    html = (STATIC / "chat.html").read_text(encoding="utf-8")
    code = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)
    assert not re.search(r"\.innerHTML\s*(\+?=)", code)
    menu = code.split("function renderSlash")[1].split("function openSlash")[0]
    assert "textContent" in menu
    # Nothing in the menu path may build markup from a string.
    assert "innerHTML" not in menu

