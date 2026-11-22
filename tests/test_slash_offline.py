"""Tests for the offline slash path used by the dashboard chat page.

``agent_core/commands/slash.py`` decides which leading-slash messages are
answered WITHOUT the model.  The behaviour it guarantees - and that
``USAGE.md`` documents - had no test before this file:

* an unlisted command returns ``None`` so the normal turn pipeline runs
  (nothing silently changes);
* ``/model``, ``/model list`` and ``/models`` return the SHARED catalogue from
  ``model_catalog``, and ``/models <name>`` is a switch, not a catalogue read;
* a state-changing subcommand (``kg add``, ``session use``) is REFUSED rather
  than executed, because the page must not edit memory behind the terminal;
* a command that prompts on stdin is refused instead of blocking an HTTP
  request on a terminal nobody is watching;
* ``run_slash`` never raises - a failure comes back as ``{"ok": False, ...}``.

These are behaviour tests against the REAL functions, with the registry and
the catalogue patched at their source so no LLM client or agent is built.
"""

from __future__ import annotations

import pytest

from agent_core.commands import slash as slash_mod
from agent_core.commands.slash import (
    OFFLINE_COMMANDS,
    parse_slash,
    run_slash,
)


class _FakeAgent:
    """Stand-in agent: these paths must not need a real one."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], object]] = []


class _FakeCommand:
    """Minimal command: records the call and prints a fixed line."""

    def __init__(self, name: str, output: str = "printed") -> None:
        self.name = name
        self.output = output
        self.calls: list[list[str]] = []

    async def execute(self, args, agent) -> bool:
        self.calls.append(list(args))
        print(self.output)
        return True


class _FakeRegistry:
    def __init__(self, commands: dict[str, object]) -> None:
        self._commands = commands

    def get(self, name: str):
        return self._commands.get(name)


@pytest.fixture()
def catalogue(monkeypatch):
    """Patch the shared catalogue so no provider is contacted."""
    seen: dict[str, object] = {}

    def fake_model_catalog(agent, fresh: bool = False):
        seen["fresh"] = fresh
        return {
            "current": "test-model",
            "active_provider": "fake",
            "groups": [
                {
                    "title": "fake",
                    "active": True,
                    "entries": [
                    {"value": "test-model", "current": True, "loaded": True},
                ],
                }
            ],
            "count": 1,
        }

    monkeypatch.setattr(slash_mod, "model_catalog", fake_model_catalog)
    return seen


@pytest.fixture()
def registry(monkeypatch):
    """Patch the registry builder and expose the commands it will serve."""
    commands: dict[str, object] = {}

    def fake_build_registry():
        return _FakeRegistry(commands)

    def install(name: str, command: object) -> None:
        commands[name] = command

    import agent_core.cli.interactive as interactive_mod

    monkeypatch.setattr(interactive_mod, "_build_registry", fake_build_registry)
    return install


# -- parse_slash ---------------------------------------------------------


def test_parse_slash_splits_name_and_args() -> None:
    assert parse_slash("/model list") == ("model", ["list"])


def test_parse_slash_returns_none_for_plain_text() -> None:
    assert parse_slash("what is the model list?") is None


def test_parse_slash_ignores_a_bare_slash() -> None:
    # A bare "/" is the autocomplete affordance, not a command.
    assert parse_slash("/") is None


def test_parse_slash_survives_unbalanced_quotes() -> None:
    # An unbalanced quote from the browser must not raise.
    assert parse_slash('/kg add "unclosed') is not None


# -- the allowlist is what makes the browser path safe -------------------


def test_offline_allowlist_is_exactly_the_registry_backed_commands() -> None:
    # Every name here is a command the REPL already has, answered without the
    # model. "clear" joined the set because the page could otherwise only
    # STAGE the request: "/clear" reached the LLM, which claimed to have
    # cleared a conversation it had never touched. Its own regression tests
    # (and its stdin-prompt trap) live in tests/test_slash_clear.py.
    assert OFFLINE_COMMANDS == {
        "model",
        "models",
        "help",
        "kg",
        "memory",
        "session",
        "clear",
    }


def test_unlisted_command_falls_through_to_the_turn_pipeline(registry) -> None:
    command = _FakeCommand("implement")
    registry("implement", command)

    assert run_slash(_FakeAgent(), "/implement foo") is None
    assert command.calls == [], "an unlisted command must not execute offline"


def test_plain_text_falls_through_to_the_turn_pipeline() -> None:
    assert run_slash(_FakeAgent(), "please fix the tests") is None


# -- the shared catalogue ------------------------------------------------


def test_models_command_returns_the_shared_catalogue(catalogue) -> None:
    result = run_slash(_FakeAgent(), "/models")

    assert result is not None
    assert result["ok"] is True
    assert result["kind"] == "models"
    assert result["command"] == "models"
    assert result["catalog"]["current"] == "test-model"


def test_model_list_shares_the_catalogue_not_terminal_output(
    catalogue, registry
) -> None:
    # Even with a real registry entry available, the structured payload wins.
    printed = _FakeCommand("model", output="THIS IS CAPTURED TEXT")
    registry("model", printed)

    result = run_slash(_FakeAgent(), "/model list")

    assert result is not None and result["kind"] == "models"
    assert result["reply"] != "THIS IS CAPTURED TEXT"
    assert printed.calls == []


def test_bare_model_is_a_catalogue_read(catalogue) -> None:
    result = run_slash(_FakeAgent(), "/model")

    assert result is not None and result["kind"] == "models"


def test_models_with_an_argument_is_a_switch_not_a_catalogue(
    registry, monkeypatch
) -> None:
    def boom(*a, **k):
        raise AssertionError("the catalogue must not be read for a switch")

    monkeypatch.setattr(slash_mod, "model_catalog", boom)
    models = _FakeCommand("models", output="switched")
    registry("models", models)

    # "/models <name>" must not be answered from the catalogue; the registry
    # runs the switch, so the REPL stays the only implementation.
    result = run_slash(_FakeAgent(), "/models some-model")

    assert result is not None
    assert result.get("kind") != "models"
    assert result["reply"] == "switched"
    assert models.calls == [["some-model"]]


def test_catalogue_is_asked_for_fresh_data() -> None:
    seen: dict[str, object] = {}

    def fake_model_catalog(agent, fresh: bool = False):
        seen["fresh"] = fresh
        return {"current": "", "active_provider": "", "groups": [], "count": 0}

    monkeypatch_ok = pytest.MonkeyPatch()
    monkeypatch_ok.setattr(slash_mod, "model_catalog", fake_model_catalog)
    try:
        run_slash(_FakeAgent(), "/models")
    finally:
        monkeypatch_ok.undo()

    assert seen["fresh"] is False, "the endpoint default must stay cacheable"


# -- regression: only /model has catalogue subcommands -------------------


@pytest.mark.parametrize("message", ["/help", "/memory", "/kg show", "/session"])
def test_other_allowlisted_commands_never_return_the_catalogue(
    message, catalogue, registry
) -> None:
    """/help and /memory once answered with the MODEL LIST instead of running.

    ``_models_payload`` compared only the SUBCOMMAND and never the command
    name, so any allowlisted command called with no args (or with "list") fell
    into the catalogue branch and returned ``kind: models``.  The regression is
    invisible from the surface: the reply looked like a plausible answer.
    """
    name = message.split()[0].lstrip("/")
    registry(name, _FakeCommand(name, output=f"real output for {name}"))

    result = run_slash(_FakeAgent(), message)

    assert result is not None
    assert result.get("kind") != "models", f"{message} must not return the catalogue"
    assert result["reply"] == f"real output for {name}"


# -- refusals -----------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "/kg add fact x y",
        "/kg remove fact x",
        "/session use other",
        "/session new scratch",
    ],
)
def test_state_changing_subcommands_are_refused(message, registry) -> None:
    kg = _FakeCommand("kg")
    session = _FakeCommand("session")
    registry("kg", kg)
    registry("session", session)

    result = run_slash(_FakeAgent(), message)

    assert result is not None
    assert result["ok"] is False
    assert "not run from the browser" in result["error"]
    assert kg.calls == [] and session.calls == [], "refused command must not execute"


def test_read_only_kg_show_is_allowed(registry) -> None:
    kg = _FakeCommand("kg", output="facts: 3")
    registry("kg", kg)

    result = run_slash(_FakeAgent(), "/kg show")

    assert result is not None and result["ok"] is True
    assert result["reply"] == "facts: 3"
    assert kg.calls == [["show"]]


def test_command_needing_terminal_input_is_refused(registry, monkeypatch) -> None:
    model = _FakeCommand("model")
    registry("model", model)
    # The module-level helper is what the path calls, so patch it there.
    monkeypatch.setattr(slash_mod, "needs_terminal_input", lambda command: True)

    # "load" is not a read-only catalogue sub, so the registry path runs.
    result = run_slash(_FakeAgent(), "/model load")

    assert result is not None
    assert result["ok"] is False
    assert "asks for input on the terminal" in result["error"]
    assert model.calls == [], "a stdin-prompting command must not execute"


# -- never raises -------------------------------------------------------


def test_missing_registry_entry_falls_through(registry) -> None:
    # Registry has no such command: hand the message to the pipeline.
    assert run_slash(_FakeAgent(), "/memory stats") is None


def test_command_exception_becomes_a_failed_result(registry) -> None:
    class Exploding(_FakeCommand):
        async def execute(self, args, agent) -> bool:
            raise RuntimeError("boom")

    registry("help", Exploding("help"))

    result = run_slash(_FakeAgent(), "/help")

    assert result is not None
    assert result["ok"] is False
    assert "boom" in result["error"]


def test_empty_command_output_says_so(registry) -> None:
    registry("memory", _FakeCommand("memory", output="   "))

    result = run_slash(_FakeAgent(), "/memory")

    assert result is not None and result["ok"] is True
    assert "produced no output" in result["reply"]
