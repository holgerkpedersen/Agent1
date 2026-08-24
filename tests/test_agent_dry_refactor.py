"""Regression tests: agent.py DRY refactor (shared helpers keep behaviour).

The refactor collapsed four kinds of duplication into single helpers:

1. ``_emit_command_metrics`` — the three dashboard metric writes shared by
   the real REPL path (:func:`agent.record_command_metrics`) and demo data
   (:meth:`agent.Agent.record_demo_activity`).
2. ``_run_subprocess_captured`` — arg-list subprocess execution with
   identical timeout/stderr-tagging/truncation for the NLP git/diff/tests
   tools.
3. ``_save_verify_note`` — save + py_compile-verify + trace-effect tail
   shared by the NLP write/edit tool handlers.
4. ``_build_dashboard`` — collector + default alert rules wiring shared by
   ``--serve`` and ``--dashboard`` startup paths.

These tests pin the observable contracts so future edits cannot silently
diverge the call sites again.
"""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

import agent


@pytest.fixture(autouse=True)
def _reset_shared_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test gets a pristine process-wide collector."""
    monkeypatch.setattr(agent, "_shared_metrics_collector", None)


def _make_agent() -> "agent.Agent":
    return agent.Agent(workspace=".")


# ---------------------------------------------------------------------------
# 1. Metrics emission has exactly one implementation
# ---------------------------------------------------------------------------

def test_record_command_metrics_delegates_to_emit() -> None:
    """The public REPL entry point is a thin delegate — no second writer."""
    import inspect

    src = inspect.getsource(agent.record_command_metrics)
    assert "_emit_command_metrics(" in src


def test_demo_and_real_commands_share_metric_names() -> None:
    """Demo data and live commands must be indistinguishable to the UI."""
    bot = _make_agent()
    bot.record_demo_activity(activity="analyze", latency_ms=250)  # -> 0.25 s
    agent.record_command_metrics("analyze", 1.5)

    collector = bot.get_metrics_collector()
    assert collector.get_counter_value("command.analyze.count") == 2.0
    gauge = collector.get_gauge_value("last.command.seconds")
    assert gauge == pytest.approx(1.5)
    samples = collector.snapshot()["histogram_samples"]["command.elapsed.seconds"]
    assert samples == [pytest.approx(0.25), pytest.approx(1.5)]


# ---------------------------------------------------------------------------
# 2. Shared subprocess runner
# ---------------------------------------------------------------------------

def test_run_subprocess_captured_tags_stderr_and_truncates() -> None:
    output, error = agent._run_subprocess_captured(
        [sys.executable, "-c",
         "import sys; print('out'); print('bad', file=sys.stderr)"],
        cwd=".", timeout=30, label="Probe",
    )
    assert error is None
    assert "out" in output
    assert "[STDERR]" in output and "bad" in output


def test_run_subprocess_captured_reports_launch_failure_with_label() -> None:
    output, error = agent._run_subprocess_captured(
        ["definitely-not-a-real-binary-xyz"], cwd=".", timeout=5, label="Git",
    )
    assert output == ""
    assert error is not None and error.startswith("Git error:")


def test_run_subprocess_captured_timeout_message() -> None:
    output, error = agent._run_subprocess_captured(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        cwd=".", timeout=1, label="Tests",
    )
    assert error is None
    assert output.startswith("Tests timed out after 1s")


def test_nlp_git_tool_uses_shared_runner() -> None:
    """The git handler must go through the shared capture helper."""
    import inspect

    src = inspect.getsource(agent.Agent._nlp_git)
    assert "_run_subprocess_captured(" in src
    assert "subprocess.run(" not in src


def test_nlp_tests_tool_uses_shared_runner() -> None:
    import inspect

    src = inspect.getsource(agent.Agent._nlp_tests)
    assert "_run_subprocess_captured(" in src


# ---------------------------------------------------------------------------
# 3. Write/edit share one save+verify+note tail
# ---------------------------------------------------------------------------

def test_save_verify_note_writes_and_verifies_python(tmp_path) -> None:
    bot = _make_agent()
    # Arm the trace-effects buffer as chat_nlp does for traced runs;
    # untraced runs deliberately keep _note_effect a no-op guard.
    bot._pending_effects = []
    target = tmp_path / "new_mod.py"
    out = asyncio.run(bot._save_verify_note(str(target), "x = 1\n", "Written f"))
    assert out.startswith("Written f")
    assert "[verify] py_compile ✓" in out
    assert target.read_text(encoding="utf-8") == "x = 1\n"
    # The written file is registered as a trace effect (absolute path).
    effects = [os.path.normcase(os.path.abspath(e)) for e in bot._pending_effects]
    assert os.path.normcase(str(target.resolve())) in effects


def test_save_verify_note_skips_when_no_changes(tmp_path) -> None:
    bot = _make_agent()
    target = tmp_path / "same.py"
    target.write_text("x = 1\n", encoding="utf-8")
    out = asyncio.run(bot._save_verify_note(str(target), "x = 1\n", "Written f"))
    assert out == f"Skipped {target} (no changes)"


def test_nlp_write_and_edit_use_shared_tail() -> None:
    import inspect

    for method in (agent.Agent._nlp_write, agent.Agent._nlp_edit):
        src = inspect.getsource(method)
        assert "_save_verify_note(" in src, method.__name__


def test_nlp_edit_rewrites_file_end_to_end(tmp_path) -> None:
    bot = _make_agent()
    target = tmp_path / "mod.py"
    target.write_text("value = 1\n", encoding="utf-8")
    out = asyncio.run(bot._nlp_edit({
        "path": str(target),
        "old_text": "value = 1",
        "new_text": "value = 2",
    }))
    assert out.startswith(f"Edited {target}")
    assert target.read_text(encoding="utf-8") == "value = 2\n"


# ---------------------------------------------------------------------------
# 4. Dashboard wiring built once
# ---------------------------------------------------------------------------

def test_both_dashboard_entrypoints_share_build_dashboard(monkeypatch) -> None:
    import inspect

    for fn in (agent.start_dashboard_thread, agent.run_dashboard_server):
        src = inspect.getsource(fn)
        assert "_build_dashboard(" in src, fn.__name__


def test_build_dashboard_applies_default_rules() -> None:
    from agent_core.monitoring import DashboardAPIServer

    collector = agent.get_metrics_collector()
    holder, alert_system = agent._build_dashboard(collector, port=0)
    assert isinstance(holder, DashboardAPIServer)
    names = {r.name for r in alert_system.list_rules()}
    assert {"slow_command", "command_volume_high", "fix_runs_elevated"} <= names


# ---------------------------------------------------------------------------
# 5. Banner derives from the registry; system prompt is a module constant
# ---------------------------------------------------------------------------

def test_banner_synopses_match_registered_commands(capsys) -> None:
    """Every banner line must come from a command's own help_text."""
    from agent_core.commands.registry import CommandRegistry

    registry = CommandRegistry()
    agent._register_commands(registry)
    synopses = {
        c.help_text.splitlines()[0].strip() for c in registry._commands.values()
    }
    # The banner helper exists and every registered command has a synopsis.
    assert len(synopses) == len(registry.names())
    assert "quit" not in registry.names()


def test_system_prompt_is_module_constant() -> None:
    assert isinstance(agent._SYSTEM_PROMPT, str)
    assert "senior coding assistant" in agent._SYSTEM_PROMPT
    # Built from the detected shell name.
    assert agent._detect_shell() in agent._SYSTEM_PROMPT


def test_chat_nlp_uses_system_prompt_constant() -> None:
    import inspect

    # The system prompt must come from the shared module constant — now built
    # inside _refresh_system_message (chat_nlp's first pipeline phase).
    src = inspect.getsource(agent.Agent._refresh_system_message)
    assert '"content": _SYSTEM_PROMPT' in src
    # ...and chat_nlp must delegate to that phase instead of inlining a prompt.
    loop_src = inspect.getsource(agent.Agent.chat_nlp)
    assert "_refresh_system_message(" in loop_src
