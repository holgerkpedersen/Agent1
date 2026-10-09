"""Regression: the Kanban -> Agent1 inbound queue must actually be drained.

``QueueProcessor`` existed, was fully tested, and was constructed *nowhere*
outside tests.  So the inbound half of the sync was dead code in production:
Agent1 wrote ``issue_create``/``issue_resolve`` to ``queue-agent1-to-kanban``
while every Kanban ``card_create``/``card_update``/``card_move``/
``card_delete`` reply sat unread in ``queue-kanban-to-agent1`` forever.

The fix wires a real boot path: ``agent.py:main()`` calls
``_start_kanban_inbound()`` -> ``kanban_bridge.start_inbound_processor()``,
gated on ``KANBAN_SYNC_ENABLED=1`` and idempotent per process.
"""
from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_boot_handles(tmp_path, monkeypatch):
    """Reset the process-wide boot sentinel and redirect queue paths.

    ``_BOOT_HANDLES`` is module state whose whole purpose is to persist across
    calls, so a test that left it populated would make every later
    ``start_inbound_processor()`` a no-op and silently pass a broken
    implementation.  Redirecting the queue globals keeps the started poller
    off the real ``data/`` queue shared with the running app.
    """
    import harnessfix.kanban_bridge as kb

    saved = list(kb._BOOT_HANDLES)
    kb._BOOT_HANDLES.clear()

    data = tmp_path / "data"
    (data / "queue-kanban-to-agent1").mkdir(parents=True)
    monkeypatch.setattr(kb, "DATA_DIR", data)
    monkeypatch.setattr(kb, "QUEUE_KANBAN_TO_AGENT1", data / "queue-kanban-to-agent1")
    monkeypatch.setattr(kb, "QUEUE_AGENT1_TO_KANBAN", data / "queue-agent1-to-kanban")

    # Keep these tests independent of whatever the developer has in the real
    # repo .env: with KANBAN_SYNC_ENABLED=1 there, every "disabled" case would
    # start a real poller from dotenv.  Process env still wins per test.
    monkeypatch.setattr(kb, "ENV_FILE_PATH", tmp_path / "absent.env")

    try:
        yield
    finally:
        for proc in kb._BOOT_HANDLES:
            proc.stop()
        kb._BOOT_HANDLES[:] = saved


def _inbound_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "kanban-inbound"]


# ---------------------------------------------------------------------------
# start_inbound_processor
# ---------------------------------------------------------------------------

class TestStartInboundProcessor:
    def test_disabled_by_default(self, monkeypatch):
        """No env var -> no poller, no thread (opt-in integration)."""
        import harnessfix.kanban_bridge as kb
        monkeypatch.delenv("KANBAN_SYNC_ENABLED", raising=False)

        assert kb.start_inbound_processor() is None
        assert kb.inbound_processor_running() is False
        assert _inbound_threads() == []

    def test_disabled_when_flag_is_zero(self, monkeypatch):
        import harnessfix.kanban_bridge as kb
        monkeypatch.setenv("KANBAN_SYNC_ENABLED", "0")

        assert kb.start_inbound_processor() is None
        assert kb.inbound_processor_running() is False

    def test_enabled_starts_daemon_poller(self, monkeypatch):
        """The core regression: enabled sync must actually spawn the poller."""
        import harnessfix.kanban_bridge as kb
        monkeypatch.setenv("KANBAN_SYNC_ENABLED", "1")

        proc = kb.start_inbound_processor(poll_interval=0.05)

        assert proc is not None, "sync enabled but no processor was started"
        assert kb.inbound_processor_running() is True
        assert kb._BOOT_HANDLES == [proc]

        threads = _inbound_threads()
        assert len(threads) == 1, f"expected exactly one poller thread, got {threads}"
        assert threads[0].daemon, (
            "poller must be a daemon or it blocks interpreter exit"
        )

    def test_second_call_reuses_first_processor(self, monkeypatch):
        """A second poller on the same queue only widens the torn-line window."""
        import harnessfix.kanban_bridge as kb
        monkeypatch.setenv("KANBAN_SYNC_ENABLED", "1")

        first = kb.start_inbound_processor(poll_interval=0.05)
        second = kb.start_inbound_processor(poll_interval=0.05)

        assert first is not None
        assert second is None, "must reuse, not start a second poller"
        assert kb._BOOT_HANDLES == [first]
        assert len(_inbound_threads()) == 1

    def test_stop_is_registered_with_atexit(self, monkeypatch):
        """atexit matters: a killed mid-append writer tears messages.jsonl."""
        import harnessfix.kanban_bridge as kb
        monkeypatch.setenv("KANBAN_SYNC_ENABLED", "1")

        registered: list = []
        monkeypatch.setattr(kb.atexit, "register", registered.append)

        proc = kb.start_inbound_processor(poll_interval=0.05)

        assert proc is not None
        assert registered == [proc.stop], "proc.stop must be the atexit hook"

    def test_construction_failure_is_swallowed(self, monkeypatch):
        """A broken queue must never stop the agent from booting."""
        import harnessfix.kanban_bridge as kb
        monkeypatch.setenv("KANBAN_SYNC_ENABLED", "1")

        def _boom(*a, **kw):
            raise RuntimeError("queue dir unreadable")

        monkeypatch.setattr(kb, "QueueProcessor", _boom)

        assert kb.start_inbound_processor() is None  # must not raise
        assert kb.inbound_processor_running() is False
        assert _inbound_threads() == []

    def test_inbound_processor_running_reflects_state(self, monkeypatch):
        import harnessfix.kanban_bridge as kb
        monkeypatch.setenv("KANBAN_SYNC_ENABLED", "1")

        assert kb.inbound_processor_running() is False
        kb.start_inbound_processor(poll_interval=0.05)
        assert kb.inbound_processor_running() is True


# ---------------------------------------------------------------------------
# agent.py boot wiring
# ---------------------------------------------------------------------------

@pytest.fixture()
def agent_module():
    import agent
    return agent


class TestAgentBootWiring:
    def test_helper_delegates_to_bridge(self, agent_module, monkeypatch):
        import harnessfix.kanban_bridge as kb
        calls: list = []
        monkeypatch.setattr(kb, "start_inbound_processor", lambda: calls.append(1))

        agent_module._start_kanban_inbound()

        assert calls == [1], "_start_kanban_inbound must start the bridge poller"

    def test_helper_swallows_bridge_errors(self, agent_module, monkeypatch):
        import harnessfix.kanban_bridge as kb

        def _boom():
            raise RuntimeError("queue exploded")

        monkeypatch.setattr(kb, "start_inbound_processor", _boom)

        agent_module._start_kanban_inbound()  # must not raise

    def test_helper_survives_missing_bridge(self, agent_module, monkeypatch):
        """harnessfix is optional for some entry points -> log, don't crash."""
        monkeypatch.setitem(sys.modules, "harnessfix.kanban_bridge", None)

        agent_module._start_kanban_inbound()  # must not raise

    def test_main_starts_kanban_before_branches(self, agent_module, monkeypatch):
        """main() must boot the poller — that call site IS the fix."""
        order: list[str] = []
        monkeypatch.setattr(
            agent_module, "_start_kanban_inbound", lambda: order.append("kanban"),
        )

        async def _fake_interactive() -> None:
            order.append("interactive")

        monkeypatch.setattr(agent_module, "run_interactive", _fake_interactive)
        monkeypatch.setattr(sys, "argv", ["agent.py"])

        asyncio.run(agent_module.main())

        assert order == ["kanban", "interactive"], (
            f"poller must start before the mode branches, got {order}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestManualCli:
    def _run_main(self, monkeypatch, argv: list[str]) -> int:
        import harnessfix.kanban_bridge as kb
        monkeypatch.setattr(sys, "argv", ["kanban_bridge", *argv])
        return kb.main()

    def test_process_in_flag(self, monkeypatch):
        import harnessfix.kanban_bridge as kb
        processed: list = []
        monkeypatch.setattr(
            kb, "process_inbound", lambda qdir: processed.append(qdir) or 0,
        )

        assert self._run_main(monkeypatch, ["--process-in"]) == 0
        assert processed, "--process-in must call process_inbound"

    def test_process_alias_accepted(self, monkeypatch):
        """The docstring advertised ``--process`` for years; keep it working."""
        import harnessfix.kanban_bridge as kb
        processed: list = []
        monkeypatch.setattr(
            kb, "process_inbound", lambda qdir: processed.append(qdir) or 0,
        )

        assert self._run_main(monkeypatch, ["--process"]) == 0
        assert processed, "--process alias must still process inbound"

    def test_queue_dir_is_honoured(self, monkeypatch, tmp_path):
        import harnessfix.kanban_bridge as kb
        seen: list[Path] = []
        monkeypatch.setattr(
            kb, "process_inbound", lambda qdir: seen.append(qdir) or 0,
        )

        target = tmp_path / "custom-queue"
        rc = self._run_main(
            monkeypatch, ["--process-in", "--queue-dir", str(target)],
        )
        assert rc == 0
        assert seen == [target]

    def test_no_flag_prints_help_and_fails(self, monkeypatch, capsys):
        assert self._run_main(monkeypatch, []) == 1
        assert "usage" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# The docstring's advertised boot path must be real
# ---------------------------------------------------------------------------

class TestDocumentedBootPath:
    def test_module_docstring_does_not_advertise_issue_loop(self):
        """It used to say ``python harnessfix/issue_loop.py`` — which has no
        ``main()`` and no argparse, so the documented boot did nothing."""
        import harnessfix.kanban_bridge as kb
        doc = kb.__doc__ or ""

        assert "issue_loop.py" not in doc
        assert "agent.py" in doc

    def test_issue_loop_has_no_main_entry_point(self):
        """Pins *why* the old docstring was wrong, so it can't be re-added."""
        import harnessfix.issue_loop as il

        assert not hasattr(il, "main"), (
            "if issue_loop gains a main(), the boot docstring may cite it again"
        )
