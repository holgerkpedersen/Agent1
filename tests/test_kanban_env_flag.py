"""Regression: ``KANBAN_SYNC_ENABLED`` in ``.env`` must actually enable sync.

``.env.example`` documents the master switch as an ``.env`` setting:

    KANBAN_SYNC_ENABLED=0

...and the surrounding comments tell the operator to set it to ``1``.  But
nothing ever bridges ``.env`` into ``os.environ``: ``agent_core.config``'s
``_load_env_file()`` returns a *dict* consumed only inside
``load_agent_settings()``, and no module calls ``load_dotenv``/``putenv``/
``os.environ.update``.  Both sync gates therefore read ``os.environ`` alone:

  * ``harnessfix.issues._should_sync``        (outbound: issue_create/update/resolve)
  * ``kanban_bridge.start_inbound_processor`` (inbound: Kanban -> Agent1 replies)

So a user who edits ``.env`` exactly as documented gets **no sync at all** --
the documented master switch is silently dead unless the variable happens to be
exported in the shell.  Verified live before the fix: with
``KANBAN_SYNC_ENABLED=1`` appended to ``.env``, ``os.environ.get(...)`` was
``None`` and ``issues._should_sync()`` returned ``False``.

The fix resolves the flag the way the rest of this repo already resolves
``.env`` values (``agent._read_env_value``, ``agent_core/hue/bridge.py``,
``conftest._load_env``): **process env wins, else the repo ``.env``**.
"""
from __future__ import annotations

import threading

import pytest


@pytest.fixture(autouse=True)
def _isolated_sync_state(tmp_path, monkeypatch):
    """Reset the process-wide boot sentinel and redirect queue + .env paths.

    ``_BOOT_HANDLES`` is module state whose whole purpose is to persist across
    calls, so a leaked handle would make later ``start_inbound_processor()``
    calls no-ops and silently pass a broken implementation.  Redirecting the
    queue globals keeps a started poller off the real ``data/`` queue, and
    pointing ``ENV_FILE_PATH`` at a temp file keeps these tests independent of
    whatever the developer has in the real repo ``.env``.
    """
    import harnessfix.kanban_bridge as kb

    saved_handles = list(kb._BOOT_HANDLES)
    kb._BOOT_HANDLES.clear()

    data = tmp_path / "data"
    (data / "queue-kanban-to-agent1").mkdir(parents=True)
    (data / "queue-agent1-to-kanban").mkdir(parents=True)
    monkeypatch.setattr(kb, "DATA_DIR", data)
    monkeypatch.setattr(kb, "QUEUE_KANBAN_TO_AGENT1", data / "queue-kanban-to-agent1")
    monkeypatch.setattr(kb, "QUEUE_AGENT1_TO_KANBAN", data / "queue-agent1-to-kanban")

    # No .env by default: every test below opts in explicitly.
    monkeypatch.setattr(kb, "ENV_FILE_PATH", tmp_path / "absent.env")
    monkeypatch.delenv("KANBAN_SYNC_ENABLED", raising=False)

    try:
        yield
    finally:
        for proc in kb._BOOT_HANDLES:
            proc.stop()
        kb._BOOT_HANDLES[:] = saved_handles


# The documented "turn sync on" line from .env.example.
_ENV_ON = "KANBAN_SYNC_ENABLED=1\n"


def _write_env(tmp_path, text: str):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def _inbound_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "kanban-inbound"]


# ---------------------------------------------------------------------------
# The flag resolver
# ---------------------------------------------------------------------------

class TestSyncEnabledResolver:
    def test_dotenv_one_enables(self, tmp_path, monkeypatch):
        """The core regression: .env alone must be enough to enable sync."""
        import harnessfix.kanban_bridge as kb
        monkeypatch.setattr(kb, "ENV_FILE_PATH", _write_env(tmp_path, _ENV_ON))

        assert kb.sync_enabled() is True

    def test_process_env_wins_over_dotenv(self, tmp_path, monkeypatch):
        """An explicit export must override the file, not be shadowed by it."""
        import harnessfix.kanban_bridge as kb
        monkeypatch.setattr(kb, "ENV_FILE_PATH", _write_env(tmp_path, _ENV_ON))
        monkeypatch.setenv("KANBAN_SYNC_ENABLED", "0")

        assert kb.sync_enabled() is False

    def test_absent_key_is_disabled(self, tmp_path, monkeypatch):
        import harnessfix.kanban_bridge as kb
        monkeypatch.setattr(kb, "ENV_FILE_PATH", _write_env(tmp_path, "OTHER_KEY=1\n"))

        assert kb.sync_enabled() is False

    def test_missing_env_file_is_disabled(self, monkeypatch):
        import harnessfix.kanban_bridge as kb
        monkeypatch.setattr(
            kb, "ENV_FILE_PATH", __import__("pathlib").Path("does-not-exist.env")
        )

        assert kb.sync_enabled() is False

    def test_comments_and_quotes_are_tolerated(self, tmp_path, monkeypatch):
        """Match the repo's .env dialect: '#' comments, quotes, spaces."""
        import harnessfix.kanban_bridge as kb
        monkeypatch.setattr(
            kb, "ENV_FILE_PATH",
            _write_env(
                tmp_path,
                "# KANBAN_SYNC_ENABLED=0  (commented out, must be ignored)\n"
                'KANBAN_SYNC_ENABLED="1"\n',
            ),
        )

        assert kb.sync_enabled() is True


# ---------------------------------------------------------------------------
# The two real gates must honour the file
# ---------------------------------------------------------------------------

class TestGatesHonourDotenv:
    def test_inbound_boot_starts_from_dotenv(self, tmp_path, monkeypatch):
        """``agent.py:main()`` -> ``start_inbound_processor()`` must honour .env."""
        import harnessfix.kanban_bridge as kb
        env = _write_env(tmp_path, "KANBAN_SYNC_ENABLED=1\n")
        monkeypatch.setattr(kb, "ENV_FILE_PATH", env)

        proc = kb.start_inbound_processor(poll_interval=0.05)

        msg = ".env enabled sync but the inbound poller never started"
        assert proc is not None, msg
        assert kb.inbound_processor_running() is True
        assert len(_inbound_threads()) == 1

    def test_outbound_should_sync_from_dotenv(self, tmp_path, monkeypatch):
        """``issues._should_sync()`` must honour .env or issues never enqueue."""
        import harnessfix.kanban_bridge as kb
        from harnessfix import issues as issue_store
        env = _write_env(tmp_path, "KANBAN_SYNC_ENABLED=1\n")
        monkeypatch.setattr(kb, "ENV_FILE_PATH", env)

        # Layer 2 (per-board opt-in on the Kanban side) has its own dedicated
        # tests; pin it to True here so this test isolates the .env master
        # switch. The fixture's temp DATA_DIR has no board files at all, so an
        # un-stubbed gate would fail-closed and mask what is under test.
        monkeypatch.setattr(kb, "active_board_allows_agent1", lambda **_: True)

        assert issue_store._should_sync() is True

    def test_outbound_disabled_when_dotenv_says_zero(self, tmp_path, monkeypatch):
        import harnessfix.kanban_bridge as kb
        from harnessfix import issues as issue_store
        env = _write_env(tmp_path, "KANBAN_SYNC_ENABLED=0\n")
        monkeypatch.setattr(kb, "ENV_FILE_PATH", env)

        assert issue_store._should_sync() is False
