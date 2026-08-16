"""Tests for the run tool's timeout handling (process-tree kill on Windows)."""
import asyncio
import subprocess

from agent import Agent


class _FakeProc:
    pid = 4242

    def __init__(self, *a, **k):
        self.calls = 0

    def communicate(self, timeout=None):
        self.calls += 1
        if self.calls == 1:
            raise subprocess.TimeoutExpired("cmd", timeout)
        return ("", "")


class TestRunTimeout:
    def test_timeout_kills_process_tree(self, tmp_path, monkeypatch):
        agent = Agent(workspace=str(tmp_path))
        captured = {}

        def fake_popen(*a, **k):
            captured["kwargs"] = k
            return _FakeProc()

        def fake_taskkill(args, **k):
            captured["taskkill"] = args
            return type("R", (), {"returncode": 0})()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        monkeypatch.setattr(subprocess, "run", fake_taskkill)

        out = asyncio.run(
            agent._execute_tool_call("run", {"command": "python -m x", "timeout": 1})
        )
        assert "timed out after 1s" in out
        assert "tree" in out
        assert captured["taskkill"][0] == "taskkill"
        assert captured["taskkill"][3] == "/T"  # tree kill
        assert "4242" in captured["taskkill"]
        # The shell runs in its own process group on Windows.
        if os_name() == "nt":
            assert captured["kwargs"]["creationflags"] == subprocess.CREATE_NEW_PROCESS_GROUP

    def test_kill_process_tree_never_raises(self, monkeypatch):
        from agent import _kill_process_tree

        def boom(args, **k):
            raise OSError("taskkill failed")

        monkeypatch.setattr(subprocess, "run", boom)
        _kill_process_tree(_FakeProc())  # must not raise


def os_name() -> str:
    import os

    return os.name
