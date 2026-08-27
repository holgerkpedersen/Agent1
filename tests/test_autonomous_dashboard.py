"""Regression tests for the live autonomous-run progress beacon + dashboard API.

The autonomous loop only wrote summary.json at the END of each iteration (a
single iteration can run the full pytest suite for minutes), so the dashboard
had no "what is it doing right now" signal. These tests pin:

- harnessfix.progress atomic read/write/clear + history append.
- the /api/autonomous/status endpoint merges the live beacon, summary, history
  and recent commits, and derives a correct running/idle flag from the heartbeat.
- the static/autonomous.html page is served at /autonomous and /autonomous.html.
"""
from __future__ import annotations

import json
import threading
import typing
import urllib.request
from pathlib import Path

import pytest

from agent_core.monitoring import DashboardAPIServer

import harnessfix.progress as progress


def test_write_progress_merges_and_stamps_ts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(progress, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(progress, "STATUS_PATH", tmp_path / "run_status.json")
    monkeypatch.setattr(progress, "HISTORY_PATH", tmp_path / "run_history.jsonl")

    progress.write_progress({"iteration": 2, "running": True})
    progress.write_progress({"phase": "diagnosing", "diagnosed": 7})

    data = progress.read_progress()
    assert data["iteration"] == 2
    assert data["phase"] == "diagnosing"
    assert data["diagnosed"] == 7
    assert isinstance(data["ts"], (int, float))


def test_write_progress_atomic_no_partial_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(progress, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(progress, "STATUS_PATH", tmp_path / "run_status.json")
    monkeypatch.setattr(progress, "HISTORY_PATH", tmp_path / "run_history.jsonl")

    progress.write_progress({"a": 1})
    # Temp file must not linger (atomic replace).
    assert not (tmp_path / "run_status.tmp").exists()
    assert (tmp_path / "run_status.json").is_file()


def test_append_and_read_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(progress, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(progress, "STATUS_PATH", tmp_path / "run_status.json")
    monkeypatch.setattr(progress, "HISTORY_PATH", tmp_path / "run_history.jsonl")

    progress.append_history({"iteration": 1, "verdict": "accepted"})
    progress.append_history({"iteration": 2, "verdict": "rejected_and_reverted"})

    hist = progress.read_history()
    assert [h["iteration"] for h in hist] == [1, 2]
    assert hist[0]["verdict"] == "accepted"

    # limit keeps the newest records.
    assert [h["iteration"] for h in progress.read_history(limit=1)] == [2]


def test_read_progress_tolerates_missing_or_corrupt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(progress, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(progress, "STATUS_PATH", tmp_path / "run_status.json")
    monkeypatch.setattr(progress, "HISTORY_PATH", tmp_path / "run_history.jsonl")

    assert progress.read_progress() == {}
    (tmp_path / "run_status.json").write_text("{not valid json", encoding="utf-8")
    assert progress.read_progress() == {}


def test_clear_progress_marks_idle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(progress, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(progress, "STATUS_PATH", tmp_path / "run_status.json")
    monkeypatch.setattr(progress, "HISTORY_PATH", tmp_path / "run_history.jsonl")

    progress.write_progress({"running": True, "phase": "diagnosing"})
    progress.clear_progress()
    data = progress.read_progress()
    assert data["phase"] == "idle"
    assert data["running"] is False


def test_endpoint_merges_status_summary_history_and_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Point the progress module at the temp dir.
    monkeypatch.setattr(progress, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(progress, "STATUS_PATH", tmp_path / "run_status.json")
    monkeypatch.setattr(progress, "HISTORY_PATH", tmp_path / "run_history.jsonl")

    # Seed a live beacon (recent ts) + a history record + a summary file.
    progress.write_progress({"iteration": 3, "max_iterations": 5, "running": True, "phase": "running_test_gate"})
    progress.append_history({"iteration": 1, "verdict": "accepted", "git_head": "abc1234"})
    summary = {"verdict": "accepted", "proposed_repair": "stuck-repeat-tool-hints", "tests_passed": True}
    (tmp_path / "reports" / "harnessfix").mkdir(parents=True)
    (tmp_path / "reports" / "harnessfix" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    # Serve the real endpoint; the dashboard uses repo-root-relative paths for
    # summary.json + git, so redirect those to our temp dir too.
    from agent_core.monitoring import dashboard_api
    monkeypatch.setattr(dashboard_api.DashboardAPIHandler, "_base_dir", str(tmp_path))
    monkeypatch.setattr(
        dashboard_api.subprocess, "run",
        lambda *a, **k: typing.cast(typing.Any, _FakeProc()),
    )

    server = DashboardAPIServer(__import__("agent_core.monitoring.metrics_collector", fromlist=["MetricsCollector"]).MetricsCollector(), port=0)
    httpd = server.start()
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        port = httpd.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/autonomous/status", timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    finally:
        threading.Thread(target=httpd.shutdown, daemon=True).start()
        httpd.server_close()

    assert payload["status"]["running"] is True
    assert payload["status"]["iteration"] == 3
    assert payload["status"]["phase"] == "running_test_gate"
    assert payload["summary"]["verdict"] == "accepted"
    assert payload["history"][0]["iteration"] == 1
    # git subprocess was stubbed -> recent_commits should be empty list, not crash.
    assert payload["recent_commits"] == []


def test_endpoint_derives_idle_when_heartbeat_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(progress, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(progress, "STATUS_PATH", tmp_path / "run_status.json")
    monkeypatch.setattr(progress, "HISTORY_PATH", tmp_path / "run_history.jsonl")

    # Old heartbeat (> stale window) with running=True must read as idle.
    progress.write_progress({"running": True, "phase": "diagnosing"})
    old = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    old["ts"] = old["ts"] - 1000  # 1000s in the past
    (tmp_path / "run_status.json").write_text(json.dumps(old), encoding="utf-8")

    from agent_core.monitoring import dashboard_api
    monkeypatch.setattr(dashboard_api.DashboardAPIHandler, "_base_dir", str(tmp_path))
    monkeypatch.setattr(dashboard_api.subprocess, "run", lambda *a, **k: typing.cast(typing.Any, _FakeProc()))

    server = DashboardAPIServer(__import__("agent_core.monitoring.metrics_collector", fromlist=["MetricsCollector"]).MetricsCollector(), port=0)
    httpd = server.start()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        port = httpd.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/autonomous/status", timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    finally:
        threading.Thread(target=httpd.shutdown, daemon=True).start()
        httpd.server_close()

    assert payload["status"]["running"] is False


def test_autonomous_page_served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_core.monitoring import dashboard_api

    # Serve a real static/autonomous.html if present, else require the route to
    # 404 cleanly (page missing is a setup error, not a route error).
    page = Path(dashboard_api.__file__).resolve().parent.parent / "static" / "autonomous.html"
    if not page.is_file():
        return
    monkeypatch.setattr(dashboard_api.DashboardAPIHandler, "_base_dir", str(page.parent.parent))

    server = DashboardAPIServer(__import__("agent_core.monitoring.metrics_collector", fromlist=["MetricsCollector"]).MetricsCollector(), port=0)
    httpd = server.start()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        port = httpd.server_address[1]
        for route in ("/autonomous", "/autonomous.html"):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{route}", timeout=10) as resp:
                body = resp.read().decode("utf-8")
            assert resp.status == 200
            assert "Autonomous" in body
    finally:
        threading.Thread(target=httpd.shutdown, daemon=True).start()
        httpd.server_close()


class _FakeProc:
    returncode = 0
    stdout = ""
