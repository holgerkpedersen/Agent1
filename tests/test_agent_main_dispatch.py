"""Regression tests for the agent entrypoint dispatch (post theme-port merge).

Upstream commit df6d0c3 (@LebToki) rewrote ``agent.main`` so that:
* ``run_dashboard_server()`` — a *synchronous* function — was awaited
  ("object NoneType can't be used in 'await' expression"), and
* without ``--serve`` ``main()`` returned immediately, silently killing the
  interactive CLI.

These tests pin the corrected dispatch behavior.
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any

import agent


def _no_interactive(monkeypatch: Any) -> list[dict[str, Any]]:
    """Replace run_interactive/run_dashboard_server with recorders."""
    calls: list[dict[str, Any]] = []

    async def fake_interactive() -> None:
        calls.append({"fn": "interactive"})

    def fake_dashboard() -> None:
        calls.append({"fn": "dashboard"})

    monkeypatch.setattr(agent, "run_interactive", fake_interactive)
    monkeypatch.setattr(agent, "run_dashboard_server", fake_dashboard)
    return calls


def test_serve_flag_launches_dashboard(monkeypatch: Any) -> None:
    calls = _no_interactive(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["agent.py", "--serve"])
    asyncio.run(agent.main())
    assert [c["fn"] for c in calls] == ["dashboard"]


def test_no_serve_flag_runs_interactive_cli(monkeypatch: Any) -> None:
    """THE regression: previously main() did NOTHING without --serve."""
    calls = _no_interactive(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["agent.py"])
    asyncio.run(agent.main())
    assert [c["fn"] for c in calls] == ["interactive"]


def test_dashboard_server_is_not_awaitable() -> None:
    """Guards the original bug: awaiting run_dashboard_server() must fail type check.

    It is a plain sync function (ThreadingHTTPServer.serve_forever loop).
    """
    import inspect

    assert not inspect.iscoroutinefunction(agent.run_dashboard_server)
