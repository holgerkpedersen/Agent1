"""Tests for plan_execution.chunking — bounded summaries & wave-based execution."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_core.plan_execution import chunking
from agent_core.plan_execution.parser import PlanTask
from agent_core.plan_execution.runner import (
    build_scheduler,
    run_plan,
)


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_env_defaults(monkeypatch):
    for var in ("AGENT_PLAN_MAX_SUMMARY_CHARS", "AGENT_PLAN_MAX_CONCURRENT"):
        monkeypatch.delenv(var, raising=False)
    assert chunking.max_summary_chars() == chunking.DEFAULT_MAX_SUMMARY_CHARS
    assert chunking.max_concurrent() == chunking.DEFAULT_MAX_CONCURRENT


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("AGENT_PLAN_MAX_SUMMARY_CHARS", "500")
    monkeypatch.setenv("AGENT_PLAN_MAX_CONCURRENT", "2")
    assert chunking.max_summary_chars() == 500
    assert chunking.max_concurrent() == 2


def test_env_invalid_values_fall_back(monkeypatch):
    monkeypatch.setenv("AGENT_PLAN_MAX_SUMMARY_CHARS", "not-a-number")
    monkeypatch.setenv("AGENT_PLAN_MAX_CONCURRENT", "-3")
    assert chunking.max_summary_chars() == chunking.DEFAULT_MAX_SUMMARY_CHARS
    assert chunking.max_concurrent() == chunking.DEFAULT_MAX_CONCURRENT


def test_summarize_result_truncates_long_text():
    result = {"status": "completed", "summary": "x" * 5000, "files": ["a.py"]}
    bounded = chunking.summarize_result(result, max_chars=100)
    assert len(bounded["summary"]) < 200
    assert bounded["summary"].startswith("xxx")
    assert bounded["summary_truncated"] is True
    # Non-summary keys survive verbatim.
    assert bounded["status"] == "completed"
    assert bounded["files"] == ["a.py"]


def test_summarize_result_leaves_short_text_alone():
    result = {"status": "completed", "summary": "short"}
    bounded = chunking.summarize_result(result, max_chars=100)
    assert bounded["summary"] == "short"
    assert "summary_truncated" not in bounded


def test_summarize_result_non_string_values_preserved():
    result = {"status": "completed", "summary": {"nested": 1}, "count": 3}
    bounded = chunking.summarize_result(result, max_chars=10)
    assert bounded["summary"] == {"nested": 1}
    assert bounded["count"] == 3


def test_chunk_boundaries():
    assert chunking.chunk_boundaries(0) == []
    assert chunking.chunk_boundaries(7, max_concurrent_tasks=3) == [
        (0, 3),
        (3, 6),
        (6, 7),
    ]


def test_plan_chunks_respects_size():
    order = ["t1", "t2", "t3", "t4", "t5"]
    chunks = chunking.plan_chunks(order, max_concurrent_tasks=2)
    assert chunks == [["t1", "t2"], ["t3", "t4"], ["t5"]]


def test_plan_chunks_uses_env_default(monkeypatch):
    monkeypatch.setenv("AGENT_PLAN_MAX_CONCURRENT", "2")
    chunks = chunking.plan_chunks(["a", "b", "c", "d"])
    assert [len(c) for c in chunks] == [2, 2]


# ---------------------------------------------------------------------------
# runner integration: bounded summaries via the real scheduler path
# ---------------------------------------------------------------------------


class _RecordingAgent:
    """Minimal agent stub; records the prompt each task runs with."""

    def __init__(self):
        self.seen: list[str] = []

    def spawn_subagent(self, name: str, workspace=None, role=None):
        agent = self

        class _Sub:
            async def respond(self, desc):
                agent.seen.append(str(desc))
                return "u" * 50_000  # verbose model answer

            def get_context_summary(self, max_messages: int = 10) -> str:
                return "s" * 50_000  # verbose summary

        return _Sub()


def _tasks():
    return [
        PlanTask(id="t1", description="first"),
        PlanTask(id="t2", description="second", depends_on=["t1"]),
        PlanTask(id="t3", description="third", depends_on=["t2"]),
    ]


def test_run_plan_bounds_downstream_summaries(monkeypatch, tmp_path):
    """A huge upstream summary reaches downstream prompts bounded."""
    monkeypatch.setenv("AGENT_PLAN_MAX_SUMMARY_CHARS", "100")
    agent = _RecordingAgent()

    snapshot = asyncio.run(run_plan(agent, _tasks(), plan_dir=tmp_path))

    assert [t["status"] for t in snapshot["tasks"]] == ["completed"] * 3
    downstream = [p for p in agent.seen if p.startswith("second")]
    assert downstream, "downstream tasks should see the upstream summary"
    for prompt in downstream:
        assert len(prompt) < 5_000, "prompt must not carry the raw 50k summary"
        assert chunking._TRUNCATION_SUFFIX.strip() in prompt


def test_scheduler_retains_bounded_results(monkeypatch, tmp_path):
    """build_scheduler keeps only bounded summaries resident per task."""
    monkeypatch.setenv("AGENT_PLAN_MAX_SUMMARY_CHARS", "100")
    agent = _RecordingAgent()
    tasks = _tasks()
    scheduler = build_scheduler(tasks, agent)

    async def run():
        for t in tasks:
            scheduler.schedule_task(t.id)
        await scheduler.start()
        while (
            scheduler.pending_count() > 0
            or scheduler.scheduled_count() > 0
            or scheduler._inflight
        ):
            await asyncio.sleep(0.01)
        await scheduler.stop()

    asyncio.run(run())
    for t in tasks:
        result = scheduler.get_result(t.id)
        assert result is not None and result["success"] is True
        # Bounded summary, not the raw 50k-char context dump.
        assert len(result["summary"]) <= 100 + len(chunking._TRUNCATION_SUFFIX)


def test_scheduler_bounded_results_are_persisted(tmp_path):
    """dep_<id>_result.json files stay small because results are summaries."""
    agent = _RecordingAgent()

    async def run():
        return await run_plan(agent, _tasks(), plan_dir=tmp_path)

    # Default cap: 2000 chars per summary.
    asyncio.run(run())
    for tid in ("t1", "t2", "t3"):
        body = (tmp_path / f"dep_{tid}_result.json").read_text(encoding="utf-8")
        assert len(body) < 5_000, f"{tid} persisted result too large: {len(body)}"
