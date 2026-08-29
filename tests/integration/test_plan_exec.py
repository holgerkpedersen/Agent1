"""Integration tests for the ``plan exec`` command surface.

Exercises ``PlanCommand`` end-to-end with a mocked agent: plan-mode refusal,
build-mode execution that spawns one isolated subagent per task, and cycle
abort (no subagents spawned).
"""
from __future__ import annotations

import pytest

from agent_core.commands.plan_cmd import PlanCommand
from agent_core.orchestration.dependency_graph import CycleError
from agent_core.plan_execution import PlanTask, run_plan
from agent_core.subagent_roles import get_role


class _FakeSubagent:
    """Stand-in for a role-gated, isolated SubAgent."""

    def __init__(self, name: str, role: str) -> None:
        self.name = name
        self.role_name = role
        self.mode = "build"
        self._history: list[str] = []

    async def respond(self, task: str) -> str:
        self._history.append(task)
        return f"did {task[:20]}"

    def get_context_summary(self, max_messages: int = 3) -> str:
        return f"summary:{self.name}"


class _FakeAgent:
    """Minimal agent double for the command: tracks mode + spawned children."""

    def __init__(self, mode: str, workspace: str) -> None:
        self.mode = mode
        self.workspace = workspace
        self.spawned: list[_FakeSubagent] = []

    def is_plan_mode(self) -> bool:
        return self.mode == "plan"

    def spawn_subagent(self, name: str, workspace=None, role=None) -> _FakeSubagent:
        assert get_role(role) is not None, f"command must validate role {role!r}"
        sub = _FakeSubagent(name, role)
        self.spawned.append(sub)
        return sub


async def _run(args, agent) -> str:
    import io
    import contextlib

    cmd = PlanCommand()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        await cmd.execute(args, agent)
    return buf.getvalue()


@pytest.mark.anyio
async def test_plan_exec_refused_in_plan_mode(tmp_path):
    agent = _FakeAgent("plan", str(tmp_path))
    out = await _run(["exec", "--dry-run"], agent)
    assert "Refused" in out
    assert agent.spawned == []


@pytest.mark.anyio
async def test_plan_exec_runs_subagents_in_build_mode(tmp_path):
    agent = _FakeAgent("build", str(tmp_path))
    plan = (
        "## Tasks\n"
        "- [T0] explore (role: planner)\n"
        "- [T1] implement (role: implementer) deps: T0\n"
    )
    plan_file = tmp_path / "plan.md"
    plan_file.write_text(plan, encoding="utf-8")

    out = await _run(["exec", str(plan_file), "--yes"], agent)
    # One isolated subagent per task, role-gated (each keeps its own history).
    assert len(agent.spawned) == 2
    roles = {s.role_name for s in agent.spawned}
    assert roles == {"planner", "implementer"}
    assert "All tasks completed" in out


@pytest.mark.anyio
async def test_plan_exec_cycle_aborts_no_spawn(tmp_path):
    agent = _FakeAgent("build", str(tmp_path))
    plan = (
        "## Tasks\n"
        "- [A] x (role: implementer) deps: B\n"
        "- [B] y (role: implementer) deps: A\n"
    )
    plan_file = tmp_path / "plan.md"
    plan_file.write_text(plan, encoding="utf-8")

    out = await _run(["exec", str(plan_file), "--yes"], agent)
    assert "cycle" in out.lower()
    assert agent.spawned == []


@pytest.mark.anyio
async def test_plan_exec_unknown_role_aborts(tmp_path):
    agent = _FakeAgent("build", str(tmp_path))
    plan = "## Tasks\n- [T1] do it (role: nope)\n"
    plan_file = tmp_path / "plan.md"
    plan_file.write_text(plan, encoding="utf-8")

    out = await _run(["exec", str(plan_file), "--yes"], agent)
    assert "Unknown role" in out
    assert agent.spawned == []


# ── real SubAgent integration (no LLM network — monkeypatched client) ──
import agent as _agent_mod  # noqa: E402  (imported late to keep stubs above)


class _FakeLLMClient:
    """Stand-in for ``agent.LLMClient`` that returns canned text cheaply."""

    def __init__(self, *args, **kwargs):
        self.model_name = "fake"

    async def chat(self, messages, tools=None, **kwargs):
        # Echo the last user message so we can assert the task reached the
        # real SubAgent.respond() path.
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        return f"ack: {last_user[-40:]}"


@pytest.mark.anyio
async def test_run_plan_drives_real_subagent(tmp_path, monkeypatch):
    """Exercise run_plan's DEFAULT executor against a genuine SubAgent.

    The real ``SubAgent.respond`` (role persona, isolated history, turn cap)
    executes; only the network LLM is faked.  This proves the executor seam
    actually talks to an isolated, role-gated subagent end-to-end.
    """
    monkeypatch.setattr(_agent_mod, "LLMClient", _FakeLLMClient)

    # Minimal real Agent double: only what SubAgent.__init__ reads.
    class _RealAgent:
        mode = "build"
        model_name = "fake"
        workspace = str(tmp_path)

        def spawn_subagent(self, name, workspace=None, role=None):
            from agent_core.subagent import SubAgent
            return SubAgent(parent=self, name=name, workspace=workspace, role=role)

    tasks = [
        PlanTask(id="T0", description="explore module", role="planner"),
        PlanTask(id="T1", description="implement feature", role="implementer",
                 depends_on=["T0"]),
    ]
    snapshot = await run_plan(agent=_RealAgent(), tasks=tasks)

    assert all(t["status"] == "completed" for t in snapshot["tasks"])
    # The dependency result from T0 flowed into T1's input_data.
    t1 = next(t for t in snapshot["tasks"] if t["task_id"] == "T1")
    assert "T0" in t1["input_data"]
    assert "result" in t1["input_data"]["T0"]

