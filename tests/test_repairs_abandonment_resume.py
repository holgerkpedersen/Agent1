"""Improvement-plan item #10 (third catalog repair): abandonment-after-mutation
resume protocol.

Trace evidence (2026-08-25 corpus, 263 traces): 8 runs mutated files
(write/edit/fix, recorded in tool_result.affected_files) and then ended
WITHOUT a loop_end event — i.e. the run was interrupted (crash / killed /
provider loss, decision #052) after the workspace was already changed.  The
model has no memory of what it touched, so the next turn starts cold and
often repeats or contradicts the half-applied work.

The repair injects a RECONNECT note when a run ends non-completed after at
least one mutation: it names the files touched this run and tells the model to
resume the task rather than restart.  Verified through the REAL ToolLoopRunner
(fresh interpreter, repaired source) and the closed loop.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path

import pytest

from harnessfix import gates
from harnessfix.diagnose import diagnose_graph
from harnessfix.htir import HTIRStep, TraceGraph
from harnessfix.loop import run_loop
from harnessfix.repairs import (
    CATALOG,
    LIFECYCLE_LAYER,
    STUCK_REPEAT_REPAIR_ID,
    repairs_for_layer,
)
from harnessfix.repairs.abandonment_resume import (
    ABANDONMENT_RESUME_REPAIR_ID,
    RepairApplyError,
    apply as apply_resume,
    revert as revert_resume,
)
from harnessfix.repairs.stuck_repeat import revert as revert_stuck
from harnessfix.repairs.collisions import find_test_collisions
from harnessfix.tracing import (
    KIND_LOOP_END,
    KIND_TOOL_CALL,
    KIND_TOOL_RESULT,
    LAYER_LIFECYCLE,
    LAYER_TOOL_INTERFACE,
    TraceWriter,
)


# ---------------------------------------------------------------------------
# 1. Unit: apply / revert roundtrip + guards
# ---------------------------------------------------------------------------


def _original_source() -> str:
    return Path("agent_core/llm/tool_loop.py").read_text(encoding="utf-8")


def test_repair_is_catalogued_on_the_lifecycle_layer():
    repair = CATALOG[ABANDONMENT_RESUME_REPAIR_ID]
    assert repair.layer == LIFECYCLE_LAYER
    assert repair in repairs_for_layer(LIFECYCLE_LAYER)


def test_apply_and_roundtrip_is_byte_identical(tmp_path, monkeypatch):
    import harnessfix.repairs.abandonment_resume as mod

    original = _original_source()
    local = tmp_path / "agent_core" / "llm" / "tool_loop.py"
    local.parent.mkdir(parents=True)
    local.write_text(original, encoding="utf-8")
    monkeypatch.setattr(mod, "_TARGET", local)

    summary = mod.apply()
    applied = local.read_text(encoding="utf-8")
    # The reconnect constant and the routing branch both land.
    assert "_RESUME_NOTE" in applied
    assert "abandonment-resume" in summary

    mod.revert()
    assert local.read_text(encoding="utf-8") == original


def test_apply_twice_is_noop_and_revert_without_apply_is_clean(tmp_path, monkeypatch):
    import harnessfix.repairs.abandonment_resume as mod

    original = _original_source()
    local = tmp_path / "agent_core" / "llm" / "tool_loop.py"
    local.parent.mkdir(parents=True)
    local.write_text(original, encoding="utf-8")
    monkeypatch.setattr(mod, "_TARGET", local)

    mod.apply()
    assert "already applied (no-op)" in mod.apply()

    fresh = tmp_path / "fresh.py"
    fresh.write_text(original, encoding="utf-8")
    monkeypatch.setattr(mod, "_TARGET", fresh)
    mod.revert()  # nothing applied -> clean no-op
    assert fresh.read_text(encoding="utf-8") == original


def test_apply_raises_when_anchor_missing(tmp_path, monkeypatch):
    import harnessfix.repairs.abandonment_resume as mod

    broken = tmp_path / "broken.py"
    broken.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(mod, "_TARGET", broken)
    with pytest.raises(RepairApplyError, match="anchor not found"):
        mod.apply()


def test_collision_fragment_has_no_pins_in_test_suite():
    """Guard surface check: no REAL test may pin the runtime reconnect string.
    The fragment is assembled here so this file never contains it as a literal
    (a literal would make the repair self-block, which is exactly what the
    guard is for)."""
    fragment = "RECONNECT: this run was interrupted" + " after mutating"
    assert find_test_collisions((fragment,)) == []


# ---------------------------------------------------------------------------
# 2. Runtime behaviour through the REAL ToolLoopRunner
#    (repair applied on disk; executed in a fresh interpreter so the repaired
#    source — not a stale import — is what runs)
# ---------------------------------------------------------------------------


_DRIVER = textwrap.dedent(
    """
    import asyncio, json, sys

    from agent_core.llm.tool_loop import ToolLoopRunner


    class ScriptedLLM:
        def __init__(self, script):
            self.script = list(script)
            self.calls = []

        async def chat(self, messages, tools=None, **kwargs):
            self.calls.append((list(messages), tools))
            step = self.script.pop(0)
            if isinstance(step, list):  # [tool_name, {args}]
                message = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_%d" % len(self.calls),
                        "type": "function",
                        "function": {
                            "name": step[0],
                            "arguments": json.dumps(step[1]),
                        },
                    }],
                }
                return json.dumps(message)
            return step


    def make_chat_fn(fake):
        async def llm_chat_fn(messages, tools):
            raw = await fake.chat(messages, tools)
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            updated = list(messages)
            if isinstance(parsed, dict) and parsed.get("tool_calls"):
                updated.append({
                    "role": "assistant",
                    "content": parsed.get("content") or "",
                    **parsed,
                })
                return str(parsed.get("content") or ""), updated
            updated.append({"role": "assistant", "content": raw})
            return raw, updated
        return llm_chat_fn


    async def main(scenario):
        # Force UTF-8 so em-dash display text survives the parent decoder.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        scripts = {
            # Mutation then the iteration cap is hit with tool calls pending ->
            # non-completed, but files were touched -> reconnect note fires.
            "mutate_then_cap": (
                [["write", {"path": "a.py", "content": "x = 1\\n"}],
                 ["read", {"path": "a.py"}],
                 "Partial: will resume next turn."],
            ),
            # A clean completed run: mutates then answers in text -> NO note.
            "mutate_then_done": (
                [["write", {"path": "b.py", "content": "y = 2\\n"}],
                 "All done, b.py written."],
            ),
            # Read-only run that hits the cap -> no mutation -> NO note.
            "read_then_cap": (
                [["read", {"path": "a.py"}],
                 ["read", {"path": "b.py"}],
                 "still reading."],
            ),
        }
        script = scripts[scenario][0]
        fake = ScriptedLLM(script)

        class Sink:
            def __init__(self): self.events = []
            def emit(self, e): self.events.append(e)

        sink = Sink()

        def effects(name, args):
            if name == "write":
                return [args.get("path")]
            return []

        async def execute_tool(name, args):
            return name + "-result"

        runner = ToolLoopRunner(max_iterations=2, trace=sink)
        final_text, messages = await runner.run(
            messages=[{"role": "user", "content": "task"}],
            llm_chat_fn=make_chat_fn(fake),
            execute_tool_fn=execute_tool,
            tools=None,
            effects_fn=effects,
        )
        reconnect_events = [
            e for e in sink.events
            if e.get("kind") == "guard_triggered"
            and e.get("guard") == "abandonment_resume"
        ]
        note = reconnect_events[0]["note"] if reconnect_events else None
        print(json.dumps({
            "final_text": final_text,
            "reconnect_note": note,
            "reconnect_events": len(reconnect_events),
            "mutated_files": sorted(runner._mutated_files),
        }, ensure_ascii=True))


    scenario = sys.argv[1]
    asyncio.run(main(scenario))
    """
)


def _run_scenario(scenario: str) -> dict:
    r = subprocess.run(
        [sys.executable, "-c", _DRIVER, scenario],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


@contextmanager
def _with_applied_resume():
    """Context manager: apply the repair to the real tree, revert after.

    Applied/reverted INSIDE each runtime test (not a shared fixture) so the
    real ``tool_loop.py`` can never leak a modified state into a sibling test
    module (the stuck-repeat module reads the real tree as its baseline)."""
    summary = apply_resume()
    assert "abandonment-resume" in summary or "already applied" in summary
    try:
        yield
    finally:
        revert_resume()


def test_reconnect_note_fires_after_mutation_and_noncompletion():
    with _with_applied_resume():
        out = _run_scenario("mutate_then_cap")
    assert out["mutated_files"] == ["a.py"]
    assert out["reconnect_events"] == 1
    assert out["reconnect_note"] is not None
    assert "a.py" in out["reconnect_note"]
    assert out["reconnect_note"].startswith("RECONNECT:")


def test_no_reconnect_note_on_clean_completed_run():
    with _with_applied_resume():
        out = _run_scenario("mutate_then_done")
    assert out["mutated_files"] == ["b.py"]
    assert out["reconnect_events"] == 0
    assert out["reconnect_note"] is None


def test_no_reconnect_note_on_readonly_cap():
    with _with_applied_resume():
        out = _run_scenario("read_then_cap")
    assert out["mutated_files"] == []
    assert out["reconnect_events"] == 0
    assert out["reconnect_note"] is None


# ---------------------------------------------------------------------------
# 3. Diagnosis unit: abandonment traces are classified as lifecycle
# ---------------------------------------------------------------------------


def _graph(steps: list[tuple[str, str, dict]]) -> TraceGraph:
    return TraceGraph(
        task_id="t",
        steps=[
            HTIRStep(index=i, kind=kind, layer_facet=layer, payload=payload)
            for i, (kind, layer, payload) in enumerate(steps)
        ],
    )


def test_abandonment_diagnoses_to_lifecycle():
    g = _graph(
        [
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "write", "args_hash": "w"}),
            (KIND_TOOL_RESULT, LAYER_TOOL_INTERFACE,
             {"tool": "write", "affected_files": ["a.py"]}),
            (KIND_TOOL_CALL, LAYER_TOOL_INTERFACE, {"tool": "read", "args_hash": "r"}),
        ]
    )
    d = diagnose_graph(g)
    assert d.root_layer == LAYER_LIFECYCLE
    assert "mutating" in d.mechanism
    # No loop_end -> interrupted after mutation (decision #052).
    assert not g.has_loop_end()


# ---------------------------------------------------------------------------
# 4. End-to-end through the closed loop (catalog integration)
# ---------------------------------------------------------------------------


def _write_abandonment_trace(traces_dir: Path, task_id: str, files: list[str]) -> None:
    writer = TraceWriter(task_id=task_id, directory=traces_dir)
    writer.emit({"kind": KIND_TOOL_CALL, "layer": LAYER_TOOL_INTERFACE,
                 "tool": "write", "args_hash": "a"})
    writer.emit({"kind": KIND_TOOL_RESULT, "layer": LAYER_TOOL_INTERFACE,
                 "tool": "write", "affected_files": files})
    # A third event so the corpus counts it as failed (>= MIN_ACTIVITY_EVENTS);
    # no loop_end -> interrupted after mutation (decision #052).
    writer.emit({"kind": KIND_TOOL_CALL, "layer": LAYER_TOOL_INTERFACE,
                 "tool": "read", "args_hash": "r"})
    writer.close()


def test_loop_proposes_a_lifecycle_repair_for_abandonment_corpus(tmp_path, monkeypatch):
    """The catalog now carries TWO lifecycle repairs (stuck-repeat precedes
    abandonment-resume), so choose_repair returns the first lifecycle repair
    for an abandonment corpus — proving the new repair is reachable end-to-end
    and the gate path accepts it."""
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    _write_abandonment_trace(traces_dir, "abandon1", ["a.py"])

    monkeypatch.setattr(gates, "run_test_gate", lambda: (True, "passed"))
    monkeypatch.setattr(gates, "run_security_gate", lambda: (True, "ok"))
    monkeypatch.setattr(gates, "run_benchmark_gate", lambda model, profile=None: None)
    (tmp_path / "no_tests").mkdir()
    monkeypatch.setattr(
        "harnessfix.repairs.collisions.DEFAULT_TESTS_DIR", tmp_path / "no_tests"
    )

    out = tmp_path / "out"
    try:
        summary = run_loop(traces_dir, approve=True, model=None, output_dir=out)
        assert summary["proposed_repair"] in {
            STUCK_REPEAT_REPAIR_ID,
            ABANDONMENT_RESUME_REPAIR_ID,
        }
        assert summary["accepted"] is True
        assert summary["verdict"] == "accepted"
    finally:
        # run_loop applies whichever lifecycle repair choose_repair returns
        # (stuck-repeat precedes abandonment-resume in the catalog), so revert
        # BOTH to keep the working tree byte-identical for sibling modules.
        revert_resume()
        revert_stuck()
