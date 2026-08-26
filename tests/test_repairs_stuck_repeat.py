"""Improvement-plan item #10: grow the repair catalog.

Two pieces are covered here:

1. Diagnosis false-positive fix (prerequisite): every "context layer"
   diagnosis on the 2026-08-25 corpus was bogus - the signature matched
   ``llm_response.text``, i.e. the model QUOTING the tracer's storage marker
   "...[truncated N chars]" in its own chat output.  Context pressure is a
   SYSTEM event and must be matched on system fields only.
2. The second catalog repair, ``stuck-repeat-tool-hints``: trace evidence
   shows stuck loops reach three identical calls before anything concrete is
   offered; the repair gives per-tool alternatives at strike TWO, while the
   model still has budget.  Runtime behaviour is verified in a FRESH
   interpreter (subprocess) so the repaired source - not a stale import -
   is what executes.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from harnessfix import gates
from harnessfix.diagnose import diagnose_graph
from harnessfix.htir import HTIRStep, TraceGraph
from harnessfix.loop import run_loop
from harnessfix.repairs import CATALOG, LIFECYCLE_LAYER, repairs_for_layer
from harnessfix.repairs.collisions import find_test_collisions
from harnessfix.repairs.stuck_repeat import (
    STUCK_REPEAT_REPAIR_ID,
    RepairApplyError,
    apply as apply_stuck,
    revert as revert_stuck,
)
from harnessfix.tracing import (
    KIND_GUARD_TRIGGERED,
    KIND_LLM_RESPONSE,
    KIND_LOOP_END,
    KIND_TOOL_CALL,
    KIND_TOOL_ERROR,
    LAYER_CONTEXT,
    LAYER_LIFECYCLE,
    LAYER_TOOL_INTERFACE,
    TraceWriter,
)

# ---------------------------------------------------------------------------
# Fixtures/helpers
# ---------------------------------------------------------------------------


def _graph(steps: list[tuple[str, str, dict]]) -> TraceGraph:
    return TraceGraph(
        task_id="t",
        steps=[
            HTIRStep(index=i, kind=kind, layer_facet=layer, payload=payload)
            for i, (kind, layer, payload) in enumerate(steps)
        ],
    )


def _loop_end(outcome: str = "completed") -> tuple[str, str, dict]:
    return KIND_LOOP_END, LAYER_LIFECYCLE, {"outcome": outcome, "termination_reason": outcome}


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
        # The loop prints display summaries (containing em dashes) to stdout;
        # force UTF-8 so the parent's strict decoder never sees cp1252 bytes.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        scripts = {
            "recovers": (
                [["read", {"path": "f.py"}],
                 ["read", {"path": "f.py"}],
                 ["list_files", {"path": "."}],
                 "Recovered using the listing."],
            ),
            "third_strike": (
                [["search", {"query": "q", "path": "."}],
                 ["search", {"query": "q", "path": "."}],
                 ["search", {"query": "q", "path": "."}],
                 "Final answer despite being stuck."],
            ),
            "default_hint": (
                [["web_search", {"query": "x"}],
                 ["web_search", {"query": "x"}],
                 "Answering from memory."],
            ),
        }
        script = scripts[scenario][0]
        fake = ScriptedLLM(script)
        executed = []

        async def execute_tool(name, args):
            executed.append(name)
            return name + "-result"

        runner = ToolLoopRunner(max_iterations=10)
        final_text, messages = await runner.run(
            messages=[{"role": "user", "content": "task"}],
            llm_chat_fn=make_chat_fn(fake),
            execute_tool_fn=execute_tool,
            tools=None,
        )
        # ASCII-only JSON: child-process stdout must survive any console
        # codepage (a cp1252 em dash crashed the parent's UTF-8 reader).
        print(json.dumps(
            {
                "final_text": final_text,
                "executed": executed,
                "llm_calls": len(fake.calls),
                "tool_msgs": [
                    str(m["content"]).encode("ascii", "replace").decode("ascii")
                    for m in messages if m["role"] == "tool"
                ],
            },
            ensure_ascii=True,
        ))


    scenario = sys.argv[1]
    asyncio.run(main(scenario))
    """
)


def _run_scenario(scenario: str) -> dict:
    """Execute a scripted tool-loop scenario in a FRESH interpreter."""
    r = subprocess.run(
        [sys.executable, "-c", _DRIVER, scenario],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.fixture()
def applied_stuck_repair():
    """Apply against the real tree, restore byte-identical state after."""
    summary = apply_stuck()
    assert "_REPEAT_HINTS" in summary
    try:
        yield
    finally:
        revert_stuck()


# ---------------------------------------------------------------------------
# 1. Context-layer diagnosis: system fields only, never model output
# ---------------------------------------------------------------------------


def test_model_quoting_tracer_marker_is_not_context_pressure():
    """Regression (2026-08-25 corpus, 5/5 bogus context diagnoses): the model
    echoing "...[truncated N chars]" in its own answer must not be diagnosed
    as history truncation."""
    g = _graph(
        [
            (
                KIND_LLM_RESPONSE,
                LAYER_CONTEXT,
                {"text": "Here is the section...[truncated 1423 chars]... end."},
            ),
            (KIND_TOOL_ERROR, LAYER_TOOL_INTERFACE, {"exception": "OSError", "message": "disk"}),
            _loop_end("cap"),
        ]
    )
    d = diagnose_graph(g)
    assert d.root_layer == "execution_environment"


def test_system_note_with_truncation_marker_still_maps_to_context():
    """A SYSTEM steering note carrying truncation wording IS context pressure:
    the signature now matches guard notes instead of free text."""
    g = _graph(
        [
            (
                KIND_GUARD_TRIGGERED,
                LAYER_LIFECYCLE,
                {"guard": "budget_exhausted", "note": "context truncated to fit budget"},
            ),
            (KIND_TOOL_ERROR, LAYER_TOOL_INTERFACE, {"exception": "OSError", "message": "disk"}),
            _loop_end("cap"),
        ]
    )
    d = diagnose_graph(g)
    assert d.root_layer == LAYER_CONTEXT
    assert d.mechanism == "history truncation / token limit pressure"


def test_llm_response_free_text_never_matches_any_signature():
    """Free-text model output is excluded for ALL signatures, not just the
    truncation one (same root cause: marker words inside quoted content)."""
    g = _graph(
        [
            (
                KIND_LLM_RESPONSE,
                LAYER_TOOL_INTERFACE,
                {"text": "the command 'rm' is not allowed by policy, so I stopped"},
            ),
            _loop_end("completed"),
        ]
    )
    d = diagnose_graph(g)
    assert d.root_layer == LAYER_LIFECYCLE
    assert d.mechanism.startswith("loop did not complete")


# ---------------------------------------------------------------------------
# 2. The stuck-repeat-tool-hints repair: unit level
# ---------------------------------------------------------------------------


def test_repair_is_catalogued_on_the_lifecycle_layer():
    repair = CATALOG[STUCK_REPEAT_REPAIR_ID]
    assert repair.layer == LIFECYCLE_LAYER
    assert repair in repairs_for_layer(LIFECYCLE_LAYER)


def test_apply_and_roundtrip(tmp_path, monkeypatch):
    """Apply rewrites the second-strike suffix + inserts the hint table;
    revert restores the original bytes exactly."""
    import harnessfix.repairs.stuck_repeat as mod

    target = Path("agent_core/llm/tool_loop.py")
    original = target.read_text(encoding="utf-8")
    local = tmp_path / "agent_core" / "llm" / "tool_loop.py"
    local.parent.mkdir(parents=True)
    local.write_text(original, encoding="utf-8")
    monkeypatch.setattr(mod, "_TARGET", local)

    summary = mod.apply()
    applied = local.read_text(encoding="utf-8")
    assert "_REPEAT_HINTS.get(tool_name, _REPEAT_HINT_DEFAULT)" in applied
    assert "Take a different action or " + "answer in text." not in applied
    assert "_REPEAT_HINTS" in summary

    mod.revert()
    assert local.read_text(encoding="utf-8") == original


def test_apply_twice_is_noop_and_revert_without_apply_is_clean(tmp_path, monkeypatch):
    import harnessfix.repairs.stuck_repeat as mod

    original = Path("agent_core/llm/tool_loop.py").read_text(encoding="utf-8")
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
    import harnessfix.repairs.stuck_repeat as mod

    broken = tmp_path / "broken.py"
    broken.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(mod, "_TARGET", broken)
    with pytest.raises(RepairApplyError, match="anchor not found"):
        mod.apply()


def test_collision_fragment_has_no_pins_in_test_suite():
    """Guard surface check: no REAL test may pin the OLD second-strike
    suffix (the pinned PREFIX stays intact by construction).  The fragment
    is assembled here so this file itself never contains it as a literal -
    a literal would make the repair self-block, which is exactly what the
    guard is for."""
    fragment = "Take a different action or " + "answer in text."
    assert find_test_collisions((fragment,)) == []


# ---------------------------------------------------------------------------
# 3. Runtime behaviour through the REAL ToolLoopRunner (repair applied,
#    executed in a fresh interpreter so the repaired source is what runs)
# ---------------------------------------------------------------------------


def test_second_strike_carries_concrete_alternatives(applied_stuck_repair):
    """Strike two must now name alternatives - BEFORE the fatal third repeat."""
    out = _run_scenario("recovers")

    assert out["final_text"] == "Recovered using the listing."
    assert out["executed"].count("read") == 1  # repeats are never re-executed
    second_strike = next(
        t for t in out["tool_msgs"] if t.startswith("NOTE: This exact call")
    )
    # pinned prefix preserved ...
    assert "NOTE: This exact call has now been executed" in second_strike
    # ... AND concrete alternatives appended (the read-specific hint).
    assert "definitions()" in second_strike
    assert "Alternatives:" in second_strike
    # Prevention worked: the model took a different action instead of a
    # third identical strike.
    assert "list_files" in out["executed"]


def test_third_strike_still_stops_the_loop(applied_stuck_repair):
    """Prevention must not weaken the existing stop guarantee."""
    out = _run_scenario("third_strike")

    assert out["final_text"] == "Final answer despite being stuck."
    assert out["llm_calls"] == 4  # 3 iterations + forced synthesis; cap=10 unused
    assert any("three times" in t for t in out["tool_msgs"])


def test_unmapped_tools_get_the_default_hint(applied_stuck_repair):
    out = _run_scenario("default_hint")

    second_strike = next(
        t for t in out["tool_msgs"] if t.startswith("NOTE: This exact call")
    )
    assert "give your final answer now." in second_strike


# ---------------------------------------------------------------------------
# 4. End-to-end through the closed loop (diagnosis -> proposal -> gates)
# ---------------------------------------------------------------------------


def _write_stuck_trace(traces_dir: Path, task_id: str) -> None:
    writer = TraceWriter(task_id=task_id, directory=traces_dir)
    for _ in range(3):
        writer.emit({"kind": KIND_TOOL_CALL, "layer": "tool_interface",
                     "tool": "read", "args_hash": "a"})
    writer.emit({"kind": KIND_GUARD_TRIGGERED, "layer": "lifecycle",
                 "guard": "stuck", "iteration": 3, "note": "repeated"})
    writer.emit({"kind": KIND_LOOP_END, "layer": "lifecycle",
                 "outcome": "stuck", "termination_reason": "stuck"})
    writer.close()


def test_loop_now_proposes_a_lifecycle_repair_for_stuck_traces(tmp_path, monkeypatch):
    """Before this change a stuck-only corpus proposed NOTHING (no lifecycle
    repair existed); now choose_repair finds stuck-repeat-tool-hints and the
    full gate path runs with it."""
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    _write_stuck_trace(traces_dir, "stuck1")

    monkeypatch.setattr(gates, "get_baseline_failures", lambda *a, **k: frozenset())
    monkeypatch.setattr(gates, "run_test_gate", lambda *a, **k: (True, "passed"))
    monkeypatch.setattr(gates, "run_security_gate", lambda: (True, "ok"))
    monkeypatch.setattr(gates, "run_benchmark_gate", lambda model, profile=None: None)
    (tmp_path / "no_tests").mkdir()
    monkeypatch.setattr(
        "harnessfix.repairs.collisions.DEFAULT_TESTS_DIR", tmp_path / "no_tests"
    )

    out = tmp_path / "out"
    try:
        summary = run_loop(traces_dir, approve=True, model=None, output_dir=out)
        assert summary["proposed_repair"] == STUCK_REPEAT_REPAIR_ID
        assert summary["accepted"] is True
        assert summary["verdict"] == "accepted"
    finally:
        revert_stuck()  # keep the working tree byte-identical
