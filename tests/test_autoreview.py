"""Agent auto-review engine: golden patterns + trust model.

The golden set encodes the evidence methodology validated against the 12
hand-labeled pre-#050 traces in docs/PRE050_TRACE_LABELS.md (4 bug / 8 noise).
Each synthetic trace reproduces one corpus pattern; the engine must classify
it exactly as the manual review did.
"""

from pathlib import Path

import pytest

from harnessfix.autoreview import AGENT_FORBIDDEN, auto_review
from harnessfix.htir import compile_trace
from harnessfix.review import ReviewRecord, label_review
from harnessfix.tracing import TraceWriter


# ── golden trace builders ─────────────────────────────────────────────────

def _base(tmp_path: Path, task_id: str) -> TraceWriter:
    w = TraceWriter(task_id=task_id, directory=tmp_path)
    w.emit({"kind": "task_begin", "layer": "context", "user_input": "do it",
            "model": "m", "profile": "p"})
    return w


def _close(w: TraceWriter, outcome: str = "", reason: str = "") -> None:
    if outcome:
        w.emit({"kind": "loop_end", "layer": "lifecycle", "outcome": outcome,
                "termination_reason": reason})
    w.close()


def _read(w: TraceWriter, iteration: int) -> None:
    w.emit({"kind": "llm_response", "layer": "observability", "iteration": iteration,
            "text": "", "tool_calls_requested": 1})
    w.emit({"kind": "tool_call", "layer": "tool_interface", "iteration": iteration,
            "tool": "read", "args_hash": f"r{iteration}"})
    w.emit({"kind": "tool_result", "layer": "tool_interface", "iteration": iteration,
            "tool": "read", "args_hash": f"r{iteration}", "output": "..."})


def trace_fixture_delivered_after_tool_error(tmp_path):
    """demo-http500: tool error, then the loop ended completed (answer path)."""
    w = _base(tmp_path, "t1")
    w.emit({"kind": "tool_call", "layer": "tool_interface", "iteration": 0,
            "tool": "run", "args_hash": "x"})
    w.emit({"kind": "tool_error", "layer": "tool_interface", "iteration": 0,
            "exception": "ValueError", "message": "boom"})
    w.emit({"kind": "llm_response", "layer": "observability", "iteration": 1,
            "text": "the tool failed twice", "tool_calls_requested": 0})
    _close(w, "completed", "answer")
    return w.path


def trace_stuck_no_answer(tmp_path):
    """stuck guard on 3x identical call, no answer — genuine agent behavior."""
    w = _base(tmp_path, "t2")
    for i in range(3):
        w.emit({"kind": "llm_response", "layer": "observability", "iteration": i,
                "text": "", "tool_calls_requested": 1})
        w.emit({"kind": "tool_call", "layer": "tool_interface", "iteration": i,
                "tool": "search", "args_hash": "same"})
        w.emit({"kind": "tool_result", "layer": "tool_interface", "iteration": i,
                "tool": "search", "args_hash": "same"})
    w.emit({"kind": "guard_triggered", "layer": "lifecycle", "iteration": 2,
            "guard": "stuck", "note": "repeated identical call"})
    _close(w, "stuck", "stuck")
    return w.path


def trace_demo_fixture(tmp_path):
    """demo-* traces are spec fixtures — always noise regardless of shape."""
    w = _base(tmp_path, "demo-stuck")
    for i in range(3):
        w.emit({"kind": "tool_call", "layer": "tool_interface", "iteration": i,
                "tool": "search", "args_hash": "sym"})
        w.emit({"kind": "tool_result", "layer": "tool_interface", "iteration": i,
                "tool": "search", "args_hash": "sym"})
    w.emit({"kind": "guard_triggered", "layer": "lifecycle", "iteration": 2,
            "guard": "stuck", "note": "repeated identical call"})
    _close(w, "stuck", "stuck")
    return w.path


def trace_interrupted_no_guard(tmp_path):
    """17131c/9c89a2/f00b30/fc5199: no loop_end, no guard, no error."""
    w = _base(tmp_path, "t3")
    for i in range(8):
        _read(w, i)
    return w.path


def trace_tool_error_no_answer(tmp_path):
    """tool_error without final answer -> bug."""
    w = _base(tmp_path, "t4")
    w.emit({"kind": "tool_call", "layer": "tool_interface", "iteration": 0,
            "tool": "write", "args_hash": "w0"})
    w.emit({"kind": "tool_error", "layer": "tool_interface", "iteration": 0,
            "exception": "PermissionError", "message": "denied"})
    _close(w, "error", "error")
    return w.path


def trace_stuck_after_edits(tmp_path):
    """536848: edits applied, then stuck guard on a repeated search."""
    w = _base(tmp_path, "t5")
    w.emit({"kind": "tool_call", "layer": "tool_interface", "iteration": 0,
            "tool": "edit", "args_hash": "e0"})
    w.emit({"kind": "tool_result", "layer": "tool_interface", "iteration": 0,
            "tool": "edit", "args_hash": "e0"})
    for i in range(1, 4):
        w.emit({"kind": "llm_response", "layer": "observability", "iteration": i,
                "text": "", "tool_calls_requested": 1})
        w.emit({"kind": "tool_call", "layer": "tool_interface", "iteration": i,
                "tool": "search", "args_hash": "same"})
        w.emit({"kind": "tool_result", "layer": "tool_interface", "iteration": i,
                "tool": "search", "args_hash": "same"})
    w.emit({"kind": "guard_triggered", "layer": "lifecycle", "iteration": 3,
            "guard": "stuck", "note": "repeated identical call"})
    # run interrupted right after the guard: no loop_end
    return w.path


def trace_no_mutation_circling(tmp_path):
    """0f7793/b5b264: silent read loop, no_mutation guard, no answer."""
    w = _base(tmp_path, "t6")
    for i in range(10):
        _read(w, i)
    w.emit({"kind": "guard_triggered", "layer": "lifecycle", "iteration": 9,
            "guard": "no_mutation", "note": "30 calls without modifying"})
    _close(w, "no_progress", "no_progress")
    return w.path


def trace_no_mutation_after_work(tmp_path):
    """9acf2d: writes done, then a long read phase tripped the guard, no answer."""
    w = _base(tmp_path, "t7")
    w.emit({"kind": "tool_call", "layer": "tool_interface", "iteration": 0,
            "tool": "write", "args_hash": "w0"})
    w.emit({"kind": "tool_result", "layer": "tool_interface", "iteration": 0,
            "tool": "write", "args_hash": "w0"})
    for i in range(1, 9):
        _read(w, i)
    w.emit({"kind": "guard_triggered", "layer": "lifecycle", "iteration": 8,
            "guard": "no_mutation", "note": "50 calls without modifying"})
    _close(w, "no_progress", "no_progress")
    return w.path


def trace_provider_error(tmp_path):
    """e0b03b/f1b4320b: loop_end outcome=error, no tool_error, no answer."""
    w = _base(tmp_path, "t8")
    _read(w, 0)
    _close(w, "error", "answer")
    return w.path


def trace_guard_after_answer(tmp_path):
    """a669a26e pattern: guard-terminated but a substantive final answer exists."""
    w = _base(tmp_path, "t9")
    _read(w, 0)
    w.emit({"kind": "guard_triggered", "layer": "lifecycle", "iteration": 0,
            "guard": "no_mutation", "note": "stopping"})
    w.emit({"kind": "llm_response", "layer": "observability", "iteration": 1,
            "text": "final answer: " + "x" * 200, "tool_calls_requested": 0})
    _close(w, "no_progress", "no_progress")
    return w.path


GOLDEN = [
    (trace_fixture_delivered_after_tool_error, "noise", "high"),
    (trace_demo_fixture, "noise", "high"),
    (trace_interrupted_no_guard, "noise", "high"),
    (trace_tool_error_no_answer, "bug", "high"),
    (trace_stuck_after_edits, "bug", "high"),
    (trace_stuck_no_answer, "bug", "high"),
    (trace_no_mutation_circling, "bug", "high"),
    (trace_no_mutation_after_work, "bug", "high"),
    (trace_provider_error, "noise", "medium"),
    (trace_guard_after_answer, "noise", "high"),
]


@pytest.mark.parametrize("builder,expected,conf", GOLDEN,
                         ids=[b.__name__ for b, _, _ in GOLDEN])
def test_golden_patterns(tmp_path, builder, expected, conf):
    path = builder(tmp_path)
    verdict = auto_review(compile_trace(path))
    assert verdict.disposition == expected
    assert verdict.confidence == conf
    assert verdict.evidence


def test_agent_note_is_prefixed(tmp_path):
    verdict = auto_review(compile_trace(trace_no_mutation_circling(tmp_path)))
    assert verdict.note.startswith("agent auto-review: ")
    assert verdict.evidence


# ── trust model ──────────────────────────────────────────────────────────

def test_agent_cannot_label_forbidden_dispositions():
    assert AGENT_FORBIDDEN == {"ok", "regression"}


def test_agent_cannot_overwrite_human_label(tmp_path):
    reviews = {"t": ReviewRecord(task_id="t")}
    label_review(reviews, "t", "noise", source="human")
    with pytest.raises(ValueError, match="human labels win"):
        label_review(reviews, "t", "bug", source="agent")


def test_agent_cannot_assign_ok_or_regression(tmp_path):
    reviews = {"t": ReviewRecord(task_id="t")}
    for bad in ("ok", "regression"):
        with pytest.raises(ValueError, match="no self-certification"):
            label_review(reviews, "t", bad, source="agent")


def test_agent_label_records_source_and_human_override(tmp_path):
    reviews = {"t": ReviewRecord(task_id="t")}
    rec = label_review(reviews, "t", "bug", source="agent")
    assert rec.source == "agent"
    rec2 = label_review(reviews, "t", "noise", source="human")
    assert rec2.source == "human" and rec2.disposition == "noise"


def test_review_table_marks_agent_labels(tmp_path):
    from harnessfix.review import review_table

    reviews = {"t": ReviewRecord(task_id="t", disposition="bug", source="agent")}
    table = review_table(reviews)
    assert "bug (agent)" in table


def test_real_corpus_matches_pre050_labels():
    """The engine must reproduce the 12 hand-labeled pre-#050 verdicts.

    Runs against the real corpus; skipped when it is not present
    (reports/ is gitignored and not part of CI).
    """
    corpus = Path("reports/traces")
    if not corpus.is_dir():
        pytest.skip("trace corpus not present (reports/ is gitignored)")

    import json

    expected = {
        "bug": ["0f7793", "536848", "9acf2d", "b5b264"],
        "noise": ["17131c", "9c89a2", "demo-h", "demo-s",
                  "e0b03b", "f00b30", "f1b432", "fc5199"],
    }
    actual: dict[str, list[str]] = {"bug": [], "noise": []}
    for p in corpus.glob("*.jsonl"):
        found = False
        for group, prefixes in expected.items():
            for prefix in prefixes:
                if p.stem.startswith(prefix):
                    found = True
                    break
            if found:
                break
        if not found:
            continue
        verdict = auto_review(compile_trace(p))
        assert verdict.disposition in ("bug", "noise"), p.stem
        actual[verdict.disposition].append(p.stem[:6])
    assert sorted(actual["bug"]) == sorted(expected["bug"])
    assert sorted(actual["noise"]) == sorted(expected["noise"])