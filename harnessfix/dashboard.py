"""Small CLI dashboard for recent harnessfix traces and diagnoses.

Usage:
  python -m harnessfix.dashboard --traces reports/traces --diagnoses reports/harnessfix/diagnoses --limit 10
  python -m harnessfix.dashboard --traces reports/traces --diagnoses reports/harnessfix/diagnoses --task <task_id>
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .reader import read_trace, TraceValidationError
from .tracing import TRACE_DIR

#: Payload fields worth showing in the per-event timeline, by event kind.
_EVENT_FIELDS: dict[str, tuple[str, ...]] = {
    "step_start": ("iteration", "budget_remaining"),
    "llm_response": ("iteration", "tool_calls_requested"),
    "tool_call": ("iteration", "tool", "args_hash"),
    "tool_result": ("iteration", "tool", "duration_s"),
    "tool_error": ("iteration", "tool", "exception"),
    "guard_triggered": ("iteration", "guard"),
    "loop_end": ("iteration", "outcome", "termination_reason"),
}


def _outcome_of(events: list[dict[str, Any]]) -> str:
    for ev in reversed(events):
        if ev.get("kind") == "loop_end":
            return str(ev.get("outcome", "unknown"))
    # No loop_end yet -> trace is still open / incomplete
    return "incomplete"


def _summarize_trace(path: Path) -> dict[str, Any]:
    try:
        events = read_trace(path)
    except TraceValidationError:
        return {"task_id": path.stem, "error": "invalid"}
    task_id = str(events[0].get("task_id", path.stem))
    outcome = _outcome_of(events)
    failed = sum(1 for e in events if e.get("kind") in ("tool_error",) or (e.get("kind") == "loop_end" and e.get("outcome") not in ("completed",)))
    # Count guard triggers for quick health signal
    guards = sum(1 for e in events if e.get("kind") == "guard_triggered")
    return {
        "task_id": task_id,
        "path": str(path),
        "events": len(events),
        "outcome": outcome,
        "failed_signals": failed,
        "guards": guards,
    }


def _load_diagnosis(diag_dir: Path, task_id: str) -> dict[str, Any] | None:
    if not diag_dir.is_dir():
        return None
    p = diag_dir / f"{task_id}.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return {
        "layer": data.get("root_layer"),
        "mechanism": data.get("mechanism"),
        "confidence": data.get("confidence"),
        "evidence": data.get("evidence", []),
        "repair_proposal": data.get("repair_proposal"),
    }


def _find_trace(traces_dir: Path, task_id: str) -> Path | None:
    """Locate the trace file for a task id (bare id or id with .jsonl)."""
    for candidate in (Path(f"{task_id}.jsonl"), Path(f"{task_id}")):
        p = traces_dir / candidate
        if p.is_file():
            return p
    return None


def _event_line(index: int, ev: dict[str, Any]) -> str:
    """One compact timeline line for a single trace event."""
    kind = str(ev.get("kind", "?"))
    layer = str(ev.get("layer", "?"))
    bits: list[str] = []
    for field in _EVENT_FIELDS.get(kind, ()):
        value = ev.get(field)
        if value is not None:
            bits.append(f"{field}={value}")
    for field, cap in (("message", 90), ("text", 90)):
        value = ev.get(field)
        if isinstance(value, str) and value.strip():
            bits.append(f"{field}={value[:cap]!r}")
            break
    marker = "FAIL" if kind == "tool_error" else ("GUARD" if kind == "guard_triggered" else "")
    line = f"[{index:>3}] {layer:<20} {kind:<16} {' '.join(bits)}"
    return f"{line}  <-- {marker}" if marker else line


def _explain_task(events: list[dict[str, Any]], summary: dict[str, Any], diag: dict[str, Any] | None) -> list[str]:
    """Build a short human-readable explanation of what happened in the task."""
    lines: list[str] = []
    outcome = summary["outcome"]
    if outcome == "completed":
        lines.append(f"The task completed after {summary['events']} trace events with no unresolved failure.")
    elif outcome == "incomplete":
        lines.append("The trace is still open: no loop_end event was recorded yet.")
    else:
        reason = ""
        for ev in reversed(events):
            if ev.get("kind") == "loop_end":
                reason = str(ev.get("termination_reason", ""))
                break
        suffix = f" (termination reason: {reason})" if reason else ""
        lines.append(f"The loop terminated with outcome {outcome!r}{suffix} after {summary['events']} trace events.")

    failures = [(i, ev) for i, ev in enumerate(events) if ev.get("kind") == "tool_error"]
    guards = [ev for ev in events if ev.get("kind") == "guard_triggered"]
    dupes = sum(1 for ev in events if ev.get("kind") == "tool_call" and ev.get("duplicate"))

    for i, ev in failures[:3]:
        lines.append(
            f"Tool error at event {i}: {ev.get('tool', '?')} raised "
            f"{ev.get('exception', '?')}: {str(ev.get('message', ''))[:90]}"
        )
    if len(failures) > 3:
        lines.append(f"... and {len(failures) - 3} more tool error(s).")
    if guards:
        names = ", ".join(str(ev.get("guard", "?")) for ev in guards)
        lines.append(f"Lifecycle guard(s) fired: {names}.")
    if dupes:
        lines.append(f"The model repeated an already-executed tool call {dupes} time(s) (duplicate flag).")

    if diag and diag.get("layer"):
        lines.append(
            f"Diagnosis: {diag['layer']} — {diag.get('mechanism', 'unknown')} "
            f"(confidence {diag.get('confidence', 'n/a')})."
        )
        if diag.get("repair_proposal"):
            lines.append(f"Suggested repair: {diag['repair_proposal']}.")
    else:
        lines.append("No stored diagnosis for this task.")
    return ["Explanation:", *("  " + line for line in lines)]


def _show_task(traces_dir: Path, diag_dir: Path, task_id: str, as_json: bool) -> int:
    """Zoom into a single task: full trace timeline, diagnosis and explanation."""
    trace_path = _find_trace(traces_dir, task_id)
    if trace_path is None:
        print(f"Unknown task: {task_id!r} (no matching trace file in {traces_dir})")
        return 1
    try:
        events = read_trace(trace_path)
    except TraceValidationError as exc:
        print(f"Invalid trace for task {task_id!r}: {exc}")
        return 1
    summary = _summarize_trace(trace_path)
    diag = _load_diagnosis(diag_dir, summary["task_id"])

    if as_json:
        print(
            json.dumps(
                {
                    "task_id": summary["task_id"],
                    "path": str(trace_path),
                    "summary": summary,
                    "diagnosis": diag,
                    "events": events,
                },
                indent=2,
                default=str,
            )
        )
        return 0

    print(f"HarnessFix task detail — {summary['task_id']}")
    print("=" * 100)
    print(f"  trace file : {trace_path}")
    print(f"  outcome    : {summary['outcome']}")
    print(f"  events     : {summary['events']}")
    print(f"  failed     : {summary['failed_signals']} failed signal(s)")
    print(f"  guards     : {summary['guards']} guard trigger(s)")
    if diag and diag.get("layer"):
        print(f"  diagnosis  : {diag['layer']} — {diag.get('mechanism', '?')}  (confidence {diag.get('confidence', 'n/a')})")
        evidence = diag.get("evidence") or []
        if evidence:
            print(f"  evidence   : {', '.join(str(e) for e in evidence)}")
        if diag.get("repair_proposal"):
            print(f"  repair     : {diag['repair_proposal']}")
    else:
        print("  diagnosis  : none stored")
    print("-" * 100)
    print("Timeline:")
    for index, ev in enumerate(events):
        print(_event_line(index, ev))
    print("-" * 100)
    for line in _explain_task(events, summary, diag):
        print(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harnessfix.dashboard", description="Print a short dashboard of recent traces/diagnoses")
    parser.add_argument("--traces", type=Path, default=TRACE_DIR, help="trace directory")
    parser.add_argument("--diagnoses", type=Path, default=Path("reports/harnessfix/diagnoses"), help="diagnoses directory")
    parser.add_argument("--limit", type=int, default=10, help="how many recent traces to show")
    parser.add_argument("--json", action="store_true", help="output JSON instead of table")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="zoom into a single task: full trace timeline, diagnosis and explanation",
    )
    args = parser.parse_args(argv)

    if not args.traces.is_dir():
        print(f"No trace directory: {args.traces}")
        return 1

    if args.task is not None:
        return _show_task(args.traces, args.diagnoses, args.task, as_json=args.json)

    files = sorted(args.traces.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[: args.limit]
    rows = []
    layer_counter = Counter()
    for p in files:
        s = _summarize_trace(p)
        diag = _load_diagnosis(args.diagnoses, s["task_id"])
        if diag and diag.get("layer"):
            layer_counter[diag["layer"]] += 1
        rows.append({
            "task_id": s.get("task_id"),
            "path": s.get("path"),
            "events": s.get("events"),
            "outcome": s.get("outcome"),
            "failed_signals": s.get("failed_signals"),
            "guards": s.get("guards"),
            "diagnosis": diag,
        })

    if args.json:
        out = {
            "traces_dir": str(args.traces),
            "count": len(rows),
            "rows": rows,
            "layer_distribution": dict(layer_counter),
        }
        print(json.dumps(out, indent=2))
        return 0

    # Header
    print(f"HarnessFix dashboard — {len(rows)} recent traces from {args.traces}")
    print("-" * 110)
    print(f"{'task_id':<36} {'outcome':<12} {'events':>6} {'failed':>6} {'guards':>6} {'layer':<18} mechanism")
    print("-" * 110)
    for r in rows:
        s = r
        diag = s.get("diagnosis")
        tid = s.get("task_id", "")[:34]
        outcome = s.get("outcome", "")
        events = s.get("events", 0)
        failed = s.get("failed_signals", 0)
        guards = s.get("guards", 0)
        layer = diag.get("layer") if diag else ""
        mech = (diag.get("mechanism") or "")[:60] if diag else ""
        print(f"{tid:<36} {outcome:<12} {events:>6} {failed:>6} {guards:>6} {layer:<18} {mech}")

    if layer_counter:
        print("\nLayer distribution (diagnosed failures):")
        for layer, cnt in layer_counter.most_common():
            print(f"  {layer:<22} {cnt}")
    else:
        print("\nNo diagnoses found for the shown traces.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
