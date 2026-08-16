"""Small CLI dashboard for recent harnessfix traces and diagnoses.

Usage:
  python -m harnessfix.dashboard --traces reports/traces --diagnoses reports/harnessfix/diagnoses --limit 10
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .reader import read_trace, TraceValidationError
from .tracing import TRACE_DIR


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
        return {
            "layer": data.get("root_layer"),
            "mechanism": data.get("mechanism"),
            "confidence": data.get("confidence"),
        }
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harnessfix.dashboard", description="Print a short dashboard of recent traces/diagnoses")
    parser.add_argument("--traces", type=Path, default=TRACE_DIR, help="trace directory")
    parser.add_argument("--diagnoses", type=Path, default=Path("reports/harnessfix/diagnoses"), help="diagnoses directory")
    parser.add_argument("--limit", type=int, default=10, help="how many recent traces to show")
    parser.add_argument("--json", action="store_true", help="output JSON instead of table")
    args = parser.parse_args(argv)

    if not args.traces.is_dir():
        print(f"No trace directory: {args.traces}")
        return 1

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
