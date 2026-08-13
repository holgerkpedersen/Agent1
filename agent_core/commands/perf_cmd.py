"""Perf command — lightweight command performance tracker for the REPL.

Usage:
    perf                  Show summary table of command timings
    perf --detail         Show every individual execution with timestamps
    perf --reset          Clear all collected stats
    perf --html           Export as self-contained HTML dashboard
"""

from datetime import datetime as _datetime

from .base import Command

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from agent import Agent


class PerfTracker:
    """In-memory command timing collector — no threads, no SQLite, no deps."""

    _records: list[dict[str, Any]] = []

    @classmethod
    def record(cls, command: str, elapsed_s: float, input_text: str = "") -> None:
        cls._records.append({
            "command": command,
            "elapsed_s": round(elapsed_s, 3),
            "input_len": len(input_text),
            "timestamp": _datetime.now().isoformat(timespec="seconds"),
        })

    @classmethod
    def summary(cls) -> list[dict[str, Any]]:
        """Aggregate by command name, return sorted by total time descending."""
        by_cmd: dict[str, list[float]] = {}
        for r in cls._records:
            by_cmd.setdefault(r["command"], []).append(r["elapsed_s"])
        result = []
        for cmd, times in sorted(by_cmd.items(), key=lambda x: -sum(x[1])):
            result.append({
                "command": cmd,
                "calls": len(times),
                "total": f"{sum(times):.1f}s",
                "avg": f"{sum(times) / len(times):.1f}s",
                "max": f"{max(times):.1f}s",
                "last": f"{times[-1]:.1f}s",
            })
        return result

    @classmethod
    def detail(cls) -> list[dict[str, Any]]:
        return cls._records

    @classmethod
    def reset(cls) -> None:
        cls._records.clear()


class PerfCommand(Command):
    """Show or reset performance statistics from the current session."""

    @property
    def name(self) -> str:
        return "perf"

    @property
    def help_text(self) -> str:
        return "perf [--detail|--reset|--html] — Command performance dashboard"

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        if "--reset" in args:
            PerfTracker.reset()
            print("  Performance stats cleared.")
            return True

        if "--detail" in args:
            records = PerfTracker.detail()
            if not records:
                print("  No commands executed yet.")
                return True
            print(f"\n  Execution log ({len(records)} entries):\n")
            for r in records:
                ts = r["timestamp"].split("T")[-1] if "T" in r["timestamp"] else r["timestamp"]
                print(f"  {ts}  {r['command']:<12} {r['elapsed_s']:>8.1f}s  ({r['input_len']} chars)")
            return True

        if "--html" in args:
            html = self._build_html(PerfTracker.summary(), PerfTracker.detail())
            path = "performance_dashboard.html"
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"  Dashboard exported to {path}")
            return True

        summary = PerfTracker.summary()
        if not summary:
            print("  No commands executed yet.")
            return True

        total_calls = sum(s["calls"] for s in summary)
        total_time = sum(float(s["total"].rstrip("s")) for s in summary)
        records = PerfTracker.detail()
        total_input = sum(r["input_len"] for r in records)
        llm_est = len([r for r in records if r["command"] not in ("read", "write", "search", "clear", "model", "cleanup")])

        print(f"\n  Commands: {total_calls}  |  Runtime: {total_time:.1f}s  |  Input: {total_input} chars  |  LLM calls: ~{llm_est}")
        print(f"  {'─'*72}")
        print(f"  {'command':<12} {'calls':>6} {'total':>8} {'avg':>8} {'max':>8} {'last':>8}")
        print(f"  {'─'*72}")
        for s in summary:
            print(f"  {s['command']:<12} {s['calls']:>6} {s['total']:>8} {s['avg']:>8} {s['max']:>8} {s['last']:>8}")
        print()

        return True

    def _build_html(self, summary: list[dict[str, Any]], detail: list[dict[str, Any]]) -> str:
        rows = "\n".join(
            f"<tr><td>{s['command']}</td><td>{s['calls']}</td><td>{s['total']}</td><td>{s['avg']}</td><td>{s['max']}</td><td>{s['last']}</td></tr>"
            for s in summary
        )
        detail_rows = "\n".join(
            f"<tr><td>{r['timestamp']}</td><td>{r['command']}</td><td>{r['elapsed_s']}s</td></tr>"
            for r in detail[:100]
        )
        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Agent Performance</title>
<style>
body {{font-family:monospace;background:#1a1a2e;color:#e0e0e0;padding:20px}}
h2 {{color:#00d4aa}} th,td {{padding:6px 14px;text-align:right}}
th {{background:#16213e;color:#00d4aa}} td {{background:#0f3460}}
tr:hover td {{background:#1a508b}}
</style></head><body>
<h2>Command Summary</h2>
<table><tr><th>Command</th><th>Calls</th><th>Total</th><th>Avg</th><th>Max</th><th>Last</th></tr>
{rows}</table>
<h2>Execution Log (last 100)</h2>
<table><tr><th>Time</th><th>Command</th><th>Elapsed</th></tr>
{detail_rows}</table>
</body></html>"""
