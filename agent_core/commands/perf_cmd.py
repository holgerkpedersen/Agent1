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
        total_calls = sum(s["calls"] for s in summary)
        total_time = sum(float(s["total"].rstrip("s")) for s in summary)
        return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<title>Agent1 Performance Dashboard</title>
<link rel="icon" type="image/x-icon" href="static/favicon.png">
<link rel="stylesheet" href="static/theme.css">
</head>
<body>
<div class="app-container">
    <div class="header">
        <div style="display:flex;align-items:center;gap:8px;">
            <img src="static/favicon.png" alt="Agent1 Logo" style="width:32px;height:32px;">
            <h6 class="fw-semibold mb-0" style="display:inline-block;background:linear-gradient(135deg,var(--gradient-start),var(--gradient-end));-webkit-background-clip:text;background-clip:text;color:transparent;font-size:24px;">AGENT1</h6>
            <span style="font-size:12px;color:var(--text-muted);margin-left:8px;">Performance Dashboard</span>
            <button type="button" class="theme-toggle"
                onclick="toggleTheme()" id="themeToggle"
                aria-label="Toggle Light/Dark Theme"></button>
        </div>
    </div>

    <div class="panel" style="position:static;margin:0 0 12px 0;min-width:0;">
        <div class="panel-header">
            <h6 class="fw-semibold mb-0">Command Summary</h6>
        </div>
        <div class="panel-content">
            <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px;">
                <div style="flex:1;min-width:160px;background:var(--input-bg);border-radius:12px;padding:14px;border:1px solid var(--border);text-align:center;">
                    <div style="font-size:28px;color:var(--accent);line-height:1;margin-bottom:6px;" class="icon-cpu"></div>
                    <div style="font-size:22px;font-weight:700;">{total_calls}</div>
                    <div style="font-size:11px;color:var(--text-muted);">Commands</div>
                </div>
                <div style="flex:1;min-width:160px;background:var(--input-bg);border-radius:12px;padding:14px;border:1px solid var(--border);text-align:center;">
                    <div style="font-size:28px;color:var(--warning);line-height:1;margin-bottom:6px;" class="icon-clock"></div>
                    <div style="font-size:22px;font-weight:700;">{total_time:.1f}s</div>
                    <div style="font-size:11px;color:var(--text-muted);">Runtime</div>
                </div>
                <div style="flex:1;min-width:160px;background:var(--input-bg);border-radius:12px;padding:14px;border:1px solid var(--border);text-align:center;">
                    <div style="font-size:28px;color:var(--success);line-height:1;margin-bottom:6px;" class="icon-chart"></div>
                    <div style="font-size:22px;font-weight:700;">{len(summary)}</div>
                    <div style="font-size:11px;color:var(--text-muted);">Command Types</div>
                </div>
            </div>
            <table class="issues-table">
                <thead><tr><th>Command</th><th>Calls</th><th>Total</th><th>Avg</th><th>Max</th><th>Last</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
    </div>

    <div class="panel" style="position:static;margin:0 0 12px 0;min-width:0;">
        <div class="panel-header">
            <h6 class="fw-semibold mb-0">Execution Log (last 100)</h6>
        </div>
        <div class="panel-content">
            <table class="issues-table">
                <thead><tr><th>Time</th><th>Command</th><th>Elapsed</th></tr></thead>
                <tbody>{detail_rows}</tbody>
            </table>
        </div>
    </div>

    <footer class="footer">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
            <div>
                <span>Developed by Holger K. Pedersen</span>
                <a href="https://github.com/holgerkpedersen/Agent1" target="_blank" rel="noopener"
                    title="Holger K. Pedersen - Agent1" class="ms-1"><span id="ghIcon"></span></a>
            </div>
            <div>
                <span>UI/UX reshaped with care by <a href="https://2tinteractive.com" target="_blank" rel="noopener" style="color:var(--accent);text-decoration:none;">2tinteractive.com - The Spatial Digital Agency</a> (<a href="https://github.com/LebToki" target="_blank" rel="noopener" style="color:var(--accent);text-decoration:none;">Tarek Tarabichi</a>)</span>
            </div>
        </div>
    </footer>
</div>
<script src="static/assets/agent1-icons.js"></script>
<script>
let currentTheme = localStorage.getItem('agent1_theme') || 'dark';
document.documentElement.setAttribute('data-theme', currentTheme);
document.getElementById('themeToggle').innerHTML = iconify(currentTheme === 'dark' ? 'solar:moon-bold' : 'solar:sun-bold', 18);
document.querySelector('.icon-cpu').innerHTML = iconify('tabler:cpu', 28);
document.querySelector('.icon-clock').innerHTML = iconify('solar:clock-circle-bold', 28);
document.querySelector('.icon-chart').innerHTML = iconify('solar:chart-2-bold', 28);
document.getElementById('ghIcon').innerHTML = iconify('mdi:github', 18);
function toggleTheme() {{
    currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', currentTheme);
    localStorage.setItem('agent1_theme', currentTheme);
    document.getElementById('themeToggle').innerHTML = iconify(currentTheme === 'dark' ? 'solar:moon-bold' : 'solar:sun-bold', 18);
}}
</script>
</body>
</html>"""
