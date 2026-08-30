"""`demo_data` — feed synthetic activity into the shared metrics collector.

Purpose: the TTTHEME dashboard (http://localhost:8081) renders whatever the
process-wide ``MetricsCollector`` holds. On a fresh start every view shows
"No counters / No activity recorded yet" until real commands run. This
command fills every dashboard surface deterministically — LLM-free:

  - stat card "Commands Executed"  <- per-command counters
  - Command Summary table          <- ``command.<name>.count`` counters
  - Gauges view                    <- gauges incl. ``last.command.seconds``
  - Histogram view                 <- ``command.elapsed.seconds`` samples
  - Execution Log                  <- raw metric events (deque store)

Examples:
    demo_data                    # 5 mixed activities (read/search/...)
    demo_data --activity analyze # only 'analyze' events
    demo_data --count 12         # 12 events instead of 5
    demo_data --latency-ms 250   # mean latency of the generated samples
    demo_data --loop 10          # one batch per second, 10 rounds
    demo_data --clear            # wipe collector + active alerts
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from .base import Command

if TYPE_CHECKING:
    from agent import Agent

#: Deterministic activity mix when --activity is not given.
_DEFAULT_ACTIVITIES = ("read", "search", "analyze", "fix", "run")


class DemoDataCommand(Command):
    """Generate dashboard-visible metrics without running real commands."""

    @property
    def name(self) -> str:
        return "demo_data"

    @property
    def help_text(self) -> str:
        return (
            "demo_data [--activity <name>] [--count N] [--latency-ms MS] "
            "[--loop N] [--clear] - Feed synthetic data into the web "
            "dashboard"
        )

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        activity: str | None = None
        count = 5
        latency_ms = 120.0
        loop_rounds = 1
        clear = False

        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--activity":
                if i + 1 >= len(args):
                    self.error("--activity expects a name.")
                    return True
                activity = args[i + 1]
                i += 2
            elif arg == "--count":
                try:
                    count = max(1, int(args[i + 1]))
                except (IndexError, ValueError):
                    self.error("--count expects a positive integer.")
                    return True
                i += 2
            elif arg == "--latency-ms":
                try:
                    latency_ms = max(1.0, float(args[i + 1]))
                except (IndexError, ValueError):
                    self.error("--latency-ms expects milliseconds (number).")
                    return True
                i += 2
            elif arg == "--loop":
                try:
                    loop_rounds = max(1, int(args[i + 1]))
                except (IndexError, ValueError):
                    self.error("--loop expects a number of rounds.")
                    return True
                i += 2
            elif arg == "--clear":
                clear = True
                i += 1
            else:
                self.error(f"Unknown argument: {arg} (see help).")
                return True

        if clear:
            agent.get_metrics_collector().reset()
            print("  [demo] metrics collector cleared.")
            return True

        activities = (activity,) if activity else _DEFAULT_ACTIVITIES
        total: float = 0
        for round_no in range(loop_rounds):
            if round_no:
                await asyncio.sleep(1.0)  # spread batches so the log scrolls
            for j in range(count):
                name = activities[j % len(activities)]
                result = agent.record_demo_activity(
                    activity=name, latency_ms=latency_ms
                )
                total += result["events"]
            print(f"  [demo] round {round_no + 1}/{loop_rounds}: "
                  f"+{count} events fed (running total {total})")

        print("  [demo] done. Open http://localhost:8081 "
              "(UI polls every 3s; llama-server LLM backend on :8080).")
        return True
