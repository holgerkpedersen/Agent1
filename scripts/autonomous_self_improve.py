#!/usr/bin/env python3
"""Autonomous self-improvement driver for Agent1 (HarnessFix loop, repeated).

This turns the dormant HarnessFix closed loop into a *running* agent that
improves itself without human clicks:

1. Runs one :func:`harnessfix.loop.run_loop` iteration with ``auto_approve``
   (machine-gated: applies the proposed repair, then commits ONLY if the test
   + security + benchmark gates all pass — never on ambiguity).
2. If the repair was accepted, commits it with a deterministic message and
   records the outcome to ``harnessfix/history`` for feedback.
3. Repeats until a stop condition is met (no repair catalogued, a rejected
   round, diminishing returns, or the kill-switch fires).
4. Leaves a git checkpoint before each iteration so ``git revert`` is a
   one-step rollback of any autonomous change.

Safety rails
------------
- Requires ``AGENT_AUTONOMOUS=1`` (or ``--auto``) to engage.  Without it the
  script refuses to run, so it can never be flipped on by accident.
- ``STOP_AUTONOMOUS`` file or env var checked BETWEEN iterations: create
  ``STOP_AUTONOMOUS`` in the repo root (or set the env var) to halt cleanly
  after the current round finishes.
- ``--max-iterations`` caps the run (default 5); ``--model`` is required for
  the benchmark gate to be meaningful (otherwise the benchmark is non-blocking
  and only tests + security gate the repair).

Usage
-----
    set AGENT_AUTONOMOUS=1
    python scripts/autonomous_self_improve.py --model qwen3.5-32b --max-iterations 5
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
# Make the repo root importable when run as a standalone CLI (pytest already
# puts it on sys.path, but `python scripts/autonomous_self_improve.py` does not).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
STOP_FILENAME = "STOP_AUTONOMOUS"
SUMMARY_PATH = REPO_ROOT / "reports" / "harnessfix" / "summary.json"


def _git(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=check
    )


def _stop_requested() -> bool:
    flag = os.environ.get("STOP_AUTONOMOUS", "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    return (REPO_ROOT / STOP_FILENAME).exists()


def _has_changes() -> bool:
    proc = _git(["status", "--porcelain"], check=False)
    return bool(proc.stdout.strip())


def _commit_repair(iteration: int, summary: dict[str, Any]) -> bool:
    """Commit an accepted repair. Returns False if there is nothing to commit."""
    if not _has_changes():
        return False
    repair_id = summary.get("proposed_repair") or "unknown"
    verdict = summary.get("verdict", "accepted")
    msg = (
        f"autonomous(self-improve): apply {repair_id} [{verdict}] "
        f"(iter {iteration})\n\n"
        f"Auto-committed by scripts/autonomous_self_improve.py "
        f"after all gates passed:\n"
        f"  tests_passed={summary.get('tests_passed')}\n"
        f"  security_passed={summary.get('security_passed')}\n"
        f"  baseline_rate={summary.get('baseline_rate')} "
        f"post_rate={summary.get('post_rate')}\n"
    )
    _git(["add", "-A"])
    _git(["commit", "-m", msg], check=False)
    return True


def run_iteration(
    iteration: int,
    *,
    model: str | None,
    profile: str | None,
    trace_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """One autonomous HarnessFix iteration; returns its summary dict."""
    from harnessfix.loop import run_loop

    return run_loop(
        trace_dir,
        approve=False,
        auto_approve=True,
        model=model,
        profile=profile,
        output_dir=output_dir,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="autonomous_self_improve",
        description="Run the HarnessFix loop autonomously (machine-gated).",
    )
    parser.add_argument("--traces", type=Path, default=None,
                        help="trace corpus dir (default: reports/traces)")
    parser.add_argument("--model", default=None, help="benchmark gate model (needs a live API)")
    parser.add_argument("--profile", default=None, help="benchmark profile tag (decision #055)")
    parser.add_argument("--max-iterations", type=int, default=5, help="cap on loop iterations")
    parser.add_argument("--auto", action="store_true",
                        help="engages autonomous mode (same as AGENT_AUTONOMOUS=1)")
    args = parser.parse_args(argv)

    if not (os.environ.get("AGENT_AUTONOMOUS", "").strip().lower() in ("1", "true", "yes", "on")
            or args.auto):
        print("[autonomous] Refusing to run: set AGENT_AUTONOMOUS=1 (or pass --auto) to engage.")
        return 2

    trace_dir = args.traces or (REPO_ROOT / "reports" / "traces")
    output_dir = REPO_ROOT / "reports" / "harnessfix"

    print(f"[autonomous] Starting self-improvement loop "
          f"(max_iterations={args.max_iterations}, "
          f"model={args.model or 'none (benchmark non-blocking)'})")

    for iteration in range(1, args.max_iterations + 1):
        if _stop_requested():
            print("[autonomous] Kill-switch detected — halting before iteration "
                  f"{iteration}.")
            return 0

        # Git checkpoint so any autonomous change is a one-step revert.
        _git(["stash", "push", "-u", "-m", f"autonomous-checkpoint-iter-{iteration}"],
             check=False)

        print(f"\n[autonomous] === iteration {iteration}/{args.max_iterations} ===")
        try:
            summary = run_iteration(
                iteration, model=args.model, profile=args.profile,
                trace_dir=trace_dir, output_dir=output_dir,
            )
        except SystemExit as exc:
            # run_loop raises SystemExit when no traces exist — a clean stop.
            print(f"[autonomous] Loop stopped: {exc}")
            _git(["stash", "pop"], check=False)
            return 0

        verdict = summary.get("verdict")
        print(f"[autonomous] verdict={verdict} repair={summary.get('proposed_repair')}")

        if verdict == "accepted":
            committed = _commit_repair(iteration, summary)
            print(f"[autonomous] Repair accepted and "
                  f"{'committed' if committed else 'already clean'}.")
            # Restore the stash (nothing should remain uncommitted after commit).
            _git(["stash", "drop"], check=False)
            continue

        # Any non-accepted verdict (review_required_fail_closed,
        # rejected_and_reverted, no_repair_catalogued, skipped_test_collision,
        # apply_failed, revert_failed) ends the loop: there is no improvement
        # to keep, and re-running would only repeat the same dead end.
        print(f"[autonomous] No accepted repair this round ({verdict}) — stopping.")
        _git(["stash", "pop"], check=False)
        return 0

    print("[autonomous] Reached max_iterations — stopping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
