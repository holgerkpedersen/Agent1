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
- ``--max-iterations`` caps the run (default 5); ``--model`` is OPTIONAL:
  the LLM benchmark is now a non-blocking cross-check, not the primary gate.
  When ``--model`` is omitted the loop gates on the offline, harness-centric
  quality signal (corpus target-alignment) plus tests + security.  Pass
  ``--no-benchmark`` to suppress the benchmark subprocess entirely even when a
  model is set.

Usage
-----
    set AGENT_AUTONOMOUS=1
    python scripts/autonomous_self_improve.py --max-iterations 5
    python scripts/autonomous_self_improve.py --model qwen3.5-32b --no-benchmark
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
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=check
        )
    except FileNotFoundError as exc:
        # git is not on PATH for this process.  Surface a clear, actionable
        # error instead of a bare FileNotFoundError pointing at the _git() call
        # site (e.g. the per-iteration `git stash push` on line 146).
        raise RuntimeError(
            "git executable not found on PATH; the autonomous driver requires "
            "git to be installed and reachable.  Add git to PATH and retry."
        ) from exc


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
    no_benchmark: bool = False,
) -> dict[str, Any]:
    """One autonomous HarnessFix iteration; returns its summary dict."""
    from harnessfix.loop import run_loop

    return run_loop(
        trace_dir,
        approve=False,
        auto_approve=True,
        model=None if no_benchmark else model,
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
    parser.add_argument("--model", default=None, help="optional LLM benchmark cross-check model (needs a live API); omit to gate on offline harness quality only")
    parser.add_argument("--profile", default=None, help="benchmark profile tag (decision #055)")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="never run the benchmark subprocess, even if --model is set (offline-only gate)")
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
          f"model={args.model or 'none (offline harness-quality gate)'}"
          f"{' [benchmark disabled]' if args.no_benchmark else ''})")

    for iteration in range(1, args.max_iterations + 1):
        if _stop_requested():
            print("[autonomous] Kill-switch detected — halting before iteration "
                  f"{iteration}.")
            return 0

        # Git checkpoint so any autonomous change is a one-step revert.
        # Track whether the checkpoint was actually created: a `git stash push`
        # can be a no-op (nothing to stash) or fail outright (e.g. no initial
        # commit), and blindly popping afterwards would pop an *unrelated*
        # stash and corrupt state.
        checkpoint = f"autonomous-checkpoint-iter-{iteration}"
        pushed = _git(["stash", "push", "-u", "-m", checkpoint], check=False)
        have_checkpoint = pushed.returncode == 0
        if not have_checkpoint:
            print(f"[autonomous] WARNING: git stash push failed "
                  f"(rc={pushed.returncode}); continuing without a checkpoint: "
                  f"{pushed.stderr.strip()}")

        print(f"\n[autonomous] === iteration {iteration}/{args.max_iterations} ===")
        try:
            summary = run_iteration(
                iteration, model=args.model, profile=args.profile,
                trace_dir=trace_dir, output_dir=output_dir,
                no_benchmark=args.no_benchmark,
            )
        except SystemExit as exc:
            # run_loop raises SystemExit when no traces exist — a clean stop.
            print(f"[autonomous] Loop stopped: {exc}")
            if have_checkpoint:
                _git(["stash", "pop"], check=False)
            return 0
        except KeyboardInterrupt:
            # A Ctrl+C during a long gate run is a BaseException, so it is NOT
            # caught by `except Exception` below.  Restore the checkpoint first
            # so the tree is not left dirty, then re-raise to actually stop.
            print("[autonomous] Interrupted (KeyboardInterrupt) — restoring "
                  "checkpoint and halting.")
            if have_checkpoint:
                _git(["stash", "pop"], check=False)
            raise
        except Exception as exc:  # noqa: BLE001 - never let one bad iteration
            # crash the whole driver with a dirty tree.  Log, restore the
            # checkpoint, and stop so the human can inspect.
            print(f"[autonomous] Iteration {iteration} raised {type(exc).__name__}: {exc}")
            if have_checkpoint:
                _git(["stash", "pop"], check=False)
            return 1

        verdict = summary.get("verdict")
        print(f"[autonomous] verdict={verdict} repair={summary.get('proposed_repair')}")

        if verdict == "accepted":
            committed = _commit_repair(iteration, summary)
            print(f"[autonomous] Repair accepted and "
                  f"{'committed' if committed else 'already clean'}.")
            # Restore the stash (nothing should remain uncommitted after commit).
            if have_checkpoint:
                _git(["stash", "drop"], check=False)
            continue

        # Any non-accepted verdict (review_required_fail_closed,
        # rejected_and_reverted, no_repair_catalogued, skipped_test_collision,
        # apply_failed, revert_failed) ends the loop: there is no improvement
        # to keep, and re-running would only repeat the same dead end.
        print(f"[autonomous] No accepted repair this round ({verdict}) — stopping.")
        if have_checkpoint:
            _git(["stash", "pop"], check=False)
        return 0

    print("[autonomous] Reached max_iterations — stopping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
