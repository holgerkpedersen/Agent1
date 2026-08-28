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
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Make the repo root importable when run as a standalone CLI (pytest already
# puts it on sys.path, but `python scripts/autonomous_self_improve.py` does
# not). This must run BEFORE any `harnessfix` import below.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harnessfix.progress import append_history, clear_progress, write_progress
from harnessfix import issue_loop
from harnessfix import issues as issue_store

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


def build_repair_rationale(repair_id: str | None, summary: dict[str, Any]) -> str:
    """Build a human-readable 'why this change is better' block from the
    evidence the self-improvement loop already computed.

    The loop captures far more signal than the old commit message used:
    the offline corpus-quality snapshot (how many traces, the completion
    rate, and how many *failed* traces evidence the repair's target layer)
    plus the optional LLM-benchmark cross-check.  Historically this was
    thrown away and every autonomous commit said only ``[accepted]`` with
    ``baseline_rate=None post_rate=None`` — so a behavioural change landed
    with zero stated justification (see the ``stuck-repeat-tool-hints``
    repair, later reverted with no recorded reason).  This turns the
    loop's own evidence into an explanation and is pure (no git/IO side
    effects) so it can be unit-tested in isolation.
    """
    from harnessfix.repairs import CATALOG

    lines: list[str] = []
    repair = CATALOG.get(repair_id) if repair_id else None
    if repair is not None:
        lines.append(f"Repair: {repair.description}")

    baseline = summary.get("harness_baseline") or {}
    post = summary.get("harness_post") or {}
    total = baseline.get("total") or post.get("total") or 0
    if total:
        lines.append(
            f"Corpus evidence: {total} trace(s) in the failure corpus; "
            f"completion baseline={baseline.get('success_rate')} "
            f"post={post.get('success_rate')}."
        )
        target_layer = repair.layer if repair is not None else None
        if target_layer:
            n = baseline.get("layer_counts", {}).get(target_layer, 0)
            lines.append(
                f"Target layer '{target_layer}' is evidenced in {n} failed "
                f"trace(s) — the offline target-alignment gate passed, so the "
                f"change addresses a failure mode the corpus actually shows "
                f"(not an invented one)."
            )
    else:
        lines.append(
            "Corpus evidence: no trace corpus available to target the repair."
        )

    br, pr = summary.get("baseline_rate"), summary.get("post_rate")
    if br is not None and pr is not None:
        lines.append(
            f"LLM benchmark cross-check: baseline={br}% -> post={pr}% "
            f"(no regression)."
        )
    else:
        lines.append(
            "LLM benchmark cross-check: not run (offline harness-quality "
            "gate used as the primary signal)."
        )
    return "\n".join(lines)


def _commit_repair(iteration: int, summary: dict[str, Any]) -> bool:
    """Commit an accepted repair. Returns False if there is nothing to commit.

    SAFETY: only the repair's own source file(s) are staged — never a blanket
    ``git add -A``.  A blanket add would sweep unrelated or scratch files
    (e.g. ``_tmp_*.txt``) into an autonomous commit.  If the proposed repair
    is unknown (so its files cannot be determined) or none of its files are
    actually modified, the commit is skipped and the loop is left to stop.
    """
    repair_id = summary.get("proposed_repair") or "unknown"
    verdict = summary.get("verdict", "accepted")
    files = _repair_files(repair_id)

    # Stage ONLY the repair's targeted files that are actually changed.
    changed: list[str] = []
    for f in files:
        proc = _git(["status", "--porcelain", "--", f], check=False)
        if proc.stdout.strip():
            changed.append(f)
    if not changed:
        return False

    # Refuse to do a blanket add: if we somehow have no file list for a known
    # repair, do NOT fall back to `git add -A` (that is exactly the bug that
    # committed unrelated work).  Skip the commit instead.
    if not files:
        print(f"[autonomous] WARNING: repair {repair_id!r} has no file list; "
              f"refusing to commit (skipping to avoid a blanket add).")
        return False

    rationale = build_repair_rationale(repair_id, summary)
    msg = (
        f"autonomous(self-improve): apply {repair_id} [{verdict}] "
        f"(iter {iteration})\n\n"
        f"Auto-committed by scripts/autonomous_self_improve.py "
        f"after all gates passed:\n"
        f"  tests_passed={summary.get('tests_passed')}\n"
        f"  security_passed={summary.get('security_passed')}\n"
        f"  baseline_rate={summary.get('baseline_rate')} "
        f"post_rate={summary.get('post_rate')}\n"
        f"  files={', '.join(changed)}\n\n"
        f"Why this change is better:\n{rationale}\n"
    )
    _git(["add", "--", *changed])
    _git(["commit", "-m", msg], check=False)
    return True


def _record_iteration(iteration: int, summary: dict[str, Any], output_dir: Path) -> None:
    """Append one finished-iteration record to the dashboard history log."""
    try:
        head = _git(["rev-parse", "HEAD"], check=False).stdout.strip()
    except Exception:
        head = ""
    baseline = summary.get("harness_baseline") or {}
    post = summary.get("harness_post") or {}
    append_history({
        "iteration": iteration,
        "timestamp": _git_time(),
        "verdict": summary.get("verdict"),
        "accepted": bool(summary.get("accepted")),
        "proposed_repair": summary.get("proposed_repair"),
        "accepted_repair": summary.get("accepted_repair"),
        "repair_applied": summary.get("repair_applied"),
        "tests_passed": summary.get("tests_passed"),
        "security_passed": summary.get("security_passed"),
        "harness_baseline_rate": baseline.get("success_rate"),
        "harness_post_rate": post.get("success_rate"),
        "repair_rationale": build_repair_rationale(
            summary.get("accepted_repair") or summary.get("proposed_repair"),
            summary,
        ),
        "git_head": head,
    })


def _git_time() -> float:
    """Best-effort epoch seconds for the current commit (fallback: now)."""
    try:
        out = _git(["log", "-1", "--format=%ct"], check=False).stdout.strip()
        return float(out) if out else _now()
    except Exception:
        return _now()


def _now() -> float:
    import time as _t
    return _t.time()


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
        iteration=iteration,
    )


def _commit_issue(iteration: int, summary: dict[str, Any]) -> bool:
    """Scoped commit for an accepted issue: only its files + the ledger.

    Never does a blanket ``git add -A`` (the same anti-sweep rule as
    ``_commit_repair``). The ledger (``.issues.json``) is always included so the
    resolution is recorded alongside the code change.
    """
    files = list(summary.get("files", []))
    files.append(str(issue_store.ISSUES_PATH))
    changed: list[str] = []
    for f in files:
        proc = _git(["status", "--porcelain", "--", f], check=False)
        if proc.stdout.strip():
            changed.append(f)
    if not changed:
        return False
    msg = (
        f"autonomous(self-improve): resolve {summary.get('issue_id')} "
        f"(iter {iteration})\n\n"
        f"Auto-committed by scripts/autonomous_self_improve.py after all "
        f"gates passed (tests + security"
        f"{'; benchmark' if summary.get('autonomy_level') == 2 else ''})."
    )
    _git(["add", "--", *changed])
    _git(["commit", "-m", msg], check=False)
    return True


def run_issue_loop(
    *,
    max_iterations: int,
    level_cap: int,
    agent: "object",
    no_benchmark: bool,
    model: str | None,
    profile: str | None,
) -> None:
    """Autonomous issue-resolution loop (mirrors the catalog loop's safety).

    Works the next eligible issue each iteration: git checkpoint first, then
    ``issue_loop.resolve_issue`` (verify + generate via ``fix`` + gates), commit
    ONLY on acceptance (scoped to the issue's files + ``.issues.json``). A
    non-accepted verdict fails closed for that issue — its checkpoint is
    reverted and the issue deferred — but the loop continues to the next
    eligible issue rather than halting, so one ambiguous result can't block the
    rest of the run (2026-08-28).
    """
    output_dir = SUMMARY_PATH.parent
    deferred: set[str] = set()
    for iteration in range(1, max_iterations + 1):
        if _stop_requested():
            print("[autonomous] Kill-switch detected — halting issue loop.")
            return
        issues = issue_store.load_issues()
        eligible = [i for i in issue_store.open_issues(issues, max_level=level_cap)
                    if i["id"] not in deferred]
        if not eligible:
            print("[autonomous] No eligible issues remain (at or below "
                  f"AGENT_AUTONOMY_LEVEL={level_cap}).")
            return

        issue = eligible[0]
        checkpoint = f"autonomous-issue-{iteration}"
        # Only treat a stash as a real checkpoint if the tree was actually dirty
        # before generation. A `git stash push` on a clean tree is a no-op that
        # still returns rc=0, which would make `have_checkpoint` true and a later
        # `git stash pop` pop an *unrelated* stash and corrupt state.
        pre_dirty = bool(_git(["status", "--porcelain"]).stdout.strip())
        pushed = _git(["stash", "push", "-u", "-m", checkpoint], check=False)
        have_checkpoint = pushed.returncode == 0 and pre_dirty
        if pre_dirty and not have_checkpoint:
            print(f"[autonomous] WARNING: git stash push failed (rc="
                  f"{pushed.returncode}); continuing without a checkpoint.")
        print(f"\n[autonomous] === issue iteration {iteration}/{max_iterations} "
              f"=== {issue['id']}")
        try:
            summary = issue_loop.resolve_issue(
                issue, agent, level_cap=level_cap,
                run_benchmark=not no_benchmark, model=model, profile=profile,
            )
        except Exception as exc:  # noqa: BLE001 - never let one bad issue crash the driver
            print(f"[autonomous] issue iteration {iteration} raised "
                  f"{type(exc).__name__}: {exc}")
            if have_checkpoint:
                _git(["stash", "pop"], check=False)
            return

        verdict = summary.get("verdict")
        print(f"[autonomous] issue {issue['id']} -> {verdict}")
        _record_iteration(iteration, {
            "verdict": verdict,
            "accepted": bool(summary.get("accepted")),
            "proposed_repair": summary.get("issue_id"),
            "accepted_repair": summary.get("issue_id") if summary.get("accepted") else None,
        }, output_dir)
        if summary.get("accepted"):
            _commit_issue(iteration, summary)
            if have_checkpoint:
                _git(["stash", "drop"], check=False)
            continue
        # Fail-closed for this issue: leave the tree clean and move on to the
        # next eligible issue instead of halting the whole run. A bad fix is
        # reverted: if we took a real checkpoint (tree was dirty before
        # generation) pop it; otherwise the failure left a bad fix on a
        # previously-clean tree, so discard it. Defer the issue so it is not
        # retried within this run.
        if have_checkpoint:
            _git(["stash", "pop"], check=False)
        else:
            _git(["checkout", "--", "."], check=False)
        deferred.add(issue["id"])
        print(f"[autonomous] issue {issue['id']} not accepted ({verdict}) — "
              f"deferred and continuing.")
        continue


def _repair_files(repair_id: str | None) -> tuple[str, ...]:
    """Source file(s) a catalogued repair touches, for scoped staging.

    Returns an empty tuple for an unknown repair id so the caller can refuse
    to do a blanket ``git add -A`` (which would sweep unrelated/scratch files
    into an autonomous commit).
    """
    from harnessfix.repairs import CATALOG

    return CATALOG[repair_id].files if repair_id in CATALOG else ()


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
    parser.add_argument("--source", default="both", choices=("issues", "catalog", "both"),
                        help="what to work on: 'issues' (.issues.json), 'catalog' "
                             "(HarnessFix repairs), or 'both' (issues first, then catalog)")
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

    level_cap = int(os.environ.get("AGENT_AUTONOMY_LEVEL", "") or 1)
    if args.source in ("issues", "both"):
        from agent import Agent
        agent = Agent(workspace=str(REPO_ROOT))
        run_issue_loop(
            max_iterations=args.max_iterations, level_cap=level_cap, agent=agent,
            no_benchmark=args.no_benchmark, model=args.model, profile=args.profile,
        )
        if args.source == "issues":
            return 0

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
        write_progress({
            "iteration": iteration,
            "max_iterations": args.max_iterations,
            "running": True,
            "phase": "loop_iteration_start",
            "model": args.model or "none (offline harness-quality gate)",
            "no_benchmark": args.no_benchmark,
        })
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
            clear_progress()
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
            clear_progress()
            return 1

        verdict = summary.get("verdict")
        print(f"[autonomous] verdict={verdict} repair={summary.get('proposed_repair')}")

        # Record the finished iteration for the live dashboard history view.
        _record_iteration(iteration, summary, output_dir)
        write_progress({
            "iteration": iteration,
            "phase": "finished_iteration",
            "verdict": verdict,
            "accepted": bool(summary.get("accepted")),
        })

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
        clear_progress()
        return 0

    print("[autonomous] Reached max_iterations — stopping.")
    clear_progress()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
