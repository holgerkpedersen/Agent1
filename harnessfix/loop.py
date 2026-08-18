"""Phase 4 - the HarnessFix closed loop (spec section 3.5).

1. Collect traces under reports/traces/ (or --traces DIR).
2. Compile each failed trace to HTIR and diagnose (reports/harnessfix/).
3. Propose the catalog repair of the highest-frequency diagnosed layer.
4. Human review gate (fail-closed headless; --approve to proceed).
5. Apply, run gates (pytest + security + optional benchmark pass-rate).
6. Accept iff tests+security pass and the benchmark did not regress.
7. Write reports/harnessfix/summary.json with the verdict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import gates
from .corpus import choose_repair, collect_traces, diagnose_corpus, layer_counts
from .repairs import collisions
from .tracing import TRACE_DIR

SUMMARY_PATH = Path("reports") / "harnessfix" / "summary.json"


def run_loop(
    trace_dir: Path,
    *,
    approve: bool,
    model: str | None,
    profile: str | None = None,
    output_dir: Path = SUMMARY_PATH.parent,
) -> dict[str, Any]:
    """One HarnessFix iteration; headless-safe (fail-closed without approval)."""
    traces = collect_traces(trace_dir)
    if not traces:
        raise SystemExit(f"no traces found under {trace_dir}")

    diagnoses = diagnose_corpus(traces, output_dir / "diagnoses")
    counts = layer_counts(diagnoses)
    repair = choose_repair(counts)
    summary: dict[str, Any] = {
        "trace_dir": str(trace_dir),
        "traces": len(traces),
        "diagnosed": len(diagnoses),
        "layer_distribution": dict(counts),
        "proposed_repair": repair.id if repair else None,
        "verdict": "no_repair_catalogued" if repair is None else "review_required",
    }
    if repair is None:
        return _finish(summary, output_dir)
    if not approve:
        summary["verdict"] = "review_required_fail_closed"
        return _finish(summary, output_dir)

    # String-collision guard: a repair that rewrites a runtime string must
    # not break test assertions pinning the old string.  Any hit skips the
    # repair (fail-safe; recorded for the human gate) instead of burning a
    # full gate run and reverting afterwards.  The guard's OWN fixture tests
    # (GUARD_TEST_FILENAMES) are excluded: they contain the fragments as
    # literals, not as pins of the OLD runtime string (decision #051).
    all_hits = collisions.find_test_collisions(
        repair.collision_fragments,
        tests_dir=collisions.DEFAULT_TESTS_DIR,
        exclude_files=frozenset(),
    )
    guard_test_names = collisions.GUARD_TEST_FILENAMES
    guard_hits = [h for h in all_hits if h.path.name in guard_test_names]
    collisions_hits = [h for h in all_hits if h.path.name not in guard_test_names]
    if guard_hits:
        summary["ignored_guard_test_hits"] = len(guard_hits)
    if collisions_hits:
        summary["verdict"] = "skipped_test_collision"
        summary["collisions"] = [c.to_dict() for c in collisions_hits]
        return _finish(summary, output_dir)

    baseline_rate = gates.run_benchmark_gate(model, profile)
    try:
        summary["repair_applied"] = repair.applied_summary()
    except Exception as exc:  # apply/revert must never corrupt the tree silently
        summary["verdict"] = "apply_failed"
        summary["error"] = str(exc)
        return _finish(summary, output_dir)

    tests_passed, tests_tail = gates.run_test_gate()
    security_passed, security_tail = gates.run_security_gate()
    post_rate = gates.run_benchmark_gate(model, profile)
    accepted = gates.should_accept(tests_passed, security_passed, baseline_rate, post_rate)
    summary.update(
        tests_passed=tests_passed,
        tests_tail=tests_tail,
        security_passed=security_passed,
        security_tail=security_tail,
        baseline_rate=baseline_rate,
        post_rate=post_rate,
        accepted=accepted,
    )
    if not accepted:
        try:
            repair.revert()
            summary["verdict"] = "rejected_and_reverted"
        except Exception as exc:  # pragma: no cover - defensive
            summary["verdict"] = "revert_failed"
            summary["error"] = str(exc)
    else:
        summary["verdict"] = "accepted"
    return _finish(summary, output_dir)


def _finish(summary: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnessfix.loop",
        description="HarnessFix closed loop: diagnose traces, apply scoped repair, verify gates.",
    )
    parser.add_argument("--traces", type=Path, default=TRACE_DIR, help="trace corpus dir")
    parser.add_argument("--approve", action="store_true", help="human review gate: approve the proposed repair")
    parser.add_argument("--model", default=None, help="benchmark gate model (needs a live API)")
    parser.add_argument("--profile", default=None, help="benchmark profile tag (decision #055)")
    parser.add_argument("--output", type=Path, default=SUMMARY_PATH.parent, help="reports/harnessfix output dir")
    args = parser.parse_args(argv)
    summary = run_loop(
        args.traces, approve=args.approve, model=args.model,
        profile=args.profile, output_dir=args.output,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
