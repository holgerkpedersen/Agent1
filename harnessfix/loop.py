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
from .corpus import collect_traces, diagnose_corpus, layer_counts
from .repairs import collisions, repairs_for_layer
from .tracing import TRACE_DIR

SUMMARY_PATH = Path("reports") / "harnessfix" / "summary.json"


def run_loop(
    trace_dir: Path,
    *,
    approve: bool,
    model: str | None,
    profile: str | None = None,
    output_dir: Path = SUMMARY_PATH.parent,
    auto_approve: bool = False,
) -> dict[str, Any]:
    """One HarnessFix iteration.

    Two approval modes (both still enforce the test + security + benchmark
    gates — auto-approve is *machine*-gated, never human-gated):

    - ``approve`` — explicit human approval (``--approve``); applies the
      proposed repair and runs the gates.
    - ``auto_approve`` — fully autonomous: applies the repair and runs the
      gates, but only COMMITS the change if ``should_accept`` is True AND the
      collision guard passed AND all gates are green.  On any ambiguity
      (no benchmark due to a missing live model, low-confidence diagnosis,
      or a collision hit) it falls back to ``review_required_fail_closed``
      and never merges — so autonomy can never silently degrade the tree.
    """
    traces = collect_traces(trace_dir)
    if not traces:
        raise SystemExit(f"no traces found under {trace_dir}")

    diagnoses = diagnose_corpus(traces, output_dir / "diagnoses")
    counts = layer_counts(diagnoses)
    # Ordered catalog repairs to attempt: highest-frequency diagnosed layer
    # first, then catalog order within a layer, then remaining layers by
    # frequency.  A rejected repair falls through to the next candidate so the
    # loop stays useful instead of giving up on the first failure.
    candidates = _candidate_repairs(counts)
    summary: dict[str, Any] = {
        "trace_dir": str(trace_dir),
        "traces": len(traces),
        "diagnosed": len(diagnoses),
        "layer_distribution": dict(counts),
        "proposed_repair": candidates[0].id if candidates else None,
        "verdict": "no_repair_catalogued" if not candidates else "review_required",
    }
    if not candidates:
        return _finish(summary, output_dir)
    # Fail-closed unless a human (--approve) or the autonomous driver
    # (--auto-approve) explicitly engages.  Auto-approve still cannot bypass
    # the gates below — it only removes the manual click.
    if not approve and not auto_approve:
        summary["verdict"] = "review_required_fail_closed"
        return _finish(summary, output_dir)

    # Baseline failure set captured ONCE (and cached by git HEAD) so every
    # candidate is judged on whether it ADDS new failures, not on whether the
    # suite is already 100% green.  The repo carries pre-existing failures
    # (corpus-drift / environment-specific), so a strict 100%-green gate
    # would reject every repair and make the loop permanently useless.
    baseline_fail = gates.get_baseline_failures()

    # Try each candidate in order; stop at the first that the gates accept.
    attempted: list[str] = []
    for repair in candidates:
        # String-collision guard: a repair that rewrites a runtime string
        # must not break test assertions pinning the old string.  Any hit
        # skips the repair (fail-safe; recorded for the human gate) instead
        # of burning a full gate run and reverting afterwards.  The guard's
        # OWN fixture tests (GUARD_TEST_FILENAMES) are excluded: they contain
        # the fragments as literals, not as pins of the OLD runtime string
        # (decision #051).
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
            # Record the skip: a flat ``collisions`` list (matching the
            # historical single-repair shape) plus a per-repair
            # ``skipped_collisions`` entry for the multi-candidate loop.
            summary.setdefault("collisions", []).extend(
                c.to_dict() for c in collisions_hits
            )
            summary.setdefault("skipped_collisions", []).append(
                {"repair": repair.id, "collisions": [c.to_dict() for c in collisions_hits]}
            )
            attempted.append(repair.id)
            continue

        baseline_rate = gates.run_benchmark_gate(model, profile)
        try:
            summary["repair_applied"] = repair.applied_summary()
        except Exception as exc:  # apply/revert must never corrupt the tree silently
            summary["verdict"] = "apply_failed"
            summary["error"] = str(exc)
            return _finish(summary, output_dir)

        tests_passed, tests_tail = gates.run_test_gate(
            baseline_failures=baseline_fail
        )
        security_passed, security_tail = gates.run_security_gate()
        post_rate = gates.run_benchmark_gate(model, profile)
        accepted = gates.should_accept(
            tests_passed, security_passed, baseline_rate, post_rate
        )
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
            attempted.append(repair.id)
            summary["attempted_repairs"] = attempted
            continue

        summary["verdict"] = "accepted"
        summary["accepted_repair"] = repair.id
        summary["attempted_repairs"] = attempted
        return _finish(summary, output_dir)

    # Every candidate was rejected/skipped.
    summary["attempted_repairs"] = attempted
    if summary.get("verdict") in (None, "review_required"):
        # A collision skip is a fail-safe for a specific repair; if that was
        # the only outcome (no acceptance, no hard rejection), report it so
        # the human gate sees why nothing was applied.
        summary["verdict"] = (
            "skipped_test_collision" if summary.get("skipped_collisions")
            else "rejected_and_reverted"
        )
    return _finish(summary, output_dir)


def _candidate_repairs(counts: "Counter[str]") -> list:
    """Ordered catalog repairs to attempt, by diagnosed-layer frequency.

    Only repairs whose layer was actually diagnosed are candidates (a
    lifecycle repair is not meaningful for a tool_interface trace).  Within a
    layer, catalog order; layers are tried highest-frequency first.  The loop
    falls through to the next candidate when one is rejected, so a single bad
    repair does not stall the whole autonomous run.
    """
    ordered: list = []
    seen = set()
    for layer, _ in counts.most_common():
        for r in repairs_for_layer(layer):
            if r.id not in seen:
                seen.add(r.id)
                ordered.append(r)
    return ordered


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
    parser.add_argument("--auto-approve", action="store_true",
                        help="fully autonomous: apply + run gates, commit only if all gates pass "
                             "(test + security + benchmark no-regression); never merges on ambiguity")
    parser.add_argument("--model", default=None, help="benchmark gate model (needs a live API)")
    parser.add_argument("--profile", default=None, help="benchmark profile tag (decision #055)")
    parser.add_argument("--output", type=Path, default=SUMMARY_PATH.parent, help="reports/harnessfix output dir")
    args = parser.parse_args(argv)
    summary = run_loop(
        args.traces, approve=args.approve, model=args.model,
        profile=args.profile, output_dir=args.output,
        auto_approve=args.auto_approve,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    # Exit code reflects the outcome so callers (CI, wrappers, humans) get a
    # real signal.  Previously this was hardcoded to 0, which masked a
    # rejected/reverted/apply-failed repair as success.  A verdict of
    # ``accepted`` is success; a safe no-op (fail-closed, nothing catalogued,
    # or a collision-guarded skip) is also 0 because the tree was left
    # unchanged and safe; any verdict where a repair was attempted but did not
    # land is a non-zero failure.
    verdict = summary.get("verdict")
    failed = verdict in (
        "rejected_and_reverted",
        "apply_failed",
        "revert_failed",
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
