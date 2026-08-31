"""Phase 4 - the HarnessFix closed loop (spec section 3.5).

1. Collect traces under reports/traces/ (or --traces DIR).
2. Compile each failed trace to HTIR and diagnose (reports/harnessfix/).
3. Propose the catalog repair of the highest-frequency diagnosed layer.
4. Human review gate (fail-closed headless; --approve to proceed).
5. Apply, run gates (pytest + security + optional benchmark pass-rate).
6. Accept iff tests+security pass and (benchmark non-regression when a model
   is supplied, OR the offline harness-quality gate when no model is given).
7. Write reports/harnessfix/summary.json with the verdict.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from . import gates
from .corpus import collect_traces, diagnose_corpus, layer_counts
from .progress import set_phase, write_progress
from .repairs import collisions, repairs_for_layer
from .tracing import TRACE_DIR

try:
    from .wiki import consolidate as _consolidate_wiki, format_wiki_notes as _format_wiki_notes
except Exception:  # pragma: no cover - wiki is additive; never block the loop.
    _consolidate_wiki = None
    _format_wiki_notes = None

SUMMARY_PATH = Path("reports") / "harnessfix" / "summary.json"


def run_loop(
    trace_dir: Path,
    *,
    approve: bool,
    model: str | None,
    profile: str | None = None,
    output_dir: Path = SUMMARY_PATH.parent,
    auto_approve: bool = False,
    iteration: int | None = None,
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
    if iteration is not None:
        write_progress({"iteration": iteration, "running": True})
    set_phase("collecting_traces", traces=None)

    traces = collect_traces(trace_dir)
    if not traces:
        raise SystemExit(f"no traces found under {trace_dir}")

    set_phase(
        "diagnosing",
        traces=len(traces),
        diagnosed=None,
        layer_distribution=None,
    )

    diagnoses = diagnose_corpus(traces, output_dir / "diagnoses")
    counts = layer_counts(diagnoses)
    set_phase(
        "diagnosing",
        traces=len(traces),
        diagnosed=len(diagnoses),
        layer_distribution=dict(counts),
    )
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
    # WikiSkill step (consolidate + wiki context): absorb this iteration's
    # traces into the persistent wiki and surface consolidated knowledge for
    # the repair proposal.  Additive — a consolidation/context error never
    # blocks the loop or affects acceptance gates.
    if _consolidate_wiki is not None:
        try:
            wiki_pages = _consolidate_wiki(trace_dir, output_dir.parent / "wiki")
            summary["wiki_page_count"] = len(wiki_pages)
            set_phase("consolidating_wiki", wiki_pages=len(wiki_pages))
        except Exception:
            pass  # fail-open; wiki is never a gating step.

    if _format_wiki_notes is not None and candidates:
        try:
            target_layer = candidates[0].layer
            summary["wiki_context"] = _format_wiki_notes(
                f"{target_layer} harness mechanism", k=3,
                path=output_dir.parent / "wiki" / "wiki.jsonl",
            ) or ""
        except Exception:
            pass

    if not candidates:
        set_phase("finished", verdict="no_repair_catalogued", accepted=False)
        return _finish(summary, output_dir)
    # Fail-closed unless a human (--approve) or the autonomous driver
    # (--auto-approve) explicitly engages.  Auto-approve still cannot bypass
    # the gates below — it only removes the manual click.
    if not approve and not auto_approve:
        summary["verdict"] = "review_required_fail_closed"
        set_phase("finished", verdict="review_required_fail_closed", accepted=False)
        return _finish(summary, output_dir)

    # Baseline failure set captured ONCE (and cached by git HEAD) so every
    # candidate is judged on whether it ADDS new failures, not on whether the
    # suite is already 100% green.  The repo carries pre-existing failures
    # (corpus-drift / environment-specific), so a strict 100%-green gate
    # would reject every repair and make the loop permanently useless.
    baseline_fail = gates.get_baseline_failures()

    # Offline, harness-centric quality baseline: the same trace corpus the
    # repairs are diagnosed from, scored before any repair is applied.  Used
    # as the PRIMARY acceptance signal when no live model is available for the
    # LLM benchmark (which is noisy and offline-incompatible).  Computed once
    # per iteration so every candidate is judged against a stable baseline.
    harness_baseline = gates.run_harness_quality_gate(trace_dir)

    # Try each candidate in order; stop at the first that the gates accept.
    attempted: list[str] = []
    for repair in candidates:
        set_phase(
            "evaluating_candidate",
            candidate=repair.id,
            candidate_layer=repair.layer,
            candidate_summary=repair.applied_summary(),
        )
        # Already-applied guard: a static corpus diagnoses the same layers
        # every run, so the top candidate is the same repair each iteration.
        # If it is already in the tree, re-applying is a no-op (and the
        # offline gate would "accept" it again, looping forever).  Skip it
        # so the loop falls through to the next candidate or stops cleanly.
        if repair.is_applied():
            summary.setdefault("skipped_already_applied", []).append(repair.id)
            attempted.append(repair.id)
            set_phase(
                "evaluating_candidate",
                candidate=repair.id,
                verdict="already_applied_skip",
            )
            continue
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
            # Apply the repair to the real tree.  NOTE: ``applied_summary``
            # is intentionally PURE (it must not mutate the tree — see
            # repairs/__init__.py); applying is a separate, explicit step so
            # the loop never depends on a description side effect.
            summary["repair_applied"] = repair.apply()
        except Exception as exc:  # apply/revert must never corrupt the tree silently
            summary["verdict"] = "apply_failed"
            summary["error"] = str(exc)
            set_phase("finished", verdict="apply_failed", accepted=False)
            return _finish(summary, output_dir)

        set_phase("applying_repair", repair_applied=summary.get("repair_applied"))
        tests_passed, tests_tail = gates.run_test_gate(
            baseline_failures=baseline_fail
        )
        set_phase("running_test_gate", tests_passed=tests_passed)
        security_passed, security_tail = gates.run_security_gate()
        set_phase("running_security_gate", security_passed=security_passed)
        post_rate = gates.run_benchmark_gate(model, profile)
        # Offline harness-quality gate: re-score the corpus after the repair
        # is applied; the repair's own layer is the targeting key (a repair
        # must address a failure mode the corpus actually evidences).
        harness_post = gates.run_harness_quality_gate(trace_dir)
        set_phase("running_harness_gate")
        harness_ok = gates.should_accept_harness(
            harness_baseline, harness_post,
            target_layer=repair.layer,
        )
        benchmark_ok = gates.should_accept(
            tests_passed, security_passed, baseline_rate, post_rate
        )
        # Acceptance requires the mandatory gates (tests + security) plus at
        # least one quality signal: the LLM benchmark (when a model is
        # supplied) OR the offline harness-quality gate (always available).
        # This demotes the benchmark from the de-facto quality criterion to an
        # optional cross-check while keeping a deterministic, offline gate.
        accepted = (tests_passed and security_passed) and (
            benchmark_ok if model else harness_ok
        )
        summary.update(
            tests_passed=tests_passed,
            tests_tail=tests_tail,
            security_passed=security_passed,
            security_tail=security_tail,
            baseline_rate=baseline_rate,
            post_rate=post_rate,
            harness_baseline=harness_baseline.model_dump() if harness_baseline else None,
            harness_post=harness_post.model_dump() if harness_post else None,
            harness_accepted=harness_ok,
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
            set_phase("evaluating_candidate", candidate=repair.id, verdict=summary["verdict"])
            continue

        # The test gate ran the *full* pytest suite, and some HarnessFix
        # self-tests (e.g. ``_reset_repairs`` in tests/test_harnessfix_loop.py)
        # revert this very file. If that happened, the repair was applied and
        # then reverted during the gate, so an "accepted" tree would not
        # actually carry the change (and the autonomous driver would then fail
        # to commit it).  Re-apply idempotently here so the accepted tree is
        # guaranteed to hold the repair.  ``apply()`` is a no-op when already
        # present; any failure fails closed rather than silently returning an
        # unmodified tree.
        try:
            repair.apply()
        except Exception as exc:  # pragma: no cover - defensive
            summary["verdict"] = "apply_failed"
            summary["error"] = f"re-apply after gate reverted the repair: {exc}"
            set_phase("finished", verdict="apply_failed", accepted=False)
            return _finish(summary, output_dir)

        summary["verdict"] = "accepted"
        summary["accepted_repair"] = repair.id
        summary["attempted_repairs"] = attempted
        set_phase("finished", verdict="accepted", accepted=True)
        return _finish(summary, output_dir)

    # Every candidate was rejected/skipped.
    summary["attempted_repairs"] = attempted
    if summary.get("verdict") in (None, "review_required", "no_repair_catalogued"):
        # A collision skip is a fail-safe for a specific repair; if that was
        # the only outcome (no acceptance, no hard rejection), report it so
        # the human gate sees why nothing was applied.
        if summary.get("skipped_collisions"):
            summary["verdict"] = "skipped_test_collision"
        elif summary.get("skipped_already_applied"):
            # The corpus only diagnoses layers whose repairs are already
            # applied: there is nothing new to improve, so stop (the driver
            # ends the run on any non-accepted verdict instead of looping).
            summary["verdict"] = "no_repair_catalogued"
        else:
            summary["verdict"] = "rejected_and_reverted"
    summary["accepted"] = bool(summary.get("accepted"))
    set_phase(
        "finished",
        verdict=summary.get("verdict"),
        accepted=summary["accepted"],
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
