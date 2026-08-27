"""Regression tests for the self-improvement driver's 'why is this better'
rationale (scripts.autonomous_self_improve.build_repair_rationale).

These guard against a repair being committed with no stated justification
(see the stuck-repeat-tool-hints repair, which landed with
`baseline_rate=None post_rate=None` and later got reverted with no recorded
reason).  The rationale builder is pure (no git/IO), so it is unit-tested in
isolation.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "autonomous_self_improve_mod",
    REPO_ROOT / "scripts" / "autonomous_self_improve.py",
)
autonomous_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(autonomous_mod)

from harnessfix.repairs import CATALOG  # noqa: E402
from harnessfix.repairs.stuck_repeat import (  # noqa: E402
    STUCK_REPEAT_REPAIR_ID,
)
from harnessfix.repairs.tool_interface import (  # noqa: E402
    TOOL_INTERFACE_REPAIR_ID,
)

build_repair_rationale = autonomous_mod.build_repair_rationale


def test_rationale_includes_repair_description():
    summary: dict = {
        "harness_baseline": None,
        "harness_post": None,
        "baseline_rate": None,
        "post_rate": None,
    }
    out = build_repair_rationale(STUCK_REPEAT_REPAIR_ID, summary)
    # The human-readable repair description must appear.
    assert CATALOG[STUCK_REPEAT_REPAIR_ID].description in out


def test_rationale_surfaces_corpus_layer_evidence():
    repair = CATALOG[STUCK_REPEAT_REPAIR_ID]
    summary: dict = {
        "harness_baseline": {
            "total": 42,
            "success_rate": 0.5,
            "layer_counts": {repair.layer: 17},
        },
        "harness_post": {
            "total": 42,
            "success_rate": 0.5,
            "layer_counts": {repair.layer: 17},
        },
        "baseline_rate": None,
        "post_rate": None,
    }
    out = build_repair_rationale(STUCK_REPEAT_ID := STUCK_REPEAT_REPAIR_ID, summary)
    assert "42 trace(s)" in out
    assert f"Target layer '{repair.layer}' is evidenced in 17 failed" in out
    # Offline gate is the primary signal when no benchmark model is set.
    assert "not run (offline harness-quality gate" in out


def test_rationale_reports_benchmark_cross_check_when_present():
    summary: dict = {
        "harness_baseline": {"total": 0},
        "harness_post": {"total": 0},
        "baseline_rate": 0.61,
        "post_rate": 0.73,
    }
    out = build_repair_rationale(TOOL_INTERFACE_REPAIR_ID, summary)
    assert "LLM benchmark cross-check: baseline=0.61% -> post=0.73%" in out
    assert "no regression" in out


def test_rationale_handles_unknown_repair_id():
    summary: dict = {
        "harness_baseline": None,
        "harness_post": None,
        "baseline_rate": None,
        "post_rate": None,
    }
    out = build_repair_rationale("does-not-exist", summary)
    # No repair description line, but it must still explain the evidence state.
    assert "Corpus evidence:" in out
    assert "no trace corpus available" in out
