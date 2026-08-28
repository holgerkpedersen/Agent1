"""Phase 4 - verification gates for the HarnessFix closed loop.

A repair is accepted only if it passes ALL enabled gates: the pytest suite,
the security-primitive checks, and (when a model is supplied) the benchmark
pass-rate gate.  Gate and scorer code is never modified by a repair run.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .corpus_quality import CorpusQuality, corpus_quality

_GATE_TIMEOUT = 1800

#: Baseline failure cache (regression-aware gate).  Keyed by git HEAD so a
#: changed tree invalidates it; see :func:`get_baseline_failures`.
_BASELINE_CACHE = Path("reports") / "harnessfix" / "baseline_failures.json"


def _git_head() -> str | None:
    """Current git HEAD sha, or None if git is unavailable / not a repo."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def collect_test_failures() -> tuple[bool, frozenset[str], str]:
    """Run the full pytest suite; return (ran_ok, failed_node_ids, tail).

    ``ran_ok`` is True when pytest actually executed (even if some tests
    failed); it is False on a crash/timeout so callers can fall back to strict
    pass/fail.  ``failed_node_ids`` is the set of ``path::Class::test`` node
    ids that FAILED, used for regression comparison: a repair is safe when it
    adds no NEW failures, even if the suite already carries pre-existing
    (environment-specific or corpus-drift) failures.

    The repo's ``pyproject.toml`` injects ``--cov=...`` via ``addopts``, but
    ``pytest-cov`` is an optional dependency that is not always present.  When
    it is missing, pytest rejects BOTH the ``--cov`` flags and the ``--no-cov``
    override, so the gate fails to *start*.  We neutralize ``addopts``
    (``-o addopts=``) so the suite runs with whatever plugins are installed.
    """
    try:
        proc = subprocess.run(
            [
                sys.executable, "-m", "pytest", "-q",
                "-o", "addopts=", "-p", "no:cacheprovider", "--no-header",
                "--tb=no", "-rf",
                # HarnessFix loop/repair self-tests (marker `harnessfix_self_test`)
                # mutate agent_core/llm/tool_loop.py and would revert an applied
                # repair mid-run; excluding them lets the gate validate the real
                # repaired functional suite while the change is actually present.
                "-m", "not harnessfix_self_test",
            ],
            capture_output=True,
            text=True,
            timeout=_GATE_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        return False, frozenset(), f"test gate timed out after {_GATE_TIMEOUT}s: {exc}"
    except OSError as exc:
        return False, frozenset(), f"test gate failed to start: {exc}"
    # pytest exited (returncode 0 or nonzero) -> the run completed.
    ran_ok = proc.returncode is not None
    failed: set[str] = set()
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("FAILED "):
            # "-rf" emits "FAILED <nodeid>" lines (no trailing reason).
            failed.add(line[len("FAILED "):].strip())
    summary = (proc.stdout or "").strip().splitlines()
    tail = " | ".join(summary[-3:]) if summary else (proc.stderr or "")[-300:]
    return ran_ok, frozenset(failed), tail


def get_baseline_failures(force: bool = False) -> frozenset[str]:
    """Return the set of failing test node ids on the *clean* tree.

    Computed once and cached to ``baseline_failures.json`` keyed by git HEAD,
    so a changed tree invalidates the cache.  Used by the regression-aware
    test gate: a repair is judged on whether it ADDS new failures, not on
    whether the suite is already 100% green (it frequently is not, due to
    pre-existing or environment-specific failures).  On a crash/timeout the
    cache is not written and an empty baseline is returned, which degrades the
    gate to strict (any failure rejects) rather than silently accepting all.
    """
    head = _git_head()
    if not force and _BASELINE_CACHE.is_file():
        try:
            data = json.loads(_BASELINE_CACHE.read_text(encoding="utf-8"))
            if data.get("git_head") == head and isinstance(data.get("failures"), list):
                return frozenset(data["failures"])
        except (ValueError, OSError):
            pass
    ran_ok, failures, _ = collect_test_failures()
    if ran_ok:
        try:
            _BASELINE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            _BASELINE_CACHE.write_text(
                json.dumps(
                    {"git_head": head, "failures": sorted(failures)}, indent=2
                ),
                encoding="utf-8",
            )
        except OSError:
            pass
        return failures
    return frozenset()


def run_test_gate(
    baseline_failures: frozenset[str] | None = None,
) -> tuple[bool, str]:
    """Run the full pytest suite; True iff the repair is safe to accept.

    Without ``baseline_failures`` (the historical behavior, used by the gate
    unit tests and the fail-closed path) the gate is strict: True iff every
    test passes.  When a baseline failure set is supplied, the gate is
    *regression-aware*: True iff the post-repair run introduces no NEW
    failures beyond that baseline — so a repair is not rejected merely because
    the suite was already red with pre-existing failures.
    """
    ran_ok, post_fail, tail = collect_test_failures()
    if baseline_failures is None:
        # Strict mode (historical): the suite must be fully green.
        return ran_ok and not post_fail, tail
    # Regression mode: accept iff no new failures vs the baseline.
    return ran_ok and post_fail <= baseline_failures, tail


def run_security_gate() -> tuple[bool, str]:
    """Fast correctness checks on security primitives (not the perf benchmark)."""
    from pathlib import Path as _P

    from agent_core.security.allowlist import is_command_allowed
    from agent_core.security.path_utils import (
        SecurityViolationError,
        normalize_path,
    )

    try:
        if not is_command_allowed("ls"):
            return False, "allowlist rejected an allowed binary (ls)"
        if is_command_allowed("rm") or is_command_allowed("bash"):
            return False, "allowlist allowed a denied binary"
        try:
            normalize_path(_P.cwd(), "../../../etc/passwd")
        except SecurityViolationError:
            pass
        else:
            return False, "path escape was not blocked"
        return True, "security gate passed"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"security gate error: {exc}"


def _benchmark_key(model: str, profile: str = "") -> str:
    """Benchmark result key.  Benchmarks are per MODEL, and per PROFILE when
    a profile is set: the same model under deep-analysis vs fast-codegen
    behaves differently, and the gate must compare like with like
    (decision #055)."""
    return f"{model}|{profile}" if profile else model


def run_benchmark_gate(
    model: str | None, profile: str | None = None
) -> float | None:
    """Single-shot benchmark pass rate (percent) via benchmark.py, or None."""
    if not model:
        return None
    out = Path("reports") / "benchmark_harnessfix.json"
    key = _benchmark_key(model, profile or "")
    cmd = [sys.executable, "benchmark.py", "--model", model,
           "--output", str(out), "--repetitions", "1"]
    if profile:
        cmd += ["--profile", profile]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_GATE_TIMEOUT,
        )
        if proc.returncode != 0:
            return None
        with out.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        # benchmark.py's --output file stores models as a LIST of records
        # (save_json_report), each carrying display_name = model|profile.
        # models.json (save_models_json) stores a dict keyed the same way.
        # Support both; never crash on unexpected shapes.
        models: Any = data.get("models")
        accuracy: Any = None
        if isinstance(models, list):
            for m in models:
                if isinstance(m, dict) and m.get("display_name") == key:
                    accuracy = m.get("overall_accuracy")
                    break
        elif isinstance(models, dict):
            entry = models.get(key)
            if isinstance(entry, dict):
                accuracy = entry.get("overall_accuracy")
        return float(accuracy) if accuracy is not None else None
    except (OSError, json.JSONDecodeError, KeyError, ValueError, AttributeError,
            TypeError, subprocess.TimeoutExpired):
        return None


def should_accept(
    tests_passed: bool,
    security_passed: bool,
    baseline_rate: float | None,
    post_rate: float | None,
    regression_tolerance: float = 0.0,
) -> bool:
    """Accept iff tests+security pass and the benchmark did not regress.

    When the benchmark gate is unavailable (baseline None), the benchmark is
    not a blocking criterion; the test and security gates always are.
    """
    if not (tests_passed and security_passed):
        return False
    if baseline_rate is None or post_rate is None:
        return True
    return post_rate >= baseline_rate - regression_tolerance


def run_harness_quality_gate(trace_dir: str | Path) -> CorpusQuality | None:
    """Offline, harness-centric quality snapshot of the trace corpus, or None.

    Returns ``None`` only on a genuinely empty/unusable corpus (no readable
    traces), so the caller can degrade to a non-blocking fallback rather than
    rejecting a repair on missing evidence.  Any other error is swallowed and
    returns ``None`` so a gate never raises into the autonomous driver.
    """
    try:
        quality = corpus_quality(trace_dir)
    except Exception:  # noqa: BLE001 - a gate must never raise into run_loop
        return None
    if quality is None or quality.total == 0:
        return None
    return quality


def should_accept_harness(
    baseline: CorpusQuality | None,
    post: CorpusQuality | None,
    target_mechanism: str | None = None,
    target_layer: str | None = None,
    success_rate_tolerance: float = 0.0,
) -> bool:
    """Accept iff the harness repair is *targeted* to the corpus failures.

    This is the *primary* quality gate when no live model is available for the
    LLM benchmark.  The trace corpus is a static record of past runs, so the
    baseline and post snapshots are structurally identical; the gate therefore
    validates **target alignment**, not a pre/post delta:

    - If the repair declares a ``target_layer``, that layer MUST appear in the
      corpus's observed failures.  A repair whose layer is absent from the
      corpus is off-target (it would "fix" a failure mode the evidence does
      not show) and is rejected — this is the core "targeted" requirement.
    - ``success_rate`` must not drop below the baseline (minus a small
      tolerance).  Retained as a guard: a corrupted/empty post snapshot that
      still reports the target layer would otherwise pass; this rejects a
      post run whose completion rate fell.

    Returns ``True`` when the gate is unavailable (no baseline/post evidence),
    so it is non-blocking in the same fail-open way as the benchmark gate;
    the test + security gates remain mandatory.
    """
    if baseline is None or post is None:
        return True
    # Guard: a post snapshot whose completion rate fell vs the baseline is
    # treated as a degraded/empty corpus and rejected rather than trusted.
    if post.success_rate < baseline.success_rate - success_rate_tolerance:
        return False
    # Target alignment: the repair's layer must be evidenced in the corpus.
    if target_layer is not None:
        if baseline.layer_counts.get(target_layer, 0) == 0:
            return False
    return True
