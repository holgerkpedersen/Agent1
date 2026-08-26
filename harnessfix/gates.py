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

_GATE_TIMEOUT = 1800


def run_test_gate() -> tuple[bool, str]:
    """Run the full pytest suite; True iff every test passes."""
    try:
        proc = subprocess.run(
            [
                sys.executable, "-m", "pytest", "-q",
                "--no-cov", "-p", "no:cacheprovider", "--no-header",
            ],
            capture_output=True,
            text=True,
            timeout=_GATE_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"test gate timed out after {_GATE_TIMEOUT}s: {exc}"
    except OSError as exc:
        return False, f"test gate failed to start: {exc}"
    summary = (proc.stdout or "").strip().splitlines()
    tail = " | ".join(summary[-3:]) if summary else (proc.stderr or "")[-300:]
    return proc.returncode == 0, tail


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
