"""Benchmark tests for security primitives: path traversal, injection, and allow-list validation."""

from __future__ import annotations

import itertools
import pathlib
import statistics
import timeit
from typing import List, Tuple

from agent_core.security.allowlist import is_command_allowed
from agent_core.security.path_utils import (
    SecurityViolationError,
    normalize_path,
)
from agent_core.tools.file_ops import read_file, write_file
from agent_core.tools.shell_ops import run_command


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ITERATIONS = 10_000

_ATTACK_PATHS: List[str] = [
    "../../../etc/passwd",
    "/etc/shadow",
    "..\\..\\..\\windows\\system32\\config\\sam",
    "normal_file.txt",
    "../sibling/other.txt",
    "subdir/../subdir/file.txt",
    "a/b/c/d/e/f/g/h/i/j/target.txt",
]

_ATTACK_COMMANDS: List[str] = [
    "; cat /etc/passwd",
    "| ls -la",
    "$(whoami)",
    "`id`",
    "&& rm -rf /",
    "echo hello",
    "ls",
]

_ALLOWED_BINARIES: List[str] = ["ls", "cat", "grep", "python3", "git"]
_DENIED_BINARIES: List[str] = ["/bin/sh", "bash", "rm", "chmod", "sudo"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_workspace_root = pathlib.Path("/tmp/benchmark_workspace")


def _ensure_workspace() -> None:
    """Create a minimal workspace so that file_ops / shell_ops have valid roots."""
    _workspace_root.mkdir(parents=True, exist_ok=True)
    (_workspace_root / "dummy.txt").write_text("benchmark payload\n")


def _mean_seconds(timer: timeit.Timer) -> float:
    return statistics.mean(timer.repeat(repeat=3, number=_ITERATIONS)) / _ITERATIONS  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_path_normalization() -> Tuple[str, float]:
    """Benchmark normalize_path against various traversal payloads."""
    timer = timeit.Timer(
        stmt=lambda: list(
            normalize_path(_workspace_root, p) for p in _ATTACK_PATHS
        ),
    )
    return "path_normalization", _mean_seconds(timer)


def bench_allowlist_check() -> Tuple[str, float]:
    """Benchmark is_command_allowed against allowed and denied binaries."""
    all_bins = _ALLOWED_BINARIES + _DENIED_BINARIES
    timer = timeit.Timer(
        stmt=lambda: list(is_command_allowed(b) for b in all_bins),
    )
    return "allowlist_check", _mean_seconds(timer)


def bench_file_read_safe() -> Tuple[str, float]:
    """Benchmark read_file with a valid target inside the workspace."""
    timer = timeit.Timer(
        stmt=lambda: read_file(_workspace_root, "dummy.txt"),
    )
    return "file_read_safe", _mean_seconds(timer)


def bench_shell_run_safe() -> Tuple[str, float]:
    """Benchmark run_command with a benign command inside the workspace."""
    timer = timeit.Timer(
        stmt=lambda: run_command(_workspace_root, "echo benchmark"),
    )
    return "shell_run_safe", _mean_seconds(timer)


def bench_path_traversal_protection() -> Tuple[str, float]:
    """Benchmark that normalize_path raises SecurityViolationError on escapes."""
    escape_paths = [p for p in _ATTACK_PATHS if ".." in p or p.startswith("/") and not p.startswith(_workspace_root.as_posix())]
    # We only benchmark the ones that are expected to raise.
    def _run() -> None:
        for p in escape_paths[:3]:  # limit scope for speed
            try:
                normalize_path(_workspace_root, p)
            except SecurityViolationError:
                pass

    timer = timeit.Timer(stmt=_run)
    return "path_traversal_protection", _mean_seconds(timer)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_benchmarks() -> List[Tuple[str, float]]:
    """Execute every security benchmark and return (name, mean seconds)."""
    _ensure_workspace()
    results: List[Tuple[str, float]] = []

    for fn in (
        bench_path_normalization,
        bench_allowlist_check,
        bench_file_read_safe,
        bench_shell_run_safe,
        bench_path_traversal_protection,
    ):
        name, secs = fn()
        results.append((name, secs))
        print(f"{name:<35} {secs:.6f}s  (over {_ITERATIONS} iterations)")

    return results


def main() -> None:
    """Entry-point for CLI invocation."""
    run_all_benchmarks()


if __name__ == "__main__":
    main()