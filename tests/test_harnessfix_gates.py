"""Regression tests: the verification gates must never raise into the caller.

A gate that raises (e.g. the benchmark subprocess exceeding the gate timeout)
would propagate out of run_loop and crash the autonomous driver with a dirty
tree.  Gates are meant to be non-blocking fallbacks, so on any failure they
must return a safe value (False for test/security, None for benchmark).
"""
import subprocess
import sys

sys.path.insert(0, ".")

import harnessfix.gates as gates


def test_benchmark_gate_returns_none_on_timeout(monkeypatch):
    """A benchmark subprocess that exceeds the timeout must yield None, not
    raise subprocess.TimeoutExpired into run_loop."""

    def _boom(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="benchmark.py", timeout=1)

    monkeypatch.setattr(subprocess, "run", _boom)
    assert gates.run_benchmark_gate("some-model", None) is None


def test_test_gate_returns_false_on_timeout(monkeypatch):
    """The pytest gate must return (False, tail) on timeout, never raise."""

    def _boom(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="pytest", timeout=1)

    monkeypatch.setattr(subprocess, "run", _boom)
    passed, tail = gates.run_test_gate()
    assert passed is False
    assert "timed out" in tail.lower()


def test_test_gate_returns_false_on_oserror(monkeypatch):
    def _boom(*_a, **_k):
        raise OSError("no python")

    monkeypatch.setattr(subprocess, "run", _boom)
    passed, tail = gates.run_test_gate()
    assert passed is False
    assert "failed to start" in tail.lower()
