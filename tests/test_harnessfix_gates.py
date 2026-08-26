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


def test_test_gate_strict_mode_requires_all_green(monkeypatch):
    """Without a baseline, the gate is strict: any failure rejects."""

    def _fake_run(*_a, **_k):
        out = (
            "FAILED tests/x.py::TestY::test_z\n"
            "1 failed, 9 passed in 0.10s\n"
        )
        return type("P", (), {"returncode": 1, "stdout": out, "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    passed, _ = gates.run_test_gate()
    assert passed is False


def test_test_gate_regression_mode_accepts_pre_existing_failures(monkeypatch):
    """With a baseline, the gate accepts when the repair adds NO new failures
    (pre-existing failures are tolerated), and rejects when it does."""

    def _fake_run(*_a, **_k):
        # Post-repair run: same single failure as the baseline, nothing new.
        out = (
            "FAILED tests/x.py::TestY::test_z\n"
            "1 failed, 9 passed in 0.10s\n"
        )
        return type("P", (), {"returncode": 1, "stdout": out, "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    baseline = frozenset({"tests/x.py::TestY::test_z"})
    passed, _ = gates.run_test_gate(baseline_failures=baseline)
    assert passed is True

    # A NEW failure beyond the baseline must be rejected.
    def _fake_run_new(*_a, **_k):
        out = (
            "FAILED tests/x.py::TestY::test_z\n"
            "FAILED tests/x.py::TestY::test_w\n"
            "2 failed, 8 passed in 0.10s\n"
        )
        return type("P", (), {"returncode": 1, "stdout": out, "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run_new)
    passed_new, _ = gates.run_test_gate(baseline_failures=baseline)
    assert passed_new is False


def test_collect_test_failures_parses_failed_nodes(monkeypatch):
    def _fake_run(*_a, **_k):
        out = (
            "FAILED tests/a.py::TestA::test_one\n"
            "FAILED tests/b.py::TestB::test_two\n"
            "2 failed, 3 passed in 0.10s\n"
        )
        return type("P", (), {"returncode": 1, "stdout": out, "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    ran_ok, failed, _ = gates.collect_test_failures()
    assert ran_ok is True
    assert failed == frozenset(
        {"tests/a.py::TestA::test_one", "tests/b.py::TestB::test_two"}
    )


def test_get_baseline_failures_caches_and_invalidates_on_head(monkeypatch, tmp_path):
    """get_baseline_failures runs pytest once, caches by git HEAD, and
    re-computes when the HEAD changes (so a dirty tree invalidates it)."""

    calls = {"n": 0}

    def _fake_run(*_a, **_k):
        calls["n"] += 1
        out = "FAILED tests/x.py::TestY::test_z\n1 failed, 9 passed in 0.10s\n"
        return type("P", (), {"returncode": 1, "stdout": out, "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(gates, "_BASELINE_CACHE", tmp_path / "baseline.json")
    monkeypatch.setattr(gates, "_git_head", lambda: "head-aaa")

    first = gates.get_baseline_failures()
    assert first == frozenset({"tests/x.py::TestY::test_z"})
    assert calls["n"] == 1  # computed once

    # Same HEAD -> served from cache, no re-run.
    second = gates.get_baseline_failures()
    assert second == first
    assert calls["n"] == 1

    # Different HEAD -> cache invalidated, re-computed.
    monkeypatch.setattr(gates, "_git_head", lambda: "head-bbb")
    third = gates.get_baseline_failures(force=True)
    assert calls["n"] == 2

