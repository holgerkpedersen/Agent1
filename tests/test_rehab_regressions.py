"""Regression tests for latent-bug fixes found during repo rehabilitation.

Covers bugs that were silent because no tests exercised these paths:

* ``implement_cmd`` used an unimported ``_difflib`` (NameError on rewrite guard)
* ``cleanup_cmd --delete`` called ``os.remove`` without importing ``os``
* ``module_similarity`` referenced an undefined ``np`` and shadowed it locally
* ``performance_dashboard`` root package crashed on import (phantom re-exports)
* ``analytics.processor`` / ``aggregator`` used isinstance()/attribute access
  against TypedDicts, which are plain dicts at runtime (TypeError/AttributeError)
* ``utils.validation`` lacked the module-level validate_* wrappers the
  collectors import; validation itself used attribute access on TypedDicts
* ``collectors.command_collector`` imported Unix-only ``resource`` unconditionally
"""

from __future__ import annotations

import ast
import builtins
import io
import sys
import tokenize
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

WS = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# static guards: the exact NameError patterns that shipped must stay dead
# --------------------------------------------------------------------------

def _module_source(rel: str) -> str:
    return (WS / rel).read_text(encoding="utf-8")


def test_implement_cmd_imports_difflib_and_has_no_private_alias() -> None:
    src = _module_source("agent_core/commands/implement_cmd.py")
    assert "import difflib" in src
    tree = ast.parse(src)
    names = {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
    }
    assert "_difflib" not in names


def test_cleanup_cmd_imports_os() -> None:
    src = _module_source("agent_core/commands/cleanup_cmd.py")
    assert "\nimport os\n" in src or src.startswith('"""') and "import os" in src


def test_module_similarity_defines_numpy_accessor_with_annotation() -> None:
    from agent_core.utils.module_similarity import _numpy  # real code path

    np = _numpy()
    assert hasattr(np, "zeros")
    # the old bug: bare `np` references resolved only via local shadowing
    tree = ast.parse(_module_source("agent_core/utils/module_similarity.py"))
    loads = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Name) and n.id == "np" and isinstance(n.ctx, ast.Load)
    ]
    assert loads, "expected typed `np` usages"


# --------------------------------------------------------------------------
# performance_dashboard: importability + TypedDict-safe analytics
# --------------------------------------------------------------------------

def test_performance_dashboard_root_imports_cleanly() -> None:
    for mod in ("performance_dashboard", "performance_dashboard.main"):
        assert mod in sys.modules or __import__(mod)


def test_root_package_exposes_real_names_not_phantoms() -> None:
    import performance_dashboard as pd

    for name in ("CommandMetric", "TaskMetric", "DatabaseConfig",
                 "PerformanceThresholds"):
        assert getattr(pd, name, None) is not None
    with pytest.raises(AttributeError):
        pd.Config  # noqa: B018 - phantom name removed on purpose


def test_lazy_attrs_resolve_without_optional_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "performance_dashboard.collectors.task_collector",
                        raising=False)
    import performance_dashboard as pd

    assert pd.TaskCollector.__name__ == "TaskCollector"
    agg = pd.RollingWindowAggregator()
    summary = agg.statistical_summary([1.0, 2.0, 3.0])
    assert summary is not None and summary["mean_value"] == pytest.approx(2.0)


def test_command_collector_survives_missing_resource_module() -> None:
    # Windows lacks `resource`; the module must import either way.
    import performance_dashboard.collectors.command_collector as cc

    log = cc.parse_execution_logs(
        "Execution time: 42.0 ms\nMemory usage: 512 KiB\nCommand: build\n")
    assert log["execution_time_ms"] == 42.0
    assert log["memory_usage_mb"] == pytest.approx(0.5)


def test_processor_handles_mixed_typeddict_records() -> None:
    from performance_dashboard.analytics.processor import (
        compute_statistical_distributions,
        calculate_performance_trends,
    )

    cmd = {"execution_time_ms": 100.0, "memory_usage_mb": 32.0,
           "cpu_utilization_percent": 50.0, "command_name": "c",
           "return_code": 0}
    task = {"duration_seconds": 4.0, "success_rate": 1.0,
            "resource_consumption_units": 2.0, "task_id": "t",
            "status": "completed"}
    dists = compute_statistical_distributions([cmd, task])
    assert set(dists) == {"command_execution_time", "task_duration"}
    trends = calculate_performance_trends([cmd, task])
    assert len(trends) == 2


def test_aggregator_ingests_records_without_isinstance_crash() -> None:
    from performance_dashboard.analytics.aggregator import (
        AggregationEngine, HierarchicalAggregator,
    )

    rec = {"timestamp": 1.0,
           "record_type": {"execution_time_ms": 10.0, "memory_usage_mb": 1.0,
                           "cpu_utilization_percent": 5.0,
                           "command_name": "x", "return_code": 0},
           "source_identifier": "s", "metadata": {}}
    engine = AggregationEngine()
    assert engine.ingest(rec) is True          # shape check, not isinstance
    assert engine.ingest({"bogus": True}) is False
    hier = HierarchicalAggregator()
    hier.ingest(rec)
    assert hier.aggregate_by_command()["x"]["sample_count"] == 1


def test_validation_wrappers_exist_and_return_bool() -> None:
    from performance_dashboard.utils.validation import (
        MetricSchemaValidator, validate_task_metric,
    )

    good = {"duration_seconds": 1.0, "success_rate": 1.0,
            "resource_consumption_units": 0.0, "task_id": "t"}
    bad = {**good, "success_rate": 7.0}
    v = MetricSchemaValidator()
    assert validate_task_metric(v, good) is True
    assert validate_task_metric(v, bad) is False   # bool contract, not raise


def test_task_collector_aggregation_end_to_end() -> None:
    from performance_dashboard.collectors.task_collector import TaskCollector

    tc = TaskCollector()
    sub = {"duration_seconds": 2.0, "success_rate": 1.0,
           "resource_consumption_units": 1.0, "task_id": "s",
           "status": "completed"}
    merged = tc.aggregate_subtask_metrics("parent", sub)
    assert merged is not None
    assert merged["duration_seconds"] == pytest.approx(2.0)
