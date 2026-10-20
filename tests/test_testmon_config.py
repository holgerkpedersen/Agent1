"""Regression: pytest-testmon must not run with branch coverage enabled.

testmon_core.setup_coverage raises TestmonException when the pytest-cov
config has ``branch = true``. The repo's ``[tool.coverage.run]`` section
must therefore keep branch coverage off.
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_coverage_run_branch_is_disabled_for_testmon() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    run_cfg = data["tool"]["coverage"]["run"]
    assert run_cfg.get("branch") is False


def test_coverage_config_loads_without_branch() -> None:
    import coverage

    cov = coverage.Coverage(config_file=str(ROOT / "pyproject.toml"))
    assert cov.config.branch is False
