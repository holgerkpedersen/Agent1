"""Packaging-metadata regression tests.

``requirements.txt`` and ``pyproject.toml`` are maintained by hand in parallel
and have drifted before: ``tzdata`` was declared only in pyproject, so a plain
``pip install -r requirements.txt`` environment raised ``ZoneInfoNotFoundError``
on Windows, and the ``performance_dashboard`` API/storage modules import
``flask``, ``marshmallow`` and ``redis.asyncio`` at module top level while no
file declared them.

These tests pin the two manifests to each other, pin the interpreter floor to
what CI and the README promise, and pin declared deps to the top-level
third-party imports the tracked code actually performs - so a new undeclared
hard import fails the suite instead of the user's install.
"""

from __future__ import annotations

import ast
from fnmatch import fnmatch
import re
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements.txt"
README = ROOT / "README.md"

# Top-level packages that are deliberately not shipped in the wheel.
_NON_DISTRIBUTION_PACKAGES: set[str] = set()

# Distribution name -> import name, for the pairs that are not identical.
_DIST_TO_IMPORT = {
    "pillow": "PIL",
    "scikit-learn": "sklearn",
    "sentence-transformers": "sentence_transformers",
    "types-networkx": "networkx",
    "types-pyyaml": "yaml",
    "pytest-cov": "pytest_cov",
    "pytest-asyncio": "pytest_asyncio",
    "pytest-testmon": "pytest_testmon",
}

# Directories whose modules are not part of the installed distribution, so
# their imports are not install requirements.
_NON_PACKAGE_DIRS = {"tests", "scripts", "docs", ".github"}

_REQ_LINE = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9_.\-]+)\s*(?:\[[^\]]*\])?\s*(?P<spec>[<>=!~].*)?$"
)


def _split_spec(spec: str) -> tuple[str, str]:
    """Return (name, version-specifier) for one requirement string."""
    match = _REQ_LINE.match(spec)
    if match is None:
        return "", spec
    name = match.group("name").lower().replace("_", "-")
    return name, (match.group("spec") or "").strip()


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _requirements_lines() -> list[str]:
    lines: list[str] = []
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and not line.startswith("-r "):
            lines.append(line)
    return lines


def test_pyproject_is_valid_toml(pyproject: dict) -> None:
    assert pyproject["project"]["name"], "project must have a name"


def test_requirements_covers_every_core_dependency(pyproject: dict) -> None:
    """Every [project.dependencies] entry must appear in requirements.txt.

    Regression: ``tzdata`` was only in pyproject, so the requirements-only
    install documented at the top of requirements.txt was broken on Windows.
    """
    core = {_split_spec(d)[0] for d in pyproject["project"]["dependencies"]}
    req = {_split_spec(line)[0] for line in _requirements_lines()}
    assert core - req == set(), (
        f"core deps missing from requirements.txt: {sorted(core - req)}"
    )


def test_requirements_entries_are_declared_in_pyproject(pyproject: dict) -> None:
    """requirements.txt must not invent dependencies pyproject does not know.

    Extras are legitimate requirements.txt content (the file lists optional
    groups), so anything extra must match a declared optional-dependency.
    """
    known = {_split_spec(d)[0] for d in pyproject["project"]["dependencies"]}
    for group in pyproject["project"].get("optional-dependencies", {}).values():
        known |= {_split_spec(d)[0] for d in group}
    req = {_split_spec(line)[0] for line in _requirements_lines()}
    assert req - known == set(), (
        f"requirements.txt deps absent from pyproject: {sorted(req - known)}"
    )


def test_core_and_requirements_versions_agree(pyproject: dict) -> None:
    """A dependency present in both files must carry the same floor version."""
    core = dict(_split_spec(d) for d in pyproject["project"]["dependencies"])
    req = dict(_split_spec(line) for line in _requirements_lines())
    for name in sorted(set(core) & set(req)):
        assert core[name] == req[name], (
            f"{name}: pyproject {core[name]!r} vs requirements {req[name]!r}"
        )


def test_project_urls_point_at_the_real_repository(pyproject: dict) -> None:
    """The published URLs must match the git remote, not a placeholder org."""
    urls = pyproject["project"].get("urls", {})
    for key in ("Homepage", "Repository"):
        url = urls.get(key, "")
        assert "agent-framework/agent-framework" not in url, (
            f"{key} is a placeholder: {url}"
        )
        assert url.count("/") >= 3, f"{key} is not a full repo URL: {url}"


def test_requires_python_matches_ci_and_readme(pyproject: dict) -> None:
    """requires-python, the CI matrix and the README must state one floor.

    Regression: pyproject said >=3.10 while README said "Requires Python
    3.12+" and CI only tested 3.12 - a 3.10 install was unsupportable.
    """
    spec = pyproject["project"]["requires-python"]
    match = re.search(r">=\s*(\d+)\.(\d+)", spec)
    assert match, f"requires-python floor not found in {spec!r}"
    floor = (int(match.group(1)), int(match.group(2)))

    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    ci_versions = {
        tuple(int(p) for p in v.split(".")[:2])
        for v in re.findall(r"\"(\d+\.\d+(?:\.\d+)?)\"", ci)
    }
    assert ci_versions, "no Python versions found in ci.yml"
    assert min(ci_versions) >= floor, (
        f"CI tests {sorted(ci_versions)} below requires-python {floor}"
    )

    readme = README.read_text(encoding="utf-8")
    readme_versions = {
        tuple(int(p) for p in v.split(".")[:2])
        for v in re.findall(r"[Pp]ython \*\*(\d+\.\d+)", readme)
    }
    for version in readme_versions:
        assert version >= floor, (
            f"README advertises Python {version} below requires-python {floor}"
        )


def _top_level_imports() -> set[str]:
    """Every top-level module imported by tracked Python source, via AST.

    Whole-file AST parse (not a regex) so docstring text, comments and
    ``from src.agent1`` strings in test fixtures are not counted as imports.
    """
    names: set[str] = set()
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if rel.parts[0] in _NON_PACKAGE_DIRS:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def test_every_hard_import_is_declared(pyproject: dict) -> None:
    """No tracked module may import an undeclared third-party package.

    Regression: ``performance_dashboard/api/endpoints.py`` imports flask and
    marshmallow, and ``storage/metrics_store.py`` imports ``redis.asyncio``, at
    module top level - all undeclared, so the modules were unimportable.
    """
    declared = {_split_spec(d)[0] for d in pyproject["project"]["dependencies"]}
    for group in pyproject["project"].get("optional-dependencies", {}).values():
        declared |= {_split_spec(d)[0] for d in group}
    importable = {_DIST_TO_IMPORT.get(name, name) for name in declared}

    local = {
        p.name for p in ROOT.iterdir() if p.is_dir() and (p / "__init__.py").exists()
    }
    local |= {p.stem for p in ROOT.glob("*.py")}
    stdlib = set(getattr(sys, "stdlib_module_names", ()))

    undeclared = sorted(
        name
        for name in _top_level_imports()
        if name
        and name not in stdlib
        and name not in local
        and name not in importable
        and not name.startswith("_")
    )
    assert undeclared == [], f"imported but undeclared in pyproject: {undeclared}"


def _fnmatch(name: str, pattern: str) -> bool:
    """setuptools' own matching rule for packages.find include/exclude."""
    return fnmatch(name, pattern)


def test_packages_find_covers_every_importable_package(pyproject: dict) -> None:
    """Every first-party top-level package must be in the wheel.

    Regression: ``[tool.setuptools.packages.find]`` listed only
    ``agent_core*``, so ``pip install agent-framework`` built a wheel that
    silently dropped ``harnessfix``, ``fixcommand``, ``performance_dashboard``
    and ``benchmarks`` - all importable packages (they have ``__init__.py``)
    that CI itself lints.  A consumer installing the wheel got
    ``ModuleNotFoundError`` for every non-agent_core entry point.
    """
    find = pyproject["tool"]["setuptools"]["packages"]["find"]
    include = list(find["include"])
    exclude = list(find.get("exclude", []))

    local_packages = {
        p.name
        for p in ROOT.iterdir()
        if p.is_dir()
        and (p / "__init__.py").exists()
        and p.name not in _NON_DISTRIBUTION_PACKAGES
    }
    assert local_packages, "no top-level packages found - fixture is broken"

    shipped = {
        name
        for name in local_packages
        if any(_fnmatch(name, pat) for pat in include)
        and not any(_fnmatch(name, pat) for pat in exclude)
    }
    dropped = local_packages - shipped
    assert dropped == set(), (
        f"packages importable from the repo but excluded from the wheel: "
        f"{sorted(dropped)}; include={include}"
    )


def test_wheel_contains_every_subpackage(tmp_path: Path) -> None:
    """The built wheel must contain every subpackage, not just top levels.

    ``packages.find`` include patterns are prefix globs: ``harnessfix*``
    matches ``harnessfix.repairs`` only if the find config is honoured. This
    builds a real wheel and inspects it, so a config regression (a dropped
    include entry, an over-broad exclude) fails here, not for the consumer.
    """
    wheel_dir = tmp_path / "wheel"
    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps",
         "--no-build-isolation", "-w", str(wheel_dir)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"pip wheel failed:\n{result.stderr[-2000:]}"

    wheels = list(wheel_dir.glob("*.whl"))
    assert wheels, "no wheel produced"
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()

    def top_and_sub_packages() -> set[str]:
        found: set[str] = set()
        for pkg in ROOT.iterdir():
            if not (pkg.is_dir() and (pkg / "__init__.py").exists()):
                continue
            if pkg.name in _NON_DISTRIBUTION_PACKAGES:
                continue
            found.add(pkg.name)
            for sub in pkg.rglob("__init__.py"):
                rel = sub.parent.relative_to(ROOT)
                found.add(".".join(rel.parts))
        return found

    expected = top_and_sub_packages()
    present = {
        ".".join(n.split("/")[:-1])
        for n in names
        if n.endswith("__init__.py") and not n.startswith("..")
    }
    missing = {p for p in expected if p not in present}
    assert missing == set(), f"packages missing from the wheel: {sorted(missing)}"

    leaks = [n for n in names if ".mypy_cache" in n or ".pytest_cache" in n]
    assert leaks == [], f"build caches leaked into the wheel: {leaks[:5]}"
