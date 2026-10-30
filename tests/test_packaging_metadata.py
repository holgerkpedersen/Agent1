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
import os
from fnmatch import fnmatch
import re
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Regression: this suite lives directly in the repo root's ``tests`` directory.
# A nested copy (tests/<pkg>/test_packaging_metadata.py) made parents[1] resolve
# to an ancestor of the repo, so the scan below walked sibling trees - a whole
# .venv-gguf site-packages - and reported environment artifacts as packaging
# violations.  Fail loudly at collection instead of silently scanning the drive.
assert (ROOT / "pyproject.toml").is_file(), (
    f"test file is nested: {__file__!r} does not sit in <repo>/tests/"
)
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

# Directories that only exist as local build/runtime artifacts (a virtualenv
# created in-tree, caches, generated reports).  Regression: creating an
# in-repo ``.venv`` made the recursive scan above walk ~15k site-packages
# files, so every optional import of every installed library (torch, pip's
# vendored distlib, PIL's ImageQt...) counted as an "undeclared" repo import
# and this test failed for an environment reason, not a packaging one.
_LOCAL_ARTIFACT_DIRS = {".venv", "venv", ".mypy_cache", ".ruff_cache", ".pytest_cache"}


def _normalise(name: str) -> str:
    """Collapse a distribution OR import name to one comparable spelling.

    PEP 503 makes distribution names case-insensitive and treats ``-``, ``_``
    and ``.`` as interchangeable, so ``huggingface-hub`` (the declared
    distribution) and ``huggingface_hub`` (what Python actually imports) name
    the SAME project.  Without this, a correctly declared dependency was
    reported as an undeclared import purely because of the separator.
    """
    return name.lower().replace("_", "-").replace(".", "-")


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


def _build_isolation_flags() -> list[str]:
    """Return the pip flags that build this repo in the CURRENT interpreter.

    ``--no-build-isolation`` is an optimisation, not a requirement: it reuses
    the environment's setuptools instead of downloading one.  It only works
    when the build backend named in ``pyproject.toml`` is importable here.  On
    a bare Python (a fresh CI runner, a stripped container) setuptools is not
    installed, so pip died with ``BackendUnavailable: Cannot import
    'setuptools.build_meta'`` and this test failed for an environment reason
    rather than a packaging regression.  Probe the backend first: when it is
    missing, ask pip for an isolated build and let it provision the backend.
    """
    backend = _build_backend()
    if _module_importable(backend):
        return ["--no-build-isolation"]
    return []


def _build_backend() -> str:
    """The ``build-backend`` declared in ``pyproject.toml``."""
    with PYPROJECT.open("rb") as handle:
        data = tomllib.load(handle)
    return data.get("build-system", {}).get("build-backend", "setuptools.build_meta")


def _module_importable(dotted: str) -> bool:
    """True when *dotted* (a module path of any depth) can be imported.

    Only the module spec is resolved - nothing is imported - so probing the
    build backend never runs code from it.
    """
    import importlib.util

    try:
        return importlib.util.find_spec(dotted) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


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
    """Every module-level import performed by tracked Python source, via AST.

    "Top-level" means a statement in the module body (including inside a
    top-level ``try``/``if`` block), not merely a top-level *module* name:
    an import nested in any function or class is lazy by definition and its
    package is an optional extra, so counting it here is a false positive.
    Regression: ``ast.walk`` descended everywhere, so the documented lazy
    import at ``strata_download.py`` (``from huggingface_hub import ...``
    inside ``_download_one``, declared as the ``strata-download`` extra) was
    reported as an undeclared hard import - and requirements.txt correctly
    omits it.  Whole-file AST parse (not a regex) so docstring text,
    comments and quoted code in test fixtures are not counted as imports.

    Pruned os.walk, not ``ROOT.rglob("*.py")``: the recursive glob follows
    directory links, and this machine has sibling trees with broken venv
    symlinks (a ``venv/bin/python`` link that raises WinError 1920 on stat),
    so rglob either died mid-scan or walked ~15k site-packages files outside
    the repo.  Pruning skips ``.git``, every dot-directory and every symlink
    before descending, so only real repo source is parsed.
    """
    names: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".")
            and d not in _NON_PACKAGE_DIRS
            and d not in _LOCAL_ARTIFACT_DIRS
            and not os.path.islink(os.path.join(dirpath, d))
        ]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            path = Path(dirpath) / filename
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, UnicodeDecodeError):  # pragma: no cover
                continue
            for node in tree.body:
                _collect_module_level_imports(node, names)
    return names


def _collect_module_level_imports(node: ast.stmt, names: set[str]) -> None:
    """Add top-level import names from *node*; never descend into defs.

    Recurses only through compound statements that execute at import time
    (``try``/``if``/``with``, and ``for``/``while``/``match`` whose bodies
    run when the module loads).  Function, class, async-function and
    lambda definitions are lazy: their imports run on first call, not at
    import time, so they belong to extras, not hard requirements.
    """
    if isinstance(node, ast.Import):
        for alias in node.names:
            names.add(alias.name.split(".")[0])
    elif isinstance(node, ast.ImportFrom):
        if node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    elif isinstance(node, ast.Try):
        for child in node.body:
            _collect_module_level_imports(child, names)
        for handler in node.handlers:
            for child in handler.body:
                _collect_module_level_imports(child, names)
        for child in node.orelse:
            _collect_module_level_imports(child, names)
        for child in node.finalbody:
            _collect_module_level_imports(child, names)
    elif isinstance(node, (ast.If, ast.With, ast.AsyncWith)):
        for child in node.body:
            _collect_module_level_imports(child, names)
        orelse = getattr(node, "orelse", None)
        if orelse is not None:
            for child in orelse:
                _collect_module_level_imports(child, names)
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        for child in node.body:
            _collect_module_level_imports(child, names)
        for child in node.orelse:
            _collect_module_level_imports(child, names)
    elif isinstance(node, ast.Match):  # pragma: no cover - py>=3.10 only
        for case in node.cases:
            for child in case.body:
                _collect_module_level_imports(child, names)


def test_every_hard_import_is_declared(pyproject: dict) -> None:
    """No tracked module may import an undeclared third-party package.

    Regression: ``performance_dashboard/api/endpoints.py`` imports flask and
    marshmallow, and ``storage/metrics_store.py`` imports ``redis.asyncio``, at
    module top level - all undeclared, so the modules were unimportable.
    """
    declared = {_split_spec(d)[0] for d in pyproject["project"]["dependencies"]}
    for group in pyproject["project"].get("optional-dependencies", {}).values():
        declared |= {_split_spec(d)[0] for d in group}
    # Both sides are normalised: the declared side because PEP 503 makes the
    # separators interchangeable, the import side because the AST records the
    # module name verbatim.  A mapping entry still wins, since some imports
    # differ in more than punctuation (pillow -> PIL).
    importable = {_normalise(_DIST_TO_IMPORT.get(name, name)) for name in declared}

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
        and _normalise(name) not in importable
        and not name.startswith("_")
    )
    assert undeclared == [], f"imported but undeclared in pyproject: {undeclared}"


def test_top_level_imports_ignores_lazy_imports() -> None:
    """Only import-time imports count; function/class bodies are lazy.

    Regression: the collector walked the whole AST, so a documented lazy
    import (``huggingface_hub`` inside ``strata_download._download_one``)
    was reported as an undeclared hard requirement.  The collector must
    recurse through import-time compound statements (try/if/with/for) and
    stop at def/class/async-def - a lambda body is lazy too.
    """
    src = (
        "import os, json as j\n"
        "from . import sibling\n"
        "from pathlib import Path\n"
        "try:\n"
        "    import trying\n"
        "except ImportError:\n"
        "    import excepting\n"
        "if True:\n"
        "    import ifing\n"
        "with open('x') as f:\n"
        "    import withing\n"
        "for i in range(1):\n"
        "    import forring\n"
        "def _lazy():\n"
        "    from huggingface_hub import hf_hub_download\n"
        "class C:\n"
        "    import inside_class\n"
        "async def alazy():\n"
        "    import async_lazy\n"
        "f = lambda: __import__('lambda_import')\n"
    )
    names: set[str] = set()
    for node in ast.parse(src).body:
        _collect_module_level_imports(node, names)
    assert names == {
        "os",
        "json",
        "pathlib",
        "trying",
        "excepting",
        "ifing",
        "withing",
        "forring",
    }, names


def test_distribution_name_matches_its_import_name() -> None:
    """A declared distribution must match the module name Python imports.

    Regression: ``tests/test_strata_download.py`` imports ``huggingface_hub``
    while pyproject declares the distribution as ``huggingface-hub``; the
    check compared the two spellings literally, so a correctly declared
    dependency was reported as an undeclared hard import on every clean CI
    runner.  PEP 503 makes ``-``, ``_`` and ``.`` interchangeable, so both
    sides must normalise before comparing - while an explicit mapping entry
    still maps names that differ by more than punctuation (pillow -> PIL).
    """
    # The separator-only case, in both directions.
    assert _normalise("huggingface_hub") == _normalise("huggingface-hub")
    assert _normalise("HuggingFace.Hub") == _normalise("huggingface-hub")

    # A real mapping whose names differ beyond punctuation still resolves.
    assert _normalise(_DIST_TO_IMPORT["pillow"]) == _normalise("PIL")

    # Distinct projects must never collapse into each other.
    assert _normalise("huggingface-hub") != _normalise("sentence-transformers")


def test_declared_dependencies_cover_the_strata_download_import() -> None:
    """The real pair from the CI failure: the declaration is reachable.

    Pins the actual repository state rather than a synthetic string, so
    removing the ``huggingface-hub`` declaration (or breaking the
    normalisation above) fails here with a readable message instead of in
    ``test_every_hard_import_is_declared`` with only a module list.
    """
    with PYPROJECT.open("rb") as handle:
        data = tomllib.load(handle)
    declared = {_split_spec(d)[0] for d in data["project"]["dependencies"]}
    for group in data["project"].get("optional-dependencies", {}).values():
        declared |= {_split_spec(d)[0] for d in group}

    normalised = {_normalise(_DIST_TO_IMPORT.get(n, n)) for n in declared}
    assert _normalise("huggingface_hub") in normalised, (
        "huggingface_hub (imported by tests/test_strata_download.py) is not a "
        f"declared dependency: {sorted(declared)}"
    )


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
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(ROOT),
            "--no-deps",
            *_build_isolation_flags(),
            "-w",
            str(wheel_dir),
        ],
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


def test_build_isolation_flags_follow_backend_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--no-build-isolation`` only when the backend is importable.

    Regression: the wheel build above hard-coded ``--no-build-isolation``, so
    on a runner without setuptools pip raised
    ``BackendUnavailable: Cannot import 'setuptools.build_meta'`` and the job
    went red for an environment reason, not a packaging one.  The flag must be
    a function of what this interpreter can actually import.
    """
    monkeypatch.setattr(
        sys.modules[__name__], "_module_importable", lambda dotted: True
    )
    assert _build_isolation_flags() == ["--no-build-isolation"]

    monkeypatch.setattr(
        sys.modules[__name__], "_module_importable", lambda dotted: False
    )
    assert _build_isolation_flags() == []


def test_build_backend_is_read_from_pyproject() -> None:
    """The probed backend is pyproject's, so a backend swap cannot go stale."""
    assert _build_backend() == "setuptools.build_meta"


def test_dev_extra_installs_the_build_requirements(pyproject: dict) -> None:
    """The ``dev`` extra must install every ``[build-system].requires`` entry.

    Regression: ``pip install -e .[dev]`` did not install setuptools, so
    :func:`test_wheel_contains_every_subpackage` had no importable backend and
    fell back to an isolated build, which DOWNLOADS a backend mid-test and
    failed with ``BackendUnavailable: Cannot import 'setuptools.build_meta'``.
    The fallback is correct, but a dev install must not depend on the network to
    build the repo's own wheel: declaring the build requirements in the dev
    extra keeps that suite local and deterministic.
    """
    build_requires = {_split_spec(d)[0] for d in pyproject["build-system"]["requires"]}
    dev = {
        _split_spec(d)[0] for d in pyproject["project"]["optional-dependencies"]["dev"]
    }
    assert build_requires - dev == set(), (
        f"[build-system].requires not installed by the dev extra: "
        f"{sorted(build_requires - dev)}"
    )


def test_module_importable_probes_without_importing() -> None:
    """Probing resolves a spec and must never execute the module."""
    assert _module_importable("json") is True
    assert _module_importable("definitely_not_a_real_module_xyz") is False
    # A module-shaped path that does not exist must be False, not an error.
    assert _module_importable("json.no_such_submodule_xyz") is False
