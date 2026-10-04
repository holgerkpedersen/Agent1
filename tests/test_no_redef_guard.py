"""Regression guard: no name may be defined twice in one function scope.

Defects this guards
-------------------
Two independent same-scope shadowings, both reported by mypy as
``[no-redef]``:

``agent_core/commands/implement_cmd.py`` (``execute``)
    ``missing = []`` collected *filenames* inside ``if retry_mode:``, while
    ``missing: list[tuple[str, str]] = []`` collected broken imports later in
    the same function -- and ``for fname, missing in broken_imports.items()``
    rebound it a third time as a loop variable.  mypy could not type-check
    ``for mod, name in missing`` because the name was ambiguous, so it raised
    ``[str-unpack]`` -- "Unpacking a string is disallowed" -- on a line that
    really does unpack tuples.

``agent.py`` (the Hue ``set_light`` action)
    ``state = "ON" if ... else "OFF"`` built a display string in the
    listing branch, while ``state: dict[str, Any] = {}`` built the payload
    pushed to ``bridge.set_light(light_id, **state)`` in the setter branch.
    mypy unified the two, so the ``state["brightness"] = ...`` writes raised
    ``[index]`` "Unsupported target for indexed assignment (\\"str\\")".

Neither was a live crash -- the branches are mutually exclusive -- but the
shared names made mypy *unable* to verify those lines, and the resulting
false alarms would have buried a real one in the same list.  A repo that
trusts its type checker cannot afford errors the checker cannot express.

Scope of the guard
------------------
``[no-redef]`` is the exact signal mypy uses for this pattern, so asserting
it is absent from the two historically-shadowed modules catches any name
that starts being bound twice in one scope again -- not just these two.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# The two modules that carried the shadowings, checked as one mypy invocation.
GUARDED = (
    "agent.py",
    "agent_core/commands/implement_cmd.py",
)

NO_REDEF = re.compile(r"^(?P<path>[^\s:]+):\d+: error:.*\[no-redef\]")

# mypy on a cold cache exceeds the harness's 120s default; children must still
# be capped (see tests/test_fix_subprocess_guard.py for why an uncapped child
# hangs the suite).
MYPY_TIMEOUT = 180


def _mypy_bin() -> str | None:
    return shutil.which("mypy")


@pytest.mark.skipif(_mypy_bin() is None, reason="mypy is not installed")
def test_no_same_scope_name_redefinition():
    """Neither guarded module may bind one name twice within a scope."""
    proc = subprocess.run(
        [_mypy_bin(), *GUARDED, "--no-error-summary", "--cache-dir", str(REPO_ROOT / ".mypy_cache")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO_ROOT),
        timeout=MYPY_TIMEOUT,
    )
    # mypy exits non-zero whenever *any* error exists; these files have plenty
    # of other pre-existing diagnostics, so only the [no-redef] code matters.
    output = (proc.stdout or "") + (proc.stderr or "")

    guarded = {path.replace("\\", "/") for path in GUARDED}
    offenders = []
    for line in output.splitlines():
        match = NO_REDEF.match(line)
        if match and match.group("path").replace("\\", "/") in guarded:
            offenders.append(line.strip())

    assert not offenders, (
        "A name is defined twice within one scope -- rename one of them so "
        "mypy can type-check both uses:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.skipif(_mypy_bin() is None, reason="mypy is not installed")
def test_guard_actually_ran_mypy():
    """A silent mypy failure must not be read as a clean result."""
    proc = subprocess.run(
        [_mypy_bin(), *GUARDED, "--no-error-summary", "--cache-dir", str(REPO_ROOT / ".mypy_cache")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO_ROOT),
        timeout=MYPY_TIMEOUT,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    assert "error:" in output, (
        "mypy produced no diagnostics at all for modules that are known to "
        "carry pre-existing errors -- the run likely failed before analysis "
        f"(exit {proc.returncode}):\n{output[:800]}"
    )
