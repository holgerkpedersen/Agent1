"""The single definition of "this is fixture noise, not a Kanban card".

Two subsystems need the same answer:

* the **outbox** (:func:`harnessfix.issues._enqueue_sync`) must not publish
  junk in the first place, and
* the **purge** tool (:mod:`harnessfix.kanban_purge`) cleans up junk that got
  published before the outbox gate existed.

If those two disagreed, junk would either leak through the gate or be
classified as junk on the board while still being publishable — so the rule
lives here, once, and both import it.  :func:`harnessfix.kanban_purge.classify_card`
is the card-shaped adapter over this module.

Three junk shapes, all of them impossible for a human to have authored:

``placeholder-title``
    ``x``, ``y``, ``low``, ``bench`` … — the one-word titles tests use.

``pytest-temp-title``
    A title built from a path inside a pytest temp root
    (``.../Temp/pytest-of-Holger/...`` or ``.../.pytest-tmp/...``).

``out-of-repo-location``
    A finding whose every ``"<file>:<line>"`` location lies outside the repo,
    i.e. a detector run over synthetic input.  Note a real issue record can be
    junk even when its title looks plausible, which is why the outbox gate
    checks this rule and not only the title rules.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

#: ``C:/x``, ``C:\x`` - a drive-rooted path, absolute on Windows only.
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:[\\/]")

#: ``Path`` resolves paths with the *host* flavour, so "is this absolute?" must
#: be answered per flavour rather than trusting ``Path.is_absolute()``.
_HOST_IS_POSIX = os.name == "posix"

#: Titles that can never be a human-written card.  Exact match, case-folded.
PLACEHOLDER_TITLES: frozenset[str] = frozenset(
    {"x", "y", "z", "low", "high", "bench", "ælø", "æølæøl", "ælæl"}
)

#: Directory names that only ever occur inside a test's scratch tree.
TEMP_PATH_MARKERS: tuple[str, ...] = ("pytest-of-", ".pytest-tmp", ".pytest_tmp")

#: ``repo_root`` default: the package sits directly in the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent


def is_placeholder_title(title: str) -> bool:
    """True when *title* is a one-word fixture title such as ``x``."""
    return title.strip().casefold() in PLACEHOLDER_TITLES


def has_temp_path_marker(text: str) -> bool:
    """True when *text* mentions a pytest scratch root."""
    low = text.casefold()
    return any(marker in low for marker in TEMP_PATH_MARKERS)


def is_temp_path_title(title: str) -> bool:
    """True when *title* was built from a pytest temp findings path."""
    return has_temp_path_marker(title)


def location_file(location: str) -> str:
    """Split ``"<file>:<line>"`` into its file part (Windows-safe).

    A Windows path is ``C:/x/y.py:12`` and a repo-relative one is
    ``agent_core/x.py:12``; both split on the last colon.
    """
    if not location:
        return ""
    return location.rsplit(":", 1)[0] if ":" in location else location


def in_repo(location: str, repo_root: Path | None = None) -> bool:
    """True when a ``"<file>:<line>"`` location points inside the repo.

    Locations are repo-relative for real findings and absolute for a
    detector's synthetic input.  A location that cannot be parsed is treated
    as outside: publishing nothing is always safe.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    file_part = location_file(location)
    if not file_part:
        return False
    # A temp-root marker is decisive regardless of where root happens to be.
    if has_temp_path_marker(file_part):
        return False
    # A drive-rooted (or UNC) path is absolute on Windows.  On a POSIX host
    # ``Path("C:/x/y.py")`` is *relative*, so joining it onto the root would
    # make it look like repo work and leak the fixture to the board.
    if _DRIVE_PREFIX.match(file_part) or file_part.startswith("\\\\"):
        if _HOST_IS_POSIX:
            return False
        candidate = Path(file_part)
    else:
        candidate = Path(file_part)
        if not candidate.is_absolute():
            candidate = root / candidate
    try:
        candidate.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return False
    return True


def classify_issue(
    issue: dict[str, Any], repo_root: Path | None = None
) -> Optional[str]:
    """Why this issue record is fixture noise, or ``None`` if it is real work."""
    if not isinstance(issue, dict):
        return "not-an-issue"
    title = str(issue.get("title") or "")
    if is_placeholder_title(title):
        return "placeholder-title"
    if is_temp_path_title(title):
        return "pytest-temp-title"
    locations = issue.get("locations") or []
    if isinstance(locations, list) and locations:
        if not any(in_repo(str(loc), repo_root) for loc in locations):
            return "out-of-repo-location"
    return None


def is_publishable(issue: dict[str, Any], repo_root: Path | None = None) -> bool:
    """True when *issue* is real repo work and may reach the Kanban outbox."""
    return classify_issue(issue, repo_root) is None
