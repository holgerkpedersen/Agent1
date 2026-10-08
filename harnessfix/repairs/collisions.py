"""String-collision guard for the repair catalog.

A repair that rewrites a literal string must not silently break a test
assertion that pins the OLD runtime string.  Observed live: the
tool-interface-error-detail repair changed "Tool error: {exc}" and broke
test_tool_loop_nlp.py's exact substring assertion, costing a full gate run
and a revert.  Before a repair is applied, the loop scans the test suite for
the RUNTIME string fragments the repair alters; any hit skips the repair and
records every occurrence for the human review gate (decision #015).

Only PINS count.  A test that merely *produces* the string — a fake executor
``return "Tool error: boom"``, a variable holding it — supplies the value
under test and cannot be broken by rewriting the production format.  Counting
those as collisions deadlocked the loop: with the tool-interface repair
already accepted, the two executor doubles in tests/test_tool_loop_nlp.py
made every iteration report ``skipped_test_collision`` while the file's real
assertion already pinned the NEW format.

The classification deliberately errs toward MISSING a pin rather than
inventing one: a missed pin costs one gate run (the test gate rejects and
reverts the repair), whereas a false collision blocks the repair forever.
Known limitation: a fragment produced on a ``return`` line and pinned
indirectly (``return "Tool error: boom"`` in a helper, ``assert helper() in
out``) is not detected, because pinning it would need real data flow.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: Tests dir scanned by the guard (monkeypatchable for hermetic tests).
DEFAULT_TESTS_DIR = Path("tests")

#: Test files that are themselves the guard's fixtures — they contain the
#: runtime fragments as LITERALS (e.g. test_harnessfix_collisions.py asserts
#: on find_test_collisions("Tool error: ")), so scanning them makes every
#: repair a self-block.  These fixture files are updated alongside the
#: repair, so they never pin the OLD runtime string (decision #051).
GUARD_TEST_FILENAMES: frozenset[str] = frozenset(
    {"test_harnessfix_collisions.py", "test_harnessfix_loop.py"}
)

#: A statement that only PRODUCES the fragment (supplies it to the code under
#: test) rather than asserting on it.  Checked BEFORE the pin patterns: a
#: ``return``/``yield`` hands the string back, and any ``=`` that is not a
#: comparison binds a value — neither can be broken by rewriting the
#: production format.  ``(?<![=!<>])=(?!=)`` matches a real assignment and
#: skips ``==``, ``!=``, ``<=``, ``>=``.
_PRODUCER_RE = re.compile(r"\b(?:return|yield)\b|(?<![=!<>])=(?!=)")

#: A statement that PINS the fragment: an ``assert``, a comparison, a
#: membership test, or an explicit substring check.  Statements with neither
#: marker are ignored — the guard errs toward MISSING a pin (see module doc).
_PIN_RE = re.compile(
    r"\bassert\b|==|!=|<=|>=|\bin\b|\.startswith\(|\.endswith\("
    r"|\.count\(|\.index\(|\.find\("
)


def _logical_statements(lines: list[str]) -> list[tuple[int, int]]:
    """Group *lines* into (first, last) 1-based logical statement spans.

    A statement ends when its bracket depth returns to zero, so a wrapped
    ``assert (\\n  "Tool error: boom"\\n) in out`` counts as ONE statement —
    otherwise the fragment's own continuation line (no ``assert`` on it)
    would be misread as a producer and the real pin would be missed.
    """
    spans: list[tuple[int, int]] = []
    start: int | None = None
    depth = 0
    for lineno, line in enumerate(lines, start=1):
        if start is None:
            if not line.strip():
                continue
            start = lineno
        depth += line.count("(") + line.count("[") + line.count("{")
        depth -= line.count(")") + line.count("]") + line.count("}")
        if depth <= 0:
            spans.append((start, lineno))
            start = None
            depth = 0
    if start is not None:
        spans.append((start, len(lines)))
    return spans


def _statement_pins(statement: str) -> bool:
    """True if *statement* asserts on the fragment instead of producing it."""
    if _PRODUCER_RE.search(statement):
        return False
    return bool(_PIN_RE.search(statement))


@dataclass(frozen=True)
class StringCollision:
    """One test-suite occurrence of a repair-affected string fragment."""

    path: Path
    line: int
    snippet: str
    fragment: str

    def to_dict(self) -> dict[str, object]:
        return {
            "file": str(self.path),
            "line": self.line,
            "snippet": self.snippet,
            "fragment": self.fragment,
        }


def find_test_collisions(
    fragments: tuple[str, ...],
    tests_dir: Path = DEFAULT_TESTS_DIR,
    exclude_files: frozenset[str] = GUARD_TEST_FILENAMES,
) -> list[StringCollision]:
    """Return every test file line containing any *fragments* fragment.

    Fragments are RUNTIME strings a repair alters (e.g. "Tool error: "), not
    source lines — assertions pin runtime output.  An empty result means the
    repair is safe to apply without touching the test contract.

    ``exclude_files`` skips the guard's OWN fixture tests (see
    GUARD_TEST_FILENAMES, decision #051): those files exercise the guard and
    contain the fragments as literals, so scanning them would make every
    repair self-block.  Real pinning tests are never excluded.

    Only PIN statements are reported (see the module doc): a test that merely
    produces the fragment — a fake executor's ``return`` — supplies the value
    under test and cannot break when the production format changes.
    """
    if not fragments or not tests_dir.is_dir():
        return []
    hits: list[StringCollision] = []
    for path in sorted(tests_dir.rglob("*.py")):
        if exclude_files and path.name in exclude_files:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        pinning = {
            lineno
            for first, last in _logical_statements(lines)
            if _statement_pins("\n".join(lines[first - 1 : last]))
            for lineno in range(first, last + 1)
        }
        for lineno, line in enumerate(lines, start=1):
            if lineno not in pinning:
                continue
            for fragment in fragments:
                if fragment in line:
                    hits.append(
                        StringCollision(
                            path=path,
                            line=lineno,
                            snippet=line.strip()[:120],
                            fragment=fragment,
                        )
                    )
    return hits
