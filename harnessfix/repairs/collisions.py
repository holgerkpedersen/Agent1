"""String-collision guard for the repair catalog.

A repair that rewrites a literal string must not silently break a test
assertion that pins the OLD runtime string.  Observed live: the
tool-interface-error-detail repair changed "Tool error: {exc}" and broke
test_tool_loop_nlp.py's exact substring assertion, costing a full gate run
and a revert.  Before a repair is applied, the loop scans the test suite for
the RUNTIME string fragments the repair alters; any hit skips the repair and
records every occurrence for the human review gate (decision #015).
"""
from __future__ import annotations

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
        for lineno, line in enumerate(lines, start=1):
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
