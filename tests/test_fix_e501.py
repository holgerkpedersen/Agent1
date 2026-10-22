"""Regression tests for the E501 line-rewrapping fixer (``scripts/fix_e501.py``).

The fixer rewrites over-long source lines.  Every rewrite must satisfy
three gates, and each gate gets its own test here:

1. VALUE GATE - the parsed AST must not change, so no runtime value moves.
2. FORMAT GATE - the result must be a fixed point of ``ruff format``,
   otherwise the next format run would undo the rewrite and the
   repository would gain churn.  This is the gate whose absence produced
   format-dirty output in the working tree.
3. LINT GATE - the target violation must be gone and the total number of
   violations must strictly decrease.

The fixer is driven through its real entry points
(``fix_text`` / ``fix_file``), never through a copy of its logic.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts import fix_e501

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ast(text: str) -> str:
    """The AST dump of ``text``; used as the value fingerprint."""
    return ast.dump(ast.parse(text))


def _is_format_fixed_point(path: Path, text: str) -> bool:
    """True when ``ruff format`` leaves ``text`` unchanged."""
    probe = fix_e501._run(
        ["format", "--no-cache", "--stdin-filename", str(path), "-"], text=text
    )
    return probe.returncode == 0 and probe.stdout == text


def _e501_rows(path: Path, text: str) -> list[int]:
    """The rows of ``text`` that violate E501."""
    return fix_e501._e501_rows_of_text(path, text, None)


def _run_fixer(tmp_path: Path, source: str) -> tuple[Path, str, list[int]]:
    """Run the fixer over ``source``; return (path, new_text, fixed_rows)."""
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    new_text, fixed, _unfixed = fix_e501.fix_text(path, source, None)
    return path, new_text, fixed


def _namespace(text: str) -> dict[str, object]:
    """Execute ``text`` and return the resulting module namespace."""
    namespace: dict[str, object] = {}
    exec(compile(ast.parse(text), "<fixture>", "exec"), namespace)  # noqa: S102
    return namespace


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
#
# The fixtures are built by concatenation so that this test file stays
# inside the 88 character line limit while the fixtures stay over it.

#: A long implicit concatenation inside a call, the ``rebalance`` strategy.
LONG_CONCAT = (
    "def f(class_conflicts, dangerous_files):\n"
    "    reasons = []\n"
    "    if class_conflicts:\n"
    "        reasons.append(\n"
    "            f\"{sum(1 for cc in class_conflicts if cc['file'] in "
    'dangerous_files)} have class-name conflicts"\n'
    "        )\n"
    "    return reasons\n"
)

#: A long f-string, the ``split`` strategy.
_LONG_WORDS = "the quick brown fox jumps over the lazy dog "

LONG_FSTRING = 'def g():\n    return f"' + _LONG_WORDS * 3 + '"\n'

#: A long plain string, the ``implicit-concatenation`` strategy.
LONG_PLAIN_STRING = 'msg = "' + _LONG_WORDS * 3 + '"\n'

#: A long comment, the ``comment`` strategy.
_LONG_COMMENT = "this is a very long explanatory comment that goes past the limit "

LONG_COMMENT = "# " + _LONG_COMMENT * 2 + "\ndef h():\n    return 1\n"

#: A long line inside a docstring, the ``continuation`` strategy.
LONG_DOCSTRING = (
    "def i():\n"
    '    """'
    + "This docstring holds one very long physical line that must be cut. " * 2
    + '"""\n'
)

#: A long line inside a multiline raw string.  The fixer has no candidate
#: for this shape, so it must be left alone rather than damaged.
LONG_JS_IN_RAW_STRING = (
    'JS = r"""\n'
    "      const els = {\n"
    "        'hgram-chart': { innerHTML: '', style: {}, textContent: '',\n"
    "                         addEventListener: function(){}, "
    "querySelector: function(){return null;} },\n"
    "      };\n"
    '"""\n'
)

ALL_CASES = [
    pytest.param(LONG_CONCAT, id="rebalance"),
    pytest.param(LONG_FSTRING, id="split"),
    pytest.param(LONG_PLAIN_STRING, id="implicit-concatenation"),
    pytest.param(LONG_COMMENT, id="comment"),
    pytest.param(LONG_DOCSTRING, id="docstring"),
]


# ---------------------------------------------------------------------------
# VALUE GATE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ALL_CASES)
def test_fix_preserves_the_ast(tmp_path: Path, source: str) -> None:
    """The rewrite must not change any runtime value."""
    _path, new_text, _fixed = _run_fixer(tmp_path, source)
    assert _ast(new_text) == _ast(source)


def test_fix_preserves_the_string_value_exactly(tmp_path: Path) -> None:
    """The rewrapped string must read back as the same characters."""
    _path, new_text, _fixed = _run_fixer(tmp_path, LONG_PLAIN_STRING)
    before = ast.literal_eval(ast.parse(LONG_PLAIN_STRING).body[0].value)
    after = ast.literal_eval(ast.parse(new_text).body[0].value)
    assert before == after
    assert before == _LONG_WORDS * 3


def test_fix_preserves_the_fstring_value_exactly(tmp_path: Path) -> None:
    """Splitting an f-string must not change the rendered text.

    The AST gate allows a split at a literal boundary, so the rendered
    value is what actually has to hold.
    """
    _path, new_text, _fixed = _run_fixer(tmp_path, LONG_FSTRING)
    before = _namespace(LONG_FSTRING)["g"]()
    after = _namespace(new_text)["g"]()
    assert before == after
    assert before == _LONG_WORDS * 3


# ---------------------------------------------------------------------------
# FORMAT GATE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ALL_CASES)
def test_fix_output_is_a_ruff_format_fixed_point(tmp_path: Path, source: str) -> None:
    """The rewrite must be a fixed point of ``ruff format``.

    A rewrite that the formatter would undo again is churn: the next
    format run moves the lines, and the repository never settles.
    """
    path, new_text, _fixed = _run_fixer(tmp_path, source)
    assert _is_format_fixed_point(path, new_text)


def test_fix_does_not_leave_the_file_format_dirty(tmp_path: Path) -> None:
    """``fix_file`` must not write a file that ``ruff format`` would change."""
    path = tmp_path / "sample.py"
    path.write_text(LONG_CONCAT, encoding="utf-8")

    fixed, remaining, _unfixed, skipped = fix_e501.fix_file(path, apply=True)

    assert not skipped
    assert fixed > 0
    assert remaining == 0
    written = path.read_text(encoding="utf-8")
    assert _is_format_fixed_point(path, written)
    assert _ast(written) == _ast(LONG_CONCAT)


# ---------------------------------------------------------------------------
# LINT GATE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ALL_CASES)
def test_fix_removes_the_e501_violation(tmp_path: Path, source: str) -> None:
    """The target violation must be gone and the count must drop."""
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    before = _e501_rows(path, source)
    assert before, "the fixture must violate E501 to begin with"

    _path, new_text, fixed = _run_fixer(tmp_path, source)

    assert fixed, "the fixer must fix the fixture"
    after = _e501_rows(path, new_text)
    assert len(after) < len(before)
    assert not set(before) & set(after)


def test_fix_leaves_a_clean_file_untouched(tmp_path: Path) -> None:
    """A file without violations is not rewritten."""
    source = "x = 1\n"
    path, new_text, fixed = _run_fixer(tmp_path, source)
    assert fixed == []
    assert new_text == source


# ---------------------------------------------------------------------------
# safety guards
# ---------------------------------------------------------------------------


def test_fix_file_skips_a_file_that_is_not_format_clean(tmp_path: Path) -> None:
    """A file the formatter would rewrite anyway is left alone.

    Editing such a file would mix the fixer's reflow with the
    formatter's reflow and produce a result nobody asked for.
    """
    unformatted = "x = 'one'\n"
    path = tmp_path / "sample.py"
    path.write_text(unformatted, encoding="utf-8")

    fixed, remaining, _unfixed, skipped = fix_e501.fix_file(path, apply=True)

    assert skipped
    assert fixed == 0
    assert path.read_text(encoding="utf-8") == unformatted


def test_fix_leaves_a_multiline_raw_string_untouched(tmp_path: Path) -> None:
    """A long line inside a multiline raw string is not rewritten.

    The fixer has no safe split for that shape.  It must report the row
    as unfixed and leave the text alone instead of guessing.
    """
    path, new_text, fixed = _run_fixer(tmp_path, LONG_JS_IN_RAW_STRING)

    assert fixed == []
    assert new_text == LONG_JS_IN_RAW_STRING
    assert _ast(new_text) == _ast(LONG_JS_IN_RAW_STRING)


# ---------------------------------------------------------------------------
# SENTENCE GATE
# ---------------------------------------------------------------------------
#
# A split inside a sentence hides the sentence from a source-text search.
# The fixer therefore prefers to cut at a sentence boundary, so a phrase
# stays contiguous in the source even though the value is unchanged.

#: Prose whose sentences are long enough to need a split.
PROSE = (
    "def prompt():\n"
    '    return ("Alpha bravo charlie delta echo foxtrot. Golf hotel india '
    'juliett kilo lima. Mike november oscar papa quebec romeo. "\n'
    ")\n"
)

#: A prose phrase that must stay contiguous in the source text.
PHRASES = [
    "Alpha bravo charlie delta echo foxtrot.",
    "Golf hotel india juliett kilo lima.",
    "Mike november oscar papa quebec romeo.",
]


def test_fix_splits_prose_at_sentence_boundaries(tmp_path: Path) -> None:
    """A prose line is cut at a sentence boundary, not inside a sentence."""
    path, new_text, fixed = _run_fixer(tmp_path, PROSE)

    assert fixed, "the fixture must violate E501"
    for phrase in PHRASES:
        assert phrase in new_text, f"sentence split inside {phrase!r}"
    assert _ast(new_text) == _ast(PROSE)


def test_fix_keeps_the_prose_value_intact(tmp_path: Path) -> None:
    """Splitting at a sentence boundary must not change the value."""
    path, new_text, _fixed = _run_fixer(tmp_path, PROSE)

    before = _namespace(PROSE)["prompt"]()
    after = _namespace(new_text)["prompt"]()
    assert before == after


def test_fix_falls_back_to_word_splits_for_a_long_sentence(tmp_path: Path) -> None:
    """A sentence too long for one part is split at word boundaries.

    The sentence points cannot fit the budget, so the fixer falls back to
    the word points.  The result must still be free of E501 violations.
    """
    long_sentence = "word " * 30
    source = (
        "def prompt():\n"
        '    return ("Short opener. ' + long_sentence + 'Short tail. ")\n'
    )
    path, new_text, fixed = _run_fixer(tmp_path, source)

    assert fixed
    assert not _e501_rows(path, new_text)
    assert _ast(new_text) == _ast(source)
