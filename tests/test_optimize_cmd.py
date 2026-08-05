"""Tests for agent_core.commands.optimize_cmd — batching, parsing, and static analysis."""

from __future__ import annotations

import textwrap
from typing import Final

import pytest

from agent_core.commands.base import show_file_diff
from agent_core.commands.optimize_cmd import (
    create_batches,
    estimate_tokens,
    format_batch_context,
    parse_llm_fixes,
    SAFETY_MARGIN,
    SYSTEM_OVERHEAD_TOKENS,
)
from agent_core.patterns import analyze as static_analyze, detect_list_append_join


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 0

    def test_four_chars_per_token(self) -> None:
        assert estimate_tokens("abcd") == 1

    def test_rounds_down(self) -> None:
        assert estimate_tokens("abc") == 0

    def test_longer_text(self) -> None:
        text = "x" * 100
        assert estimate_tokens(text) == 25


# ---------------------------------------------------------------------------
# create_batches
# ---------------------------------------------------------------------------

class TestCreateBatches:
    def _contents(self, files: dict[str, str]) -> dict[str, str]:
        return files

    def test_single_file_single_batch(self) -> None:
        batches = create_batches({"a.py": "code"}, {"a.py": []})
        assert len(batches) == 1
        assert batches[0]["files"] == ["a.py"]

    def test_many_small_files_fit_in_one_batch(self) -> None:
        files = {f"f{i}.py": "x" * 10 for i in range(10)}
        findings = {k: [] for k in files}
        batches = create_batches(files, findings)
        assert len(batches) == 1

    def test_large_files_split_into_batches(self) -> None:
        big = "x" * 100_000  # ~25k tokens
        files = {"big1.py": big, "big2.py": big}
        findings = {k: [] for k in files}
        batches = create_batches(files, findings)
        assert len(batches) >= 2

    def test_files_sorted_smallest_first(self) -> None:
        files = {"large.py": "x" * 10_000, "small.py": "y" * 10}
        findings = {k: [] for k in files}
        batches = create_batches(files, findings)
        assert batches[0]["files"][0] == "small.py"

    def test_findings_add_to_token_count(self) -> None:
        files = {"a.py": "code"}
        findings = {"a.py": [{"line": 1, "pattern": "x", "suggestion": "y" * 200}]}
        batches = create_batches(files, findings)
        assert batches[0]["total_tokens"] > SYSTEM_OVERHEAD_TOKENS

    def test_empty_input(self) -> None:
        assert create_batches({}, {}) == []


# ---------------------------------------------------------------------------
# format_batch_context
# ---------------------------------------------------------------------------

class TestFormatBatchContext:
    def test_includes_filename_and_code(self) -> None:
        batch = {
            "files": ["foo.py"],
            "contents": {"foo.py": "print('hello')"},
            "findings": {"foo.py": []},
            "total_tokens": 10,
        }
        result = format_batch_context(batch)
        assert "foo.py" in result
        assert "print('hello')" in result

    def test_includes_findings(self) -> None:
        batch = {
            "files": ["bar.py"],
            "contents": {"bar.py": "x = 1"},
            "findings": {"bar.py": [{"line": 1, "pattern": "bare_except", "suggestion": "fix this"}]},
            "total_tokens": 10,
        }
        result = format_batch_context(batch)
        assert "bare_except" in result
        assert "fix this" in result

    def test_no_findings_shows_none(self) -> None:
        batch = {
            "files": ["c.py"],
            "contents": {"c.py": "pass"},
            "findings": {"c.py": []},
            "total_tokens": 10,
        }
        result = format_batch_context(batch)
        assert "(none)" in result


# ---------------------------------------------------------------------------
# parse_llm_fixes
# ---------------------------------------------------------------------------

class TestParseLLMFixes:
    def test_extracts_file_blocks(self) -> None:
        response = textwrap.dedent("""\
            [FILE: a.py]
            ```python
            import os
            x = 1
            ```
        """)
        fixes = parse_llm_fixes(response, ["a.py"])
        assert "a.py" in fixes
        assert "import os" in fixes["a.py"]

    def test_multiple_files(self) -> None:
        response = textwrap.dedent("""\
            [FILE: a.py]
            ```python
            import os
            ```
            [FILE: b.py]
            ```python
            import sys
            ```
        """)
        fixes = parse_llm_fixes(response, ["a.py", "b.py"])
        assert "a.py" in fixes
        assert "b.py" in fixes

    def test_fallback_to_python_blocks(self) -> None:
        response = textwrap.dedent("""\
            ```python
            import json
            x = 2
            ```
        """)
        fixes = parse_llm_fixes(response, ["only.py"])
        assert "only.py" in fixes
        assert "import json" in fixes["only.py"]

    def test_empty_response(self) -> None:
        assert parse_llm_fixes("nothing here", ["a.py"]) == {}

    def test_no_import_skipped(self) -> None:
        """Code blocks without import are treated as incomplete."""
        response = textwrap.dedent("""\
            [FILE: a.py]
            ```python
            x = 1
            ```
        """)
        fixes = parse_llm_fixes(response, ["a.py"])
        assert fixes == {}


# ---------------------------------------------------------------------------
# Static analysis: list_append_join detector
# ---------------------------------------------------------------------------

class TestListAppendJoinDetector:
    """Verify the detector flags the unfixed code and clears after the fix."""

    UNFIXED: Final[str] = textwrap.dedent("""\
        def build_messages(contexts):
            pending = []
            for ctx in contexts:
                pending.append({"role": "system", "content": ctx})
            return pending
    """)

    FIXED: Final[str] = textwrap.dedent("""\
        def build_messages(contexts):
            return [{"role": "system", "content": ctx} for ctx in contexts]
    """)

    def test_unfixed_code_triggers_detector(self) -> None:
        findings = detect_list_append_join(self.UNFIXED)
        assert len(findings) >= 1
        assert findings[0][1] == "list_append_join"

    def test_fixed_code_no_findings(self) -> None:
        findings = detect_list_append_join(self.FIXED)
        assert findings == []

    def test_full_analyze_unfixed_has_pattern(self) -> None:
        results = static_analyze(self.UNFIXED)
        patterns = [r["pattern"] for r in results]
        assert "list_append_join" in patterns

    def test_full_analyze_fixed_has_no_pattern(self) -> None:
        results = static_analyze(self.FIXED)
        patterns = [r["pattern"] for r in results]
        assert "list_append_join" not in patterns


# ---------------------------------------------------------------------------
# show_file_diff
# ---------------------------------------------------------------------------

class TestShowFileDiff:
    def test_shows_added_and_removed_lines(self, capsys: pytest.CaptureFixture[str]) -> None:
        show_file_diff("test.py", "line1\nline2\n", "line1\nline3\n")
        out = capsys.readouterr().out
        assert "line2" in out
        assert "line3" in out
        assert "1 lines removed" in out
        assert "1 lines added" in out

    def test_identical_files_no_diff(self, capsys: pytest.CaptureFixture[str]) -> None:
        show_file_diff("test.py", "same\n", "same\n")
        out = capsys.readouterr().out
        assert "no changes" in out

    def test_empty_original_all_additions(self, capsys: pytest.CaptureFixture[str]) -> None:
        show_file_diff("test.py", "", "new_line\n")
        out = capsys.readouterr().out
        assert "new_line" in out
        assert "0 lines removed" in out
        assert "1 lines added" in out

    def test_shows_side_by_side_columns(self, capsys: pytest.CaptureFixture[str]) -> None:
        original = "line1\nline2\nline3\nline4\n"
        new = "line1\nmodified\nline3\nline4\n"
        show_file_diff("test.py", original, new)
        out = capsys.readouterr().out
        assert "@@" in out
        assert "line2" in out
        assert "modified" in out

    def test_context_lines_show_both_columns(self, capsys: pytest.CaptureFixture[str]) -> None:
        original = "a\nb\nc\n"
        new = "a\nX\nc\n"
        show_file_diff("test.py", original, new)
        out = capsys.readouterr().out
        assert "a" in out
        assert "c" in out

    def test_removed_has_red_color(self, capsys: pytest.CaptureFixture[str]) -> None:
        show_file_diff("test.py", "line1\nline2\n", "line1\nline3\n")
        out = capsys.readouterr().out
        assert "\033[41m" in out

    def test_added_has_green_color(self, capsys: pytest.CaptureFixture[str]) -> None:
        show_file_diff("test.py", "line1\nline2\n", "line1\nline3\n")
        out = capsys.readouterr().out
        assert "\033[42m" in out
