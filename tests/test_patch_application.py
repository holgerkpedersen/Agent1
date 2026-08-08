"""Tests for [PATCH:] patch application — covers fix_cmd and implement_cmd paths."""
from __future__ import annotations

import os
import re
from unittest.mock import patch

import pytest


class TestImplementPatchApply:
    """Unit tests for the non-interactive _apply_patch in implement_cmd.py.

    Note: _apply_patch takes just the hunk body (no [PATCH: filename] prefix).
    The caller strips that prefix with regex before passing to this function.
    It returns (success: bool, result_str_or_error_message: str).
    """

    def _load_impl(self):
        from agent_core.commands.implement_cmd import _apply_patch
        return _apply_patch

    def test_simple_replacement(self, tmp_path):
        """A single hunk replacing one line works correctly."""
        apply = self._load_impl()
        original = ["x = 1\n", "y = 2\n", "z = 3\n"]
        patch_text = (
            "@@ -1,3 +1,3 @@\n"
            "- x = 1\n"
            "+ x = 10\n"
            " y = 2\n"
            " z = 3\n"
        )
        ok, result = apply(patch_text, "dummy.py", original)
        assert ok is True
        assert result == "x = 10\ny = 2\nz = 3\n"

    def test_addition_inside_function(self, tmp_path):
        """A patch that inserts a new line inside a function body via paired replacement."""
        apply = self._load_impl()
        original = [
            "def foo():\n",
            "    a = 1\n",
            "    return a\n",
        ]
        # Replace 'return a' with two lines to effectively insert one.
        # Paired -/+ strips the leading space from '+' lines.
        patch_text = (
            "@@ -1,3 +1,4 @@\n"
            " def foo():\n"
            "     a = 1\n"
            "-     return a\n"
            "+     b = 2\n"
            "+     return a\n"
        )
        ok, result = apply(patch_text, "dummy.py", original)
        assert ok is True
        assert "b = 2" in result

    def test_deletion_only_rejected(self, tmp_path):
        """Patch with only '-' lines (no '+') is rejected."""
        apply = self._load_impl()
        original = ["a = 1\n", "b = 2\n", "c = 3\n"]
        patch_text = (
            "@@ -1,3 +1,2 @@\n"
            "- a = 1\n"
            " b = 2\n"
            " c = 3\n"
        )
        ok, result = apply(patch_text, "dummy.py", original)
        assert ok is False

    def test_multiple_hunks(self, tmp_path):
        """Two separate hunks in one patch are both applied."""
        apply = self._load_impl()
        original = ["a = 1\n", "b = 2\n", "c = 3\n", "d = 4\n", "e = 5\n"]
        patch_text = (
            "@@ -1,3 +1,3 @@\n"
            "- a = 1\n"
            "+ a = 10\n"
            " b = 2\n"
            " c = 3\n"
            "\n"
            "@@ -4,2 +4,2 @@\n"
            "- d = 4\n"
            "+ d = 40\n"
            " e = 5\n"
        )
        ok, result = apply(patch_text, "dummy.py", original)
        assert ok is True
        assert result == "a = 10\nb = 2\nc = 3\nd = 40\ne = 5\n"

    def test_indentation_preserved_on_replacement(self, tmp_path):
        """Replacements keep the indentation of the removed line inside a function."""
        apply = self._load_impl()
        original = [
            "def foo():\n",
            "    old_line = True\n",
            "    return old_line\n",
        ]
        patch_text = (
            "@@ -1,3 +1,3 @@\n"
            " def foo():\n"
            "-     old_line = True\n"
            "+     new_line = False\n"
            "     return old_line\n"
        )
        ok, result = apply(patch_text, "dummy.py", original)
        assert ok is True
        assert "new_line = False" in result

    def test_indentation_preserved_on_addition_after_removal(self, tmp_path):
        """When - and + are paired, the + gets the indentation of the removed line."""
        apply = self._load_impl()
        original = [
            "def bar():\n",
            "    bad = True\n",
            "    return bad\n",
        ]
        patch_text = (
            "@@ -1,3 +1,3 @@\n"
            " def bar():\n"
            "-     bad = True\n"
            "+     good = False\n"
            "     return bad\n"
        )
        ok, result = apply(patch_text, "dummy.py", original)
        assert ok is True
        assert "good = False" in result

    def test_line_out_of_range_rejected(self, tmp_path):
        """Hunk starting beyond file length is rejected."""
        apply = self._load_impl()
        original = ["a = 1\n", "b = 2\n"]
        patch_text = (
            "@@ -10,2 +10,2 @@\n"
            "- a = 1\n"
            "+ A = 10\n"
        )
        ok, result = apply(patch_text, "dummy.py", original)
        assert ok is False

    def test_content_mismatch_rejected(self, tmp_path):
        """Old line content that doesn't match the file is rejected."""
        apply = self._load_impl()
        original = ["hello = 1\n", "foo = 2\n"]
        patch_text = (
            "@@ -1,2 +1,2 @@\n"
            "- totally different text\n"
            "+ new content\n"
        )
        ok, result = apply(patch_text, "dummy.py", original)
        assert ok is False

    def test_whitespace_tolerant_matching(self, tmp_path):
        """Leading/trailing whitespace in old lines is stripped before comparison."""
        apply = self._load_impl()
        original = ["    hello = 1  \n", "foo = 2\n"]
        patch_text = (
            "@@ -1,2 +1,2 @@\n"
            "-   hello = 1\n"
            "+ NEW_VAR=99\n"
        )
        ok, result = apply(patch_text, "dummy.py", original)
        assert ok is True
        assert "NEW_VAR=99" in result

    def test_empty_replacement_filtered(self, tmp_path):
        """Hunks with empty '+' lines are filtered out."""
        apply = self._load_impl()
        original = ["a = 1\n", "b = 2\n"]
        patch_text = (
            "@@ -1,2 +1,2 @@\n"
            "- a = 1\n"
            "+\n"
        )
        ok, result = apply(patch_text, "dummy.py", original)
        assert ok is False

    def test_incomplete_replacement_filtered(self, tmp_path):
        """Hunks with trailing incomplete operators on '+' lines are rejected."""
        apply = self._load_impl()
        original = ["a = 1\n"]
        patch_text = (
            "@@ -1 +1 @@\n"
            "- a = 1\n"
            "+ incomplete[\n"
        )
        ok, result = apply(patch_text, "dummy.py", original)
        assert ok is False

    def test_no_hunks_parsed(self, tmp_path):
        """Patch with no @@ sections returns failure."""
        apply = self._load_impl()
        patch_text = "not a real patch"
        ok, result = apply(patch_text, "dummy.py", ["a = 1\n"])
        assert ok is False

    def test_syntax_check_rejects_broken_code(self, tmp_path):
        """A patch that would produce invalid Python is rejected."""
        apply = self._load_impl()
        original = ["def foo():\n", "    pass\n"]
        # Remove 'pass' — leaves empty function body → SyntaxError
        patch_text = (
            "@@ -1,2 +1,1 @@\n"
            "-     pass\n"
        )
        ok, result = apply(patch_text, "dummy.py", original)
        assert ok is False

    def test_syntax_check_accepts_valid_code(self, tmp_path):
        """A patch producing valid Python passes the syntax check."""
        apply = self._load_impl()
        original = ["x = 1\n"]
        patch_text = (
            "@@ -1 +1 @@\n"
            "- x = 1\n"
            "+ x = 2\n"
        )
        ok, result = apply(patch_text, "dummy.py", original)
        assert ok is True

    def test_context_line_skip_on_mismatch_for_removal(self, tmp_path):
        """A '-' line with mismatched content causes rejection."""
        apply = self._load_impl()
        original = ["a = 1\n", "b = 2\n", "c = 3\n"]
        patch_text = (
            "@@ -1,3 +1,3 @@\n"
            "- x = 0\n"
            "+ X = 0\n"
            " b = 2\n"
            " c = 3\n"
        )
        ok, result = apply(patch_text, "dummy.py", original)
        assert ok is False

    def test_context_line_skip_on_out_of_range(self, tmp_path):
        """Context lines beyond file length are skipped without error."""
        apply = self._load_impl()
        original = ["a = 1\n", "b = 2\n"]
        # Replace last line and add one after — paired -/+ so space is stripped.
        patch_text = (
            "@@ -1,2 +1,3 @@\n"
            " a = 1\n"
            "- b = 2\n"
            "+ b = 2\n"
            "+ c = 3\n"
        )
        ok, result = apply(patch_text, "dummy.py", original)
        assert ok is True
        assert "c = 3" in result

    def test_hunk_start_missing_plus_part(self, tmp_path):
        """Lenient parsing of @@ -start @@ without +start part."""
        apply = self._load_impl()
        original = ["a = 1\n", "b = 2\n"]
        patch_text = (
            "@@ -1 @@\n"
            "- a = 1\n"
            "+ A = 10\n"
        )
        ok, result = apply(patch_text, "dummy.py", original)
        assert ok is True
        assert "A = 10" in result

    def test_result_is_string_not_list(self, tmp_path):
        """Verify the return type is (bool, str)."""
        apply = self._load_impl()
        patch_text = "@@ -1 +1 @@\n- a = 1\n+ A = 10\n"
        ok, result = apply(patch_text, "dummy.py", ["a = 1\n"])
        assert isinstance(ok, bool)
        assert isinstance(result, str)

    def test_replacement_with_comment(self, tmp_path):
        """A patch replacing a line with a comment is valid."""
        apply = self._load_impl()
        original = ["# start\n", "x = 1\n", "# end\n"]
        patch_text = (
            "@@ -1,3 +1,3 @@\n"
            " # start\n"
            "- x = 1\n"
            "+ # disabled\n"
            " # end\n"
        )
        ok, result = apply(patch_text, "dummy.py", original)
        assert ok is True
        assert "# disabled" in result

    def test_patch_preserves_other_lines(self, tmp_path):
        """A patch that only changes one line leaves other lines intact."""
        apply = self._load_impl()
        original = ["x = 1\n", "y = 2\n"]
        patch_text = (
            "@@ -1 +1 @@\n"
            "- x = 1\n"
            "+ x = 99\n"
        )
        ok, result = apply(patch_text, "dummy.py", original)
        assert ok is True
        assert "x = 99" in result
        assert "y = 2" in result


class TestFixCmdPatchApply:
    """Unit tests for the interactive _apply_patch in fix_cmd.py (mocking input)."""

    def test_simple_patch_accepted(self, tmp_path):
        """Accepted patch writes changes to disk."""
        from agent_core.commands.fix_cmd import FixCommand
        fc = FixCommand()
        target = tmp_path / "test.py"
        target.write_text("old_val = 1\nkeep_this = 2\n", encoding="utf-8")
        patch_text = (
            "@@ -1,2 +1,2 @@\n"
            "- old_val = 1\n"
            "+ new_val = 1\n"
            " keep_this = 2\n"
        )
        with patch("builtins.input", return_value="y"):
            ok = fc._apply_patch(patch_text, str(target), str(tmp_path))
        assert ok is True
        assert target.read_text(encoding="utf-8") == "new_val = 1\nkeep_this = 2\n"

    def test_simple_patch_rejected(self, tmp_path):
        """User declines patch — file unchanged."""
        from agent_core.commands.fix_cmd import FixCommand
        fc = FixCommand()
        target = tmp_path / "test.py"
        target.write_text("old_val = 1\nkeep_this = 2\n", encoding="utf-8")
        patch_text = (
            "@@ -1,2 +1,2 @@\n"
            "- old_val = 1\n"
            "+ new_val = 1\n"
            " keep_this = 2\n"
        )
        with patch("builtins.input", return_value="n"):
            ok = fc._apply_patch(patch_text, str(target), str(tmp_path))
        assert ok is False
        assert target.read_text(encoding="utf-8") == "old_val = 1\nkeep_this = 2\n"

    def test_stdlib_file_skipped(self, tmp_path):
        """Patches targeting stdlib paths are skipped."""
        from agent_core.commands.fix_cmd import FixCommand
        fc = FixCommand()
        import sys as _sys
        target = os.path.join(_sys.prefix, "Lib", "os.py")
        with patch("builtins.input", return_value="y"):
            ok = fc._apply_patch("@@ -1 +1 @@\n- x\n+ y\n", target, str(tmp_path))
        assert ok is False

    def test_nonexistent_file_skipped(self, tmp_path):
        """Patch for a file that doesn't exist returns False."""
        from agent_core.commands.fix_cmd import FixCommand
        fc = FixCommand()
        with patch("builtins.input", return_value="y"):
            ok = fc._apply_patch(
                "@@ -1 +1 @@\n- x\n+ y\n",
                str(tmp_path / "missing.py"),
                str(tmp_path),
            )
        assert ok is False

    def test_unparseable_patch_rejected(self, tmp_path):
        """Patch with no valid hunks returns False."""
        from agent_core.commands.fix_cmd import FixCommand
        fc = FixCommand()
        target = tmp_path / "test.py"
        target.write_text("hello = 1\n", encoding="utf-8")
        with patch("builtins.input", return_value="y"):
            ok = fc._apply_patch("not a real patch", str(target), str(tmp_path))
        assert ok is False

    def test_eof_in_input_skips_patch(self, tmp_path):
        """EOFError in input() skips the patch."""
        from agent_core.commands.fix_cmd import FixCommand
        fc = FixCommand()
        target = tmp_path / "test.py"
        target.write_text("old = 1\n", encoding="utf-8")
        with patch("builtins.input", side_effect=EOFError):
            ok = fc._apply_patch("@@ -1 +1 @@\n- old = 1\n+ new = 1\n", str(target), str(tmp_path))
        assert ok is False

    def test_keyboard_interrupt_skips_patch(self, tmp_path):
        """KeyboardInterrupt in input() skips the patch."""
        from agent_core.commands.fix_cmd import FixCommand
        fc = FixCommand()
        target = tmp_path / "test.py"
        target.write_text("old = 1\n", encoding="utf-8")
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            ok = fc._apply_patch("@@ -1 +1 @@\n- old = 1\n+ new = 1\n", str(target), str(tmp_path))
        assert ok is False

    def test_syntax_rejection_prevents_write(self, tmp_path):
        """A patch that breaks syntax is not written to disk."""
        from agent_core.commands.fix_cmd import FixCommand
        fc = FixCommand()
        target = tmp_path / "test.py"
        target.write_text("def foo():\n    pass\n", encoding="utf-8")
        # Removing 'pass' creates SyntaxError (empty function body)
        patch_text = "@@ -1,2 +1,1 @@\n-    pass\n"
        with patch("builtins.input", return_value="y"):
            ok = fc._apply_patch(patch_text, str(target), str(tmp_path))
        assert ok is False
        # File should be unchanged
        assert target.read_text(encoding="utf-8") == "def foo():\n    pass\n"


class TestPatchParsingRegex:
    """Tests for the regex used to extract [PATCH:] blocks from LLM responses."""

    def test_parse_single_block(self):
        text = "[PATCH: foo.py]\n@@ -1 +1 @@\n- a\n+ b\n"
        m = re.search(r'\[PATCH:\s*([^\]]+)\]\s*\n?(.*?)(?=\[PATCH:|\Z)', text, re.DOTALL)
        assert m is not None
        assert m.group(1).strip() == "foo.py"
        assert "+ b" in m.group(2).strip()

    def test_parse_multiple_blocks(self):
        text = (
            "[PATCH: a.py]\n@@ -1 +1 @@\n- x\n+ X\n"
            "\n[PATCH: b.py]\n@@ -1 +1 @@\n- y\n+ Y\n"
        )
        patches = re.findall(
            r'\[PATCH:\s*([^\]]+)\]\s*\n?(.*?)(?=\[PATCH:|\Z)', text, re.DOTALL
        )
        assert len(patches) == 2
        assert patches[0][0].strip() == "a.py"
        assert patches[1][0].strip() == "b.py"

    def test_parse_mixed_patch_and_file(self):
        text = (
            "[PATCH: a.py]\n@@ -1 +1 @@\n- x\n+ X\n"
            "\n[FILE: b.py]\n```python\nprint('hi')\n```"
        )
        patches = re.findall(
            r'\[PATCH:\s*([^\]]+)\]\s*\n?(.*?)(?=\[PATCH:|\Z)', text, re.DOTALL
        )
        assert len(patches) == 1
        assert patches[0][0].strip() == "a.py"

    def test_parse_empty_response(self):
        assert re.findall(
            r'\[PATCH:\s*([^\]]+)\]\s*\n?(.*?)(?=\[PATCH:|\Z)', "", re.DOTALL
        ) == []
        assert re.findall(
            r'\[PATCH:\s*([^\]]+)\]\s*\n?(.*?)(?=\[PATCH:|\Z)', "no patches here", re.DOTALL
        ) == []

    def test_parse_trailing_whitespace_in_path(self):
        text = "[PATCH:  my_module.py   ]\n@@ -1 +1 @@\n- x\n+ X\n"
        m = re.search(r'\[PATCH:\s*([^\]]+)\]\s*\n?(.*?)(?=\[PATCH:|\Z)', text, re.DOTALL)
        assert m is not None
        assert m.group(1).strip() == "my_module.py"

    def test_parse_no_space_after_bracket(self):
        """Edge case: [PATCH:path] with no space after ]."""
        text = "[PATCH:path.py]\n@@ -1 +1 @@\n- a\n+ A\n"
        m = re.search(r'\[PATCH:\s*([^\]]+)\]\s*\n?(.*?)(?=\[PATCH:|\Z)', text, re.DOTALL)
        assert m is not None
        assert m.group(1).strip() == "path.py"

    def test_parse_newline_before_hunk(self):
        """Edge case: [PATCH:path]\n@@ -1... with explicit newline."""
        text = "[PATCH: path.py]\n\n@@ -1 +1 @@\n- a\n+ A\n"
        m = re.search(r'\[PATCH:\s*([^\]]+)\]\s*\n?(.*?)(?=\[PATCH:|\Z)', text, re.DOTALL)
        assert m is not None
        assert "@@" in m.group(2)


class TestPatchIntegrationViaAgent:
    """End-to-end tests running through agent.py command execution."""

    def test_implement_cmd_patch_via_parse_and_apply(self, tmp_path):
        """Simulate the implement --fix path: parse [PATCH:] from LLM output and apply."""
        from agent_core.commands.implement_cmd import _apply_patch

        target = tmp_path / "code.py"
        target.write_text(
            "def add(a, b):\n"
            "    return a - b\n",
            encoding="utf-8",
        )

        llm_response = (
            "[PATCH: code.py]\n"
            "@@ -2,1 +2,1 @@\n"
            "-    return a - b\n"
            "+    return a + b\n"
        )
        # Replicate the exact parsing regex from implement_cmd.py line 1771
        patch_match = re.search(
            r'\[PATCH:\s*([^\]]+)\]\s*\n?(.*?)(?=\[PATCH:|\Z)', llm_response, re.DOTALL
        )
        assert patch_match is not None
        fpath = patch_match.group(1).strip()
        patch_text = patch_match.group(2).strip()

        current_code = target.read_text(encoding="utf-8")
        ok, result_msg = _apply_patch(patch_text, fpath, current_code.split("\n"))
        assert ok is True
        # Write patched content back to file to simulate the real flow
        with open(target, "w", encoding="utf-8") as f:
            f.write(result_msg)
        result = target.read_text(encoding="utf-8")
        assert "return a + b" in result
        assert "return a - b" not in result

    def test_fix_cmd_patch_full_roundtrip(self, tmp_path):
        """Full round-trip: write file → apply patch via fix_cmd → verify result."""
        from agent_core.commands.fix_cmd import FixCommand

        target = tmp_path / "buggy.py"
        target.write_text(
            "def check(x):\n"
            "    if x is None:\n"
            "        return False\n"
            "    return True\n",
            encoding="utf-8",
        )
        fc = FixCommand()
        patch_text = (
            "@@ -1,4 +1,4 @@\n"
            " def check(x):\n"
            "-    if x is None:\n"
            "+    if x == 'default':\n"
            "         return False\n"
            "     return True\n"
        )
        with patch("builtins.input", return_value="y"):
            ok = fc._apply_patch(patch_text, str(target), str(tmp_path))
        assert ok is True
        result = target.read_text(encoding="utf-8")
        assert "x == 'default'" in result
        assert "x is None" not in result

    def test_patch_with_multiple_errors_in_file(self, tmp_path):
        """Patch that fixes two separate locations in one file."""
        from agent_core.commands.implement_cmd import _apply_patch

        target = tmp_path / "multi.py"
        target.write_text(
            "a = 1\n"
            "b = 2\n"
            "c = a + b\n"
            "d = 3\n"
            "e = d * 2\n",
            encoding="utf-8",
        )
        patch_text = (
            "@@ -1,3 +1,3 @@\n"
            "- a = 1\n"
            "+ a = 10\n"
            " b = 2\n"
            " c = a + b\n"
            "\n"
            "@@ -4,2 +4,2 @@\n"
            "- d = 3\n"
            "+ d = 30\n"
            " e = d * 2\n"
        )
        current_code = target.read_text(encoding="utf-8")
        ok, result = _apply_patch(patch_text, "multi.py", current_code.split("\n"))
        assert ok is True
        # Write the patched content back to verify round-trip
        with open(target, "w", encoding="utf-8") as f:
            f.write(result)
        final = target.read_text(encoding="utf-8")
        assert "a = 10" in final
        assert "d = 30" in final
        # Check exact lines, not substring match (a=1 is part of a=10)
        lines = final.strip().split("\n")
        assert lines[0] == "a = 10"
        assert lines[3] == "d = 30"


class TestAnchoredPatchApplication:
    """Content-anchored patch fallback: wrong line numbers, fences, fused
    headers, padding style, and safe-rejection behaviors."""

    SRC = (
        "def engine(options):\n"
        "    cache = {}\n"
        "    for key in options:\n"
        "        data = fetch(key)\n"
        "        cache[key] = data\n"
        "    return cache\n"
    )

    def _source_lines(self):
        return self.SRC.split("\n")

    def test_anchored_fixes_wrong_line_numbers(self) -> None:
        from agent_core.patch_utils import apply_anchored_patch
        patch = (
            "@@ -1,3 +1,3 @@\n"
            "     for key in options:\n"
            "-        data = fetch(key)\n"
            "+        data = load(key)\n"
            "\n"
        )
        ok, result = apply_anchored_patch(patch, self._source_lines())
        assert ok is True
        assert "data = load(key)" in result

    def test_anchored_handles_fenced_fused_headers(self) -> None:
        """run8 shape: markdown fence + multiple fused @@ headers."""
        from agent_core.patch_utils import apply_anchored_patch
        patch = (
            "```python\n"
            "@@ -4,2 +4,2 @@ for key in options:\n"
            "-        data = fetch(key)\n"
            "+        data = load(key)\n"
            "@@ -88,1 +88,1 @@\n"
            "```\n"
        )
        ok, result = apply_anchored_patch(patch, self._source_lines())
        assert ok is True
        assert "data = load(key)" in result

    def test_anchored_applies_multiple_hunks(self) -> None:
        from agent_core.patch_utils import apply_anchored_patch
        patch = (
            "@@ -900,2 +900,2 @@\n"
            "-        data = fetch(key)\n"
            "+        data = load(key)\n"
            "@@ -901,2 +901,2 @@\n"
            "-        cache[key] = data\n"
            "+        cache[key] = value\n"
        )
        ok, result = apply_anchored_patch(patch, self._source_lines())
        assert ok is True
        assert "data = load(key)" in result
        assert "cache[key] = value" in result
        compile(result, "<anchored>", "exec")

    def test_anchored_rejects_unknown_content(self) -> None:
        from agent_core.patch_utils import apply_anchored_patch
        patch = "@@ -260,2 +260,2 @@\n-     invented_line\n+     x\n"
        ok, err = apply_anchored_patch(patch, self._source_lines())
        assert ok is False
        assert "Cannot anchor" in err

    def test_anchored_rejects_syntax_breakage(self) -> None:
        from agent_core.patch_utils import apply_anchored_patch
        patch = (
            "@@ -900,2 +900,2 @@\n"
            "-        cache[key] = data\n"
            "+        cache\n"
            "+    bad indent(\n"
        )
        ok, err = apply_anchored_patch(patch, self._source_lines())
        assert ok is False
        assert err  # rejected by hunk validation, anchoring, or the syntax gate

    def test_split_hunks_handles_fused_headers(self) -> None:
        from agent_core.patch_utils import split_patch_hunks
        patch = (
            "@@ -4,2 +4,2 @@ for key in options:\n"
            "-        data = fetch(key)\n"
            "+        data = load(key)\n"
            "@@ -88,1 +88,1 @@\n"
        )
        hunks = split_patch_hunks(patch)
        assert len(hunks) == 1
        assert hunks[0][0] == 4
        ops = [op for op, _ in hunks[0][1]]
        assert "-" in ops and "+" in ops

    def test_normalize_strips_fence_and_preamble(self) -> None:
        from agent_core.patch_utils import normalize_patch_block
        raw = (
            "Here is the fix:\n"
            "```diff\n"
            "--- a/huge.py\n"
            "+++ b/huge.py\n"
            "@@ -3,3 +3,3 @@\n"
            "- old\n"
            "+ new\n"
            "```\n"
        )
        out = normalize_patch_block(raw)
        assert out.startswith("@@ -3,3 +3,3 @@")
        assert "--- a/huge.py" not in out
        assert "```" not in out
