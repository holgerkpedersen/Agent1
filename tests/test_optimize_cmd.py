"""Tests for agent_core.commands.optimize_cmd — batching, parsing, and static analysis."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Final

import pytest

from agent_core.commands.base import show_file_diff
from agent_core.commands.optimize_cmd import (
    create_batches,
    estimate_tokens,
    format_batch_context,
    parse_llm_fixes,
    OPTIMIZE_RULES,
    SYSTEM_OVERHEAD_TOKENS,
    _surgically_revert_regressions,
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
# _request_max_tokens (context-aware output budget)
# ---------------------------------------------------------------------------

class TestRequestMaxTokens:
    def test_context_aware_budget_fits_window(self) -> None:
        """With a known context window the budget shrinks so prompt + output
        (plus the reserve) always fit — the HTTP-400 overflow is impossible."""
        from agent_core.commands.optimize_cmd import _request_max_tokens
        assert _request_max_tokens(20000, context_tokens=32768) <= 32768 - 20000

    def test_context_near_full_window_still_positive(self) -> None:
        from agent_core.commands.optimize_cmd import _request_max_tokens
        assert _request_max_tokens(32000, context_tokens=32768) >= 1

    def test_heuristic_budget_when_context_unknown(self) -> None:
        from agent_core.commands.optimize_cmd import _request_max_tokens
        assert _request_max_tokens(100) >= 8192

    def test_hard_ceiling_respected(self) -> None:
        from agent_core.commands.optimize_cmd import _request_max_tokens, REGION_HARD_MAX_TOKENS
        assert _request_max_tokens(100, context_tokens=1000000) <= REGION_HARD_MAX_TOKENS
        assert _request_max_tokens(100000000, context_tokens=None) == REGION_HARD_MAX_TOKENS


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
            x = os.getcwd()
            ```
        """)
        fixes, _ = parse_llm_fixes(response, ["a.py"])
        assert "a.py" in fixes
        assert "import os" in fixes["a.py"]

    def test_multiple_files(self) -> None:
        response = textwrap.dedent("""\
            [FILE: a.py]
            ```python
            import os
            x = os.getcwd()
            ```
            [FILE: b.py]
            ```python
            import sys
            print(sys.version)
            ```
        """)
        fixes, _ = parse_llm_fixes(response, ["a.py", "b.py"])
        assert "a.py" in fixes
        assert "b.py" in fixes

    def test_fallback_to_python_blocks(self) -> None:
        response = textwrap.dedent("""\
            ```python
            import json
            x = json.loads("{}")
            ```
        """)
        fixes, _ = parse_llm_fixes(response, ["only.py"])
        assert "only.py" in fixes
        assert "import json" in fixes["only.py"]

    def test_empty_response(self) -> None:
        fixes, _ = parse_llm_fixes("nothing here", ["a.py"])
        assert fixes == {}

    def test_code_without_imports_is_accepted(self) -> None:
        """The old 'must contain import' gate is dropped: small modules with
        no imports are now accepted (previously silently skipped forever)."""
        response = textwrap.dedent("""\
            [FILE: a.py]
            ```python
            x = 1
            ```
        """)
        fixes, _ = parse_llm_fixes(response, ["a.py"])
        assert "a.py" in fixes

    def test_syntax_error_is_rejected(self) -> None:
        response = textwrap.dedent("""\
            [FILE: a.py]
            ```python
            import os
            def broken(
            ```
        """)
        fixes, failures = parse_llm_fixes(response, ["a.py"])
        assert fixes == {}
        assert "a.py" in failures
        assert failures["a.py"][0]["pattern"] == "syntax_error"


# ---------------------------------------------------------------------------
# Static analysis: list_append_join detector
# ---------------------------------------------------------------------------

class TestListAppendJoinDetector:
    """Verify the detector flags the unfixed code and clears after the fix."""

    UNFIXED: Final[str] = textwrap.dedent("""\
        def build_messages(content_blocks):
            pending = []
            for block in content_blocks:
                pending.append(block)
            return "\\n".join(pending)
    """)

    FIXED: Final[str] = textwrap.dedent("""\
        def build_messages(content_blocks):
            return "\\n".join(block for block in content_blocks)
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

    def test_complex_loop_no_finding(self) -> None:
        """Complex loops with if/await should NOT be flagged.

        .append() in loops with conditional logic, awaits, or breaks
        cannot be easily replaced with a list comprehension.
        """
        code = textwrap.dedent("""\
            async def run_conversation(messages):
                all_text = []
                for msg in messages:
                    text, updated = await chat(msg)
                    if text:
                        all_text.append(text)
                    else:
                        break
                return "\\n".join(all_text)
        """)
        findings = detect_list_append_join(code)
        assert findings == []

    def test_tool_loop_no_finding(self) -> None:
        """Real-world case: tool_loop.py should not be flagged."""
        from pathlib import Path
        code = Path("agent_core/llm/tool_loop.py").read_text()
        findings = detect_list_append_join(code)
        assert findings == []

    def test_append_feed_sorted_set_not_flagged(self) -> None:
        """Loop-built list consumed by sorted()/set()/return is NOT the
        append->join pattern and must not trigger a rewrite finding."""
        code = textwrap.dedent("""\
            import re

            def _parse_imports(source):
                result = []
                for m in re.finditer(r"^(?:from|import)\\s+(\\S+)", source, re.MULTILINE):
                    module = m.group(1)
                    if module.startswith("."):
                        continue
                    result.append(module)
                return sorted(set(result))
        """)
        findings = detect_list_append_join(code)
        assert findings == []

    def test_append_join_custom_separator_flagged(self) -> None:
        code = textwrap.dedent("""\
            def f(items):
                out = []
                for x in items:
                    out.append(str(x))
                return ",".join(out)
        """)
        findings = detect_list_append_join(code)
        assert any(f[1] == "list_append_join" for f in findings)


# ---------------------------------------------------------------------------
# Loop-scope detectors: dedent ends loop (not just column-0)
# ---------------------------------------------------------------------------

class TestLoopScopeDedent:
    """Regression guard for the column-0-only loop-exit false-positive bug.

    Indented code that follows a for/while loop inside a method body must NOT
    be treated as still-inside-the-loop.
    """

    def test_post_loop_concat_in_method_not_flagged(self) -> None:
        """`combined += "".join(...)` placed after a loop, inside a method,
        must not trigger string_concat_in_loop (the bug from analyze_cmd.py)."""
        from agent_core.patterns import detect_string_concat_in_loop, detect_file_read_in_loop, detect_regex_in_loop
        code = textwrap.dedent("""\
            async def _deep_analyze(self, path):
                combined = content
                for ref in refs:
                    if len(ref_candidates) >= 4:
                        break
                    full = os.path.join(ws, ref)
                combined += "".join(f"\\n# {p}" for p, c in parts)
                user = f"## Code:\\n\\n{combined}"
                return user
        """)
        assert detect_string_concat_in_loop(code) == []
        assert detect_regex_in_loop(code) == []
        assert detect_file_read_in_loop(code) == []

    def test_genuine_in_loop_still_flagged(self) -> None:
        """A real `path += ".x"` inside a loop body is still detected."""
        from agent_core.patterns import detect_string_concat_in_loop
        code = textwrap.dedent("""\
            def build(rows):
                for r in rows:
                    path += ".x"
                return path
        """)
        assert len(detect_string_concat_in_loop(code)) == 1

    def test_genuine_file_read_in_loop_still_flagged(self) -> None:
        """A real repeated read of the same file inside a loop is detected."""
        from agent_core.patterns import detect_file_read_in_loop
        code = textwrap.dedent("""\
            def collect():
                cfg = "config.json"
                for p in paths:
                    data = open(cfg).read()
                return data
        """)
        assert len(detect_file_read_in_loop(code)) == 1

    def test_distinct_file_reads_in_loop_not_flagged(self) -> None:
        """Reads of distinct files per iteration are NOT flagged: nothing to
        hoist, so the suggested fix does not apply."""
        from agent_core.patterns import detect_file_read_in_loop
        code = textwrap.dedent("""\
            def collect():
                for p in paths:
                    data = open(p).read()
                return data
        """)
        assert detect_file_read_in_loop(code) == []

    def test_regex_in_loop_still_flagged(self) -> None:
        from agent_core.patterns import detect_regex_in_loop
        code = textwrap.dedent("""\
            for i in range(3):
                re.compile("^x")
        """)
        assert len(detect_regex_in_loop(code)) == 1

    def test_concatenation_after_loop_in_method_not_flagged(self) -> None:
        """Regression: a string ``+=`` after a loop, still inside a method
        body, must not be falsely reported (the column-0 bug)."""
        from agent_core.patterns import detect_string_concat_in_loop
        code = textwrap.dedent("""\
            class C:
                def f(self):
                    for r in refs:
                        combined += r
                    combined += "tail"
                    return combined
        """)
        assert detect_string_concat_in_loop(code) == []

    def test_file_read_in_loop_flagged_unless_cached(self) -> None:
        from agent_core.patterns import detect_file_read_in_loop
        # cached reads must not be flagged
        cached = textwrap.dedent("""\
            async def f():
                read_cache: dict[str, str] = {}
                for p in paths:
                    if p in read_cache:
                        data = read_cache[p]
                    else:
                        data = await read_file(p)
                        read_cache[p] = data
                return data
        """)
        assert detect_file_read_in_loop(cached) == []
        # repeated reads of a *constant* path must still be flagged
        plain = textwrap.dedent("""\
            def f():
                cfg = "config.json"
                for p in paths:
                    data = open(cfg).read()
                return data
        """)
        assert len(detect_file_read_in_loop(plain)) == 1

    def test_file_read_in_loop_not_flagged_when_derived(self) -> None:
        """Reads of paths derived from the iteration variable (e.g. inside
        ``os.walk``, or ``tmp = os.path.join(root, f)``) must not be flagged:
        each iteration reads a distinct file."""
        from agent_core.patterns import detect_file_read_in_loop
        walked = textwrap.dedent("""\
            import os
            def collect():
                for root, dirs, files in os.walk(ws):
                    if ".git" in root:
                        continue
                    for f in files:
                        fp = os.path.normpath(os.path.join(root, f))
                        with open(fp, "r", encoding="utf-8") as sf:
                            content = sf.read()
        """)
        assert detect_file_read_in_loop(walked) == []
        read_file = textwrap.dedent("""\
            def f():
                for p in paths:
                    data = await read_file(os.path.join(base, p))
                return data
        """)
        assert detect_file_read_in_loop(read_file) == []

    def test_file_read_through_with_alias_not_flagged(self) -> None:
        """``with open(path) as fh: fh.read()`` — the path used on the
        ``.read()`` line is the loop variable, so not flagged."""
        from agent_core.patterns import detect_file_read_in_loop
        code = textwrap.dedent("""\
            def f():
                for p in paths:
                    with open(p) as fh:
                        data = fh.read()
                return data
        """)
        assert detect_file_read_in_loop(code) == []

    def test_transitive_derived_path_not_flagged(self) -> None:
        """``rel = os.path.join(root, f); full = os.path.join(base, rel)``:
        the second assignment is derived transitively from the loop vars."""
        from agent_core.patterns import detect_file_read_in_loop
        code = textwrap.dedent("""\
            def f():
                for root, f in files:
                    rel = os.path.join(root, f)
                    full = os.path.join(base, rel)
                    data = open(full).read()
                return data
        """)
        assert detect_file_read_in_loop(code) == []


# ---------------------------------------------------------------------------
# New detectors: dead_assignment, unreachable_code, unused_imports
# ---------------------------------------------------------------------------

class TestNewDetectors:
    """Tests for dead_assignment, unreachable_code, and unused_imports."""

    def test_dead_assignment_detected(self) -> None:
        from agent_core.patterns import detect_dead_assignment
        code = textwrap.dedent("""\
            x = [i for i in range(5)]
            x = []
            print(x)
        """)
        findings = detect_dead_assignment(code)
        assert len(findings) == 1
        assert findings[0][1] == "dead_assignment"

    def test_dead_assignment_no_false_positive(self) -> None:
        from agent_core.patterns import detect_dead_assignment
        code = textwrap.dedent("""\
            x = 1
            y = 2
            return x
        """)
        findings = detect_dead_assignment(code)
        assert findings == []

    def test_dead_assignment_overwrite_reading_rhs_not_flagged(self) -> None:
        from agent_core.patterns import detect_dead_assignment
        code = textwrap.dedent("""\
            ws = agent.workspace
            ws = to_windows_path(ws)
            ws_path = Path(ws)
        """)
        findings = detect_dead_assignment(code)
        assert findings == []

    def test_dead_assignment_overwrite_reading_rhs_clamp_not_flagged(self) -> None:
        from agent_core.patterns import detect_dead_assignment
        code = textwrap.dedent("""\
            max_old_no = max(old_lineno, 0)
            max_old_no = max(max_old_no, 4)
            print(max_old_no)
        """)
        findings = detect_dead_assignment(code)
        assert findings == []

    def test_unreachable_code_detected(self) -> None:
        from agent_core.patterns import detect_unreachable_code
        code = textwrap.dedent("""\
            def foo():
                return 1
                x = 2  # unreachable
        """)
        findings = detect_unreachable_code(code)
        assert len(findings) == 1
        assert findings[0][1] == "unreachable_code"

    def test_unreachable_code_no_false_positive(self) -> None:
        from agent_core.patterns import detect_unreachable_code
        code = textwrap.dedent("""\
            def foo():
                if x:
                    return 1
                else:
                    return 2
        """)
        findings = detect_unreachable_code(code)
        assert findings == []

    def test_unreachable_code_multiline_return_no_false_positive(self) -> None:
        """Multi-line return statements (unclosed brackets) should not flag
        continuation lines as unreachable."""
        from agent_core.patterns import detect_unreachable_code
        code = textwrap.dedent("""\
            def _parse_imports(source):
                result = []
                for m in re.finditer(r"x", source):
                    module = m.group(1)
                    result.append(module)
                return sorted(set(
                    result
                ))


            def _parse_file_refs(text):
                refs = []
                for m in re.finditer(r"y", text):
                    refs.append(m.group(0))
                return sorted(set(
                    refs
                ))
        """)
        findings = detect_unreachable_code(code)
        assert findings == []

    def test_unused_imports_detected(self) -> None:
        from agent_core.patterns import detect_unused_imports
        code = textwrap.dedent("""\
            import os
            import re
            print(re.findall(r"a", "abc"))
        """)
        findings = detect_unused_imports(code)
        assert len(findings) == 1
        assert findings[0][1] == "unused_import"
        assert "os" in findings[0][2]

    def test_unused_imports_with_alias(self) -> None:
        from agent_core.patterns import detect_unused_imports
        code = textwrap.dedent("""\
            import os as operating_system
            print(operating_system.getcwd())
        """)
        findings = detect_unused_imports(code)
        assert findings == []

    def test_dead_assignment_never_used_detected(self) -> None:
        """An assignment inside a function whose name is never referenced again
        — reproducing the LLM's dead `combined`-store pattern — must flag."""
        from agent_core.patterns import detect_dead_assignment
        code = textwrap.dedent("""\
            async def _deep_analyze(path):
                combined_parts = [path]
                combined_parts.append(x)
                combined = chr(10).join(combined_parts)
                msg = combined
                combined_parts.append(y)
                combined = chr(10).join(combined_parts)
                return msg
        """)
        findings = detect_dead_assignment(code)
        # exactly one dead store: the last `combined = ...`
        dead = [f for f in findings if "never used" in f[2]]
        assert len(dead) == 1
        assert dead[0][1] == "dead_assignment"
        assert "combined" in dead[0][2]

    def test_dead_assignment_annotated_store_detected(self) -> None:
        """Annotated assignment (`x: T = ...`) that is never used must flag."""
        from agent_core.patterns import detect_dead_assignment
        code = textwrap.dedent("""\
            async def f():
                combined_parts: list[str] = [x]
                combined: str = chr(10).join(combined_parts)
                return combined_parts
        """)
        findings = detect_dead_assignment(code)
        dead = [f for f in findings if "never used" in f[2]]
        assert len(dead) == 1
        assert dead[0][1] == "dead_assignment"
        assert "combined" in dead[0][2]

    def test_dead_assignment_module_level_not_flagged(self) -> None:
        """Module-level assignments may be imported elsewhere — skip them."""
        from agent_core.patterns import detect_dead_assignment
        code = textwrap.dedent("""\
            API_KEY = os.environ["KEY"]
            def helper():
                return 1
        """)
        findings = detect_dead_assignment(code)
        dead = [f for f in findings if "never used" in f[2]]
        assert dead == []

    def test_dead_assignment_underscore_skip(self) -> None:
        """Conventionally-unused (`_`) locals are intentionally ignored."""
        from agent_core.patterns import detect_dead_assignment
        code = textwrap.dedent("""\
            async def f():
                _unused = 1
                _x = compute()
                return 2
        """)
        findings = detect_dead_assignment(code)
        dead = [f for f in findings if "never used" in f[2]]
        assert dead == []

    def test_dead_assignment_used_in_fstring_not_flagged(self) -> None:
        """Usage inside an f-string counts as a real use."""
        from agent_core.patterns import detect_dead_assignment
        code = textwrap.dedent("""\
            async def f():
                name = "x"
                return f"hi {name}"
        """)
        findings = detect_dead_assignment(code)
        assert findings == []

    def test_dead_assignment_kwargs_in_multiline_call_not_flagged(self) -> None:
        """Kwargs like ``capture_output=True`` inside a multi-line call are
        not assignments — a kwarg-context false positive regression test."""
        from agent_core.patterns import detect_dead_assignment
        code = textwrap.dedent("""\
            def f(path):
                result = subprocess.run(
                    ["python", "-m", "py_compile", path],
                    capture_output=True,
                    text=True,
                )
                return result.returncode
        """)
        findings = detect_dead_assignment(code)
        assert findings == []

    def test_dead_assignment_kwargs_inline_call_not_flagged(self) -> None:
        """Single-line kwarg calls must also stay clean: the detector's
        bracket-depth tracking covers both single- and multi-line forms."""
        from agent_core.patterns import detect_dead_assignment
        code = textwrap.dedent("""\
            def f():
                r = subprocess.run(
                    ["ls"], capture_output=True, text=True
                )
                return r
        """)
        findings = detect_dead_assignment(code)
        assert findings == []

    def test_dead_assignment_loop_back_edge_not_flagged(self) -> None:
        """A store at the end of a loop body feeds a read on the next
        iteration (e.g. ``prev_error_sigs = dict(...)`` compared at the top
        of the body).  Such loop-state variables are live, never dead."""
        from agent_core.patterns import detect_dead_assignment
        code = textwrap.dedent("""\
            def fix_signatures(tree):
                prev_error_sigs = {}
                for node in tree:
                    if signature(node) in prev_error_sigs:
                        continue
                    prev_error_sigs = dict(node.error_sigs)
                return tree
        """)
        findings = detect_dead_assignment(code)
        assert not any(f[1] == "dead_assignment" for f in findings)

    def test_dead_assignment_inside_loop_dead_store_still_detected(self) -> None:
        """A store inside a loop that no other line of the loop body reads is
        still a dead store — the back-edge rule only suppresses real loop state."""
        from agent_core.patterns import detect_dead_assignment
        code = textwrap.dedent("""\
            def walk(items):
                total = 0
                for item in items:
                    scratch = item * 2
                return total
        """)
        findings = detect_dead_assignment(code)
        assert any(f[1] == "dead_assignment" for f in findings)


# ---------------------------------------------------------------------------
# LLM output validation
# ---------------------------------------------------------------------------

class TestValidateLLMCode:
    """Tests for validate_llm_code blocking LLM output with quality issues."""

    def test_silent_except_blocked(self) -> None:
        from agent_core.commands.optimize_cmd import validate_llm_code
        code = textwrap.dedent("""\
            import os
            try:
                os.mkdir("/test")
            except Exception:
                pass
        """)
        issues = validate_llm_code(code)
        assert len(issues) == 1
        assert issues[0]["pattern"] == "silent_except"

    def test_dead_assignment_blocked(self) -> None:
        from agent_core.commands.optimize_cmd import validate_llm_code
        code = textwrap.dedent("""\
            def foo():
                x = [i for i in range(5)]
                x = []
                return x
        """).strip()
        issues = validate_llm_code(code)
        assert any(i["pattern"] == "dead_assignment" for i in issues)

    def test_clean_code_passes(self) -> None:
        from agent_core.commands.optimize_cmd import validate_llm_code
        code = textwrap.dedent("""\
            import os
            try:
                os.mkdir("/test")
            except Exception:
                logger.warning("Failed: %s", exc_info=True)
        """)
        issues = validate_llm_code(code)
        assert issues == []

    def test_dead_store_blocked(self) -> None:
        """The `combined`/`combined_parts` dead store from an earlier LLM
        rewrite must now be rejected by the validator."""
        from agent_core.commands.optimize_cmd import validate_llm_code
        code = textwrap.dedent("""\
            import os
            async def _deep_analyze(path):
                combined_parts: list[str] = [path]
                combined_parts.append(x)
                combined = chr(10).join(combined_parts)
                msg = combined
                combined_parts.append(y)
                combined = chr(10).join(combined_parts)
                return msg
        """)
        issues = validate_llm_code(code)
        assert any(i["pattern"] == "dead_assignment" for i in issues)

    def test_optimize_rules_encourage_surgical_changes(self) -> None:
        """The optimizer prompt must instruct the LLM not to create dead
        assignments and not to refactor variables it need not touch."""
        assert "dead assignment" in OPTIMIZE_RULES.lower()
        assert "rename" in OPTIMIZE_RULES.lower()


class TestWalrusInComprehension:
    def test_walrus_in_listcomp_flagged(self) -> None:
        from agent_core.patterns import detect_walrus_in_comprehension
        code = textwrap.dedent("""\
            def f(items):
                return [y for x in items if (y := x % 2) == 1]
        """)
        findings = detect_walrus_in_comprehension(code)
        assert len(findings) == 1
        assert findings[0][1] == "walrus_in_comprehension"

    def test_walrus_in_genexpr_flagged(self) -> None:
        from agent_core.patterns import detect_walrus_in_comprehension
        code = textwrap.dedent("""\
            def f(items):
                return sum(y for x in items if (y := x * 2) > 0)
        """)
        findings = detect_walrus_in_comprehension(code)
        assert len(findings) == 1
        assert findings[0][1] == "walrus_in_comprehension"

    def test_walrus_in_dict_comp_flagged(self) -> None:
        from agent_core.patterns import detect_walrus_in_comprehension
        code = textwrap.dedent("""\
            def f(items):
                return {x: (y := x) for x in items}
        """)
        findings = detect_walrus_in_comprehension(code)
        assert len(findings) == 1

    def test_walrus_outside_comprehension_not_flagged(self) -> None:
        from agent_core.patterns import detect_walrus_in_comprehension
        code = textwrap.dedent("""\
            import re
            def f(text):
                if (m := re.search(r"x", text)):
                    return m.group(1)
                return None
        """)
        assert detect_walrus_in_comprehension(code) == []

    def test_clean_comprehension_not_flagged(self) -> None:
        from agent_core.patterns import detect_walrus_in_comprehension
        code = textwrap.dedent("""\
            def f(items):
                return [x * 2 for x in items if x > 0]
        """)
        assert detect_walrus_in_comprehension(code) == []

    def test_walrus_in_comprehension_detected_via_analyze(self) -> None:
        code = textwrap.dedent("""\
            def f(items):
                return [y for x in items if (y := x) > 0]
        """)
        patterns = {finding["pattern"] for finding in static_analyze(code)}
        assert "walrus_in_comprehension" in patterns

    def test_walrus_in_comprehension_blocked_by_validator(self) -> None:
        """An LLM rewrite that introduces a walrus in a comprehension must be
        rejected by the validator so it can never be applied."""
        from agent_core.commands.optimize_cmd import validate_llm_code
        code = textwrap.dedent("""\
            import re

            def parse(source):
                return sorted(set(
                    f"{path}.py"
                    for m in re.finditer(r"^(?:from|import)\\s+(\\S+)", source)
                    if (top := m.group(1).split(".", 1)[0]) in ("agent_core",)
                    for path in [m.group(1).replace(".", "/")]
                ))
        """)
        issues = validate_llm_code(code)
        assert any(i["pattern"] == "walrus_in_comprehension" for i in issues)

    def test_parse_llm_fixes_routes_walrus_to_failures(self) -> None:
        response = textwrap.dedent("""\
            [FILE: a.py]
            ```python
            def f(items):
                return [y for x in items if (y := x) > 0]
            ```
        """)
        fixes, failures = parse_llm_fixes(response, ["a.py"])
        assert fixes == {}
        assert "a.py" in failures
        assert any(f["pattern"] == "walrus_in_comprehension" for f in failures["a.py"])


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

    def test_blank_added_line_shows_line_number(self, capsys: pytest.CaptureFixture[str]) -> None:
        show_file_diff("test.py", "line1\nline2\n", "line1\n\nline2\n")
        out = capsys.readouterr().out
        assert "2" in out
        assert "3" in out
        assert "\033[42m" in out

    def test_blank_removed_line_shows_line_number(self, capsys: pytest.CaptureFixture[str]) -> None:
        show_file_diff("test.py", "line1\n\nline2\n", "line1\nline2\n")
        out = capsys.readouterr().out
        assert "\033[41m" in out

    def test_long_added_line_is_wrapped(self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        import agent_core.commands.base as base
        monkeypatch.setattr(base, "_diff_terminal_width", lambda: 80)
        huge = "x" * 300
        show_file_diff("test.py", "line1\n", f"line1\n{huge}\n")
        out = capsys.readouterr().out
        # Each visible (stripped of ANSI) printed line stays within the width cap.
        import re as _re
        stripped = _re.sub(r"\033\[[0-9]*m", "", out)
        for line in stripped.splitlines():
            assert len(line) <= 80, f"line too wide ({len(line)}): {line!r}"
        # The long token must be split across multiple lines.
        assert stripped.count("x") == 300

    def test_long_line_word_wraps(self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        import agent_core.commands.base as base
        monkeypatch.setattr(base, "_diff_terminal_width", lambda: 80)
        words = " ".join(["word"] * 30)  # ~119 chars
        show_file_diff("test.py", "line1\n", f"line1\n{words}\n")
        out = capsys.readouterr().out
        import re as _re
        stripped = _re.sub(r"\033\[[0-9]*m", "", out)
        for line in stripped.splitlines():
            assert len(line) <= 80, f"line too wide ({len(line)}): {line!r}"
        # No "word" instance should be cut mid-word across a wrap boundary
        # (word-wrap kicks in before hard-slicing).
        assert stripped.count("word") == 30


class TestOptimizeEmptyOutput:
    def test_empty_llm_response_retries_then_gives_up(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A model that burns its budget in reasoning and returns an empty
        message must NOT silently drop the batch: the pipeline retries with
        feedback, then reports that no fixes were generated."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        target = tmp_path / "sloppy.py"
        target.write_text(textwrap.dedent("""\
            import os
            def read(path):
                try:
                    return open(path).read()
                except Exception:
                    pass
        """), encoding="utf-8")

        calls = 0

        async def chat(messages, **kwargs):
            nonlocal calls
            calls += 1
            return ""

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert calls == 3  # initial + 2 retries (max_retries=2)
        assert "No code blocks parsed" in out
        assert "No fixes were generated" in out
        assert "pass" in target.read_text(encoding="utf-8")  # file untouched

    def test_prose_only_response_retries_then_recovers(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A response containing prose but no code blocks triggers the retry;
        a subsequent valid response is accepted and applied."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        target = tmp_path / "sloppy.py"
        target.write_text(textwrap.dedent("""\
            import os
            def read(path):
                try:
                    return open(path).read()
                except Exception:
                    pass
        """), encoding="utf-8")

        responses = [
            "I considered refactoring but here are no code blocks yet.",
            "[FILE: sloppy.py]\n```python\n"
            "def read(path):\n"
            "    try:\n"
            "        with open(path, encoding=\"utf-8\") as f:\n"
            "            return f.read()\n"
            "    except OSError:\n"
            "        return \"\"\n"
            "```\n",
        ]

        async def chat(messages, **kwargs):
            return responses.pop(0)

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert "No code blocks parsed" in out
        assert "Got fixes for 1 file(s)" in out
        assert "Applied: sloppy.py" in out


class TestChangedImports:
    def test_import_entries_normalization(self) -> None:
        from agent_core.commands.optimize_cmd import _import_entries
        code = textwrap.dedent("""\
            import os
            from .base import Command
            from agent_core import workspace_path, normalize_path
            import ast as _ast
        """)
        entries = _import_entries(code)
        assert "import os" in entries
        assert "from .base import Command" in entries
        assert "from agent_core import normalize_path, workspace_path" in entries
        assert "import ast" in entries  # alias target normalized to the module name

    def test_removing_unused_import_is_subset(self) -> None:
        from agent_core.commands.optimize_cmd import _import_entries
        original = "import os\nimport re\n\ndef f():\n    return re.compile('x')\n"
        fixed = "import re\n\ndef f():\n    return re.compile('x')\n"
        added = _import_entries(fixed) - _import_entries(original)
        assert added == set()

    def test_duplicate_stdlib_import_is_blocked(self) -> None:
        """A rewrite that adds a *second* copy of an import the file already
        has (region splices re-inject import os/import re) is a new
        duplicate_import defect and must be rejected even for stdlib modules."""
        from agent_core.commands.optimize_cmd import _blocked_added_imports
        original = "import os\nimport re\n\ndef f():\n    return re.compile('x')\n"
        candidate = "import os\nimport re\nimport os\n\ndef f():\n    return re.compile('x')\n"
        assert "import os" in _blocked_added_imports(candidate, original)

    def test_new_stdlib_import_is_allowed(self) -> None:
        """A genuinely new stdlib import (e.g. import json for a logging fix)
        is safe and must NOT be blocked."""
        from agent_core.commands.optimize_cmd import _blocked_added_imports
        original = "import re\n\ndef f():\n    return re.compile('x')\n"
        candidate = "import re\nimport json\n\ndef f():\n    return json.dumps(re.compile('x').pattern)\n"
        assert _blocked_added_imports(candidate, original) == set()

    def test_added_import_rejected_and_not_applied(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A rewrite that adds a non-stdlib import the original never had is
        rejected (changed_imports) and routed into the retry feedback, never
        applied.  Stdlib additions (e.g. import json) are safe and allowed;
        relative/project imports are the risk this gate blocks."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        target = tmp_path / "sample.py"
        target.write_text(textwrap.dedent("""\
            import os
            import re

            def f():
                return re.compile("x")
        """), encoding="utf-8")

        responses = [
            "[FILE: sample.py]\n```python\n"
            "import re\n"
            "from .helpers import now\n"
            "\n"
            "def f():\n"
            "    return now(re.compile(\"x\").pattern)\n"
            "```\n",
            "[FILE: sample.py]\n```python\n"
            "import re\n"
            "\n"
            "def f():\n"
            "    return re.compile(\"x\")\n"
            "```\n",
        ]

        async def chat(messages, **kwargs):
            return responses.pop(0)

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert "Rejected 1 fix(es) for changed imports" in out
        assert "changed_imports" in out
        assert "Got fixes for 1 file(s)" in out  # retry succeeded
        assert "Applied: sample.py" in out
        final = target.read_text(encoding="utf-8")
        assert "from .helpers" not in final
        assert "import os" not in final  # unused import removed on retry

    def test_added_stdlib_import_accepted(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A rewrite that adds a stdlib import (logging, to implement the
        silent_except fix) passes the changed_imports gate and is applied."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        target = tmp_path / "sample.py"
        target.write_text(textwrap.dedent("""\
            import os
            import re

            def f():
                try:
                    return re.compile("x")
                except Exception:
                    pass
                return None
        """), encoding="utf-8")

        async def chat(messages, **kwargs):
            return "[FILE: sample.py]\n```python\n" \
                "import logging\n" \
                "import re\n" \
                "\n" \
                "def f():\n" \
                "    try:\n" \
                "        return re.compile(\"x\")\n" \
                "    except Exception:\n" \
                "        logging.error(\"compile failed\")\n" \
                "        raise\n" \
                "    return None\n" \
                "```\n"

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert "changed_imports" not in out
        assert "Applied: sample.py" in out
        final = target.read_text(encoding="utf-8")
        assert "import logging" in final
        assert "logging.error" in final

    def test_unchanged_imports_accepted(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A fix that only touches code (imports untouched) passes the check."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        target = tmp_path / "sample.py"
        target.write_text(textwrap.dedent("""\
            import os
            import re

            def f():
                return re.compile("x")
        """), encoding="utf-8")

        async def chat(messages, **kwargs):
            return "[FILE: sample.py]\n```python\n" \
                   "import re\n\n" \
                   "def f():\n" \
                   "    return re.compile(\"x\")\n" \
                   "```\n"

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert "Rejected" not in out
        assert "Applied: sample.py" in out


class TestNoopRewriteGate:
    """The merged-rewrite gate must reject a diff that touches no finding line:
    an LLM dodging silent_except findings with a comment/whitespace-only edit
    is a no-op rewrite, not a fix."""

    def test_whitespace_only_change_to_unrelated_line_is_noop(self) -> None:
        from agent_core.commands.optimize_cmd import _blocked_regressions
        original = textwrap.dedent("""\
            def probe(a):
                try:
                    return a + 1
                except Exception:
                    pass
                return 0


            def other(b):
                return b * 2
        """)
        # The only change: a comment line elsewhere + one extra blank line;
        # the silent_except (original line 3) is untouched.
        candidate = textwrap.dedent("""\
            def probe(a):
                try:
                    return a + 1
                except Exception:
                    pass
                return 0

            # unrelated comment
            def other(b):
                return b * 2
        """)
        issues = _blocked_regressions(original, candidate)
        assert [i["pattern"] for i in issues] == ["noop_rewrite"]

    def test_real_fix_touching_finding_line_is_not_noop(self) -> None:
        from agent_core.commands.optimize_cmd import _blocked_regressions
        original = textwrap.dedent("""\
            def probe(a):
                try:
                    return a + 1
                except Exception:
                    pass
                return 0
        """)
        candidate = textwrap.dedent("""\
            def probe(a):
                try:
                    return a + 1
                except Exception:
                    print("probe failed")
                return 0
        """)
        assert _blocked_regressions(original, candidate) == []


class TestRegionSplitting:
    """Divide & conquer: oversized files are split into line regions, each
    fixed in its own LLM call, then spliced back and validated as a whole."""

    def _big_file(self) -> str:
        """A file big enough (input estimate > REGION_SPLIT_TOKENS) to trigger
        region splitting: many top-level defs, one silent-except finding."""
        parts = ["import os\nimport re\n\n"]
        for i in range(60):
            parts.append(
                f"def func_{i}(value):\n"
                f"    result = value * {i}\n"
                f"    fname = os.path.join('data', re.sub(r'\\d+', '', 'f{i}.txt'))\n"
                "    try:\n"
                f"        return open(fname).read()\n"
                "    except Exception:\n"
                "        pass\n"
                "    return result\n\n"
            )
        return "".join(parts)

    def _big_class_file(self) -> str:
        """A file whose bulk lives inside a class: many methods, one
        silent-except finding per method.  Splitting should happen at the
        indented method boundaries, not only at column-0 statements."""
        parts = ["import os\nimport re\n\n"]
        parts.append("class BigService:\n")
        parts.append("    def __init__(self):\n")
        parts.append("        self.data = []\n\n")
        for i in range(60):
            parts.append(
                f"    def func_{i}(self, value):\n"
                f"        result = value * {i}\n"
                f"        fname = os.path.join('data', re.sub(r'\\d+', '', 'f{i}.txt'))\n"
                "        try:\n"
                f"            return open(fname).read()\n"
                "        except Exception:\n"
                "            pass\n"
                "        return result\n\n"
            )
        return "".join(parts)

    def _monolith_file(self) -> str:
        """A file that is ONE un-splittable statement: a giant assignment
        (dict/tuple literal ~9k tokens) inside a function, whose only finding
        is the dead assignment itself.  No AST seam exists inside the literal,
        so the region stays whole and patch-mode (hunks) is the only way."""
        parts = ["def big_engine():\n", "    x = (\n", "        'start',\n"]
        for i in range(1100):
            parts.append(f"        '# padding {i}',\n")
        parts.append("        'key',\n")
        parts.append("    )\n")
        return "".join(parts)

    def _assert_statement_seam_boundaries(self, code: str, regions: list[tuple[int, int]]) -> None:
        """Every region boundary must be a statement seam (start of a complete
        statement that ends the previous one) or a column-0 def/import line."""
        import ast as ast_mod
        from agent_core.commands.optimize_cmd import _ast_statement_seams
        lines = code.split("\n")
        seams = _ast_statement_seams(code)
        tree = ast_mod.parse(code)
        stmt_starts = {0, len(lines)}
        for node in ast_mod.walk(tree):
            if getattr(node, "lineno", None) is None:
                continue
            first = node.decorator_list[0] if getattr(node, "decorator_list", None) else node
            stmt_starts.add(first.lineno - 1)
        for start, end in regions:
            assert start in (seams | stmt_starts), f"{start} is not a statement start"
            assert end == 0 or end in (seams | stmt_starts) or end == len(lines), \
                f"{end} is not a statement boundary"

    def test_statement_seams_never_split_a_statement(self) -> None:
        """Every seam is the start line of SOME complete statement — a cut can
        never land strictly inside a statement's own lines.  (Seams naturally
        lie inside enclosing statements: that is exactly what lets a giant
        method split at its inner statements.)"""
        import ast as ast_mod
        from agent_core.commands.optimize_cmd import _ast_statement_seams
        for code in (self._big_file(), self._big_class_file(), self._monolith_file()):
            seams = _ast_statement_seams(code)
            tree = ast_mod.parse(code)
            stmt_starts = set()
            for node in ast_mod.walk(tree):
                if getattr(node, "lineno", None) is None:
                    continue
                first = node.decorator_list[0] if getattr(node, "decorator_list", None) else node
                stmt_starts.add(first.lineno - 1)
            assert seams <= stmt_starts, f"seams outside statement starts: {seams - stmt_starts}"

    def test_giant_function_body_splits_at_inner_statement_seams(self) -> None:
        """A 1300-line single method body is no longer one monolithic region:
        statement seams inside it produce multiple regions, each of which
        compiles when re-attached to its enclosing function header."""
        from agent_core.commands.optimize_cmd import split_into_regions
        parts = ["def big_engine():\n"]
        for j in range(32):
            parts.append(f"    x{j} = " + "'" + "y" * 180 + "'\n")
            parts.append(f"    if x{j}:\n        data = x{j}\n    else:\n        data = 0\n")
        parts.append("    return data\n")
        code = "".join(parts)
        lines = code.split("\n")
        regions = split_into_regions(code)
        assert len(regions) >= 2, "giant method must not stay a single region"
        assert regions[0][0] == 0 and regions[-1][1] == len(lines)
        for start, end in regions[1:]:
            assert lines[start][:1].isspace(), "mid-body regions start indented"
            joined = "def big_engine():\n" + "\n".join(lines[start:end])
            compile(joined, "<seam>", "exec")

    def test_split_into_regions_splits_class_bodies(self) -> None:
        """A class-heavy file (like fix_cmd.py) is split at indented method
        boundaries so the whole class doesn't have to be rewritten one-shot."""
        from agent_core.commands.optimize_cmd import split_into_regions
        code = self._big_class_file()
        lines = code.split("\n")
        regions = split_into_regions(code)
        assert len(regions) >= 2
        # Regions are contiguous and cover the whole file.
        assert regions[0][0] == 0
        assert regions[-1][1] == len(lines)
        for (_, end), (next_start, _) in zip(regions, regions[1:]):
            assert end == next_start
        # Some region starts inside the class (indented) — big class bodies
        # must never have to be rewritten in one shot.
        indented_starts = [s for s, _ in regions if lines[s][:1].isspace()]
        assert indented_starts, "expected at least one region to start inside the class"
        self._assert_statement_seam_boundaries(code, regions)

    def test_split_into_regions_never_cuts_mid_function(self) -> None:
        from agent_core.commands.optimize_cmd import split_into_regions
        code = self._big_file()
        lines = code.split("\n")
        regions = split_into_regions(code)
        assert len(regions) >= 2
        # Regions are contiguous and cover the whole file.
        assert regions[0][0] == 0
        assert regions[-1][1] == len(lines)
        for (_, end), (next_start, _) in zip(regions, regions[1:]):
            assert end == next_start
        # Every boundary is a statement seam or a top-level def/import — a
        # region never starts mid-statement.
        self._assert_statement_seam_boundaries(code, regions)

    def test_merge_regions_splices_fixed_code(self) -> None:
        from agent_core.commands.optimize_cmd import _merge_regions
        original = "a\nb\nc\nd\ne\n"
        regions = {0: (0, 2, "A\nB"), 1: (2, 5, "C\nD\nE")}
        assert _merge_regions(original, regions) == "A\nB\nC\nD\nE\n"

    def test_merge_regions_keeps_unfixed_slices(self) -> None:
        from agent_core.commands.optimize_cmd import _merge_regions
        original = "a\nb\nc\nd\ne\n"
        regions = {0: (0, 2, "A\nB")}  # second region never returned a fix
        assert _merge_regions(original, regions) == "A\nB\nc\nd\ne\n"

    def test_large_file_fixed_in_regions(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An oversized file is split into regions; each region is fixed in its
        own LLM call; the merged file is applied."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand, split_into_regions

        target = tmp_path / "big.py"
        target.write_text(self._big_file(), encoding="utf-8")

        code = self._big_file()
        regions = split_into_regions(code)
        assert len(regions) >= 2  # precondition: this file triggers splitting

        # Mock LLM: fix each region by replacing `except Exception: pass` with
        # `except OSError: return ""` (satisfies silent_except).
        async def chat(messages, **kwargs):
            user = messages[-1]["content"]
            code_block = re.search(r"```python\n(.*?)\n```", user, re.DOTALL)
            assert code_block is not None
            fixed = code_block.group(1).replace(
                "except Exception:\n        pass",
                "except OSError:\n        return \"\"",
            )
            return f"[FILE: big.py]\n```python\n{fixed}\n```\n"

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert "split into" in out
        assert "divide & conquer" in out
        assert "Merged" in out
        assert "Applied: big.py" in out
        final = target.read_text(encoding="utf-8")
        assert "except Exception:\n        pass" not in final
        assert final.count("def func_") == 60

    def test_class_file_fixed_in_regions_preserves_indentation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A class-heavy file is split at method boundaries; a region whose
        fix drops the method's leading indentation is rejected and retried so
        the merged file stays valid Python."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand, split_into_regions

        target = tmp_path / "svc.py"
        target.write_text(self._big_class_file(), encoding="utf-8")
        code = self._big_class_file()
        regions = split_into_regions(code)
        assert len(regions) >= 2  # precondition

        bad_indent_done = False
        seen_feedback: list[str] = []
        saw_disable_thinking = False

        async def chat(messages, **kwargs):
            nonlocal bad_indent_done, saw_disable_thinking
            if kwargs.get("disable_thinking") is True:
                saw_disable_thinking = True
            user = messages[-1]["content"]
            code_block = re.search(r"```python\n(.*?)\n```", user, re.DOTALL)
            assert code_block is not None
            raw = code_block.group(1)
            if "region_indent_changed" in user:
                seen_feedback.append(user)
            if not bad_indent_done and raw[:1].isspace():
                # Simulate the model dropping the method's first-line indent.
                bad_indent_done = True
                dedented = raw[4:]
                return f"[FILE: svc.py]\n```python\n{dedented}\n```\n"
            fixed = raw.replace(
                "except Exception:\n            pass",
                "except OSError:\n            return \"\"",
            )
            return f"[FILE: svc.py]\n```python\n{fixed}\n```\n"

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert "split into" in out
        assert seen_feedback, "expected the bad-indent region to be rejected with feedback"
        assert saw_disable_thinking, "optimize rewrite calls must pass disable_thinking=True"
        final = target.read_text(encoding="utf-8")
        compile(final, "<svc>", "exec")  # must remain valid Python
        assert final.count("def func_") == 60
        assert "except Exception:\n            pass" not in final

    def test_region_prompt_asks_for_verbatim_indentation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The region prompt tells the model to keep the original indentation."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        target = tmp_path / "svc.py"
        target.write_text(self._big_class_file(), encoding="utf-8")
        seen_prompt = ""

        async def chat(messages, **kwargs):
            nonlocal seen_prompt
            seen_prompt = messages[-1]["content"]
            user = messages[-1]["content"]
            code_block = re.search(r"```python\n(.*?)\n```", user, re.DOTALL)
            raw = code_block.group(1)
            fixed = raw.replace(
                "except Exception:\n            pass",
                "except OSError:\n            return \"\"",
            )
            return f"[FILE: svc.py]\n```python\n{fixed}\n```\n"

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        assert asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent)) is True
        assert "Preserve the original indentation" in seen_prompt

    def test_region_syntax_failure_retries_region(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A region whose fix has a syntax error is rejected and retried with
        feedback; the corrected region is accepted."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        target = tmp_path / "big.py"
        target.write_text(self._big_file(), encoding="utf-8")

        responses = [
            "[FILE: big.py]\n```python\n    def broken(:\n```\n",  # region 0: bad syntax
        ]
        used_bad = False

        async def chat(messages, **kwargs):
            nonlocal used_bad
            user = messages[-1]["content"]
            code_block = re.search(r"```python\n(.*?)\n```", user, re.DOTALL)
            assert code_block is not None
            code = code_block.group(1)
            if not used_bad:
                used_bad = True
                return responses.pop(0)
            fixed = code.replace(
                "except Exception:\n        pass",
                "except OSError:\n        return \"\"",
            )
            return f"[FILE: big.py]\n```python\n{fixed}\n```\n"

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert "Retry 1/2" in out
        assert "Applied: big.py" in out
        assert "except Exception:\n        pass" not in target.read_text(encoding="utf-8")

    def test_merged_rewrite_repaired_when_region_leaves_finding(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A region that leaves a pre-existing finding unfixed is NOT a
        regression: the original had many silent-excepts, the merged rewrite
        has fewer, so it is applied directly — no repair pass, no skip."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        target = tmp_path / "big.py"
        target.write_text(self._big_file(), encoding="utf-8")
        repair_passes = 0

        async def chat(messages, **kwargs):
            nonlocal repair_passes
            user = messages[-1]["content"]
            code_block = re.search(r"```python\n(.*?)\n```", user, re.DOTALL)
            assert code_block is not None
            code = code_block.group(1)
            if "REGION of a larger file" in user:
                if "func_59" in code:
                    # Leave this region's silent-except unfixed on purpose.
                    return f"[FILE: big.py]\n```python\n{code}\n```\n"
                fixed = code.replace(
                    "except Exception:\n        pass",
                    "except OSError:\n        return \"\"",
                )
                return f"[FILE: big.py]\n```python\n{fixed}\n```\n"
            # Whole-file repair pass only runs if the merged rewrite regresses.
            repair_passes += 1
            return f"[FILE: big.py]\n```python\n{code}\n```\n"

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out

        assert ok is True
        assert repair_passes == 0, (
            "leaving a pre-existing silent-except must NOT trigger a repair "
            "pass (no regression vs the original)"
        )
        assert "Skipping" not in out
        assert "Applied: big.py" in out
        # The residual silent-except is tolerated (original had 60 of them).
        final = target.read_text(encoding="utf-8")
        assert "except Exception:\n        pass" in final
        compile(final, "<big>", "exec")  # merged file must stay valid

    def test_merged_region_adds_stdlib_import_accepted(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A merged rewrite that ADDS a stdlib import (`import logging`, the
        fix its silent_except finding demands) must not be skipped: stdlib
        additions are safe and pass the changed_imports gate.  Only
        relative/project/third-party additions block."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        target = tmp_path / "big.py"
        target.write_text(self._big_file(), encoding="utf-8")
        repair_passes = 0

        async def chat(messages, **kwargs):
            nonlocal repair_passes
            user = messages[-1]["content"]
            code_block = re.search(r"```python\n(.*?)\n```", user, re.DOTALL)
            assert code_block is not None
            code = code_block.group(1)
            if "REGION of a larger file" in user:
                # The first region covers the file top (module imports); add
                # `import logging` there and fix silent-excepts with it. This
                # mirrors the live fix_cmd.py run that added import logging.
                if "import os" in code and code.split("\n")[0].startswith("import "):
                    fixed = code.replace(
                        "import os\n",
                        "import logging\nimport os\n",
                        1,
                    )
                    fixed = fixed.replace(
                        "except Exception:\n        pass",
                        "except Exception:\n            logging.error('read failed')\n            return \"\"",
                    )
                    return f"[FILE: big.py]\n```python\n{fixed}\n```\n"
                fixed = code.replace(
                    "except Exception:\n        pass",
                    "except OSError:\n        return \"\"",
                )
                return f"[FILE: big.py]\n```python\n{fixed}\n```\n"
            repair_passes += 1
            return f"[FILE: big.py]\n```python\n{code}\n```\n"

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out

        assert ok is True
        assert repair_passes == 0, "default stdlib import must NOT trigger a repair pass"
        assert "added import" not in out, "stdlib import addition must not be skipped"
        assert "Skipping" not in out
        assert "Applied: big.py" in out
        final = target.read_text(encoding="utf-8")
        assert "import logging" in final
        assert "logging.error" in final
        compile(final, "<big>", "exec")  # merged file must stay valid

    def test_merged_regression_triggers_repair_and_applies(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A merged rewrite that introduces an extra dead-store line inside a
        replaced block: the surgical pass removes the inserted line without
        any LLM repair call, and the cleaned merge is applied."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        target = tmp_path / "big.py"
        target.write_text(self._big_file(), encoding="utf-8")
        repair_passes = 0

        async def chat(messages, **kwargs):
            nonlocal repair_passes
            user = messages[-1]["content"]
            code_block = re.search(r"```python\n(.*?)\n```", user, re.DOTALL)
            assert code_block is not None
            code = code_block.group(1)
            assert "REGION of a larger file" in user
            # Region rewrite injects a dead store `leak = 1` never read.
            leaked = code.replace(
                "def func_5(value):\n    result = value * 5\n",
                "def func_5(value):\n    leak = 1\n    result = value * 5\n",
            )
            fixed = leaked.replace(
                "except Exception:\n        pass",
                "except OSError:\n        return \"\"",
            )
            return f"[FILE: big.py]\n```python\n{fixed}\n```\n"

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert repair_passes == 0, (
            "the injected line is removable; no whole-file repair needed"
        )
        assert "regressions fixed surgically" in out
        assert "Applied: big.py" in out
        final = target.read_text(encoding="utf-8")
        assert "leak = 1" not in final
        compile(final, "<big>", "exec")

    def test_oversized_region_uses_patch_mode(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A file whose first region is ONE un-splittable giant statement (a
        giant dict literal ~8k tokens, no AST seams exist inside a literal)
        must not be re-emitted whole: patch-mode asks for [PATCH:] hunks and
        applies them with a tiny output budget."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import (
            OptimizeCommand, estimate_tokens, REGION_PATCH_MODE_TOKENS,
            split_into_regions,
        )

        code = self._monolith_file()
        target = tmp_path / "huge.py"
        target.write_text(code, encoding="utf-8")

        lines = code.split("\n")
        regions = split_into_regions(code)
        assert regions == [(0, len(lines))]
        assert estimate_tokens(code) > REGION_PATCH_MODE_TOKENS  # precondition

        system_prompt = ""
        max_tokens_seen = 0

        async def chat(messages, **kwargs):
            nonlocal system_prompt, max_tokens_seen
            system_prompt = messages[0]["content"]
            max_tokens_seen = max(max_tokens_seen, kwargs.get("max_tokens", 0))
            idx = next(i for i, l in enumerate(lines)
                       if l.strip() == "'# padding 0',")
            hunk = (
                f"[PATCH: huge.py]\n"
                f"@@ -{idx},3 +{idx},3 @@\n"
                f" {lines[idx - 1]}\n"
                f"-{lines[idx]}\n"
                f"+        'engine_ok_springs'\n"
            )
            return hunk

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert "PATCH: huge.py" in system_prompt, "patch-mode prompt must demand hunks"
        assert max_tokens_seen <= 4096, "patch-mode output budget must stay small"
        assert "Applied: huge.py" in out
        final = target.read_text(encoding="utf-8")
        assert "'engine_ok_springs'" in final
        compile(final, "<huge>", "exec")

    def test_patch_mode_retries_after_missing_hunks(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A patch-mode unit that returns no [PATCH:] block is retried with
        explicit feedback; the corrected hunk response is applied."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        code = self._monolith_file()
        lines = code.split("\n")
        target = tmp_path / "huge.py"
        target.write_text(code, encoding="utf-8")

        calls = 0

        async def chat(messages, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return "no hunks here, just prose"
            idx = next(i for i, l in enumerate(lines)
                       if l.strip() == "'# padding 0',")
            return (
                f"[PATCH: huge.py]\n"
                f"@@ -{idx},3 +{idx},3 @@\n"
                f" {lines[idx - 1]}\n"
                f"-{lines[idx]}\n"
                f"+        'engine_ok_springs'\n"
            )

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute(["huge.py", "--apply", "--yes"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert calls == 2, "first empty hunk call must be retried"
        final = target.read_text(encoding="utf-8")
        assert "engine_ok_springs" in final
        compile(final, "<huge>", "exec")
class TestSurgicalRevertRegressions:
    """Deterministic excision of regressing lines: run9 showed LLM repair
    loops keep reintroducing walrus-in-comprehension and unused-import
    regressions, so the merge gate must revert them without the LLM."""

    def test_reverts_introduced_walrus_line(self) -> None:
        original = "def pick(nums):\n    vals = [n for n in nums if n > 0]\n    return vals\n"
        candidate = "def pick(nums):\n    vals = [n for n in nums if (p := n) > 0]\n    return vals\n"
        assert "walrus_in_comprehension" in {
            f["pattern"] for f in static_analyze(candidate)
        }
        cleaned, residual = _surgically_revert_regressions(original, candidate)
        assert not residual, residual
        assert "p := n" not in cleaned
        compile(cleaned, "<clean>", "exec")

    def test_removes_added_unused_import(self) -> None:
        original = "import os\n\ndef f():\n    return os.name\n"
        candidate = "import os\nimport collections\n\ndef f():\n    return os.name\n"
        assert "unused_import" in {f["pattern"] for f in static_analyze(candidate)}
        cleaned, residual = _surgically_revert_regressions(original, candidate)
        assert not residual, residual
        assert "collections" not in cleaned
        compile(cleaned, "<clean>", "exec")

    def test_keeps_legitimate_fixes(self) -> None:
        """The fix must not share a replacement block with the regression:
        difflib preserves equal blocks, so the fixed comprehension survives
        while only the walrus line is restored."""
        original = (
            "def gen(nums):\n"
            "    out = []\n"
            "    for n in nums:\n"
            "        out.append(n)\n"
            "    return out\n"
            "\n"
            "def gen2(xs):\n"
            "    return [n for n in xs if n > 0]\n"
        )
        candidate = (
            "def gen(nums):\n"
            "    out = [n for n in nums]\n"
            "    return out\n"
            "\n"
            "def gen2(xs):\n"
            "    return [n for n in xs if (q := n) > 0]\n"
        )
        cleaned, residual = _surgically_revert_regressions(original, candidate)
        assert not residual, residual
        assert "out = [n for n in nums]" in cleaned
        assert "out = []" not in cleaned and "out.append" not in cleaned
        assert "q := n" not in cleaned
        compile(cleaned, "<clean>", "exec")
