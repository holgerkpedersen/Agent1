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
from agent_core.patterns import analyze as static_analyze, detect_list_append_join, detect_none_eq, detect_fstring_without_placeholder, detect_iter_dict_keys, detect_type_comparison, detect_mutable_default_arg, detect_redundant_bool_expr


def _numbered_context(user_content: str) -> list[tuple[int, str]]:
    """Parse the 'N | line' numbered-context block from a per-finding user
    prompt into (absolute_line_number, text) pairs — lets a fake LLM build
    hunks from the exact lines the optimizer showed it."""
    block = re.search(r"```\n(.*?)\n```", user_content, re.DOTALL)
    if not block:
        return []
    out: list[tuple[int, str]] = []
    for line in block.group(1).split("\n"):
        m = re.match(r"^\s*(\d+)\s*\|\s?(.*)$", line)
        if m:
            out.append((int(m.group(1)), m.group(2)))
    return out


def _finding_pattern(user_content: str) -> str:
    """Pattern name from a per-finding prompt's 'Finding to resolve' line."""
    m = re.search(r"Finding to resolve\s*\n\S+:\d+ \[([^\]]+)\]", user_content)
    return m.group(1) if m else ""


def _fix_hunk_for_pattern(basename: str, user_content: str) -> str:
    """The canonical one-hunk fix for whichever finding the prompt targets.

    * silent_except: ``except Exception`` -> ``except OSError``
    * missing_context_manager: ``return open(X).read()`` -> with-block (keeps
      the finding line's indentation)
    """
    numbered = _numbered_context(user_content)
    pat = _finding_pattern(user_content)
    if pat == "silent_except":
        idx = next(i for i, (num, text) in enumerate(numbered)
                   if text.strip() == "except Exception:")
        start = numbered[idx - 1][0] if idx > 0 else numbered[idx][0]
        ctx = numbered[idx - 1][1] if idx > 0 else None
        body = ""
        if ctx is not None:
            body += f" {ctx}\n"
        body += (f"-{numbered[idx][1]}\n"
                 f"+{numbered[idx][1].replace('Exception', 'OSError')}\n")
        pass_next = idx + 1 < len(numbered) and numbered[idx + 1][1].strip() == "pass"
        if idx + 1 < len(numbered) and not pass_next:
            body += f" {numbered[idx + 1][1]}\n"
        out_hunks = f"[PATCH: {basename}]\n@@ -{start},3 +{start},3 @@\n{body}"
        if pass_next:
            pass_line = numbered[idx + 1][1]
            indent = pass_line[: len(pass_line) - len(pass_line.lstrip())]
            pass_num = numbered[idx + 1][0]
            out_hunks += (
                f"@@ -{pass_num},1 +{pass_num},1 @@\n"
                f"-{pass_line}\n"
                f"+{indent}return None\n"
            )
        return out_hunks
    if pat == "missing_context_manager":
        idx = next(i for i, (num, text) in enumerate(numbered)
                   if text.strip().startswith("return open("))
        line = numbered[idx][1]
        indent = line[: len(line) - len(line.lstrip())]
        expr = line.strip()[len("return "):].removesuffix(".read()")
        num = numbered[idx][0]
        return (
            f"[PATCH: {basename}]\n"
            f"@@ -{num},1 +{num},2 @@\n"
            f"-{line}\n"
            f"+{indent}with {expr} as f:\n"
            f"+{indent}    return f.read()\n"
        )
    if pat == "unused_import":
        num, text = numbered[0]
        modname = text.replace("import ", "", 1).strip().split()[0].split(".")[0]
        if " as " in text:
            modname = text.strip().split(" as ")[1].strip()
        indent = text[: len(text) - len(text.lstrip())]
        return (
            f"[PATCH: {basename}]\n"
            f"@@ -{num},1 +{num},1 @@\n"
            f"-{text}\n"
            f"+{indent}{modname}  # keep referenced\n"
        )
    if pat == "string_concat_in_loop":
        idx = next(i for i, (num, text) in enumerate(numbered)
                   if "+=" in text and isinstance(num, int))
        num = numbered[idx][0]
        old_line = numbered[idx][1]
        new_line = old_line.replace("+=", "=", 1).rstrip() + "  # FIXME"
        return (
            f"[PATCH: {basename}]\n"
            f"@@ -{num},1 +{num},1 @@\n"
            f"-{old_line}\n"
            f"+{new_line}\n"
        )
    if pat == "dead_assignment":
        num, text = next((n, t) for n, t in numbered if "=" in t)
        return (
            f"[PATCH: {basename}]\n"
            f"@@ -{num},1 +{num},1 @@\n"
            f"-{text}\n"
            f"+{text[:len(text)-len(text.lstrip())]}pass  # removed\n"
        )
    raise AssertionError(f"fake LLM does not know pattern {pat!r}")


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

    def test_stateful_loop_skipped(self) -> None:
        """Loops with loop-carried state (name assigned in body, referenced
        by the append/extend expression) must NOT be flagged."""
        code = textwrap.dedent("""\
            def merge(original, regions):
                lines = original.split("\\n")
                out = []
                prev = 0
                for _, (start, end, code) in sorted(regions.items()):
                    out.extend(lines[prev:start])
                    out.extend(code.split("\\n"))
                    prev = end
                out.extend(lines[prev:])
                return "\\n".join(out)
        """)
        findings = detect_list_append_join(code)
        assert findings == []


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

    def test_distinct_read_through_nested_loops_not_flagged(self) -> None:
        """``f.read()`` inside ``while`` inside ``with open(fp)`` inside
        ``os.walk``: the path is derived from outer loop targets, so distinct
        files are read per iteration — no hoistable work."""
        from agent_core.patterns import detect_file_read_in_loop
        code = textwrap.dedent("""\
            def search(path, query):
                results = []
                for root, dirs, files in os.walk(path):
                    for file in files:
                        filepath = os.path.join(root, file)
                        try:
                            with open(filepath, 'r') as f:
                                while True:
                                    chunk = f.read(8192)
                                    if not chunk:
                                        break
                                    if query in chunk:
                                        results.append(filepath)
                                        break
                        except Exception:
                            pass
                return results
        """)
        assert detect_file_read_in_loop(code) == []

    def test_invariant_read_in_nested_loop_still_flagged(self) -> None:
        """``open(CONFIG_PATH).read()`` inside nested ``for`` loops: the path is
        a constant — same file re-read every iteration, must be flagged."""
        from agent_core.patterns import detect_file_read_in_loop
        code = textwrap.dedent("""\
            CONFIG_PATH = "/etc/config.yaml"
            def f():
                for root, dirs, files in os.walk('.'):
                    for file in files:
                        data = open(CONFIG_PATH).read()
                        return data
        """)
        findings = detect_file_read_in_loop(code)
        assert len(findings) == 1
        assert findings[0][1] == "file_read_in_loop"


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
        message must NOT silently drop the finding: the per-finding loop
        retries with feedback, then reports the finding as unresolved."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        target = tmp_path / "sloppy.py"
        target.write_text(textwrap.dedent("""\
            def read(path):
                s = ""
                for i in range(10):
                    s += "x"
                return s
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
        assert calls == 3  # initial + 2 retries (FINDING_MAX_ATTEMPTS=3)
        assert "Unresolved" in out
        assert "No fixes were generated" in out
        assert "for i in range(10):" in target.read_text(encoding="utf-8")  # file untouched

    def test_prose_only_response_retries_then_recovers(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A response containing no [PATCH:] block triggers the retry; a
        subsequent hunk response is accepted and applied."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        file_text = textwrap.dedent("""\
            def read(path):
                s = ""
                for i in range(10):
                    s += "x"
                return s
        """)
        target = tmp_path / "sloppy.py"
        target.write_text(file_text, encoding="utf-8")

        calls = 0

        async def chat(messages, **kwargs):
            nonlocal calls
            calls += 1
            user = messages[-1]["content"]
            if calls == 1:
                return "I considered refactoring but here are no hunks yet."
            return _fix_hunk_for_pattern("sloppy.py", user)

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert calls == 2
        assert "no [PATCH: sloppy.py] block" in out
        assert "Fixed line" in out
        assert "Applied: sloppy.py" in out
        final = target.read_text(encoding="utf-8")
        assert 's += "' not in final
        assert 's += "x"' not in final

    def test_list_append_join_rejects_extend_evasion(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """For list_append_join, .append(x) → .extend([x]) is a lateral move.
        The detector now flags .extend( in loops too, so the acceptance gate
        rejects it (after_cnt stays 1).  A proper comprehension on retry is
        accepted."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        file_text = textwrap.dedent("""\
            def build():
                lines = []
                for item in items:
                    lines.append(f"{item}")
                return "".join(lines)
        """)
        target = tmp_path / "joiner.py"
        target.write_text(file_text, encoding="utf-8")

        calls = 0

        async def chat(messages, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                # lateral move — rejected (pattern count doesn't shrink)
                return (
                    "[PATCH: joiner.py]\n"
                    "@@ -3,1 +3,1 @@\n"
                    "-    lines.append(f\"{item}\")\n"
                    "+    lines.extend([f\"{item}\"])\n"
                )
            # proper comprehension fix — accepted
            return (
                "[PATCH: joiner.py]\n"
                "@@ -2,3 +2,1 @@\n"
                "-    lines = []\n"
                "-    for item in items:\n"
                "-        lines.append(f\"{item}\")\n"
                "+    lines = [f\"{item}\" for item in items]\n"
            )

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert calls == 0
        assert "Fixed line" in out
        assert "Applied: joiner.py" in out
        final = target.read_text(encoding="utf-8")
        assert "for item in items:" not in final
        assert ".append(" not in final
        assert "for item in items]" in final


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
        assert "from agent_core import normalize_path" in entries
        assert "from agent_core import workspace_path" in entries
        assert "import ast" in entries  # alias target normalized to the module name

    def test_removing_unused_import_is_subset(self) -> None:
        from agent_core.commands.optimize_cmd import _import_entries
        original = "import os\nimport re\n\ndef f():\n    return re.compile('x')\n"
        fixed = "import re\n\ndef f():\n    return re.compile('x')\n"
        added = _import_entries(fixed) - _import_entries(original)
        assert added == set()

    def test_removing_name_from_relative_import_not_blocked(self) -> None:
        """Removing an unused name from a multi-name ``from .`` import reduces
        the import set and must NOT be blocked by the import gate."""
        from agent_core.commands.optimize_cmd import _blocked_added_imports
        original = textwrap.dedent("""\
            from .types import TaskType, ProfileType

            class P:
                x: ProfileType
        """)
        candidate = textwrap.dedent("""\
            from .types import ProfileType

            class P:
                x: ProfileType
        """)
        assert _blocked_added_imports(candidate, original) == set()

    def test_adding_name_to_relative_import_is_blocked(self) -> None:
        """Adding a new name to a ``from .`` import must still be blocked."""
        from agent_core.commands.optimize_cmd import _blocked_added_imports
        original = textwrap.dedent("""\
            from .types import ProfileType

            class P:
                x: ProfileType
        """)
        candidate = textwrap.dedent("""\
            from .types import TaskType, ProfileType

            class P:
                x: ProfileType
        """)
        blocked = _blocked_added_imports(candidate, original)
        assert "from .types import TaskType" in blocked

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
        """A hunk that adds a non-stdlib import the original never had is
        rejected (changed_imports) and routed into the retry feedback, never
        applied.  Stdlib additions (e.g. import re) are safe and allowed."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        file_text = textwrap.dedent("""\
            import os
            import re

            def f():
                return re.compile("x")
        """)
        target = tmp_path / "sample.py"
        target.write_text(file_text, encoding="utf-8")

        calls = 0

        async def chat(messages, **kwargs):
            nonlocal calls
            calls += 1
            user = messages[-1]["content"]
            if calls == 1:
                return (
                    "[PATCH: sample.py]\n"
                    "@@ -1,3 +1,4 @@\n"
                    " import os\n"
                    " import re\n"
                    "+from .helpers import now\n"
                    "\n"
                )
            return _fix_hunk_for_pattern("sample.py", user)

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert calls == 0  # mechanical fix handled unused_import
        assert "Applied: sample.py" in out
        final = target.read_text(encoding="utf-8")
        assert "from .helpers" not in final
        assert "import os" not in final

    def test_added_stdlib_import_accepted(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A hunk that adds a stdlib import (logging, to implement the
        silent_except fix) passes the changed_imports gate and is applied."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        file_text = textwrap.dedent("""\
            import os
            import re

            def f():
                try:
                    return re.compile("x")
                except Exception:
                    pass
                return None
        """)
        target = tmp_path / "sample.py"
        target.write_text(file_text, encoding="utf-8")

        async def chat(messages, **kwargs):
            numbered = _numbered_context(messages[-1]["content"])
            idx = next(i for i, (num, text) in enumerate(numbered)
                       if text.strip() == "except Exception:")
            return (
                "[PATCH: sample.py]\n"
                "@@ -5,1 +5,1 @@\n"
                f" {numbered[idx - 1][1]}\n"
                f"-{numbered[idx][1]}\n"
                f"+{numbered[idx][1].replace('Exception', 'OSError')}\n"
                "@@ -7,1 +7,1 @@\n"
                f"-{numbered[idx + 1][1]}\n"
                f"+{numbered[idx + 1][1].replace('pass', 'return None')}\n"
                "@@ -8,1 +8,0 @@\n"
                f"-{numbered[idx + 2][1]}\n"
            )

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert "Applied: sample.py" in out
        final = target.read_text(encoding="utf-8")
        assert "import os" not in final
        assert "Warning: silenced exception" in final
        assert "pass" not in final

    def test_unchanged_imports_accepted(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A fix that only touches code (imports untouched) passes the check."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        file_text = textwrap.dedent("""\
            import os
            import re

            def f():
                return re.compile("x")
        """)
        target = tmp_path / "sample.py"
        target.write_text(file_text, encoding="utf-8")

        async def chat(messages, **kwargs):
            return _fix_hunk_for_pattern("sample.py", messages[-1]["content"])

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


class TestPerFindingPatches:
    """Targeted patches: one tiny [PATCH:] request per finding, context
    limited to the enclosing block with ABSOLUTE numbered lines, applied
    atomically per finding.  No whole-file/region re-emission anywhere."""

    def _big_file(self) -> str:
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
        parts = ["def big_engine():\n", "    x = (\n", "        'start',\n"]
        for i in range(1100):
            parts.append(f"        '# padding {i}',\n")
        parts.append("        'key',\n")
        parts.append("    )\n\n")
        parts.append("def read_all(path):\n")
        parts.append("    s = ''\n")
        parts.append("    for i in range(10):\n")
        parts.append("        s += \"x\"\n")
        parts.append("    return s\n")
        return "".join(parts)

    def _except_fix_hunk(self, basename: str, numbered: list[tuple[int, str]]) -> str:
        """Build a hunk that converts the first `except Exception:` line in
        the numbered context to `except OSError:` — the canonical finding fix."""
        idx = next(i for i, (num, text) in enumerate(numbered)
                   if text.strip() == "except Exception:")
        num = numbered[idx - 1][0] if idx > 0 else numbered[idx][0]
        ctx = numbered[idx - 1][1] if idx > 0 else None
        body = ""
        if ctx is not None:
            body += f" {ctx}\n"
        body += f"-{numbered[idx][1]}\n+{numbered[idx][1].replace('Exception', 'OSError')}\n"
        pass_next = idx + 1 < len(numbered) and numbered[idx + 1][1].strip() == "pass"
        if idx + 1 < len(numbered) and not pass_next:
            body += f" {numbered[idx + 1][1]}\n"
        out_hunks = f"[PATCH: {basename}]\n@@ -{num},3 +{num},3 @@\n{body}"
        if pass_next:
            pass_line = numbered[idx + 1][1]
            indent = pass_line[: len(pass_line) - len(pass_line.lstrip())]
            pass_num = numbered[idx + 1][0]
            out_hunks += (
                f"@@ -{pass_num},1 +{pass_num},1 @@\n"
                f"-{pass_line}\n"
                f"+{indent}return None\n"
            )
        return out_hunks

    def test_large_file_fixed_by_targeted_patches(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A file with 60 findings gets one tiny hunk request per finding;
        every hunk is applied against the live copy and the result compiles."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        target = tmp_path / "big.py"
        target.write_text(self._big_file(), encoding="utf-8")

        file_prompts: list[str] = []
        calls = 0

        async def chat(messages, **kwargs):
            nonlocal calls
            calls += 1
            user = messages[-1]["content"]
            file_prompts.append(user)
            return _fix_hunk_for_pattern("big.py", user)

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert calls == 60, "one LLM call per non-mechanical finding"
        assert "split into" not in out and "Merged" not in out
        assert "Applied: big.py" in out
        final = target.read_text(encoding="utf-8")
        assert "except Exception:\n        pass" not in final
        assert final.count("def func_") == 60
        compile(final, "<big>", "exec")

    def test_class_file_patches_preserve_indentation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Method-level findings inside a class are patched while the class
        indentation stays intact (the context keeps original indentation and
        hunks carry the lines verbatim)."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        target = tmp_path / "svc.py"
        target.write_text(self._big_class_file(), encoding="utf-8")

        async def chat(messages, **kwargs):
            user = messages[-1]["content"]
            numbered = _numbered_context(user)
            assert all(text.startswith(" ") for _, text in numbered
                       if text.strip()), "context must preserve indentation"
            return self._except_fix_hunk("svc.py", numbered)

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert "Applied: svc.py" in out
        final = target.read_text(encoding="utf-8")
        compile(final, "<svc>", "exec")
        assert final.count("def func_") == 60
        assert "except Exception:\n            pass" not in final

    def test_prompt_uses_absolute_numbers_and_hunks_only(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The per-finding prompt must show absolute numbered lines, demand
        [PATCH:] output, and never ask for a re-emission ([FILE:])."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        target = tmp_path / "svc.py"
        target.write_text(self._big_class_file(), encoding="utf-8")
        seen_system = ""
        seen_user = ""

        async def chat(messages, **kwargs):
            nonlocal seen_system, seen_user
            seen_system = messages[0]["content"]
            seen_user = messages[-1]["content"]
            numbered = _numbered_context(seen_user)
            return self._except_fix_hunk("svc.py", numbered)

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        capsys.readouterr()
        assert ok is True
        assert "absolute" in seen_user  # numbers ARE the absolute line numbers
        assert "[PATCH: svc.py]" in seen_system
        assert "[FILE: svc.py]" not in seen_user and "[FILE: svc.py]" not in seen_system
        assert " | " in seen_user  # numbered context


    def test_syntax_broken_patch_retries_until_valid(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A patch that leaves the file unparseable is rejected with
        feedback and retried; the valid retry is applied."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        target = tmp_path / "big.py"
        target.write_text(self._big_file(), encoding="utf-8")
        calls = 0

        async def chat(messages, **kwargs):
            nonlocal calls
            calls += 1
            user = messages[-1]["content"]
            numbered = _numbered_context(user)
            if calls == 1:
                num, text = numbered[0]
                indent = text[: len(text) - len(text.lstrip())]
                return (
                    "[PATCH: big.py]\n"
                    f"@@ -{num},1 +{num},1 @@\n"
                    f"-{text}\n"
                    f"+{indent}@@ {text.strip()}\n"
                )
            return self._except_fix_hunk("big.py", numbered)

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert calls >= 2
        assert "Patch breaks syntax" in out
        assert "Applied: big.py" in out

    def test_patch_that_leaves_finding_is_rejected_and_retried(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A hunk that edits a nearby line but leaves the finding itself
        standing fails the acceptance check (the pattern must shrink) and is
        retried with feedback pinned to the finding."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        target = tmp_path / "big.py"
        target.write_text(self._big_file(), encoding="utf-8")
        calls = 0

        async def chat(messages, **kwargs):
            nonlocal calls
            calls += 1
            user = messages[-1]["content"]
            numbered = _numbered_context(user)
            if calls == 1:
                idx = next(i for i, (num, text) in enumerate(numbered)
                           if text.strip().startswith("return open"))
                num = numbered[idx][0]
                return (
                    "[PATCH: big.py]\n"
                    f"@@ -{num},1 +{num},1 @@\n"
                    f"-{numbered[idx][1]}\n"
                    f"+{numbered[idx][1].replace('open(fname)', 'open(fname, mode=\"r\")')}\n"
                )
            return self._except_fix_hunk("big.py", numbered)

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert "still contains" in out
        assert "Applied: big.py" in out
        final = target.read_text(encoding="utf-8")
        assert "mode=\"r\"" not in final  # the unrelated edit was never applied

    def test_whitespace_only_patch_rejected_as_cosmetic(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A patch that only drops trailing spaces on blank lines (the
        laguna base.py behavior) is a no-op: rejected, retried, fixed for
        real."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        file_text = textwrap.dedent("""\
            import os
            import re

            def read(path):
                s = ""
                for i in range(10):
                    s += "x"
                return s
        """)
        target = tmp_path / "baseish.py"
        target.write_text(file_text, encoding="utf-8")
        calls = 0

        async def chat(messages, **kwargs):
            nonlocal calls
            calls += 1
            user = messages[-1]["content"]
            numbered = _numbered_context(user)
            if calls == 1:
                idx = next(i for i, (num, text) in enumerate(numbered)
                           if text.strip() == "for i in range(10):")
                num = numbered[idx][0]
                return (
                    "[PATCH: baseish.py]\n"
                    f"@@ -{num},1 +{num},1 @@\n"
                    f"-{numbered[idx][1]}\n"
                    f"+{numbered[idx][1]}  \n"
                )
            return _fix_hunk_for_pattern("baseish.py", user)

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert "whitespace" in out
        assert "Applied: baseish.py" in out
        final = target.read_text(encoding="utf-8")
        assert 's += "' not in final

    def test_giant_statement_file_patched_with_tiny_budget(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A file dominated by one giant statement (millions of tokens) is
        still fixed by a small hunk with a capped output budget; the giant
        literal is never re-emitted whole."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        code = self._monolith_file()
        target = tmp_path / "huge.py"
        target.write_text(code, encoding="utf-8")
        system_prompt = ""
        max_tokens_seen = 0

        async def chat(messages, **kwargs):
            nonlocal system_prompt, max_tokens_seen
            system_prompt = messages[0]["content"]
            max_tokens_seen = max(max_tokens_seen, kwargs.get("max_tokens", 0))
            numbered = _numbered_context(messages[-1]["content"])
            return _fix_hunk_for_pattern("huge.py", messages[-1]["content"])

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert max_tokens_seen <= 8192, "hunk output budget must stay small"
        assert "[FILE: huge.py]" not in system_prompt
        assert "Applied: huge.py" in out
        final = target.read_text(encoding="utf-8")
        assert 's += "' not in final
        compile(final, "<huge>", "exec")

    def test_missing_hunk_response_retried_with_feedback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A response with no [PATCH:] block at all is retried with explicit
        feedback; the corrected hunk response is applied."""
        import asyncio
        from types import SimpleNamespace
        from agent_core.commands.optimize_cmd import OptimizeCommand

        target = tmp_path / "huge.py"
        target.write_text(self._monolith_file(), encoding="utf-8")
        calls = 0

        async def chat(messages, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return "no hunks here, just prose"
            return _fix_hunk_for_pattern("huge.py", messages[-1]["content"])

        agent = SimpleNamespace(
            workspace=str(tmp_path),
            llm=SimpleNamespace(chat=chat),
        )
        ok = asyncio.run(OptimizeCommand().execute([target.name, "--apply", "--yes"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert calls == 2
        assert "no [PATCH: huge.py] block" in out
        assert "Applied: huge.py" in out


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


# ---------------------------------------------------------------------------
# New detectors (added 2026-08-09)
# ---------------------------------------------------------------------------

class TestNoneEqDetector:
    """``x == None`` / ``x != None`` → ``is None`` / ``is not None``."""

    def test_none_eq_flagged(self) -> None:
        code = "if x == None:\n    return"
        findings = detect_none_eq(code)
        assert len(findings) == 1
        assert findings[0][1] == "none_eq"

    def test_none_ne_flagged(self) -> None:
        code = "if x != None:\n    return"
        findings = detect_none_eq(code)
        assert len(findings) == 1

    def test_is_none_not_flagged(self) -> None:
        code = "if x is None:\n    pass"
        findings = detect_none_eq(code)
        assert findings == []

    def test_comment_not_flagged(self) -> None:
        code = "# check x == None"
        findings = detect_none_eq(code)
        assert findings == []

    def test_full_analyze(self) -> None:
        results = static_analyze("x = None\nif x == None:\n    pass")
        assert "none_eq" in {r["pattern"] for r in results}

    def test_fixed_clears(self) -> None:
        results = static_analyze("if x is None:\n    pass")
        assert "none_eq" not in {r["pattern"] for r in results}


class TestFstringWithoutPlaceholderDetector:
    """``f\"static\"`` → ``\"static\"``."""

    def test_fstring_no_interp_flagged(self) -> None:
        code = 'msg = f"hello world"'
        findings = detect_fstring_without_placeholder(code)
        assert len(findings) == 1
        assert findings[0][1] == "fstring_without_placeholder"

    def test_fstring_with_interp_not_flagged(self) -> None:
        code = 'msg = f"hello {name}"'
        findings = detect_fstring_without_placeholder(code)
        assert findings == []

    def test_fstring_escaped_braces_flagged(self) -> None:
        code = 'msg = f"hello {{name}}"'  # {{ → literal {, no interpolation
        findings = detect_fstring_without_placeholder(code)
        assert len(findings) == 1

    def test_triple_quoted_flagged(self) -> None:
        code = 'msg = f"""no interp here\nstill none"""'
        findings = detect_fstring_without_placeholder(code)
        assert len(findings) == 1

    def test_full_analyze(self) -> None:
        results = static_analyze('x = f"static"')
        assert "fstring_without_placeholder" in {r["pattern"] for r in results}

    def test_fixed_clears(self) -> None:
        results = static_analyze('x = "static"')
        assert "fstring_without_placeholder" not in {r["pattern"] for r in results}


class TestIterDictKeysDetector:
    """``for k in d.keys():`` → ``for k in d:``."""

    def test_for_in_keys_flagged(self) -> None:
        code = "for k in d.keys():\n    print(k)"
        findings = detect_iter_dict_keys(code)
        assert len(findings) == 1
        assert findings[0][1] == "iter_dict_keys"

    def test_containment_flagged(self) -> None:
        code = "if name in config.keys():\n    return"
        findings = detect_iter_dict_keys(code)
        assert len(findings) == 1

    def test_plain_keys_call_not_flagged(self) -> None:
        code = "x = d.keys()"
        findings = detect_iter_dict_keys(code)
        assert findings == []

    def test_fixed_clears(self) -> None:
        code = "for k in d:\n    print(k)"
        findings = detect_iter_dict_keys(code)
        assert findings == []

    def test_full_analyze(self) -> None:
        results = static_analyze("for k in d.keys():\n    pass")
        assert "iter_dict_keys" in {r["pattern"] for r in results}


class TestTypeComparisonDetector:
    """``type(x) == Foo`` → ``isinstance(x, Foo)``."""

    def test_type_eq_flagged(self) -> None:
        code = "if type(x) == str:\n    return True"
        findings = detect_type_comparison(code)
        assert len(findings) == 1
        assert findings[0][1] == "type_comparison"

    def test_type_ne_flagged(self) -> None:
        code = "if type(x) != int:\n    return"
        findings = detect_type_comparison(code)
        assert len(findings) == 1

    def test_type_in_tuple_flagged(self) -> None:
        code = "if type(x) in (int, float):\n    return"
        findings = detect_type_comparison(code)
        assert len(findings) == 1

    def test_isinstance_not_flagged(self) -> None:
        code = "if isinstance(x, str):\n    return True"
        findings = detect_type_comparison(code)
        assert findings == []

    def test_full_analyze(self) -> None:
        results = static_analyze("if type(x) == str:\n    pass")
        assert "type_comparison" in {r["pattern"] for r in results}

    def test_fixed_clears(self) -> None:
        results = static_analyze("if isinstance(x, str):\n    pass")
        assert "type_comparison" not in {r["pattern"] for r in results}


class TestMutableDefaultArgDetector:
    """``def f(x=[])`` / ``def f(y={})`` → ``None`` + guard."""

    def test_list_default_flagged(self) -> None:
        code = "def f(x=[]):\n    return x"
        findings = detect_mutable_default_arg(code)
        assert len(findings) == 1
        assert findings[0][1] == "mutable_default_arg"

    def test_dict_default_flagged(self) -> None:
        code = "def f(x={}):\n    return x"
        findings = detect_mutable_default_arg(code)
        assert len(findings) == 1

    def test_none_default_not_flagged(self) -> None:
        code = "def f(x=None):\n    if x is None:\n        x = []\n    return x"
        findings = detect_mutable_default_arg(code)
        assert findings == []

    def test_int_default_not_flagged(self) -> None:
        code = "def f(x=0):\n    return x"
        findings = detect_mutable_default_arg(code)
        assert findings == []

    def test_full_analyze(self) -> None:
        results = static_analyze("def f(params=[]):\n    return params")
        assert "mutable_default_arg" in {r["pattern"] for r in results}


class TestRedundantBoolExprDetector:
    """``return True if cond else False`` → ``return bool(cond)``."""

    def test_true_if_else_false_flagged(self) -> None:
        code = "def f(x):\n    return True if x else False"
        findings = detect_redundant_bool_expr(code)
        assert len(findings) == 1
        assert findings[0][1] == "redundant_bool_expr"

    def test_false_if_else_true_flagged(self) -> None:
        code = "def f(x):\n    return False if x else True"
        findings = detect_redundant_bool_expr(code)
        assert len(findings) == 1

    def test_normal_return_not_flagged(self) -> None:
        code = "def f(x):\n    return bool(x)"
        findings = detect_redundant_bool_expr(code)
        assert findings == []

    def test_full_analyze(self) -> None:
        results = static_analyze("def f(x):\n    return True if x else False")
        assert "redundant_bool_expr" in {r["pattern"] for r in results}


class TestRegexInLoopDetector:
    """Narrowed regex_in_loop: only literal string/r-string patterns are flagged."""

    def test_literal_flagged(self) -> None:
        from agent_core.patterns import detect_regex_in_loop
        code = textwrap.dedent("""\
            for f in files:
                m = re.search(r"^class", f)
                re.compile("^x")
        """)
        findings = detect_regex_in_loop(code)
        assert len(findings) == 2

    def test_variable_not_flagged(self) -> None:
        from agent_core.patterns import detect_regex_in_loop
        code = textwrap.dedent("""\
            for pattern in patterns:
                re.compile(pattern)
        """)
        findings = detect_regex_in_loop(code)
        assert findings == []

    def test_fstring_not_flagged(self) -> None:
        from agent_core.patterns import detect_regex_in_loop
        code = textwrap.dedent("""\
            for cn in class_names:
                m = re.match(rf"^class\\s+{cn}", line)
        """)
        findings = detect_regex_in_loop(code)
        assert findings == []

    def test_concat_not_flagged(self) -> None:
        from agent_core.patterns import detect_regex_in_loop
        code = textwrap.dedent("""\
            for prefix in prefixes:
                re.compile("pre_" + prefix)
        """)
        findings = detect_regex_in_loop(code)
        assert findings == []

    def test_full_analyze(self) -> None:
        code = textwrap.dedent("""\
            for f in files:
                re.compile("static_pattern")
                re.compile(dynamic_var)
                re.match(rf"pre_{x}", line)
        """)
        results = [r for r in static_analyze(code) if r["pattern"] == "regex_in_loop"]
        assert len(results) == 1

    def test_fixed_clears(self) -> None:
        from agent_core.patterns import detect_regex_in_loop
        code = textwrap.dedent("""\
            def f():
                _RE_1 = re.compile("static_pattern")
                for f in files:
                    _RE_1.match(f)
        """)
        findings = detect_regex_in_loop(code)
        assert findings == []


class TestMechanicalRegexHoist:
    """Mechanical _fix_regex_in_loop produces correct hunks."""

    def test_hoist_produces_hunk(self) -> None:
        from agent_core.commands.optimize_cmd import _fix_regex_in_loop
        code = "def scan():\n    for f in files:\n        m = re.search(r'^class', f)"
        wl = code.split("\n")
        hunk = _fix_regex_in_loop(wl, 2, 3, "test.py", {})
        assert hunk is not None
        assert "_RE_" in hunk
        assert "re.compile" in hunk

    def test_hoist_applies_and_compiles(self) -> None:
        from agent_core.commands.optimize_cmd import _fix_regex_in_loop
        from agent_core.patch_utils import apply_patch
        code = textwrap.dedent("""\
            import re

            def scan(files):
                for f in files:
                    m = re.search(r'^class', f)
                    print(m)
        """)
        wl = code.split("\n")
        hunk = _fix_regex_in_loop(wl, 4, 5, "test.py", {})
        assert hunk is not None
        ok, patched = apply_patch(hunk, wl)
        assert ok
        compile(patched, "<test>", "exec")
        assert "re.compile" in patched
        assert "re.search(" not in patched

    def test_variable_arg_returns_none(self) -> None:
        from agent_core.commands.optimize_cmd import _fix_regex_in_loop
        code = "for p in pats:\n    re.compile(p)"
        wl = code.split("\n")
        hunk = _fix_regex_in_loop(wl, 1, 2, "test.py", {})
        assert hunk is None

    def test_concat_arg_returns_none(self) -> None:
        from agent_core.commands.optimize_cmd import _fix_regex_in_loop
        code = "for p in pats:\n    re.compile('pre_' + p)"
        wl = code.split("\n")
        hunk = _fix_regex_in_loop(wl, 1, 2, "test.py", {})
        assert hunk is None


    def test_nested_loop_hoists_above_outermost(self) -> None:
        """When a re call is inside a nested loop, hoist above the OUTERMOST enclosing loop."""
        from agent_core.commands.optimize_cmd import _fix_regex_in_loop
        from agent_core.patch_utils import apply_patch
        code = textwrap.dedent("""\
            import re

            def run():
                for attempt in range(3):
                    for fname in files:
                        class_names = re.findall(r'^class\\s+(\\w+)', source, re.MULTILINE)
                        print(class_names)
        """)
        wl = code.split("\n")
        # The findall is at 0-based index 5 (1-based line 6)
        hunk = _fix_regex_in_loop(wl, 5, 6, "test.py", {})
        assert hunk is not None
        ok, patched = apply_patch(hunk, wl)
        assert ok
        compile(patched, "<test>", "exec")
        # The inserted _RE_N = re.compile(...) must NOT be inside any loop body
        # Verify count drops (no new finding at the insert position)
        from agent_core.commands.optimize_cmd import _count_pattern
        before = _count_pattern("\n".join(wl), "regex_in_loop")
        after = _count_pattern(patched, "regex_in_loop")
        assert after < before, f"count gate must pass: {before} -> {after}"


class TestUndefinedHoistedNamesGate:
    """_undefined_hoisted_names catches broken hoist patches."""

    def test_detects_undefined(self) -> None:
        from agent_core.commands.optimize_cmd import _undefined_hoisted_names
        code = "_INIT_PATTERN.match(x)\n_PREFIX_RE.sub('', y)"
        result = _undefined_hoisted_names(code)
        assert "_INIT_PATTERN" in result
        assert "_PREFIX_RE" in result

    def test_passes_defined(self) -> None:
        from agent_core.commands.optimize_cmd import _undefined_hoisted_names
        code = "_RE_1 = re.compile(r'x')\n_RE_1.search(line)"
        result = _undefined_hoisted_names(code)
        assert result == []

    def test_passes_import_alias(self) -> None:
        from agent_core.commands.optimize_cmd import _undefined_hoisted_names
        code = "import _io\n_io.open('f')"
        result = _undefined_hoisted_names(code)
        assert result == []

    def test_passes_function_def(self) -> None:
        from agent_core.commands.optimize_cmd import _undefined_hoisted_names
        code = textwrap.dedent("""\
            def _helper():
                return 1

            _helper()
        """)
        result = _undefined_hoisted_names(code)
        assert result == []

    def test_passes_module_dunders(self) -> None:
        from agent_core.commands.optimize_cmd import _undefined_hoisted_names
        code = 'import os\nos.path.abspath(__file__)\n__doc__\n__package__'
        result = _undefined_hoisted_names(code)
        assert result == []


class TestRegressesDefinedNames:
    """_regresses_defined_names catches LLM patches that silently drop a
    previously-defined name (e.g. removing ``prev = 0`` while keeping
    ``out.extend(lines[prev:])``)."""

    def test_catches_removed_def(self) -> None:
        from agent_core.commands.optimize_cmd import _regresses_defined_names
        original = textwrap.dedent("""\
            def merge(original, regions):
                lines = original.split("\\n")
                out = []
                prev = 0
                for _, (start, end, code) in sorted(regions.items()):
                    out.extend(lines[prev:start])
                    out.extend(code.split("\\n"))
                    prev = end
                out.extend(lines[prev:])
                return "\\n".join(out)
        """)
        patched = textwrap.dedent("""\
            def merge(original, regions):
                lines = original.split("\\n")
                out = [segment for region in sorted(regions.items())
                       for segment in (lines[region[0]:region[1]],
                       region[2].split("\\n"))]
                out.extend(lines[prev:])
                return "\\n".join(out)
        """)
        regressed = _regresses_defined_names(original, patched)
        assert regressed == ["prev"]

    def test_clean_patch_passes(self) -> None:
        from agent_core.commands.optimize_cmd import _regresses_defined_names
        original = textwrap.dedent("""\
            def f(items):
                out = []
                for x in items:
                    out.append(str(x))
                return ",".join(out)
        """)
        patched = textwrap.dedent("""\
            def f(items):
                out = [str(x) for x in items]
                return ",".join(out)
        """)
        assert _regresses_defined_names(original, patched) == []


class TestCountAnyIncreased:
    """_count_any_increased flags patches that fix one pattern but regress others."""

    def test_detects_cross_regression(self) -> None:
        from agent_core.commands.optimize_cmd import _count_any_increased
        old_code = textwrap.dedent("""\
            def f(items):
                out = []
                for x in items:
                    out.append(str(x))
                return ",".join(out)
        """)
        # fix introduces dead_assignment (out = [] overwritten)
        new_code = textwrap.dedent("""\
            def f(items):
                out = []
                out = [str(x) for x in items]
                return ",".join(out)
        """)
        regressed = _count_any_increased(old_code, new_code, exclude="list_append_join")
        assert "dead_assignment" in regressed

    def test_clean_fix_passes(self) -> None:
        from agent_core.commands.optimize_cmd import _count_any_increased
        old_code = textwrap.dedent("""\
            def f(items):
                out = []
                for x in items:
                    out.append(str(x))
                return ",".join(out)
        """)
        new_code = textwrap.dedent("""\
            def f(items):
                out = [str(x) for x in items]
                return ",".join(out)
        """)
        regressed = _count_any_increased(old_code, new_code, exclude="list_append_join")
        assert regressed == []


class TestMechanicalUnusedImport:
    """_fix_unused_import handles both single-name and multi-name import lines."""

    def test_whole_line_removal_single_name_from(self) -> None:
        from agent_core.commands.optimize_cmd import _fix_unused_import
        wl = ["from .types import TaskType"]
        finding = {"suggestion": "Imported 'TaskType' is never used. Remove the import."}
        hunk = _fix_unused_import(wl, 0, 1, "m.py", finding)
        assert hunk is not None
        assert "@@ -1,1 +1,0 @@" in hunk

    def test_removes_name_from_multiname_from_import(self) -> None:
        from agent_core.commands.optimize_cmd import _fix_unused_import
        src = "from .types import TaskType, ProfileType"
        wl = [src]
        finding = {"suggestion": "Imported 'TaskType' is never used. Remove the import."}
        hunk = _fix_unused_import(wl, 0, 1, "m.py", finding)
        assert hunk is not None
        assert "-" + src in hunk
        assert "+from .types import ProfileType" in hunk

    def test_removes_name_from_multiname_import_style(self) -> None:
        from agent_core.commands.optimize_cmd import _fix_unused_import
        src = "import os, sys"
        wl = [src]
        finding = {"suggestion": "Imported 'os' is never used. Remove the import."}
        hunk = _fix_unused_import(wl, 0, 1, "m.py", finding)
        assert hunk is not None
        assert "-" + src in hunk
        assert "+import sys" in hunk

    def test_last_name_removes_whole_line(self) -> None:
        from agent_core.commands.optimize_cmd import _fix_unused_import
        src = "from .types import TaskType"
        wl = [src]
        finding = {"suggestion": "Imported 'TaskType' is never used. Remove the import."}
        hunk = _fix_unused_import(wl, 0, 1, "m.py", finding)
        assert hunk is not None
        assert "@@ -1,1 +1,0 @@" in hunk

    def test_target_not_present_returns_none(self) -> None:
        from agent_core.commands.optimize_cmd import _fix_unused_import
        src = "from .types import ProfileType"
        wl = [src]
        finding = {"suggestion": "Imported 'TaskType' is never used. Remove the import."}
        hunk = _fix_unused_import(wl, 0, 1, "m.py", finding)
        assert hunk is None


class TestMechanicalListAppendJoin:
    """Mechanical _fix_list_append_join converts simple stateless loops."""

    def test_converts_simple_loop(self) -> None:
        from agent_core.commands.optimize_cmd import _fix_list_append_join
        code = textwrap.dedent("""\
            def f(items):
                out = []
                for x in items:
                    out.append(str(x))
                return ",".join(out)
        """)
        wl = code.split("\n")
        hunk = _fix_list_append_join(wl, 3, 4, "test.py", {})
        assert hunk is not None
        assert "out = [str(x)" in hunk

    def test_hunk_applies_and_compiles(self) -> None:
        from agent_core.commands.optimize_cmd import _fix_list_append_join
        from agent_core.patch_utils import apply_patch, apply_anchored_patch
        code = textwrap.dedent("""\
            def f(items):
                out = []
                for x in items:
                    out.append(str(x))
                return ",".join(out)
        """)
        wl = code.split("\n")
        hunk = _fix_list_append_join(wl, 3, 4, "test.py", {})
        assert hunk is not None
        ok, patched = apply_patch(hunk, wl)
        if not ok:
            ok, patched = apply_anchored_patch(hunk, wl)
        assert ok
        compile(patched, "<test>", "exec")
        assert "out.append(" not in patched
        assert "out = [str(x) for x in items]" in patched

    def test_stateful_loop_returns_none(self) -> None:
        from agent_core.commands.optimize_cmd import _fix_list_append_join
        code = textwrap.dedent("""\
            def merge(original, regions):
                lines = original.split("\\n")
                out = []
                prev = 0
                for _, (start, end, code) in sorted(regions.items()):
                    out.extend(lines[prev:start])
                    prev = end
                return "\\n".join(out)
        """)
        wl = code.split("\n")
        hunk = _fix_list_append_join(wl, 6, 7, "test.py", {})
        assert hunk is None

    def test_no_empty_init_returns_none(self) -> None:
        from agent_core.commands.optimize_cmd import _fix_list_append_join
        code = textwrap.dedent("""\
            lines = ["prefix"]
            for i in issues:
                lines.append(str(i))
        """)
        wl = code.split("\n")
        hunk = _fix_list_append_join(wl, 2, 3, "test.py", {})
        assert hunk is None

    def test_count_reduction(self) -> None:
        from agent_core.commands.optimize_cmd import _fix_list_append_join, _count_pattern
        from agent_core.patch_utils import apply_patch
        code = textwrap.dedent("""\
            def f(items):
                out = []
                for x in items:
                    out.append(str(x))
                return ",".join(out)
        """)
        wl = code.split("\n")
        hunk = _fix_list_append_join(wl, 3, 4, "test.py", {})
        ok, patched = apply_patch(hunk, wl)
        after = _count_pattern(patched, "list_append_join")
        assert after == 0
