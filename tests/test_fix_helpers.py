"""Tests for fix_cmd helpers: stdlib detection and trackable file checks."""

from __future__ import annotations
import os
import sys
import tempfile

from agent_core.commands.fix_cmd import (
    _is_stdlib_path,
    _is_trackable_file,
    _max_identical_run,
    _looks_corrupted,
    _collapse_duplicate_runs,
    _fix_unused_ignore,
    _fix_redundant_cast,
    _fix_implicit_optional,
    _fix_attr_defined_rename,
    _fix_missing_return_none,
    _fix_bare_generic,
    _fix_container_optional,
    _fix_tuple_arity,
    _fix_untyped_params,
    _type_context,
    _function_returns_value,
    _enclosing_function_name,
)


class TestIsStdlibPath:
    def test_python_install_path_is_stdlib(self):
        p = os.path.join(sys.prefix, "Lib", "functools.py")
        assert _is_stdlib_path(p)

    def test_base_prefix_path_is_stdlib(self):
        p = os.path.join(sys.base_prefix, "Lib", "os.py")
        assert _is_stdlib_path(p)

    def test_user_project_path_is_not_stdlib(self):
        p = os.path.join(os.getcwd(), "agent.py")
        assert not _is_stdlib_path(p)

    def test_tmp_path_is_not_stdlib(self):
        with tempfile.TemporaryDirectory() as td:
            assert not _is_stdlib_path(os.path.join(td, "test.py"))

    def test_case_insensitive_match(self):
        p = os.path.join(sys.prefix.upper(), "LIB", "FUNCTOOLS.PY")
        assert _is_stdlib_path(p)

    def test_empty_string_is_not_stdlib(self):
        assert not _is_stdlib_path("")

    def test_relative_path_in_cwd_is_not_stdlib(self):
        assert not _is_stdlib_path("agent.py")


class TestIsTrackableFile:
    def test_existing_user_file_is_trackable(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"print('hello')")
        try:
            assert _is_trackable_file(f.name)
        finally:
            os.unlink(f.name)

    def test_stdlib_file_is_not_trackable(self):
        p = os.path.join(sys.prefix, "Lib", "functools.py")
        assert not _is_trackable_file(p)

    def test_frozen_entry_is_not_trackable(self):
        assert not _is_trackable_file("<frozen runpy>")
        assert not _is_trackable_file("<stdin>")
        assert not _is_trackable_file("<frozen importlib.util>")

    def test_nonexistent_file_is_not_trackable(self):
        assert not _is_trackable_file("/nonexistent/path/file.py")


class TestCollapseDuplicateRuns:
    def test_collapses_long_run(self):
        text = "x = 1\n" + ("# todo\n" * 50) + "y = 2\n"
        out, runs = _collapse_duplicate_runs(text)
        assert runs == 1
        assert out == "x = 1\n# todo\ny = 2\n"

    def test_leaves_short_runs_alone(self):
        text = "a\nb\nb\nb\ny\n"  # run of 3 (< threshold 5)
        out, runs = _collapse_duplicate_runs(text)
        assert runs == 0
        assert out == text

    def test_leaves_blank_runs_alone(self):
        text = "a\n" + ("\n" * 20) + "b\n"
        out, runs = _collapse_duplicate_runs(text)
        assert runs == 0
        assert out == text


class TestLooksCorrupted:
    def test_duplicate_run_explosion(self):
        original = "x = 1\n# todo\ny = 2\n"
        result = "x = 1\n" + ("# todo\n" * 40) + "y = 2\n"
        assert _looks_corrupted(original, result) is not None

    def test_runaway_growth(self):
        original = "\n".join(f"line{i}" for i in range(40)) + "\n"
        result = original + ("\nextra\n" * 60)
        assert _looks_corrupted(original, result) is not None

    def test_clean_result_passes(self):
        original = ("def f():\n    return 1\n" * 3)
        result = "def f():\n    return 2\n"
        assert _looks_corrupted(original, result) is None


class TestMaxIdenticalRun:
    def test_counts_longest(self):
        assert _max_identical_run("a\nb\nb\nb\nc") == 3
    def test_ignores_blank_lines(self):
        assert _max_identical_run("\n\n\n\n\nx") == 1


class TestUnusedIgnore:
    def test_strips_trailing_directive(self):
        lines = ["        return foo()  # type: ignore[no-any-return]"]
        out = _fix_unused_ignore(lines, 1)
        assert out == ["        return foo()"]

    def test_drops_whole_line_when_only_comment(self):
        lines = ["    # type: ignore", "    x = 1"]
        out = _fix_unused_ignore(lines, 1)
        assert out == ["    x = 1"]

    def test_noop_without_directive(self):
        lines = ["    x = 1  # normal comment"]
        assert _fix_unused_ignore(lines, 1) is None


class TestRedundantCast:
    def test_removes_simple_cast(self):
        lines = ["    return cast(str, await fs.write(p, c))"]
        out = _fix_redundant_cast(lines, 1, "str")
        assert out == ["    return await fs.write(p, c)"]

    def test_wrong_type_noop(self):
        lines = ["    return cast(str, x)"]
        assert _fix_redundant_cast(lines, 1, "int") is None

    def test_no_cast_noop(self):
        lines = ["    return x"]
        assert _fix_redundant_cast(lines, 1, "str") is None


class TestImplicitOptional:
    def test_adds_optional(self):
        lines = ["    async def search_file(self, query: str, path: str = None) -> str:"]
        out = _fix_implicit_optional(lines, 1)
        assert out is not None
        assert "path: str | None = None" in out[0]

    def test_skips_already_optional(self):
        lines = ["    def f(self, path: str | None = None) -> str:"]
        assert _fix_implicit_optional(lines, 1) is None

    def test_noop_without_default_none(self):
        lines = ["    def f(self, path: str = '.') -> str:"]
        assert _fix_implicit_optional(lines, 1) is None


class TestAttrDefinedRename:
    SRC = (
        "class Agent:\n"
        "    def _tool_delete_file(self, p):\n"
        "        return self._delete_file(p)\n"
        "\n"
        "    def other(self, p):\n"
        "        return self._delete_file(p)\n"
    )

    def test_renames_in_safe_caller(self):
        lines = self.SRC.split("\n")
        out = _fix_attr_defined_rename(lines.copy(), 6, self.SRC, "_delete_file", "_tool_delete_file")
        assert out is not None
        assert out[5] == "        return self._tool_delete_file(p)"

    def test_skips_recursive_case(self):
        lines = self.SRC.split("\n")
        # line 3 is inside def _tool_delete_file; renaming there = recursion
        out = _fix_attr_defined_rename(lines.copy(), 3, self.SRC, "_delete_file", "_tool_delete_file")
        assert out is None


class TestMissingReturnNone:
    def test_adds_to_void_function(self):
        lines = ["def cleanup():", "    cache.clear()"]
        out = _fix_missing_return_none(lines, 1)
        assert out is not None
        assert out[0] == "def cleanup() -> None:"

    def test_skips_value_returning_function(self):
        src = "def get():\n    return 1\n"
        lines = src.split("\n")
        out = _fix_missing_return_none(lines, 1)
        assert out is None

    def test_skips_already_annotated(self):
        lines = ["    def f(self) -> str:", "        return 'x'"]
        assert _fix_missing_return_none(lines, 1) is None

    def test_multiline_signature(self):
        src = (
            "def long_signature(\n"
            "    a,\n"
            "    b,\n"
            "):\n"
            "    pass\n"
        )
        out = _fix_missing_return_none(src.split("\n"), 1)
        assert out is not None
        assert out[3] == ") -> None:"


class TestEnclosingFunction:
    def test_finds_enclosing(self):
        src = "def outer():\n    x = 1\n    def inner():\n        y = 2\n"
        assert _enclosing_function_name(src, 4) == "inner"
        assert _enclosing_function_name(src, 2) == "outer"

    def test_none_outside_function(self):
        assert _enclosing_function_name("x = 1\n", 1) is None


class TestFixBareGeneric:
    def test_fills_bare_dict_in_annotation(self):
        lines = ["def f() -> list[dict]:", "    return []"]
        out = _fix_bare_generic(lines, 1, 'error: Missing type arguments for generic type "dict" [type-arg]')
        assert out is not None
        assert "def f() -> list[dict[str, Any]]:" in out

    def test_leaves_parameterized_alone(self):
        lines = ["x: dict[str, int] = {}"]
        assert _fix_bare_generic(lines, 1, 'error: Missing type arguments for generic type "dict" [type-arg]') is None

    def test_leaves_call_alone(self):
        lines = ["d = dict()"]
        assert _fix_bare_generic(lines, 1, 'error: Missing type arguments for generic type "dict" [type-arg]') is None

    def test_match_generic(self):
        lines = ["m: re.Match | None = None"]
        out = _fix_bare_generic(lines, 1, 'error: Missing type arguments for generic type "Match" [type-arg]')
        assert out is not None
        assert "m: re.Match[Any] | None = None" in out


class TestFixContainerOptional:
    ERR = 'error: Item "None" of "str | None" has no attribute "strip" [union-attr]'

    def test_narrows_container_element(self):
        lines = [
            "def apply():",
            "    chunks: list[tuple[str, str | None]] = []",
            "    chunks.append(('-', line[1:]))",
            "    if any(op == '+' and not text.strip() for op, text in chunks):",
            "        pass",
        ]
        out = _fix_container_optional(lines, TestFixContainerOptional.ERR)
        assert out is not None
        assert out[1] == "    chunks: list[tuple[str, str]] = []"

    def test_ignores_plain_local_optional(self):
        lines = [
            "def f():",
            "    guard_match: Match[str] | None = re.search('x', body)",
            "    return guard_match.group(1)",
        ]
        assert _fix_container_optional(lines, TestFixContainerOptional.ERR) is None

    def test_ignores_without_container(self):
        lines = ["x: str | None = None"]
        assert _fix_container_optional(lines, TestFixContainerOptional.ERR) is None

    def test_ignores_docstring_prose(self):
        """A docstring that merely *mentions* ``str | None`` next to
        ``list[...]`` is prose, not an annotation â€” it must never be edited."""
        lines = [
            "def f():",
            '    """Narrow ``X | None`` to ``X`` inside a container element.',
            "    ``str | None`` lives in a ``list[...]``/``tuple[...]``/``dict[...]``",
            '    element slot."""',
            "    chunks: list[tuple[str, str | None]] = []",
        ]
        out = _fix_container_optional(lines, TestFixContainerOptional.ERR)
        assert out is not None
        assert out[4] == "    chunks: list[tuple[str, str]] = []"
        assert out[2] == "    ``str | None`` lives in a ``list[...]``/``tuple[...]``/``dict[...]``"

    def test_ignores_pure_docstring(self):
        lines = [
            "def f():",
            '    """``str | None`` lives in a ``list[...]`` slot."""',
            "    pass",
        ]
        assert _fix_container_optional(lines, TestFixContainerOptional.ERR) is None

    def test_ignores_comment_mention(self):
        lines = [
            "def f():",
            "    # note: ``str | None`` in a ``list[...]``",
            "    chunks: list[tuple[str, str | None]] = []",
        ]
        out = _fix_container_optional(lines, TestFixContainerOptional.ERR)
        assert out is not None
        assert out[2] == "    chunks: list[tuple[str, str]] = []"
        assert out[1] == "    # note: ``str | None`` in a ``list[...]``"


class TestFixTupleArity:
    ERR = "error: Too many values to unpack (2 expected, 3 provided) [misc]"

    def test_truncates_annotation(self):
        lines = [
            "def apply():",
            "    valid: list[tuple[int, list[tuple[str, str]], bool]] = []",
            "    for claimed, chunks in valid:",
            "        pass",
        ]
        out = _fix_tuple_arity(lines, 3, TestFixTupleArity.ERR)
        assert out is not None
        assert out[1] == "    valid: list[tuple[int, list[tuple[str, str]]]] = []"

    def test_skips_when_arity_matches(self):
        lines = [
            "def apply():",
            "    valid: list[tuple[int, str]] = []",
            "    for a, b in valid:",
            "        pass",
        ]
        assert _fix_tuple_arity(lines, 3, TestFixTupleArity.ERR) is None


class TestFixUntypedParams:
    def test_annotates_all_params(self):
        src = (
            "from typing import Optional\n"
            "def f(a, b: int, c=3):\n"
            "    pass\n"
        )
        out = _fix_untyped_params(src.split("\n"), 2)
        assert out is not None
        assert out[0] == "from typing import Optional, Any"
        assert out[1] == "def f(a: Any, b: int, c: Any = 3):"

    def test_adds_typing_import_when_missing(self):
        src = "def f(x):\n    pass\n"
        out = _fix_untyped_params(src.split("\n"), 1)
        assert out is not None
        assert "from typing import Any" in out

    def test_skips_self(self):
        src = "class C:\n    def m(self):\n        pass\n"
        out = _fix_untyped_params(src.split("\n"), 2)
        assert out is None

class TestTypeContext:
    SRC = [
        "def _fix_untyped_params(lines: list[str], def_lineno: int) -> list[str] | None:",
        "    idx = def_lineno - 1",
        "    edits: list[tuple[int, int]] = []",
        "    for a in args:",
        "        if a.annotation is None:",
        "            edits.append((a.end_lineno - 1, a.end_col_offset))",
        "    return None",
    ]

    def test_operator_error_uses_error_line_identifiers(self):
        err = 't.py:6: error: Unsupported operand types for - ("None" and "int")  [operator]'
        ctx = _type_context(self.SRC, err, 6)
        assert "edits: list[tuple[int, int]]" in ctx
        assert "a in args" in ctx

    def test_message_identifiers_win_over_line(self):
        err = 't.py:5: error: Item "None" of "str | None" has no attribute "strip"  [union-attr]'
        ctx = _type_context(self.SRC, err, 5)
        # "a" is on the error line but the message names no variable ->
        # the error line is used, so "a in args" must be included
        assert "a in args" in ctx

    def test_empty_outside_function(self):
        src = ["x = 1", "y = 2"]
        assert _type_context(src, "t.py:1: error: Name 'x' is not defined  [name-defined]", 1) == ""

    def test_value_returning_no_annotations(self):
        """A function with neither param nor return annotations (the class
        where the LLM fails) must get : Any params + -> Any."""
        src = [
            "def _fix_dead_assignment(wl, idx, line, basename, finding):",
            "    if 'never used after' in finding.get('suggestion', ''):",
            "        return f'@@ -{line},1 +{line},0 @@\\n-{wl[idx]}'",
            "    return None",
        ]
        out = _fix_untyped_params(src, 1)
        assert out is not None
        assert "def _fix_dead_assignment(wl: Any, idx: Any, line: Any, basename: Any, finding: Any) -> Any:" in out
        assert "from typing import Any" in out

    def test_annotated_params_missing_return(self):
        """Annotated params but a missing return annotation -> only -> Any."""
        src = [
            "def f(a: int, b: str):",
            "    return a + len(b)",
        ]
        out = _fix_untyped_params(src, 1)
        assert out is not None
        assert "def f(a: int, b: str) -> Any:" in out
        assert "a: Any" not in out

    def test_fully_annotated_no_change(self):
        src = ["def g(a: int, b: str) -> int:", "    return a"]
        assert _fix_untyped_params(src, 1) is None
