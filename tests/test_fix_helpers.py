"""Tests for fix_cmd helpers: stdlib detection and trackable file checks."""
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
