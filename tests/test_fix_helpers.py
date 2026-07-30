"""Tests for fix_cmd helpers: stdlib detection and trackable file checks."""
import os
import sys
import tempfile

from agent_core.commands.fix_cmd import _is_stdlib_path, _is_trackable_file


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
