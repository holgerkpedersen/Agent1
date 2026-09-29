"""Regression tests for agent_core.utils.lint_guard (was at 0% coverage).

Covers the full public surface of ``LintGuard`` / ``LintIssue`` against real
source strings and temp files: duplicate module-level definitions, bare
excepts, file/workspace scanning with ignore rules, unreadable inputs, and
the newly-introduced-only semantics of ``scan_regression``.
"""

import dataclasses

import pytest

from agent_core.utils.lint_guard import LintGuard, LintIssue


@pytest.fixture()
def guard():
    return LintGuard()


class TestLintIssue:
    def test_fields_and_default_name(self):
        issue = LintIssue(kind="k", path="p.py", line=3, message="m")
        assert (issue.kind, issue.path, issue.line) == ("k", "p.py", 3)
        assert issue.message == "m"
        assert issue.name == ""

    def test_frozen(self):
        issue = LintIssue(kind="k", path="p", line=1, message="m")
        with pytest.raises(dataclasses.FrozenInstanceError):
            issue.kind = "other"


class TestDuplicateDefinitions:
    def test_single_definition_is_clean(self, guard):
        assert guard.find_duplicate_definitions("def foo():\n    pass\n") == []

    def test_duplicate_def_flagged_at_second_occurrence(self, guard):
        source = "def foo():\n    pass\n\n\ndef foo():\n    pass\n"
        issues = guard.find_duplicate_definitions(source, "x.py")
        assert len(issues) == 1
        issue = issues[0]
        assert issue.kind == "duplicate-definition"
        assert issue.path == "x.py"
        assert issue.line == 5
        assert issue.name == "foo"
        assert "first defined at line 1" in issue.message

    def test_duplicate_across_kinds_shares_namespace(self, guard):
        source = "class A:\n    pass\n\n\ndef A():\n    pass\n"
        issues = guard.find_duplicate_definitions(source)
        assert [issue.name for issue in issues] == ["A"]

    def test_async_def_duplicates_counted(self, guard):
        source = (
            "async def work():\n    pass\n\n\n"
            "async def work():\n    pass\n"
        )
        issues = guard.find_duplicate_definitions(source)
        assert len(issues) == 1 and issues[0].name == "work"

    def test_third_occurrence_reported_against_first(self, guard):
        source = "def f():\n    pass\n\ndef f():\n    pass\n\ndef f():\n    pass\n"
        issues = guard.find_duplicate_definitions(source)
        assert [issue.line for issue in issues] == [4, 7]
        assert all("first defined at line 1" in i.message for i in issues)

    def test_nested_and_method_definitions_ignored(self, guard):
        source = (
            "def outer():\n"
            "    def inner():\n"
            "        pass\n"
            "\n"
            "class C:\n"
            "    def m(self):\n"
            "        pass\n"
            "\n"
            "def outer2():\n"
            "    def inner():\n"
            "        pass\n"
            "\n"
            "class D:\n"
            "    def m(self):\n"
            "        pass\n"
        )
        assert guard.find_duplicate_definitions(source) == []

    def test_syntax_error_returns_empty(self, guard):
        assert guard.find_duplicate_definitions("def broken(:") == []


class TestBareExcepts:
    def test_bare_except_flagged(self, guard):
        source = "try:\n    pass\nexcept:\n    pass\n"
        issues = guard.find_bare_excepts(source, "y.py")
        assert len(issues) == 1
        issue = issues[0]
        assert issue.kind == "bare-except"
        assert issue.path == "y.py"
        assert issue.line == 3
        assert issue.name == ""
        assert issue.message == (
            "Bare except clause; catch a concrete exception instead"
        )

    def test_typed_except_not_flagged(self, guard):
        source = (
            "try:\n"
            "    pass\n"
            "except ValueError:\n"
            "    pass\n"
            "\n"
            "try:\n"
            "    pass\n"
            "except (OSError, KeyError):\n"
            "    pass\n"
        )
        assert guard.find_bare_excepts(source) == []

    def test_nested_bare_excepts_all_found(self, guard):
        source = (
            "def f():\n"
            "    try:\n"
            "        pass\n"
            "    except:\n"
            "        pass\n"
            "\n"
            "try:\n"
            "    pass\n"
            "except:\n"
            "    pass\n"
        )
        assert len(guard.find_bare_excepts(source)) == 2

    def test_syntax_error_returns_empty(self, guard):
        assert guard.find_bare_excepts("try:") == []


class TestScanSource:
    def test_combines_duplicates_then_bare_excepts(self, guard):
        source = (
            "def f():\n"
            "    try:\n"
            "        pass\n"
            "    except:\n"
            "        pass\n"
            "\n"
            "def f():\n"
            "    pass\n"
        )
        issues = guard.scan_source(source)
        assert [i.kind for i in issues] == ["duplicate-definition", "bare-except"]

    def test_default_path_is_string_sentinel(self, guard):
        source = "try:\n    pass\nexcept:\n    pass\n"
        issues = guard.find_bare_excepts(source)
        assert issues[0].path == "<string>"


class TestScanFile:
    def test_real_file_reports_with_resolved_path(self, guard, tmp_path):
        target = tmp_path / "bad.py"
        target.write_text(
            "def f():\n    pass\n\ndef f():\n    pass\n", encoding="utf-8"
        )
        issues = guard.scan_file(target)
        assert len(issues) == 1
        assert issues[0].path == str(target)

    def test_missing_file_returns_empty(self, guard, tmp_path):
        assert guard.scan_file(tmp_path / "nope.py") == []

    def test_undecodable_bytes_return_empty(self, guard, tmp_path):
        target = tmp_path / "binary.py"
        target.write_bytes(b"\xff\xfe\x00def f():\n    pass\n")
        assert guard.scan_file(target) == []


class TestScanWorkspace:
    def test_scans_sorted_and_skips_ignored_dirs(self, tmp_path):
        (tmp_path / "a_bad.py").write_text(
            "def a():\n    pass\n\ndef a():\n    pass\n", encoding="utf-8"
        )
        (tmp_path / "z_ok.py").write_text("VALUE = 1\n", encoding="utf-8")
        venv_pkg = tmp_path / ".venv" / "pkg"
        venv_pkg.mkdir(parents=True)
        (venv_pkg / "hidden.py").write_text(
            "def h():\n    pass\n\ndef h():\n    pass\n", encoding="utf-8"
        )
        issues = LintGuard(workspace=tmp_path).scan_workspace()
        assert [i.path for i in issues] == [str(tmp_path / "a_bad.py")]

    def test_default_workspace_is_cwd(self, monkeypatch, tmp_path):
        (tmp_path / "here.py").write_text(
            "def c():\n"
            "    try:\n"
            "        pass\n"
            "    except:\n"
            "        pass\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        guard = LintGuard()
        assert guard.workspace == tmp_path
        assert [i.kind for i in guard.scan_workspace()] == ["bare-except"]


class TestScanRegression:
    def test_no_change_no_issues(self, guard):
        source = "def f():\n    pass\n"
        assert guard.scan_regression(source, source) == []

    def test_new_duplicate_reported_preexisting_suppressed(self, guard):
        before = "def old():\n    pass\n\ndef old():\n    pass\n"
        after = before + "\ndef new():\n    pass\n\ndef new():\n    pass\n"
        issues = guard.scan_regression(before, after, "r.py")
        assert [i.name for i in issues] == ["new"]

    def test_new_bare_except_reported_when_before_clean(self, guard):
        before = "def f():\n    pass\n"
        after = (
            "def f():\n"
            "    try:\n"
            "        pass\n"
            "    except:\n"
            "        pass\n"
        )
        issues = guard.scan_regression(before, after)
        assert [i.kind for i in issues] == ["bare-except"]

    def test_bare_excepts_suppressed_when_before_had_one(self, guard):
        before = "try:\n    pass\nexcept:\n    pass\n"
        after = (
            "try:\n    pass\nexcept:\n    pass\n"
            "\ndef f():\n    try:\n        pass\n    except:\n        pass\n"
        )
        assert guard.scan_regression(before, after) == []

    def test_unparseable_before_treated_as_clean_baseline(self, guard):
        before = "def broken(:\n"
        after = "def g():\n    pass\n\ndef g():\n    pass\n"
        issues = guard.scan_regression(before, after)
        assert [i.name for i in issues] == ["g"]
