"""Regression tests for scripts/setup_git_auth.py.

`.git/config` is machine-local, so the credential.helper ORDER has to be
re-applied once per clone. These tests drive the real script against real
throwaway git repositories (`git init` in tmp_path) rather than a simulated
config, so the assertions cover git's actual multi-value semantics.

The load-bearing property is ORDER: `manager` must be first, so that a dead
`.env` token can never shadow the working OS-store credential and cause
`401 Bad credentials` on every push.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "setup_git_auth_mod",
    REPO_ROOT / "scripts" / "setup_git_auth.py",
)
setup_git_auth = importlib.util.module_from_spec(SPEC)
# dataclasses looks the class's module up in sys.modules to resolve annotations,
# so the module must be registered BEFORE exec_module (the @dataclass decorator
# runs at import time and would otherwise hit a None module).
sys.modules[SPEC.name] = setup_git_auth
SPEC.loader.exec_module(setup_git_auth)


def _git(repo: Path, *args: str) -> str:
    """Run git in *repo* and return stdout (asserts the call succeeded)."""
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result.stdout


@pytest.fixture()
def tmp_repo(tmp_path, monkeypatch):
    """A real git repo with a stub helper script inside it."""
    _git(tmp_path, "init")
    helper = tmp_path / "scripts" / "git_credential_helper.cmd"
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_text("@echo off\n", encoding="utf-8")
    monkeypatch.setattr(setup_git_auth, "REPO_ROOT", tmp_path)
    return tmp_path


def _helpers(repo: Path) -> list[str]:
    """Read the ordered credential.helper list from a real repo."""
    result = subprocess.run(
        ["git", "config", "--local", "--get-all", "credential.helper"],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return result.stdout.splitlines()


class TestBuildEntries:
    def test_manager_comes_before_the_env_helper(self, tmp_repo):
        """The whole fix in one assertion: order is not cosmetic."""
        entries = setup_git_auth.build_entries(tmp_repo)
        helper_values = [e.value for e in entries if e.append]
        assert helper_values[0] == setup_git_auth.GCM_HELPER
        assert helper_values[0] == "manager"
        assert helper_values[1].startswith("!")
        assert helper_values[1].endswith("git_credential_helper.cmd")

    def test_helper_uses_absolute_forward_slash_path(self, tmp_repo):
        entries = setup_git_auth.build_entries(tmp_repo)
        helper_value = [e.value for e in entries if e.append][1]
        assert "\\" not in helper_value, "backslashes break git's ! helper"
        assert "scripts/git_credential_helper.cmd" in helper_value

    def test_username_is_pinned_and_interactive_disabled(self, tmp_repo):
        entries = setup_git_auth.build_entries(tmp_repo)
        by_key = {e.key: e.value for e in entries if not e.append}
        assert by_key["credential.https://github.com.username"] == "holgerkpedersen"
        assert by_key["credential.interactive"] == "never"

    def test_custom_username_is_honoured(self, tmp_repo):
        entries = setup_git_auth.build_entries(tmp_repo, username="someone-else")
        by_key = {e.key: e.value for e in entries if not e.append}
        assert by_key["credential.https://github.com.username"] == "someone-else"


class TestApply:
    def test_writes_manager_first_in_real_repo(self, tmp_repo):
        setup_git_auth.apply(tmp_repo)
        helpers = _helpers(tmp_repo)
        assert len(helpers) == 2
        assert helpers[0] == "manager"

    def test_is_idempotent(self, tmp_repo):
        """Re-running must not accumulate duplicates (clear-then-add)."""
        setup_git_auth.apply(tmp_repo)
        first = _helpers(tmp_repo)
        setup_git_auth.apply(tmp_repo)
        assert _helpers(tmp_repo) == first

    def test_clears_a_pre_existing_wrong_order(self, tmp_repo):
        """Reproduces the original breakage and proves apply repairs it."""
        _git(tmp_repo, "config", "--local", "--add", "credential.helper", "helper-env")
        _git(tmp_repo, "config", "--local", "--add", "credential.helper", "manager")
        assert _helpers(tmp_repo)[0] == "helper-env"

        setup_git_auth.apply(tmp_repo)
        assert _helpers(tmp_repo)[0] == "manager"

    def test_removes_a_stray_empty_entry(self, tmp_repo):
        """An empty credential.helper CLEARS all earlier helpers."""
        _git(tmp_repo, "config", "--local", "--add", "credential.helper", "")
        setup_git_auth.apply(tmp_repo)
        assert "" not in _helpers(tmp_repo)
        assert _helpers(tmp_repo)[0] == "manager"

    def test_does_not_touch_the_global_config(self, tmp_repo):
        """Only --local is written: a setup step must not alter other repos."""
        result = subprocess.run(
            ["git", "config", "--local", "--get", "credential.interactive"],
            cwd=str(tmp_repo),
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == ""  # not yet applied
        setup_git_auth.apply(tmp_repo)
        local = subprocess.run(
            ["git", "config", "--local", "--get", "credential.interactive"],
            cwd=str(tmp_repo),
            capture_output=True,
            text=True,
        )
        assert local.stdout.strip() == "never"


class TestIsConfigured:
    def test_false_on_a_fresh_clone(self, tmp_repo):
        assert setup_git_auth.is_configured(tmp_repo) is False

    def test_true_after_apply(self, tmp_repo):
        setup_git_auth.apply(tmp_repo)
        assert setup_git_auth.is_configured(tmp_repo) is True

    def test_false_when_order_is_reversed(self, tmp_repo):
        setup_git_auth.apply(tmp_repo)
        _git(tmp_repo, "config", "--local", "--unset-all", "credential.helper")
        for value in ("helper-env", "manager"):
            _git(tmp_repo, "config", "--local", "--add", "credential.helper", value)
        assert setup_git_auth.is_configured(tmp_repo) is False

    def test_false_when_username_pin_is_missing(self, tmp_repo):
        setup_git_auth.apply(tmp_repo)
        _git(tmp_repo, "config", "--local", "--unset", "credential.https://github.com.username")
        assert setup_git_auth.is_configured(tmp_repo) is False


class TestMain:
    def test_check_exits_zero_when_configured(self, tmp_repo, capsys):
        setup_git_auth.apply(tmp_repo)
        assert setup_git_auth.main(["--check"]) == 0
        assert "configured correctly" in capsys.readouterr().out

    def test_check_exits_one_when_not_configured(self, tmp_repo, capsys):
        assert setup_git_auth.main(["--check"]) == 1
        assert "MISCONFIGURED" in capsys.readouterr().out

    def test_check_does_not_mutate(self, tmp_repo):
        """A verification mode that writes would defeat its purpose."""
        setup_git_auth.main(["--check"])
        assert _helpers(tmp_repo) == []

    def test_plain_run_applies_and_exits_zero(self, tmp_repo):
        assert setup_git_auth.main([]) == 0
        assert _helpers(tmp_repo)[0] == "manager"

    def test_missing_helper_script_exits_two(self, tmp_repo, capsys):
        (tmp_repo / "scripts" / "git_credential_helper.cmd").unlink()
        assert setup_git_auth.main([]) == 2
        assert "helper missing" in capsys.readouterr().err
