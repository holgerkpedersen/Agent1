"""Regression tests for scripts/git_credential_helper.py.

The helper supplies the non-interactive `git push` credential, so its contract
is narrow and must not drift:

  * `get` emits `username=git` plus the token, and nothing else.
  * `store` / `erase` emit nothing (git only needs `get`).
  * No token -> emit nothing, so git falls through to the NEXT helper in the
    ordered `credential.helper` list (Git Credential Manager).

That last point is the one that matters operationally: a broken or expired
`.env` token must not poison the chain. If this helper emitted an empty or
placeholder password, git would send it instead of falling through to the
working OS-store credential, and every push would fail with
`401 Bad credentials` even though a valid credential existed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "git_credential_helper_mod",
    REPO_ROOT / "scripts" / "git_credential_helper.py",
)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)

TOKEN = "ghp_" + "a" * 36


@pytest.fixture()
def env_repo(tmp_path, monkeypatch):
    """Point the helper at a throwaway repo root and clear the env fallback."""
    monkeypatch.setattr(helper, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    return tmp_path


def _write_env(root: Path, body: str) -> None:
    (root / ".env").write_text(body, encoding="utf-8")


def _run(monkeypatch, capsys, action: str | None) -> str:
    argv = ["git_credential_helper.py"]
    if action is not None:
        argv.append(action)
    monkeypatch.setattr(helper.sys, "argv", argv)
    helper.main()
    return capsys.readouterr().out


class TestReadToken:
    def test_reads_token_from_dotenv(self, env_repo):
        _write_env(env_repo, f"GITHUB_TOKEN={TOKEN}\n")
        assert helper._read_token() == TOKEN

    def test_key_match_is_case_insensitive(self, env_repo):
        _write_env(env_repo, f"github_token={TOKEN}\n")
        assert helper._read_token() == TOKEN

    def test_strips_quotes_and_whitespace(self, env_repo):
        """Leading/trailing padding on the line and around the value is fine."""
        _write_env(env_repo, f'  GITHUB_TOKEN=  "{TOKEN}"  \n')
        assert helper._read_token() == TOKEN

    def test_space_before_equals_is_not_a_match(self, env_repo):
        """Pin the known limitation: the key must be `GITHUB_TOKEN=` exactly.

        A space before `=` (`GITHUB_TOKEN = ...`) is NOT parsed. This is
        intentional-by-omission rather than a desired feature: the repo's real
        .env has no space, and emitting nothing on an unparsed line is the safe
        failure, because git then falls through to the next helper instead of
        sending a bogus password.
        """
        _write_env(env_repo, f"GITHUB_TOKEN = {TOKEN}\n")
        assert helper._read_token() == ""

    def test_ignores_comments_and_other_keys(self, env_repo):
        _write_env(
            env_repo,
            "# GITHUB_TOKEN=commented_out\n"
            "OTHER_KEY=value\n"
            f"GITHUB_TOKEN={TOKEN}\n",
        )
        assert helper._read_token() == TOKEN

    def test_falls_back_to_process_environment(self, env_repo, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "from-env")
        assert helper._read_token() == "from-env"

    def test_dotenv_wins_over_environment(self, env_repo, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "from-env")
        _write_env(env_repo, f"GITHUB_TOKEN={TOKEN}\n")
        assert helper._read_token() == TOKEN

    def test_absent_env_file_and_variable_gives_empty(self, env_repo):
        assert helper._read_token() == ""

    def test_missing_token_key_gives_empty(self, env_repo):
        _write_env(env_repo, "OTHER=1\n")
        assert helper._read_token() == ""


class TestMain:
    def test_get_emits_username_and_password(self, env_repo, monkeypatch, capsys):
        _write_env(env_repo, f"GITHUB_TOKEN={TOKEN}\n")
        lines = _run(monkeypatch, capsys, "get").splitlines()
        assert lines == ["username=git", f"password={TOKEN}"]

    def test_get_without_token_emits_nothing(self, env_repo, monkeypatch, capsys):
        """No token must mean NO output, so the next helper is tried."""
        assert _run(monkeypatch, capsys, "get") == ""

    @pytest.mark.parametrize("action", ["store", "erase"])
    def test_store_and_erase_are_silent(self, env_repo, monkeypatch, capsys, action):
        _write_env(env_repo, f"GITHUB_TOKEN={TOKEN}\n")
        assert _run(monkeypatch, capsys, action) == ""

    def test_no_action_argument_emits_nothing(self, env_repo, monkeypatch, capsys):
        _write_env(env_repo, f"GITHUB_TOKEN={TOKEN}\n")
        assert _run(monkeypatch, capsys, None) == ""

    def test_password_is_the_token_with_no_trailing_junk(
        self, env_repo, monkeypatch, capsys
    ):
        """A stray \\r in the password is what breaks a git send."""
        _write_env(env_repo, f"GITHUB_TOKEN={TOKEN}\n")
        out = _run(monkeypatch, capsys, "get")
        password = next(
            line for line in out.splitlines() if line.startswith("password=")
        )
        assert password == f"password={TOKEN}"
        assert "\r" not in password
        assert password[len("password=") :] == TOKEN
