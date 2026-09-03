"""Regression tests for the full-suite pytest time-budget watchdog in conftest.py.

The watchdog must:
  * resolve a budget of max(floor, last full-run duration * (1 + margin)),
    falling back to the default floor on invalid/missing values;
  * treat only bare `pytest` invocations (no explicit test paths) as full runs;
  * never duplicate keys when updating the repo .env across runs;
  * abort the process with exit code 124 (timeout convention) when a full run
    exceeds its budget, and stay silent when the budget is respected;
  * record the elapsed time of full runs only, never targeted runs.
"""

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import conftest

_REPO_ROOT = Path(conftest.__file__).resolve().parent


class TestResolveBudget:
    def test_floor_wins_when_last_run_small(self, monkeypatch):
        monkeypatch.setenv(conftest._FULL_SUITE_TIMEOUT_KEY, "600")
        monkeypatch.setenv(conftest._LAST_FULL_RUN_KEY, "10")
        assert conftest._resolve_budget() == 600.0

    def test_last_run_with_margin_wins_when_larger(self, monkeypatch):
        monkeypatch.setenv(conftest._FULL_SUITE_TIMEOUT_KEY, "600")
        monkeypatch.setenv(conftest._LAST_FULL_RUN_KEY, "1000")
        assert conftest._resolve_budget() == 1200.0  # 1000 * 1.20

    def test_invalid_values_fall_back_to_default(self, monkeypatch):
        monkeypatch.delenv(conftest._FULL_SUITE_TIMEOUT_KEY, raising=False)
        monkeypatch.delenv(conftest._LAST_FULL_RUN_KEY, raising=False)
        monkeypatch.setattr(
            conftest, "_load_env",
            lambda: {
                conftest._FULL_SUITE_TIMEOUT_KEY: "abc",
                conftest._LAST_FULL_RUN_KEY: "xyz",
            },
        )
        assert conftest._resolve_budget() == conftest._DEFAULT_FULL_SUITE_TIMEOUT

    def test_no_env_uses_default(self, monkeypatch):
        monkeypatch.delenv(conftest._FULL_SUITE_TIMEOUT_KEY, raising=False)
        monkeypatch.delenv(conftest._LAST_FULL_RUN_KEY, raising=False)
        monkeypatch.setattr(conftest, "_load_env", lambda: {})
        assert conftest._resolve_budget() == conftest._DEFAULT_FULL_SUITE_TIMEOUT


class TestGetEnvValue:
    def test_process_env_wins_over_env_file(self, monkeypatch):
        monkeypatch.setenv(conftest._FULL_SUITE_TIMEOUT_KEY, "42")
        monkeypatch.setattr(
            conftest, "_load_env",
            lambda: {conftest._FULL_SUITE_TIMEOUT_KEY: "600"},
        )
        assert conftest._get_env_value(conftest._FULL_SUITE_TIMEOUT_KEY) == "42"

    def test_env_file_value_used_when_no_process_env(self, monkeypatch):
        monkeypatch.delenv(conftest._FULL_SUITE_TIMEOUT_KEY, raising=False)
        monkeypatch.setattr(
            conftest, "_load_env",
            lambda: {conftest._FULL_SUITE_TIMEOUT_KEY: "600"},
        )
        assert conftest._get_env_value(conftest._FULL_SUITE_TIMEOUT_KEY) == "600"


class TestIsFullRun:
    @staticmethod
    def _config(args):
        return type(
            "C", (), {"invocation_params": type("P", (), {"args": args})}
        )()

    def test_bare_pytest_is_full_run(self):
        assert conftest._is_full_run(self._config([])) is True

    def test_only_flags_is_full_run(self):
        assert conftest._is_full_run(self._config(["-q", "--tb=short"])) is True

    def test_explicit_path_is_not_full_run(self):
        assert conftest._is_full_run(self._config(["tests/test_x.py"])) is False

    def test_path_with_flags_is_not_full_run(self):
        assert conftest._is_full_run(self._config(["-q", "tests/test_x.py"])) is False


class TestEnvRoundTrip:
    def test_save_preserves_comments_and_other_keys(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# comment\nGITHUB_TOKEN=abc\n\nPYTEST_FULL_SUITE_TIMEOUT=600\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(conftest, "_ENV_FILE", env_file)
        conftest._save_env({conftest._LAST_FULL_RUN_KEY: "123.4"})
        content = env_file.read_text(encoding="utf-8")
        assert "# comment" in content
        assert "GITHUB_TOKEN=abc" in content
        assert "PYTEST_FULL_SUITE_TIMEOUT=600" in content
        assert "PYTEST_LAST_FULL_RUN_SECONDS=123.4" in content

    def test_repeated_saves_never_duplicate_keys(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("", encoding="utf-8")
        monkeypatch.setattr(conftest, "_ENV_FILE", env_file)
        for value in ("100.0", "200.0", "300.0"):
            conftest._save_env({conftest._LAST_FULL_RUN_KEY: value})
        content = env_file.read_text(encoding="utf-8")
        assert content.count("PYTEST_LAST_FULL_RUN_SECONDS=") == 1
        assert "PYTEST_LAST_FULL_RUN_SECONDS=300.0" in content

    def test_load_skips_comments_and_blank_lines(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# only a comment\n\nA=1\n# another comment\nB=2\n", encoding="utf-8"
        )
        monkeypatch.setattr(conftest, "_ENV_FILE", env_file)
        assert conftest._load_env() == {"A": "1", "B": "2"}


class TestWatchdogAborts:
    def test_over_budget_exits_124(self):
        code = textwrap.dedent(
            """
            import sys, threading, time
            sys.path.insert(0, {root!r})
            import conftest
            stop = threading.Event()
            # started 5s in the past, budget 1s -> elapsed already exceeds it.
            threading.Thread(
                target=conftest._watchdog_loop,
                args=(time.monotonic() - 5.0, 1.0, stop),
                daemon=True,
            ).start()
            time.sleep(2.0)
            print("SURVIVED")
            """
        ).format(root=str(_REPO_ROOT))
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
        )
        assert proc.returncode == 124  # timeout convention, same as `timeout(1)`
        assert "exceeded" in proc.stdout

    def test_within_budget_stays_alive(self):
        code = textwrap.dedent(
            """
            import sys, threading, time
            sys.path.insert(0, {root!r})
            import conftest
            stop = threading.Event()
            threading.Thread(
                target=conftest._watchdog_loop,
                args=(time.monotonic(), 3600.0, stop),
                daemon=True,
            ).start()
            time.sleep(1.5)
            stop.set()
            print("OK")
            """
        ).format(root=str(_REPO_ROOT))
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
        )
        assert proc.returncode == 0
        assert "OK" in proc.stdout


class TestSessionHooks:
    def test_sessionfinish_records_elapsed_for_full_run(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("", encoding="utf-8")
        monkeypatch.setattr(conftest, "_ENV_FILE", env_file)
        session = type(
            "S", (), {"_is_full_run": True, "_full_run_started": time.monotonic() - 42.0}
        )()
        conftest.pytest_sessionfinish(session, 0)
        assert "PYTEST_LAST_FULL_RUN_SECONDS=42.0" in env_file.read_text(encoding="utf-8")

    def test_sessionfinish_skips_targeted_run(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("", encoding="utf-8")
        monkeypatch.setattr(conftest, "_ENV_FILE", env_file)
        session = type("S", (), {"_is_full_run": False})()
        conftest.pytest_sessionfinish(session, 0)
        assert env_file.read_text(encoding="utf-8") == ""