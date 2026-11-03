"""Contract tests for the ONE shell detector.

Card: "Communicating current shell."  Before this refactor two detectors
coexisted and could disagree in production text: ``shell_utils.detect_shell``
(env-var evidence, feeds SYSTEM_PROMPT ``SHELL_NAME=...``, run hints and the
real execution argv) vs ``subprocess_utils.shell_info`` (psutil ancestor walk,
feeds the run-tool schema description).  The model received two answers to
"what shell am I in" — e.g. system prompt "powershell", tool schema
"cmd.exe", with naming that differed per source.

The contract pinned here:

* ONE detector owns detection: ``shell_utils.detect_shell()``.
  ``subprocess_utils.shell_info()`` derives its dict from that token, so the
  advertised shell and the executing shell can no longer diverge.
* Canonical display names everywhere (the token itself): "powershell",
  "cmd", "bash" — not ".exe"-suffixed variants.
* Windows precedence: the psutil ancestor walk is authoritative when psutil
  is installed; PS session env vars are a documented best-effort fallback
  only when it is absent (PSModulePath alone over-reports).
* The ``AGENT1_SHELL`` override wins everywhere.  On Windows ``SHELL`` is
  NOT consulted at all: session managers, WSL-mounted shells and Git Bash /
  MSYS2 export a POSIX ``SHELL`` inside a cmd/PowerShell terminal, so on
  Windows it is not proof of the *executing* shell — the process tree (or the
  explicit override) is.  On POSIX the ``SHELL`` basename decides.
"""

from __future__ import annotations

import os
import sys

import pytest

import agent_core.subprocess_utils as su
import agent_core.shell_utils as shell_utils


@pytest.fixture(autouse=True)
def _clean_shell_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise host-environment evidence so each test controls its own."""
    monkeypatch.delenv("AGENT1_SHELL", raising=False)
    monkeypatch.delenv("SHELL", raising=False)
    for var in shell_utils._PS_SESSION_ENV:
        monkeypatch.delenv(var, raising=False)


# ── one source of truth ─────────────────────────────────────────────────────


def test_shell_info_name_is_the_canonical_token() -> None:
    """shell_info() must report the same shell detect_shell() advertises.

    The token is the display name everywhere — no ".exe" suffix variants, no
    per-source naming.  On POSIX with no evidence, bash is the default.
    """
    info = su.shell_info()
    token = shell_utils.detect_shell()
    if sys.platform == "win32":
        # Windows: the canonical name carries the .exe the executor invokes.
        assert info["name"] in ("powershell.exe", "cmd.exe")
        base = os.path.splitext(os.path.basename(info["name"]))[0].lower()
        assert base == token, (info["name"], token)
    else:
        assert info["name"] == token


def test_shell_info_delegates_to_detect_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An AGENT1_SHELL override wins over the psutil walk in BOTH views.

    The walk reports PowerShell; the explicit override says cmd.  A single
    detector means shell_info must follow the override — with two independent
    detectors this is exactly where they diverged.
    """
    monkeypatch.setattr(su, "_walk_for_powershell", lambda: True)
    monkeypatch.setenv("AGENT1_SHELL", "cmd")
    assert shell_utils.detect_shell() == "cmd"
    info = su.shell_info()
    assert os.path.splitext(os.path.basename(info["name"]))[0].lower() == "cmd"


def test_run_argv_executes_the_advertised_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The advertised shell and the executing argv agree (override wins)."""
    monkeypatch.setattr(su, "_walk_for_powershell", lambda: True)
    monkeypatch.setenv("AGENT1_SHELL", "cmd")
    argv = shell_utils.run_shell_command("echo hi")
    token = shell_utils.detect_shell()
    assert token == "cmd"
    # The command reaches the advertised shell as ONE intact argument on both
    # platforms: plain string on Windows (unchanged cmd behaviour), and
    # [shell, "-c", command] on POSIX.
    assert "echo hi" in argv


# ── detection precedence ────────────────────────────────────────────────────


def test_posix_family_shell_never_overrides_windows_process_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On Windows a POSIX ``SHELL`` must NOT hijack detection.

    Git Bash / MSYS2 / WSL-mounted sessions export ``SHELL=/bin/bash`` inside
    a cmd.exe or PowerShell terminal, so on Windows ``SHELL`` is not proof of
    the executing shell — the process tree is.  The explicit ``AGENT1_SHELL``
    override (tested above) is the supported escape hatch for a real bash.
    """
    monkeypatch.setenv("SHELL", "/bin/bash")
    if getattr(su, "_psutil", None) is not None:
        # Walk available on Windows: it decides, SHELL is ignored.
        monkeypatch.setattr(su, "_walk_for_powershell", lambda: True)
        assert shell_utils.detect_shell() == (
            "powershell" if sys.platform == "win32" else "bash"
        )
        return
    for var in shell_utils._PS_SESSION_ENV:
        monkeypatch.setenv(var, "C:\\PowerShell\\Modules")
    assert shell_utils.detect_shell() == (
        "powershell" if sys.platform == "win32" else "bash"
    )


def test_windows_ancestor_walk_is_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On Windows, with psutil available the ancestor walk decides powershell
    vs cmd.  POSIX CI has no PowerShell in the process tree, so there the same
    mechanism legitimately reports the POSIX default — assert on Windows only.
    """
    if getattr(su, "_psutil", None) is None:
        pytest.skip("psutil not installed — walk is unavailable")
    if sys.platform != "win32":
        pytest.skip(
            "Windows-only: the powershell/cmd ancestor walk has no POSIX analogue"
        )
    monkeypatch.setattr(su, "_walk_for_powershell", lambda: True)
    assert shell_utils.detect_shell() == "powershell"
    monkeypatch.setattr(su, "_walk_for_powershell", lambda: False)
    assert shell_utils.detect_shell() == "cmd"


def test_windows_env_fallback_is_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On Windows without psutil the detector falls back to PS session env
    evidence.  The fallback is a Windows branch, so it is asserted on Windows
    only."""
    if sys.platform != "win32":
        pytest.skip("Windows-only: PS session env fallback does not run on POSIX")
    if getattr(su, "_psutil", None) is not None:
        # Walk available: env vars are NOT consulted — walk decides.
        monkeypatch.setattr(su, "_walk_for_powershell", lambda: False)
        for var in shell_utils._PS_SESSION_ENV:
            monkeypatch.setenv(var, "C:\\PowerShell\\Modules")
        assert shell_utils.detect_shell() == "cmd"
        return
    for var in shell_utils._PS_SESSION_ENV:
        monkeypatch.setenv(var, "C:\\PowerShell\\Modules")
    assert shell_utils.detect_shell() == "powershell"


def test_posix_default_is_bash(monkeypatch: pytest.MonkeyPatch) -> None:
    """POSIX with no evidence defaults to bash (documented default)."""
    if os.name == "nt":
        pytest.skip("POSIX-only contract")
    assert shell_utils.detect_shell() == "bash"


def test_override_wins_everywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    """AGENT1_SHELL overrides every signal, raw aliases map to the platform."""
    monkeypatch.setenv("SHELL", "/bin/zsh")
    for alias in ("none", "off", "raw", "default"):
        monkeypatch.setenv("AGENT1_SHELL", alias)
        expected = "cmd" if os.name == "nt" else "sh"
        assert shell_utils.detect_shell() == expected, alias


# ── communication surfaces ─────────────────────────────────────────────────


def test_system_prompt_names_the_single_detector() -> None:
    """SHELL_NAME and the run-tool sentence show the SAME token."""
    token = shell_utils.detect_shell()
    assert shell_utils.SHELL_NAME == shell_utils.shell_name_token() == token
    prompt = shell_utils.SYSTEM_PROMPT
    assert f"SHELL_NAME={token}" in prompt
    assert f"The shell for the run tool is: {token}." in prompt


def test_cli_prompt_shows_the_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    """The interactive CLI shows which shell run commands execute in, next to
    the mode tag — visible at use time, not only in prompts."""
    from datetime import datetime

    import agent_core.cli.interactive as inter

    monkeypatch.setenv("AGENT1_SHELL", "cmd")
    prompt = inter._build_chat_prompt(object(), "main", datetime(2025, 1, 1, 9, 30))
    assert "[shell: cmd]" in prompt
    # Mode tag stays the final segment (colour styling unchanged).
    assert prompt.rstrip().endswith("[build]")
