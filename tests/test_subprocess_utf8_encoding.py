"""Regression tests: subprocess.run(text=True) must always pin an encoding.

History: the git REPL command (and sibling git/mypy/py_compile child calls
in agent.py, fix_cmd, proposal_core, implement_cmd, harnessfix/gates and
the scripts) used ``text=True`` without an explicit encoding, so on
Windows child output was decoded with the locale codepage (cp1252).
git always emits UTF-8: CJK/emoji in commit messages or filenames came
back as mojibake ("中" -> "Ã¤Â¸\xad"), and any byte the codepage cannot
map (e.g. 0x81) crashed the command handler with a hard
``UnicodeDecodeError``.  Every such call now passes
``encoding="utf-8", errors="replace"`` — the pattern
``agent_core/llm/lmstudio.py`` already used for the lms CLI.

The behavioural tests use a fake ``git`` on PATH that writes raw bytes,
so they exercise the REAL decode path of each fixed function.  The
source-level guard at the end is deterministic on every machine (including
UTF-8-locale ones where the behavioural tests would pass even unfixed).
"""
from __future__ import annotations

import asyncio
import locale
import os
import re
import shutil
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _locale_invalid_byte() -> int | None:
    """A byte the machine's preferred encoding cannot decode, if any.

    cp1252 (the classic Windows locale) rejects 0x81; latin-1 accepts all
    256 bytes, in which case there is no crash byte and the mojibake
    assertion alone guards the test.
    """
    enc = locale.getpreferredencoding(False)
    for b in range(1, 256):
        try:
            bytes([b]).decode(enc)
        except (UnicodeDecodeError, LookupError):
            return b
    return None


#: "中" as UTF-8 (what git emits), plus a byte the local codepage rejects
#: so an un-pinned decode RAISES instead of silently mojibake-ing.
_BAD_BYTE = _locale_invalid_byte()
BAD_BYTES = "中".encode("utf-8") + (bytes([_BAD_BYTE]) if _BAD_BYTE is not None else b"")

#: cp1252 decoding of the first UTF-8 byte of "中" — the mojibake marker.
MOJIBAKE_MARK = "Ã¤"


def _write_git_shim(bin_dir: Path, work_dir: Path, subcommand: str) -> None:
    """Install a fake ``git`` that emits BAD_BYTES when run as ``git <subcommand>``.

    Windows CreateProcess resolves a bare ``git`` by trying every PATHEXT
    extension against the whole PATH, and a real ``git.exe`` (e.g. from
    Git-for-Windows) beats a ``git.bat``/``git.cmd`` shim in an earlier
    PATH directory.  So the fake ``git`` is a copy of the Python
    interpreter: ``git.exe <subcommand>`` makes Python execute a script
    file named ``<subcommand>`` from the working directory, which writes
    the raw bytes to stdout.
    """
    git_exe = bin_dir / "git.exe"
    if not git_exe.exists():
        shutil.copy(sys.executable, git_exe)
    work_dir.joinpath(subcommand).write_text(
        "import sys\n"
        f"sys.stdout.buffer.write({BAD_BYTES!r})\n"
        "sys.stdout.buffer.flush()\n",
        encoding="utf-8",
    )


@pytest.fixture()
def fake_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake ``git`` on PATH (use with a work dir that gets a subcommand script)."""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    monkeypatch.setenv(
        "PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    )
    return bin_dir


class TestGitCommandEncoding:
    def test_non_ascii_output_decoded_as_utf8(
        self, fake_git: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agent_core.commands.git_cmd import GitCommand

        workspace = tmp_path / "ws"
        workspace.mkdir()
        _write_git_shim(fake_git, workspace, "status")

        agent = types.SimpleNamespace(workspace=str(workspace))
        # Without the fix this RAISES UnicodeDecodeError on cp1252 (0x81)
        # or prints mojibake on other codepages.
        ok = asyncio.run(GitCommand().execute(["status"], agent))
        out = capsys.readouterr().out
        assert ok is True
        assert "中" in out
        assert MOJIBAKE_MARK not in out


class TestRunCappedEncoding:
    def test_invalid_bytes_do_not_crash(self) -> None:
        from agent_core.commands.fix_cmd import _run_capped

        code = (
            "import sys\n"
            f"sys.stdout.buffer.write({BAD_BYTES!r})\n"
            "sys.stdout.buffer.flush()\n"
        )
        # _run_capped only catches TimeoutExpired — without the fix,
        # UnicodeDecodeError propagates out of subprocess.run.
        r = _run_capped([sys.executable, "-c", code], timeout_s=30)
        assert r.returncode == 0
        assert "中" in r.stdout


class TestProposalGitEncoding:
    def test_non_ascii_output_returned_not_error(
        self, fake_git: Path, tmp_path: Path
    ) -> None:
        from agent_core.commands.proposal_core import _git

        work = tmp_path / "ws"
        work.mkdir()
        _write_git_shim(fake_git, work, "status")

        # Without the fix the broad except swallows the decode crash and
        # returns (1, "git error: ...") instead of the git output.
        rc, out = _git(["status"], cwd=str(work))
        assert rc == 0
        assert "中" in out
        assert "git error" not in out


class TestAllTextTrueCallsPinUtf8:
    """Source-level guard: deterministic on every machine.

    Behavioural tests above cannot fail on a UTF-8-locale machine even when
    a call site regresses (the locale decode happens to match git's UTF-8).
    This guard pins the invariant at the source level instead.
    """

    SKIP_DIRS = {
        ".git", ".venv", "venv", "node_modules", "backups", "reports",
        "tests", "benchmark", "dashboard", ".docs", "skills", "harnessfix_selftest",
    }

    @staticmethod
    def _subprocess_call_bodies(src: str) -> list[str]:
        """Text of every subprocess.run(...) argument list (paren-matched)."""
        bodies: list[str] = []
        for m in re.finditer(r"subprocess\.run\(", src):
            depth = 1
            i = m.end()
            while i < len(src) and depth:
                if src[i] == "(":
                    depth += 1
                elif src[i] == ")":
                    depth -= 1
                i += 1
            bodies.append(src[m.end() : i - 1])
        return bodies

    def test_text_true_calls_always_pass_encoding(self) -> None:
        offenders: list[str] = []
        for p in sorted(REPO_ROOT.rglob("*.py")):
            if any(part in self.SKIP_DIRS for part in p.parts):
                continue
            src = p.read_text(encoding="utf-8", errors="replace")
            for body in self._subprocess_call_bodies(src):
                if "text=True" in body and "encoding=" not in body:
                    snippet = " ".join(body.split())[:80]
                    offenders.append(
                        f"{p.relative_to(REPO_ROOT)}: {snippet!r}"
                    )
        assert not offenders, (
            "subprocess.run(text=True) without an explicit encoding — on "
            "Windows this decodes child output with the locale codepage and "
            "crashes/mojibakes on git's UTF-8 output.  Add "
            'encoding="utf-8", errors="replace".\n' + "\n".join(offenders)
        )
