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

    POSIX has no PATHEXT dance: execvp matches the bare name ``git``
    exactly, so the same interpreter copy must ALSO exist under the
    extension-less name (``shutil.copy`` carries the exec bit over).
    """
    git_exe = bin_dir / "git.exe"
    if not git_exe.exists():
        shutil.copy(sys.executable, git_exe)
    if os.name != "nt":
        git_bin = bin_dir / "git"
        if not git_bin.exists():
            shutil.copy(sys.executable, git_bin)
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


class TestNlpRunEncoding:
    """The NLP ``run`` tool spawns the shell via subprocess.Popen(text=True).

    That call site is the one the user actually hits (every command the
    agent runs goes through it), so it gets a behavioural test against the
    REAL ``Agent._nlp_run``, not just the source-level guard.
    """

    def test_utf8_child_output_not_mojibaked(self, tmp_path: Path) -> None:
        from agent import Agent

        # A child script that writes UTF-8 bytes straight to its stdout
        # buffer, so the bytes on the pipe are UTF-8 regardless of the
        # parent's locale.  A file (not ``python -c``) is used because the
        # command runs through the shell, where newlines/quotes would be
        # re-parsed by cmd.exe.
        script = tmp_path / "_utf8_child.py"
        script.write_text(
            "import sys\n"
            "sys.stdout.buffer.write('中'.encode('utf-8'))\n"
            "sys.stdout.buffer.flush()\n",
            encoding="utf-8",
        )
        bot = Agent(workspace=str(tmp_path))
        cmd = f'"{sys.executable}" "{script}"'
        out = asyncio.run(bot._nlp_run({"command": cmd}))

        assert "中" in out, f"UTF-8 output mangled by the locale decode: {out!r}"
        assert MOJIBAKE_MARK not in out

    def test_invalid_byte_does_not_crash_the_run_tool(self, tmp_path: Path) -> None:
        from agent import Agent

        script = tmp_path / "_badbyte_child.py"
        script.write_text(
            "import sys\n"
            f"sys.stdout.buffer.write({BAD_BYTES!r})\n"
            "sys.stdout.buffer.flush()\n",
            encoding="utf-8",
        )
        bot = Agent(workspace=str(tmp_path))
        cmd = f'"{sys.executable}" "{script}"'
        # Without the encoding pin, communicate() raises UnicodeDecodeError
        # on a cp1252 locale and _nlp_run returns "Error: ..." instead of the
        # command's real output.
        out = asyncio.run(bot._nlp_run({"command": cmd}))

        assert not out.startswith("Error:"), out
        assert "中" in out
        assert "[EXIT CODE" not in out


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

    #: Every subprocess entry point that accepts ``text=True`` and would
    #: therefore decode child output with the locale codepage when no
    #: explicit ``encoding=`` is given.  ``Popen`` matters most: the NLP
    #: ``run`` tool (``agent.py`` ``_nlp_run``) spawns the shell through it,
    #: so a missing pin there mojibakes/crashes every command the agent runs.
    ENTRY_POINTS = ("run", "Popen", "check_output", "call")

    @staticmethod
    def _subprocess_call_bodies(src: str) -> list[str]:
        """Text of every subprocess entry-point argument list (paren-matched)."""
        bodies: list[str] = []
        pattern = r"subprocess\.(?:" + "|".join(
            TestAllTextTrueCallsPinUtf8.ENTRY_POINTS
        ) + r")\("
        for m in re.finditer(pattern, src):
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
