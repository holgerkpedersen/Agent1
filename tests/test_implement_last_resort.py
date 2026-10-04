"""Regression: the last-resort direct-generation fallback must be reachable.

Defect
------
In ``ImplementCommand.execute`` the self-correction retry block sits inside
the ``for filename, patch_text in patch_matches`` loop::

    ok, patched = apply_patch(...)          # patch fails -> ok is False
    if not ok:                              # self-correction (Options A-D)
        ...
        if not ok:
            print("... retry produced no usable content")
            continue                        # <-- jumps past the fallback
        except Exception:
            print("... retry failed")
            continue                        # <-- jumps past the fallback
    # Last resort: generate the file directly with a focused prompt
    if not ok:                              # unreachable: ok is True here
        print("  Attempting direct generation ...")

Both ways of leaving the retry block with ``ok`` still False are a
``continue``, so every path that reaches the last-resort guard has ``ok is
True``.  The fallback can never execute -- mypy flags the block
``[unreachable]``.  The ``if not ok: continue`` that sits *after* the block
only makes sense if the block is meant to run.

Effect: when a generated patch fails to apply and all four self-correction
options come back unusable, the file is skipped with a warning instead of
being regenerated directly -- exactly the recovery path the comment promises.

The test drives the REAL ``ImplementCommand.execute`` with a stub agent that
returns (1) a patch that cannot apply, (2) a retry response with no usable
block, (3) a valid full-file block for the direct generation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_core.commands.implement_cmd import ImplementCommand

BROKEN_PATCH = (
    "[PATCH: pkg/mod_me.py]\n"
    "@@ -1,2 +1,2 @@\n"
    "-def nowhere():\n"
    "-    pass\n"
    "+def nowhere():\n"
    "+    return 2\n"
)

DIRECT_FILE = (
    "[FILE: pkg/mod_me.py]\n"
    "```python\n"
    "def foo():\n"
    "    return 42\n"
    "```"
)


def _make_workspace(tmp_path: Path) -> Path:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mod_me.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (tmp_path / "tasks.md").write_text(
        "1. `pkg/mod_me.py` — extend foo\n", encoding="utf-8"
    )
    return pkg / "mod_me.py"


def _run(tmp_path: Path, responses: list[str]) -> tuple[bool, str, int]:
    """Run ``implement`` against a stub agent returning *responses* in order.

    Returns (command result, captured stdout, number of chat calls).
    """
    target = _make_workspace(tmp_path)
    assert target.exists()

    calls: list[object] = []

    async def chat(messages, **kwargs):
        index = len(calls)
        calls.append(messages)
        if index < len(responses):
            return responses[index]
        # Anything past the scripted exchange gets a usable full file so the
        # run can finish normally.
        return DIRECT_FILE

    agent = SimpleNamespace(workspace=str(tmp_path), llm=SimpleNamespace(chat=chat))
    with patch(
        "agent_core.commands.implement_cmd.auto_choice", return_value="y"
    ):
        # --modify selects modify mode, without which a [PATCH:] block is
        # never parsed at all (see `if modify_mode and not matches:`).
        result = asyncio.run(
            ImplementCommand().execute(
                [str(tmp_path / "tasks.md"), "--modify"], agent
            )
        )
    return result, "", len(calls)


def test_failed_self_correction_reaches_direct_generation(tmp_path, capsys):
    """A patch plus an unusable retry must fall through to direct generation."""
    result, _, calls = _run(
        tmp_path,
        [
            BROKEN_PATCH,       # generated patch cannot apply -> ok is False
            "I am unable to produce a corrected patch.",  # no usable block
        ],
    )
    out = capsys.readouterr().out

    assert "retry produced no usable content" in out, (
        "expected the self-correction retry to run at all:\n" + out
    )
    assert "Attempting direct generation for pkg/mod_me.py" in out, (
        "the last-resort direct-generation fallback never ran, so a failed "
        "patch silently skips the file instead of regenerating it:\n" + out
    )
    assert calls >= 3, f"expected a third chat call for direct generation, got {calls}"


def test_direct_generation_recovers_the_file(tmp_path, capsys):
    """After the fallback runs, the file is actually regenerated."""
    _run(
        tmp_path,
        [
            BROKEN_PATCH,
            "I am unable to produce a corrected patch.",
        ],
    )
    out = capsys.readouterr().out

    assert "Generated (patch): pkg/mod_me.py" in out, (
        "direct generation did not recover the file:\n" + out
    )
    written = (tmp_path / "pkg" / "mod_me.py").read_text(encoding="utf-8")
    assert "return 42" in written, (
        "the recovered file still holds the original content:\n" + written
    )
