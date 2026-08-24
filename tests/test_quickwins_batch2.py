"""Regression tests for quick wins #6 (self-review note) and #19
(uncommitted-changes reminder at REPL shutdown) — improvement plan batch 2.

#6: after a chat turn whose write/edit tool results carry a py_compile
verification line, ``chat_nlp`` prints one ``[self-review]`` nudge listing
the changed files (relative paths).  Extraction goes through the REAL tool
handlers' output format.

#19: every REPL shutdown path calls ``_warn_uncommitted``, which prints an
invariant-#4 reminder when ``git status --porcelain`` is non-empty, stays
silent on a clean repo, and never raises outside a git repo.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path

import pytest

import agent
from agent import Agent


@pytest.fixture()
def bot() -> Agent:
    return Agent(workspace=".")


def _tool_msg(content: str) -> dict:
    return {"role": "tool", "content": content}


# ---------------------------------------------------------------------------
# #6 — mutated-file extraction + self-review note
# ---------------------------------------------------------------------------

class TestMutatingFilesExtraction:
    def test_extracts_from_real_write_and_edit_output(self, bot: Agent) -> None:
        """Drive the REAL write/edit handlers, then extract."""
        target = Path(bot.workspace) / "tmp_qw6_probe.py"
        out_write = asyncio.run(bot._execute_tool_call(
            "write", {"path": str(target), "content": "x = 1\n"},
        ))
        out_edit = asyncio.run(bot._execute_tool_call(
            "edit",
            {"path": str(target), "old_text": "x = 1", "new_text": "x = 2"},
        ))
        assert "[verify] py_compile" in out_write
        assert "[verify] py_compile" in out_edit
        bot._chat_history.append(_tool_msg(out_write))
        bot._chat_history.append(_tool_msg(out_edit))
        files = bot._mutating_files_this_turn()
        assert len(files) == 1
        assert os.path.normcase(str(target)) == os.path.normcase(files[0])
        target.unlink()

    def test_skipped_no_change_message_is_not_counted(self, bot: Agent) -> None:
        target = Path(bot.workspace) / "tmp_qw6_same.py"
        target.write_text("x = 1\n", encoding="utf-8")
        out_skip = asyncio.run(bot._execute_tool_call(
            "write", {"path": str(target), "content": "x = 1\n"},
        ))
        assert out_skip.startswith("Skipped")
        bot._chat_history.append(_tool_msg(out_skip))
        assert bot._mutating_files_this_turn() == []
        target.unlink()

    def test_plain_tool_results_are_ignored(self, bot: Agent) -> None:
        bot._chat_history.append(_tool_msg("read result with no marker"))
        bot._chat_history.append(
            _tool_msg("Written fake.py (10 bytes)")  # no [verify] line
        )
        assert bot._mutating_files_this_turn() == []

    def test_deduplicates_repeated_writes_to_same_file(self, bot: Agent) -> None:
        msg = _tool_msg("Written C:\\x\\a.py (5 bytes)\n[verify] py_compile ✓")
        bot._chat_history.extend([msg, msg])
        assert len(bot._mutating_files_this_turn()) == 1


class TestSelfReviewNote:
    def test_note_printed_after_mutating_turn(self, bot: Agent, capsys) -> None:
        bot._print_self_review_note(["c:/ws/a.py", "c:/ws/b.py"])
        out = capsys.readouterr().out
        assert "[self-review]" in out
        assert "2 file(s)" in out
        assert "py_compile verified syntax only" in out

    def test_paths_shown_relative_to_workspace(self, bot: Agent, capsys) -> None:
        ws = bot.workspace
        bot._print_self_review_note([os.path.join(ws, "deep", "mod.py")])
        out = capsys.readouterr().out
        assert "deep/mod.py" in out or "deep\\mod.py" in out

    def test_more_than_six_files_gets_overflow_marker(
        self, bot: Agent, capsys,
    ) -> None:
        bot._print_self_review_note([f"f{i}.py" for i in range(8)])
        out = capsys.readouterr().out
        assert "(+2 more)" in out

    def test_chat_nlp_prints_note_only_when_files_changed(
        self, bot: Agent, tmp_path: Path, monkeypatch, capsys,
    ) -> None:
        """End-to-end through chat_nlp with the REAL write handler."""
        hist_file = tmp_path / "chat_history.json"
        mem_file = tmp_path / "agent_memory.json"
        hist_file.write_text("[]", encoding="utf-8")
        mem_file.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(agent, "CHAT_HISTORY_JSON_PATH", str(hist_file))
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem_file))
        fresh = Agent(workspace=".")
        # Point the agent's workspace at tmp so writes stay isolated.
        fresh.workspace = str(tmp_path)

        async def fake_loop_run(self_loop, **kwargs):
            # self_loop is the ToolLoopRunner; the agent is closed over here.
            # Mirror the REAL loop message shape: assistant(tool_calls) +
            # matching tool result — otherwise _drop_orphan_tool_messages
            # would legitimately drop the bare tool message.
            target = Path(fresh.workspace) / "qw6_e2e.py"
            result = await fresh._execute_tool_call(
                "write", {"path": str(target), "content": "y = 1\n"},
            )
            kwargs["messages"].append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_qw6",
                    "type": "function",
                    "function": {"name": "write", "arguments": "{}"},
                }],
            })
            kwargs["messages"].append({
                "role": "tool",
                "tool_call_id": "call_qw6",
                "content": result,
            })
            return f"Wrote {target.name}.", kwargs["messages"]

        monkeypatch.setattr(agent.ToolLoopRunner, "run", fake_loop_run)
        asyncio.run(fresh.chat_nlp("write the file"))
        out = capsys.readouterr().out
        assert "[self-review]" in out
        assert "qw6_e2e.py" in out


class TestTurnBoundaryAfterRestart:
    """Regression: _mutating_files_this_turn scanned the WHOLE restored
    history, so after a session restart the self-review note listed files
    written by PREVIOUS sessions (observed 2026-08-25: a fresh Agent in the
    real repo reported 7 stale tmp_* writes from an earlier session).  The
    turn boundary (_turn_start_index, set at the top of chat_nlp) must fence
    per-turn scans off from restored tool results."""

    def test_restored_history_tool_results_are_ignored(self) -> None:
        bot = Agent(workspace=".")
        # Simulate a session that RESTORED previous turns' tool results.
        bot._chat_history.extend([
            {"role": "user", "content": "old task"},
            _tool_msg("Written old_session.py (5 bytes)\n[verify] py_compile ✓"),
        ])
        bot._turn_start_index = len(bot._chat_history)
        assert bot._mutating_files_this_turn() == []

    def test_current_turn_writes_still_detected_after_restart(self) -> None:
        bot = Agent(workspace=".")
        bot._chat_history.append(
            _tool_msg("Written old_session.py (5 bytes)\n[verify] py_compile ✓")
        )
        bot._turn_start_index = len(bot._chat_history)
        bot._chat_history.append(_tool_msg("user asks something new"))
        bot._chat_history.append(
            _tool_msg("Written this_turn.py (7 bytes)\n[verify] py_compile ✓")
        )
        files = bot._mutating_files_this_turn()
        assert len(files) == 1
        assert files[0].endswith("this_turn.py")

    def test_end_to_end_restart_then_write_prints_only_new_file(
        self, tmp_path: Path, monkeypatch, capsys,
    ) -> None:
        """Full path: persisted history with an old write -> restart ->
        one new write -> note lists ONLY the new file."""
        hist_file = tmp_path / "chat_history.json"
        mem_file = tmp_path / "agent_memory.json"
        stale = (
            "Written C:\\somewhere\\else\\stale.py (5 bytes)\n"
            "[verify] py_compile ✓"
        )
        hist_file.write_text(
            json.dumps([
                {"role": "system", "content": "s"},
                {"role": "user", "content": "previous session prompt"},
                {"role": "tool", "tool_call_id": "x", "content": stale},
            ]),
            encoding="utf-8",
        )
        mem_file.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(agent, "CHAT_HISTORY_JSON_PATH", str(hist_file))
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem_file))
        restarted = Agent(workspace=str(tmp_path))

        async def fake_loop_run(self_loop, **kwargs):
            target = Path(restarted.workspace) / "new_turn.py"
            result = await restarted._execute_tool_call(
                "write", {"path": str(target), "content": "z = 1\n"},
            )
            kwargs["messages"].append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_new",
                    "type": "function",
                    "function": {"name": "write", "arguments": "{}"},
                }],
            })
            kwargs["messages"].append({
                "role": "tool",
                "tool_call_id": "call_new",
                "content": result,
            })
            return f"Wrote {target.name}.", kwargs["messages"]

        monkeypatch.setattr(agent.ToolLoopRunner, "run", fake_loop_run)
        asyncio.run(restarted.chat_nlp("write again"))
        out = capsys.readouterr().out
        assert "new_turn.py" in out
        assert "stale.py" not in out


# ---------------------------------------------------------------------------
# #19 — uncommitted-changes reminder at shutdown
# ---------------------------------------------------------------------------

class TestWarnUncommitted:
    def test_clean_repo_is_silent(self, tmp_path: Path, capsys) -> None:
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        bot = Agent(workspace=str(tmp_path))
        agent._warn_uncommitted(bot)
        assert capsys.readouterr().out == ""

    def test_non_repo_is_silent_and_never_raises(self, tmp_path: Path, capsys) -> None:
        bot = Agent(workspace=str(tmp_path))
        # Must not raise even though git status fails here.
        agent._warn_uncommitted(bot)
        assert capsys.readouterr().out == ""

    def test_dirty_repo_prints_invariant_reminder(
        self, tmp_path: Path, capsys,
    ) -> None:
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        (tmp_path / "tracked.txt").write_text("v1", encoding="utf-8")
        subprocess.run(
            ["git", "add", "tracked.txt"], cwd=tmp_path, check=True,
        )
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t",
             "commit", "-qm", "init"],
            cwd=tmp_path, check=True,
        )
        (tmp_path / "tracked.txt").write_text("v2", encoding="utf-8")
        (tmp_path / "untracked.txt").write_text("new", encoding="utf-8")

        bot = Agent(workspace=str(tmp_path))
        agent._warn_uncommitted(bot)
        out = capsys.readouterr().out
        assert "uncommitted change(s)" in out
        assert "invariant #4" in out
        assert "tracked.txt" in out

    def test_lists_at_most_five_then_overflow(self, tmp_path: Path, capsys) -> None:
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        for i in range(9):
            (tmp_path / f"f{i}.txt").write_text("x", encoding="utf-8")
        bot = Agent(workspace=str(tmp_path))
        agent._warn_uncommitted(bot)
        out = capsys.readouterr().out
        assert "and 4 more" in out
