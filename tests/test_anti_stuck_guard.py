"""Regression tests: long-term anti-stuck strategy (2026-08-25 stall post-mortem).

Two layers are covered here (the third, subprocess caps, lives in
test_fix_subprocess_guard.py):

Layer 1 — engine stall cap: LM Studio chat requests no longer hard-wire a
3600s socket timeout; the cap defaults to 600s and is env-overridable.  A
stalled engine (observed live as a silent 12-minute freeze at ~310 tokens)
now surfaces as a timeout error instead of holding the turn hostage.

Layer 2 — read-loop guard: traces a01f1bde/39a90f8f show dozens of
consecutive ``read`` calls ballooning the prompt until prefill crawls and
the turn dies as stuck/no_progress.  After _MAX_CONSECUTIVE_READS reads in
one turn, results carry a steering note; any non-read tool resets the count;
a new turn starts from zero.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import agent
from agent import Agent
from agent_core.llm import lmstudio


@pytest.fixture()
def bot(tmp_path: Path) -> Agent:
    return Agent(workspace=str(tmp_path))


class TestEngineStallCap:
    def test_default_is_600_not_3600(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LMSTUDIO_CHAT_TIMEOUT", raising=False)
        assert lmstudio.chat_timeout() == 600

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LMSTUDIO_CHAT_TIMEOUT", "120")
        assert lmstudio.chat_timeout() == 120

    def test_garbage_env_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LMSTUDIO_CHAT_TIMEOUT", "abc")
        assert lmstudio.chat_timeout() == 600

    def test_floor_at_30s(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LMSTUDIO_CHAT_TIMEOUT", "5")
        assert lmstudio.chat_timeout() == 30

    def test_no_hardwired_3600_left(self) -> None:
        src = Path(lmstudio.__file__).read_text(encoding="utf-8")
        assert "timeout=3600" not in src, (
            "a chat request still bypasses the configurable stall cap"
        )


class TestReadLoopGuard:
    def test_reads_below_threshold_are_unmodified(self, bot: Agent, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("x = 1\n", encoding="utf-8")
        for _ in range(agent._MAX_CONSECUTIVE_READS - 1):
            out = asyncio.run(bot._nlp_read({"path": str(f)}))
            assert "read-loop guard" not in out

    def test_note_appended_at_threshold(self, bot: Agent, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("x = 1\n", encoding="utf-8")
        out = ""
        for i in range(agent._MAX_CONSECUTIVE_READS):
            out = asyncio.run(bot._nlp_read({"path": str(f)}))
        assert "read-loop guard" in out
        assert f"{agent._MAX_CONSECUTIVE_READS} consecutive reads" in out

    def test_non_read_tool_resets_streak(self, bot: Agent, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("x = 1\n", encoding="utf-8")
        for _ in range(agent._MAX_CONSECUTIVE_READS - 1):
            asyncio.run(bot._nlp_read({"path": str(f)}))
        asyncio.run(bot._execute_tool_call(
            "list_files", {"path": "."},
        ))
        assert bot._read_streak == 0
        out = asyncio.run(bot._nlp_read({"path": str(f)}))
        assert "read-loop guard" not in out

    def test_failed_reads_do_not_count(self, bot: Agent, tmp_path: Path) -> None:
        for _ in range(agent._MAX_CONSECUTIVE_READS):
            out = asyncio.run(bot._nlp_read({"path": str(tmp_path / "nope.py")}))
            assert out.startswith("File not found")
        assert bot._read_streak == 0

    def test_new_turn_starts_from_zero(self, bot: Agent, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("x = 1\n", encoding="utf-8")
        for _ in range(agent._MAX_CONSECUTIVE_READS + 2):
            asyncio.run(bot._nlp_read({"path": str(f)}))
        assert bot._read_streak >= agent._MAX_CONSECUTIVE_READS
        bot._chat_history.append({"role": "user", "content": "next"})
        bot._turn_start_index = len(bot._chat_history)
        bot._read_streak = 0  # what chat_nlp does at turn start
        out = asyncio.run(bot._nlp_read({"path": str(f)}))
        assert "read-loop guard" not in out

    def test_truncated_read_carries_note_too(self, bot: Agent, tmp_path: Path) -> None:
        f = tmp_path / "big.py"
        f.write_text("\n".join(f"line {i}" for i in range(100)), encoding="utf-8")
        out = ""
        for _ in range(agent._MAX_CONSECUTIVE_READS):
            out = asyncio.run(bot._nlp_read({"path": str(f), "limit": 10}))
        assert "[truncated" in out and "read-loop guard" in out
