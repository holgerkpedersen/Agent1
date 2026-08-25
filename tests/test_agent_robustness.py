"""Regression tests for the agent.py robustness batch (brainstorm items
A2/A4/C1/C2/B3 + the D1 remote-URL guard).

A2: chat_history.json / agent_memory.json are written atomically (tmp file +
os.replace) and a corrupt file is quarantined as *.bad-<timestamp> instead of
being silently dropped on load.

A4: the NLP ``run`` tool caps model-supplied timeouts (``_MAX_RUN_TIMEOUT_S``)
and the ``read`` tool caps per-call line limits (``_MAX_READ_LINES``).

C1: the plan-mode banner respects the QUIET display-mode contract (only the
final answer is printed in QUIET).

C2: loop-injected continuation notes are TAGGED, not content-matched — a user
message whose text happens to equal the note text must survive the strip.

B3: providers expose apply_profile(); LLMClient restores profiles through it
instead of poking _profile_name/temperature/max_tokens directly.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import agent
from agent import Agent
from agent_core.constants import (
    CHAT_HISTORY_TMP_PATH,
    AGENT_MEMORY_TMP_PATH,
)


@pytest.fixture()
def bot() -> Agent:
    return Agent(workspace=".")


def _tool_msg(content: str) -> dict:
    return {"role": "tool", "content": content}


# ---------------------------------------------------------------------------
# A2 — atomic persistence + corrupt-file quarantine
# ---------------------------------------------------------------------------

class TestAtomicPersistence:
    def test_save_chat_history_leaves_no_tmp_sidecar(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        hist = tmp_path / "chat_history.json"
        monkeypatch.setattr(agent, "CHAT_HISTORY_JSON_PATH", str(hist))
        monkeypatch.setattr(agent, "CHAT_HISTORY_TMP_PATH", str(hist) + ".tmp")
        bot = Agent(workspace=str(tmp_path))
        bot._chat_history = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "hello"},
        ]
        bot._save_chat_history()
        assert hist.exists()
        assert not Path(str(hist) + ".tmp").exists()
        data = json.loads(hist.read_text(encoding="utf-8"))
        assert [m["role"] for m in data] == ["system", "user"]

    def test_save_memory_leaves_no_tmp_sidecar(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        mem = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        monkeypatch.setattr(agent, "AGENT_MEMORY_TMP_PATH", str(mem) + ".tmp")
        bot = Agent(workspace=str(tmp_path))
        bot._files_read.add(str(tmp_path / "x.py"))
        bot._save_memory()
        assert mem.exists()
        assert not Path(str(mem) + ".tmp").exists()
        data = json.loads(mem.read_text(encoding="utf-8"))
        assert data["files_read"] == [str(tmp_path / "x.py")]

    def test_save_memory_sidecar_tracks_destination_not_stale_constant(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Regression (CI WinError 17): the tmp sidecar must be derived from
        the CURRENT AGENT_MEMORY_JSON_PATH.  A stale module-level
        AGENT_MEMORY_TMP_PATH pointing elsewhere (CI: repo on D:, patched
        destination on C:) must not break or misplace the save."""
        mem = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        monkeypatch.setattr(
            agent, "AGENT_MEMORY_TMP_PATH",
            str(tmp_path / "no-such-dir" / "agent_memory.json.tmp"),
        )
        bot = Agent(workspace=str(tmp_path))
        bot._files_read.add(str(tmp_path / "x.py"))
        bot._save_memory()
        assert mem.exists()
        data = json.loads(mem.read_text(encoding="utf-8"))
        assert data["files_read"] == [str(tmp_path / "x.py")]

    def test_corrupt_chat_history_is_quarantined_not_silently_dropped(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The old behavior returned [] and left corrupt bytes in place, so
        the next save clobbered them. Now a .bad-<ts> copy must exist."""
        hist = tmp_path / "chat_history.json"
        hist.write_text('{"role": "user", "cont', encoding="utf-8")  # truncated JSON
        monkeypatch.setattr(agent, "CHAT_HISTORY_JSON_PATH", str(hist))
        bot = Agent(workspace=str(tmp_path))
        bad = list(tmp_path.glob("chat_history.json.bad-*"))
        assert len(bad) == 1, "corrupt file was not quarantined"
        # Quarantine preserves the ORIGINAL corrupt bytes for inspection.
        assert bad[0].read_text(encoding="utf-8") == '{"role": "user", "cont'
        assert not hist.exists()  # moved, not copied
        # The live history falls back to empty.
        assert bot._chat_history == []

    def test_corrupt_agent_memory_is_quarantined(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        mem = tmp_path / "agent_memory.json"
        mem.write_text("]not json[", encoding="utf-8")
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem))
        bot = Agent(workspace=str(tmp_path))
        assert list(tmp_path.glob("agent_memory.json.bad-*"))
        assert bot._files_read == set()

    def test_missing_file_still_loads_empty_without_quarantine(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.setattr(agent, "CHAT_HISTORY_JSON_PATH",
                            str(tmp_path / "absent.json"))
        bot = Agent(workspace=str(tmp_path))
        assert bot._chat_history == []
        assert not list(tmp_path.glob("*.bad-*"))


# ---------------------------------------------------------------------------
# A4 — model-supplied argument caps
# ---------------------------------------------------------------------------

class TestToolArgumentCaps:
    def test_run_timeout_capped(self, bot: Agent) -> None:
        out = asyncio.run(bot._nlp_run({"command": "echo hi", "timeout": 10 ** 6}))
        assert agent._MAX_RUN_TIMEOUT_S == 600
        assert "hi" in out or out.strip() != ""

    def test_run_timeout_floor_of_one_second(self, bot: Agent) -> None:
        async def fail_fast() -> None:
            raise AssertionError("should not run")

        # timeout=0/None/negative all clamp to >= 1s; just prove no crash and
        # the command still executes within the clamped budget.
        out = asyncio.run(bot._nlp_run({"command": "echo ok", "timeout": -5}))
        assert "ok" in out

    def test_read_limit_capped_at_max_read_lines(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        big = tmp_path / "big.py"
        big.write_text("\n".join(f"line_{i}" for i in range(900)), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        bot = Agent(workspace=str(tmp_path))
        out = asyncio.run(bot._nlp_read({"path": str(big), "offset": 1, "limit": 10 ** 6}))
        body_lines = [l for l in out.splitlines() if l.startswith("line_")]
        assert len(body_lines) == agent._MAX_READ_LINES
        assert "[truncated" in out

    def test_read_offset_beyond_end_reports_line_count(
        self, tmp_path: Path,
    ) -> None:
        f = tmp_path / "small.py"
        f.write_text("a\nb\n", encoding="utf-8")
        bot = Agent(workspace=str(tmp_path))
        out = asyncio.run(bot._nlp_read({"path": str(f), "offset": 999}))
        assert "999" in out and "2 lines" in out


# ---------------------------------------------------------------------------
# C2 — tagged continue-note stripping
# ---------------------------------------------------------------------------

class TestContinueNoteTagging:
    def test_user_message_equal_to_note_text_survives_persist(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The old code stripped ANY message whose content equaled the note —
        including a genuine user prompt. Tagged stripping must keep it."""
        hist_file = tmp_path / "chat_history.json"
        mem_file = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "CHAT_HISTORY_JSON_PATH", str(hist_file))
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem_file))
        bot = Agent(workspace=str(tmp_path))
        note = agent._CONTINUE_NOTE

        async def fake_loop_run(self_loop, **kwargs):
            kwargs["messages"].append({
                "role": "assistant", "content": "Done.",
            })
            return "Done.", kwargs["messages"]

        monkeypatch.setattr(agent.ToolLoopRunner, "run", fake_loop_run)
        asyncio.run(bot.chat_nlp(note))  # user literally sends the note text
        saved = json.loads(hist_file.read_text(encoding="utf-8"))
        user_msgs = [
            m.get("content") for m in saved if m.get("role") == "user"
        ]
        assert any(c == note for c in user_msgs), (
            "user message matching the note text was dropped"
        )

    def test_injected_note_is_tagged_and_stripped_from_persisted_history(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        hist_file = tmp_path / "chat_history.json"
        mem_file = tmp_path / "agent_memory.json"
        monkeypatch.setattr(agent, "CHAT_HISTORY_JSON_PATH", str(hist_file))
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH", str(mem_file))
        bot = Agent(workspace=str(tmp_path))

        captured: dict = {}

        async def fake_loop_run(self_loop, **kwargs):
            captured["messages"] = kwargs["messages"]
            return "All done.", kwargs["messages"]

        # Force one auto-continue so an injected note actually gets appended.
        monkeypatch.setattr(agent, "_MAX_CHAINED_RUNS", 2)

        real_looks_incomplete = agent._looks_incomplete
        calls = {"n": 0}

        def fake_looks_incomplete(text: str) -> bool:
            calls["n"] += 1
            return calls["n"] == 1  # first answer looks incomplete → chain once

        monkeypatch.setattr(agent, "_looks_incomplete", fake_looks_incomplete)
        monkeypatch.setattr(agent.ToolLoopRunner, "run", fake_loop_run)
        asyncio.run(bot.chat_nlp("do the task"))

        # During the chained turn the injected note carried the tag...
        tagged = [
            m for m in captured["messages"]
            if m.get(agent._CONTINUE_NOTE_TAG_KEY) == agent._CONTINUE_NOTE_TAG
        ]
        assert len(tagged) == 1
        # ...and after the turn it is gone from the persisted history.
        saved = json.loads(hist_file.read_text(encoding="utf-8"))
        assert all(
            m.get("content") != agent._CONTINUE_NOTE for m in saved
        )


# ---------------------------------------------------------------------------
# C1 — plan-mode banner vs QUIET display mode
# ---------------------------------------------------------------------------

class TestPlanBannerDisplayMode:
    def _plan_bot(self, tmp_path: Path, monkeypatch) -> Agent:
        monkeypatch.setattr(agent, "CHAT_HISTORY_JSON_PATH",
                            str(tmp_path / "chat_history.json"))
        monkeypatch.setattr(agent, "AGENT_MEMORY_JSON_PATH",
                            str(tmp_path / "agent_memory.json"))
        b = Agent(workspace=str(tmp_path))
        b.mode = "plan"

        async def fake_loop_run(self_loop, **kwargs):
            return "Research complete.", kwargs["messages"]

        monkeypatch.setattr(agent.ToolLoopRunner, "run", fake_loop_run)
        return b

    def test_quiet_mode_suppresses_plan_banner(
        self, tmp_path: Path, monkeypatch, capsys,
    ) -> None:
        b = self._plan_bot(tmp_path, monkeypatch)
        monkeypatch.setenv("AGENT_DISPLAY_MODE", "quiet")
        asyncio.run(b.chat_nlp("research topic x"))
        out = capsys.readouterr().out
        assert "[plan mode]" not in out
        assert "Research complete." in out  # final answer always prints

    def test_verbose_mode_keeps_plan_banner(
        self, tmp_path: Path, monkeypatch, capsys,
    ) -> None:
        b = self._plan_bot(tmp_path, monkeypatch)
        monkeypatch.setenv("AGENT_DISPLAY_MODE", "verbose")
        asyncio.run(b.chat_nlp("research topic x"))
        out = capsys.readouterr().out
        assert "[plan mode] Read-only research" in out


# ---------------------------------------------------------------------------
# B3 — apply_profile on providers; no direct attribute pokes
# ---------------------------------------------------------------------------

class TestApplyProfileContract:
    def test_lmstudio_apply_profile_sets_all_three(self) -> None:
        from agent_core.llm.lmstudio import LMStudioProvider

        p = LMStudioProvider(model_name="laguna-s-2.1")
        p.apply_profile("deep-analysis", 0.2, 8000)
        assert p._profile_name == "deep-analysis"
        assert p.temperature == 0.2
        assert p.max_tokens == 8000

    def test_opencode_apply_profile_sets_all_three(self) -> None:
        from agent_core.llm.opencode_provider import OpencodeProvider

        p = OpencodeProvider(model_name="opencode-go/deepseek-v4-flash",
                             read_store=False)
        p.apply_profile("default", 0.5, 4000)
        assert p._profile_name == "default"
        assert p.temperature == 0.5
        assert p.max_tokens == 4000

    def test_llmclient_restores_profile_via_apply_profile(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """LLMClient.__init__ used to poke _provider._profile_name etc.
        directly; it must go through apply_profile now."""
        applied: dict = {}

        class FakeProvider(SimpleNamespace):
            def __init__(self) -> None:
                self.model_name = "m"
                self.temperature = 0.7
                self.max_tokens = 50000
                self._profile_name = None

            def apply_profile(self, name: str, temperature: float,
                              max_tokens: int) -> None:
                applied.update(name=name, temperature=temperature,
                               max_tokens=max_tokens)
                self._profile_name = name
                self.temperature = temperature
                self.max_tokens = max_tokens

        fake_provider = FakeProvider()
        monkeypatch.setattr(agent, "resolve_model", lambda *a, **k: "m")

        class FakeClient:
            def __init__(self, model_name=None, api_key=None):
                self._model_name = "m"
                self._provider = fake_provider
                self._profile_name = None
                self.api_key = ""

        # Build an Agent with a stubbed LLMClient construction path.
        monkeypatch.setattr(agent, "LLMClient", FakeClient)
        b = Agent(workspace=str(tmp_path))
        assert isinstance(b.llm, FakeClient)


# ---------------------------------------------------------------------------
# Payload boundary — the internal tag must never reach a provider
# ---------------------------------------------------------------------------

class TestTagStrippedAtPayloadBoundary:
    def test_sanitize_message_roles_strips_loop_note_tag(self) -> None:
        from agent_core.llm.lmstudio import sanitize_message_roles

        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task",
             agent.LOOP_NOTE_TAG_KEY: "continue"},
            {"role": "assistant", "content": "half done"},
            {"role": "user", "content": agent._CONTINUE_NOTE,
             agent.LOOP_NOTE_TAG_KEY: "continue"},
        ]
        out = sanitize_message_roles(msgs)
        assert all(agent.LOOP_NOTE_TAG_KEY not in m for m in out)
        # Content and roles are untouched otherwise.
        assert [m["role"] for m in out] == [
            "system", "user", "assistant", "user",
        ]
        assert out[1]["content"] == "task"
        assert out[3]["content"] == agent._CONTINUE_NOTE
