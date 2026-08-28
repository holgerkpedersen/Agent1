"""Regression test: the on-demand `fix --desc` path must cap its LLM context.

Before the fix, a single best-effort-except issue pulled full source of the
top-5 files (~575KB for agent.py + friends) into ONE non-streaming POST. On a
local 27B model the prefill couldn't finish inside the socket-timeout window,
so the call timed out and RetryPolicy resent the same giant prompt 4x (~40min).

The on-demand path now caps per-file and total context size: large files get a
focused window (centered on the issue line) and invite [READ:] for the rest.
"""

from __future__ import annotations
import asyncio
import os
from pathlib import Path

from agent_core.commands.fix_cmd import FixCommand


class _FakeLLM:
    """Captures the user prompt sent to the LLM and returns a plain reply so
    the on-demand round loop ends after the first call (no file writes)."""

    def __init__(self) -> None:
        self.captured: list[str] = []

    async def chat(self, messages, *args, **kwargs):
        user = next((m for m in messages if m.get("role") == "user"), None)
        if user is not None:
            self.captured.append(user["content"])
        return "No fix produced in this simulated run."


class _FakeAgent:
    def __init__(self) -> None:
        self.llm = _FakeLLM()


def _write_huge_file(path: Path, lines: int, keyword: str) -> None:
    body = "\n".join(
        f"def func_{i}():  # {keyword}\n    raise {keyword.title()}() if False else None\n"
        for i in range(lines)
    )
    path.write_text(body, encoding="utf-8")


def test_ondemand_context_capped_for_large_files(tmp_path: Path) -> None:
    big = tmp_path / "bigmod.py"
    _write_huge_file(big, 4000, "exception")  # far larger than the per-file cap
    (tmp_path / "helper.py").write_text("def helper():\n    return 1\n", encoding="utf-8")

    agent = _FakeAgent()
    cmd = FixCommand()
    desc = "fix best-effort-except in bigmod.py:200 — wrap in context manager"
    asyncio.run(cmd.execute([str(big), "--desc", desc], agent))

    assert agent.llm.captured, "fix command never called the LLM"
    prompt = agent.llm.captured[0]

    budget = int(os.environ.get("FIX_CONTEXT_MAX_CHARS", "50000"))
    # Allow slack for the other project files' signatures and the issue text.
    assert len(prompt) <= budget + 20000, (
        f"on-demand prompt {len(prompt)} bytes exceeds budget {budget} (+slack)"
    )
    # The keyword must still be present (the window covers the issue line)...
    assert "exception" in prompt.lower()
    # ...and the model must be told it can pull the full file on demand,
    # proving the huge file was NOT inlined wholesale.
    assert "request full with [READ:" in prompt


def test_ondemand_small_file_still_inlined(tmp_path: Path) -> None:
    small = tmp_path / "smallmod.py"
    small.write_text(
        "def do_work():\n    print('start')\n    # best-effort\n    try:\n        risky()\n"
        "    except Exception:\n        log()\n",
        encoding="utf-8",
    )
    agent = _FakeAgent()
    cmd = FixCommand()
    desc = "wrap the try body in smallmod.py:5 in a context manager"
    asyncio.run(cmd.execute([str(small), "--desc", desc], agent))

    prompt = agent.llm.captured[0]
    # Small file fits the per-file cap, so its full source is inlined.
    assert "def do_work():" in prompt
    assert "request full with [READ:" not in prompt
