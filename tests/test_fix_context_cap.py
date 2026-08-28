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


def test_ondemand_read_upgrades_windowed_file(tmp_path: Path) -> None:
    """Regression (2026-08-28): a top file larger than FIX_FILE_MAX_CHARS is
    sent as a focused WINDOW, not full source. The model must be able to upgrade
    it to full source via [READ:] so it can actually make the edit. Previously
    the READ handler skipped already-"included" top files, so the model never
    saw the whole file and the loop ended with no fix.
    """
    big = tmp_path / "bigmod.py"
    lines = [f"def f{i}(): return {i}  # except" for i in range(4000)]
    lines[3000] = "# UNIQUE_DEEP_MARKER_LINE"
    big.write_text("\n".join(lines), encoding="utf-8")
    (tmp_path / "helper.py").write_text("def helper():\n    return 1\n", encoding="utf-8")

    class _ReadThenFixLLM:
        def __init__(self) -> None:
            self.captured: list[str] = []
            self.round = 0

        async def chat(self, messages, *args, **kwargs):
            user = next((m for m in messages if m.get("role") == "user"), None)
            if user is not None:
                self.captured.append(user["content"])
            self.round += 1
            if self.round == 1:
                return "[READ: bigmod.py]"
            return ("[FILE: bigmod.py]\n```python\n"
                    "def f0(): return 0\n# UNIQUE_DEEP_MARKER_LINE\n"
                    "```")

    agent = _FakeAgent()
    agent.llm = _ReadThenFixLLM()
    cmd = FixCommand()
    desc = "fix best-effort-except in bigmod.py:200 — wrap in context manager"
    asyncio.run(cmd.execute([str(big), "--desc", desc], agent))

    # Round-2 context must contain the FULL file (the deep marker only appears
    # once the windowed file was upgraded to full source via [READ:]).
    assert any("UNIQUE_DEEP_MARKER_LINE" in p for p in agent.llm.captured), (
        "windowed file was not upgraded to full source on [READ:]"
    )
    # And the fix should have been applied to disk.
    assert "def f0(): return 0" in big.read_text(encoding="utf-8")


def test_ondemand_read_bounded_by_ceiling(tmp_path: Path, monkeypatch) -> None:
    """Regression (2026-08-28): a [READ:] of a large module must NOT re-bloat the
    prompt past the read ceiling. The Fix A change made [READ:] load full source,
    which reintroduced the 500KB+ timeout when the LLM read several big modules.
    Now a [READ:] only loads the full file when it fits under the ceiling;
    otherwise a focused window is returned (the deep marker must not leak in).
    """
    monkeypatch.setenv("FIX_CONTEXT_READ_CEILING", "40000")
    big = tmp_path / "bigmod.py"
    lines = [f"def f{i}(): return {i}  # except" for i in range(4000)]
    lines[3000] = "# UNIQUE_DEEP_MARKER_LINE"
    big.write_text("\n".join(lines), encoding="utf-8")
    (tmp_path / "helper.py").write_text("def helper():\n    return 1\n", encoding="utf-8")

    class _ReadThenFixLLM:
        def __init__(self) -> None:
            self.captured: list[str] = []
            self.round = 0

        async def chat(self, messages, *args, **kwargs):
            user = next((m for m in messages if m.get("role") == "user"), None)
            if user is not None:
                self.captured.append(user["content"])
            self.round += 1
            if self.round == 1:
                return "[READ: bigmod.py]"
            return "[FILE: bigmod.py]\n```python\nimport os\n\ndef f0(): return 0\n```"

    agent = _FakeAgent()
    agent.llm = _ReadThenFixLLM()
    cmd = FixCommand()
    desc = "fix best-effort-except in bigmod.py:200 — wrap in context manager"
    asyncio.run(cmd.execute([str(big), "--desc", desc], agent))

    # A windowed [READ:] must not inline the FULL file (deep marker stays out).
    assert any("focused window" in p for p in agent.llm.captured), (
        "expected a bounded window on [READ:], not full source"
    )
    assert not any("UNIQUE_DEEP_MARKER_LINE" in p for p in agent.llm.captured), (
        "large [READ:] was not bounded by the ceiling — full source leaked in"
    )
    # No single prompt may approach the size that timed out the local model.
    assert max(len(p) for p in agent.llm.captured) < 100000


def test_ondemand_strips_shell_tool_calls(tmp_path: Path) -> None:
    """Regression (2026-08-28): the model sometimes emits <tool_call>shell ...>
    invocations (which the fix path cannot execute). These must be stripped so
    they don't pollute the parse, and a [FILE:] fix in the same response still
    applies.
    """
    small = tmp_path / "smallmod.py"
    small.write_text("def do_work():\n    print('x')\n", encoding="utf-8")

    class _ToolCallLLM:
        def __init__(self) -> None:
            self.captured: list[str] = []

        async def chat(self, messages, *args, **kwargs):
            user = next((m for m in messages if m.get("role") == "user"), None)
            if user is not None:
                self.captured.append(user["content"])
            return ("<tool_call>shell<arg_key>cmd</arg_key>"
                    "<arg_value>grep -rn x .</arg_value></tool_call>\n"
                    "[FILE: smallmod.py]\n```python\nimport os\n\n\n"
                    "def do_work() -> None:\n    print('y')\n    return None\n```")

    agent = _FakeAgent()
    agent.llm = _ToolCallLLM()
    cmd = FixCommand()
    desc = "change print in smallmod.py:2"
    asyncio.run(cmd.execute([str(small), "--desc", desc], agent))

    # The shell tool call is inert; the [FILE:] fix must still be applied.
    assert "print('y')" in small.read_text(encoding="utf-8")


def test_default_generate_false_when_no_tree_change() -> None:
    """Regression (2026-08-28): `_default_generate` must report False when the
    generator runs but changes nothing, so the caller raises `generate_failed`
    instead of a misleading `verify_failed`. The target is an existing tracked
    file and the generator is a no-op, so the tree stays clean.
    """
    from harnessfix.issue_loop import _default_generate

    issue = {
        "id": "iss-tmp-1",
        "category": "best-effort-except",
        "title": "do nothing",
        # An existing, tracked file: the no-op generator leaves it unchanged, so
        # git status stays clean and _tree_changed must return False.
        "locations": ["agent_core/llm/retry.py"],
        "autonomy_level": 1,
        "suggested_approach": "no-op",
    }
    agent = _FakeAgent()  # llm returns plain text -> no fix applied
    assert _default_generate(issue, agent) is False
