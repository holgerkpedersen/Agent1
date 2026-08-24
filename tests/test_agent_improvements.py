"""Regression tests for the agent.py NLP tool-dispatch refactor.

Covers:
- dispatch-table completeness vs ``NLP_TOOL_NAMES`` (schema ↔ executor parity)
- uniform handler signature (no positional ws_dir leaks)
- per-tool behaviour preserved after extraction (_nlp_read/_nlp_edit/...)
- new guards: unknown-tool message, handler-exception containment,
  list_files missing-dir handling, edit file-not-found path-recovery prefix
- shared ``_truncate_output`` helper used by run/git/diff/tests tools
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agent import (
    Agent,
    _blocked_shell_command,
    _truncate_output,
)
from agent_core.tool_schemas import NLP_TOOL_NAMES


@pytest.fixture()
def agent() -> Agent:
    return Agent(workspace=".")


# ---------------------------------------------------------------------------
# Dispatch table integrity
# ---------------------------------------------------------------------------

def test_dispatch_table_covers_every_advertised_tool(agent: Agent) -> None:
    handlers = agent._nlp_tool_handlers()
    assert set(handlers) == set(NLP_TOOL_NAMES)


def test_unknown_tool_names_available_set(agent: Agent) -> None:
    result = asyncio.run(agent._execute_tool_call("teleport", {}))
    assert result.startswith("Unknown tool: teleport")
    for tool in ("search", "read", "write", "edit", "run", "tests"):
        assert tool in result


def test_tool_name_is_case_insensitive(agent: Agent, tmp_path: Path) -> None:
    target = tmp_path / "case_probe.txt"
    target.write_text("hello", encoding="utf-8")
    result = asyncio.run(agent._execute_tool_call(
        "READ", {"path": str(target), "offset": 1, "limit": 10}
    ))
    assert result == "hello"


def test_handler_exception_is_contained(agent: Agent) -> None:
    """A raising handler must not blow up chat_nlp: the loop turns exceptions
    into 'Tool error:' strings, so the dispatcher itself must never raise."""
    with patch.object(Agent, "_nlp_read", AsyncMock(side_effect=RuntimeError("boom"))):
        result = asyncio.run(agent._execute_tool_call("read", {"path": "x"}))
    assert "error" in result.lower()
    assert "boom" in result


# ---------------------------------------------------------------------------
# _nlp_read
# ---------------------------------------------------------------------------

def test_nlp_read_reports_offset_beyond_eof(agent: Agent, tmp_path: Path) -> None:
    target = tmp_path / "short.txt"
    target.write_text("one\ntwo\n", encoding="utf-8")
    result = asyncio.run(agent._nlp_read({"path": str(target), "offset": 99}))
    assert "beyond the end" in result
    assert "2 lines" in result


def test_nlp_read_paging_hint_carries_next_offset(agent: Agent, tmp_path: Path) -> None:
    target = tmp_path / "long.txt"
    target.write_text("\n".join(f"line-{i}" for i in range(50)), encoding="utf-8")
    result = asyncio.run(agent._nlp_read({"path": str(target), "offset": 1, "limit": 10}))
    assert "[truncated — use read with offset=11 to continue]" in result
    assert "line-0" in result and "line-10" not in result


def test_nlp_read_rejects_non_integer_offset(agent: Agent, tmp_path: Path) -> None:
    target = tmp_path / "x.txt"
    target.write_text("data", encoding="utf-8")
    result = asyncio.run(agent._nlp_read({"path": str(target), "offset": "abc"}))
    assert "must be integers" in result


# ---------------------------------------------------------------------------
# _nlp_list_files
# ---------------------------------------------------------------------------

def test_nlp_list_files_marks_directories_and_sorts(agent: Agent, tmp_path: Path) -> None:
    (tmp_path / "z_dir").mkdir()
    (tmp_path / "a_file.txt").write_text("", encoding="utf-8")
    result = asyncio.run(agent._nlp_list_files({"path": str(tmp_path)}))
    lines = result.splitlines()
    assert lines[0] == "  a_file.txt"
    assert lines[-1] == "  z_dir/"


def test_nlp_list_files_missing_directory(agent: Agent, tmp_path: Path) -> None:
    result = asyncio.run(
        agent._nlp_list_files({"path": str(tmp_path / "does_not_exist")})
    )
    assert result.startswith("Directory not found:")


def test_nlp_list_files_not_a_directory(agent: Agent, tmp_path: Path) -> None:
    f = tmp_path / "plain.txt"
    f.write_text("", encoding="utf-8")
    result = asyncio.run(agent._nlp_list_files({"path": str(f)}))
    assert result.startswith("Not a directory:")


def test_nlp_list_files_caps_at_fifty_sorted_entries(
    agent: Agent, tmp_path: Path
) -> None:
    for i in range(80):
        (tmp_path / f"f{i:03d}.txt").write_text("", encoding="utf-8")
    result = asyncio.run(agent._nlp_list_files({"path": str(tmp_path)}))
    assert len(result.splitlines()) == 50
    # Sorted before capping: the first 50 names are kept, not the first created.
    assert "f000.txt" in result and "f079.txt" not in result


# ---------------------------------------------------------------------------
# _nlp_edit
# ---------------------------------------------------------------------------

def test_nlp_edit_missing_file_uses_path_miss_prefix(agent: Agent, tmp_path: Path) -> None:
    """'File not found:' is the prefix ToolLoopRunner's path-recovery guard
    recognises for read/edit/write — edit must keep using it so a mistyped
    path triggers the parent-listing recovery instead of a dead end."""
    result = asyncio.run(agent._nlp_edit({
        "path": str(tmp_path / "ghost.py"),
        "old_text": "a", "new_text": "b",
    }))
    assert result.startswith("File not found:")


def test_nlp_edit_old_text_not_found_message(agent: Agent, tmp_path: Path) -> None:
    target = tmp_path / "code.py"
    target.write_text("value = 1\n", encoding="utf-8")
    result = asyncio.run(agent._nlp_edit({
        "path": str(target), "old_text": "NOPE", "new_text": "b",
    }))
    assert "Text not found" in result


def test_nlp_edit_applies_first_occurrence_and_verifies_py(
    agent: Agent, tmp_path: Path
) -> None:
    target = tmp_path / "code.py"
    target.write_text("value = 1\nvalue = 1\n", encoding="utf-8")
    result = asyncio.run(agent._nlp_edit({
        "path": str(target), "old_text": "value = 1", "new_text": "value = 2",
    }))
    assert result.startswith(f"Edited {target}")
    assert "[verify] py_compile" in result
    content = target.read_text(encoding="utf-8")
    assert content == "value = 2\nvalue = 1\n"  # only the FIRST occurrence


# ---------------------------------------------------------------------------
# _nlp_web_search (offline validation only — no network in unit tests)
# ---------------------------------------------------------------------------

def test_nlp_web_search_requires_query(agent: Agent) -> None:
    result = asyncio.run(agent._nlp_web_search({}))
    assert result == "Error: web_search requires a query."


# ---------------------------------------------------------------------------
# Shared output truncation
# ---------------------------------------------------------------------------

def test_truncate_output_short_pass_through() -> None:
    assert _truncate_output("ok") == "ok"


def test_truncate_output_empty_becomes_placeholder() -> None:
    assert _truncate_output("") == "(no output)"


def test_truncate_output_bounds_long_output() -> None:
    out = "A" * 3000 + "\n" + "B" * 3000
    truncated = _truncate_output(out)
    assert len(truncated) < len(out)
    assert "... [truncated] ..." in truncated
    assert truncated.startswith("A" * 2500)
    assert truncated.endswith("B" * 2500)


# ---------------------------------------------------------------------------
# Effective workspace resolution
# ---------------------------------------------------------------------------

def test_effective_ws_dir_falls_back_when_override_is_invalid(
    agent: Agent, tmp_path: Path
) -> None:
    agent._nlp_workspace = str(tmp_path / "missing_dir")
    assert agent._effective_ws_dir() == agent.workspace


def test_effective_ws_dir_prefers_existing_override(
    agent: Agent, tmp_path: Path
) -> None:
    agent._nlp_workspace = str(tmp_path)
    assert agent._effective_ws_dir() == str(Path(tmp_path).resolve())
    agent._nlp_workspace = None


# ---------------------------------------------------------------------------
# Destructive-command scan stays wired into the run tool
# ---------------------------------------------------------------------------

def test_run_tool_still_blocks_destructive_commands(agent: Agent) -> None:
    assert _blocked_shell_command("rm -rf /") is not None
    result = asyncio.run(agent._nlp_run({"command": "format c:"}))
    assert "Dangerous command blocked" in result


# ---------------------------------------------------------------------------
# Chat-history projection keeps its bounded-window contract
# ---------------------------------------------------------------------------

def test_trim_chat_history_handles_non_system_head() -> None:
    from agent import _trim_chat_history

    messages = [
        {"role": "user", "content": "first"},
        *[{"role": "user", "content": f"m{i}"} for i in range(70)],
    ]
    trimmed = _trim_chat_history(messages)
    # No system prompt at index 0 → nothing is treated as one; window still bounded.
    assert len(trimmed) <= 60
    assert trimmed[-1]["content"] == "m69"
