"""Regression tests for the per-file history RAG block (plan item #2, second half).

Goal: make ``harnessfix/history.py``'s execution corpus *readable from chat*.
Before this feature the trace index and the executions ledger were only ever
consulted by implement/fix; a normal chat turn never saw "this file's last
three tool runs errored".  The contract mirrors the habits feature:

1. :func:`extract_file_paths` — path tokens mentioned in a user message,
   mention order preserved, deduplicated, capped at ``MAX_FILES``.
2. :func:`file_notes_block` — renders nothing (empty string) when no query,
   no paths, or no history exists; with history it renders the marker exactly
   once and hard-caps its body lines.  It never raises.
3. Wiring: ``Agent._file_history_block`` uses the real ledger through
   ``_effective_ws_dir``; the marker is registered in
   ``_strip_dynamic_system_blocks`` so a stale block cannot accumulate across
   refreshes, and an empty workspace keeps the prompt byte-identical.

Every test writes only into ``tmp_path`` workspaces.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from agent_core.file_history import (
    FILE_HISTORY_MARKER,
    LINE_CAP,
    MAX_FILES,
    PER_FILE,
    extract_file_paths,
    file_notes_block,
)


def _seed_execution(workspace: Path, command: str, files: list[str], outcome: str = "ok", ts: float | None = None) -> None:
    """Append one structured execution record through the real ledger writer."""
    from harnessfix.history import append_execution

    ledger_dir = workspace / "reports" / "history"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    # append_execution stamps ts=now; write the file directly to control recency.
    import json as _json

    record = {
        "ts": time.time() if ts is None else ts,
        "id": f"{command}_{len(files)}",
        "command": command,
        "outcome": outcome,
        "files": files,
        "note": "",
    }
    with (ledger_dir / "executions.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(_json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# 1. extract_file_paths
# ---------------------------------------------------------------------------

def test_extract_finds_slashed_and_extension_paths_in_mention_order() -> None:
    text = "check agent_core/cli/commands/clear.py then docs/AGENT_ARCHITECTURE.md please"
    assert extract_file_paths(text) == [
        "agent_core/cli/commands/clear.py",
        "docs/AGENT_ARCHITECTURE.md",
    ]


def test_extract_normalizes_windows_separators() -> None:
    assert extract_file_paths(r"edit agent_core\cli\commands\clear.py now") == [
        "agent_core/cli/commands/clear.py"
    ]


def test_extract_deduplicates_case_insensitively() -> None:
    text = "see Agent_Core/Cli/Commands/CLEAR.PY and again agent_core/cli/commands/clear.py"
    paths = extract_file_paths(text)
    assert len(paths) == 1
    assert paths[0].lower() == "agent_core/cli/commands/clear.py"


def test_extract_caps_at_max_files_preserving_order() -> None:
    text = " ".join(f"a/f{i}.py" for i in range(10))
    paths = extract_file_paths(text)
    assert len(paths) == MAX_FILES
    assert paths[0] == "a/f0.py"


def test_extract_ignores_plain_prose() -> None:
    assert extract_file_paths("what's the best way to structure this module") == []


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_extract_empty_inputs_yield_nothing(empty) -> None:
    assert extract_file_paths(empty) == []


# ---------------------------------------------------------------------------
# 2. file_notes_block
# ---------------------------------------------------------------------------

def test_block_is_empty_without_a_query(tmp_path: Path) -> None:
    _seed_execution(tmp_path, "run", ["agent_core/cli/commands/clear.py"])
    assert file_notes_block("", tmp_path) == ""
    assert file_notes_block(None, tmp_path) == ""


def test_block_is_empty_when_no_history_exists(tmp_path: Path) -> None:
    """An untouched workspace keeps the prompt byte-identical to the baseline."""
    assert file_notes_block("read agent_core/cli/commands/clear.py", tmp_path) == ""


def test_block_renders_marker_once_and_names_the_file(tmp_path: Path) -> None:
    _seed_execution(
        tmp_path, "run", ["agent_core/cli/commands/clear.py"], outcome="2 passed"
    )

    block = file_notes_block("fix agent_core/cli/commands/clear.py", tmp_path)

    assert block.startswith(FILE_HISTORY_MARKER)
    assert block.count("RECENT FILE NOTES") == 1
    assert "clear.py" in block
    assert "2 passed" in block


def test_block_uses_the_history_matchers_not_just_exact_paths(tmp_path: Path) -> None:
    """A directory-arg ledger entry counts for the file mentioned by the query.

    ``history._matches_arg`` says a dir arg matches a direct child; the block
    must inherit that matcher behaviour, not naive string equality.
    """
    from harnessfix.history import clear_history_cache

    clear_history_cache()
    _seed_execution(
        tmp_path, "run", ["agent_core/cli/commands"], outcome="tests green"
    )

    block = file_notes_block("read agent_core/cli/commands/clear.py", str(tmp_path))
    assert FILE_HISTORY_MARKER in block
    assert "tests green" in block


def test_block_caps_events_per_file(tmp_path: Path) -> None:
    for i in range(6):
        _seed_execution(tmp_path, f"cmd{i}", ["agent_core/cli/commands/clear.py"], ts=100.0 + i)

    block = file_notes_block("read agent_core/cli/commands/clear.py", tmp_path)
    event_lines = [ln for ln in block.splitlines() if "cmd" in ln]
    assert len(event_lines) <= PER_FILE


def test_block_caps_total_body_lines(tmp_path: Path) -> None:
    for i in range(10):
        _seed_execution(tmp_path, f"c{i}", [f"a/f{i}.py"], ts=100.0 + i)

    block = file_notes_block(" ".join(f"a/f{i}.py" for i in range(10)), tmp_path)
    body_lines = [ln for ln in block.splitlines() if ln.strip()]
    assert len(body_lines) <= LINE_CAP + 1  # header marker line + capped body


def test_block_never_raises_on_garbage_inputs() -> None:
    assert file_notes_block(object(), object()) == ""  # type: ignore[arg-type]
    assert file_notes_block("agent.py", "/nonexistent/drive/Q:/nope") == ""


# ---------------------------------------------------------------------------
# 3. Agent wiring
# ---------------------------------------------------------------------------

def _bare_agent(workspace: Path):
    """A real Agent instance without the heavy __init__ (same trick as habits tests)."""
    from agent import Agent

    bot = Agent.__new__(Agent)
    bot._effective_ws_dir = lambda: str(workspace)  # type: ignore[method-assign]
    return bot


def test_agent_method_renders_the_block_from_the_real_ledger(tmp_path: Path) -> None:
    _seed_execution(tmp_path, "run", ["agent_core/cli/commands/clear.py"], outcome="3 passed")

    block = _bare_agent(tmp_path)._file_history_block("fix agent_core/cli/commands/clear.py")

    assert block.startswith(FILE_HISTORY_MARKER)
    assert "3 passed" in block


def test_agent_method_is_empty_without_history(tmp_path: Path) -> None:
    assert _bare_agent(tmp_path)._file_history_block("read agent_core/cli/commands/clear.py") == ""


def test_agent_method_never_raises() -> None:
    bot = _bare_agent(Path("/nonexistent/nope"))
    assert bot._file_history_block(None) == ""  # type: ignore[arg-type]


def test_strip_dynamic_system_blocks_removes_the_file_history_block() -> None:
    from agent import _strip_dynamic_system_blocks

    text = "BASE PROMPT" + FILE_HISTORY_MARKER + "\n- some/file.py: 1 past event(s)\n"
    assert _strip_dynamic_system_blocks(text) == "BASE PROMPT"


def test_prompt_rebuild_keeps_the_block_exactly_once(tmp_path: Path, monkeypatch) -> None:
    import agent as agent_mod

    ws = tmp_path / "ws"
    ws.mkdir()
    _seed_execution(ws, "run", ["agent_core/cli/commands/clear.py"], outcome="green")

    monkeypatch.setattr(agent_mod, "CHAT_HISTORY_JSON_PATH", str(tmp_path / "chat_history.json"))
    monkeypatch.setattr(agent_mod, "AGENT_MEMORY_JSON_PATH", str(tmp_path / "agent_memory.json"))
    bot = agent_mod.Agent(workspace=str(ws))

    bot._refresh_system_message(query="read agent_core/cli/commands/clear.py")
    first = bot._chat_history[0]["content"]
    bot._refresh_system_message(query="read agent_core/cli/commands/clear.py")
    second = bot._chat_history[0]["content"]

    assert FILE_HISTORY_MARKER in first, "file-history block missing from the prompt"
    assert second == first, "dynamic blocks accumulated across refreshes"
    assert second.count("RECENT FILE NOTES") == 1


def test_empty_workspace_keeps_the_prompt_byte_identical(tmp_path: Path, monkeypatch) -> None:
    import agent as agent_mod

    ws = tmp_path / "ws"
    ws.mkdir()

    monkeypatch.setattr(agent_mod, "CHAT_HISTORY_JSON_PATH", str(tmp_path / "chat_history.json"))
    monkeypatch.setattr(agent_mod, "AGENT_MEMORY_JSON_PATH", str(tmp_path / "agent_memory.json"))
    bot = agent_mod.Agent(workspace=str(ws))

    bot._refresh_system_message(query="read agent_core/cli/commands/clear.py")
    content = bot._chat_history[0]["content"]
    assert FILE_HISTORY_MARKER not in content
    assert agent_mod._strip_dynamic_system_blocks(content) == agent_mod._SYSTEM_PROMPT
