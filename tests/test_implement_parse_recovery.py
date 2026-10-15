"""Regression tests for implement's block parsing and self-correction fallbacks.

* Blocks whose closing code fence is missing were silently dropped
  (``[FILE:]`` / ``[COMPUTE_DIFF:]``), forcing a full batch retry.
* The self-correction fallbacks set ``patched_text`` but not ``patched``, and
  the post-loop ``patched_text = str(patched)`` then overwrote the recovered
  content with the old patch error string.
"""
from __future__ import annotations

import inspect
from pathlib import Path

from agent_core.commands import implement_cmd
from agent_core.commands.implement_cmd import (
    _describe_unparsed_response,
    _recover_unterminated_blocks,
    _save_unparsed_response,
    _strip_block_fences,
)


def test_strip_closed_block_removes_both_fences() -> None:
    assert _strip_block_fences("```python\nx = 1\n```\n") == ("x = 1", True)


def test_strip_unterminated_block_keeps_code() -> None:
    assert _strip_block_fences("```python\nx = 1\n") == ("x = 1", False)


def test_strip_bare_opening_fence_gives_empty_code() -> None:
    assert _strip_block_fences("```python") == ("", False)


def test_recover_unterminated_file_block() -> None:
    resp = "[FILE: agent_core/x.py]\n```python\ndef f():\n    return 1\n"
    files, diffs = _recover_unterminated_blocks(resp)
    assert files == [("agent_core/x.py", "def f():\n    return 1")]
    assert diffs == []


def test_recover_unterminated_compute_diff_block() -> None:
    resp = "[COMPUTE_DIFF: a.py]\n```python\nx = 2\n"
    files, diffs = _recover_unterminated_blocks(resp)
    assert files == []
    assert diffs == [("a.py", "x = 2")]


def test_recover_stops_at_next_header() -> None:
    resp = (
        "[FILE: a.py]\n```python\nx = 1\n"
        "[FILE: b.py]\n```python\ny = 2\n"
    )
    files, _ = _recover_unterminated_blocks(resp)
    assert files == [("a.py", "x = 1"), ("b.py", "y = 2")]


def test_recover_ignores_closed_blocks() -> None:
    resp = "[FILE: a.py]\n```python\nx = 1\n```\n"
    assert _recover_unterminated_blocks(resp) == ([], [])


def test_recover_ignores_header_only_response() -> None:
    # Observed in the failed run: the model emitted only the header.
    resp = "[COMPUTE_DIFF: agent_core/commands/registry.py]\n"
    assert _recover_unterminated_blocks(resp) == ([], [])


def test_describe_unparsed_reports_length_fences_and_tags() -> None:
    resp = "[COMPUTE_DIFF: a.py]\n```python\nx = 1\n"
    text = _describe_unparsed_response(resp)
    assert f"{len(resp)} chars" in text
    assert "1 fence marker(s)" in text
    assert "tags=['COMPUTE_DIFF']" in text


def test_describe_unparsed_handles_none_and_no_tags() -> None:
    assert "0 chars" in _describe_unparsed_response(None)
    assert "tags=none" in _describe_unparsed_response("just prose")


def test_save_unparsed_response_writes_beside_taskplan(tmp_path: Path) -> None:
    taskplan = tmp_path / "project_tasks.md"
    taskplan.write_text("# tasks\n", encoding="utf-8")
    saved = Path(_save_unparsed_response(str(taskplan), ["agent_core/config.py"], "raw text"))
    assert saved.parent == tmp_path / ".implement_unparsed"
    assert saved.name.startswith("agent_core_config.py_")
    assert saved.read_text(encoding="utf-8") == "raw text"


def test_fallbacks_set_patched_so_post_loop_keeps_content() -> None:
    """Each fallback that sets patched_text must also set patched.

    Otherwise ``patched_text = str(patched)`` replaces the recovered content
    with the stale patch error message.
    """
    src = inspect.getsource(implement_cmd)
    for name in ("retry_content", "fallback_content", "direct_content"):
        assert f"patched_text = {name}\n" in src, name
        assert f"patched = {name}\n" in src, name
