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
    resp = "[FILE: a.py]\n```python\nx = 1\n[FILE: b.py]\n```python\ny = 2\n"
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
    saved = Path(
        _save_unparsed_response(str(taskplan), ["agent_core/config.py"], "raw text")
    )
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


class TestComputeDiffBatchParse:
    """[COMPUTE_DIFF:] batch answers must reach the compute-diff processor.

    Regression (batch-1 "no blocks parsed"): the discard gate after the parse
    retries checked only ``matches``/``patch_matches`` -- never
    ``compute_diff_matches`` -- and the retry paths never parsed
    ``[COMPUTE_DIFF:]`` at all.  A response that followed the retry
    instruction ("Output exactly one block ... [COMPUTE_DIFF: ...] with
    complete code (we compute diff)") was therefore always classified as
    unparsed, saved to .implement_unparsed, and the batch was dropped.
    """

    @staticmethod
    def _run(tmp_path, chat):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import patch

        from agent_core.commands.implement_cmd import ImplementCommand

        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "mod_me.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        (tmp_path / "tasks.md").write_text(
            "1. `pkg/mod_me.py` - extend foo\n", encoding="utf-8"
        )
        agent = SimpleNamespace(workspace=str(tmp_path), llm=SimpleNamespace(chat=chat))
        with patch("agent_core.commands.implement_cmd.auto_choice", return_value="y"):
            return asyncio.run(
                ImplementCommand().execute(
                    [str(tmp_path / "tasks.md"), "--modify"], agent
                )
            )

    def test_compute_diff_only_response_is_applied(self, tmp_path) -> None:
        async def chat(messages, **kwargs):
            return (
                "[COMPUTE_DIFF: pkg/mod_me.py]\n"
                "```python\ndef foo():\n    return 2\n```\n"
            )

        ok = self._run(tmp_path, chat)
        target = tmp_path / "pkg" / "mod_me.py"
        assert ok is True
        assert target.read_text(encoding="utf-8") == "def foo():\n    return 2\n"
        assert not (tmp_path / ".implement_unparsed").exists()

    def test_compute_diff_on_retry_is_applied(self, tmp_path) -> None:
        calls = {"n": 0}

        async def chat(messages, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return "Sorry, I can only explain the change in prose."
            return (
                "[COMPUTE_DIFF: pkg/mod_me.py]\n"
                "```python\ndef foo():\n    return 2\n```\n"
            )

        ok = self._run(tmp_path, chat)
        target = tmp_path / "pkg" / "mod_me.py"
        assert ok is True
        assert target.read_text(encoding="utf-8") == "def foo():\n    return 2\n"
        assert calls["n"] >= 2
        assert not (tmp_path / ".implement_unparsed").exists()


def test_discard_gate_checks_all_three_block_kinds() -> None:
    """Structural: the unparsed-response gate must consider all three kinds.

    The discard gate is the bare two-way condition directly guarding the
    "Could not parse files" warning; it must be gone (the compute step inside
    ``_parse_response_blocks`` legitimately keeps a two-way form).  The full
    three-way condition must appear at least twice -- recovery + discard.
    """
    import re as _re

    src = inspect.getsource(implement_cmd)
    assert (
        _re.search(
            r'if not matches and not patch_matches:\s*\n\s*print\(\s*"  Warning: Could not parse files',
            src,
        )
        is None
    )
    assert (
        src.count("not matches and not patch_matches and not compute_diff_matches") >= 2
    )


def test_parse_response_blocks_file_wins_over_others() -> None:
    from agent_core.commands.implement_cmd import _parse_response_blocks

    resp = (
        "[FILE: a.py]\n```python\nx = 1\n```\n"
        "[PATCH: b.py]\n@@ -1 +1 @@\n"
        "[COMPUTE_DIFF: c.py]\n```python\ny = 2\n```\n"
    )
    files, patches, diffs = _parse_response_blocks(resp, modify_mode=True)
    assert files == [("a.py", "x = 1")]
    assert patches == []
    assert diffs == []


def test_parse_response_blocks_patch_requires_modify_mode() -> None:
    from agent_core.commands.implement_cmd import _parse_response_blocks

    resp = "[PATCH: b.py]\n@@ -1 +1 @@\n-old\n+new\n"
    files, patches, diffs = _parse_response_blocks(resp, modify_mode=False)
    assert files == [] and patches == [] and diffs == []
    files, patches, diffs = _parse_response_blocks(resp, modify_mode=True)
    assert patches and patches[0][0] == "b.py"
    assert files == [] and diffs == []


def test_parse_response_blocks_compute_when_no_other_kind() -> None:
    from agent_core.commands.implement_cmd import _parse_response_blocks

    resp = "[COMPUTE_DIFF: a.py]\n```python\nx = 2\n```\n"
    files, patches, diffs = _parse_response_blocks(resp, modify_mode=True)
    assert files == [] and patches == []
    assert diffs == [("a.py", "x = 2")]


def test_parse_response_blocks_handles_none_response() -> None:
    from agent_core.commands.implement_cmd import _parse_response_blocks

    assert _parse_response_blocks(None, modify_mode=True) == ([], [], [])


class TestMultiBlockParse:
    """Regression: a clean multi-block response must yield EVERY block.

    The first (``$``-anchored) FILE pattern let its lazy body backtrack to
    the LAST closing fence, collapsing two well-formed [FILE:] blocks into a
    single over-captured match whose body contained the second block's
    header -- the compile gate rejected it and the batch produced zero
    files (batch-1 failure with clean LLM output).
    """

    @staticmethod
    def _run(tmp_path, chat):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import patch

        from agent_core.commands.implement_cmd import ImplementCommand

        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (tmp_path / "tasks.md").write_text(
            "1. `pkg/alpha_report.py` - alpha report\n"
            "2. `pkg/json_serializer.py` - json serializer\n",
            encoding="utf-8",
        )
        agent = SimpleNamespace(workspace=str(tmp_path), llm=SimpleNamespace(chat=chat))
        with patch("agent_core.commands.implement_cmd.auto_choice", return_value="y"):
            return asyncio.run(
                ImplementCommand().execute([str(tmp_path / "tasks.md")], agent)
            )

    def test_two_file_blocks_both_written(self, tmp_path) -> None:
        async def chat(messages, **kwargs):
            return (
                "[FILE: pkg/alpha_report.py]\n"
                "```python\nclass AlphaReport:\n"
                "    name = 'alpha'\n\n"
                "    def label(self) -> str:\n"
                "        return f'label for {self.name}'\n```\n"
                "[FILE: pkg/json_serializer.py]\n"
                "```python\nimport json\n\n"
                "def serialize(items: list[int]) -> str:\n"
                "    return json.dumps(items, sort_keys=True)\n```\n"
            )

        ok = self._run(tmp_path, chat)
        assert ok is True
        a = tmp_path / "pkg" / "alpha_report.py"
        b = tmp_path / "pkg" / "json_serializer.py"
        assert a.is_file(), "first block dropped"
        assert b.is_file(), "second block dropped"
        assert "class AlphaReport" in a.read_text(encoding="utf-8")
        assert "def serialize" in b.read_text(encoding="utf-8")
        assert not (tmp_path / ".implement_unparsed").exists()


def test_parse_response_blocks_returns_every_file_block() -> None:
    from agent_core.commands.implement_cmd import _parse_response_blocks

    resp = "[FILE: a.py]\n```python\nx = 1\n```\n[FILE: b.py]\n```python\ny = 2\n```\n"
    files, patches, diffs = _parse_response_blocks(resp, modify_mode=False)
    assert files == [("a.py", "x = 1"), ("b.py", "y = 2")]
    assert patches == [] and diffs == []
