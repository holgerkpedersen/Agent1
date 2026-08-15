"""Tests for the NLP search tool and FileSearcher.

Regression coverage: the search handler once sliced the result STRING
char-by-char (per-character output lines), and the searcher matched
runtime state files (.git logs, chat_history.json, mypy caches) instead
of source code.
"""
import asyncio

from agent_core.file_searcher import FileSearcher


class TestFileSearcher:
    def test_returns_line_numbered_matches(self, tmp_path):
        (tmp_path / "mod.py").write_text(
            "def a():\n    x = 1\n\n\ndef _execute_nlp_tool():\n    pass\n",
            encoding="utf-8",
        )
        (tmp_path / "other.py").write_text(
            "def _execute_nlp_tool():\n    return 2\n",
            encoding="utf-8",
        )

        result = asyncio.run(FileSearcher().search("_execute_nlp_tool", str(tmp_path)))
        lines = result.splitlines()
        assert len(lines) == 2
        assert str(tmp_path / "mod.py") + ":5" in lines[0]
        assert str(tmp_path / "other.py") + ":1" in lines[1]
        assert "def _execute_nlp_tool():" in lines[0]

    def test_excludes_runtime_state_and_caches(self, tmp_path):
        (tmp_path / "src.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        (tmp_path / "chat_history.json").write_text('{"x": "foo"}', encoding="utf-8")
        (tmp_path / "model.json").write_text('{"model": "foo"}', encoding="utf-8")
        (tmp_path / ".env").write_text("FOO=1", encoding="utf-8")
        (tmp_path / "cache.db").write_text("foo binary", encoding="utf-8")
        (tmp_path / "data.png").write_text("foo", encoding="utf-8")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "logs").mkdir(parents=True)
        (tmp_path / ".git" / "logs" / "HEAD").write_text("foo commit", encoding="utf-8")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "src.cpython-312.pyc").write_text(
            "foo", encoding="utf-8"
        )
        (tmp_path / ".mypy_cache").mkdir()
        (tmp_path / ".mypy_cache" / "cache.6.db").write_text("foo", encoding="utf-8")

        result = asyncio.run(FileSearcher().search("foo", str(tmp_path)))
        lines = result.splitlines()
        # Only src.py may match — every other hit is excluded state/binary.
        assert len(lines) == 1
        assert "src.py:1" in lines[0]

    def test_no_matches_message(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        result = asyncio.run(FileSearcher().search("zzz-not-there", str(tmp_path)))
        assert result == "No matches found"

    def test_project_workflow_docs_are_not_code_matches(self, tmp_path):
        """project_*.md are temporary workflow artifacts — a symbol that only
        lives in them (like '_execute_nlp_tool') must not appear as a match."""
        (tmp_path / "project_plan.md").write_text(
            "[MUST] Patch `_execute_nlp_tool` command injection vulnerability",
            encoding="utf-8",
        )
        (tmp_path / "project_tasks.md").write_text(
            "1. `_execute_nlp_tool` shell=True replacement",
            encoding="utf-8",
        )
        (tmp_path / "src.py").write_text("def _execute_tool_call():\n    pass\n", encoding="utf-8")

        result = asyncio.run(FileSearcher().search("_execute_nlp_tool", str(tmp_path)))
        assert result == "No matches found"

        result2 = asyncio.run(FileSearcher().search("_execute_tool_call", str(tmp_path)))
        assert "src.py" in result2
        assert "project_plan.md" not in result2
        assert "project_tasks.md" not in result2

    def test_caps_results(self, tmp_path):
        for i in range(60):
            (tmp_path / f"f{i}.py").write_text(f"needle = {i}", encoding="utf-8")
        result = asyncio.run(FileSearcher().search("needle", str(tmp_path)))
        lines = result.splitlines()
        assert len(lines) == 51  # 50 matches + "… and N more"
        assert "and 10 more matches" in lines[-1]

    def test_docs_run_folders_are_not_code_matches(self, tmp_path):
        """Workflow docs now live in .docs/<timestamp>/ — a symbol that only
        lives in them must not appear as a code match (like the old
        root-level project_*.md rule)."""
        run = tmp_path / ".docs" / "2026-08-15_11-17-11"
        run.mkdir(parents=True)
        (run / "project_plan.md").write_text(
            "[MUST] Patch `_execute_nlp_tool` command injection vulnerability",
            encoding="utf-8",
        )
        (run / "project_tasks.md").write_text(
            "1. `_execute_nlp_tool` shell=True replacement",
            encoding="utf-8",
        )
        (tmp_path / "src.py").write_text("def _execute_tool_call():\n    pass\n", encoding="utf-8")

        result = asyncio.run(FileSearcher().search("_execute_nlp_tool", str(tmp_path)))
        assert result == "No matches found"

        result2 = asyncio.run(FileSearcher().search("_execute_tool_call", str(tmp_path)))
        assert "src.py" in result2
        assert ".docs" not in result2


class TestSearchToolHandler:
    """Drive Agent._execute_tool_call('search', ...) — regression for the
    char-by-char result formatting bug."""

    def test_search_tool_returns_one_line_per_match(self, tmp_path):
        from agent import Agent
        (tmp_path / "mod.py").write_text(
            "def target_fn():\n    return 1\n", encoding="utf-8"
        )

        async def run():
            return await Agent(workspace=str(tmp_path))._execute_tool_call(
                "search", {"query": "target_fn", "path": "."},
            )

        result = asyncio.run(run())
        lines = result.splitlines()
        assert len(lines) == 1
        assert "mod.py" in lines[0]
        assert "def target_fn" in lines[0]

    def test_search_tool_no_match_message(self, tmp_path):
        from agent import Agent
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")

        async def run():
            return await Agent(workspace=str(tmp_path))._execute_tool_call(
                "search", {"query": "never-present", "path": "."},
            )

        assert "No files found matching that query." in asyncio.run(run())

    def test_search_tool_excludes_chat_history(self, tmp_path):
        from agent import Agent
        (tmp_path / "src.py").write_text("def real_symbol():\n    pass\n", encoding="utf-8")
        (tmp_path / "chat_history.json").write_text(
            '{"content": "real_symbol mention in conversation"}', encoding="utf-8"
        )

        async def run():
            return await Agent(workspace=str(tmp_path))._execute_tool_call(
                "search", {"query": "real_symbol", "path": "."},
            )

        result = asyncio.run(run())
        assert "chat_history" not in result
        assert "src.py" in result
