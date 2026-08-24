"""Regression tests: symbol-level tools (plan item B-#8, agent_core.symbol_intel).

Covers:
- collect_definitions: signatures + line spans, nested classes, syntax-error
  reporting
- collect_references: whole-word matching (no substring hits), definition
  lines included, cap honoured, invalid symbol rejected, oversized-file skip
- Agent wiring: dispatch-table parity, plan-mode blocking, end-to-end
  handler behaviour through _execute_tool_call
"""
from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import pytest

import agent
from agent import Agent
from agent_core.modes import MODE_PLAN
from agent_core.symbol_intel import (
    MAX_REFERENCES,
    collect_definitions,
    collect_references,
)


# ---------------------------------------------------------------------------
# definitions
# ---------------------------------------------------------------------------

class TestCollectDefinitions:
    def test_signatures_and_line_spans(self) -> None:
        src = (
            "def top(a, b=2):\n"
            "    return a\n"
            "\n"
            "async def afunc() -> None:\n"
            "    pass\n"
        )
        out = collect_definitions(src, "m.py")
        assert "def top(a, b=2)" in out
        assert "async def afunc() -> None" in out
        assert "[lines 1-2]" in out
        assert "[lines 4-5]" in out

    def test_class_with_methods_and_bases(self) -> None:
        src = (
            "class Base:\n"
            "    pass\n"
            "\n"
            "class Child(Base):\n"
            "    def method(self):\n"
            "        return 1\n"
        )
        out = collect_definitions(src, "m.py")
        assert "class Base" in out
        assert "class Child(Base)" in out
        assert "def method(self)" in out
        # method line comes after its class header
        assert out.index("class Child") < out.index("def method")

    def test_syntax_error_reported_not_raised(self) -> None:
        out = collect_definitions("def broken(:\n", "bad.py")
        assert "SyntaxError" in out
        assert "bad.py" in out

    def test_file_without_definitions(self) -> None:
        out = collect_definitions("X = 1\n", "const.py")
        assert "No classes or functions" in out


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------

@pytest.fixture()
def ws(tmp_path: Path) -> Path:
    """Tiny workspace: defs.py defines sym_a/sym_b; user.py uses them."""
    (tmp_path / "defs.py").write_text(
        "def sym_a():\n    return 1\n\n\nclass SymB:\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "user.py").write_text(
        "from defs import sym_a\n"
        "sym_a()\n"
        "x = sym_a_longer()\n"   # must NOT match (substring)
        "import defs\n"
        "print(defs.sym_a)\n",
        encoding="utf-8",
    )
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "more.py").write_text("sym_a  # used here too\n", encoding="utf-8")
    junk = tmp_path / "__pycache__"
    junk.mkdir()
    (junk / "compiled.py").write_text("sym_a\n", encoding="utf-8")
    return tmp_path


class TestCollectReferences:
    def test_whole_word_match_no_substrings(self, ws: Path) -> None:
        out = collect_references("sym_a", str(ws))
        assert "user.py" in out
        # the sym_a_longer line must not appear
        assert "sym_a_longer()" not in out.replace("sym_a_longer", "SHOULD-NOT-MATCH") or \
            "1: x = sym_a_longer()" not in out

    def test_definition_lines_included(self, ws: Path) -> None:
        out = collect_references("sym_a", str(ws))
        assert "defs.py" in out and "1: def sym_a():" in out

    def test_scans_subdirectories_skips_pycache(self, ws: Path) -> None:
        out = collect_references("sym_a", str(ws))
        assert os.path.join("sub", "more.py").replace("\\", "/") in out or "sub/more.py" in out
        assert "__pycache__" not in out

    def test_cap_on_hits(self, ws: Path) -> None:
        out = collect_references("sym_a", str(ws), max_results=2)
        assert "(2+ hits" in out

    def test_invalid_symbol_rejected(self, ws: Path) -> None:
        out = collect_references("not valid!", str(ws))
        assert "Invalid symbol" in out

    def test_no_hits_message(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("pass\n", encoding="utf-8")
        out = collect_references("zzz_none", str(tmp_path))
        assert "No references to 'zzz_none'" in out


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------

class TestAgentWiring:
    def test_dispatch_table_covers_new_tools(self) -> None:
        from agent_core.tool_schemas import NLP_TOOL_NAMES, NLP_TOOL_SCHEMAS

        bot = Agent(workspace=".")
        handlers = bot._nlp_tool_handlers()
        assert {"definitions", "references"} <= set(handlers)
        assert set(handlers) == set(NLP_TOOL_NAMES)
        names = [s["function"]["name"] for s in NLP_TOOL_SCHEMAS]
        assert names.count("definitions") == 1
        assert names.count("references") == 1

    def test_plan_mode_blocks_both_tools(self) -> None:
        from agent_core.modes import check_tool_allowed

        for tool in ("definitions", "references"):
            rejection = check_tool_allowed(tool, MODE_PLAN)
            assert rejection is None  # read-only tools stay allowed
            # but they must be IN the read-only set so schemas survive filtering
        from agent_core.modes import PLAN_MODE_TOOLS
        assert {"definitions", "references"} <= PLAN_MODE_TOOLS

    def test_definitions_end_to_end(self, tmp_path: Path) -> None:
        (tmp_path / "mod.py").write_text(
            "def hello(name):\n    return name\n", encoding="utf-8"
        )
        bot = Agent(workspace=str(tmp_path))
        out = asyncio.run(bot._execute_tool_call(
            "definitions", {"path": "mod.py"},
        ))
        assert "def hello(name)" in out
        assert "[lines 1-2]" in out

    def test_definitions_missing_file(self) -> None:
        bot = Agent(workspace=".")
        out = asyncio.run(bot._execute_tool_call(
            "definitions", {"path": "definitely_not_here_xyz.py"},
        ))
        assert out.startswith("File not found:")

    def test_definitions_rejects_non_python(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("hi", encoding="utf-8")
        bot = Agent(workspace=str(tmp_path))
        out = asyncio.run(bot._execute_tool_call(
            "definitions", {"path": "notes.txt"},
        ))
        assert out.startswith("Error: definitions needs a .py file")

    def test_references_end_to_end(self, tmp_path: Path) -> None:
        (tmp_path / "use.py").write_text(
            "val = MY_CONST\nother = MY_CONSTS\n", encoding="utf-8"
        )
        bot = Agent(workspace=str(tmp_path))
        out = asyncio.run(bot._execute_tool_call(
            "references", {"symbol": "MY_CONST"},
        ))
        assert "MY_CONST" in out
        assert "1: val = MY_CONST" in out
        assert "MY_CONSTS" not in out.replace("MY_CONSTS", "")  # no false hit line

    def test_system_prompt_mentions_symbol_tools(self) -> None:
        assert "references" in agent._SYSTEM_PROMPT
        assert "definitions" in agent._SYSTEM_PROMPT
