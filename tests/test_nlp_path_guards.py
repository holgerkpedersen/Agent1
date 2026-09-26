"""NLP tools must reject a DIRECTORY path with a clear message.

Regression (2026-09-26): `analyze(path="agent_core/")` opened the directory
and surfaced Windows' opaque ``Error reading file: [Errno 13] Permission
denied: 'C:\\Dev\\Agent1\\agent_core'``.  The tool schemas also said nothing
about file-vs-directory, so the model had no instruction to avoid it.
"""

import asyncio

from agent import Agent


def _agent(workspace) -> Agent:
    """Minimal Agent for handler tests (no provider/memory construction)."""
    agent = Agent.__new__(Agent)
    agent.workspace = str(workspace)
    agent._nlp_workspace = None
    agent._pending_effects = None
    agent._files_read = set()
    agent._file_mtimes = {}
    return agent


def test_read_file_directory_message(tmp_path):
    agent = _agent(tmp_path)
    out = asyncio.run(agent.read_file(str(tmp_path)))
    assert "Not a file (directory)" in out
    assert "Permission denied" not in out
    assert "Errno 13" not in out


def test_nlp_read_directory_message(tmp_path):
    agent = _agent(tmp_path)
    out = asyncio.run(agent._nlp_read({"path": str(tmp_path)}))
    assert "Not a file (directory)" in out
    assert "list_files" in out


def test_nlp_analyze_empty_directory(tmp_path):
    agent = _agent(tmp_path)
    out = asyncio.run(agent._nlp_analyze({"path": str(tmp_path)}))
    assert "No analyzable files found" in out


class _CaptureLLM:
    def __init__(self):
        self.messages = None

    async def chat(self, messages, tools=None, **kwargs):
        self.messages = messages
        return "ANALYSIS"


def test_nlp_analyze_directory_summarizes_files(tmp_path):
    """`analyze <folder>` is supported: a bounded sample of the directory's
    files is summarized (it used to fail with PermissionError)."""
    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    agent = _agent(tmp_path)
    agent.llm = _CaptureLLM()
    out = asyncio.run(agent._nlp_analyze({"path": str(tmp_path)}))
    assert "ANALYSIS" in out
    prompt = agent.llm.messages[-1]["content"]
    assert "mod.py" in prompt
    assert "x = 1" in prompt


def test_read_file_still_reads_real_files(tmp_path):
    target = tmp_path / "x.py"
    target.write_text("print('hi')\n", encoding="utf-8")
    agent = _agent(tmp_path)
    assert asyncio.run(agent.read_file(str(target))) == "print('hi')\n"


def test_read_file_missing_file_message(tmp_path):
    agent = _agent(tmp_path)
    out = asyncio.run(agent.read_file(str(tmp_path / "nope.py")))
    assert out.startswith("File not found:")


def test_schemas_guide_file_vs_directory():
    from agent_core.tool_schemas import NLP_TOOL_SCHEMAS

    by_name = {s["function"]["name"]: s["function"] for s in NLP_TOOL_SCHEMAS}
    assert "not a directory" in by_name["read"]["description"].lower()
    assert "file or directory" in by_name["analyze"]["description"].lower()
