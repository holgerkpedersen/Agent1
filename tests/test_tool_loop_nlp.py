"""Tests for the structured NLP tool-calling loop.

The conversational agent must actually EXECUTE tool calls — never just
describe them.  These tests drive ``ToolLoopRunner`` with a scripted fake
LLM that emits native ``tool_calls`` and verify that the dispatcher runs
the tools, feeds results back, and terminates on a plain-text answer.
"""
import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from agent_core.llm.tool_loop import ToolLoopRunner
from agent_core.tool_schemas import NLP_TOOL_NAMES, NLP_TOOL_SCHEMAS


class _ScriptedLLM:
    """A fake LLM that returns a fixed sequence of responses.

    Each entry is either a plain-text answer (loop terminates) or a
    ``(tool_name, args_dict)`` call that gets serialized to OpenAI-style
    ``tool_calls`` in the assistant message.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def chat(self, messages, tools=None, **kwargs):
        self.calls.append((list(messages), tools))
        step = self.script.pop(0)
        if isinstance(step, tuple):
            tool_name, args = step
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args),
                    },
                }],
            }
            return json.dumps(message)
        return step


def _make_llm_chat_fn(fake_llm):
    """Wrap a _ScriptedLLM the same way chat_nlp wraps the real provider."""
    async def llm_chat_fn(messages, tools):
        raw = await fake_llm.chat(messages, tools)
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict) and parsed.get("tool_calls"):
            parsed.pop("role", None)
            updated = list(messages)
            updated.append({"role": "assistant", "content": parsed.get("content") or "", **parsed})
            return str(parsed.get("content") or ""), updated
        updated = list(messages)
        updated.append({"role": "assistant", "content": raw})
        return raw, updated

    return llm_chat_fn


class TestToolLoopExecution:
    def test_executes_tools_and_terminates_on_text(self):
        fake = _ScriptedLLM([
            ("read", {"path": "agent.py"}),
            ("search", {"query": "def chat_nlp"}),
            "Summary of what I found.",
        ])
        executed = []

        async def execute_tool(name, args):
            executed.append((name, args))
            return f"result-of-{name}"

        runner = ToolLoopRunner(max_iterations=5)
        final_text, messages = _loop_runner_sync(runner, fake, execute_tool)

        assert final_text == "Summary of what I found."
        assert [e[0] for e in executed] == ["read", "search"]
        # Tool results must be fed back to the LLM as role:tool messages.
        assert sum(1 for m in messages if m["role"] == "tool") == 2

    def test_stops_when_no_tool_calls(self):
        fake = _ScriptedLLM(["Just an answer, no tools."])
        executed = []

        async def execute_tool(name, args):
            executed.append(name)
            return "x"

        runner = ToolLoopRunner(max_iterations=5)
        final_text, _ = _loop_runner_sync(runner, fake, execute_tool)

        assert final_text == "Just an answer, no tools."
        assert executed == []

    def test_tool_error_is_fed_back_not_crashing(self):
        fake = _ScriptedLLM([
            ("read", {"path": "missing.py"}),
            "I got an error.",
        ])
        executed = []

        async def execute_tool(name, args):
            executed.append(name)
            raise RuntimeError("boom")

        runner = ToolLoopRunner(max_iterations=5)
        final_text, messages = _loop_runner_sync(runner, fake, execute_tool)

        assert final_text == "I got an error."
        tool_msg = next(m for m in messages if m["role"] == "tool")
        assert "Tool error: boom" in tool_msg["content"]

    def test_cap_hit_while_tool_calls_pending_forces_final_synthesis(self):
        """If the model burns all iterations on tool calls, the loop must still
        produce a final text answer via a forced tool-less synthesis call."""
        fake = _ScriptedLLM([
            ("list_files", {"path": "."}),
            ("list_files", {"path": "."}),
            ("list_files", {"path": "."}),
            "Final synthesis: the project has these files.",
        ])
        executed = []

        async def execute_tool(name, args):
            executed.append(name)
            return "entries"

        runner = ToolLoopRunner(max_iterations=3)
        final_text, messages = _loop_runner_sync(runner, fake, execute_tool)

        assert final_text == "Final synthesis: the project has these files."
        # Consecutive identical calls are NOT re-executed — the first runs,
        # the two repeats get a steering note instead.
        assert executed == ["list_files"]
        # The forced call must run WITHOUT tools so a text answer is forced.
        assert fake.calls[-1][1] == []
        # The "no more tools" steering note must not leak into the history
        # that a follow-up turn would continue from.
        assert not any(
            m.get("role") == "system" and "no more tools" in str(m.get("content"))
            for m in messages
        )

    def test_consecutive_identical_calls_get_note_not_repeat_execution(self):
        """A repeated identical call must not silently re-execute; the model
        must be told it already got that result."""
        fake = _ScriptedLLM([
            ("search", {"query": "_execute_nlp_tool", "path": "."}),
            ("search", {"query": "_execute_nlp_tool", "path": "."}),
            "The symbol does not exist in current code.",
        ])
        executed = []

        async def execute_tool(name, args):
            executed.append(name)
            return "only state files match"

        runner = ToolLoopRunner(max_iterations=5)
        final_text, messages = _loop_runner_sync(runner, fake, execute_tool)

        assert executed == ["search"]
        assert final_text == "The symbol does not exist in current code."
        notes = [m["content"] for m in messages if m["role"] == "tool"]
        assert len(notes) == 2
        assert "NOTE: This exact call was just executed" in notes[1]

    def test_non_consecutive_identical_call_still_executes(self):
        """A legit re-read (read -> edit -> read) must still work."""
        fake = _ScriptedLLM([
            ("read", {"path": "f.py"}),
            ("edit", {"path": "f.py", "old_text": "a", "new_text": "b"}),
            ("read", {"path": "f.py"}),
            "Verified.",
        ])
        executed = []

        async def execute_tool(name, args):
            executed.append(name)
            return f"result-{name}"

        runner = ToolLoopRunner(max_iterations=5)
        final_text, messages = _loop_runner_sync(runner, fake, execute_tool)

        assert executed == ["read", "edit", "read"]
        assert final_text == "Verified."

    def test_third_consecutive_duplicate_forces_synthesis_immediately(self):
        """A model stuck on the same call must be stopped after the third
        repeat and forced to answer — not left to burn the iteration budget.
        The scripted LLM raises IndexError if the loop asks for one more
        response after the forced call, proving early termination."""
        fake = _ScriptedLLM([
            ("read", {"path": "agent.py"}),
            ("read", {"path": "agent.py"}),
            ("read", {"path": "agent.py"}),
            "I found the implementation in agent.py.",
        ])
        executed = []

        async def execute_tool(name, args):
            executed.append(name)
            return "content..."

        runner = ToolLoopRunner(max_iterations=20)
        final_text, messages = _loop_runner_sync(runner, fake, execute_tool)

        assert executed == ["read"]
        assert final_text == "I found the implementation in agent.py."
        # The forced call runs without tools and only 4 LLM calls happened
        # (3 loop iterations + 1 forced synthesis) despite max_iterations=20.
        assert len(fake.calls) == 4
        assert fake.calls[-1][1] == []
        tool_msgs = [m["content"] for m in messages if m["role"] == "tool"]
        assert any("three times" in m for m in tool_msgs)

    def test_stuck_synthesis_note_does_not_leak_into_history(self):
        fake = _ScriptedLLM([
            ("read", {"path": "a.py"}),
            ("read", {"path": "a.py"}),
            ("read", {"path": "a.py"}),
            "Answer.",
        ])
        executed = []

        async def execute_tool(name, args):
            executed.append(name)
            return "x"

        runner = ToolLoopRunner(max_iterations=20)
        final_text, messages = _loop_runner_sync(runner, fake, execute_tool)

        assert final_text == "Answer."
        assert not any(
            m.get("role") == "system" and "repeated the same tool call" in str(m.get("content"))
            for m in messages
        )

    def test_deadline_note_is_injected_before_cap(self):
        """A budget warning must reach the LLM before the cap hits — but must
        NOT leak into the persisted history (a fresh turn has a fresh budget)."""
        fake = _ScriptedLLM([
            ("list_files", {"path": "."}),
            "Answer before the cap.",
        ])
        executed = []

        async def execute_tool(name, args):
            executed.append(name)
            return "entries"

        runner = ToolLoopRunner(max_iterations=3, deadline_window=2)
        final_text, messages = _loop_runner_sync(runner, fake, execute_tool)

        sent_notes = [
            m["content"] for call in fake.calls for m in call[0]
            if m.get("role") == "system" and "BUDGET WARNING" in str(m.get("content"))
        ]
        assert len(sent_notes) == 1
        assert "1 tool call" in sent_notes[0] or "tool call" in sent_notes[0]
        # The steering note must not persist into the returned history.
        assert not any(
            m.get("role") == "system" and "BUDGET WARNING" in str(m.get("content"))
            for m in messages
        )

    def test_no_deadline_note_when_model_answers_early(self):
        fake = _ScriptedLLM(["Just an answer, no tools."])
        executed = []

        async def execute_tool(name, args):
            executed.append(name)
            return "x"

        runner = ToolLoopRunner(max_iterations=5, deadline_window=2)
        final_text, messages = _loop_runner_sync(runner, fake, execute_tool)

        assert final_text == "Just an answer, no tools."
        notes = [
            m["content"] for m in messages
            if m["role"] == "system" and "BUDGET WARNING" in m["content"]
        ]
        assert notes == []


def _loop_runner_sync(runner, fake_llm, execute_tool):
    import asyncio
    return asyncio.run(runner.run(
        messages=[{"role": "user", "content": "do the thing"}],
        llm_chat_fn=_make_llm_chat_fn(fake_llm),
        execute_tool_fn=execute_tool,
        tools=list(NLP_TOOL_SCHEMAS),
    ))


class TestNlpToolSchemas:
    def test_all_schema_names_are_valid_identifiers(self):
        for name in NLP_TOOL_NAMES:
            assert name.isidentifier(), name

    def test_every_schema_has_name_in_names_set(self):
        for schema in NLP_TOOL_SCHEMAS:
            assert schema["function"]["name"] in NLP_TOOL_NAMES


class TestAgentExecuteToolCall:
    """Drive Agent._execute_tool_call with a real temp file so the
    verification summary is produced for write/edit."""

    def test_write_verifies_python_file(self, tmp_path):
        import asyncio
        from agent import Agent
        agent = Agent(workspace=str(tmp_path))
        target = tmp_path / "hello.py"

        async def run():
            return await agent._execute_tool_call("write", {
                "path": str(target),
                "content": "def hello() -> str:\n    return 'hi'\n",
            })

        result = asyncio.run(run())
        assert "Written" in result
        assert "[verify] py_compile" in result
        assert target.read_text(encoding="utf-8").startswith("def hello")

    def test_read_supports_offset_pagination(self, tmp_path):
        """read must page through large files with offset/limit instead of
        re-returning the same first chunk (the cause of repeated read loops)."""
        import asyncio
        from agent import Agent
        agent = Agent(workspace=str(tmp_path))
        target = tmp_path / "big.py"
        target.write_text("line0\n" + "x" * 6000, encoding="utf-8")

        async def first_page():
            return await agent._execute_tool_call("read", {"path": str(target)})

        async def second_page():
            return await agent._execute_tool_call(
                "read", {"path": str(target), "offset": 5000},
            )

        page1 = asyncio.run(first_page())
        assert "line0" in page1
        assert "use read with offset=5000" in page1
        page2 = asyncio.run(second_page())
        assert "x" in page2
        assert "offset=5000" not in page2

    def test_read_offset_beyond_end(self, tmp_path):
        import asyncio
        from agent import Agent
        agent = Agent(workspace=str(tmp_path))
        target = tmp_path / "small.py"
        target.write_text("abc", encoding="utf-8")

        async def run():
            return await agent._execute_tool_call(
                "read", {"path": str(target), "offset": 99},
            )

        assert "beyond the end" in asyncio.run(run())

    def test_edit_verifies_and_replaces_first_occurrence(self, tmp_path):
        import asyncio
        from agent import Agent
        agent = Agent(workspace=str(tmp_path))
        target = tmp_path / "t.py"
        target.write_text("x = 1\nx = 2\n", encoding="utf-8")

        async def run():
            return await agent._execute_tool_call("edit", {
                "path": str(target),
                "old_text": "x = 1",
                "new_text": "x = 10",
            })

        result = asyncio.run(run())
        assert "Edited" in result
        assert "[verify] py_compile" in result
        assert target.read_text(encoding="utf-8") == "x = 10\nx = 2\n"

    def test_unknown_tool_returns_error(self):
        import asyncio
        from agent import Agent
        agent = Agent(workspace=".")

        async def run():
            return await agent._execute_tool_call("no_such_tool", {})

        assert "Unknown tool" in asyncio.run(run())

    def test_git_bash_workspace_is_normalized(self, tmp_path):
        """A Git-Bash /c/... workspace must not break subprocess-based tools
        (regression: WinError 267 'directory name is invalid' on git/run/diff).
        """
        import asyncio
        from pathlib import Path
        from agent import Agent
        repo = Path(tmp_path)
        (repo / ".git").mkdir()
        (repo / "file.txt").write_text("hello\n", encoding="utf-8")
        bash_path = "/c" + str(repo)[2:].replace("\\", "/")

        agent = Agent(workspace=bash_path)
        assert os.path.isdir(agent.workspace)
        assert agent.workspace == os.path.normpath(str(repo))

        async def run():
            return await agent._execute_tool_call("git", {"subcommand": "status"})

        result = asyncio.run(run())
        assert "Git error" not in result

    def test_git_bash_override_in_nlp_workspace_is_normalized(self, tmp_path):
        """A bad _nlp_workspace override (paste --workspace) must fall back to a
        valid cwd instead of failing subprocess tools."""
        import asyncio
        from agent import Agent
        agent = Agent(workspace=".")
        agent._nlp_workspace = "/c/definitely/not/a/dir"

        async def run():
            return await agent._execute_tool_call("git", {"subcommand": "status"})

        result = asyncio.run(run())
        assert "Git error" not in result


class TestNoShellInjection:
    """The NLP subprocess tools must not let model-supplied text reach a shell.

    Regression: git/diff/tests built shell strings with shell=True, so an
    argument like ``--oneline && echo PWNED`` would have executed both parts.
    """

    @staticmethod
    def _echo_ran(result: str) -> bool:
        """True only if `echo PWNED` actually executed (standalone line)."""
        return any(line.strip() == "PWNED" for line in result.splitlines())

    def test_git_args_cannot_chain_shell_commands(self, tmp_path):
        import asyncio
        import subprocess
        from agent import Agent
        repo = tmp_path
        subprocess.run(["git", "init", "-q", str(repo)], check=False)
        (repo / "f.txt").write_text("x\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "f.txt"], check=False)
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-qm", "init"], check=False,
        )
        agent = Agent(workspace=str(repo))

        async def run():
            return await agent._execute_tool_call(
                "git", {"subcommand": "log", "args": "--oneline && echo PWNED"},
            )

        result = asyncio.run(run())
        assert not self._echo_ran(result)

    def test_tests_path_cannot_chain_shell_commands(self, tmp_path):
        import asyncio
        from agent import Agent
        agent = Agent(workspace=str(tmp_path))

        async def run():
            return await agent._execute_tool_call(
                "tests", {"path": "x; echo PWNED", "framework": "pytest"},
            )

        result = asyncio.run(run())
        assert not self._echo_ran(result)

    def test_diff_path_cannot_chain_shell_commands(self, tmp_path):
        import asyncio
        from agent import Agent
        repo = tmp_path
        (repo / "a.txt").write_text("a\n", encoding="utf-8")
        (repo / "b.txt").write_text("b\n", encoding="utf-8")
        agent = Agent(workspace=str(repo))

        async def run():
            return await agent._execute_tool_call(
                "diff", {"file1": 'a.txt" && echo PWNED && "'},
            )

        result = asyncio.run(run())
        assert not self._echo_ran(result)

    def test_run_tool_blocks_destructive_patterns(self):
        import asyncio
        from agent import Agent
        agent = Agent(workspace=".")

        async def run(cmd):
            return await agent._execute_tool_call("run", {"command": cmd})

        for cmd in [
            "del /s /q C:\\x",
            "rm -rf /",
            "rmdir /s C:\\x",
            "FORMAT D:",
            "shutdown /s",
            "diskpart",
            "powershell -Command Remove-Item -Recurse C:\\x",
        ]:
            result = asyncio.run(run(cmd))
            assert "blocked" in result.lower(), cmd

    def test_run_tool_still_executes_harmless_commands(self):
        import asyncio
        from agent import Agent
        agent = Agent(workspace=".")

        async def run():
            return await agent._execute_tool_call("run", {"command": "echo hello-ok"})

        assert "hello-ok" in asyncio.run(run())


class TestPersistentChatHistory:
    """The NLP conversation must survive REPL restarts via chat_history.json."""

    def test_loads_history_from_previous_session(self, tmp_path):
        from unittest.mock import patch
        from agent import Agent
        history_file = tmp_path / "chat_history.json"
        history_file.write_text(
            json.dumps([
                {"role": "user", "content": "Analyser projektet"},
                {"role": "assistant", "content": "Jeg fandt 5 ting."},
            ]),
            encoding="utf-8",
        )
        with patch("agent.CHAT_HISTORY_JSON_PATH", str(history_file)):
            agent = Agent(workspace=".")
            assert len(agent._chat_history) == 2
            assert agent._chat_history[0]["content"] == "Analyser projektet"

    def test_corrupt_history_file_falls_back_to_empty(self, tmp_path):
        from unittest.mock import patch
        from agent import Agent
        history_file = tmp_path / "chat_history.json"
        history_file.write_text("{not valid json", encoding="utf-8")
        with patch("agent.CHAT_HISTORY_JSON_PATH", str(history_file)):
            agent = Agent(workspace=".")
            assert agent._chat_history == []

    def test_saves_history_after_nlp_turn(self, tmp_path):
        import asyncio
        from unittest.mock import patch
        from agent import Agent
        history_file = tmp_path / "chat_history.json"

        class FakeLLM:
            async def chat(self, messages, tools=None, **kwargs):
                return "Summary of the analysis."

        with patch("agent.CHAT_HISTORY_JSON_PATH", str(history_file)):
            agent = Agent(workspace=".")
            agent.llm = FakeLLM()

            async def run():
                await agent.chat_nlp("Analyser projektet")

            asyncio.run(run())

            assert history_file.exists()
            saved = json.loads(history_file.read_text(encoding="utf-8"))
            roles = [m["role"] for m in saved]
            assert roles == ["system", "user", "assistant"]
            assert saved[1]["content"] == "Analyser projektet"
            assert saved[2]["content"] == "Summary of the analysis."

    def test_trim_keeps_system_prompt_and_tail(self, tmp_path):
        import asyncio
        from unittest.mock import patch
        from agent import Agent, _trim_chat_history
        history_file = tmp_path / "chat_history.json"
        messages = [{"role": "system", "content": "SYS"}]
        messages += [{"role": "user", "content": f"msg-{i}"} for i in range(80)]

        trimmed = _trim_chat_history(messages)
        assert len(trimmed) == 60
        assert trimmed[0]["content"] == "SYS"
        assert trimmed[-1]["content"] == "msg-79"

        class FakeLLM:
            async def chat(self, messages, tools=None, **kwargs):
                return "done"

        with patch("agent.CHAT_HISTORY_JSON_PATH", str(history_file)):
            agent = Agent(workspace=".")
            agent._chat_history = list(messages)
            agent.llm = FakeLLM()

            async def run():
                await agent.chat_nlp("hello")

            asyncio.run(run())
            # The SAVED history is projected: only the last exchange survives.
            saved = json.loads(history_file.read_text(encoding="utf-8"))
            assert saved[0]["content"] == "SYS"
            assert saved[1]["content"] == "hello"
            assert saved[2]["content"] == "done"

    def test_projection_keeps_only_last_exchange(self):
        from agent import _project_chat_history
        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "gammel opgave"},
            {"role": "assistant", "content": "gammelt svar"},
            {"role": "user", "content": "ny opgave"},
            {"role": "assistant", "content": "nyt svar"},
        ]
        projected = _project_chat_history(messages)
        roles = [m["content"] for m in projected]
        assert roles == ["SYS", "ny opgave", "nyt svar"]

    def test_projection_drops_steering_notes_and_empty_placeholders(self):
        from agent import _project_chat_history
        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "opgave"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
            {"role": "tool", "tool_call_id": "t1", "content": "resultat"},
            {"role": "tool", "tool_call_id": "t2", "content": "NOTE: This exact call was just executed ..."},
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "endeligt svar"},
        ]
        projected = _project_chat_history(messages)
        assert [m["content"] for m in projected if m["content"]] == [
            "SYS", "opgave", "resultat", "endeligt svar",
        ]
        # The empty assistant WITH tool_calls stays (it pairs with the result)…
        tool_calls = [m for m in projected if m.get("tool_calls")]
        assert len(tool_calls) == 1

    def test_clear_history_deletes_file(self, tmp_path):
        from unittest.mock import patch
        from agent import Agent
        history_file = tmp_path / "chat_history.json"
        history_file.write_text(
            json.dumps([{"role": "user", "content": "hi"}]),
            encoding="utf-8",
        )
        with patch("agent.CHAT_HISTORY_JSON_PATH", str(history_file)):
            agent = Agent(workspace=".")
            assert agent._chat_history
            agent.clear_history()
            assert agent._chat_history == []
            assert not history_file.exists()
