"""speculate REPL command — probabilistic deliberation exposed at the REPL.

TDD red->green: these tests were written (and watched fail with ImportError)
before ``agent_core/commands/speculate_cmd.py`` existed.

The command must wire the REAL pipeline
(``Orchestrator.dispatch_speculative`` -> ``ProbabilisticOrchestrator.
run_speculative`` -> ``decide_commitment``) so the previously programmatic-
only phase 3/4 machinery is reachable from the interactive command surface.
"""

import asyncio
import json


class FakeLLM:
    """Branch + judge responder.

    Branch prompts:   "Speculative branch {i}. {question}"
    Judge prompts:    "Quality judge: ... answer: {answer}"
    """

    model_name = "fake-model"

    async def chat(self, messages, tools=None, **kwargs):
        content = str(messages[-1]["content"])
        if content.startswith("Quality judge:"):
            if "good answer" in content:
                return "0.9"
            if "mediocre answer" in content:
                return "0.5"
            return "0.2"
        branch_id = int(content.split()[2].rstrip("."))
        if "[bad]" in content:
            return "bad answer %d" % branch_id
        if branch_id == 0:
            return "good answer %d" % branch_id
        return "mediocre answer %d" % branch_id


def _agent():
    return type("A", (), {"llm": FakeLLM(), "workspace": "C:/Dev/Agent1"})()


def _run(args, agent=None):
    from agent_core.commands.speculate_cmd import SpeculateCommand

    return asyncio.run(SpeculateCommand().execute(args, agent or _agent()))


def test_speculate_is_registered_for_dispatch():
    from agent import _build_registry

    assert "speculate" in _build_registry().names()


def test_speculate_commits_best_branch(capsys):
    # posix=False shlex keeps the literal quotes (REPL convention).
    assert _run(['"which branch is best?"']) is True
    out = capsys.readouterr().out
    assert "COMMIT" in out
    assert "good answer 0" in out


def test_speculate_refuses_when_all_below_threshold(capsys):
    # Every branch answers "[bad]" -> judge scores 0.2 < 0.7 -> REFUSE.
    assert _run(['"rate this [bad] idea"']) is True
    out = capsys.readouterr().out
    assert "REFUSE" in out
    assert "good answer" not in out


def test_speculate_rejects_threshold_out_of_range(capsys):
    assert _run(['"q"', "--threshold", "2"]) is True
    out = capsys.readouterr().out
    assert "threshold" in out
    assert "COMMIT" not in out
    assert "REFUSE" not in out


def test_speculate_requires_a_question(capsys):
    assert _run([]) is True
    out = capsys.readouterr().out
    assert "Usage: speculate" in out


def test_speculate_branches_receive_the_agent_system_prompt():
    """Regression: branches used to send only a user message, so they replied
    as a generic assistant ("I can't access your system") instead of as the
    agent.  Each branch must lead with the agent's real system prompt."""
    from agent_core.commands.speculate_cmd import SpeculateCommand

    seen = []

    class CaptureLLM:
        model_name = "fake-model"

        async def chat(self, messages, tools=None, **kwargs):
            seen.append(messages)
            content = str(messages[-1]["content"])
            return "0.9" if content.startswith("Quality judge:") else "branch answer"

    class SysAgent:
        llm = CaptureLLM()
        workspace = "C:/Dev/Agent1"

        def _refresh_system_message(self):
            pass

        @property
        def _chat_history(self):
            return [{"role": "system", "content": "SYSTEM-CONTEXT-XYZ"}]

    asyncio.run(SpeculateCommand().execute(['"q"'], SysAgent()))

    branch_calls = [m for m in seen if m and m[0]["role"] == "system"]
    assert branch_calls, "speculative branch did not receive a system message"
    assert branch_calls[0][0]["content"] == "SYSTEM-CONTEXT-XYZ"
    assert branch_calls[0][-1]["role"] == "user"  # question is the user turn


class _ToolLLM:
    """Branch LLM that asks for one tool, then answers; judge always 0.9."""

    model_name = "fake-model"

    def __init__(self, tool_name):
        self.tool_name = tool_name

    async def chat(self, messages, tools=None, **kwargs):
        content = str(messages[-1].get("content") or "")
        if content.startswith("Quality judge:"):
            return "0.9"
        if any(m.get("role") == "tool" for m in messages):
            return "grounded answer"
        return json.dumps({
            "content": "",
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {
                    "name": self.tool_name,
                    "arguments": json.dumps({"path": "x.py"}),
                },
            }],
        })


class _ToolAgent:
    def __init__(self, tool_name):
        self.llm = _ToolLLM(tool_name)
        self.executed = []

    def _refresh_system_message(self):
        pass

    @property
    def _chat_history(self):
        return [{"role": "system", "content": "SYS"}]

    async def _execute_tool_call(self, name, args):
        self.executed.append((name, args))
        return "file contents"


def test_speculate_branch_runs_read_only_tools():
    """A branch that asks for a read-only tool executes it via the agent and
    feeds the result back before answering (so answers can be grounded)."""
    from agent_core.commands.speculate_cmd import SpeculateCommand

    agent = _ToolAgent("read")
    asyncio.run(SpeculateCommand().execute(['"q"'], agent))
    assert agent.executed, "branch did not execute its read-only tool"
    assert all(name == "read" for name, _ in agent.executed)


def test_speculate_branch_refuses_mutating_tools():
    """A hallucinated write/edit/run must never reach the executor — branches
    are read-only, even though the agent's mode is build."""
    from agent_core.commands.speculate_cmd import SpeculateCommand

    agent = _ToolAgent("write")
    asyncio.run(SpeculateCommand().execute(['"q"'], agent))
    assert agent.executed == []  # allowlist blocked it before the executor


def test_looks_like_tool_call_detects_leaked_syntax():
    from agent_core.commands.speculate_cmd import _looks_like_tool_call

    assert _looks_like_tool_call('<|tool_call>call:run{command:"x"}<tool_call|>')
    assert _looks_like_tool_call('call:read{path:"a.py"}')
    assert _looks_like_tool_call('{"tool_calls": [{"id": "1"}]}')
    assert not _looks_like_tool_call("The repo is doing well - about 80/100.")
    assert not _looks_like_tool_call("")


def test_speculate_rejects_branch_that_emits_a_tool_call_as_text(capsys):
    """Regression: a model whose tool-call syntax leaks through as TEXT
    (observed with gemma: '<|tool_call>call:run{...}<tool_call|>') was scored
    1.0 by the judge and COMMITted as the final answer."""
    from agent_core.commands.speculate_cmd import SpeculateCommand

    class LeakyLLM:
        model_name = "fake-model"

        async def chat(self, messages, tools=None, **kwargs):
            content = str(messages[-1].get("content") or "")
            if content.startswith("Quality judge:"):
                return "1.0"  # even a perfect judge score must not save it
            return '<|tool_call>call:run{command:"pytest tests/"}<tool_call|>'

    agent = type("A", (), {"llm": LeakyLLM(), "workspace": "C:/Dev/Agent1"})()
    assert asyncio.run(SpeculateCommand().execute(['"q"'], agent)) is True
    out = capsys.readouterr().out
    assert "REFUSE" in out
    assert "COMMIT" not in out
    assert "<|tool_call>" not in out  # never presented as the answer


def test_provider_error_string_scores_zero():
    """Regression: LMStudioProvider.chat returns '[Error: HTTP Error 400: ...]'
    as plain text when the server has no model loaded.  _parse_score's
    first-number regex then extracted the '400' and clamped it to 1.0 —
    a failed branch was COMMITted as the answer (observed live against a
    llama-server on WSL Ubuntu 24.04 with no model loaded)."""
    from agent_core.commands.speculate_cmd import _is_provider_error, _parse_score

    err = ('[Error: HTTP Error 400: {\n'
           '    "error": {\n'
           '        "message": "No models loaded. Please load a model in the '
           'developer page or use the \'lms load\' command.",\n'
           '        "type": "invalid_request_error",\n'
           '        "param": "model",\n'
           '        "code": null\n'
           '    }\n'
           '}]')
    assert _is_provider_error(err)
    assert _parse_score(err) == 0.0  # was 1.0 before the fix

    # A normal judge reply still parses correctly.
    assert _parse_score("0.9") == 0.9
    assert _parse_score("0.5") == 0.5
    assert _parse_score("") == 0.0
    assert _parse_score("I think it is fine") == 0.0  # no number -> 0.0

    # A real answer is NOT a provider error.
    assert not _is_provider_error("The retry policy is 3 attempts.")
    assert not _is_provider_error("")


def test_speculate_refuses_when_provider_fails(capsys):
    """Regression: when the LLM returns a provider error string for every
    branch AND the judge, no branch should be COMMITted."""
    from agent_core.commands.speculate_cmd import SpeculateCommand

    err_str = ('[Error: HTTP Error 400: {\n'
               '    "error": {\n'
               '        "message": "No models loaded.",\n'
               '        "type": "invalid_request_error",\n'
               '        "param": "model",\n'
               '        "code": null\n'
               '    }\n'
               '}]')

    class BrokenLLM:
        model_name = "broken-model"

        async def chat(self, messages, tools=None, **kwargs):
            # Every call (branch AND judge) returns the same error string.
            return err_str

    agent = type("A", (), {"llm": BrokenLLM(), "workspace": "C:/Dev/Agent1"})()
    assert asyncio.run(SpeculateCommand().execute(['"q"'], agent)) is True
    out = capsys.readouterr().out
    assert "REFUSE" in out
    assert "COMMIT" not in out
    assert "[Error:" not in out  # error never presented as the answer
