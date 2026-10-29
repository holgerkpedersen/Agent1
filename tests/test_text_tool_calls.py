"""Text-form tool-call recovery (Qwen3-Coder XML emitted as plain text).

Regression: Qwen3.8-Flash-Next served through Strata emitted

    <tool_call>
    <function=grep>
    <parameter=pattern>
    class Command
    </parameter>
    </function>
    </tool_call>

as the assistant CONTENT because ``grep`` was not a declared tool.  The harness
treated it as the final answer and the REPL printed the XML.  These tests pin
the recovery parser and the main tool-loop wiring that executes the call.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_core.llm.tool_reply import recover_text_tool_calls


OBSERVED = (
    "<tool_call>\n"
    "<function=grep>\n"
    "<parameter=glob>\n*.py\n</parameter>\n"
    "<parameter=head_limit>\n30\n</parameter>\n"
    "<parameter=output_mode>\ncontent\n</parameter>\n"
    "<parameter=path>\nagent_core\n</parameter>\n"
    "<parameter=pattern>\nclass Command\n</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


def _args(call: dict) -> dict:
    return json.loads(call["function"]["arguments"])


class TestRecoverTextToolCalls:
    def test_observed_qwen_xml_call_is_recovered_and_aliased(self) -> None:
        text, calls = recover_text_tool_calls(OBSERVED)
        assert text == ""
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "search"
        assert _args(calls[0]) == {
            "query": "class Command",
            "path": "agent_core",
        }

    def test_surrounding_prose_is_kept(self) -> None:
        text, calls = recover_text_tool_calls(
            "Let me check the class.\n" + OBSERVED + "\nThen I will report."
        )
        assert text == "Let me check the class.\n\nThen I will report."
        assert len(calls) == 1

    def test_multiple_wrapped_calls_in_one_message(self) -> None:
        text, calls = recover_text_tool_calls(
            "<tool_call><function=read><parameter=path>a.py</parameter>"
            "</function></tool_call>\n"
            "<tool_call><function=search><parameter=query>x</parameter>"
            "</function></tool_call>"
        )
        assert text == ""
        assert [c["function"]["name"] for c in calls] == ["read", "search"]
        assert _args(calls[0]) == {"path": "a.py"}
        assert _args(calls[1]) == {"query": "x"}
        assert len({c["id"] for c in calls}) == 2

    def test_parameter_values_follow_schema_types(self) -> None:
        _, calls = recover_text_tool_calls(
            "<tool_call><function=read>"
            "<parameter=path>a.py</parameter>"
            "<parameter=offset>10</parameter>"
            "<parameter=limit>2</parameter>"
            "</function></tool_call>"
        )
        assert _args(calls[0]) == {"path": "a.py", "offset": 10, "limit": 2}

    def test_boolean_parameter_is_typed(self) -> None:
        _, calls = recover_text_tool_calls(
            "<tool_call><function=plan_start>"
            "<parameter=dry_run>false</parameter>"
            "</function></tool_call>"
        )
        assert _args(calls[0]) == {"dry_run": False}

    def test_string_parameter_is_never_coerced(self) -> None:
        """A write of the literal text ``true`` must stay text, not become a bool."""
        _, calls = recover_text_tool_calls(
            "<tool_call><function=write>"
            "<parameter=path>a.txt</parameter>"
            "<parameter=content>true</parameter>"
            "</function></tool_call>"
        )
        assert _args(calls[0]) == {"path": "a.txt", "content": "true"}

    def test_array_parameter_is_decoded(self) -> None:
        _, calls = recover_text_tool_calls(
            "<tool_call><function=read>"
            '<parameter=paths>["a.py", "b.py"]</parameter>'
            "</function></tool_call>"
        )
        assert _args(calls[0]) == {"paths": ["a.py", "b.py"]}

    def test_json_body_inside_wrapper_is_supported(self) -> None:
        _, calls = recover_text_tool_calls(
            '<tool_call>{"name": "search", "arguments": {"query": "x"}}'
            "</tool_call>"
        )
        assert calls[0]["function"]["name"] == "search"
        assert _args(calls[0]) == {"query": "x"}

    def test_bare_function_block_without_wrapper(self) -> None:
        _, calls = recover_text_tool_calls(
            "<function=read><parameter=path>a.py</parameter></function>"
        )
        assert calls[0]["function"]["name"] == "read"
        assert _args(calls[0]) == {"path": "a.py"}

    def test_bash_alias_maps_to_run(self) -> None:
        _, calls = recover_text_tool_calls(
            "<tool_call><function=bash>"
            "<parameter=command>pytest -q</parameter>"
            "</function></tool_call>"
        )
        assert calls[0]["function"]["name"] == "run"
        assert _args(calls[0]) == {"command": "pytest -q"}

    def test_unknown_tool_name_is_not_renamed(self) -> None:
        _, calls = recover_text_tool_calls(
            "<tool_call><function=mystery><parameter=x>1</parameter>"
            "</function></tool_call>"
        )
        assert calls[0]["function"]["name"] == "mystery"

    def test_incomplete_block_is_left_untouched(self) -> None:
        raw = "<tool_call>\n<function=read>\n<parameter=path>a.py"
        text, calls = recover_text_tool_calls(raw)
        assert text == raw
        assert calls == []

    def test_plain_prose_and_quoted_syntax_are_not_executed(self) -> None:
        raw = "The docs describe the `<function=read>` tag without arguments."
        text, calls = recover_text_tool_calls(raw)
        assert text == raw
        assert calls == []

    def test_plain_text_is_unchanged(self) -> None:
        raw = "The class lives in agent_core/commands/base.py:12."
        assert recover_text_tool_calls(raw) == (raw, [])


class _ScriptedLLM:
    """Provider double: returns scripted replies, records requests."""

    model_name = "llama/fake"
    _provider = None
    _profile_name = ""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[list[dict]] = []

    async def chat(self, messages, tools=None, disable_thinking=False, **kwargs):
        self.requests.append(list(messages))
        return self.replies.pop(0)


class _FakeAgent:
    def __init__(self, llm: _ScriptedLLM) -> None:
        self.llm = llm
        self.mode = "build"
        self.model_name = llm.model_name
        self.tool_calls: list[tuple[str, dict]] = []
        self._turn_run_id = None
        self._active_trace_writer = None
        self._pending_effects = None

    async def _execute_tool_call(self, name: str, args: dict) -> str:
        self.tool_calls.append((name, args))
        return "agent_core/commands/base.py:12:class Command"

    def _take_trace_effects(self):
        return []

    def _record_llm_experience(self, **kwargs) -> None:
        return None


class TestParseToolReply:
    def test_structured_json_envelope_still_parses(self) -> None:
        from agent_core.llm.tool_reply import parse_tool_reply

        raw = json.dumps(
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": '{"path": "a.py"}',
                        },
                    }
                ],
            }
        )
        text, messages = parse_tool_reply(
            raw, [{"role": "user", "content": "x"}]
        )
        assert text == ""
        assert messages[-1]["tool_calls"][0]["function"]["name"] == "read"

    def test_malformed_arguments_are_repaired_when_hook_given(self) -> None:
        from agent_core.llm.tool_reply import parse_tool_reply

        raw = json.dumps(
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": '{"path": a.py}',
                        },
                    }
                ],
            }
        )
        _, messages = parse_tool_reply(
            raw,
            [{"role": "user", "content": "x"}],
            repair_arguments=lambda _raw: {"path": "a.py"},
        )
        assert json.loads(
            messages[-1]["tool_calls"][0]["function"]["arguments"]
        ) == {"path": "a.py"}

    def test_text_form_call_is_recovered(self) -> None:
        from agent_core.llm.tool_reply import parse_tool_reply

        text, messages = parse_tool_reply(
            OBSERVED, [{"role": "user", "content": "x"}]
        )
        assert text == ""
        assert messages[-1]["tool_calls"][0]["function"]["name"] == "search"


class TestTurnLoopWiring:
    def test_text_form_call_is_executed_instead_of_answered(self) -> None:
        from agent_core.config import AgentDisplayMode
        from agent_core.turn_loop import _run_chained_tool_loop

        llm = _ScriptedLLM([OBSERVED, "The class lives in base.py."])
        agent = _FakeAgent(llm)
        final_text, messages, llm_error, loop = asyncio.run(
            _run_chained_tool_loop(
                agent,  # type: ignore[arg-type]
                user_input="where is class Command defined?",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "where is class Command defined?"},
                ],
                display_mode=AgentDisplayMode.QUIET,
                seen_calls={},
            )
        )

        assert llm_error is None
        assert agent.tool_calls == [
            ("search", {"query": "class Command", "path": "agent_core"})
        ]
        assert final_text == "The class lives in base.py."
        assert loop.tool_calls_made == 1
        assert loop.termination_reason == "answer"
        assert not any("<tool_call>" in str(m.get("content")) for m in messages)
        assistant = next(
            m
            for m in messages
            if m.get("role") == "assistant" and m.get("tool_calls")
        )
        assert assistant["tool_calls"][0]["function"]["name"] == "search"

    def test_plain_text_answer_still_passes_through(self) -> None:
        from agent_core.config import AgentDisplayMode
        from agent_core.turn_loop import _run_chained_tool_loop

        llm = _ScriptedLLM(["All done."])
        agent = _FakeAgent(llm)
        final_text, _, llm_error, loop = asyncio.run(
            _run_chained_tool_loop(
                agent,  # type: ignore[arg-type]
                user_input="say done",
                messages=[{"role": "user", "content": "say done"}],
                display_mode=AgentDisplayMode.QUIET,
                seen_calls={},
            )
        )
        assert llm_error is None
        assert final_text == "All done."
        assert agent.tool_calls == []
        assert loop.tool_calls_made == 0


class TestParallelWiring:
    def test_parallel_chat_fn_recovers_text_form_call(self) -> None:
        from agent_core.llm.parallel import _make_tool_llm_chat_fn

        class _Provider:
            async def chat(self, messages, tools=None, max_tokens=None, **kwargs):
                return OBSERVED

        fn = _make_tool_llm_chat_fn(_Provider(), None, True)
        text, messages = asyncio.run(
            fn([{"role": "user", "content": "find it"}], [])
        )
        assert text == ""
        assert messages[-1]["tool_calls"][0]["function"]["name"] == "search"
        assert json.loads(
            messages[-1]["tool_calls"][0]["function"]["arguments"]
        ) == {"query": "class Command", "path": "agent_core"}


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
