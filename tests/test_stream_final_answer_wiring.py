"""Tests: the streamed final answer is wired in, opt-in and capability-gated.

``tests/test_stream_final_answer.py`` covers the RUNNER contract (a ``stream_fn``
serves the forced-synthesis call and ``final_answer_streamed`` is reported).  This
file covers the CALLER side, where the dangerous part lives:

1. ``_resolve_stream_fn`` returns None unless the user opted in
   (``AGENT_STREAM_FINAL_ANSWER``, default off) — an existing run must stay
   byte-identical by default;
2. it returns None on a provider whose ``chat_stream`` prints NOTHING
   (Opencode/OpenRouter/Lemonade are a bare ``return await self.chat(...)``).
   This is the regression that matters: on those providers the returned text has
   not appeared on screen, so marking it "already streamed" would make
   ``_finish_turn`` skip its print and show the user NO answer at all;
3. a ``FailoverProvider`` is judged by the provider that actually serves the
   stream (its ``chat_stream`` delegates to ``_providers[0]`` only);
4. ``_finish_turn`` prints the answer exactly once — suppressed only when the
   loop reports it was already streamed, and still printed for a loop-like
   object that lacks the flag.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import agent as agent_mod
from agent_core.config import AgentDisplayMode, AgentSettings, load_agent_settings
from agent_core.llm import provider as provider_mod
from agent_core.llm.lemonade_provider import LemonadeProvider
from agent_core.llm.llama_provider import LlamaProvider
from agent_core.llm.lmstudio import LMStudioProvider
from agent_core.llm.opencode_provider import OpencodeProvider
from agent_core.llm.openrouter_provider import OpenRouterProvider
from agent_core.llm.provider import FailoverProvider


def _bare(cls):
    """An instance of *cls* without running ``__init__`` (no server needed)."""
    return object.__new__(cls)


# ---------------------------------------------------------------------------
# 1. provider_supports_streaming — only the providers that really print
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "provider, expected",
    [
        (_bare(LMStudioProvider), True),
        (_bare(LlamaProvider), True),
        (_bare(OpencodeProvider), False),
        (_bare(OpenRouterProvider), False),
        (_bare(LemonadeProvider), False),
    ],
    ids=lambda v: type(v).__name__ if not isinstance(v, bool) else str(v),
)
def test_provider_supports_streaming(provider, expected: bool) -> None:
    assert provider_mod.provider_supports_streaming(provider) is expected


def test_non_streaming_providers_really_print_nothing() -> None:
    """Guard the assumption behind the gate: those three chat_stream impls are
    a bare delegation to chat, so nothing reaches the console."""
    source = (
        Path(provider_mod.__file__).parent / "opencode_provider.py"
    ).read_text(encoding="utf-8")
    assert "No streaming support" in source


def test_failover_provider_judged_by_the_provider_that_serves_it() -> None:
    """FailoverProvider.chat_stream delegates to _providers[0] only — no
    failover — so that provider's capability is the one that counts."""
    fo = _bare(FailoverProvider)
    fo._providers = [_bare(OpencodeProvider), _bare(LMStudioProvider)]
    assert provider_mod.provider_supports_streaming(fo) is False, (
        "first provider prints nothing, so the stream shows the user nothing"
    )
    fo._providers = [_bare(LMStudioProvider), _bare(OpencodeProvider)]
    assert provider_mod.provider_supports_streaming(fo) is True


def test_unknown_provider_object_is_not_streaming() -> None:
    assert provider_mod.provider_supports_streaming(object()) is False


# ---------------------------------------------------------------------------
# 2. _resolve_stream_fn — opt-in AND capability, never raises
# ---------------------------------------------------------------------------

def _llm_with(provider):
    """A minimal llm stand-in exposing the two attributes the resolver reads."""
    class _LLM:
        _provider = provider

        async def chat_stream(self, messages):
            return "streamed"

    return _LLM()


def _settings(**overrides) -> AgentSettings:
    """Real settings with streaming forced on/off."""
    base = load_agent_settings()
    return AgentSettings(
        workspace_root=base.workspace_root,
        llm_provider=base.llm_provider,
        llm_providers=base.llm_providers,
        stream_final_answer=overrides.get("stream_final_answer", False),
    )


def test_resolver_returns_none_when_setting_is_off(monkeypatch) -> None:
    """Default: OFF -> no streaming, so behaviour is byte-identical."""
    monkeypatch.setattr(
        agent_mod, "load_agent_settings",
        lambda: _settings(stream_final_answer=False),
    )
    llm = _llm_with(_bare(LMStudioProvider))
    assert agent_mod._resolve_stream_fn(llm) is None


def test_resolver_returns_callable_when_opted_in_and_capable(monkeypatch) -> None:
    monkeypatch.setattr(
        agent_mod, "load_agent_settings",
        lambda: _settings(stream_final_answer=True),
    )
    llm = _llm_with(_bare(LMStudioProvider))
    stream_fn = agent_mod._resolve_stream_fn(llm)
    assert stream_fn is not None
    assert asyncio.run(stream_fn([])) == "streamed"


@pytest.mark.parametrize(
    "provider",
    [_bare(OpencodeProvider), _bare(OpenRouterProvider), _bare(LemonadeProvider)],
    ids=lambda p: type(p).__name__,
)
def test_resolver_returns_none_on_non_streaming_provider(
    monkeypatch, provider
) -> None:
    """THE regression guard: opted in, but the provider prints nothing — so the
    caller must keep printing the answer itself (otherwise: no answer at all)."""
    monkeypatch.setattr(
        agent_mod, "load_agent_settings",
        lambda: _settings(stream_final_answer=True),
    )
    assert agent_mod._resolve_stream_fn(_llm_with(provider)) is None


def test_resolver_survives_settings_failure(monkeypatch) -> None:
    """A broken settings load means 'no streaming', never a crashed turn."""
    def _boom():
        raise RuntimeError("settings exploded")

    monkeypatch.setattr(agent_mod, "load_agent_settings", _boom)
    assert agent_mod._resolve_stream_fn(_llm_with(_bare(LMStudioProvider))) is None


def test_resolver_handles_llm_without_provider_attribute(monkeypatch) -> None:
    monkeypatch.setattr(
        agent_mod, "load_agent_settings",
        lambda: _settings(stream_final_answer=True),
    )

    class _NoProvider:
        async def chat_stream(self, messages):
            return "x"

    assert agent_mod._resolve_stream_fn(_NoProvider()) is None


# ---------------------------------------------------------------------------
# 3. _finish_turn prints the answer exactly once
# ---------------------------------------------------------------------------

class _Loop:
    """Minimal loop stand-in carrying only the streaming flag."""

    def __init__(self, streamed: bool) -> None:
        self.final_answer_streamed = streamed


def _make_agent(tmp_path: Path, monkeypatch) -> "agent_mod.Agent":
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        agent_mod, "CHAT_HISTORY_JSON_PATH", str(tmp_path / "chat_history.json")
    )
    monkeypatch.setattr(
        agent_mod, "AGENT_MEMORY_JSON_PATH", str(tmp_path / "agent_memory.json")
    )
    monkeypatch.setattr(agent_mod, "TURN_LOG_PATH", str(tmp_path / "turn_log.jsonl"))
    return agent_mod.Agent(workspace=str(workspace))


def test_finish_turn_prints_answer_when_not_streamed(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    bot = _make_agent(tmp_path, monkeypatch)
    bot._last_user_input = "hi"
    bot._finish_turn("The answer.", None, _Loop(streamed=False),
                     AgentDisplayMode.QUIET)
    assert "The answer." in capsys.readouterr().out


def test_finish_turn_skips_print_when_already_streamed(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Already on screen token-by-token -> printing again would duplicate it."""
    bot = _make_agent(tmp_path, monkeypatch)
    bot._last_user_input = "hi"
    bot._finish_turn("The answer.", None, _Loop(streamed=True),
                     AgentDisplayMode.QUIET)
    assert "The answer." not in capsys.readouterr().out


def test_finish_turn_prints_for_loop_without_the_flag(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Existing callers pass a bare object() as the loop — that must still
    print (absent flag means 'not streamed'), never raise AttributeError."""
    bot = _make_agent(tmp_path, monkeypatch)
    bot._last_user_input = "hi"
    bot._finish_turn("Still printed.", None, object(), AgentDisplayMode.QUIET)
    assert "Still printed." in capsys.readouterr().out


def test_finish_turn_llm_error_takes_precedence_over_streamed_flag(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """An error is reported as an error even if a flag claims it streamed."""
    bot = _make_agent(tmp_path, monkeypatch)
    bot._last_user_input = "hi"
    bot._finish_turn("Should not appear.", "boom", _Loop(streamed=True),
                     AgentDisplayMode.QUIET)
    out = capsys.readouterr().out
    assert "boom" in out
    assert "Should not appear." not in out


# ---------------------------------------------------------------------------
# 4. end-to-end: chat_nlp really streams, and the answer is shown ONCE
# ---------------------------------------------------------------------------

def test_chat_nlp_streams_final_answer_and_prints_it_once(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Opted in + capable provider: the forced-synthesis answer is served by
    ``chat_stream`` (proving the wiring is live, not dead code) and the caller
    does not print it a second time.

    Chaining is disabled (``_MAX_CHAINED_RUNS = 0``) so exactly ONE run streams
    once — otherwise a cap verdict chains a second run that legitimately
    streams its own answer, and a print count could not tell a duplicate
    ``_finish_turn`` print from a second chained run.
    """
    from agent import Agent
    from agent_core.llm.tool_loop import ToolLoopRunner

    monkeypatch.setattr(
        agent_mod, "CHAT_HISTORY_JSON_PATH", str(tmp_path / "chat_history.json")
    )
    monkeypatch.setattr(
        agent_mod, "AGENT_MEMORY_JSON_PATH", str(tmp_path / "agent_memory.json")
    )
    monkeypatch.setattr(agent_mod, "TURN_LOG_PATH", str(tmp_path / "turn_log.jsonl"))
    monkeypatch.setattr(
        agent_mod, "load_agent_settings",
        lambda: _settings(stream_final_answer=True),
    )
    monkeypatch.setattr(
        agent_mod, "_resolve_display_mode", lambda: AgentDisplayMode.QUIET
    )
    monkeypatch.setattr(agent_mod, "_MAX_CHAINED_RUNS", 0)
    monkeypatch.setattr(
        agent_mod, "ToolLoopRunner",
        lambda *a, **k: ToolLoopRunner(max_iterations=3),
    )

    answer = "STREAMED-FINAL-ANSWER"

    class StreamingLLM:
        """Distinct reads -> cap -> forced synthesis served by chat_stream."""

        def __init__(self) -> None:
            self.chat_calls = 0
            self.stream_calls = 0
            self._provider = _bare(LMStudioProvider)

        async def chat(self, messages, tools=None, **kwargs):
            self.chat_calls += 1
            # Tool-less call = forced synthesis; must be served by chat_stream.
            if not tools:
                return answer
            return json.dumps({
                "content": "",
                "tool_calls": [{
                    "id": f"c{self.chat_calls}", "type": "function",
                    "function": {
                        "name": "read",
                        # distinct paths: never a duplicate/stuck verdict
                        "arguments": json.dumps({"path": f"f{self.chat_calls}.py"}),
                    },
                }],
            })

        async def chat_stream(self, messages):
            self.stream_calls += 1
            print(answer)  # what a real streaming provider does
            return answer

    bot = Agent(workspace=".")
    llm = StreamingLLM()
    bot.llm = llm

    async def go():
        await bot.chat_nlp("read the files")

    asyncio.run(go())

    assert llm.stream_calls == 1, (
        "chat_nlp did not route the forced-synthesis answer through chat_stream "
        "— the stream_fn wiring is not live"
    )
    out = capsys.readouterr().out
    assert out.count(answer) == 1, (
        f"answer must be shown exactly once, saw {out.count(answer)}: {out!r}"
    )


def test_chat_nlp_does_not_stream_when_not_opted_in(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Default (setting off): byte-identical to before — the blocking call
    serves the answer and chat_stream is never used."""
    from agent import Agent
    from agent_core.llm.tool_loop import ToolLoopRunner

    monkeypatch.setattr(
        agent_mod, "CHAT_HISTORY_JSON_PATH", str(tmp_path / "chat_history.json")
    )
    monkeypatch.setattr(
        agent_mod, "AGENT_MEMORY_JSON_PATH", str(tmp_path / "agent_memory.json")
    )
    monkeypatch.setattr(agent_mod, "TURN_LOG_PATH", str(tmp_path / "turn_log.jsonl"))
    monkeypatch.setattr(
        agent_mod, "load_agent_settings",
        lambda: _settings(stream_final_answer=False),
    )
    monkeypatch.setattr(
        agent_mod, "_resolve_display_mode", lambda: AgentDisplayMode.QUIET
    )
    monkeypatch.setattr(agent_mod, "_MAX_CHAINED_RUNS", 0)
    monkeypatch.setattr(
        agent_mod, "ToolLoopRunner",
        lambda *a, **k: ToolLoopRunner(max_iterations=3),
    )

    answer = "BLOCKING-FINAL-ANSWER"

    class BlockingLLM:
        def __init__(self) -> None:
            self.chat_calls = 0
            self.stream_calls = 0
            self._provider = _bare(LMStudioProvider)

        async def chat(self, messages, tools=None, **kwargs):
            self.chat_calls += 1
            if not tools:
                return answer
            return json.dumps({
                "content": "",
                "tool_calls": [{
                    "id": f"c{self.chat_calls}", "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps({"path": f"f{self.chat_calls}.py"}),
                    },
                }],
            })

        async def chat_stream(self, messages):
            self.stream_calls += 1
            return answer

    bot = Agent(workspace=".")
    llm = BlockingLLM()
    bot.llm = llm

    async def go():
        await bot.chat_nlp("read the files")

    asyncio.run(go())

    assert llm.stream_calls == 0, "streaming must stay off unless opted in"
    assert answer in capsys.readouterr().out, "the answer must still be printed"


def test_chained_runs_each_stream_their_own_answer_without_a_double_print(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Chaining is ON (the real default, ``_MAX_CHAINED_RUNS = 6``): a turn that
    keeps hitting the cap re-enters the loop, and EVERY run streams its own
    forced synthesis.

    So the user may legitimately see more than one streamed answer — run 2's is
    a *continuation*, not a duplicate of run 1's.  What must NEVER happen is a
    print on TOP of a streamed answer: the number of times the answer appears on
    the console must equal the number of ``chat_stream`` calls (each of which
    printed it live), i.e. ``_finish_turn`` added none.

    The chain is deterministic here: run 1 caps (``reason == "cap"`` ->
    ``needs_more``) and chains; run 2 caps with the SAME answer, which the
    existing stuck-detection catches (``final_text == last_answer``) and breaks
    — exactly two runs, two streams.
    """
    from agent import Agent
    from agent_core.llm.tool_loop import ToolLoopRunner

    monkeypatch.setattr(
        agent_mod, "CHAT_HISTORY_JSON_PATH", str(tmp_path / "chat_history.json")
    )
    monkeypatch.setattr(
        agent_mod, "AGENT_MEMORY_JSON_PATH", str(tmp_path / "agent_memory.json")
    )
    monkeypatch.setattr(agent_mod, "TURN_LOG_PATH", str(tmp_path / "turn_log.jsonl"))
    monkeypatch.setattr(
        agent_mod, "load_agent_settings",
        lambda: _settings(stream_final_answer=True),
    )
    monkeypatch.setattr(
        agent_mod, "_resolve_display_mode", lambda: AgentDisplayMode.QUIET
    )
    # Small cap keeps the run fast; chaining is deliberately LEFT ON.
    monkeypatch.setattr(
        agent_mod, "ToolLoopRunner",
        lambda *a, **k: ToolLoopRunner(max_iterations=2),
    )

    answer = "CHAINED-STREAMED-ANSWER"

    class CappingLLM:
        def __init__(self) -> None:
            self.chat_calls = 0
            self.stream_calls = 0
            self._provider = _bare(LMStudioProvider)

        async def chat(self, messages, tools=None, **kwargs):
            self.chat_calls += 1
            if not tools:
                return answer
            return json.dumps({
                "content": "",
                "tool_calls": [{
                    "id": f"c{self.chat_calls}", "type": "function",
                    "function": {
                        "name": "read",
                        # distinct paths: never a duplicate/stuck tool verdict
                        "arguments": json.dumps({"path": f"f{self.chat_calls}.py"}),
                    },
                }],
            })

        async def chat_stream(self, messages):
            self.stream_calls += 1
            print(answer)  # what a real streaming provider does
            return answer

    bot = Agent(workspace=".")
    llm = CappingLLM()
    bot.llm = llm

    async def go():
        await bot.chat_nlp("read the files")

    asyncio.run(go())

    assert llm.stream_calls == 2, (
        "each chained run must stream its own forced synthesis "
        f"(saw {llm.stream_calls})"
    )
    out = capsys.readouterr().out
    assert out.count(answer) == llm.stream_calls, (
        "every occurrence of the answer must come from a chat_stream print — "
        f"_finish_turn added {out.count(answer) - llm.stream_calls} extra"
    )

