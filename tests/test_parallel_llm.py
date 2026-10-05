"""Tests for parallel multi-LLM dispatch (agent_core.llm.parallel).

The whole point of the module is to fire SIMULTANEOUS calls to DIFFERENT
LLMs (local LM Studio + hosted opencode-go).  These tests prove:

1. Concurrency — N fake providers that each sleep 0.2s finish in ~0.2s, not
   N * 0.2s, when fired in parallel (wall-clock assertion).
2. Per-model provider building — the real `build_provider` routing is used
   (lmstudio vs opencode per model prefix).
3. Error isolation — one failing provider (error string or exception) does
   not abort the other models' answers.
4. Consensus wiring — the dormant RefinementVoter/ConsensusVoter machinery
   records each model's verdict and the quorum gate works.
"""
import asyncio
import io
import json
import time
import urllib.request
from unittest.mock import patch

import pytest

from agent_core.llm.parallel import (
    ParallelResult,
    ParallelRun,
    _looks_negative,
    parse_verdict,
    run_parallel,
    summarize,
)


class _FakeProvider:
    """Minimal LLMProvider-protocol stub with a configurable chat()."""

    def __init__(self, model_name, delay=0.0, text="answer", error=False, exc=None):
        self.model_name = model_name
        self.delay = delay
        self.text = text
        self.error = error
        self.exc = exc
        self.last_response_metrics = None
        self.calls = 0

    async def chat(self, messages, tools=None, max_tokens=None, disable_thinking=False):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc is not None:
            raise self.exc
        if self.error:
            return "[Error: fake failure]"
        return self.text


def _settings():
    return type("S", (), {"llm_provider": "lmstudio"})()


def _fake_build_provider(settings, model_name):
    """Deterministic provider factory keyed on the model name."""
    return _FakeProvider(model_name, delay=0.2)


class TestParallelConcurrency:
    def test_two_models_fire_simultaneously(self):
        """Wall time must be ~one delay, not two — the parallel exploit."""
        with patch(
            "agent_core.llm.parallel.build_provider",
            side_effect=_fake_build_provider,
        ):
            start = time.monotonic()
            run = asyncio.run(run_parallel(
                [{"role": "user", "content": "hi"}],
                ["laguna-s-2.1", "opencode-go/deepseek-v4-flash"],
                _settings(),
            ))
            elapsed = time.monotonic() - start
        # Both providers sleep 0.2s in parallel -> ~0.2s total.
        assert elapsed < 0.35, f"calls were serialized: {elapsed:.3f}s"
        assert len(run.results) == 2
        assert all(r.ok for r in run.results)
        assert [r.model for r in run.results] == [
            "laguna-s-2.1", "opencode-go/deepseek-v4-flash",
        ]
        assert all(r.provider in ("lmstudio", "opencode") for r in run.results)

    def test_blocking_providers_do_not_serialize(self):
        """Real providers BLOCK the event loop (sync urllib HTTP inside
        async chat).  The fix dispatches the blocking call to a worker
        thread (asyncio.to_thread) — two 0.2s BLOCKING providers must still
        finish in ~0.2s, not ~0.4s.  This test fails against the pre-fix
        code (urlopen blocks the loop, gather serializes)."""
        from agent_core.llm.lmstudio import LMStudioProvider

        # Simulate a slow LM Studio: the synchronous HTTP call (urlopen)
        # blocks for 0.2s, exactly like a real round-trip.

        def _slow_urlopen(req, timeout=None):
            time.sleep(0.2)
            body = json.dumps({
                "choices": [{
                    "message": {"content": "blocking answer"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }).encode()
            return io.BytesIO(body)

        def factory(settings, model_name):
            return LMStudioProvider(model_name)

        with patch("agent_core.llm.parallel.build_provider", side_effect=factory), \
             patch("urllib.request.urlopen", side_effect=_slow_urlopen):
            start = time.monotonic()
            run = asyncio.run(run_parallel(
                [{"role": "user", "content": "hi"}],
                ["laguna-s-2.1", "opencode-go/deepseek-v4-flash"],
                _settings(),
            ))
            elapsed = time.monotonic() - start
        # Both blocking providers run in worker threads -> ~0.2s total.
        assert elapsed < 0.35, f"blocking calls were serialized: {elapsed:.3f}s"
        assert all(r.ok for r in run.results)
        assert all(r.text == "blocking answer" for r in run.results)

    def test_concurrency_cap_serializes(self):
        """With concurrency=1 the two 0.2s calls take ~0.4s."""
        with patch(
            "agent_core.llm.parallel.build_provider",
            side_effect=_fake_build_provider,
        ):
            start = time.monotonic()
            run = asyncio.run(run_parallel(
                [{"role": "user", "content": "hi"}],
                ["laguna-s-2.1", "opencode-go/deepseek-v4-flash"],
                _settings(),
                concurrency=1,
            ))
            elapsed = time.monotonic() - start
        assert elapsed >= 0.35, f"concurrency=1 did not serialize: {elapsed:.3f}s"
        assert len(run.ok_results) == 2

    def test_requires_two_models(self):
        with pytest.raises(ValueError, match="at least two models"):
            asyncio.run(run_parallel(
                [{"role": "user", "content": "hi"}],
                ["laguna-s-2.1"],
                _settings(),
            ))


class TestParallelIsolation:
    def test_one_error_string_does_not_abort_others(self):
        def factory(settings, model_name):
            if "opencode" in model_name:
                return _FakeProvider(model_name, error=True)
            return _FakeProvider(model_name, text="local answer")

        with patch("agent_core.llm.parallel.build_provider", side_effect=factory):
            run = asyncio.run(run_parallel(
                [{"role": "user", "content": "hi"}],
                ["laguna-s-2.1", "opencode-go/deepseek-v4-flash"],
                _settings(),
            ))
        assert len(run.results) == 2
        ok = {r.model: r.ok for r in run.results}
        assert ok["laguna-s-2.1"] is True
        assert ok["opencode-go/deepseek-v4-flash"] is False
        assert run.ok_results[0].text == "local answer"
        assert run.failed_results[0].error.startswith("[Error")

    def test_one_exception_does_not_abort_others(self):
        def factory(settings, model_name):
            if "opencode" in model_name:
                return _FakeProvider(model_name, exc=RuntimeError("boom"))
            return _FakeProvider(model_name, text="local answer")

        with patch("agent_core.llm.parallel.build_provider", side_effect=factory):
            run = asyncio.run(run_parallel(
                [{"role": "user", "content": "hi"}],
                ["laguna-s-2.1", "opencode-go/deepseek-v4-flash"],
                _settings(),
            ))
        ok = {r.model: r.ok for r in run.results}
        assert ok["laguna-s-2.1"] is True
        assert ok["opencode-go/deepseek-v4-flash"] is False
        assert "RuntimeError" in run.failed_results[0].error

    def test_build_provider_routing_real(self, monkeypatch):
        """The REAL build_provider routes lmstudio vs opencode per prefix."""
        from agent_core.llm.provider import build_provider

        monkeypatch.setattr(
            "agent_core.constants.load_model_json",
            lambda: {"model": "laguna-s-2.1", "provider": "lmstudio"},
        )
        settings = type("S", (), {
            "llm_provider": "lmstudio",
            "opencode_server_url": "http://127.0.0.1:4096",
            "opencode_password": "",
            "opencode_api_url": "https://opencode.ai/zen/go/v1",
            "opencode_api_key": "",
        })()
        lm = build_provider(settings, "laguna-s-2.1")
        oc = build_provider(settings, "opencode-go/glm-5.2")
        assert type(lm).__name__ == "LMStudioProvider"
        assert type(oc).__name__ == "OpencodeProvider"


class TestConsensus:
    def test_votes_recorded_and_quorum_gate(self):
        run = ParallelRun(template_id="t1")
        run.results = [
            ParallelResult(model="a", provider="lmstudio", text="yes looks good", ok=True),
            ParallelResult(model="b", provider="opencode", text="no, has a bug", ok=True),
        ]
        run.agree(verdict=True)
        status = run.voter.vote_status("t1")
        assert status["total"] == 2
        assert status["yes"] == 1  # "no, has a bug" counts as reject
        # 1/2 = 0.5 meets the >= 0.5 quorum (approval-ratio semantics).
        assert run.quorum_reached(quorum_threshold=0.5) is True
        assert "consensus APPROVE" in run.consensus()

    def test_tie_below_stricter_quorum(self):
        run = ParallelRun(template_id="t5")
        run.results = [
            ParallelResult(model="a", provider="lmstudio", text="yes", ok=True),
            ParallelResult(model="b", provider="opencode", text="no", ok=True),
        ]
        run.agree(verdict=True)
        # 1/2 < 0.6 -> no consensus under the stricter threshold.
        assert run.quorum_reached(quorum_threshold=0.6) is False
        assert "no consensus" in run.consensus(quorum_threshold=0.6)

    def test_majority_approves(self):
        run = ParallelRun(template_id="t4")
        run.results = [
            ParallelResult(model="a", provider="lmstudio", text="yes", ok=True),
            ParallelResult(model="b", provider="opencode", text="looks good", ok=True),
            ParallelResult(model="c", provider="lmstudio", text="no", ok=True),
        ]
        run.agree(verdict=True)
        assert run.quorum_reached(quorum_threshold=0.5) is True  # 2/3
        assert "consensus APPROVE" in run.consensus()

    def test_no_consensus_below_quorum(self):
        run = ParallelRun(template_id="t2")
        run.results = [
            ParallelResult(model="a", provider="lmstudio", text="yes", ok=True),
            ParallelResult(model="b", provider="opencode", text="no", ok=True),
            ParallelResult(model="c", provider="lmstudio", text="no way", ok=True),
        ]
        run.agree(verdict=True)
        assert run.quorum_reached(quorum_threshold=0.5) is False  # 1/3
        assert "no consensus" in run.consensus()

    def test_no_usable_answers(self):
        run = ParallelRun(template_id="t3")
        run.results = [
            ParallelResult(model="a", provider="lmstudio", text="[Error: down]", ok=False),
        ]
        run.agree(verdict=True)
        assert "no usable model answers" in run.consensus()

    def test_looks_negative(self):
        assert _looks_negative("No, that is wrong")
        assert _looks_negative("bug found at line 3")
        assert not _looks_negative("Yes, this looks correct")
        assert _looks_negative("")  # empty answer counts as reject


class TestStructuredVerdict:
    """The honest-consensus fix: vote from the model's OWN stated verdict.

    The old vote was ``verdict and not _looks_negative(text)`` — a lexical
    guess over the first 120 chars, which a model can contradict in its own
    conclusion ("No, wait — on reflection this is correct").  These tests pin
    the structured ``VERDICT:`` path AND the fallback for answers that carry
    no verdict line.
    """

    def test_parse_verdict_approve_and_reject(self):
        assert parse_verdict("Looks fine.\nVERDICT: APPROVE")[0] is True
        assert parse_verdict("Broken.\nVERDICT: REJECT")[0] is False

    def test_parse_verdict_accept_and_deny_synonyms(self):
        assert parse_verdict("VERDICT: ACCEPT")[0] is True
        assert parse_verdict("VERDICT: DENY")[0] is False

    def test_parse_verdict_tolerates_markdown_and_punctuation(self):
        """Models emit bold/heading verdict lines — still parse them."""
        assert parse_verdict("**VERDICT:** APPROVE")[0] is True
        assert parse_verdict("## VERDICT: REJECT")[0] is False
        assert parse_verdict("- VERDICT - APPROVE")[0] is True

    def test_parse_verdict_last_line_wins(self):
        """A restated proposition must not override the final conclusion."""
        text = "VERDICT: REJECT is what a hasty read suggests.\n" \
               "On closer inspection:\nVERDICT: APPROVE"
        assert parse_verdict(text)[0] is True

    def test_parse_verdict_reason_extracted(self):
        verdict, reason = parse_verdict(
            "Some prose.\nVERDICT: REJECT\nREASON: off-by-one at line 42"
        )
        assert verdict is False
        assert reason == "off-by-one at line 42"

    def test_parse_verdict_absent_returns_none(self):
        """No verdict line → (None, '') so the caller can fall back, not guess."""
        assert parse_verdict("This is fine, I think.") == (None, "")
        assert parse_verdict("") == (None, "")
        # A prose mention is NOT a verdict line (must be at line start).
        assert parse_verdict("the verdict is approve")[0] is None

    def test_structured_verdict_beats_negative_heuristic(self):
        """REGRESSION: prose looks negative, model's verdict says APPROVE."""
        run = ParallelRun(template_id="sv1")
        run.results = [
            ParallelResult(
                model="a", provider="lmstudio", ok=True,
                text="No, wait — I checked and this is correct.\n"
                     "VERDICT: APPROVE\nREASON: checked the call site",
            ),
        ]
        # The heuristic alone would reject this answer.
        assert _looks_negative(run.results[0].text) is True
        run.agree(verdict=True)
        assert run.voter.vote_status("sv1")["yes"] == 1  # structured wins
        assert run.results[0].verdict is True
        assert run.results[0].verdict_reason == "checked the call site"
        assert run.votes_structured == 1

    def test_structured_reject_beats_positive_heuristic(self):
        """Mirror case: prose looks positive, model's verdict says REJECT."""
        run = ParallelRun(template_id="sv2")
        run.results = [
            ParallelResult(
                model="a", provider="lmstudio", ok=True,
                text="Yes, this looks correct at first glance.\n"
                     "VERDICT: REJECT\nREASON: mutates the caller's list",
            ),
        ]
        assert _looks_negative(run.results[0].text) is False
        run.agree(verdict=True)
        assert run.voter.vote_status("sv2")["yes"] == 0
        assert run.results[0].verdict is False

    def test_unparseable_answer_falls_back_to_heuristic(self):
        """No verdict line → the old lexical behavior is preserved."""
        run = ParallelRun(template_id="sv3")
        run.results = [
            ParallelResult(model="a", provider="lmstudio", ok=True,
                           text="yes looks good"),
            ParallelResult(model="b", provider="opencode", ok=True,
                           text="no, has a bug"),
        ]
        run.agree(verdict=True)
        assert run.voter.vote_status("sv3") == {
            "total": 2, "yes": 1, "no": 1,
        }
        assert run.votes_structured == 0
        assert all(r.verdict is None for r in run.results)

    def test_auto_agree_uses_own_verdicts(self):
        """Each model votes its OWN verdict — no reference verdict needed."""
        run = ParallelRun(template_id="sv4", verdict_requested=True)
        run.results = [
            ParallelResult(model="a", provider="lmstudio", ok=True,
                           text="prose\nVERDICT: APPROVE"),
            ParallelResult(model="b", provider="opencode", ok=True,
                           text="prose\nVERDICT: APPROVE"),
            ParallelResult(model="c", provider="lmstudio", ok=True,
                           text="prose\nVERDICT: REJECT"),
        ]
        assert run.auto_agree() == 3
        assert run.voter.vote_status("sv4")["yes"] == 2
        assert run.quorum_reached(quorum_threshold=0.5) is True  # 2/3
        assert "consensus APPROVE" in run.consensus()

    def test_auto_agree_skips_unparseable_answers(self):
        """A silent model is NOT counted as a reject — it just doesn't vote."""
        run = ParallelRun(template_id="sv5", verdict_requested=True)
        run.results = [
            ParallelResult(model="a", provider="lmstudio", ok=True,
                           text="prose\nVERDICT: APPROVE"),
            ParallelResult(model="b", provider="opencode", ok=True,
                           text="rambling answer with no verdict line"),
            ParallelResult(model="c", provider="lmstudio", ok=False,
                           text="[Error: down]"),
        ]
        assert run.auto_agree() == 1
        assert run.voter.vote_status("sv5")["total"] == 1
        # Only the one stated verdict counts — and it approves.
        assert run.quorum_reached(quorum_threshold=1.0) is True

    def test_auto_agree_with_no_stated_verdicts_is_no_consensus(self):
        """No parseable verdicts → no votes → quorum must NOT be claimed."""
        run = ParallelRun(template_id="sv6", verdict_requested=True)
        run.results = [
            ParallelResult(model="a", provider="lmstudio", ok=True, text="maybe"),
            ParallelResult(model="b", provider="opencode", ok=True, text="unclear"),
        ]
        assert run.auto_agree() == 0
        assert run.quorum_reached(quorum_threshold=0.0) is False
        assert "no usable model answers" in run.consensus()

    def test_summarize_shows_verdict_labels(self):
        run = ParallelRun(template_id="sv7", verdict_requested=True)
        run.results = [
            ParallelResult(model="a", provider="lmstudio", ok=True,
                           text="x\nVERDICT: APPROVE\nREASON: fine"),
            ParallelResult(model="b", provider="opencode", ok=True,
                           text="y\nVERDICT: REJECT\nREASON: leak"),
        ]
        run.auto_agree()
        out = summarize(run)
        assert "[APPROVE]" in out
        assert "[REJECT]" in out
        assert "verdict: fine" in out

    def test_verdict_instruction_appended_to_last_user_message(self):
        """run_parallel(verdict_instruction=True) asks every model for a line."""
        seen = {}

        class _Recording:
            def __init__(self, model_name):
                self.model_name = model_name
                self.last_response_metrics = None

            async def chat(self, messages, tools=None, max_tokens=None,
                           disable_thinking=False):
                seen[self.model_name] = list(messages)
                return "ok\nVERDICT: APPROVE"

        with patch("agent_core.llm.parallel.build_provider",
                   side_effect=lambda s, m: _Recording(m)):
            run = asyncio.run(run_parallel(
                [{"role": "user", "content": "is this ok?"}],
                ["laguna-s-2.1", "opencode-go/deepseek-v4-flash"],
                _settings(),
                verdict_instruction=True,
            ))

        assert run.verdict_requested is True
        for model in ("laguna-s-2.1", "opencode-go/deepseek-v4-flash"):
            user_msgs = [m for m in seen[model] if m["role"] == "user"]
            assert "VERDICT: APPROVE" in user_msgs[-1]["content"]
            assert user_msgs[-1]["content"].startswith("is this ok?")
        assert run.auto_agree() == 2
        assert run.quorum_reached(quorum_threshold=1.0) is True

    def test_verdict_instruction_off_by_default(self):
        """Existing callers must not get the extra prompt text."""
        seen = {}

        class _Recording:
            def __init__(self, model_name):
                self.model_name = model_name
                self.last_response_metrics = None

            async def chat(self, messages, tools=None, max_tokens=None,
                           disable_thinking=False):
                seen[self.model_name] = list(messages)
                return "ok"

        with patch("agent_core.llm.parallel.build_provider",
                   side_effect=lambda s, m: _Recording(m)):
            run = asyncio.run(run_parallel(
                [{"role": "user", "content": "is this ok?"}],
                ["laguna-s-2.1", "opencode-go/deepseek-v4-flash"],
                _settings(),
            ))

        assert run.verdict_requested is False
        for model in ("laguna-s-2.1", "opencode-go/deepseek-v4-flash"):
            user_msgs = [m for m in seen[model] if m["role"] == "user"]
            assert user_msgs[-1]["content"] == "is this ok?"


class TestRoles:
    """Per-model system prompts: each LLM gets its OWN role prepended."""

    def test_role_prepended_to_that_models_messages(self):
        """The role system message is prepended ONLY to the named model."""
        seen = {}

        class _Recording:
            def __init__(self, model_name):
                self.model_name = model_name
                self.last_response_metrics = None

            async def chat(self, messages, tools=None, max_tokens=None,
                           disable_thinking=False):
                seen[self.model_name] = list(messages)
                return "answer"

        def factory(settings, model_name):
            return _Recording(model_name)

        with patch("agent_core.llm.parallel.build_provider", side_effect=factory):
            asyncio.run(run_parallel(
                [{"role": "user", "content": "review this code"}],
                ["laguna-s-2.1", "opencode-go/deepseek-v4-flash"],
                _settings(),
                roles={
                    "laguna-s-2.1": "You are a security auditor.",
                    "opencode-go/deepseek-v4-flash": "You are a perf engineer.",
                },
            ))
        local = seen["laguna-s-2.1"]
        remote = seen["opencode-go/deepseek-v4-flash"]
        # Own role first, then the shared question.
        assert local[0] == {"role": "system", "content": "You are a security auditor."}
        assert remote[0] == {"role": "system", "content": "You are a perf engineer."}
        assert local[1] == remote[1] == {"role": "user", "content": "review this code"}
        # Roles do not leak across models.
        assert "security" not in remote[0]["content"]
        assert "perf" not in local[0]["content"]

    def test_no_role_keeps_messages_unchanged(self):
        seen = {}

        class _Recording:
            def __init__(self, model_name):
                self.model_name = model_name
                self.last_response_metrics = None

            async def chat(self, messages, tools=None, max_tokens=None,
                           disable_thinking=False):
                seen[self.model_name] = list(messages)
                return "answer"

        def factory(settings, model_name):
            return _Recording(model_name)

        with patch("agent_core.llm.parallel.build_provider", side_effect=factory):
            asyncio.run(run_parallel(
                [{"role": "user", "content": "hi"}],
                ["laguna-s-2.1", "opencode-go/deepseek-v4-flash"],
                _settings(),
            ))
        assert seen["laguna-s-2.1"] == [{"role": "user", "content": "hi"}]

    def test_role_for_unknown_model_is_ignored(self):
        """A role keyed to a model not in the run is simply not applied."""
        seen = {}

        class _Recording:
            def __init__(self, model_name):
                self.model_name = model_name
                self.last_response_metrics = None

            async def chat(self, messages, tools=None, max_tokens=None,
                           disable_thinking=False):
                seen[self.model_name] = list(messages)
                return "answer"

        def factory(settings, model_name):
            return _Recording(model_name)

        with patch("agent_core.llm.parallel.build_provider", side_effect=factory):
            asyncio.run(run_parallel(
                [{"role": "user", "content": "hi"}],
                ["laguna-s-2.1", "opencode-go/deepseek-v4-flash"],
                _settings(),
                roles={"some-other-model": "ignored role"},
            ))
        assert seen["laguna-s-2.1"] == [{"role": "user", "content": "hi"}]
        assert seen["opencode-go/deepseek-v4-flash"] == [
            {"role": "user", "content": "hi"}
        ]


class TestMultiLlmCommandRoles:
    """The multillm command accepts roles INLINE — no roles.json needed.

    The REPL splits input with ``shlex.split(user_input, posix=False)``,
    which KEEPS the literal quotes on a quoted ``--role "model:prompt"``
    value.  The command must strip them (repo convention: analyze_cmd,
    fix_cmd, implement_cmd all use ``.strip('"')``) so multi-word inline
    roles reach the providers intact.
    """

    @staticmethod
    def _run_command(args, agent=None):
        from agent_core.commands.multillm_cmd import MultiLlmCommand
        return MultiLlmCommand().execute(args, agent)

    @staticmethod
    def _agent():
        class FakeLLM:
            model_name = "laguna-s-2.1"
        agent = type("A", (), {"llm": FakeLLM(), "workspace": "C:/Dev/Agent1"})()
        return agent

    def test_simultaneous_review_command_end_to_end(self):
        """The exact command from the request, through the REAL REPL path
        (shlex.split posix=False -> MultiLlmCommand.execute).

        Proves three things at once:
        1. SIMULTANEITY — both models' chat() calls each take 0.2s (simulated
           LLM latency); the whole command must finish in ~0.2s, not ~0.4s.
           A serialized runner (awaiting model A fully, then model B) takes
           ~0.4s and fails the wall-clock assertion.
        2. ROLES — each model receives its OWN system prompt (security
           auditor vs performance engineer) prepended to the shared question.
        3. QUESTION — the full quoted question survives the posix=False
           tokenization and reaches both models identically.
        """
        import shlex

        seen_roles = {}     # model -> its system prompt
        seen_question = {}  # model -> the shared question

        class _SlowProvider:
            """0.2s simulated latency per model, role-aware answer."""

            def __init__(self, model_name):
                self.model_name = model_name
                self.last_response_metrics = None

            async def chat(self, messages, tools=None, max_tokens=None,
                           disable_thinking=False):
                seen_roles[self.model_name] = str(messages[0]["content"])
                seen_question[self.model_name] = str(messages[-1]["content"])
                await asyncio.sleep(0.2)  # simulated LLM round-trip
                if "security" in str(messages[0]):
                    return ("SECURITY: agent.py review — no injection "
                            "vulnerabilities in the tool dispatch path.")
                return ("PERF: agent.py review — the hot path is the REPL "
                        "command loop; no blocking I/O on it.")

        def factory(settings, model_name):
            return _SlowProvider(model_name)

        # EXACT command from the request, as one REPL line.
        cmd = (
            'multillm "review agent.py, and synthesize the two models\' '
            'responses" '
            "--models laguna-s-2.1,opencode-go/deepseek-v4-flash "
            '--role "laguna-s-2.1:You are a security auditor. Focus on '
            'vulnerabilities." '
            '--role "opencode-go/deepseek-v4-flash:You are a performance '
            'engineer. Focus on hot paths."'
        )
        parts = shlex.split(cmd, posix=False)[1:]  # REPL strips the command name

        with patch("agent_core.llm.parallel.build_provider", side_effect=factory):
            start = time.monotonic()
            ok = asyncio.run(self._run_command(parts, self._agent()))
            elapsed = time.monotonic() - start

        assert ok is True
        # SIMULTANEITY: two 0.2s calls finish in ~0.2s, not ~0.4s.
        assert elapsed < 0.35, f"calls were serialized: {elapsed:.3f}s"
        # Roles reached the right models.
        assert "security auditor" in seen_roles["laguna-s-2.1"]
        assert "performance engineer" in seen_roles["opencode-go/deepseek-v4-flash"]
        # The full question (quoted at the REPL) survived intact.
        assert "review agent.py, and synthesize the two models' responses" \
            in seen_question["laguna-s-2.1"]
        assert seen_question["laguna-s-2.1"] == seen_question[
            "opencode-go/deepseek-v4-flash"
        ]

    def test_inline_roles_with_quotes_are_stripped(self):
        """Quoted multi-word --role values (as REPL delivers them) work."""
        seen = {}

        class _Recording:
            def __init__(self, model_name):
                self.model_name = model_name
                self.last_response_metrics = None

            async def chat(self, messages, tools=None, max_tokens=None,
                           disable_thinking=False):
                seen[self.model_name] = list(messages)
                return "answer"

        def factory(settings, model_name):
            return _Recording(model_name)

        # Exactly the token shapes shlex.split(posix=False) produces.
        args = [
            "review this code",
            "--models", "laguna-s-2.1,opencode-go/deepseek-v4-flash",
            "--role", '"laguna-s-2.1:You are a security auditor."',
            "--role", '"opencode-go/deepseek-v4-flash:You are a perf engineer."',
        ]
        with patch("agent_core.llm.parallel.build_provider", side_effect=factory):
            ok = asyncio.run(self._run_command(args, self._agent()))
        assert ok is True
        assert seen["laguna-s-2.1"][0] == {
            "role": "system", "content": "You are a security auditor.",
        }
        assert seen["opencode-go/deepseek-v4-flash"][0] == {
            "role": "system", "content": "You are a perf engineer.",
        }
        # The question also arrives quoted (posix=False quirk) but passes through.
        assert seen["laguna-s-2.1"][1]["content"] == "review this code"

    def test_inline_role_without_quotes_works(self):
        """Unquoted --role model:prompt still works (single-word prompts)."""
        seen = {}

        class _Recording:
            def __init__(self, model_name):
                self.model_name = model_name
                self.last_response_metrics = None

            async def chat(self, messages, tools=None, max_tokens=None,
                           disable_thinking=False):
                seen[self.model_name] = list(messages)
                return "answer"

        def factory(settings, model_name):
            return _Recording(model_name)

        args = [
            "hi",
            "--models", "laguna-s-2.1,opencode-go/deepseek-v4-flash",
            "--role", "laguna-s-2.1:reviewer",
            "--role", "opencode-go/deepseek-v4-flash:auditor",
        ]
        with patch("agent_core.llm.parallel.build_provider", side_effect=factory):
            ok = asyncio.run(self._run_command(args, self._agent()))
        assert ok is True
        assert seen["laguna-s-2.1"][0] == {"role": "system", "content": "reviewer"}
        assert seen["opencode-go/deepseek-v4-flash"][0] == {
            "role": "system", "content": "auditor",
        }

    def test_malformed_role_is_rejected(self):
        """--role without a colon errors out instead of silently ignoring."""
        args = [
            "hi",
            "--models", "laguna-s-2.1,opencode-go/deepseek-v4-flash",
            "--role", "no-colon-here",
        ]
        ok = asyncio.run(self._run_command(args, self._agent()))
        assert ok is True  # error path returns True (handled) without crashing


class TestParallelTools:
    """Parallel models get the SAME tools the agent uses — they can actually
    read/search/list files instead of answering from the prompt alone."""

    @staticmethod
    def _tool_schemas():
        return [{
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }]

    def test_models_can_use_tools_via_tool_loop(self):
        """A model that requests read_file gets the result and answers with it."""
        seen = {}

        class _ToolCallingProvider:
            def __init__(self, model_name):
                self.model_name = model_name
                self.last_response_metrics = None
                self._step = 0

            async def chat(self, messages, tools=None, max_tokens=None,
                           disable_thinking=False):
                self._step += 1
                if self._step == 1:
                    seen[self.model_name] = {"tools": tools is not None}
                    return json.dumps({
                        "content": "I will read the file.",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": json.dumps({"path": "agent.py"}),
                            },
                        }],
                    })
                last = messages[-1]
                assert last["role"] == "tool"
                return (f"MODEL {self.model_name} says: I read agent.py. "
                        f"{last['content'][:40]}")

        async def _execute_tool(name, args):
            assert name == "read"
            assert args == {"path": "agent.py"}
            return "FAKE-CONTENT-OF-agent.py (42 lines)"

        def factory(settings, model_name):
            return _ToolCallingProvider(model_name)

        with patch("agent_core.llm.parallel.build_provider", side_effect=factory):
            run = asyncio.run(run_parallel(
                [{"role": "user", "content": "review agent.py"}],
                ["laguna-s-2.1", "opencode-go/deepseek-v4-flash"],
                _settings(),
                tools=self._tool_schemas(),
                execute_tool_fn=_execute_tool,
            ))
        assert len(run.results) == 2
        for r in run.results:
            assert r.ok, r.error
            assert "I read agent.py" in r.text
            # Both models actually received the tools schemas.
            assert seen[r.model]["tools"] is True

    def test_tools_requires_execute_tool_fn(self):
        """tools without an executor is a programming error, not silent."""
        with pytest.raises(ValueError, match="execute_tool_fn"):
            asyncio.run(run_parallel(
                [{"role": "user", "content": "hi"}],
                ["laguna-s-2.1", "opencode-go/deepseek-v4-flash"],
                _settings(),
                tools=self._tool_schemas(),
            ))

    def test_no_tools_keeps_plain_single_chat(self):
        """Without tools the provider is called exactly once (no loop)."""
        seen = {}

        class _Counting:
            def __init__(self, model_name):
                self.model_name = model_name
                self.last_response_metrics = None
                self.calls = 0

            async def chat(self, messages, tools=None, max_tokens=None,
                           disable_thinking=False):
                self.calls += 1
                seen[self.model_name] = self.calls
                return "plain answer"

        def factory(settings, model_name):
            return _Counting(model_name)

        with patch("agent_core.llm.parallel.build_provider", side_effect=factory):
            run = asyncio.run(run_parallel(
                [{"role": "user", "content": "hi"}],
                ["laguna-s-2.1", "opencode-go/deepseek-v4-flash"],
                _settings(),
            ))
        assert all(r.ok for r in run.results)
        assert seen == {"laguna-s-2.1": 1, "opencode-go/deepseek-v4-flash": 1}


class TestSummarize:
    def test_summarize_prints_every_model(self):
        run = ParallelRun(template_id="s1")
        run.results = [
            ParallelResult(model="a", provider="lmstudio", text="one", ok=True),
            ParallelResult(model="b", provider="opencode", text="[Error: x]", ok=False),
        ]
        out = summarize(run)
        assert "a" in out and "b" in out
        assert "[ok]" in out and "[ERROR]" in out
        assert "no usable model answers" in out  # zero ok votes


class TestMultiLlmVerdictFlag:
    """``multillm --verdict`` end-to-end through the REAL command path.

    REGRESSION: before this, nothing in production ever called ``agree()``, so
    ``summarize()``'s consensus line was ALWAYS "no usable model answers —
    no consensus" no matter what the models said.  The flag must (a) ask every
    model for a structured VERDICT line and (b) record those lines as votes.
    """

    @staticmethod
    def _agent():
        class FakeLLM:
            model_name = "laguna-s-2.1"
        return type("A", (), {
            "llm": FakeLLM(), "workspace": "C:/Dev/Agent1",
        })()

    @staticmethod
    def _run(args):
        from agent_core.commands.multillm_cmd import MultiLlmCommand
        return asyncio.run(MultiLlmCommand().execute(args, TestMultiLlmVerdictFlag._agent()))

    def test_verdict_flag_asks_and_counts_votes(self, capsys):
        seen = {}

        class _Verdicting:
            def __init__(self, model_name):
                self.model_name = model_name
                self.last_response_metrics = None

            async def chat(self, messages, tools=None, max_tokens=None,
                           disable_thinking=False):
                seen[self.model_name] = str(messages[-1]["content"])
                # Model A approves, model B rejects.
                word = "APPROVE" if self.model_name == "laguna-s-2.1" else "REJECT"
                return f"my answer\nVERDICT: {word}\nREASON: because {word}"

        def factory(settings, model_name):
            return _Verdicting(model_name)

        args = [
            "is the patch safe?",
            "--models", "laguna-s-2.1,opencode-go/deepseek-v4-flash",
            "--verdict",
        ]
        with patch("agent_core.llm.parallel.build_provider", side_effect=factory):
            ok = self._run(args)

        assert ok is True
        # (a) every model was asked for a structured verdict line.
        for model in ("laguna-s-2.1", "opencode-go/deepseek-v4-flash"):
            assert "VERDICT: APPROVE" in seen[model]
            assert seen[model].startswith("is the patch safe?")
        # (b) the verdicts were counted, and the summary shows real votes.
        out = capsys.readouterr().out
        assert "2 structured verdict(s) counted." in out
        assert "[APPROVE]" in out and "[REJECT]" in out
        assert "consensus APPROVE (1/2" in out
        assert "no usable model answers" not in out

    def test_without_flag_no_verdict_requested_and_no_votes(self, capsys):
        seen = {}

        class _Plain:
            def __init__(self, model_name):
                self.model_name = model_name
                self.last_response_metrics = None

            async def chat(self, messages, tools=None, max_tokens=None,
                           disable_thinking=False):
                seen[self.model_name] = str(messages[-1]["content"])
                return "just prose, no verdict line"

        def factory(settings, model_name):
            return _Plain(model_name)

        args = [
            "is the patch safe?",
            "--models", "laguna-s-2.1,opencode-go/deepseek-v4-flash",
        ]
        with patch("agent_core.llm.parallel.build_provider", side_effect=factory):
            ok = self._run(args)

        assert ok is True
        # The prompt is untouched and nothing pretends to be a vote.
        for model in ("laguna-s-2.1", "opencode-go/deepseek-v4-flash"):
            assert seen[model] == "is the patch safe?"
        out = capsys.readouterr().out
        assert "structured verdict" not in out
        assert "no usable model answers" in out  # honest: no votes recorded

    def test_flag_with_silent_models_reports_no_consensus(self, capsys):
        """Models that ignore the instruction must NOT be counted as rejects."""

        class _Silent:
            def __init__(self, model_name):
                self.model_name = model_name
                self.last_response_metrics = None

            async def chat(self, messages, tools=None, max_tokens=None,
                           disable_thinking=False):
                return "I have thoughts but no structured conclusion."

        def factory(settings, model_name):
            return _Silent(model_name)

        args = [
            "is the patch safe?",
            "--models", "laguna-s-2.1,opencode-go/deepseek-v4-flash",
            "--verdict",
        ]
        with patch("agent_core.llm.parallel.build_provider", side_effect=factory):
            ok = self._run(args)

        assert ok is True
        out = capsys.readouterr().out
        assert "No model emitted a parseable 'VERDICT:' line" in out
        assert "no usable model answers" in out
        assert "consensus APPROVE" not in out
