"""Jev decision engine — typed probabilistic answers from a dedicated small model.

Covers the pure parsers, the voting/logprobs mechanisms, abstention/UNKNOWN
semantics, and that the engine binds to ``settings.jev_model`` (never the
agent's main model).

No real LLM / network: providers are tiny fakes.
"""

import asyncio
import math
from types import SimpleNamespace

import pytest

from agent_core.jev_engine import (
    KIND_CHOICE,
    KIND_SCORE,
    KIND_YESNO,
    MECHANISM_LOGPROBS,
    MECHANISM_VOTE,
    JevEngine,
    JevQuestion,
    JevResult,
    build_jev_engine,
    is_failure,
    parse_choice,
    parse_score,
    parse_yesno,
)

_ABC = ("alpha", "beta", "gamma")


@pytest.fixture(autouse=True)
def _disable_jev_telemetry(monkeypatch):
    """Fake-provider decisions must never pollute the real ledger."""
    monkeypatch.setenv("AGENT_NO_JEV_LOG", "1")

# ---------------------------------------------------------------------------
#  Fakes
# ---------------------------------------------------------------------------


class VoteProvider:
    """Returns one scripted answer per ``chat`` call, cycling."""

    def __init__(self, answers, model_name="google/gemma-4-e4b"):
        self._answers = list(answers)
        self._i = 0
        self.model_name = model_name

    async def chat(
        self, messages, tools=None, max_tokens=None, disable_thinking=False,
    ):
        answer = self._answers[self._i % len(self._answers)]
        self._i += 1
        return answer


class LogprobsProvider:
    """Provides only the logprobs readout."""

    model_name = "google/gemma-4-e4b"

    def __init__(self, top):
        self._top = top
        self.chat_calls = 0

    async def chat(self, messages, **kwargs):  # pragma: no cover
        self.chat_calls += 1
        return "yes"

    async def chat_logprobs(self, messages, max_tokens=None, top_logprobs=20):
        return "yes", self._top


def _engine(provider, **kwargs):
    kwargs.setdefault("samples", 5)
    return JevEngine(provider, model_name="google/gemma-4-e4b", **kwargs)


def _decide(engine, question, state=""):
    return asyncio.run(engine.decide(question, state))


# ---------------------------------------------------------------------------
#  Parsers
# ---------------------------------------------------------------------------


class TestParsers:
    @pytest.mark.parametrize(
        "reply,expected",
        [
            ("yes", "yes"), ("Yes.", "yes"), ("YES", "yes"),
            ("yes, definitely", "yes"), ("true", "yes"), ("1", "yes"),
            ("no", "no"), ("No!", "no"), ("false", "no"), ("0", "no"),
            ("The proposition is false", "no"),
            ("I cannot answer that", None),
            ("", None),
        ],
    )
    def test_parse_yesno(self, reply, expected):
        assert parse_yesno(reply) == expected

    def test_parse_yesno_rejects_failures(self):
        assert parse_yesno("[Error: HTTP Error 400: no models loaded]") is None
        assert parse_yesno('<|tool_call>call:run{command:"x"}<tool_call|>') is None

    def test_parse_choice(self):
        opts = ("alpha", "beta", "gamma")
        assert parse_choice("A", opts) == "alpha"
        assert parse_choice("b", opts) == "beta"
        assert parse_choice("B)", opts) == "beta"
        assert parse_choice("2", opts) == "beta"
        assert parse_choice("Option B", opts) == "beta"
        assert parse_choice("gamma", opts) == "gamma"
        assert parse_choice("The answer is alpha", opts) == "alpha"
        assert parse_choice("i don't know", opts) is None

    def test_parse_score(self):
        assert parse_score("85", 100.0) == 85.0
        assert parse_score("0.85", 100.0) == 85.0
        assert parse_score("100", 100.0) == 100.0
        assert parse_score("150", 100.0) == 100.0  # clamped
        assert parse_score("-5", 100.0) == 0.0  # clamped
        assert parse_score("no idea", 100.0) is None
        assert parse_score("[Error: boom]", 100.0) is None

    def test_is_failure(self):
        assert is_failure("[Error: x]")
        assert is_failure("<|tool_call>call:read{}")
        assert is_failure('{"tool_calls": []}') is False  # not our marker
        assert not is_failure("yes")


# ---------------------------------------------------------------------------
#  Question validation
# ---------------------------------------------------------------------------


class TestQuestion:
    def test_bad_kind_raises(self):
        with pytest.raises(ValueError):
            JevQuestion(kind="magic", text="x")

    def test_empty_text_raises(self):
        with pytest.raises(ValueError):
            JevQuestion(kind=KIND_YESNO, text="   ")

    def test_choice_needs_two_options(self):
        with pytest.raises(ValueError):
            JevQuestion(kind=KIND_CHOICE, text="pick", options=("only",))

    def test_score_scale_must_be_positive(self):
        with pytest.raises(ValueError):
            JevQuestion(kind=KIND_SCORE, text="rate", max_value=0)


# ---------------------------------------------------------------------------
#  Voting mechanism
# ---------------------------------------------------------------------------


class TestVoting:
    def test_yesno_agreement_is_probability(self):
        engine = _engine(VoteProvider(["yes", "yes", "yes", "no", "yes"]))
        result = _decide(engine, JevQuestion(kind=KIND_YESNO, text="is it true?"))
        assert result.mechanism == MECHANISM_VOTE
        assert result.probabilities["yes"] == pytest.approx(0.8)
        assert result.probabilities["no"] == pytest.approx(0.2)
        assert result.decision == "TRUE"
        assert result.confidence == pytest.approx(0.8)
        assert result.n == 5
        assert result.abstentions == 0

    def test_yesno_below_threshold_is_false(self):
        engine = _engine(VoteProvider(["yes", "no", "no", "no", "no"]), threshold=0.7)
        result = _decide(engine, JevQuestion(kind=KIND_YESNO, text="q"))
        assert result.decision == "FALSE"
        assert result.probabilities["yes"] == pytest.approx(0.2)

    def test_majority_abstention_is_unknown(self):
        engine = _engine(VoteProvider(["yes", "yes", "??", "??", "??"]))
        result = _decide(engine, JevQuestion(kind=KIND_YESNO, text="q"))
        assert result.decision == "UNKNOWN"
        assert result.abstentions == 3

    def test_provider_errors_are_abstentions_not_votes(self):
        err = '[Error: HTTP Error 400: {"message": "No models loaded."}]'
        engine = _engine(VoteProvider([err] * 5))
        result = _decide(engine, JevQuestion(kind=KIND_YESNO, text="q"))
        assert result.decision == "UNKNOWN"
        assert result.abstentions == 5
        assert result.probabilities == {}
        assert result.error

    def test_leaked_tool_call_is_abstention(self):
        leak = '<|tool_call>call:run{command:"pytest"}<tool_call|>'
        engine = _engine(VoteProvider([leak] * 5))
        result = _decide(engine, JevQuestion(kind=KIND_YESNO, text="q"))
        assert result.decision == "UNKNOWN"

    def test_choice_distribution_and_winner(self):
        engine = _engine(VoteProvider(["A", "A", "A", "B", "C"]), threshold=0.5)
        question = JevQuestion(kind=KIND_CHOICE, text="pick", options=_ABC)
        result = _decide(engine, question)
        assert result.probabilities == {
            "alpha": pytest.approx(0.6),
            "beta": pytest.approx(0.2),
            "gamma": pytest.approx(0.2),
        }
        assert result.decision == "alpha"
        assert result.confidence == pytest.approx(0.6)

    def test_choice_below_threshold_is_undecided(self):
        engine = _engine(VoteProvider(["A", "B", "C", "A", "B"]), threshold=0.7)
        question = JevQuestion(kind=KIND_CHOICE, text="pick", options=_ABC)
        result = _decide(engine, question)
        assert result.decision == "UNDECIDED"

    def test_score_mean_and_spread(self):
        engine = _engine(VoteProvider(["80", "90", "100", "70", "60"]))
        result = _decide(engine, JevQuestion(kind=KIND_SCORE, text="rate it"))
        assert result.kind == KIND_SCORE
        assert result.value == pytest.approx(0.8)
        assert result.confidence == pytest.approx(0.8)
        assert result.spread == pytest.approx(0.1414, abs=1e-3)

    def test_score_all_abstentions_is_unknown(self):
        engine = _engine(VoteProvider(["n/a"] * 5))
        result = _decide(engine, JevQuestion(kind=KIND_SCORE, text="rate it"))
        assert result.decision == "UNKNOWN"
        assert result.value is None


# ---------------------------------------------------------------------------
#  Logprobs mechanism
# ---------------------------------------------------------------------------


class TestLogprobs:
    def test_logprobs_distribution(self):
        top = [
            {"token": "yes", "logprob": math.log(0.8)},
            {"token": "no", "logprob": math.log(0.2)},
        ]
        engine = _engine(LogprobsProvider(top))
        result = _decide(engine, JevQuestion(kind=KIND_YESNO, text="q"))
        assert result.mechanism == MECHANISM_LOGPROBS
        assert result.probabilities["yes"] == pytest.approx(0.8, abs=1e-6)
        assert result.decision == "TRUE"
        assert result.n == 1

    def test_choice_logprobs(self):
        top = [
            {"token": "A", "logprob": math.log(0.6)},
            {"token": "B", "logprob": math.log(0.3)},
            {"token": "C", "logprob": math.log(0.1)},
        ]
        engine = _engine(LogprobsProvider(top), threshold=0.5)
        question = JevQuestion(kind=KIND_CHOICE, text="pick", options=_ABC)
        result = _decide(engine, question)
        assert result.mechanism == MECHANISM_LOGPROBS
        assert result.probabilities["alpha"] == pytest.approx(0.6, abs=1e-6)
        assert result.decision == "alpha"

    def test_missing_logprobs_falls_back_to_vote(self):
        provider = LogprobsProvider(None)
        engine = _engine(provider)
        result = _decide(engine, JevQuestion(kind=KIND_YESNO, text="q"))
        assert result.mechanism == MECHANISM_VOTE
        assert provider.chat_calls == 5  # fell back to the default 5 samples

    def test_score_never_uses_logprobs(self):
        provider = LogprobsProvider([{"token": "8", "logprob": math.log(0.9)}])
        engine = _engine(provider)
        result = _decide(engine, JevQuestion(kind=KIND_SCORE, text="rate"))
        assert result.mechanism == MECHANISM_VOTE

    def test_explicit_logprobs_falls_back_when_unsupported(self):
        engine = _engine(VoteProvider(["yes"]), mechanism=MECHANISM_LOGPROBS)
        result = _decide(engine, JevQuestion(kind=KIND_YESNO, text="q"))
        assert result.mechanism == MECHANISM_VOTE
        assert result.decision == "TRUE"


# ---------------------------------------------------------------------------
#  Result rendering
# ---------------------------------------------------------------------------


class TestResult:
    def test_summary_and_dict(self):
        result = JevResult(
            kind=KIND_YESNO, model="m", mechanism=MECHANISM_VOTE,
            probabilities={"yes": 0.8, "no": 0.2}, decision="TRUE",
            confidence=0.8, n=5, abstentions=1,
        )
        text = result.summary()
        assert "P(yes)=0.80" in text
        assert "TRUE" in text
        data = result.as_dict()
        assert data["decision"] == "TRUE"
        assert data["probabilities"]["yes"] == 0.8


# ---------------------------------------------------------------------------
#  Provider binding — the small model is Jev-only
# ---------------------------------------------------------------------------


class TestBuildEngine:
    def test_builds_provider_from_jev_model(self, monkeypatch):
        import agent_core.llm.provider as prov

        calls = {}

        def fake_build(settings, model, provider_override=None, **kwargs):
            calls["model"] = model
            calls["override"] = provider_override
            calls["single"] = kwargs.get("single")
            return VoteProvider(["yes"])

        monkeypatch.setattr(prov, "build_provider", fake_build)
        settings = SimpleNamespace(
            jev_model="google/gemma-4-e4b", jev_provider="", jev_samples=3,
            jev_temperature=0.5, jev_max_tokens=4, jev_timeout=10.0,
        )
        engine = build_jev_engine(settings=settings)
        assert calls["model"] == "google/gemma-4-e4b"
        assert calls["override"] is None
        assert calls["single"] is True  # no failover chain for Jev
        assert engine.samples == 3
        assert engine.model_name == "google/gemma-4-e4b"

    def test_explicit_provider_override_is_forwarded(self, monkeypatch):
        import agent_core.llm.provider as prov

        calls = {}

        def fake_build(settings, model, provider_override=None, **kwargs):
            calls["override"] = provider_override
            return VoteProvider(["yes"])

        monkeypatch.setattr(prov, "build_provider", fake_build)
        settings = SimpleNamespace(
            jev_model="x", jev_provider="llama", jev_samples=1,
            jev_temperature=0.7, jev_max_tokens=8, jev_timeout=5.0,
        )
        build_jev_engine(settings=settings)
        assert calls["override"] == "llama"

    def test_injected_provider_skips_build(self):
        engine = build_jev_engine(
            settings=SimpleNamespace(jev_samples=2, jev_temperature=0.7,
                                     jev_max_tokens=8, jev_timeout=5.0),
            provider=VoteProvider(["no"]),
            model_name="small",
        )
        assert engine.model_name == "small"
        assert engine.samples == 2

    def test_no_model_configured_raises(self, monkeypatch):
        import agent_core.jev_engine as jev

        monkeypatch.setattr(jev, "DEFAULT_JEV_MODEL", "")
        settings = SimpleNamespace(
            jev_model="", jev_provider="", jev_samples=1,
            jev_temperature=0.7, jev_max_tokens=8, jev_timeout=5.0,
        )
        with pytest.raises(ValueError):
            build_jev_engine(settings=settings)

    def test_build_engine_pins_one_provider_no_failover(self):
        """Regression: build_jev_engine used the failover chain, so a Jev call
        silently drifted to DeepSeek/OpenRouter/the 27B when LM Studio was
        unreachable — the "small model for Jev only" invariant was broken.
        Jev must build ONE provider pinned to the small model."""
        settings = SimpleNamespace(
            jev_model="qwen2.5-coder-1.5b-instruct", jev_provider="",
            jev_samples=1, jev_temperature=0.7, jev_max_tokens=8,
            jev_timeout=5.0,
            llm_providers=(
                "opencode:opencode-go/deepseek-v4.1-flash",
                "openrouter", "lmstudio", "llama",
            ),
            llm_provider="opencode", failover_strategy="ordered",
        )
        engine = build_jev_engine(settings=settings)
        assert type(engine.provider).__name__ == "LMStudioProvider"
        assert not hasattr(engine.provider, "providers")  # not a FailoverProvider
        assert engine.provider.model_name == "qwen2.5-coder-1.5b-instruct"

    def test_build_provider_single_is_concrete(self):
        from agent_core.llm.provider import build_provider

        settings = SimpleNamespace(
            llm_providers=("opencode:opencode-go/deepseek-v4.1-flash",
                           "lmstudio", "llama"),
            llm_provider="opencode", failover_strategy="ordered",
        )
        provider = build_provider(
            settings, "qwen2.5-coder-1.5b-instruct", single=True,
        )
        assert type(provider).__name__ == "LMStudioProvider"
        assert provider.model_name == "qwen2.5-coder-1.5b-instruct"


class TestEnsureReady:
    """The engine auto-selects its dedicated model before the first request."""

    class _Provider:
        model_name = "fake-small"

        def __init__(self, ensure_result=(True, "already loaded")):
            self._ensure_result = ensure_result
            self.ensure_calls = 0

        def apply_profile(self, *args):
            pass

        def ensure_model_loaded(self):
            self.ensure_calls += 1
            return self._ensure_result

        async def chat(self, messages, **kwargs):
            return "yes"

    def test_ensure_called_once_per_engine(self):
        provider = self._Provider()
        engine = JevEngine(
            provider, model_name="fake-small", samples=1, mechanism="vote",
        )
        question = JevQuestion(kind=KIND_YESNO, text="q")
        asyncio.run(engine.decide(question))
        asyncio.run(engine.decide(question))
        assert provider.ensure_calls == 1

    def test_failed_select_warns_but_still_decides(self, capsys):
        provider = self._Provider(ensure_result=(False, "model not found"))
        engine = JevEngine(
            provider, model_name="fake-small", samples=1, mechanism="vote",
        )
        result = asyncio.run(engine.decide(JevQuestion(kind=KIND_YESNO, text="q")))
        assert result.decision == "TRUE"
        out = capsys.readouterr().out
        assert "could not select model" in out
        assert "model not found" in out

    def test_provider_without_hook_is_fine(self):
        engine = JevEngine(VoteProvider(["yes"]), samples=1, mechanism="vote")
        result = asyncio.run(engine.decide(JevQuestion(kind=KIND_YESNO, text="q")))
        assert result.decision == "TRUE"


class TestSettings:
    """The Jev settings default to the catalog's dedicated small model."""

    def test_defaults_present(self):
        from agent_core.config import AgentSettings

        settings = AgentSettings()
        assert settings.jev_model == "qwen2.5-coder-1.5b-instruct"
        assert settings.jev_provider == ""
        assert settings.jev_samples == 5
        assert settings.jev_temperature == pytest.approx(0.7)
        assert settings.jev_max_tokens == 512
        assert settings.jev_timeout == pytest.approx(60.0)

    def test_invalid_provider_rejected(self):
        from agent_core.config import AgentSettings, ConfigurationError

        with pytest.raises(ConfigurationError):
            AgentSettings(jev_provider="bogus")

    def test_nonpositive_samples_rejected(self):
        from agent_core.config import AgentSettings, ConfigurationError

        with pytest.raises(ConfigurationError):
            AgentSettings(jev_samples=0)

