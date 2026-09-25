"""jev REPL command — typed probabilistic decisions at the REPL.

The command must build the engine from the DEDICATED small model
(``build_jev_engine``) and never touch ``agent.llm``.  No real LLM: the engine
factory is monkeypatched to a fake vote provider.
"""

import asyncio
import json

import pytest

from agent_core.jev_engine import JevEngine


@pytest.fixture(autouse=True)
def _disable_jev_telemetry(monkeypatch):
    """Fake-provider decisions must never pollute the real ledger."""
    monkeypatch.setenv("AGENT_NO_JEV_LOG", "1")


class VoteProvider:
    def __init__(self, answers, model_name="google/gemma-4-e4b"):
        self._answers = list(answers)
        self._i = 0
        self.model_name = model_name

    async def chat(self, messages, tools=None, max_tokens=None, disable_thinking=False):
        answer = self._answers[self._i % len(self._answers)]
        self._i += 1
        return answer


def _patch_engine(monkeypatch, answers, captured=None):
    import agent_core.jev_engine as jev

    def fake_build(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return JevEngine(
            VoteProvider(answers),
            model_name=kwargs.get("model_name") or "google/gemma-4-e4b",
            samples=kwargs.get("samples") or 5,
            mechanism=kwargs.get("mechanism") or "auto",
            threshold=kwargs.get("threshold", 0.7),
        )

    monkeypatch.setattr(jev, "build_jev_engine", fake_build)


def _agent(workspace="C:/Dev/Agent1"):
    return type("A", (), {"workspace": workspace})()


def _run(args, agent=None):
    from agent_core.commands.jev_cmd import JevCommand

    return asyncio.run(JevCommand().execute(args, agent or _agent()))


def test_jev_is_registered():
    from agent import _build_registry

    assert "jev" in _build_registry().names()


def test_jev_requires_a_kind(capsys):
    assert _run(["what?"]) is True
    assert "Usage: jev" in capsys.readouterr().out


def test_jev_yesno_commits(monkeypatch, capsys):
    _patch_engine(monkeypatch, ["yes", "yes", "yes", "no", "yes"])
    assert _run(["yesno", '"is the sky blue?"']) is True
    out = capsys.readouterr().out
    assert "P(yes)=0.80" in out
    assert "TRUE" in out


def test_jev_yesno_json(monkeypatch, capsys):
    _patch_engine(monkeypatch, ["no", "no", "no", "no", "no"])
    assert _run(["yesno", '"is it true?"', "--json"]) is True
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{"):])
    assert payload["decision"] == "FALSE"
    assert payload["probabilities"]["no"] == 1.0


def test_jev_choice_needs_options(monkeypatch, capsys):
    _patch_engine(monkeypatch, ["A"])
    assert _run(["choice", '"pick one"']) is True
    assert "--options" in capsys.readouterr().out


def test_jev_choice_distribution(monkeypatch, capsys):
    _patch_engine(monkeypatch, ["A", "A", "B", "A", "B"])
    assert _run(
        ["choice", "--options", "red|green", '"pick"', "--threshold", "0.5"]
    ) is True
    out = capsys.readouterr().out
    assert "P(red)=0.60" in out
    assert "red" in out


def test_jev_rejects_bad_threshold(monkeypatch, capsys):
    _patch_engine(monkeypatch, ["yes"])
    assert _run(["yesno", '"q"', "--threshold", "2"]) is True
    out = capsys.readouterr().out
    assert "threshold" in out
    assert "P(yes)" not in out


def test_jev_forwards_model_and_mechanism(monkeypatch, capsys):
    captured = {}
    _patch_engine(monkeypatch, ["yes"], captured=captured)
    _run(["yesno", '"q"', "--model", "google/gemma-4-e4b", "--mechanism", "vote",
          "--samples", "3"])
    assert captured["model_name"] == "google/gemma-4-e4b"
    assert captured["mechanism"] == "vote"
    assert captured["samples"] == 3


def test_jev_reads_state_file(monkeypatch, tmp_path, capsys):
    (tmp_path / "ctx.txt").write_text("The repo uses pytest.", encoding="utf-8")
    seen = {}

    import agent_core.jev_engine as jev

    class RecordingEngine:
        model_name = "google/gemma-4-e4b"
        mechanism = "vote"
        samples = 5

        async def decide(self, question, state=""):
            seen["state"] = state
            from agent_core.jev_engine import JevResult
            return JevResult(kind="yesno", model="m", mechanism="vote",
                             decision="TRUE", confidence=1.0)

    monkeypatch.setattr(jev, "build_jev_engine", lambda **kw: RecordingEngine())
    assert _run(["yesno", '"is pytest used?"', "--file", "ctx.txt"],
                _agent(str(tmp_path))) is True
    assert "pytest" in seen["state"]


def test_jev_missing_state_file_errors(monkeypatch, tmp_path, capsys):
    _patch_engine(monkeypatch, ["yes"])
    assert _run(["yesno", '"q"', "--file", "nope.txt"], _agent(str(tmp_path))) is True
    assert "not found" in capsys.readouterr().out


def test_jev_engine_build_failure_is_reported(monkeypatch, capsys):
    import agent_core.jev_engine as jev

    def boom(**kwargs):
        raise ValueError("no Jev model configured")

    monkeypatch.setattr(jev, "build_jev_engine", boom)
    assert _run(["yesno", '"q"']) is True
    assert "could not build" in capsys.readouterr().out


def test_jev_does_not_touch_agent_llm(monkeypatch, capsys):
    """The dedicated small model is used — agent.llm must never be called."""
    _patch_engine(monkeypatch, ["yes"])

    class MainLLM:
        model_name = "main-model"

        async def chat(self, *a, **k):  # pragma: no cover - must not run
            raise AssertionError("agent.llm was used by jev")

    agent = type("A", (), {"workspace": "C:/Dev/Agent1", "llm": MainLLM()})()
    assert _run(["yesno", '"q"'], agent) is True
    assert "P(yes)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
#  NLP tool jev_decide
# ---------------------------------------------------------------------------


def test_jev_decide_is_a_known_read_only_tool():
    from agent_core.modes import PLAN_MODE_TOOLS, filter_tool_schemas
    from agent_core.tool_schemas import NLP_TOOL_NAMES, NLP_TOOL_SCHEMAS

    assert "jev_decide" in NLP_TOOL_NAMES
    assert "jev_decide" in PLAN_MODE_TOOLS
    plan_names = {
        s["function"]["name"] for s in filter_tool_schemas(NLP_TOOL_SCHEMAS, "plan")
    }
    assert "jev_decide" in plan_names


def test_jev_decide_handler_dispatch_registered():
    from agent import Agent

    handlers = Agent.__new__(Agent)._nlp_tool_handlers()
    assert "jev_decide" in handlers


def test_jev_decide_handler_uses_small_model(monkeypatch):
    _patch_engine(monkeypatch, ["yes", "yes", "yes", "yes", "yes"])
    from agent import Agent

    out = asyncio.run(
        Agent.__new__(Agent)._nlp_jev_decide(
            {"kind": "yesno", "question": "is the build green?"},
        )
    )
    assert "P(yes)=1.00" in out
    assert "TRUE" in out


def test_jev_decide_handler_validates_question():
    from agent import Agent

    out = asyncio.run(
        Agent.__new__(Agent)._nlp_jev_decide({"kind": "yesno", "question": "  "})
    )
    assert "requires 'question'" in out


def test_system_prompt_steers_the_agent_to_jev():
    """The main model only uses jev_decide if the system prompt tells it to.

    Regression: the tool existed but was never mentioned, so the reasoning
    model never exploited the cheap typed decisions.
    """
    from agent import _SYSTEM_PROMPT

    assert "jev_decide" in _SYSTEM_PROMPT
    assert "yesno" in _SYSTEM_PROMPT
    assert "choice" in _SYSTEM_PROMPT
    assert "score" in _SYSTEM_PROMPT
    assert "probability" in _SYSTEM_PROMPT.lower()
    # And it must constrain usage to bounded judgements, not generation.
    assert "decision model" in _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
#  jev stats (calibration report)
# ---------------------------------------------------------------------------


def _seed_ledger(tmp_path):
    from agent_core.jev_engine import JevQuestion, JevResult
    from harnessfix.jev_telemetry import record_decision, record_outcome

    question = JevQuestion(kind="yesno", text="is the sky blue?")
    for p_yes, outcome in ((0.9, "correct"), (0.8, "correct"),
                           (0.2, "incorrect"), (0.6, "incorrect")):
        result = JevResult(
            kind="yesno", model="qwen2.5-coder-1.5b-instruct", mechanism="vote",
            probabilities={"yes": p_yes, "no": 1.0 - p_yes},
            decision="TRUE" if p_yes >= 0.7 else "FALSE",
            confidence=max(p_yes, 1.0 - p_yes), threshold=0.7, n=5,
        )
        decision_id = record_decision(
            result, question, workspace=str(tmp_path), source="repl",
        )
        record_outcome(decision_id, outcome, workspace=str(tmp_path))


def test_jev_stats_reports_calibration(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("AGENT_NO_JEV_LOG", raising=False)  # seed the ledger
    _seed_ledger(tmp_path)
    assert _run(["stats"], _agent(str(tmp_path))) is True
    out = capsys.readouterr().out
    assert "decisions=4" in out
    assert "labeled=4" in out
    assert "suggested yesno threshold" in out


def test_jev_stats_json(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("AGENT_NO_JEV_LOG", raising=False)  # seed the ledger
    _seed_ledger(tmp_path)
    assert _run(["stats", "--json"], _agent(str(tmp_path))) is True
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{"):])
    assert payload["total"] == 4
    assert payload["labeled"] == 4
    assert payload["suggested_threshold"]["accuracy"] == 1.0


def test_jev_stats_empty_explains_labeling(tmp_path, capsys):
    assert _run(["stats"], _agent(str(tmp_path))) is True
    out = capsys.readouterr().out
    assert "decisions=0" in out
    assert "record_outcome" in out


# ---------------------------------------------------------------------------
#  jev label (calibration outcomes)
# ---------------------------------------------------------------------------


def test_jev_label_records_outcome(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("AGENT_NO_JEV_LOG", raising=False)  # seed the ledger
    _seed_ledger(tmp_path)
    from harnessfix.jev_telemetry import load_decisions

    decision_id = load_decisions(workspace=str(tmp_path))[0]["id"]
    assert _run(["label", decision_id, "correct"], _agent(str(tmp_path))) is True
    out = capsys.readouterr().out
    assert "labeled" in out
    assert load_decisions(workspace=str(tmp_path))[0]["outcome"] == "correct"


def test_jev_label_unknown_id(tmp_path, capsys):
    assert _run(["label", "ghost", "correct"], _agent(str(tmp_path))) is True
    assert "not found" in capsys.readouterr().out


def test_jev_label_requires_an_outcome(tmp_path, capsys):
    assert _run(["label", "some-id"], _agent(str(tmp_path))) is True
    assert "Usage: jev label" in capsys.readouterr().out



