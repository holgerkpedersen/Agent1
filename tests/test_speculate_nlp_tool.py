"""speculate as an NLP tool — the LLM's on-demand speculative deliberation.

The main model must be able to reach the SAME pipeline as the REPL
``speculate`` command (parallel read-only-grounded branches + judge scoring
+ COMMIT/REFUSE gate) from inside a conversation, so it can get a fast,
independently corroborated answer instead of one long reasoning chain —
the LLM-side twin of ``jev_decide`` for questions that need an ANSWER, not
just a typed probability.

The functional tests drive the REAL pipeline (Orchestrator thread pool ->
ProbabilisticOrchestrator.run_speculative) with a fake LLM and assert on the
captured tool result — including that nothing leaks to REPL stdout.
"""

import asyncio


class FakeLLM:
    """Branch + judge responder.

    Branch prompts:   "Speculative branch {i}. {question}"
    Judge prompts:    "Quality judge: ... answer: {answer}"

    Same contract as the FakeLLM in tests/test_speculate_cmd.py so the tool
    path and the REPL path are held to identical expectations.
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


def _bare_agent():
    """Minimal Agent stand-in for the handler (same trick as test_jev_cmd).

    ``SpeculateCommand.execute`` only needs ``agent.llm`` and (optionally)
    ``agent.workspace`` / ``agent._refresh_system_message``; everything else
    is getattr-guarded inside the command.
    """
    from agent import Agent

    a = Agent.__new__(Agent)
    a.llm = FakeLLM()
    a.workspace = "C:/Dev/Agent1"
    return a


# ---------------------------------------------------------------------------
#  Registration / schema / plan-mode surface
# ---------------------------------------------------------------------------


def test_speculate_is_a_known_tool():
    from agent_core.modes import filter_tool_schemas
    from agent_core.tool_schemas import NLP_TOOL_NAMES, NLP_TOOL_SCHEMAS

    assert "speculate" in NLP_TOOL_NAMES
    # Read-only (branch allowlist) -> must survive plan-mode filtering.
    plan_names = {
        s["function"]["name"] for s in filter_tool_schemas(NLP_TOOL_SCHEMAS, "plan")
    }
    assert "speculate" in plan_names


def test_speculate_handler_dispatch_registered():
    from agent import Agent

    handlers = Agent.__new__(Agent)._nlp_tool_handlers()
    assert "speculate" in handlers


def test_schema_documents_the_speed_knobs():
    from agent_core.tool_schemas import NLP_TOOL_SCHEMAS

    schema = next(
        s for s in NLP_TOOL_SCHEMAS if s["function"]["name"] == "speculate"
    )
    params = schema["function"]["parameters"]
    assert set(params["properties"]) >= {
        "question", "branches", "judge", "branch_model", "threshold",
    }
    assert params["required"] == ["question"]
    # The model must be told the fast paths exist.
    desc = schema["function"]["description"]
    assert "jev" in desc
    assert "parallel" in desc.lower()


# ---------------------------------------------------------------------------
#  Validation (no LLM traffic)
# ---------------------------------------------------------------------------


def test_requires_a_question():
    out = asyncio.run(_bare_agent()._nlp_speculate({"question": "   "}))
    assert "requires 'question'" in out


def test_rejects_flag_like_question():
    # A question starting with '--' would be eaten as a command flag.
    out = asyncio.run(
        _bare_agent()._nlp_speculate({"question": "--branches 2"})
    )
    assert "command flag" in out


def test_rejects_unknown_judge():
    out = asyncio.run(
        _bare_agent()._nlp_speculate({"question": "q", "judge": "vibes"})
    )
    assert "llm, jev, both" in out


def test_rejects_unknown_branch_model():
    out = asyncio.run(
        _bare_agent()._nlp_speculate(
            {"question": "q", "branch_model": "quantum"}
        )
    )
    assert "chat, jev" in out


def test_rejects_threshold_out_of_range():
    out = asyncio.run(
        _bare_agent()._nlp_speculate({"question": "q", "threshold": 2})
    )
    assert "[0.0, 1.0]" in out


# ---------------------------------------------------------------------------
#  End-to-end through the REAL pipeline (fake LLM)
# ---------------------------------------------------------------------------


def test_tool_commits_best_branch(capsys):
    out = asyncio.run(
        _bare_agent()._nlp_speculate({"question": "which branch is best?"})
    )
    assert "[speculate]" in out  # header with judge/branch model info
    assert "COMMIT" in out
    assert "good answer 0" in out
    # The verdict must be CAPTURED into the tool result, not printed to the
    # REPL stdout — a human at the prompt should not see LLM-internal traffic.
    assert "[speculate]" not in capsys.readouterr().out


def test_tool_refuses_when_all_below_threshold():
    out = asyncio.run(
        _bare_agent()._nlp_speculate({"question": "rate this [bad] idea"})
    )
    assert "REFUSE" in out
    assert "good answer" not in out


def test_tool_passes_speed_knobs_to_the_command(monkeypatch):
    """branches/judge/branch_model/threshold must reach the command as CLI."""
    from agent_core.commands import speculate_cmd

    seen = {}

    async def fake_execute(self, args, agent):
        seen["args"] = list(args)
        print("[speculate] fake run")
        return True

    monkeypatch.setattr(
        speculate_cmd.SpeculateCommand, "execute", fake_execute, raising=True,
    )
    out = asyncio.run(_bare_agent()._nlp_speculate({
        "question": "q?",
        "branches": 5,
        "judge": "jev",
        "branch_model": "jev",
        "threshold": 0.42,
    }))
    assert "fake run" in out
    args = seen["args"]
    assert args[0] == "q?"
    assert "--branches" in args and args[args.index("--branches") + 1] == "5"
    assert "--judge" in args and args[args.index("--judge") + 1] == "jev"
    assert (
        "--branch-model" in args
        and args[args.index("--branch-model") + 1] == "jev"
    )
    assert (
        "--threshold" in args
        and float(args[args.index("--threshold") + 1]) == 0.42
    )


def test_system_prompt_steers_the_agent_to_speculate():
    """The main model only uses speculate if the system prompt tells it to."""
    from agent import _SYSTEM_PROMPT

    assert "speculate" in _SYSTEM_PROMPT
    assert "COMMITs only the best when it meets the threshold" in _SYSTEM_PROMPT
    # And it must frame it as the fast/parallel alternative to long reasoning.
    assert "chain of thought" in _SYSTEM_PROMPT
    assert "judge='jev'" in _SYSTEM_PROMPT
