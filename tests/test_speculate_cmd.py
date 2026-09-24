"""speculate REPL command — probabilistic deliberation exposed at the REPL.

TDD red->green: these tests were written (and watched fail with ImportError)
before ``agent_core/commands/speculate_cmd.py`` existed.

The command must wire the REAL pipeline
(``Orchestrator.dispatch_speculative`` -> ``ProbabilisticOrchestrator.
run_speculative`` -> ``decide_commitment``) so the previously programmatic-
only phase 3/4 machinery is reachable from the interactive command surface.
"""

import asyncio


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
