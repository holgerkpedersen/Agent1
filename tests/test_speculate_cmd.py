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
    def __init__(self, tool_name, tool_result="file contents"):
        self.llm = _ToolLLM(tool_name)
        self.executed = []
        self.tool_result = tool_result

    def _refresh_system_message(self):
        pass

    @property
    def _chat_history(self):
        return [{"role": "system", "content": "SYS"}]

    async def _execute_tool_call(self, name, args):
        self.executed.append((name, args))
        return self.tool_result


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


# ---------------------------------------------------------------------------
#  Grounding + claim verification
# ---------------------------------------------------------------------------


def test_is_repo_question():
    from agent_core.commands.speculate_cmd import _is_repo_question

    assert _is_repo_question("How does agent_core/jev_engine.py work?")
    assert _is_repo_question(
        "What is the highest-value improvement to the Jev integration?"
    )
    assert not _is_repo_question("What is the capital of France?")


def test_question_subject_terms_keeps_distinctive_words():
    from agent_core.commands.speculate_cmd import _question_subject_terms

    terms = _question_subject_terms(
        "What is the single highest-value improvement to the Jev integration "
        "right now?"
    )
    assert "jev" in terms
    assert "integration" in terms
    assert "improvement" not in terms  # stopword
    assert "highest" not in terms      # stopword
    assert "jev_engine" in _question_subject_terms(
        "How does agent_core/jev_engine.py work?"
    )


def test_grounded_in_subject():
    from agent_core.commands.speculate_cmd import _grounded_in_subject

    assert _grounded_in_subject(["agent_core/jev_engine.py ..."], {"jev"})
    assert not _grounded_in_subject(["nothing relevant"], {"jev"})
    assert not _grounded_in_subject([], {"jev"})
    assert _grounded_in_subject(["anything"], set())  # no terms -> any tool


def test_verify_claims_checks_cited_line_content(tmp_path):
    from agent_core.commands.speculate_cmd import _verify_claims

    lines = ["x = 1"] + [f"y{i} = {i}" for i in range(2, 10)] + ["branch_llm = 10"]
    (tmp_path / "mod.py").write_text("\n".join(lines), encoding="utf-8")
    assert _verify_claims("`branch_llm` is set at `mod.py:10`", str(tmp_path)) == []
    problems = _verify_claims("`branch_llm` is set at `mod.py:1`", str(tmp_path))
    assert problems
    assert "does not mention branch_llm" in problems[0]


def test_verify_claims_checks_workspace(tmp_path):
    from agent_core.commands.speculate_cmd import _verify_claims

    (tmp_path / "real.py").write_text("line1\nline2\n", encoding="utf-8")
    assert _verify_claims("see real.py:2", str(tmp_path)) == []
    assert _verify_claims("see real.py:99", str(tmp_path))  # past EOF
    assert _verify_claims("see ghost.py:1", str(tmp_path))  # missing file
    assert _verify_claims("no file claims here", str(tmp_path)) == []


def test_verify_claims_resolves_bare_basename(tmp_path):
    from agent_core.commands.speculate_cmd import _verify_claims

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "mod.py").write_text("a\nb\n", encoding="utf-8")
    assert _verify_claims("see mod.py:2", str(tmp_path)) == []  # found in pkg/
    assert _verify_claims("see mod.py:99", str(tmp_path))  # past EOF
    assert _verify_claims("see ghost.py:1", str(tmp_path))  # nowhere


def test_speculate_requires_grounding_for_repo_questions(capsys):
    """A repo question answered from memory is not a candidate — the 27B did
    exactly that and a same-model judge scored it 1.00."""
    from agent_core.commands.speculate_cmd import SpeculateCommand

    class UngroundedLLM:
        model_name = "fake-model"

        async def chat(self, messages, tools=None, **kwargs):
            content = str(messages[-1].get("content") or "")
            if content.startswith("Quality judge:"):
                return "0.9"
            return "The Jev engine uses sample votes."  # never calls a tool

    agent = type("A", (), {"llm": UngroundedLLM(), "workspace": "C:/Dev/Agent1"})()
    assert asyncio.run(SpeculateCommand().execute(
        ['"How does agent_core/jev_engine.py work?"'], agent,
    )) is True
    out = capsys.readouterr().out
    assert "grounding=on" in out
    assert "REFUSE" in out
    assert "COMMIT" not in out


def test_speculate_refuse_reports_branch_failure_reasons(capsys):
    """A bare REFUSE is opaque — the disqualification reasons must be shown."""
    from agent_core.commands.speculate_cmd import SpeculateCommand

    class UngroundedLLM:
        model_name = "fake-model"

        async def chat(self, messages, tools=None, **kwargs):
            content = str(messages[-1].get("content") or "")
            if content.startswith("Quality judge:"):
                return "0.9"
            return "The Jev engine uses sample votes."

    agent = type("A", (), {"llm": UngroundedLLM(), "workspace": "C:/Dev/Agent1"})()
    assert asyncio.run(SpeculateCommand().execute(
        ['"How does agent_core/jev_engine.py work?"'], agent,
    )) is True
    out = capsys.readouterr().out
    assert "REFUSE" in out
    assert "branch failures (deterministic guards):" in out
    assert "without calling any tool" in out


def test_speculate_no_grounding_flag_allows_ungrounded(capsys):
    from agent_core.commands.speculate_cmd import SpeculateCommand

    class UngroundedLLM:
        model_name = "fake-model"

        async def chat(self, messages, tools=None, **kwargs):
            content = str(messages[-1].get("content") or "")
            if content.startswith("Quality judge:"):
                return "0.9"
            return "The Jev engine uses sample votes."

    agent = type("A", (), {"llm": UngroundedLLM(), "workspace": "C:/Dev/Agent1"})()
    assert asyncio.run(SpeculateCommand().execute(
        ['"How does agent_core/jev_engine.py work?"', "--no-grounding"], agent,
    )) is True
    out = capsys.readouterr().out
    assert "grounding=off" in out
    assert "COMMIT" in out


def test_speculate_grounded_branch_can_commit(capsys):
    from agent_core.commands.speculate_cmd import SpeculateCommand

    agent = _ToolAgent(
        "read", tool_result="agent_core/jev_engine.py: sample votes + logprobs",
    )
    asyncio.run(SpeculateCommand().execute(
        ['"How does agent_core/jev_engine.py work?"'], agent,
    ))
    assert agent.executed  # the branch grounded itself
    out = capsys.readouterr().out
    assert "COMMIT" in out
    assert "grounded answer" in out


def test_speculate_rejects_irrelevant_grounding(capsys):
    """A tool call that never touches the subject is not grounding: the branch
    called a tool but its result says nothing about the Jev engine."""
    from agent_core.commands.speculate_cmd import SpeculateCommand

    agent = _ToolAgent("read", tool_result="nothing about the topic")
    asyncio.run(SpeculateCommand().execute(
        ['"How does the Jev engine decide?"'], agent,
    ))
    assert agent.executed  # a tool WAS called...
    out = capsys.readouterr().out
    assert "REFUSE" in out
    assert "without tool evidence about the subject" in out


def test_speculate_refuses_unverified_claims(capsys):
    """A COMMIT-worthy answer citing code that is not there is refused."""
    from agent_core.commands.speculate_cmd import SpeculateCommand

    class ClaimingLLM:
        model_name = "fake-model"

        async def chat(self, messages, tools=None, **kwargs):
            content = str(messages[-1].get("content") or "")
            if content.startswith("Quality judge:"):
                return "0.9"
            return "The engine lives at agent_core/jev_engine.py:999999."

    agent = type("A", (), {"llm": ClaimingLLM(), "workspace": "C:/Dev/Agent1"})()
    assert asyncio.run(SpeculateCommand().execute(['"q"'], agent)) is True
    out = capsys.readouterr().out
    assert "REFUSE - unverified file/line claim" in out
    assert "999999" in out
    assert "COMMIT" not in out


def test_looks_like_tool_call_detects_leaked_syntax():
    from agent_core.commands.speculate_cmd import _looks_like_tool_call

    assert _looks_like_tool_call('<|tool_call>call:run{command:"x"}<tool_call|>')
    assert _looks_like_tool_call('call:read{path:"a.py"}')
    assert _looks_like_tool_call('{"tool_calls": [{"id": "1"}]}')
    assert not _looks_like_tool_call("The repo is doing well - about 80/100.")
    assert not _looks_like_tool_call("")


def test_looks_like_fabricated_output_detects_pseudo_tool_xml():
    """A weak branch model writes fake tool XML with placeholder paths instead
    of calling the tool (observed with qwen2.5-coder-1.5b in pure Jev mode)."""
    from agent_core.commands.speculate_cmd import _looks_like_fabricated_output

    assert _looks_like_fabricated_output('<definitions path="path/to/x.py">')
    assert _looks_like_fabricated_output('<references symbol="X" max_results="30">')
    assert _looks_like_fabricated_output('<search query="X">')
    assert _looks_like_fabricated_output('<web_search query="X">')
    assert _looks_like_fabricated_output("see path/to/file.py")
    assert _looks_like_fabricated_output("docs at example.com")
    assert not _looks_like_fabricated_output("The fix command edits files in place.")
    assert not _looks_like_fabricated_output("use the <definitions> tool first")
    assert not _looks_like_fabricated_output("")


def test_speculate_rejects_fabricated_branch_answer(capsys):
    """Regression: a branch fabricated tool XML with placeholder paths and its
    own judge scored it 0.83 — it was COMMITted as the answer."""
    from agent_core.commands.speculate_cmd import SpeculateCommand

    class FabricatingLLM:
        model_name = "fake-model"

        async def chat(self, messages, tools=None, **kwargs):
            content = str(messages[-1].get("content") or "")
            if content.startswith("Quality judge:"):
                return "0.9"  # even a perfect judge score must not save it
            return (
                '<definitions path="path/to/jev_integration.py">\n'
                '  <definition class="Jev" line="100"/>\n'
                '</definitions>\n'
                '<search query="Jev">\n'
                '  <file path="path/to/x.py">line 1</file>\n</search>'
            )

    agent = type("A", (), {"llm": FabricatingLLM(), "workspace": "C:/Dev/Agent1"})()
    assert asyncio.run(SpeculateCommand().execute(['"q"'], agent)) is True
    out = capsys.readouterr().out
    assert "REFUSE" in out
    assert "COMMIT" not in out
    assert "<definitions" not in out  # never presented as the answer


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


# ---------------------------------------------------------------------------
#  Jev judge (--judge jev|both)
# ---------------------------------------------------------------------------


class _FakeJevEngine:
    """P(yes)=good for a 'good' answer, bad otherwise.

    ``provider`` is the branch LLM in pure `--judge jev` mode (the engine's
    pinned small-model provider); it defaults to a branch-answering FakeLLM.
    """

    model_name = "small"

    def __init__(self, good=0.9, bad=0.2, provider=None):
        self._good = good
        self._bad = bad
        self.provider = provider if provider is not None else FakeLLM()

    async def ensure_ready(self):
        pass

    async def decide(self, question, state=""):
        from agent_core.jev_engine import JevResult

        good = "good answer" in question.text
        p = self._good if good else self._bad
        return JevResult(
            kind="yesno", model="small", mechanism="vote",
            probabilities={"yes": p, "no": 1.0 - p},
            decision="TRUE" if p >= 0.5 else "FALSE",
            confidence=max(p, 1.0 - p),
        )


class _CountingLLM:
    """Branch answers like FakeLLM but counts branch and LLM-judge calls."""

    model_name = "fake-model"

    def __init__(self, judge_score="0.5"):
        self.judge_calls = 0
        self.branch_calls = 0
        self.judge_score = judge_score

    async def chat(self, messages, tools=None, **kwargs):
        content = str(messages[-1]["content"])
        if content.startswith("Quality judge:"):
            self.judge_calls += 1
            return self.judge_score
        self.branch_calls += 1
        branch_id = int(content.split()[2].rstrip("."))
        if branch_id == 0:
            return "good answer %d" % branch_id
        return "mediocre answer %d" % branch_id


def _patch_jev(monkeypatch, engine):
    import agent_core.jev_engine as jev

    monkeypatch.setattr(jev, "build_jev_engine", lambda **kw: engine)


def test_speculate_jev_judge_uses_chat_branches_by_default(monkeypatch, capsys):
    """The smart split: the reasoning model THINKS (branches), the small Jev
    model DECIDES (judge)."""
    from agent_core.commands.speculate_cmd import SpeculateCommand

    chat = _CountingLLM()
    _patch_jev(monkeypatch, _FakeJevEngine())
    agent = type("A", (), {"llm": chat, "workspace": "C:/Dev/Agent1"})()
    assert asyncio.run(
        SpeculateCommand().execute(['"which branch is best?"', "--judge", "jev"], agent)
    ) is True
    out = capsys.readouterr().out
    assert "branch_model=chat" in out
    assert "jev_model=small" in out
    assert "COMMIT" in out
    assert chat.branch_calls > 0   # chat model generated the branches
    assert chat.judge_calls == 0   # the small model judged


def test_speculate_jev_mode_runs_everything_on_the_jev_model(monkeypatch, capsys):
    """`--branch-model jev` runs the WHOLE command on the dedicated Jev model —
    branches AND judge — and never touches the selected chat model."""
    from agent_core.commands.speculate_cmd import SpeculateCommand

    branch = _CountingLLM()  # stands in for the Jev provider (engine.provider)
    chat = _CountingLLM()    # the selected chat model — must stay untouched
    _patch_jev(monkeypatch, _FakeJevEngine(provider=branch))
    agent = type("A", (), {"llm": chat, "workspace": "C:/Dev/Agent1"})()
    assert asyncio.run(
        SpeculateCommand().execute(
            ['"which branch is best?"', "--judge", "jev", "--branch-model", "jev"],
            agent,
        )
    ) is True
    out = capsys.readouterr().out
    assert "COMMIT" in out
    assert "good answer 0" in out
    assert "judge=jev" in out
    assert "jev_model=small" in out
    assert "branch_model=fake-model" in out
    assert "timeout=600s" in out
    assert branch.branch_calls > 0   # branches ran on the Jev provider
    assert branch.judge_calls == 0   # ...and the LLM judge was never asked
    assert chat.branch_calls == 0    # the selected chat model was NOT used
    assert chat.judge_calls == 0


def test_speculate_both_judges_average(monkeypatch, capsys):
    from agent_core.commands.speculate_cmd import SpeculateCommand

    _patch_jev(monkeypatch, _FakeJevEngine())
    llm = _CountingLLM()
    agent = type("A", (), {"llm": llm, "workspace": "C:/Dev/Agent1"})()
    # good: (0.9 + 0.5) / 2 = 0.7 >= 0.7 -> COMMIT
    assert asyncio.run(
        SpeculateCommand().execute(
            ['"which branch is best?"', "--judge", "both"], agent,
        )
    ) is True
    out = capsys.readouterr().out
    assert "judge=both" in out
    assert "COMMIT" in out
    assert llm.judge_calls > 0


def test_speculate_jev_escalate_uses_the_chat_model_in_gray_band(monkeypatch, capsys):
    """`--escalate` is the ONLY way `--judge jev` touches the chat model."""
    from agent_core.commands.speculate_cmd import SpeculateCommand

    _patch_jev(monkeypatch, _FakeJevEngine(good=0.5))
    chat = _CountingLLM(judge_score="0.9")
    agent = type("A", (), {"llm": chat, "workspace": "C:/Dev/Agent1"})()
    assert asyncio.run(
        SpeculateCommand().execute(
            ['"which branch is best?"', "--judge", "jev", "--escalate",
             "--branch-model", "jev"],
            agent,
        )
    ) is True
    out = capsys.readouterr().out
    assert "escalate=on" in out
    assert "COMMIT" in out
    assert chat.judge_calls > 0   # escalated in the gray band


def test_speculate_jev_gray_band_without_escalate_stays_pure(monkeypatch, capsys):
    from agent_core.commands.speculate_cmd import SpeculateCommand

    _patch_jev(monkeypatch, _FakeJevEngine(good=0.5))
    chat = _CountingLLM(judge_score="0.9")
    agent = type("A", (), {"llm": chat, "workspace": "C:/Dev/Agent1"})()
    assert asyncio.run(
        SpeculateCommand().execute(
            ['"which branch is best?"', "--judge", "jev", "--branch-model", "jev"],
            agent,
        )
    ) is True
    out = capsys.readouterr().out
    assert "escalate=off" in out
    assert "REFUSE" in out  # 0.5 < 0.7 and no escalation
    assert chat.branch_calls == 0
    assert chat.judge_calls == 0


def test_speculate_jev_mode_rejects_fabricated_answer(monkeypatch, capsys):
    """Pure-Jev mode must reject a fabricated branch answer and stay on the
    Jev model (the selected chat model is never touched)."""
    from agent_core.commands.speculate_cmd import SpeculateCommand

    class FabricatingLLM:
        model_name = "fake-model"

        async def chat(self, messages, tools=None, **kwargs):
            return (
                '<references symbol="X" max_results="30">\n'
                '<reference file="path/to/a.py" line="5"/>\n</references>'
            )

    chat = _CountingLLM()
    _patch_jev(monkeypatch, _FakeJevEngine(provider=FabricatingLLM()))
    agent = type("A", (), {"llm": chat, "workspace": "C:/Dev/Agent1"})()
    assert asyncio.run(
        SpeculateCommand().execute(
            ['"q"', "--judge", "jev", "--branch-model", "jev"], agent,
        )
    ) is True
    out = capsys.readouterr().out
    assert "REFUSE" in out
    assert chat.branch_calls == 0
    assert chat.judge_calls == 0


def test_speculate_jev_judge_confident_reject_skips_llm(monkeypatch, capsys):
    from agent_core.commands.speculate_cmd import SpeculateCommand

    _patch_jev(monkeypatch, _FakeJevEngine(good=0.2))
    chat = _CountingLLM(judge_score="0.9")
    agent = type("A", (), {"llm": chat, "workspace": "C:/Dev/Agent1"})()
    assert asyncio.run(
        SpeculateCommand().execute(
            ['"which branch is best?"', "--judge", "jev", "--branch-model", "jev"],
            agent,
        )
    ) is True
    out = capsys.readouterr().out
    assert "REFUSE" in out
    assert chat.judge_calls == 0  # confident reject, no reasoning-model cost
    assert chat.branch_calls == 0


def test_speculate_rejects_low_above_threshold(capsys):
    assert _run(['"q"', "--low", "0.9", "--threshold", "0.5"]) is True
    out = capsys.readouterr().out
    assert "--low must be <= --threshold" in out
    assert "COMMIT" not in out


def test_speculate_jev_judge_build_failure_falls_back(monkeypatch, capsys):
    import agent_core.jev_engine as jev

    def boom(**kwargs):
        raise ValueError("no Jev model configured")

    monkeypatch.setattr(jev, "build_jev_engine", boom)
    assert _run(['"which branch is best?"', "--judge", "jev"]) is True
    out = capsys.readouterr().out
    assert "Jev judge unavailable" in out
    assert "judge=llm" in out
    assert "COMMIT" in out
    assert "good answer 0" in out


def test_speculate_rejects_bad_judge(capsys):
    assert _run(['"q"', "--judge", "bogus"]) is True
    out = capsys.readouterr().out
    assert "--judge" in out
    assert "COMMIT" not in out
