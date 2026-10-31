"""Regression tests: pytest must never be invoked on a non-Python file.

History (2026-10-31): a documentation plan turn ended with
``run(command="python -m pytest docs/PROSPECTOR_PLAN.md -q")`` - a pytest
invocation aimed at a Markdown PLAN document.  pytest collects nothing from a
``.md`` file, so the run reported ``no tests ran`` (exit code 5) and proved
nothing about the plan; the turn burned a cycle and the transcript suggested
the plan had been "tested".

Two rules now hold (decision #124):

1. **A plan/document file is never a pytest target.**  ``run`` refuses the
   command before Popen and ``tests``/``unittest`` refuse the path, with a
   message naming a valid replacement (a real ``.py`` test path, or a
   structural check such as ``python scripts/audit_invariants.py``).
2. **The refusal is narrow.**  Every real pytest shape - bare, whole tree,
   subset selectors, ``-k``/``-m`` expressions, options with values, node ids,
   dotted module ids, piped and redirected commands - must keep running, and
   a non-Python token that pytest does not receive as a *target* (an option
   value, ``--ignore=``, a ``K=V`` env assignment, a second segment of a
   compound command) must not trigger a refusal.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest

import agent
from agent import Agent
from agent_core import pytest_cmd_policy
from agent_core.decisions import load_decisions
from agent_core.handlers import nlp_tools
from agent_core.pytest_cmd_policy import (
    NON_PYTHON_TARGET_REFUSAL,
    pytest_non_python_target,
)

#: Targets pytest cannot collect a test from.  The plan document is the
#: reported case and is listed first on purpose.
NON_PYTHON_TARGETS = [
    "PROSPECTOR_PLAN.md",
    "docs/PLAN.md",
    "docs/refactor_agent_py_taskplan.md",
    "Architecture.md",
    "README.md",
    ".decisions.json",
    "reports/traces/run.json",
    "pyproject.toml",
    ".gitignore",
    "requirements.txt",
    "AGENTS",
]


@pytest.fixture()
def bot() -> Agent:
    return Agent(workspace=".")


def _run_tool(bot: Agent, args: dict[str, Any]) -> str:
    async def go() -> str:
        return await bot._nlp_run(args)

    return asyncio.run(go())


def _tests_tool(bot: Agent, args: dict[str, Any]) -> str:
    async def go() -> str:
        return await bot._nlp_tests(args)

    return asyncio.run(go())


# ---------------------------------------------------------------------------
# The classifier: what counts as a non-Python pytest target
# ---------------------------------------------------------------------------


class TestNonPythonTargetDetection:
    @pytest.mark.parametrize("target", NON_PYTHON_TARGETS)
    def test_markdown_and_data_targets_detected(self, target: str) -> None:
        command = f"python -m pytest {target} -q --no-cov"
        assert pytest_non_python_target(command) == target

    def test_reported_incident_is_detected(self) -> None:
        """The exact command from the plan turn."""
        command = "python -m pytest docs/PROSPECTOR_PLAN.md -q"
        assert pytest_non_python_target(command) == "docs/PROSPECTOR_PLAN.md"

    @pytest.mark.parametrize(
        "command",
        [
            "python -m pytest tests/test_x.py -q --no-cov",
            "python -m pytest tests/ -q --no-cov",
            "python -m pytest . -q",
            "python -m pytest -q --no-cov",
            "python -m pytest --lf -q --no-cov",
            "python -m pytest --testmon -q --no-cov",
            "python -m pytest tests/test_x.py::TestA::test_b -q",
            "python -m pytest tests.test_x -q",
            "python -m pytest -k 'plan or doc' -q",
            "python -m pytest -m 'not slow' -q",
            'python -m pytest "tests/test_x.py::test_y" -q --no-cov 2>&1',
            "python -m pytest tests/test_x.py -q >nul",
            # A non-Python token pytest never receives as a TARGET:
            "python -m pytest --ignore=docs/PLAN.md -q",
            "python -m pytest -p no:cacheprovider -q",
            "python -m pytest -c pyproject.toml tests/test_x.py -q",
            "python -m pytest --deselect tests/test_x.py::test_y -q",
            # Compound commands: see below, the scrubber removes them first.
            "python -m pytest tests/test_x.py -q && python scripts/check.py plan.md",
            "git status && python -m pytest -q",
            "echo pytest plan.md",
            "python -c \"print('pytest plan.md')\"",
        ],
    )
    def test_real_pytest_shapes_are_not_flagged(self, command: str) -> None:
        assert pytest_non_python_target(command) == ""

    def test_unittest_module_target_flagged(self) -> None:
        command = "python -m unittest docs/PLAN.md"
        assert pytest_non_python_target(command) == "docs/PLAN.md"

    @pytest.mark.parametrize(
        "token",
        [
            "PROSPECTOR_PLAN.md",
            "docs/PLAN.md",
            "./ARCHITECTURE.MD",
            "pyproject.toml",
            ".gitignore",
            ".decisions.json",
            "AGENTS",
        ],
    )
    def test_target_token_predicate(self, token: str) -> None:
        assert pytest_cmd_policy._is_non_python_test_target(token) is True

    @pytest.mark.parametrize(
        "token",
        [
            "tests/test_x.py",
            "test_x.py",
            "tests/",
            ".",
            "tests.test_x",
            "tests/test_x.py::TestA::test_b",
            "tests/unit",
            "",
            # A directory that only CONTAINS a dot is still collectable.
            "tests/v1.0",
            "agent_core/tests",
        ],
    )
    def test_collectable_tokens_not_flagged(self, token: str) -> None:
        assert pytest_cmd_policy._is_non_python_test_target(token) is False


# ---------------------------------------------------------------------------
# The run tool refuses before launching a process
# ---------------------------------------------------------------------------


class TestRunToolRefusesNonPythonTarget:
    def test_plan_document_refused_without_spawning(
        self, bot: Agent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spawned: list[object] = []

        def _popen(cmd: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            spawned.append(cmd)

            class _Proc:
                returncode = 0

                def communicate(self, timeout: int | None = None) -> tuple[str, str]:
                    return ("", "")

            return _Proc()

        monkeypatch.setattr(agent.subprocess, "Popen", _popen)
        out = _run_tool(bot, {"command": "python -m pytest docs/PLAN.md -q"})
        assert not spawned, "no process may start for a non-Python target"
        assert "docs/PLAN.md" in out
        assert ".py" in out  # the message names a valid replacement

    @pytest.mark.parametrize("target", NON_PYTHON_TARGETS)
    def test_every_non_python_target_refused(
        self, bot: Agent, target: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            agent.subprocess,
            "Popen",
            lambda *a, **kw: pytest.fail("Popen called for " + target),
        )
        out = _run_tool(bot, {"command": f"python -m pytest {target} -q"})
        assert out.startswith("Error:")

    def test_real_test_path_still_runs(
        self, bot: Agent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[object] = []

        class _Ok:
            returncode = 0

            def communicate(self, timeout: int | None = None) -> tuple[str, str]:
                seen.append(True)
                return ("1 passed", "")

        monkeypatch.setattr(agent.subprocess, "Popen", lambda *a, **kw: _Ok())
        out = _run_tool(
            bot, {"command": "python -m pytest tests/test_x.py -q --no-cov"}
        )
        assert seen, "a real .py test path must still execute"
        assert "Error:" not in out

    def test_non_pytest_command_on_document_still_runs(
        self, bot: Agent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only pytest invocations are restricted; reading a doc is fine."""
        seen: list[object] = []

        class _Ok:
            returncode = 0

            def communicate(self, timeout: int | None = None) -> tuple[str, str]:
                seen.append(True)
                return ("plan ok", "")

        monkeypatch.setattr(agent.subprocess, "Popen", lambda *a, **kw: _Ok())
        out = _run_tool(bot, {"command": "type docs/PLAN.md"})
        assert seen
        assert "plan ok" in out


# ---------------------------------------------------------------------------
# The tests tool refuses the same targets
# ---------------------------------------------------------------------------


class TestTestsToolRefusesNonPythonTarget:
    def test_plan_document_refused_without_running(
        self, bot: Agent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            nlp_tools,
            "_run_subprocess_captured",
            lambda *a, **kw: pytest.fail("tests tool ran a non-Python target"),
        )
        out = _tests_tool(bot, {"path": "docs/PLAN.md"})
        assert out.startswith("Error:")
        assert "docs/PLAN.md" in out

    def test_json_file_refused(
        self, bot: Agent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            nlp_tools,
            "_run_subprocess_captured",
            lambda *a, **kw: pytest.fail("tests tool ran a non-Python target"),
        )
        assert _tests_tool(bot, {"path": ".decisions.json"}).startswith("Error:")

    def test_real_test_file_runs(
        self, bot: Agent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[list[str]] = []

        def _captured(
            cmd: list[str],
            cwd: str = ".",
            timeout: int = 60,
            label: str = "Run",
        ) -> tuple[str, None]:
            calls.append(list(cmd))
            return ("1 passed", None)

        monkeypatch.setattr(nlp_tools, "_run_subprocess_captured", _captured)
        out = _tests_tool(bot, {"path": "tests/test_x.py"})
        assert calls and calls[0][3].endswith("test_x.py")
        assert out == "1 passed"


# ---------------------------------------------------------------------------
# The refusal text itself: actionable, no ambiguity
# ---------------------------------------------------------------------------


class TestRefusalMessage:
    def test_names_the_target_and_the_replacement(self) -> None:
        message = NON_PYTHON_TARGET_REFUSAL.format(target="docs/PLAN.md")
        assert "docs/PLAN.md" in message
        assert ".py" in message
        assert "audit_invariants.py" in message
        assert "python -m pytest" in message

    def test_states_the_rule_not_just_the_error(self) -> None:
        message = NON_PYTHON_TARGET_REFUSAL.format(target="README.md")
        lowered = message.lower()
        assert "never" in lowered or "not" in lowered
        assert "pytest" in lowered


# ---------------------------------------------------------------------------
# ONE classifier and ONE message for every pytest ENTRY POINT.
#
# Decision #124 forbids a pytest run on a non-Python file.  The rule is only
# unambiguous if every way of starting pytest (run tool, tests tool, self_heal
# command) asks the SAME predicate and prints the SAME text: a second,
# hand-rolled check in one entry point would let the others drift apart.
# ---------------------------------------------------------------------------

#: Modules that can start a pytest process and must therefore defer to
#: ``agent_core.pytest_cmd_policy`` instead of re-implementing the rule.
ENTRY_POINT_MODULES = [
    "agent_core/handlers/nlp_tools.py",
    "agent_core/commands/self_heal_cmd.py",
]


class TestOneClassifierForEveryEntryPoint:
    def test_path_and_command_refusals_are_identical(self) -> None:
        """The path twin and the command classifier cannot disagree."""
        for target in NON_PYTHON_TARGETS:
            from_path = pytest_cmd_policy.non_python_path_refusal(target)
            from_command = pytest_cmd_policy.non_python_pytest_target_refusal(
                f"python -m pytest {target} -q"
            )
            assert from_path == from_command, target
            assert from_path == NON_PYTHON_TARGET_REFUSAL.format(target=target)

    @pytest.mark.parametrize(
        "target", ["tests/test_x.py", "tests/", ".", "", "agent_core"]
    )
    def test_collectable_path_produces_no_refusal(self, target: str) -> None:
        assert pytest_cmd_policy.non_python_path_refusal(target) == ""

    @pytest.mark.parametrize("rel", ENTRY_POINT_MODULES)
    def test_entry_point_imports_the_shared_classifier(self, rel: str) -> None:
        source = (Path(__file__).resolve().parent.parent / rel).read_text(
            encoding="utf-8", errors="replace"
        )
        assert "agent_core.pytest_cmd_policy" in source, (
            f"{rel} starts pytest and must use the shared classifier"
        )
        assert "non_python_path_refusal" in source or (
            "non_python_pytest_target_refusal" in source
        ), f"{rel} must call a pytest_cmd_policy refusal helper"

    def test_self_heal_prints_the_shared_message(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agent_core.commands.self_heal_cmd import SelfHealCommand

        target = "docs/PLAN.md"
        assert SelfHealCommand().refuse_non_python_target(target) is True
        out = capsys.readouterr().out
        assert NON_PYTHON_TARGET_REFUSAL.format(target=target) in out

    def test_tests_tool_prints_the_shared_message(self, bot: Agent) -> None:
        out = _tests_tool(bot, {"path": "docs/PLAN.md"})
        assert NON_PYTHON_TARGET_REFUSAL.format(target="docs/PLAN.md") in out


# ---------------------------------------------------------------------------
# The rule is a RECORDED decision, not just a comment in the code.
#
# Decision #124 is cited by name in ``pytest_cmd_policy`` and in this module.
# A citation without a record is an ambiguity for every later reader (and for
# the fix/history prompts, which inject recorded decisions only), so the ledger
# entry is asserted here: it must exist, be active, carry no open
# contradiction, and name forward-existing files.
# ---------------------------------------------------------------------------

DECISION_ID = "124"
RULE_FILES = [
    "agent_core/pytest_cmd_policy.py",
    "agent_core/handlers/nlp_tools.py",
    "agent_core/tool_schemas.py",
    "agent_core/commands/self_heal_cmd.py",
    "tests/test_pytest_non_python_targets.py",
    "tests/unit/test_self_heal_cmd.py",
]


def _rule_decision() -> dict[str, Any]:
    workspace = Path(__file__).resolve().parent.parent
    matches = [
        d for d in load_decisions(workspace) if str(d.get("id")) == DECISION_ID
    ]
    assert len(matches) == 1, f"decision #{DECISION_ID} must be recorded exactly once"
    return matches[0]


class TestRuleIsARecordedDecision:
    def test_decision_is_recorded_and_active(self) -> None:
        record = _rule_decision()
        assert record["status"] == "active"
        assert not record["contradictions"]
        assert record["resolved_by"] in (None, "")

    def test_decision_states_target_rule_without_hedging(self) -> None:
        record = _rule_decision()
        text = (record["decision"] + " " + record["context"]).lower()
        for needle in ("pytest", "non-python", ".py"):
            assert needle in text, f"the decision must name {needle!r}"
        for hedge in ("maybe", "perhaps", "if possible", "tbd", "unclear"):
            assert hedge not in text, (
                f"hedging word {hedge!r} leaves the rule ambiguous"
            )

    def test_decision_names_the_implementation_files(self) -> None:
        root = Path(__file__).resolve().parent.parent
        files = set(_rule_decision()["affected_files"])
        for rel in RULE_FILES:
            assert rel in files, f"{rel} must be listed in decision #{DECISION_ID}"
            on_disk = (root / rel).exists()
            assert on_disk, f"{rel} is listed but missing on disk (stale entry)"

    def test_every_citation_of_the_id_resolves_to_the_record(self) -> None:
        """The id cited in source must be the id recorded in the ledger."""
        root = Path(__file__).resolve().parent.parent
        cited: set[str] = set()
        for rel in RULE_FILES:
            path = root / rel
            text = path.read_text(encoding="utf-8", errors="replace")
            cited.update(re.findall(r"decision\s*#(\d{3})", text, re.I))
        assert DECISION_ID in cited, "the source must cite the recorded decision id"
        known = {str(d.get("id")) for d in load_decisions(root)}
        missing = sorted(cited - known)
        assert not missing, f"cited but unrecorded decision ids: {missing}"


class TestTestsToolSchemaIsUnambiguous:
    """The model-facing schema must state the rule, not only the guard."""

    def test_tests_tool_description_forbids_non_python_targets(self) -> None:
        from agent_core.tool_schemas import NLP_TOOL_SCHEMAS

        schema = next(
            s for s in NLP_TOOL_SCHEMAS if s["function"]["name"] == "tests"
        )
        description = schema["function"]["description"].lower()
        assert ".py" in description
        assert "non-python" in description or "not a python" in description
        assert "refus" in description

    def test_run_tool_description_forbids_non_python_pytest_targets(self) -> None:
        from agent_core.tool_schemas import NLP_TOOL_SCHEMAS

        schema = next(s for s in NLP_TOOL_SCHEMAS if s["function"]["name"] == "run")
        description = schema["function"]["description"].lower()
        assert "non-python" in description or "not a python" in description
