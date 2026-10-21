import sys
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agent_core.commands.analysis_verifier import verify_analysis_claims
from agent_core.commands.workflow_cmd import (
    _scan_workspace_context,
    _analysis_flag_gate,
    _extract_decisions_if_any,
    _repair_analysis,
    _repair_doc_until_clean,
    _specs_match,
    _verify_or_repair_gate,
)


class TestSpecsMatch:
    """Carry-over of plan/entities/taskplan is only safe for the SAME task."""

    def test_identical_spec_matches(self, tmp_path: Path) -> None:
        spec = tmp_path / "project_spec.md"
        spec.write_text("# Spec\n\nsearch the web via duckduckgo", encoding="utf-8")
        assert _specs_match(spec, "# Spec\n\nsearch the web via duckduckgo") is True

    def test_different_spec_does_not_match(self, tmp_path: Path) -> None:
        spec = tmp_path / "project_spec.md"
        spec.write_text("# Spec\n\nold task", encoding="utf-8")
        assert _specs_match(spec, "# Spec\n\nnew task") is False

    def test_missing_prev_spec_does_not_match(self, tmp_path: Path) -> None:
        assert _specs_match(tmp_path / "missing.md", "anything") is False


class TestAnalysisFlagGate:
    """Flagged claims halt AFTER autonomous repair/prune — never a prompt."""

    def test_clean_analysis_passes_without_input(self) -> None:
        assert _analysis_flag_gate(checked=10, flagged=0, force=False) is True

    def test_flagged_with_force_only_warns(self) -> None:
        with patch("builtins.input", side_effect=AssertionError("must not prompt")):
            assert _analysis_flag_gate(checked=10, flagged=3, force=True) is True

    def test_flagged_without_force_halts_without_prompting(self, capsys) -> None:
        report = (
            "Some claim\n## Verification Report\n"
            "- fake.py:12: not found\n- missing symbol"
        )
        with patch("builtins.input", side_effect=AssertionError("must not prompt")):
            assert (
                _analysis_flag_gate(
                    checked=2, flagged=2, force=False, report_text=report
                )
                is False
            )
        out = capsys.readouterr().out
        assert "fake.py" in out
        assert "missing symbol" in out


class TestScanWorkspaceContext:
    """_scan_workspace_context combines workspace python files under a line budget."""

    def _write(self, ws: Path, name: str, lines: list[str]) -> None:
        (ws / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_skips_when_spec_not_agent_oriented(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            self._write(ws, "main.py", ["x = 1"])
            used, combined = _scan_workspace_context(ws, "a csv converter")
            assert used is False
            assert combined == ""

    def test_combines_files_with_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            self._write(ws, "main.py", ["x = 1", "print(x)"])
            self._write(ws, "util.py", ["y = 2"])
            used, combined = _scan_workspace_context(
                ws, "optimize the agent orchestration"
            )
            assert used is True
            assert "# ---- main.py ----" in combined
            assert "# ---- util.py ----" in combined
            assert "x = 1" in combined

    def test_overflow_caps_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            self._write(ws, "huge.py", [f"x = {i}" for i in range(6100)])
            used, combined = _scan_workspace_context(ws, "agent self-improvement")
            assert used is True
            assert len(combined.splitlines()) <= 6000
            assert "# ---- huge.py ----" in combined

    def test_overflow_after_existing_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            self._write(ws, "a.py", [f"a = {i}" for i in range(6000)])
            self._write(ws, "b.py", ["b = 1"])
            used, combined = _scan_workspace_context(ws, "agent self-improvement")
            assert used is True
            assert "# ---- b.py ----" not in combined
            assert len(combined.splitlines()) <= 6000


class TestReuseWithoutSpecFile:
    """Regression: _reuse() must not raise NameError when no spec file is
    provided (e.g. --brainstorm without --desc/--from).  Previously
    `spec_content` was a free variable only assigned inside an `if spec_file:`
    block that came AFTER _reuse's definition, so referencing it raised:

        NameError: cannot access free variable 'spec_content' while ...
    """

    def test_specs_match_with_empty_current_spec(self, tmp_path: Path) -> None:
        prev = tmp_path / "project_spec.md"
        prev.write_text("# Spec\n\nsome old task", encoding="utf-8")
        # Empty spec content is the value used when no spec file exists.
        assert _specs_match(prev, "") is False

    def test_specs_match_with_empty_current_when_prev_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            assert _specs_match(ws / "missing.md", "") is False


class TestRepairAnalysis:
    """The repair round must run with thinking disabled — a reasoning model
    otherwise burns the whole 8000-token budget on reasoning_content and
    emits nothing (observed 2026-08-18: finish_reason=length, content="")."""

    def test_repair_round_disables_thinking(self, tmp_path: Path) -> None:
        (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
        agent = AsyncMock()
        agent.llm.chat = AsyncMock(return_value="Plain text, no backticked claims.")
        repaired, checked, flagged = asyncio.run(
            _repair_analysis(
                agent,
                "## 1. SCOPE\n\nPlain text, no claims.",
                "## Verification Report\n- [UNVERIFIED] `run` — not found",
                str(tmp_path),
            )
        )
        assert agent.llm.chat.call_args.kwargs.get("disable_thinking") is True
        assert repaired is not None
        assert flagged == 0


class TestExtractDecisionsGate:
    """Warned candidates are skipped unless the user explicitly confirms."""

    _WARNED = [
        {
            "title": "Refactor missing_module",
            "context": "Improve it.",
            "decision": "Refactor.",
            "affected_files": ["missing_module.py"],
            "warnings": [
                "Affected file does not exist in workspace: missing_module.py"
            ],
        }
    ]
    _CLEAN = [
        {
            "title": "Clean decision",
            "context": "Do it.",
            "decision": "Do it.",
            "affected_files": [],
        }
    ]

    def _run(self, tmp_path: Path, candidates: list, inputs: list[str]) -> None:
        analysis_md = tmp_path / "project_analysis.md"
        analysis_md.write_text("Some analysis.\n", encoding="utf-8")
        agent = AsyncMock()
        with (
            patch(
                "agent_core.commands.workflow_cmd.extract_from_analysis",
                new=AsyncMock(return_value=candidates),
            ),
            patch(
                "agent_core.commands.workflow_cmd.annotate_candidates",
                side_effect=lambda c, ws, verification_report="": c,
            ),
            patch.object(sys.stdin, "isatty", return_value=True),
            patch("builtins.input", side_effect=list(inputs)),
        ):
            asyncio.run(_extract_decisions_if_any(agent, str(analysis_md), tmp_path))

    def test_warned_candidate_skipped_without_confirm(self, tmp_path: Path) -> None:
        self._run(tmp_path, self._WARNED, ["1", "n"])
        assert not (tmp_path / ".decisions.json").exists()

    def test_warned_candidate_recorded_with_explicit_yes(self, tmp_path: Path) -> None:
        self._run(tmp_path, self._WARNED, ["1", "y"])
        assert (tmp_path / ".decisions.json").exists()

    def test_clean_candidate_records_without_confirm(self, tmp_path: Path) -> None:
        self._run(tmp_path, self._CLEAN, ["1"])
        assert (tmp_path / ".decisions.json").exists()


class TestVerifyOrRepairGate:
    """Autonomous analysis gate: multi-round repair keeps partial progress,
    then a deterministic auto-prune fallback — the user is never prompted."""

    _BODY = (
        "## 1. CODE QUALITY\n\n"
        "- Overview sentence with no claims.\n"
        "- The file `real_mod.py` holds the logic.\n"
        "- Another plain prose line.\n"
        "- A fourth line without backticked claims.\n"
        "- The symbol `_ghost_alpha` is broken.\n"
        "- The symbol `_ghost_beta` is broken.\n"
    )

    @staticmethod
    def _ws(tmp_path: Path) -> Path:
        (tmp_path / "real_mod.py").write_text(
            "def real_fn():\n    return 1\n", encoding="utf-8"
        )
        return tmp_path

    def test_multi_round_repair_keeps_partial_progress(
        self, tmp_path: Path, capsys
    ) -> None:
        ws = self._ws(tmp_path)
        md = tmp_path / "project_analysis.md"
        base = asyncio.run(verify_analysis_claims(self._BODY, ws))
        assert base.flagged == 2
        round1 = self._BODY.replace("- The symbol `_ghost_alpha` is broken.\n", "")
        round2 = round1.replace("- The symbol `_ghost_beta` is broken.\n", "")
        agent = AsyncMock()
        agent.llm.chat = AsyncMock(side_effect=[round1, round2])
        with patch("builtins.input", side_effect=AssertionError("must not prompt")):
            ok = asyncio.run(
                _verify_or_repair_gate(
                    agent,
                    self._BODY,
                    str(md),
                    ws,
                    base.checked,
                    base.flagged,
                    base.text,
                    force=False,
                )
            )
        assert ok is True
        assert agent.llm.chat.call_count == 2
        # Round 2 was fed round 1's partially repaired text: progress is kept,
        # not discarded (the old code threw away any non-perfect round).
        second_analysis = agent.llm.chat.call_args_list[1].args[0][1]["content"]
        second_analysis = second_analysis.split("## Analysis to repair:")[-1]
        assert "ghost_alpha" not in second_analysis
        body = md.read_text(encoding="utf-8").split("## Verification Report")[0]
        assert "ghost_alpha" not in body
        assert "ghost_beta" not in body
        assert "Auto-pruned" not in capsys.readouterr().out

    def test_stalled_repair_falls_back_to_auto_prune(
        self, tmp_path: Path, capsys
    ) -> None:
        ws = self._ws(tmp_path)
        md = tmp_path / "project_analysis.md"
        base = asyncio.run(verify_analysis_claims(self._BODY, ws))
        assert base.flagged == 2
        agent = AsyncMock()
        agent.llm.chat = AsyncMock(return_value=self._BODY)  # no progress
        with patch("builtins.input", side_effect=AssertionError("must not prompt")):
            ok = asyncio.run(
                _verify_or_repair_gate(
                    agent,
                    self._BODY,
                    str(md),
                    ws,
                    base.checked,
                    base.flagged,
                    base.text,
                    force=False,
                )
            )
        assert ok is True
        assert agent.llm.chat.call_count == 1  # one stalled attempt, then prune
        out = capsys.readouterr().out
        assert "Auto-pruned" in out
        body = md.read_text(encoding="utf-8").split("## Verification Report")[0]
        assert "ghost_alpha" not in body
        assert "ghost_beta" not in body
        assert "real_mod.py" in body

    def test_auto_prune_refuses_to_gut_the_document(
        self, tmp_path: Path, capsys
    ) -> None:
        ws = self._ws(tmp_path)
        md = tmp_path / "project_analysis.md"
        body = (
            "## 1. CODE QUALITY\n"
            "- `real_mod.py` mentions `_ghost_alpha`.\n"
            "- `real_mod.py` mentions `_ghost_beta`.\n"
        )
        base = asyncio.run(verify_analysis_claims(body, ws))
        assert base.flagged == 2
        agent = AsyncMock()
        agent.llm.chat = AsyncMock(return_value=body)
        with patch("builtins.input", side_effect=AssertionError("must not prompt")):
            ok = asyncio.run(
                _verify_or_repair_gate(
                    agent,
                    body,
                    str(md),
                    ws,
                    base.checked,
                    base.flagged,
                    base.text,
                    force=False,
                )
            )
        assert ok is False
        out = capsys.readouterr().out
        assert "could not be verified" in out

    def test_force_skips_the_repair_loop(self, tmp_path: Path) -> None:
        ws = self._ws(tmp_path)
        md = tmp_path / "project_analysis.md"
        base = asyncio.run(verify_analysis_claims(self._BODY, ws))
        agent = AsyncMock()
        with patch("builtins.input", side_effect=AssertionError("must not prompt")):
            ok = asyncio.run(
                _verify_or_repair_gate(
                    agent,
                    self._BODY,
                    str(md),
                    ws,
                    base.checked,
                    base.flagged,
                    base.text,
                    force=True,
                )
            )
        assert ok is True
        assert agent.llm.chat.call_count == 0


class TestRepairDocUntilClean:
    """Workflow plan/entities/taskplan stages repair flagged output through
    the LLM and halt on failure — the interactive _plan_doc_gate prompt is
    unreachable without --force."""

    @staticmethod
    def _ws(tmp_path: Path) -> Path:
        (tmp_path / "real_mod.py").write_text("x = 1\n", encoding="utf-8")
        return tmp_path

    def test_repairs_until_clean(self, tmp_path: Path) -> None:
        ws = self._ws(tmp_path)
        fixed = "## Plan\n\n- [FIX] touch `real_mod.py` - small fix\n"
        agent = AsyncMock()
        agent.llm.chat = AsyncMock(return_value=fixed)
        with patch("builtins.input", side_effect=AssertionError("must not prompt")):
            out = asyncio.run(
                _repair_doc_until_clean(agent, "plan", "Do the thing somehow.", ws)
            )
        assert out is not None
        assert "real_mod.py" in out
        assert agent.llm.chat.call_count == 1

    def test_gives_up_after_rounds_without_prompting(self, tmp_path: Path) -> None:
        ws = self._ws(tmp_path)
        agent = AsyncMock()
        agent.llm.chat = AsyncMock(return_value="Do the thing somehow.")
        with patch("builtins.input", side_effect=AssertionError("must not prompt")):
            out = asyncio.run(
                _repair_doc_until_clean(agent, "plan", "Do the thing somehow.", ws)
            )
        assert out is None
        assert agent.llm.chat.call_count == 2


class TestWorkflowStagesNeverPrompt:
    """Structural guard: every workflow stage pre-repairs before the shared
    gate, and the analysis gate itself contains no interactive call."""

    def test_all_stage_sites_pre_repair(self) -> None:
        import inspect

        from agent_core.commands import workflow_cmd

        src = inspect.getsource(workflow_cmd)
        # def + 9 stage call sites (plan/entities/taskplan x 3 paths).
        assert src.count("_repair_doc_until_clean(") >= 10

    def test_analysis_flag_gate_has_no_interactive_call(self) -> None:
        import inspect

        from agent_core.commands import workflow_cmd

        gate_src = inspect.getsource(workflow_cmd._analysis_flag_gate)
        assert "auto_choice" not in gate_src
        assert "input(" not in gate_src
