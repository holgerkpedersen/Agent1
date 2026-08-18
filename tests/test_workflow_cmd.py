import os
import sys
import asyncio
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agent_core.commands.workflow_cmd import (
    _scan_workspace_context,
    _analysis_flag_gate,
    _extract_decisions_if_any,
    _repair_analysis,
    _specs_match,
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
    """Any unverifiable code claim pauses the run for confirmation."""

    def test_clean_analysis_passes_without_input(self) -> None:
        assert _analysis_flag_gate(checked=10, flagged=0, force=False) is True

    def test_flagged_with_force_only_warns(self) -> None:
        with patch("builtins.input", side_effect=AssertionError("must not prompt")):
            assert _analysis_flag_gate(checked=10, flagged=3, force=True) is True

    def test_flagged_without_force_confirms(self) -> None:
        with patch("builtins.input", return_value="y"):
            assert _analysis_flag_gate(checked=10, flagged=3, force=False) is True
        with patch("builtins.input", return_value="n"):
            assert _analysis_flag_gate(checked=10, flagged=3, force=False) is False

    def test_flagged_eof_defaults_to_halt(self) -> None:
        with patch("builtins.input", side_effect=EOFError):
            assert _analysis_flag_gate(checked=10, flagged=1, force=False) is False

    def test_report_text_display_does_not_crash(self, capsys) -> None:
        report = "Some claim\n## Verification Report\n- fake.py:12: not found\n- missing symbol"
        with patch("builtins.input", return_value="n"):
            _analysis_flag_gate(checked=2, flagged=2, force=False, report_text=report)
        out = capsys.readouterr().out
        assert "fake.py" in out


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
            used, combined = _scan_workspace_context(ws, "optimize the agent orchestration")
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

    _WARNED = [{
        "title": "Refactor missing_module",
        "context": "Improve it.",
        "decision": "Refactor.",
        "affected_files": ["missing_module.py"],
        "warnings": ["Affected file does not exist in workspace: missing_module.py"],
    }]
    _CLEAN = [{
        "title": "Clean decision",
        "context": "Do it.",
        "decision": "Do it.",
        "affected_files": [],
    }]

    def _run(self, tmp_path: Path, candidates: list, inputs: list[str]) -> None:
        analysis_md = tmp_path / "project_analysis.md"
        analysis_md.write_text("Some analysis.\n", encoding="utf-8")
        agent = AsyncMock()
        with patch(
            "agent_core.commands.workflow_cmd.extract_from_analysis",
            new=AsyncMock(return_value=candidates),
        ), patch(
            "agent_core.commands.workflow_cmd.annotate_candidates",
            side_effect=lambda c, ws, verification_report="": c,
        ), patch.object(sys.stdin, "isatty", return_value=True), patch(
            "builtins.input", side_effect=list(inputs)
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