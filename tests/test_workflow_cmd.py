import os
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch

from agent_core.commands.workflow_cmd import (
    _scan_workspace_context,
    _analysis_flag_gate,
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