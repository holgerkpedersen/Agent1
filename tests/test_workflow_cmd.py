import os
import tempfile
import textwrap
from pathlib import Path

from agent_core.commands.workflow_cmd import _scan_workspace_context


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