"""Tests for deterministic code-claim verification of LLM-generated analysis."""
import asyncio
import re
from pathlib import Path

from agent_core.commands.analysis_verifier import VerificationResult, verify_analysis_claims

_AGENT_PY = "\n".join([
    '"""Sample agent module for verification tests."""',
    "import subprocess",
    "",
    "def _execute_nlp_tool(self):",
    "    result = subprocess.run(",
    "        cmd_to_run,",
    "        shell=True,",
    "    )",
    "    return result",
    "",
    "class FileSystem:",
    "    def read(self, path: str):",
    "        local_path = self._safe_path(path)",
    "        with open(local_path) as f:",
    "            return f.read()",
    "",
    "def _safe_path(self, path: str) -> str:",
    "    if path.startswith('./'):",
    "        path = path[2:]",
    "    return path",
    "",
    "_FORBIDDEN_PATTERNS = ('eval(',)",
    *[f"filler_{i} = {i}" for i in range(40)],
])


class TestVerifyAnalysisClaims:
    """verify_analysis_claims flags fabricated file/symbol/line/snippet claims."""

    async def _verify(self, ws: Path, analysis: str) -> VerificationResult:
        return await verify_analysis_claims(analysis, ws)

    @staticmethod
    def _flagged_lines(result: VerificationResult) -> list[str]:
        """Return the individual [UNVERIFIED] bullet lines from a result."""
        return re.findall(r"- \[UNVERIFIED\] `[^`]*` — .*", result.text)

    @staticmethod
    def _build_ws(tmp_path: Path) -> Path:
        (tmp_path / "agent.py").write_text(_AGENT_PY, encoding="utf-8")
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "util.py").write_text("def util_fn():\n    pass\n", encoding="utf-8")
        return tmp_path

    def test_no_claims_leaves_text_unchanged(self, tmp_path: Path) -> None:
        ws = self._build_ws(tmp_path)
        analysis = "## 1. SCOPE\nA csv converter with streaming support."
        result = asyncio.run(self._verify(ws, analysis))
        assert result.checked == 0
        assert result.flagged == 0
        assert result.text == analysis
        assert "Verification Report" not in result.text

    def test_verified_claims_report_clean(self, tmp_path: Path) -> None:
        ws = self._build_ws(tmp_path)
        analysis = (
            "The `_execute_nlp_tool` method in `agent.py` runs `subprocess.run(..., shell=True)` "
            "without validation."
        )
        result = asyncio.run(self._verify(ws, analysis))
        assert result.checked >= 3
        assert result.flagged == 0
        assert "## Verification Report" in result.text
        assert "0 flagged" in result.text

    def test_missing_file_flagged(self, tmp_path: Path) -> None:
        ws = self._build_ws(tmp_path)
        result = asyncio.run(self._verify(ws, "The flaw lives in `non_existent.py`."))
        assert result.flagged >= 1
        assert "[UNVERIFIED]" in result.text
        assert "file not found in workspace" in result.text

    def test_absence_claim_confirmed_clean(self, tmp_path: Path) -> None:
        ws = self._build_ws(tmp_path)
        result = asyncio.run(self._verify(ws, "`_is_safe_extension` is not called anywhere."))
        assert result.checked >= 1
        assert result.flagged == 0
        assert "0 flagged" in result.text

    def test_absence_claim_contradicted_by_code(self, tmp_path: Path) -> None:
        ws = self._build_ws(tmp_path)
        analysis = "`_execute_nlp_tool` is not called anywhere in `agent.py`."
        result = asyncio.run(self._verify(ws, analysis))
        assert result.flagged >= 1
        assert "claim says it is absent" in result.text

    def test_line_out_of_range_flagged(self, tmp_path: Path) -> None:
        ws = self._build_ws(tmp_path)
        result = asyncio.run(self._verify(ws, "See `agent.py` line 9999."))
        assert result.flagged >= 1
        assert "out of range" in result.text

    def test_line_in_range_clean(self, tmp_path: Path) -> None:
        ws = self._build_ws(tmp_path)
        result = asyncio.run(self._verify(ws, "The bug is near `agent.py` line 4."))
        assert result.flagged == 0

    def test_symbol_line_mismatch_flagged(self, tmp_path: Path) -> None:
        ws = self._build_ws(tmp_path)
        analysis = "`_execute_nlp_tool` is defined at line ~40 in `agent.py`."
        result = asyncio.run(self._verify(ws, analysis))
        assert result.flagged >= 1
        assert "defined at agent.py:4" in result.text
        assert "claimed line 40" in result.text

    def test_symbol_line_match_not_flagged(self, tmp_path: Path) -> None:
        ws = self._build_ws(tmp_path)
        analysis = "`_execute_nlp_tool` is defined at line 4 in `agent.py`."
        result = asyncio.run(self._verify(ws, analysis))
        assert result.flagged == 0

    def test_symbol_line_does_not_bleed_across_bullets(self, tmp_path: Path) -> None:
        ws = self._build_ws(tmp_path)
        analysis = (
            "- Broad exception swallowing masks failures at `agent.py` line ~86.\n"
            "- DRY violations between agent.py and FileSystem/FileSearcher: `_safe_path`, "
            "`FileSystem.read` duplicate path normalization."
        )
        result = asyncio.run(self._verify(ws, analysis))
        assert not any("claimed line 86" in flag for flag in self._flagged_lines(result))

    def test_stdlib_symbol_not_flagged(self, tmp_path: Path) -> None:
        ws = self._build_ws(tmp_path)
        result = asyncio.run(self._verify(ws, "`subprocess` drives all execution."))
        assert result.flagged == 0

    def test_code_pattern_not_found_flagged(self, tmp_path: Path) -> None:
        ws = self._build_ws(tmp_path)
        analysis = "`agent.py` relies on `shell=False` everywhere."
        result = asyncio.run(self._verify(ws, analysis))
        assert result.flagged >= 1
        assert "code pattern not found in file" in result.text

    def test_context_file_scoped_symbol_check(self, tmp_path: Path) -> None:
        ws = self._build_ws(tmp_path)
        analysis = (
            "`tools/util.py` uses `util_fn` for invocation.\n\n"
            "`agent.py` calls `util_fn` directly."
        )
        result = asyncio.run(self._verify(ws, analysis))
        assert result.flagged >= 1
        # Mis-scoped symbol (exists elsewhere but claimed in wrong file) is flagged
        # with an informative reason pointing to the actual definition location.
        assert "but not in claimed file agent.py" in result.text

    def test_backslash_path_in_backticks(self, tmp_path: Path) -> None:
        ws = self._build_ws(tmp_path)
        analysis = "See `tools\\util.py` for `util_fn`."
        result = asyncio.run(self._verify(ws, analysis))
        assert result.flagged == 0

    def test_bare_file_with_colon_line(self, tmp_path: Path) -> None:
        ws = self._build_ws(tmp_path)
        analysis = "agent.py:3 calls subprocess."
        result = asyncio.run(self._verify(ws, analysis))
        assert result.flagged == 0
        assert "Code claims checked" in result.text

    def test_empty_workspace_returns_unchanged(self, tmp_path: Path) -> None:
        result = asyncio.run(verify_analysis_claims("`agent.py` has bugs.", tmp_path))
        assert result.checked == 0
        assert result.text == "`agent.py` has bugs."

    def test_constants_are_recognized_symbols(self, tmp_path: Path) -> None:
        ws = self._build_ws(tmp_path)
        result = asyncio.run(self._verify(ws, "`_FORBIDDEN_PATTERNS` guards the sanitizer."))
        assert result.flagged == 0

    def test_symbol_without_file_context_uses_workspace(self, tmp_path: Path) -> None:
        ws = self._build_ws(tmp_path)
        result = asyncio.run(self._verify(ws, "`util_fn` is reused across the codebase."))
        assert result.flagged == 0


class TestWriteVerifiedAnalysis:
    """_write_verified_analysis writes only when verification changes the text."""

    def test_writes_verified_text_when_changed(self, tmp_path: Path) -> None:
        from agent_core.commands.workflow_cmd import _write_verified_analysis

        (tmp_path / "agent.py").write_text(_AGENT_PY, encoding="utf-8")
        analysis_md = str(tmp_path / "project_analysis.md")
        analysis = "The bug is in `non_existent.py`."
        written, checked, flagged = asyncio.run(_write_verified_analysis(analysis, analysis_md, tmp_path))
        assert written == Path(analysis_md).read_text(encoding="utf-8")
        assert "## Verification Report" in written
        assert "[UNVERIFIED]" in written
        assert checked == 1
        assert flagged == 1

    def test_skips_rewrite_when_no_claims(self, tmp_path: Path) -> None:
        from agent_core.commands.workflow_cmd import _write_verified_analysis

        (tmp_path / "agent.py").write_text(_AGENT_PY, encoding="utf-8")
        analysis_md = str(tmp_path / "project_analysis.md")
        with open(analysis_md, "w", encoding="utf-8") as f:
            f.write("placeholder")
        written, checked, flagged = asyncio.run(_write_verified_analysis("No code references here.", analysis_md, tmp_path))
        assert written == "No code references here."
        assert checked == 0
        assert flagged == 0
        assert Path(analysis_md).read_text(encoding="utf-8") == "placeholder"