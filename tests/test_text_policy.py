"""Tests for agent_core.text_policy (decision #079: no emojis in files).

Covers the detector primitives, the monochrome-glyph allowlist, tree
scanning (skips), and the audit-script wiring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_core import text_policy as tp


class TestIsEmojiChar:
    def test_true_emojis_flagged(self):
        for ch in ("\u2705", "\u26a1", "\U0001f3d7", "\U0001f600", "\ufe0f"):
            assert tp.is_emoji_char(ch), f"U+{ord(ch):04X} should be flagged"

    @pytest.mark.parametrize("ch", ["\u2713", "\u2717", "\u26a0"])
    def test_monochrome_status_marks_allowed(self, ch):
        assert not tp.is_emoji_char(ch)

    @pytest.mark.parametrize(
        "ch", ["\u2500", "\u2502", "\u250c", "\u2514", "\u25ba", "\u2550"]
    )
    def test_box_drawing_and_arrows_allowed(self, ch):
        assert not tp.is_emoji_char(ch)

    def test_typography_never_flagged(self):
        for ch in ("\u2014", "\u2264", "é", "\u4e2d", "a", "-", "?"):
            assert not tp.is_emoji_char(ch)

    def test_replacement_char_is_not_allowed(self):
        # U+FFFD marks corruption and must be reported, never silently allowed
        assert tp.is_emoji_char("\ufffd")

    def test_ascii_fast_path(self):
        assert not any(tp.is_emoji_char(c) for c in "abc XYZ 123\n\t")


class TestScanText:
    def test_finds_offending_line(self):
        rows = tp.scan_text("ok line\nbad \u2705 line\nfine")
        assert rows == [(2, "\u2705")]

    def test_clean_text_has_no_findings(self):
        text = "- [DONE] item — with em dash, \u2264 math, \u2713 mark, \u2500 box"
        assert tp.scan_text(text) == []

    def test_unique_chars_sorted(self):
        rows = tp.scan_text("\U0001f600\u26a1\u2705")
        assert rows == [(1, "".join(sorted("\U0001f600\u26a1\u2705")))]


class TestScanFile:
    def test_reads_utf8_file(self, tmp_path: Path):
        p = tmp_path / "x.md"
        p.write_text("hello \u26a1 world", encoding="utf-8")
        assert tp.scan_file(p) == [(1, "\u26a1")]

    def test_unreadable_file_yields_nothing(self, tmp_path: Path):
        p = tmp_path / "bin.md"
        p.write_bytes(b"\xff\xfe\x00\x00")
        assert tp.scan_file(p) == []

    def test_missing_file_yields_nothing(self, tmp_path: Path):
        assert tp.scan_file(tmp_path / "nope.md") == []


class TestScanTree:
    def _make(self, root: Path, rel: str, text: str) -> None:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def test_scans_tree_and_skips_dirs_and_state(self, tmp_path: Path):
        self._make(tmp_path, "a/b.py", "x = '\u2705'")
        self._make(tmp_path, "reports/traces/t.json", "{}")
        self._make(tmp_path, ".docs/run/plan.md", "#p")
        self._make(tmp_path, "chat_history.json", "[{}]")
        self._make(tmp_path, "notes.txt", "plain ascii")
        findings = tp.scan_tree(tmp_path)
        assert list(findings) == ["a/b.py"]

    def test_non_target_extensions_skipped(self, tmp_path: Path):
        self._make(tmp_path, "logo.svg", "<svg>\u2705</svg>")
        assert tp.scan_tree(tmp_path) == {}


class TestSummarizeFindings:
    def test_compact_output(self):
        findings = {"a.md": [(2, "\u2705")], "b.md": [(9, "\u26a1")] * 10}
        out = tp.summarize_findings(findings)
        assert "a.md L2" in out and "b.md" in out

    def test_overflow_note(self):
        findings = {f"f{i}.md": [(1, "\u2705")] for i in range(8)}
        out = tp.summarize_findings(findings, max_files=2)
        assert "...and 6 more file(s)" in out


class TestAuditScriptWiring:
    """The audit script must fail on emoji findings (decision #079 gate)."""

    def _load_audit_module(self):
        import importlib.util

        script = Path(__file__).resolve().parents[1] / "scripts" / "audit_invariants.py"
        spec = importlib.util.spec_from_file_location("audit_inv_tmp", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_real_repo_is_emoji_free(self):
        import subprocess

        script = Path(__file__).resolve().parents[1] / "scripts" / "audit_invariants.py"
        proc = subprocess.run(
            ["python", "-X", "utf8", str(script)],
            capture_output=True, text=True, encoding="utf-8",
            cwd=str(script.parent.parent),
        )
        out = proc.stdout + proc.stderr
        assert "emoji/pictograph symbols" not in out, out
        assert proc.returncode == 0, out

    def test_planted_emoji_fails_the_audit(self, tmp_path, monkeypatch, capsys):
        mod = self._load_audit_module()
        monkeypatch.setattr(mod, "ROOT", tmp_path)
        monkeypatch.setattr(mod, "scan_tree", lambda _r, **kw: {"bad.md": [(1, "\u2705")]})
        monkeypatch.setattr("sys.argv", ["audit_invariants.py"])

        rc = mod.main()

        assert rc == 1
        err_text = capsys.readouterr().out
        assert "decision #079" in err_text
        assert "bad.md" in err_text

    def test_clean_tree_passes_the_audit(self, tmp_path, monkeypatch, capsys):
        mod = self._load_audit_module()
        monkeypatch.setattr(mod, "ROOT", tmp_path)
        monkeypatch.setattr(mod, "scan_tree", lambda _r, **kw: {})
        monkeypatch.setattr("sys.argv", ["audit_invariants.py"])

        rc = mod.main()

        assert rc == 0
        assert "emoji" not in capsys.readouterr().out
