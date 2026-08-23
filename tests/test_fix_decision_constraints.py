"""Regression tests: fix must inject decisions from the WORKSPACE ledger.

Bug (2026-08-19): ``_fix_traceback`` passed ``Path(fpath).parent`` as the
workspace to ``decisions_as_system_prompt``.  The decision ledger lives at
the workspace root, so for any nested file ``load_decisions`` silently
returned ``[]`` and the LLM never saw its design constraints.  Matching on
a bare basename also never hit records whose affected_files are stored in
canonical workspace-relative form.
"""
from pathlib import Path

from agent_core.commands.fix_cmd import _decision_constraints_for, _decision_workspace_for
from agent_core.decisions import add_decision, load_decisions


def _make_ws(tmp_path: Path) -> Path:
    """Workspace root with a ledger + a nested package file."""
    pkg = tmp_path / "agent_core" / "commands"
    pkg.mkdir(parents=True)
    target = pkg / "fix_cmd.py"
    target.write_text("x = 1\n", encoding="utf-8")
    # A decoy ledger in the file's own directory must NOT be required —
    # only the root one exists here.
    add_decision(
        str(tmp_path),
        "No _v1/_v2 variants",
        decision="Never create duplicate function variants.",
        rationale="Corruption source.",
        affected_files=["agent_core/commands/fix_cmd.py"],
    )
    return tmp_path


class TestDecisionConstraintsFor:
    def test_nested_file_finds_root_ledger(self, tmp_path):
        """THE regression: nested file previously got an empty constraints block."""
        ws = _make_ws(tmp_path)
        target = ws / "agent_core" / "commands" / "fix_cmd.py"
        block = _decision_constraints_for(str(target))
        assert block != "", "constraints lost for nested file"
        assert "CRITICAL DESIGN CONSTRAINTS" in block
        assert "No _v1/_v2 variants" in block

    def test_absolute_path_matches_canonical_record(self, tmp_path):
        """Absolute caller path matches records stored workspace-relative."""
        ws = _make_ws(tmp_path)
        target = ws / "agent_core" / "commands" / "fix_cmd.py"
        assert str(target.resolve()) in _decision_constraints_for(str(target)) or \
            "No _v1/_v2 variants" in _decision_constraints_for(str(target))
        # The record's canonical relative form must appear in matching.
        block = _decision_constraints_for(str(target))
        assert "Never create duplicate function variants." in block

    def test_unrelated_file_gets_no_constraints(self, tmp_path):
        ws = _make_ws(tmp_path)
        other = ws / "unrelated.py"
        other.write_text("y = 2\n", encoding="utf-8")
        assert _decision_constraints_for(str(other)) == ""

    def test_no_ledger_anywhere_returns_empty(self, tmp_path):
        lone = tmp_path / "lone.py"
        lone.write_text("z = 3\n", encoding="utf-8")
        assert _decision_constraints_for(str(lone)) == ""

    def test_none_and_missing_paths_are_safe(self, tmp_path):
        assert _decision_constraints_for(None) == ""
        assert _decision_constraints_for("") == ""
        missing = tmp_path / "does" / "not" / "exist.py"
        assert _decision_constraints_for(str(missing)) == ""


class TestDecisionWorkspaceFor:
    """Recording side: new decisions must land in the EXISTING root ledger."""

    def test_records_go_to_root_ledger_not_file_dir(self, tmp_path):
        """THE regression: add_decision used Path(fpath).parent, fragmenting
        the ledger into the file's own directory."""
        ws = _make_ws(tmp_path)
        target = ws / "agent_core" / "commands" / "fix_cmd.py"
        ws_str = _decision_workspace_for(str(target))
        record = add_decision(ws_str, "Second decision", decision="Y")
        # The record must be readable from the ROOT ledger...
        root_ledger = load_decisions(str(ws))
        titles = [d["title"] for d in root_ledger]
        assert "No _v1/_v2 variants" in titles
        assert "Second decision" in titles
        assert record["id"] == "002", "fragmented ledger would restart ids at 001"
        # ...and no decoy ledger may appear next to the file.
        assert not (target.parent / ".decisions.json").exists()

    def test_no_existing_ledger_defaults_to_parent(self, tmp_path):
        lone = tmp_path / "lone.py"
        lone.write_text("z = 3\n", encoding="utf-8")
        assert _decision_workspace_for(str(lone)) == str(lone.parent)

    def test_none_path_is_safe(self, tmp_path):
        assert _decision_workspace_for(None) == ""
