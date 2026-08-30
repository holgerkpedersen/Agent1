"""Tests for first-class propose mode (agent_core.commands.proposal_core).

These are pure unit tests — no LLM, no real tree writes.  They lock in the
core invariants of propose mode:

* the agent NEVER writes the working tree (the whole point — stronger than a
  git-stash checkpoint);
* [PATCH:] diffs are applied in memory and validated;
* the existing safety gates (wholesale-rewrite, syntax, dangerous filename)
  still fire inside propose mode and mark a file rejected rather than accepted;
* the emitted bundle is a self-contained, git-apply-compatible artifact.
"""

from __future__ import annotations

from pathlib import Path

from agent_core.commands import proposal_core as pc


def _write_tree(tmp_path: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


# --------------------------------------------------------------------------- #
# in_memory_apply_patch
# --------------------------------------------------------------------------- #
def test_in_memory_apply_patch_adds_line(tmp_path: Path) -> None:
    _write_tree(tmp_path, {"mod.py": "def f():\n    return 1\n"})
    existing = (tmp_path / "mod.py").read_text()
    patch = (
        "@@ -1,2 +1,3 @@\n"
        " def f():\n"
        "+    pass\n"
        "     return 1\n"
    )
    ok, result = pc.in_memory_apply_patch("mod.py", existing, patch)
    assert ok, result
    assert "    pass\n" in result
    # The source tree is untouched.
    assert (tmp_path / "mod.py").read_text() == existing


def test_in_memory_apply_patch_rejects_syntax_break(tmp_path: Path) -> None:
    _write_tree(tmp_path, {"mod.py": "def f():\n    return 1\n"})
    existing = (tmp_path / "mod.py").read_text()
    patch = (
        "@@ -1,2 +1,2 @@\n"
        " def f():\n"
        "-    return 1\n"
        "+    return (\n"
    )
    ok, result = pc.in_memory_apply_patch("mod.py", existing, patch)
    assert not ok
    assert "syntax" in str(result).lower()


def test_in_memory_apply_patch_no_change_is_false(tmp_path: Path) -> None:
    _write_tree(tmp_path, {"mod.py": "x = 1\n"})
    existing = (tmp_path / "mod.py").read_text()
    patch = "@@ -1,1 +1,1 @@\n x = 1\n"
    ok, _ = pc.in_memory_apply_patch("mod.py", existing, patch)
    assert not ok


# --------------------------------------------------------------------------- #
# gate_file — reuse of implement_cmd's pre-write gates
# --------------------------------------------------------------------------- #
def test_gate_accepts_clean_new_file(tmp_path: Path) -> None:
    status, reason, eff = pc.gate_file(
        "pkg/newmod.py", "def g():\n    return 2\n", str(tmp_path)
    )
    assert status == "accepted"
    assert eff == "pkg/newmod.py"


def test_gate_rejects_broken_syntax(tmp_path: Path) -> None:
    status, reason, _ = pc.gate_file(
        "broken.py", "def h(\n    pass\n", str(tmp_path)
    )
    assert status == "rejected"
    assert "compile" in reason.lower()


def test_gate_rejects_wholesale_rewrite_of_existing(tmp_path: Path) -> None:
    # Use a subpackage path so the root-level auto-repair does NOT move it
    # (which would turn it into a "new file" and trip the dup-check instead).
    _write_tree(tmp_path, {"pkg/keep.py": "A = 1\nB = 2\nC = 3\nD = 4\n"})
    # A drastically different body (similarity < 0.5) must be rejected unless
    # allow_rewrite is set — same guard as the real implement write loop.
    status, reason, _ = pc.gate_file(
        "pkg/keep.py", "import os\n\ndef totally_different():\n    return os.getcwd()\n",
        str(tmp_path),
    )
    assert status == "rejected"
    assert "wholesale" in reason.lower()


def test_gate_allows_rewrite_with_allow_rewrite(tmp_path: Path) -> None:
    _write_tree(tmp_path, {"pkg/keep.py": "A = 1\n"})
    status, _, _ = pc.gate_file(
        "pkg/keep.py", "import os\n\ndef totally_different():\n    return os.getcwd()\n",
        str(tmp_path), allow_rewrite=True,
    )
    assert status == "accepted"


def test_gate_auto_repairs_dangerous_root_filename(tmp_path: Path) -> None:
    # A bare root-level .py (collision target) is auto-repaired into a subpkg.
    status, _, eff = pc.gate_file("utils.py", "def q():\n    return 1\n", str(tmp_path))
    assert status == "accepted"
    assert "/" in eff, "dangerous root filename should be auto-repaired into a subpackage"


# --------------------------------------------------------------------------- #
# emit_proposal — bundle + patch, and the NEVER-WRITE invariant
# --------------------------------------------------------------------------- #
def test_emit_proposal_creates_bundle_and_patch(tmp_path: Path) -> None:
    _write_tree(tmp_path, {"orig.py": "x = 1\n"})
    generated = {"orig.py": "x = 2\n", "new.py": "def f():\n    return 1\n"}
    outcomes = {"orig.py": "accepted", "new.py": "accepted"}
    out = pc.emit_proposal(
        generated, str(tmp_path), outcomes,
        meta={"command": "implement --propose", "model": "test"},
        out_dir=str(tmp_path / "reports" / "proposals" / "t1"),
    )
    base = Path(out)
    md = (base / "proposal.md").read_text()
    patch = (base / "proposal.patch").read_text()
    assert "Proposal Bundle" in md
    assert "## ACCEPTED:" in md
    assert "def f()" in md
    # git-apply-compatible unified diff with a/ b/ prefixes.
    assert "--- a/orig.py" in patch
    assert "+++ b/orig.py" in patch
    assert "+x = 2" in patch


def test_emit_proposal_records_rejected_files(tmp_path: Path) -> None:
    generated = {"bad.py": "def h(\n"}
    outcomes = {"bad.py": "rejected — does not compile: ..."}
    out = pc.emit_proposal(
        generated, str(tmp_path), outcomes,
        meta={"command": "implement --propose"},
        out_dir=str(tmp_path / "reports" / "proposals" / "t2"),
    )
    md = (Path(out) / "proposal.md").read_text()
    assert "## REJECTED" in md
    assert "bad.py" in md


def test_propose_never_writes_the_tree(tmp_path: Path) -> None:
    """The single most important invariant: emitting a proposal must not
    create or modify any tracked source file in the workspace."""
    _write_tree(tmp_path, {"src.py": "old = 1\n"})
    before = {p.name: p.read_text() for p in tmp_path.rglob("*.py")}
    generated = {"src.py": "old = 2\n", "extra.py": "y = 9\n"}
    pc.emit_proposal(
        generated, str(tmp_path),
        {"src.py": "accepted", "extra.py": "accepted"},
        meta={"command": "implement --propose"},
        out_dir=str(tmp_path / "reports" / "proposals" / "t3"),
    )
    after = {p.name: p.read_text() for p in tmp_path.rglob("*.py")}
    # src.py must be byte-identical — propose did not touch the tree.
    assert after["src.py"] == "old = 1\n"
    assert "extra.py" not in after  # new file was NOT written to the tree
    # The only new artifacts live under reports/.
    assert (tmp_path / "reports" / "proposals" / "t3" / "proposal.md").exists()
    assert (tmp_path / "reports" / "proposals" / "t3" / "proposal.patch").exists()


def test_build_patch_is_empty_for_identical_content(tmp_path: Path) -> None:
    _write_tree(tmp_path, {"a.py": "z = 1\n"})
    patch = pc.build_patch({"a.py": "z = 1\n"}, str(tmp_path))
    assert patch == ""
