"""First-class *propose* mode — generate code, validate it in memory, and emit
a self-contained proposal bundle instead of mutating the working tree.

This module is the shared core behind three surfaces:

* ``implement --propose`` — generate modules/files but never write them.
* ``fix --propose`` — produce a fix diff but never apply it.
* the standalone ``propose`` command — a thin wrapper that forwards to
  ``implement --propose`` so 100% of the generation logic is reused.

Why this is stronger than the driver's ``git stash`` checkpoint: the agent
never touches the tree at all.  The checkpoint becomes *the artifact* (a
diff + rationale + test result), so there is no dirty-tree window and no
``stash pop`` that can corrupt state.

All of the existing safety gates are reused verbatim — they are run, and a
file that would be rejected is recorded as ``rejected`` in the bundle rather
than being silently dropped or written.
"""

from __future__ import annotations

import difflib
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_core.commands.base import show_file_diff

# Reuse implement_cmd's module-level gate predicates (no duplication).
from agent_core.commands.implement_cmd import (  # noqa: F401
    _check_planned_duplicates,
    _find_safe_subpackage,
    _is_dangerous_filename,
    _is_planned_test_file,
    _STDLIB_COMMON,
)
from agent_core.patch_utils import apply_anchored_patch, apply_patch, split_source_lines


# --------------------------------------------------------------------------- #
# Git / metadata helpers
# --------------------------------------------------------------------------- #
def _git(args: list[str], cwd: str) -> tuple[int, str]:
    try:
        r = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30
        )
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as exc:  # pragma: no cover - defensive
        return 1, f"git error: {exc}"


def _git_head(cwd: str) -> str:
    rc, out = _git(["rev-parse", "--short", "HEAD"], cwd)
    return out.strip() if rc == 0 else "unknown"


def _git_status(cwd: str) -> str:
    rc, out = _git(["status", "--porcelain"], cwd)
    return out.strip() if rc == 0 else ""


# --------------------------------------------------------------------------- #
# In-memory apply (the only genuinely new logic vs. implement/fix)
# --------------------------------------------------------------------------- #
def in_memory_apply_patch(
    filename: str, existing_text: str, patch_text: str
) -> tuple[bool, str]:
    """Apply a ``[PATCH:]`` unified-diff to *existing_text* in memory.

    Returns ``(ok, patched_text_or_error)``.  Never touches the filesystem —
    this is the propose-mode analogue of ``fix_cmd``'s disk write.
    """
    if not existing_text:
        return False, "file does not exist on disk"
    ok, patched = apply_patch(patch_text, split_source_lines(existing_text))
    if not ok:
        ok, patched = apply_anchored_patch(patch_text, split_source_lines(existing_text))
    if not ok:
        return False, str(patched)
    patched_text = str(patched)
    if patched_text == existing_text:
        return False, "patch produced no change"
    try:
        compile(patched_text, filename, "exec")
    except SyntaxError as exc:
        return False, f"patched file does not compile: {exc}"
    return True, patched_text


# --------------------------------------------------------------------------- #
# Gate predicates (mirror implement_cmd's write-loop gates)
# --------------------------------------------------------------------------- #
_RE_3 = re.compile(r"_\d+$|_v\d+$|_clean$|_final$")
_DEF_RE = re.compile(r"def\s+(\w+)")


def gate_file(
    filename: str,
    content: str,
    ws: str,
    *,
    allow_rewrite: bool = False,
    force_mode: bool = False,
) -> tuple[str, str, str]:
    """Run implement_cmd's pre-write gates in memory.

    Returns ``(status, reason, effective_filename)`` where *status* is one of
    ``accepted`` / ``rejected`` / ``skipped`` and *effective_filename* is the
    (possibly auto-repaired) path the proposal should record.
    """
    effective = filename
    workspace = Path(ws)

    # Dangerous filename (root-level / stdlib-shadow) — mirror implement_cmd.
    dangerous, reason = _is_dangerous_filename(filename, workspace)
    if dangerous:
        if "/" not in filename and "\\" not in filename:
            safe_dir = _find_safe_subpackage(workspace)
            new_filename = f"{safe_dir}/{filename}"
            new_filepath = workspace / new_filename
            new_dangerous, _ = _is_dangerous_filename(new_filename, workspace)
            if not new_dangerous:
                if new_filepath.exists() and new_filepath.stat().st_size > 100:
                    return (
                        "skipped",
                        f"{new_filename} already exists (avoiding overwrite)",
                        effective,
                    )
                effective = new_filename
            else:
                return "rejected", reason, effective
        elif "shadows stdlib" in reason:
            f_parts = filename.replace("\\", "/").split("/")
            for idx, part in enumerate(f_parts[:-1]):
                if part in _STDLIB_COMMON:
                    f_parts[idx] = part + "_utils"
                    effective = "/".join(f_parts)
                    break
        else:
            return "rejected", reason, effective

    if filename.endswith(".py"):
        func_names = _DEF_RE.findall(content)
        if len(func_names) > 20:
            similar: dict[str, int] = {}
            for name in func_names:
                prefix = _RE_3.sub("", name)
                similar[prefix] = similar.get(prefix, 0) + 1
            max_dupes = max(similar.values()) if similar else 1
            if max_dupes > 10:
                return (
                    "rejected",
                    f"{max_dupes} near-duplicate functions",
                    effective,
                )
        if len(content) > 50000:
            return "rejected", f"{len(content)} bytes, max 50KB", effective
        if len(content) < 10:
            return "rejected", "empty response from LLM", effective

    # Near-duplicate module gate (new modules only).
    if not force_mode and not effective.endswith(".py"):
        pass
    elif not force_mode:
        filepath = workspace / effective
        if not filepath.exists() and not _is_planned_test_file(effective):
            dup_reasons = _check_planned_duplicates([effective], ws, "")
            if dup_reasons:
                return (
                    "rejected",
                    f"near-duplicate of existing module — {dup_reasons[0].split(' — ', 1)[-1]}",
                    effective,
                )

    # Wholesale-rewrite guard for existing files.
    filepath = workspace / effective
    if (
        filepath.exists()
        and filename.endswith(".py")
        and not allow_rewrite
        and not force_mode
    ):
        try:
            existing = filepath.read_text(encoding="utf-8")
        except OSError:
            existing = ""
        if existing.strip():
            similarity = difflib.SequenceMatcher(None, existing, content).ratio()
            if similarity < 0.5:
                return (
                    "rejected",
                    f"wholesale rewrite of existing file (similarity {similarity:.2f})",
                    effective,
                )

    # Final syntax gate.
    if filename.endswith(".py"):
        try:
            compile(content, effective, "exec")
        except SyntaxError as exc:
            return "rejected", f"does not compile: {exc}", effective

    return "accepted", "", effective


# --------------------------------------------------------------------------- #
# Test / security validation (best-effort, optional)
# --------------------------------------------------------------------------- #
def run_proposal_tests(
    ws: str, generated_content: dict[str, str]
) -> tuple[bool | None, str]:
    """Validate the proposal by overlaying it onto a temp copy of *ws* and
    running pytest there.  Returns ``(passed, detail)``; ``passed`` is ``None``
    when validation could not be performed.
    """
    ws_path = Path(ws)
    if not (ws_path / "tests").exists() and not (ws_path / "test").exists():
        return None, "no tests/ directory — skipping validation"
    tmp = None
    try:
        tmp = tempfile.mkdtemp(prefix="propose_")
        # Copy the workspace, excluding heavy / VCS / cache dirs.
        exclude = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache"}
        for root, dirs, files in os.walk(ws_path):
            dirs[:] = [d for d in dirs if d not in exclude]
            rel = os.path.relpath(root, ws_path)
            dest = os.path.join(tmp, rel) if rel != "." else tmp
            os.makedirs(dest, exist_ok=True)
            for fn in files:
                if fn.endswith(".pyc"):
                    continue
                try:
                    shutil.copy2(os.path.join(root, fn), os.path.join(dest, fn))
                except OSError:
                    pass
        # Overlay the proposed files.
        for fname, content in generated_content.items():
            target = os.path.join(tmp, fname)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
        r = subprocess.run(
            ["python", "-m", "pytest", tmp, "-q", "--no-header", "-p", "no:cacheprovider"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        tail = (r.stdout or "") + (r.stderr or "")
        tail = tail[-1500:]
        return (r.returncode == 0, f"pytest rc={r.returncode}\n{tail}")
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"test validation skipped: {exc}"
    finally:
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


def run_proposal_security() -> tuple[bool | None, str]:
    """Reuse the existing fast security gate as a proposal sanity check."""
    try:
        from harnessfix.gates import run_security_gate

        return run_security_gate()
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"security validation skipped: {exc}"


# --------------------------------------------------------------------------- #
# Bundle rendering
# --------------------------------------------------------------------------- #
def build_patch(generated_content: dict[str, str], ws: str) -> str:
    """Render a ``git apply``-compatible unified diff for every proposed file."""
    chunks: list[str] = []
    for filename in sorted(generated_content):
        filepath = Path(ws) / filename
        original = ""
        if filepath.exists():
            try:
                original = filepath.read_text(encoding="utf-8")
            except OSError:
                original = ""
        new = generated_content[filename]
        diff = difflib.unified_diff(
            original.splitlines(),
            new.splitlines(),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm="",
        )
        chunks.append("\n".join(diff))
    joined = "\n".join(chunks)
    return joined + ("\n" if joined else "")


def build_bundle(
    generated_content: dict[str, str],
    ws: str,
    file_outcomes: dict[str, str],
    meta: dict[str, Any],
) -> str:
    """Render the human-readable markdown proposal bundle."""
    lines: list[str] = []
    lines.append("# Proposal Bundle")
    lines.append("")
    lines.append(f"- **command**: `{meta.get('command', 'propose')}`")
    lines.append(f"- **generated**: {meta.get('timestamp', '')}")
    lines.append(f"- **model**: {meta.get('model', 'unknown')}")
    lines.append(f"- **workspace**: `{ws}`")
    head = meta.get("git_head", "unknown")
    lines.append(f"- **base HEAD**: `{head}`")
    status = meta.get("git_status", "")
    if status:
        lines.append("")
        lines.append("**Base working-tree status (must match before merge):**")
        lines.append("")
        lines.append("```")
        lines.append(status)
        lines.append("```")
    tr = meta.get("test_result")
    sr = meta.get("security_result")
    if tr is not None:
        passed, detail = tr
        lines.append(f"- **tests**: {'PASS' if passed else 'FAIL'} — {detail}")
    if sr is not None:
        passed, detail = sr
        lines.append(f"- **security**: {'PASS' if passed else 'FAIL'} — {detail}")
    lines.append("")

    accepted = [f for f, o in file_outcomes.items() if o == "accepted"]
    rejected = [f for f, o in file_outcomes.items() if o.startswith("rejected")]
    skipped = [f for f, o in file_outcomes.items() if o.startswith("skipped")]

    lines.append(f"**Summary**: {len(accepted)} accepted, "
                 f"{len(rejected)} rejected, {len(skipped)} skipped.")
    lines.append("")
    lines.append("## Merge")
    lines.append("")
    lines.append("```bash")
    lines.append(f"git apply reports/proposals/{meta.get('ts', 'latest')}/proposal.patch")
    lines.append("```")
    lines.append("")

    for filename in sorted(generated_content):
        outcome = file_outcomes.get(filename, "accepted")
        status_label = outcome.upper()
        lines.append(f"## {status_label}: `{filename}`")
        lines.append("")
        rationale = meta.get("rationale", {}).get(filename)
        if rationale:
            lines.append(f"**Rationale**: {rationale}")
            lines.append("")
        filepath = Path(ws) / filename
        original = ""
        if filepath.exists():
            try:
                original = filepath.read_text(encoding="utf-8")
            except OSError:
                original = ""
        diff = "\n".join(
            difflib.unified_diff(
                original.splitlines(),
                generated_content[filename].splitlines(),
                fromfile=f"a/{filename}",
                tofile=f"b/{filename}",
                lineterm="",
            )
        )
        if diff.strip():
            lines.append("```diff")
            lines.append(diff)
            lines.append("```")
        else:
            lines.append("_(no diff — file unchanged)_")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Emit
# --------------------------------------------------------------------------- #
def emit_proposal(
    generated_content: dict[str, str],
    ws: str,
    file_outcomes: dict[str, str],
    meta: dict[str, Any] | None = None,
    out_dir: str | None = None,
) -> str:
    """Write the proposal bundle (markdown + .patch) and return its directory.

    Never writes anything into the tracked tree besides the ``reports/``
    artifact folder — the proposal is an output, not a source change.
    """
    meta = dict(meta or {})
    ts = meta.get("ts") or datetime.now().strftime("%Y%m%d_%H%M%S")
    meta["ts"] = ts
    meta.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    meta.setdefault("git_head", _git_head(ws))
    meta.setdefault("git_status", _git_status(ws))

    base = Path(out_dir) if out_dir else (Path(ws) / "reports" / "proposals" / ts)
    base.mkdir(parents=True, exist_ok=True)

    bundle_md = build_bundle(generated_content, ws, file_outcomes, meta)
    patch_txt = build_patch(generated_content, ws)

    (base / "proposal.md").write_text(bundle_md, encoding="utf-8")
    (base / "proposal.patch").write_text(patch_txt, encoding="utf-8")

    print(f"\n{'='*50}")
    print(f"PROPOSAL emitted (tree NOT modified): {base}")
    print(f"  proposal.md  ({len(bundle_md)} bytes)")
    print(f"  proposal.patch ({len(patch_txt)} bytes)")
    accepted = [f for f, o in file_outcomes.items() if o == "accepted"]
    rejected = [f for f, o in file_outcomes.items() if o.startswith("rejected")]
    skipped = [f for f, o in file_outcomes.items() if o.startswith("skipped")]
    print(f"  accepted={len(accepted)} rejected={len(rejected)} skipped={len(skipped)}")
    if accepted:
        for f in accepted:
            print(f"    + {f}")
    if rejected:
        for f in rejected:
            print(f"    - {f}: {file_outcomes[f]}")
    print(f"\n  Merge with: git apply {base / 'proposal.patch'}")
    print(f"{'='*50}")
    return str(base)
