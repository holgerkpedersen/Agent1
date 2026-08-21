"""Delete protection for sensitive agent workspace files.

Purpose
-------
The autonomous harness can run cleanup / repair actions that sweep the
workspace and delete unreferenced or stale files (see ``cleanup_cmd --delete``).
Two categories of file must NEVER be deleted by such sweeps, regardless of how
"unreferenced" they look:

* ``.env`` — holds live LLM API keys / provider credentials. Deleting it would
  silently break the agent's ability to talk to its model and could leak secrets
  if a backup copy were later committed.
* ``reports/*`` — captured trace/diagnosis artifacts that are the *point* of
  harnessfix runs; deleting them destroys the evidence base for every repair
  decision.

This module is the single choke point: any agent-driven file removal should go
through :func:`safe_remove` / :func:`safe_rmtree`, which raise
``SecurityViolationError`` on protected targets instead of deleting them.
"""
from __future__ import annotations

import os
from pathlib import Path

from .entities import SecurityViolationError


#: Filename (no path) treated as protected wherever it appears in the workspace.
PROTECTED_FILENAMES: frozenset[str] = frozenset({".env"})

#: Path prefixes, matched case-insensitively against forward-slash rel paths,
#: that are recursively protected under the workspace root.
PROTECTED_PREFIXES: tuple[str, ...] = ("reports/",)


def _rel_posix(path: os.PathLike[str] | str, workspace_root: os.PathLike[str] | str) -> str:
    """Return ``path`` as a forward-slash relative string against workspace_root.

    Raises ``SecurityViolationError`` if the path cannot be expressed as a
    clean relative path (i.e. it escapes the workspace), so callers never get
    a misleading "not protected" answer for an out-of-tree target.
    """
    root = Path(workspace_root).resolve()
    raw_path = Path(path)
    try:
        if raw_path.is_absolute():
            resolved = raw_path.resolve(strict=False)
        else:
            # Anchor bare relative paths against the workspace root, not CWD —
            # mirrors agent_core.path_utils.normalize_path so a rel string like
            # ".env" / "reports/traces/t.jsonl" is classified correctly.
            resolved = (root / raw_path).resolve(strict=False)
    except (OSError, RuntimeError):
        # Fall back to the literal path so we can still classify it; this only
        # happens for non-existent targets being considered for deletion.
        if raw_path.is_absolute():
            resolved = raw_path
        else:
            resolved = root / raw_path
    rel = os.path.relpath(resolved, root).replace("\\", "/")
    if rel.startswith(".."):
        raise SecurityViolationError(
            f"Cannot evaluate protection for path outside workspace: {resolved}"
        )
    return rel


def is_protected(path: os.PathLike[str] | str, workspace_root: os.PathLike[str] | str) -> bool:
    """True if ``path`` (relative to ``workspace_root``) must never be deleted.

    Matches the two protected categories:
      * a file literally named ``.env``;
      * anything under the ``reports/`` directory tree.

    A path that escapes ``workspace_root`` raises ``SecurityViolationError``
    rather than returning False — out-of-tree targets are not "unprotected",
    they're invalid inputs for an in-workspace sweep.
    """
    rel = _rel_posix(path, workspace_root)
    # .env anywhere in the tree (filename match, case-sensitive on POSIX).
    if Path(rel).name == ".env":
        return True
    # reports/ recursive prefix: protect the directory itself AND everything
    # under it.  Match "reports" exactly or any path starting with "reports/".
    lowered = rel.lower()
    for prefix in PROTECTED_PREFIXES:
        if lowered == prefix.rstrip("/") or lowered.startswith(prefix):
            return True
    return False


def safe_remove(path: os.PathLike[str] | str, workspace_root: os.PathLike[str] | str) -> None:
    """Delete a single file unless it is protected.

    Raises ``SecurityViolationError`` for ``.env`` / ``reports/*`` targets so
    the caller can skip them and report rather than silently destroying them.
    """
    if is_protected(path, workspace_root):
        raise SecurityViolationError(f"Protected file — refusing to delete: {path}")
    os.remove(os.fspath(path))


def safe_rmtree(path: os.PathLike[str] | str, workspace_root: os.PathLike[str] | str) -> None:
    """Recursively remove a directory tree unless it is protected.

    ``reports/`` itself and any sub-directory of it are refused; everything else
    under the workspace root is removed normally.  Out-of-tree targets raise
    ``SecurityViolationError``.
    """
    if is_protected(path, workspace_root):
        raise SecurityViolationError(f"Protected directory — refusing to delete: {path}")
    import shutil

    shutil.rmtree(os.fspath(path))


def protected_targets(paths: list[os.PathLike[str] | str], workspace_root: os.PathLike[str] | str) -> list[str]:
    """Filter ``paths`` down to only those that are protected (for reporting)."""
    out: list[str] = []
    for p in paths:
        try:
            if is_protected(p, workspace_root):
                out.append(str(p))
        except SecurityViolationError:
            # Out-of-tree targets aren't "protected" per the policy set; skip.
            continue
    return out
