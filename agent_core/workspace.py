"""Single source of truth for "which directory is the agent working on?".

Before this module the answer was implicit and derived from the *code*::

    AgentCore.DEFAULT_WORKSPACE = dirname(dirname(abspath(__file__)))

That is Agent1's own checkout, resolved from ``agent_core/agent.py``'s
location, so starting the agent inside any other repository (or from a plain
working directory) silently aimed every tool at Agent1 instead.  Three
different layers each had their own idea of the root:

* ``AgentCore.DEFAULT_WORKSPACE`` (package dir) — what the REPL used,
* ``agent_core.config.AgentConfig.workspace_root`` (``AGENT_WORKSPACE_ROOT``
  or ``Path.cwd()``) — documented but never read by ``Agent``,
* ``session.store.repo_root()`` (``parents[2]``) — what startup pinned.

:func:`resolve_workspace` collapses those into one precedence chain:

1. an explicit path (CLI ``--workspace``, REPL ``workspace <path>``),
2. the ``AGENT_WORKSPACE_ROOT`` environment variable,
3. the current working directory, but only when it looks like a project
   (``.git`` / ``pyproject.toml``) so a stray ``cd`` into e.g. ``C:\\Windows``
   never becomes the workspace,
4. ``session.store.repo_root()`` — the historical default, kept as the last
   resort so an existing single-repo setup behaves exactly as before.

The function never raises: a bad override falls through to the next source,
because every caller (agent boot, CLI, tools) needs a usable path, not an
exception.  ``session.store.repo_root`` is imported lazily to keep this module
free of an import cycle with the session package.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

#: Environment variable that overrides the workspace for a whole process.
WORKSPACE_ENV = "AGENT_WORKSPACE_ROOT"

#: Files that mark a directory as a project root worth adopting as workspace.
REPO_MARKERS = (".git", "pyproject.toml", "setup.py", "package.json")


def looks_like_repo(path: str | os.PathLike[str]) -> bool:
    """Return True when *path* is a directory carrying a project marker."""
    try:
        root = Path(path)
    except (TypeError, ValueError):
        return False
    if not root.is_dir():
        return False
    return any((root / marker).exists() for marker in REPO_MARKERS)


def _repo_root() -> str:
    """The historical default workspace (Agent1's own checkout)."""
    try:
        from agent_core.session.store import repo_root

        return str(repo_root())
    except Exception:  # pragma: no cover - defensive: a broken import must not
        # stop the agent from starting with SOME workspace.
        logger.debug("repo_root() unavailable; falling back to cwd", exc_info=True)
        return os.getcwd()


def _normalise(raw: str | os.PathLike[str] | None) -> str:
    """Expand and absolutise *raw* (Windows-style paths included)."""
    text = os.path.expanduser(str(raw).strip())
    try:
        from agent_core.path_utils import to_windows_path

        text = to_windows_path(text)
    except Exception:  # pragma: no cover - pure convenience helper
        logger.debug("to_windows_path unavailable", exc_info=True)
    return os.path.abspath(text)


def resolve_workspace(
    explicit: str | os.PathLike[str] | None = None,
    *,
    env: dict[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
) -> str:
    """Resolve the agent workspace, honouring the documented precedence.

    *explicit* wins, then ``AGENT_WORKSPACE_ROOT``, then *cwd* when it looks
    like a project, then the repository root.  Returns an absolute path; never
    raises.  *env* and *cwd* exist so tests can pin the inputs.
    """
    environ = os.environ if env is None else env
    here = os.getcwd() if cwd is None else str(cwd)

    if explicit:
        text = str(explicit).strip()
        if text:
            resolved = _normalise(text)
            if os.path.isdir(resolved):
                return resolved
            # A non-existent override must not silently become the default --
            # that is the failure mode this module exists to kill.
            logger.warning(
                "Workspace override %r is not a directory; ignoring it.", text
            )

    from_env = str(environ.get(WORKSPACE_ENV) or "").strip()
    if from_env:
        resolved = _normalise(from_env)
        if os.path.isdir(resolved):
            return resolved
        logger.warning(
            "%s=%r is not a directory; ignoring it.", WORKSPACE_ENV, from_env
        )

    if looks_like_repo(here):
        return _normalise(here)

    return _normalise(_repo_root())


def describe_workspace_mismatch(
    workspace: str | os.PathLike[str] | None,
    *,
    cwd: str | os.PathLike[str] | None = None,
) -> str:
    """Return a one-line warning when the agent works outside the caller's cwd.

    The agent can legitimately work in a root other than the shell's directory
    (``--workspace``, ``AGENT_WORKSPACE_ROOT``, a resumed session), but the
    human in front of the terminal is standing in *cwd* and tends to assume the
    two match.  That assumption is how a run meant for one checkout lands in
    another.  Returns ``""`` when they are the same directory, so a normal
    single-repo start stays silent.
    """
    if not workspace:
        return ""
    if not isinstance(workspace, (str, os.PathLike)):
        return ""
    here = os.getcwd() if cwd is None else str(cwd)
    try:
        left = os.path.normcase(os.path.abspath(str(workspace)))
        right = os.path.normcase(os.path.abspath(here))
    except (TypeError, ValueError):
        return ""
    if left == right:
        return ""
    return (
        f"[workspace] working in {workspace} -- started from {here}. "
        "Relative paths and tool calls use the workspace, not your shell."
    )


def describe_target_divergence(
    target: str | os.PathLike[str] | None,
    agent_workspace: str | os.PathLike[str] | None,
) -> str:
    """Return a warning when a command works on a root the chat is not using.

    Commands such as ``implement`` and ``workflow`` accept their own
    ``--workspace``.  That is deliberate, but it silently forks the command's
    world from the conversation's: the chat keeps discussing one checkout
    while the command writes files into another.  Returns ``""`` when both
    point at the same directory, so the common case stays quiet.
    """
    if not target or not agent_workspace:
        return ""
    if not isinstance(target, (str, os.PathLike)):
        return ""
    if not isinstance(agent_workspace, (str, os.PathLike)):
        return ""
    try:
        left = os.path.normcase(os.path.abspath(str(target)))
        right = os.path.normcase(os.path.abspath(str(agent_workspace)))
    except (TypeError, ValueError):
        return ""
    if left == right:
        return ""
    return (
        f"[workspace] this command targets {target}, but the chat is working "
        f"in {agent_workspace}. Files are written to the command's target."
    )


__all__ = [
    "REPO_MARKERS",
    "WORKSPACE_ENV",
    "describe_target_divergence",
    "describe_workspace_mismatch",
    "looks_like_repo",
    "resolve_workspace",
]
