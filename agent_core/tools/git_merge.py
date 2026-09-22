"""Consistent, non-interactive ``git merge`` handling for the agent.

Why this module exists
----------------------
Live sessions drove ``git merge`` through the generic ``git`` tool and hit
three failure modes, each reproduced against a throwaway repo while writing
this module:

1. **``git merge --continue`` hangs forever.**  With no ``GIT_EDITOR`` set,
   git opens an editor to confirm the merge commit message; a subprocess
   with no tty and no timeout blocks indefinitely.
2. **``--continue`` rejects every argument.**  ``git merge --continue
   --no-edit`` exits 129 with ``fatal: --continue expects no arguments``, so
   the obvious "just make it non-interactive" fix is itself a dead end.  The
   only safe form is the bare ``git merge --continue``.
3. **No-op calls waste a turn.**  Starting a second merge while one is in
   progress answers ``fatal: Merging is not possible because you have
   unmerged files``, and ``--continue``/``--abort`` with nothing in progress
   answer a bare ``fatal: There is no merge in progress (MERGE_HEAD
   missing)`` — noise the model then misreads.

The fix is to own the merge state machine instead of forwarding a string:
this module inspects the real repository state (``MERGE_HEAD``, unmerged
paths), always runs non-interactively (``GIT_EDITOR`` no-op +
``GIT_MERGE_AUTOEDIT=no`` + ``stdin=DEVNULL`` + a hard timeout), never passes
an argument to ``--continue``, and answers impossible requests with
actionable guidance rather than a wasted git call.

The module is import-free of ``agent`` (the ``agent_core`` namespace rule);
``Agent._nlp_merge`` is a thin adapter over :func:`run_merge`.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

#: Hard cap on the returned text so one call can never flood the context.
MAX_OUTPUT = 5000

#: Default wall-clock budget for a single git invocation.  A merge of a large
#: history can legitimately take a while, but nothing here may ever block.
DEFAULT_TIMEOUT = 60.0

#: Timeout for the cheap state probes (``MERGE_HEAD``, unmerged paths).
PROBE_TIMEOUT = 15.0

#: Merge strategies git actually accepts (``git merge -s <name>``).  Checked
#: here so a typo is answered with the valid list instead of git's
#: ``Could not find merge strategy 'nope'``.
KNOWN_STRATEGIES: tuple[str, ...] = (
    "ort", "recursive", "resolve", "octopus", "subtree", "ours",
)

#: Actions the tool understands, in the order they are advertised.
VALID_ACTIONS: tuple[str, ...] = ("status", "start", "continue", "abort", "quit")

#: Accepted ``ff`` values -> git flag (``auto`` adds nothing).
_FF_FLAGS: dict[str, str | None] = {
    "auto": None,
    "only": "--ff-only",
    "no": "--no-ff",
}

#: Aliases accepted for the ``action`` argument.
_ACTION_ALIASES: dict[str, str] = {
    "": "status",
    "status": "status",
    "state": "status",
    "start": "start",
    "merge": "start",
    "continue": "continue",
    "resume": "continue",
    "abort": "abort",
    "quit": "quit",
}

#: Editor values that would block a headless subprocess forever.
_INTERACTIVE_EDITORS = frozenset({"", "vi", "vim", "nano", "emacs", "notepad", "edit"})

_TRUNCATION_MARK = "\n... [truncated] ...\n"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def normalize_action(raw: object) -> str | None:
    """Map a model-supplied ``action`` to a canonical action name.

    Returns ``None`` for an unrecognised value so the caller can answer with
    the valid list instead of guessing.
    """
    key = "" if raw is None else str(raw).strip().lower()
    return _ACTION_ALIASES.get(key)


def merge_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment that guarantees git never opens an editor or a prompt.

    ``base`` defaults to a copy of the current environment; it is never
    mutated, so callers can pass their own mapping safely.
    """
    env = dict(os.environ if base is None else base)
    # A no-op command: git "opens" the editor, which exits 0 immediately.
    env["GIT_EDITOR"] = "true"
    env["GIT_MERGE_AUTOEDIT"] = "no"
    # Credential/username prompts must fail fast rather than block.
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def parse_conflicted_files(output: str | None) -> list[str]:
    """Parse ``git diff --name-only --diff-filter=U`` output into paths."""
    if not output:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def build_merge_argv(
    action: str,
    branch: str | None = None,
    strategy: str | None = None,
    strategy_option: str | None = None,
    ff: str = "auto",
    message: str | None = None,
    squash: bool = False,
    no_commit: bool = False,
    allow_unrelated: bool = False,
) -> list[str]:
    """Build the exact argv for *action* — the consistency contract.

    ``continue``/``abort``/``quit`` take **no** arguments (git exits 129 for
    ``git merge --continue --no-edit``), and every start is forced
    non-interactive with ``--no-edit`` so no editor can ever open.

    Raises :class:`ValueError` for an unknown action, a branch/strategy that
    looks like a flag, or an unknown strategy/``ff`` value — a model-supplied
    string must never be reinterpreted as an option.
    """
    if action in ("continue", "abort", "quit"):
        return ["git", "merge", f"--{action}"]
    if action != "start":
        raise ValueError(
            f"Unsupported merge action: {action!r}. "
            f"Valid actions: {', '.join(VALID_ACTIONS)}"
        )

    argv = ["git", "merge"]
    # `-m` already supplies the message; git rejects it together with
    # --no-edit, so the flag is added only when no message was given.
    if message:
        argv += ["-m", str(message)]
    else:
        argv.append("--no-edit")

    if strategy:
        name = str(strategy).strip()
        if name.startswith("-"):
            raise ValueError(f"Invalid merge strategy: {name!r}")
        if name not in KNOWN_STRATEGIES:
            raise ValueError(
                f"Unknown merge strategy: {name!r}. "
                f"Available: {', '.join(KNOWN_STRATEGIES)}."
            )
        argv += ["-s", name]

    if strategy_option:
        opt = str(strategy_option).strip()
        if opt.startswith("-"):
            raise ValueError(f"Invalid strategy option: {opt!r}")
        argv += ["-X", opt]

    ff_flag = _FF_FLAGS.get(str(ff or "auto").strip().lower())
    if ff_flag:
        argv.append(ff_flag)

    if squash:
        argv.append("--squash")
    if no_commit:
        argv.append("--no-commit")
    if allow_unrelated:
        argv.append("--allow-unrelated-histories")

    target = str(branch or "").strip()
    if not target:
        raise ValueError(
            "merge(action='start') needs a 'branch' — the branch or commit to "
            "merge into the current one."
        )
    if target.startswith("-"):
        raise ValueError(f"Invalid branch name: {target!r}")
    argv.append(target)
    return argv


# ---------------------------------------------------------------------------
# Git plumbing
# ---------------------------------------------------------------------------

def _bounded(text: str, limit: int = MAX_OUTPUT) -> str:
    """Cap *text* at *limit* characters, keeping the head and the tail."""
    if len(text) <= limit:
        return text
    keep_tail = 140
    head = max(0, limit - len(_TRUNCATION_MARK) - keep_tail)
    return text[:head] + _TRUNCATION_MARK + text[-keep_tail:]


def _run_git(
    repo: str | Path, argv: list[str], timeout: float,
) -> tuple[int, str, str]:
    """Run *argv* in *repo* non-interactively, never blocking.

    ``stdin`` is closed and the environment disables every editor/prompt, so
    the only way this returns late is the timeout — which is reported as a
    message rather than an exception.
    """
    try:
        proc = subprocess.run(
            argv,
            cwd=str(repo),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=merge_environment(),
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", f"git timed out after {timeout:g}s"
    except OSError as e:  # git missing, cwd gone, ...
        return -1, "", f"git could not be started: {e}"


def merge_in_progress(repo: str | Path) -> bool:
    """True when ``MERGE_HEAD`` exists, i.e. a merge is mid-flight."""
    rc, _out, _err = _run_git(
        repo, ["git", "rev-parse", "-q", "--verify", "MERGE_HEAD"], PROBE_TIMEOUT,
    )
    return rc == 0


def conflicted_files(repo: str | Path) -> list[str]:
    """Paths still carrying unresolved conflict markers."""
    _rc, out, _err = _run_git(
        repo, ["git", "diff", "--name-only", "--diff-filter=U"], PROBE_TIMEOUT,
    )
    return parse_conflicted_files(out)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def _invocation_line(argv: list[str]) -> str:
    return "Invocation: " + " ".join(argv)


def _state_lines(repo: str | Path) -> list[str]:
    """``Merge in progress`` plus the conflict list, shared by every report."""
    if not merge_in_progress(repo):
        return ["Merge in progress: no"]
    conflicts = conflicted_files(repo)
    lines = ["Merge in progress: yes"]
    if conflicts:
        lines.append(f"Unresolved conflicts ({len(conflicts)}):")
        lines += [f"  {path}" for path in conflicts]
    else:
        lines.append("All conflicts are resolved — commit them with "
                     'merge(action="continue").')
    return lines


def _git_body(rc: int, out: str, err: str) -> str:
    body = "\n".join(part for part in (out.strip(), err.strip()) if part)
    return body or "(no git output)"


def _report(repo: str | Path, argv: list[str], rc: int, out: str, err: str) -> str:
    text = "\n".join([
        _invocation_line(argv),
        f"Exit code: {rc}",
        *_state_lines(repo),
        _git_body(rc, out, err),
    ])
    return _bounded(text)


def _guidance(argv: list[str], note: str, body: str = "(git not invoked)") -> str:
    return _bounded("\n".join([_invocation_line(argv), note, body]))


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _do_status(repo: str | Path) -> str:
    if not merge_in_progress(repo):
        return (
            "Merge in progress: no\n"
            "No merge is in progress; nothing to continue or abort."
        )
    lines = _state_lines(repo)
    conflicts = conflicted_files(repo)
    if conflicts:
        lines.append(
            'Next: resolve each conflict, stage with git(add, "-A"), then call '
            'merge(action="continue"). To back out: merge(action="abort").'
        )
    return _bounded("\n".join(lines))


def _do_start(repo: str | Path, args: dict[str, Any], timeout: float) -> str:
    branch = str(args.get("branch") or "").strip()
    if not branch:
        return _guidance(
            ["git", "merge"],
            "merge(action=\"start\") needs a 'branch' — the branch or commit to "
            "merge into the current one. Use merge(action=\"status\") to inspect "
            "the current state first.",
        )
    if merge_in_progress(repo):
        # A second merge would answer `fatal: Merging is not possible because
        # you have unmerged files` — a wasted call.  Answer the real question.
        return _bounded("\n".join([
            "A merge is already in progress — finish or cancel it first.",
            *_state_lines(repo),
            'Next: resolve the conflicts and call merge(action="continue"), or '
            'back out with merge(action="abort").',
        ]))

    try:
        argv = build_merge_argv(
            "start",
            branch=branch,
            strategy=args.get("strategy"),
            strategy_option=args.get("strategy_option"),
            ff=str(args.get("ff") or "auto"),
            message=args.get("message"),
            squash=bool(args.get("squash")),
            no_commit=bool(args.get("no_commit")),
            allow_unrelated=bool(args.get("allow_unrelated")),
        )
    except ValueError as e:
        return _guidance(["git", "merge"], str(e))

    rc, out, err = _run_git(repo, argv, timeout)
    text = _report(repo, argv, rc, out, err)

    if bool(args.get("squash")):
        text += ("\nSquash: changes are staged but not committed — commit them "
                 "yourself (the squash leaves no MERGE_HEAD).")
    elif bool(args.get("no_commit")):
        text += ("\n--no-commit: the merge result is staged but not committed — "
                 "commit it yourself when ready.")
    if conflicted_files(repo):
        text += ("\nNext: resolve each conflict, stage with git(add, \"-A\"), "
                 'then call merge(action="continue"). To back out: '
                 'merge(action="abort").')
    return _bounded(text)


def _do_continue(repo: str | Path, timeout: float) -> str:
    # `--continue` takes NO arguments — passing one exits 129, so the bare
    # form is the only correct invocation.
    argv = ["git", "merge", "--continue"]
    if not merge_in_progress(repo):
        return _guidance(
            argv,
            "No merge in progress — there is nothing to continue.",
        )
    conflicts = conflicted_files(repo)
    if conflicts:
        return _bounded("\n".join([
            _invocation_line(argv),
            f"Cannot continue: {len(conflicts)} unresolved conflict(s).",
            *[f"  {path}" for path in conflicts],
            "Resolve each file, stage it with git(add, \"-A\"), then call "
            "merge(action=\"continue\") again.",
        ]))
    rc, out, err = _run_git(repo, argv, timeout)
    return _report(repo, argv, rc, out, err)


def _do_abort(repo: str | Path, timeout: float) -> str:
    argv = ["git", "merge", "--abort"]
    if not merge_in_progress(repo):
        return _guidance(
            argv, "No merge in progress — there is nothing to abort.",
        )
    rc, out, err = _run_git(repo, argv, timeout)
    return _report(repo, argv, rc, out, err)


def _do_quit(repo: str | Path, timeout: float) -> str:
    argv = ["git", "merge", "--quit"]
    rc, out, err = _run_git(repo, argv, timeout)
    return _report(repo, argv, rc, out, err)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_merge(
    repo: str | Path,
    args: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Execute one merge request against *repo* and return a report.

    Never raises for a bad request: an unusable ``action`` is answered with
    the valid list, and every git invocation is non-interactive and bounded.
    """
    args = args or {}
    action = normalize_action(args.get("action"))
    if action is None:
        raw = args.get("action")
        return _guidance(
            ["git", "merge"],
            f"Unknown merge action: {raw!r}. "
            f"Valid actions: {', '.join(VALID_ACTIONS)}.",
        )
    if action == "status":
        return _do_status(repo)
    if action == "continue":
        return _do_continue(repo, timeout)
    if action == "abort":
        return _do_abort(repo, timeout)
    if action == "quit":
        return _do_quit(repo, timeout)
    return _do_start(repo, args, timeout)
