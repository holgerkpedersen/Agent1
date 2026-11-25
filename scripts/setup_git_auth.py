"""Configure repo-scope git credential helpers so push/ls-remote never prompt.

A fresh clone does NOT inherit this: ``.git/config`` is machine-local, so the
helper ORDER must be re-applied once per clone. This script does that
idempotently.

Run once per clone::

    python scripts/setup_git_auth.py
    python scripts/setup_git_auth.py --check   # verify only; exit 1 if wrong

Why order matters
-----------------
``credential.helper`` is an ORDERED list and git stops at the first helper that
yields a credential. If the ``.env`` token helper runs before Git Credential
Manager, a dead token is offered and every push fails with
``401 Bad credentials`` even though a working credential sits in the OS store.
An EMPTY entry (``credential.helper =``) silently clears every earlier helper,
which is how the ordering got broken originally, so :func:`apply` removes the
whole list before re-adding it in the correct order.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

_HELPER_REL = Path("scripts") / "git_credential_helper.cmd"
_CREDENTIAL_HELPER_KEY = "credential.helper"
_USERNAME_KEY = "credential.https://github.com.username"
_INTERACTIVE_KEY = "credential.interactive"

#: Git Credential Manager: owns the working credential in the OS store.
GCM_HELPER = "manager"
#: Pinned so GCM does not stop to ask which of several stored accounts to use.
DEFAULT_USERNAME = "holgerkpedersen"

GitRunner = Callable[[Sequence[str], Path], "subprocess.CompletedProcess[str]"]


@dataclass(frozen=True)
class Entry:
    """One ``git config`` assignment.

    ``append`` marks the multi-valued ``credential.helper`` key, which must be
    appended to preserve order rather than replaced.
    """

    key: str
    value: str
    append: bool = False


def _git(
    args: Sequence[str], repo_root: Path
) -> subprocess.CompletedProcess[str]:
    """Run ``git`` in *repo_root* and capture its output."""
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )


def build_entries(
    repo_root: Path,
    username: str = DEFAULT_USERNAME,
    helper_path: Path | None = None,
) -> list[Entry]:
    """Return the config entries to apply, in the order they must be written.

    Order is load-bearing: Git Credential Manager (``manager``) MUST come before
    the ``.env`` token helper, so a working OS-store credential always wins.
    """
    helper = Path(helper_path) if helper_path else repo_root / _HELPER_REL
    # git runs ``!`` helpers itself, so use an absolute path with forward
    # slashes: unambiguous regardless of the directory git invokes it from.
    helper_cmd = "!" + helper.resolve().as_posix()
    return [
        Entry(_CREDENTIAL_HELPER_KEY, GCM_HELPER, append=True),
        Entry(_CREDENTIAL_HELPER_KEY, helper_cmd, append=True),
        Entry(_USERNAME_KEY, username),
        Entry(_INTERACTIVE_KEY, "never"),
    ]


def apply(
    repo_root: Path = REPO_ROOT,
    username: str = DEFAULT_USERNAME,
    helper_path: Path | None = None,
    git_fn: GitRunner = _git,
) -> list[Entry]:
    """Apply the credential config to *repo_root*'s local git config.

    Idempotent: the helper list is cleared first, so re-running cannot
    accumulate duplicates or leave a stale empty entry that would clear
    everything after it.
    """
    entries = build_entries(repo_root, username, helper_path)
    # Fails with exit 5 when unset -- nothing to clear is fine, so ignore it.
    git_fn(["config", "--local", "--unset-all", _CREDENTIAL_HELPER_KEY], repo_root)
    for entry in entries:
        flag = "--add" if entry.append else "--replace-all"
        git_fn(
            ["config", "--local", flag, entry.key, entry.value], repo_root
        )
    return entries


def read_all(
    repo_root: Path, key: str, git_fn: GitRunner = _git
) -> list[str]:
    """Read every value of *key*; empty list when the key is absent."""
    result = git_fn(["config", "--local", "--get-all", key], repo_root)
    if result.returncode != 0:
        return []
    return result.stdout.splitlines()


def is_configured(
    repo_root: Path = REPO_ROOT,
    username: str = DEFAULT_USERNAME,
    helper_path: Path | None = None,
    git_fn: GitRunner = _git,
) -> bool:
    """True when *repo_root* already matches the required configuration."""
    expected = build_entries(repo_root, username, helper_path)
    for entry in expected:
        if entry.append:
            continue  # checked as part of the ordered list below
        actual = read_all(repo_root, entry.key, git_fn)
        if actual != [entry.value]:
            return False
    expected_helpers = [e.value for e in expected if e.append]
    return read_all(repo_root, _CREDENTIAL_HELPER_KEY, git_fn) == expected_helpers


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="setup_git_auth",
        description=(
            "Configure repo-scope git credential helpers so pushes never prompt."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify only; do not change anything (exit 1 when wrong)",
    )
    parser.add_argument(
        "--username",
        default=DEFAULT_USERNAME,
        help=f"pin this account for github.com (default: {DEFAULT_USERNAME})",
    )
    args = parser.parse_args(argv)

    if not (REPO_ROOT / _HELPER_REL).exists():
        print(f"error: helper missing: {REPO_ROOT / _HELPER_REL}", file=sys.stderr)
        return 2

    if args.check:
        if is_configured(REPO_ROOT, args.username):
            print("git auth: configured correctly")
            return 0
        print("git auth: MISCONFIGURED -- run: python scripts/setup_git_auth.py")
        for value in read_all(REPO_ROOT, _CREDENTIAL_HELPER_KEY):
            print(f"  credential.helper = {value}")
        return 1

    entries = apply(REPO_ROOT, args.username)
    print(f"git auth: wrote {len(entries)} entries to {REPO_ROOT / '.git' / 'config'}")
    for value in read_all(REPO_ROOT, _CREDENTIAL_HELPER_KEY):
        print(f"  credential.helper = {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
