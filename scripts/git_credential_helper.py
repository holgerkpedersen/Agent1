"""Git credential helper that authenticates GitHub from the repo's .env file.

Invoked by git as ``git_credential_helper.py <get|store|erase>`` with the
request attributes on stdin.  For ``get`` it emits the GITHUB_TOKEN from
``.env`` (never printed to logs) so pushes are non-interactive.

The token is read only from ``.env`` (gitignored) -- it is never written into
``.git/config`` or the remote URL.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_token() -> str:
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        try:
            text = env_path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("github_token="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    # Fall back to the process environment if .env is absent.
    return os.environ.get("GITHUB_TOKEN", "")


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else None
    if action != "get":
        return  # store/erase: nothing to do, let git continue.
    token = _read_token()
    if not token:
        return  # No token available: emit nothing, git may try the next helper.
    # GitHub accepts any username with a PAT; "git" is the conventional choice.
    sys.stdout.write("username=git\n")
    sys.stdout.write(f"password={token}\n")


if __name__ == "__main__":
    main()
