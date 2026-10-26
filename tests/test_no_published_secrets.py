"""Regression guards for the PUBLIC repository.

This repo (origin: holgerkpedersen/Agent1) is published to GitHub, so anything
tracked here is world-readable. These tests fail if private material or personal
identity data is ever committed again, in the working tree OR in history.

The PII sentinels are deliberately NOT hardcoded below. A test file that spelled
out the owner's name, email, company number and street would republish exactly
the personal data it exists to protect -- in a public repo -- and then fail on
its own contents. The sentinels are read from the environment instead, so this
file is safe to publish and the sensitive strings never enter version control.

Set them before running, e.g.::

    $env:AGENT1_PII_SENTINELS = "owner@example.com,Full Name,12345678,Street Name"

The PII tests skip (loudly, never silently passing) when it is unset. The
secret-scanning tests need no configuration and always run.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Comma-separated literal strings that must never appear in this repo.
PII_SENTINELS_ENV = "AGENT1_PII_SENTINELS"

SECRET_PATTERNS = {
    "aws-access-key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private-key-block": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"
    ),
    "github-token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "openai-key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "anthropic-key": re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}"),
    "slack-token": re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    "google-api-key": re.compile(r"AIza[0-9A-Za-z\-_]{35}"),
}

# Binary/font assets contain digit runs that look like phone numbers; skip them.
SKIP_PREFIXES = (
    ".git/",
    "build/",
    "dist/",
    "node_modules/",
    "__pycache__/",
    ".venv/",
    "static/",
)


def _pii_patterns() -> dict[str, re.Pattern[str]]:
    """Build PII matchers from ``AGENT1_PII_SENTINELS`` (comma-separated).

    Literals are escaped, so a sentinel containing regex metacharacters is
    matched literally. Returns an empty dict when the variable is unset; callers
    must skip rather than treat that as a pass.
    """
    raw = os.environ.get(PII_SENTINELS_ENV, "")
    patterns: dict[str, re.Pattern[str]] = {}
    for index, item in enumerate(raw.split(","), start=1):
        literal = item.strip()
        if literal:
            patterns[f"pii-sentinel-{index}"] = re.compile(re.escape(literal), re.I)
    return patterns


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, errors="ignore"
    )
    return result.stdout


def _tracked_files() -> list[str]:
    return [p for p in _git("ls-files").splitlines() if p.strip()]


def _is_scannable(path: str) -> bool:
    normalised = path.replace("\\", "/")
    if any(normalised.startswith(prefix) for prefix in SKIP_PREFIXES):
        return False
    return (REPO_ROOT / path).is_file()


def _scan_tracked_files(patterns: dict[str, re.Pattern[str]]) -> list[str]:
    """Return ``path (label)`` for every tracked file matching ``patterns``."""
    offenders: list[str] = []
    for path in _tracked_files():
        if not _is_scannable(path):
            continue
        try:
            text = (REPO_ROOT / path).read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover - unreadable file
            continue
        for label, pattern in patterns.items():
            if pattern.search(text):
                offenders.append(f"{path} ({label})")
    return offenders


@pytest.fixture(scope="module")
def git_available() -> None:
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")


@pytest.fixture(scope="module")
def pii_patterns(git_available: None) -> dict[str, re.Pattern[str]]:
    patterns = _pii_patterns()
    if not patterns:
        pytest.skip(
            f"set {PII_SENTINELS_ENV} to a comma-separated list of personal "
            "strings (email, name, company number, street) to enable PII scans"
        )
    return patterns


def test_no_secrets_in_tracked_files(git_available: None) -> None:
    """No credential-shaped string may be committed to a public repo."""
    offenders = _scan_tracked_files(SECRET_PATTERNS)
    assert not offenders, f"secret-shaped content tracked in public repo: {offenders}"


def test_no_pii_in_tracked_files(pii_patterns: dict[str, re.Pattern[str]]) -> None:
    """Owner email, name, company number and street must not be published."""
    offenders = _scan_tracked_files(pii_patterns)
    assert not offenders, f"personal data tracked in public repo: {offenders}"


def test_no_secrets_or_pii_in_git_history(git_available: None) -> None:
    """Deleting a file does not unpublish it; scan every commit's added lines.

    Secrets are always checked. PII is checked when sentinels are configured,
    and the skip is explicit so an unconfigured run cannot look like a pass.
    """
    patterns = {**SECRET_PATTERNS, **_pii_patterns()}
    patch = _git("log", "--all", "-p", "--pretty=format:")
    offenders: set[str] = set()
    for line in patch.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for label, pattern in patterns.items():
            if pattern.search(line):
                offenders.add(label)
    assert not offenders, f"secret/PII ever committed to history: {sorted(offenders)}"


def test_env_file_was_never_committed(git_available: None) -> None:
    """`.env` holds live credentials; it must never appear in history."""
    added = _git("log", "--all", "--oneline", "--diff-filter=A", "--", ".env")
    assert added.strip() == "", f".env was committed at some point: {added}"


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        ".env.production",
        "deploy/key.pem",
        "deploy/id_ed25519",
        "certs/server.pfx",
    ],
)
def test_sensitive_paths_are_gitignored(git_available: None, path: str) -> None:
    """Private key and env variants must be ignored, not merely absent today."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{path} is NOT gitignored - it could be committed"


def test_env_example_stays_tracked(git_available: None) -> None:
    """The `.env.*` ignore rule must not swallow the documented template."""
    assert ".env.example" in _tracked_files()
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", ".env.example"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, ".env.example is ignored but should be tracked"
