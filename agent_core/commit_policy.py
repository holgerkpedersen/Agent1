"""Commit-message policy: the single source of truth for a usable subject.

Live run (2026-09-28) produced HEAD ``9db6ca2 "commit changes"`` -- a subject
with literally zero information, because "commit changes" was handed straight
to ``git commit -m``.  ``CONTRIBUTING.md`` documents the real format
(type-prefixed, imperative) and nothing enforced it, so the convention rotted
silently: 150 of 612 subjects in history carry no type prefix at all.

Two severity tiers, deliberately:

* **errors** -- only for subjects that carry no information (empty, a bare
  filename, "wip"/"commit changes").  These BLOCK the commit, because letting
  them in is what produced the mess this module exists to stop.
* **warnings** -- format deviations (no type prefix, unknown type, trailing
  period, over 72 columns).  Advisory only; an author is never locked out of
  committing over a period.

Stdlib only and **no intra-package imports on purpose**: ``.githooks/commit-msg``
loads this file by path so a git hook never drags in the agent package (whose
``__init__`` pulls entities, file_system and the rest of the stack).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Types CONTRIBUTING.md documents.  Advisory: an unknown type warns, never fails.
ALLOWED_TYPES: frozenset[str] = frozenset(
    {"feat", "fix", "refactor", "docs", "test", "chore"}
)

#: Soft limit (git log / GitHub truncate around here) and the hard limit that
#: turns a long subject into an error.
SOFT_MAX_LEN = 72
HARD_MAX_LEN = 120

#: A type prefix (``fix:``, ``feat(jev)!:``) is evidence of intent, so anything
#: carrying one is exempt from the vacuity check.  Case-insensitive: a
#: capitalised type gets a "lowercase it" warning, not a vacuity error.
_SUBJECT_RE = re.compile(r"^([A-Za-z]+)(\([^)]*\))?!?:\s*")

#: Words that on their own say nothing.  A subject made ONLY of these is
#: vacuous ("commit changes", "minor fixes", "various improvements").
_FILLER_WORDS: frozenset[str] = frozenset(
    {
        "a", "add", "added", "address", "addresses", "again", "asdf", "bug",
        "bugs", "change", "changed", "changes", "cleanup", "code", "commit",
        "committed", "done", "edit", "edits", "file", "files", "final",
        "fix", "fixed", "fixes", "improve", "improvement", "improvements",
        "js", "minor", "misc", "more", "next", "ok", "okay", "patch", "py",
        "refactor", "resolve", "resolved", "small", "some", "source",
        "sources", "stuff", "sundry", "sync", "temp", "test", "tests", "thing",
        "things", "tmp", "tweak", "tweaks", "untitled", "update", "updated",
        "updates", "various", "version", "whatever", "wip", "work",
    }
)

#: Suffixes that make a whole-subject string a bare filename reference.
_FILE_SUFFIXES: frozenset[str] = frozenset(
    {
        ".cfg", ".cmd", ".ini", ".json", ".md", ".py", ".ps1", ".sh", ".toml",
        ".txt", ".yaml", ".yml",
    }
)


def _normalize(subject: str) -> str:
    """Lowercase, strip, and drop trailing sentence punctuation."""
    return subject.strip().strip(".").strip().casefold()


def is_placeholder_subject(subject: str) -> bool:
    """True when *subject* carries no information worth putting in history.

    Conservative by design: a subject with a type prefix is never a
    placeholder, and anything mentioning a concrete symbol, path fragment or
    domain noun alongside the filler words is kept.
    """
    norm = _normalize(subject)
    if not norm:
        return True
    if _SUBJECT_RE.match(subject.strip()):
        return False
    # A bare filename / path ("agent.py", "agent_core/jev_engine.py").
    if norm.endswith(tuple(_FILE_SUFFIXES)):
        return True
    words = re.split(r"[^a-z0-9+]+", norm)
    words = [w for w in words if w]
    if not words:
        return True
    # Every word is generic filler -> nothing was said.
    return all(w in _FILLER_WORDS for w in words)


@dataclass
class SubjectResult:
    """Errors block the commit; warnings are advisory."""

    subject: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_subject(subject: str) -> SubjectResult:
    """Check a single-line commit subject (the first line of the message)."""
    result = SubjectResult(subject=subject)
    text = subject.strip()
    if not text:
        result.errors.append("commit message subject is empty")
        return result

    if is_placeholder_subject(text):
        result.errors.append(
            f"vacuous commit subject {text!r} -- say WHAT changed and WHY, "
            "e.g. 'fix(jev): correct the module docstring punctuation'"
        )

    if len(text) > HARD_MAX_LEN:
        result.errors.append(
            f"subject is {len(text)} characters -- too long; keep it under "
            f"{HARD_MAX_LEN} and put detail in the body"
        )
    elif len(text) > SOFT_MAX_LEN:
        result.warnings.append(
            f"subject is {len(text)} characters; prefer {SOFT_MAX_LEN} or fewer"
        )

    match = _SUBJECT_RE.match(text)
    if match is None:
        result.warnings.append(
            "no type prefix -- use one of "
            + "/".join(sorted(ALLOWED_TYPES))
            + ", e.g. 'fix(commit): block vacuous subjects'"
        )
    else:
        raw_type = match.group(1)
        if raw_type != raw_type.lower():
            result.warnings.append(
                f"type prefix {raw_type!r} should be lowercase "
                f"({raw_type.lower()})"
            )
        elif raw_type not in ALLOWED_TYPES:
            result.warnings.append(
                f"unknown type {raw_type!r} -- CONTRIBUTING.md documents "
                + "/".join(sorted(ALLOWED_TYPES))
            )

    if text.endswith("."):
        result.warnings.append("subject ends with a period -- drop it")
    return result


@dataclass
class MessageReport:
    """Result of checking a whole commit message file."""

    subject: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        """Human-readable block for the hook, or "" when everything is clean."""
        lines: list[str] = []
        for err in self.errors:
            lines.append(f"  ERROR  {err}")
        for warn in self.warnings:
            lines.append(f"  warn   {warn}")
        if not lines:
            return ""
        return "\n".join(
            [
                "",
                "commit-msg: commit message rejected by agent_core.commit_policy",
                *(f"  subject: {self.subject!r}" if self.subject else []),
                *lines,
                "",
                "  Format: <type>(<scope>): <imperative summary>",
                "  Types:  " + "/".join(sorted(ALLOWED_TYPES)),
                "  Errors block the commit; warnings are advisory.",
                "",
            ]
        )


def check_message(message: str) -> MessageReport:
    """Check a full commit message: first non-blank line is the subject.

    Tolerates the CRLF line endings Windows editors write and the leading blank
    line a cleanup strip can leave behind.
    """
    text = message.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    subject = next((line for line in lines if line.strip()), "")
    result = validate_subject(subject)
    return MessageReport(
        subject=subject.strip(),
        errors=list(result.errors),
        warnings=list(result.warnings),
    )
