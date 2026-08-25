"""Text-content policy: detect emoji / pictographic symbols in repo files.

Implements decision #079: no emojis or pictographic symbols in repository
text files. Plain-text markers (`[DONE]`, `*(quick win)*`, ...) survive any
encoding; emoji do not (see the mojibake precedent in CHANGES.md).

Scope: TRUE emojis only — pictographs, dingbats, symbol blocks, and the
variation selector that forces emoji presentation. Monochrome CLI glyphs
used by this repo's terminal output and ASCII diagrams (check/cross/warning
marks, box drawing, flow arrows) are explicitly ALLOWED: they are
load-bearing output, not decoration.

Detection is stdlib-only: Unicode general-category So/Sk (symbol, other /
symbol, modifier) plus the dedicated emoji code-point ranges. Typography
(em dash, ≤, accented letters, CJK) is never flagged.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

# Dedicated emoji territory: Misc Symbols and Dingbats start at 0x2600,
# Supplemental Symbols and Pictographs through Symbols and Pictographs
# Extended-A cover the colored-emoji planes. (Glyphs are named, not shown,
# per decision #079.)
_EMOJI_RANGES: tuple[tuple[int, int], ...] = (
    (0x2600, 0x27BF),    # Misc Symbols + Dingbats
    (0x2B00, 0x2BFF),    # Misc Symbols and Arrows
    (0x1F000, 0x1FAFF),  # Mahjong tiles .. Symbols and Pictographs Ext-A
    (0xFE0F, 0xFE0F),    # Variation Selector-16 (forces emoji presentation)
)

_EMOJI_RE = re.compile(
    "[" + "".join(fr"\U0000{lo:04X}-\U0000{hi:04X}" for lo, hi in _EMOJI_RANGES) + "]"
)

# Monochrome glyphs this repo deliberately keeps: terminal status marks and
# box-drawing diagram lines (colors.py constants, verifiers, README trees).
# (Characters listed literally below; named here per decision #079.)
ALLOWED_MONO_CHARS: frozenset[str] = frozenset(
    "\u2713\u2717\u26a0\u00d7"                 # check, cross, warning, times
    "\u2500\u2502\u250c\u2510\u2514\u2518"     # box drawing: single line parts
    "\u251c\u2524\u252c\u2534\u253c"           # box drawing: tees and crosses
    "\u2550\u2551\u2554\u2557\u255a\u255d"     # box drawing: double line parts
    "\u2560\u2563\u2566\u2569\u256c"           # box drawing: double tees/cross
    "\u2190\u2192\u2191\u2193\u25ba\u25c4"     # arrows left/right/up/down
    "\u25b2\u25bc"                             # filled triangles up/down
    "\u2022\u00b7"                             # bullets
)
# NOTE: U+FFFD (replacement char) is deliberately NOT allowed — it marks
# encoding corruption and must be reported (decision #079 rationale).


DEFAULT_TEXT_EXTS: tuple[str, ...] = (
    ".py", ".md", ".json", ".txt", ".toml", ".cfg", ".ini", ".yml", ".yaml",
)

SKIP_DIRS: set[str] = {
    ".git", "__pycache__", ".docs", "backups", "reports",
    "node_modules", ".venv", "venv",
}

# Runtime state (gitignored): REPL transcripts and memory stores accumulate
# whatever the model printed — they are not authored repo content.
RUNTIME_STATE_FILES: frozenset[str] = frozenset(
    {"chat_history.json", "agent_memory.json"}
)


def is_emoji_char(ch: str) -> bool:
    """True for emoji / pictographic characters (never for typography/CJK).

    Monochrome CLI glyphs in ALLOWED_MONO_CHARS are excluded by design
    (decision #079: they are load-bearing terminal output, not decoration).
    """
    if ch in ALLOWED_MONO_CHARS:
        return False
    cp = ord(ch)
    if cp < 0x80:
        return False
    for lo, hi in _EMOJI_RANGES:
        if lo <= cp <= hi:
            return True
    return unicodedata.category(ch) in ("So", "Sk")


def find_emoji_chars(text: str) -> list[str]:
    """Sorted unique emoji characters present in *text*."""
    return sorted({ch for ch in text if is_emoji_char(ch)})


def scan_text(text: str) -> list[tuple[int, str]]:
    """(line_number, unique_emoji_chars) for every offending line."""
    rows: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.split("\n"), 1):
        found = find_emoji_chars(line)
        if found:
            rows.append((lineno, "".join(found)))
    return rows


def scan_file(path: str | Path) -> list[tuple[int, str]]:
    """Scan one file; unreadable/binary files yield no findings."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return scan_text(text)


def scan_tree(
    root: str | Path,
    *,
    extensions: tuple[str, ...] = DEFAULT_TEXT_EXTS,
    skip_dirs: set[str] = SKIP_DIRS,
) -> dict[str, list[tuple[int, str]]]:
    """Scan a directory tree; returns relative-path -> findings.

    Only **git-tracked** files are scanned when git is available — the policy
    (decision #079) governs repository content, not whatever untracked
    directories happen to sit inside the checkout (a stray virtualenv full of
    third-party license files otherwise produces bogus findings).  Falls back
    to a plain walk outside a repo or when git is missing.
    """
    findings: dict[str, list[tuple[int, str]]] = {}
    root_path = Path(root)
    tracked = _git_tracked_files(root_path)
    if tracked is not None:
        candidates = [
            root_path / rel for rel in sorted(tracked)
            if Path(rel).suffix.lower() in extensions
        ]
    else:
        candidates = [
            path for path in sorted(root_path.rglob("*"))
            if path.is_file() and path.suffix.lower() in extensions
        ]
        # keep the legacy dir-skip behaviour only for the fallback walk;
        # the tracked-file list already excludes ignored paths.
        candidates = [
            path for path in candidates
            if not any(part in skip_dirs for part in path.parts[:-1])
        ]
    for path in candidates:
        rel = path.relative_to(root_path).as_posix()
        if rel in RUNTIME_STATE_FILES:
            continue
        rows = scan_file(path)
        if rows:
            findings[rel] = rows
    return findings


def _git_tracked_files(root: Path) -> set[str] | None:
    """Relative posix paths of files git tracks under *root*.

    ``None`` when the directory is not a git worktree or git is unavailable —
    callers then fall back to a plain filesystem walk.
    """
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=str(root), capture_output=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return {
        item.replace("\\", "/")
        for item in proc.stdout.decode("utf-8", errors="replace").split("\0")
        if item
    }


def summarize_findings(
    findings: dict[str, list[tuple[int, str]]], max_files: int = 5
) -> str:
    """Compact one-line report for audit output."""
    parts: list[str] = []
    for path, rows in list(findings.items())[:max_files]:
        lines = ",".join(str(lineno) for lineno, _ in rows[:8])
        more = f"+{len(rows) - 8}" if len(rows) > 8 else ""
        parts.append(f"{path} L{lines}{more}")
    if len(findings) > max_files:
        parts.append(f"...and {len(findings) - max_files} more file(s)")
    return "; ".join(parts)
