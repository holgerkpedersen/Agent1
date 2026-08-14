"""ANSI color helpers for CLI/agent output.

Provides lightweight, dependency-free color wrappers so that tool names,
commands, results and status messages are nicer to read in the interactive
REPL.  Colors degrade gracefully (disabled) when stdout is not a TTY or when
the ``NO_COLOR`` env var / ``AGENT_NO_COLOR`` flag is set.
"""

from __future__ import annotations

import os
import sys

#: Whether color output should be emitted at all.
_ENABLED: bool = (
    sys.stdout.isatty()
    and not os.environ.get("NO_COLOR")
    and not os.environ.get("AGENT_NO_COLOR")
)


# ------------------------------------------------------------------
#  ANSI escape sequences
# ------------------------------------------------------------------
_RESET = "\033[0m"

_BOLD = "\033[1m"

#: Foreground palette used across the REPL.
_PALETTE: dict[str, str] = {
    "cyan": "\033[36m",   # tool/command names
    "green": "\033[32m",  # successful results / verify ✓
    "red": "\033[31m",    # errors, warnings, blocked commands
    "yellow": "\033[33m",  # status/info markers ([STDERR], hints)
    "blue": "\033[34m",   # info banners / headers
    "magenta": "\033[35m",  # special markers (auto-continue, stopped)
    "gray": "\033[90m",   # muted secondary text
}


def _wrap(text: str, color: str, bold: bool = False) -> str:
    """Wrap *text* in ANSI codes if color is enabled; otherwise return as-is."""
    if not _ENABLED or not text:
        return text
    prefix = (color + (_BOLD if bold else ""))
    # Reset only at the end so nested spans stay consistent.
    return f"{prefix}{text}{_RESET}"


def cyan(text: str, *, bold: bool = False) -> str:
    """Color *text* cyan — used for tool/command names."""
    return _wrap(text, _PALETTE["cyan"], bold=bold)


def green(text: str, *, bold: bool = False) -> str:
    """Color *text* green — successful results / verification ✓."""
    return _wrap(text, _PALETTE["green"], bold=bold)


def red(text: str, *, bold: bool = False) -> str:
    """Color *text* red — errors and blocked commands."""
    return _wrap(text, _PALETTE["red"], bold=bold)


def yellow(text: str, *, bold: bool = False) -> str:
    """Color *text* yellow — status/info markers ([STDERR], hints)."""
    return _wrap(text, _PALETTE["yellow"], bold=bold)


def blue(text: str, *, bold: bool = False) -> str:
    """Color *text* blue — banners / headers."""
    return _wrap(text, _PALETTE["blue"], bold=bold)


def magenta(text: str, *, bold: bool = False) -> str:
    """Color *text* magenta — special flow markers (auto-continue, stopped)."""
    return _wrap(text, _PALETTE["magenta"], bold=bold)


def gray(text: str, *, bold: bool = False) -> str:
    """Color *text* gray/muted — secondary text."""
    return _wrap(text, _PALETTE["gray"], bold=bold)


def colorize_result(result: str) -> str:
    """Heuristic color-coding of a tool result string.

    - Lines starting with ``Error`` / ``Search error`` / ``Write error`` etc.
      are colored red.
    - The ``[verify] py_compile ✓`` marker is green; the ✗ variant is red.
    - ``[STDERR]`` blocks and hints are yellow.
    - Success markers (``Successfully wrote``, ``Written ...``, ``Edited``)
      get a green prefix on their first line.

    Returns the result with ANSI codes applied to qualifying lines only, so
    multi-line command output (e.g. git diff / test runs) keeps its own shape.
    """
    if not _ENABLED or not result:
        return result
    out_lines: list[str] = []
    for line in result.splitlines():
        low = line.lower()

        # Verification marker — green ✓, red ✗.
        if "[verify]" in low and "✓" in line:
            out_lines.append(green(line))
            continue
        if "[verify]" in low and ("✗" in line or "could not run" in low):
            out_lines.append(red(line))
            continue

        # STDERR / hints — yellow.
        if line.startswith("[stderr]") or low.startswith("hint:") or "unix command hint" in low:
            out_lines.append(yellow(line))
            continue

        # Explicit error lines — red.
        if (
            low.startswith("error")
            or low.startswith("search error")
            or low.startswith("write error")
            or low.startswith("read error")
            or low.startswith("edit error")
            or low.startswith("analyze error")
            or low.startswith("diff error")
            or low.startswith("git error")
            or low.startswith("list error")
            or "error:" in low
        ):
            out_lines.append(red(line))
            continue

        # Blocked / dangerous command notice — red.
        if low.startswith("dangerous command blocked"):
            out_lines.append(red(line, bold=True))
            continue

        # Success markers — green prefix on the first line only.
        if (
            "successfully wrote" in low
            or "successfully edited" in low
            or low.startswith("written ")
            or low.startswith("edited ")
            or "patch applied successfully" in low
        ):
            out_lines.append(green(line))
            continue

        # No-match / skip notices — yellow (informational).
        if (
            "no files found matching that query" in low
            or "skipped" in low
            or "not a directory" in low
            or "(no differences found)" in line
            or "(no output)" in line
        ):
            out_lines.append(yellow(line))
            continue

        # Default — leave unchanged so raw command/test output is readable.
        out_lines.append(line)

    return "\n".join(out_lines)


__all__ = [
    "cyan",
    "green",
    "red",
    "yellow",
    "blue",
    "magenta",
    "gray",
    "colorize_result",
]
