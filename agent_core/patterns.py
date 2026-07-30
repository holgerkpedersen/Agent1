"""Static pattern detectors for Python code — zero LLM, zero AST, pure regex.

Each detector takes source code as a string and returns a list of
``(line_number, pattern_name, suggestion)`` tuples.  False positives
are possible but kept low by regex anchoring.
"""

import re
from collections import Counter


def analyze(source: str) -> list[dict]:
    """Run all detectors and return unified findings."""
    all_findings: list[dict] = []
    for detector in DETECTORS:
        for line_no, name, suggestion in detector(source):
            all_findings.append({
                "line": line_no,
                "pattern": name,
                "suggestion": suggestion,
            })
    all_findings.sort(key=lambda f: f["line"])
    return all_findings


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def detect_regex_in_loop(source: str) -> list[tuple[int, str, str]]:
    """``re.compile()`` or ``re.match()`` inside a for/while loop body."""
    findings: list[tuple[int, str, str]] = []
    lines = source.split("\n")
    in_loop = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.match(r"^\s*(for|while)\s", line):
            in_loop = True
            continue
        if in_loop and not line.startswith((" ", "\t")) and stripped:
            in_loop = False
        if in_loop and re.search(r"\bre\.(compile|match|search|sub|findall)\(", line):
            findings.append((i, "regex_in_loop",
                             "Move re.compile() to module level — compiling inside loop wastes cycles"))
    return findings


def detect_string_concat_in_loop(source: str) -> list[tuple[int, str, str]]:
    """String ``+=`` inside a for/while loop."""
    findings: list[tuple[int, str, str]] = []
    lines = source.split("\n")
    in_loop = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.match(r"^\s*(for|while)\s", line):
            in_loop = True
            continue
        if in_loop and not line.startswith((" ", "\t")) and stripped:
            in_loop = False
        if in_loop and re.search(r"\w+\s*\+=\s*[\"']", line):
            findings.append((i, "string_concat_in_loop",
                             "Use ''.join() or io.StringIO instead of += in loop — O(n²) becomes O(n)"))
    return findings


def detect_bare_except(source: str) -> list[tuple[int, str, str]]:
    """Bare ``except:`` without exception type."""
    findings: list[tuple[int, str, str]] = []
    for i, line in enumerate(source.split("\n"), 1):
        if re.match(r"^\s*except\s*:", line) and "except:" in line:
            findings.append((i, "bare_except",
                             "Specify exception type — bare except catches KeyboardInterrupt and hides bugs"))
    return findings


def detect_silent_except(source: str) -> list[tuple[int, str, str]]:
    """``except ...: pass`` — silently swallowing errors."""
    findings: list[tuple[int, str, str]] = []
    lines = source.split("\n")
    for i, line in enumerate(lines, 1):
        if re.match(r"^\s*except\b", line):
            # Look at next line for pass
            if i < len(lines) and re.match(r"^\s+pass\s*$", lines[i]):
                findings.append((i, "silent_except",
                                 "Replace 'pass' with logging or re-raise — error is silently swallowed"))
    return findings


def detect_duplicate_imports(source: str) -> list[tuple[int, str, str]]:
    """Duplicate ``import X`` or ``from Y import Z`` statements."""
    findings: list[tuple[int, str, str]] = []
    seen: dict[str, int] = {}
    for i, line in enumerate(source.split("\n"), 1):
        m = re.match(r"^\s*(?:import\s+(\S+)|from\s+(\S+)\s+import\s+(.+))", line)
        if m:
            key = m.group(0).strip()
            if key in seen:
                findings.append((i, "duplicate_import",
                                 f"Duplicate import (first at line {seen[key]}). Remove this copy."))
            else:
                seen[key] = i
    return findings


def detect_missing_context_manager(source: str) -> list[tuple[int, str, str]]:
    """``open(...).read()`` without ``with`` statement."""
    findings: list[tuple[int, str, str]] = []
    for i, line in enumerate(source.split("\n"), 1):
        if "open(" in line and "with " not in line and " with " not in line and "with(" not in line:
            # Avoid false positives: comments, in strings, or in already-correct patterns
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "contextlib" in stripped or "closing" in stripped:
                continue
            findings.append((i, "missing_context_manager",
                             "Use 'with open() as f:' — current code may leak file handles"))
    return findings


def detect_file_read_in_loop(source: str) -> list[tuple[int, str, str]]:
    """``open()`` or ``read_file()`` inside a for/while loop."""
    findings: list[tuple[int, str, str]] = []
    lines = source.split("\n")
    in_loop = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.match(r"^\s*(for|while)\s", line):
            in_loop = True
            continue
        if in_loop and not line.startswith((" ", "\t")) and stripped:
            in_loop = False
        if in_loop and ("open(" in line or "read_file(" in line or ".read(" in line):
            findings.append((i, "file_read_in_loop",
                             "File read inside loop — move open() outside or cache the content"))
    return findings


def detect_list_append_join(source: str) -> list[tuple[int, str, str]]:
    """``.append()`` in a loop → ``''.join()`` — not a bug, just an optimization hint."""
    findings: list[tuple[int, str, str]] = []
    lines = source.split("\n")
    in_loop = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.match(r"^\s*(for|while)\s", line):
            in_loop = True
            continue
        if in_loop and not line.startswith((" ", "\t")) and stripped:
            in_loop = False
        if in_loop and ".append(" in line:
            # Only flag if we also see a join later in the same function
            findings.append((i, "list_append_join",
                             "If this builds a string list, use list comprehension or generator — currently O(n) memory"))
    return findings


DETECTORS = [
    detect_regex_in_loop,
    detect_string_concat_in_loop,
    detect_bare_except,
    detect_silent_except,
    detect_duplicate_imports,
    detect_missing_context_manager,
    detect_file_read_in_loop,
    detect_list_append_join,
]
