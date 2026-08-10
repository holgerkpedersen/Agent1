"""Deterministic verification of code claims inside LLM-generated analysis text.

The workflow ``analyze`` step produces its output purely by LLM text
generation, which can hallucinate file paths, symbol names, and line numbers.
This module extracts concrete claims (backticked symbols, ``file.py``
references, ``line ~N`` references, and inline code snippets) from the text
and checks each one against the actual workspace, so unverifiable statements
are flagged instead of silently trusted. Claim text is never rewritten; the
analysis is preserved and a ``## Verification Report`` section is appended.
"""
import asyncio
import builtins
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_STDLIB_NAMES = frozenset(sys.stdlib_module_names) | frozenset(dir(builtins))

_EXTRA_MODULES = frozenset({
    "pydantic", "httpx", "numpy", "networkx", "openai", "pytest", "mypy",
    "ruff", "setuptools", "coverage", "tqdm", "yaml", "dotenv", "requests",
})

_SKIP_DIRS = frozenset({
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".coverage", "backups", "venv", ".venv", "node_modules", ".tox",
})

_BACKTICK_RE = re.compile(r"`([^`]+)`")
_FILE_RE = re.compile(r"(?<![\w/])((?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.py)(?![\w/])")
_LINE_RE = re.compile(r"\bline\s*(?:no\.?|#)?\s*(~|≈)?\s*(\d+)", re.IGNORECASE)

_ABSENCE_PHRASES = (
    "does not exist", "doesn't exist",
    "is not defined", "isn't defined", "not defined",
    "is not found", "isn't found", "not found anywhere",
    "is not called", "isn't called", "not called anywhere", "never called",
    "not called", "never used", "is not used", "isn't used",
    "unused", "unreferenced", "never referenced",
)

_ATTR_RE = re.compile(r"self\.([A-Za-z_][A-Za-z0-9_]*)\s*=")
_DEF_RE = re.compile(r"^\s*(?:async\s+)?(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)")
_IMPORT_FROM_RE = re.compile(r"^\s*from\s+[\w.]+\s+import\s+([\w,\s]+)")
_IMPORT_RE = re.compile(r"^\s*import\s+([\w.]+)")
_CONST_RE = re.compile(r"(_?[A-Z][A-Z0-9_]*)(?:\s*:\s*[^=]+)?\s*=")

_LINE_TOLERANCE_EXACT = 10
_LINE_TOLERANCE_APPROX = 30

_STATUS_OK = "ok"
_STATUS_FLAGGED = "flagged"
_STATUS_SKIP = "skip"


@dataclass
class VerificationResult:
    """Outcome of a verification pass over one analysis text."""

    text: str
    checked: int
    flagged: int


@dataclass
class _Claim:
    kind: str
    text: str
    offset: int
    file: str | None = None
    line: int | None = None
    approx: bool = False
    check_absence: bool = False
    status: str = _STATUS_SKIP
    reason: str = ""


def _list_files(ws_path: Path) -> list[str]:
    """Return workspace-relative file paths, skipping cache and venv dirs."""
    rel_files: list[str] = []
    if not ws_path.is_dir():
        return rel_files
    for root, dirs, files in os.walk(ws_path):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), ws_path).replace("\\", "/")
            rel_files.append(rel)
    return sorted(rel_files)


async def _read_files_async(ws_path: Path, rel_files: list[str]) -> dict[str, str]:
    """Read every workspace file in parallel; unreadable files are skipped."""

    def _read(rel: str) -> tuple[str, str | None]:
        try:
            content = (ws_path / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return rel, None
        return rel, content

    results = await asyncio.gather(*(asyncio.to_thread(_read, rel) for rel in rel_files))
    return {rel: content for rel, content in results if content is not None}


def _index_file(content: str) -> tuple[dict[str, int], set[str]]:
    """Index definitions, imports, module constants, and ``self`` attributes.

    Imports are indexed so claims like "agent.py uses `to_windows_path`" hold
    for names that are imported rather than defined in the file.
    """
    defs: dict[str, int] = {}
    attrs: set[str] = set()
    for lineno, line in enumerate(content.splitlines(), 1):
        m = _DEF_RE.match(line)
        if m:
            defs.setdefault(m.group(1), lineno)
            continue
        m = _IMPORT_FROM_RE.match(line)
        if m:
            for name in re.split(r"[,\s]+", m.group(1)):
                if name and name != "as":
                    defs.setdefault(name, lineno)
            continue
        m = _IMPORT_RE.match(line)
        if m:
            defs.setdefault(m.group(1).split(".", 1)[0], lineno)
            continue
        m = _CONST_RE.match(line)
        if m:
            defs.setdefault(m.group(1), lineno)
        attrs.update(_ATTR_RE.findall(line))
    return defs, attrs


def _mask_backticks(text: str) -> str:
    """Return *text* with backticked regions replaced by spaces (offsets kept)."""
    chars = list(text)
    for m in _BACKTICK_RE.finditer(text):
        for i in range(m.start(), m.end()):
            if chars[i] != "\n":
                chars[i] = " "
    return "".join(chars)


def _classify_backtick(content: str) -> str:
    if content.endswith((".py", ".md", ".toml", ".json", ".txt")) or content.startswith("."):
        return "file"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", content):
        return "symbol"
    return "snippet"


_SNIPPET_MARKERS = ("=", "(", ".", "[", "{", ":", "<")


def _extract_claims(segment: str, seg_start: int) -> list[_Claim]:
    """Extract file, symbol, snippet, and line claims from one source line."""
    claims: list[_Claim] = []

    # Backticked regions are masked so _FILE_RE only sees bare file refs
    # (backticked file paths become dedicated file claims below).
    masked = _mask_backticks(segment)
    for m in _FILE_RE.finditer(masked):
        rel = m.group(1).replace("\\", "/")
        offset = seg_start + m.start()
        claims.append(_Claim(kind="file", text=rel, offset=offset, file=rel))
        line_m = re.match(r":(\d+)", segment[m.end():])
        if line_m:
            claims.append(_Claim(
                kind="line",
                text=f"{rel}:{line_m.group(1)}",
                offset=offset + m.end() - m.start(),
                file=rel,
                line=int(line_m.group(1)),
            ))

    for m in _BACKTICK_RE.finditer(segment):
        content = m.group(1)
        kind = _classify_backtick(content)
        offset = seg_start + m.start()
        if kind == "file":
            rel = content.replace("\\", "/")
            claims.append(_Claim(kind="file", text=rel, offset=offset, file=rel))
        elif kind == "symbol":
            claims.append(_Claim(kind="symbol", text=content, offset=offset))
        else:
            claims.append(_Claim(kind="snippet", text=content, offset=offset))

    for m in _LINE_RE.finditer(segment):
        if not _is_inside_backticks(segment, m.start()):
            claims.append(_Claim(
                kind="line",
                text=f"line {'~' if m.group(1) else ''}{m.group(2)}",
                offset=seg_start + m.start(),
                line=int(m.group(2)),
                approx=bool(m.group(1)),
            ))

    for c in claims:
        if c.kind == "symbol":
            line_start = segment.rfind("\n", 0, c.offset - seg_start) + 1
            line_end = segment.find("\n", c.offset - seg_start)
            line_end = len(segment) if line_end == -1 else line_end
            line_text = segment[line_start:line_end].lower()
            c.check_absence = any(p in line_text for p in _ABSENCE_PHRASES)

    return claims


def _is_inside_backticks(segment: str, start: int) -> bool:
    return any(m.start() < start < m.end() for m in _BACKTICK_RE.finditer(segment))


def _choose_file(claims: list[_Claim], offset: int) -> str | None:
    """Pick the file context for a claim: nearest preceding file claim wins."""
    file_claims = [c for c in claims if c.kind == "file"]
    if not file_claims:
        return None
    before = [c for c in file_claims if c.offset <= offset]
    return (before[-1] if before else file_claims[0]).file


def _attach_context(claim: _Claim, claims: list[_Claim]) -> None:
    """Attach line-level file context to claims that lack one.

    Only ``.py`` files scope symbol/snippet/line checks; a ``.gitignore`` or
    markdown mention only verifies that the file itself exists.
    """
    if claim.file is None:
        candidate = _choose_file(claims, claim.offset)
        if candidate is not None and candidate.endswith(".py"):
            claim.file = candidate


def _verify_existence(file: str | None, rel_files: list[str]) -> tuple[str, str]:
    if file is None:
        return _STATUS_SKIP, "no file context"
    if file in rel_files:
        return _STATUS_OK, "file exists in workspace"
    return _STATUS_FLAGGED, "file not found in workspace"


def _symbol_exists_anywhere(
    name: str,
    defs_by_file: dict[str, dict[str, int]],
    attrs_by_file: dict[str, set[str]],
) -> bool:
    return any(name in defs for defs in defs_by_file.values()) or any(
        name in attrs for attrs in attrs_by_file.values()
    )


def _verify_symbol(
    name: str,
    file: str | None,
    defs_by_file: dict[str, dict[str, int]],
    attrs_by_file: dict[str, set[str]],
) -> tuple[str, str]:
    if name in _STDLIB_NAMES or name in _EXTRA_MODULES:
        return _STATUS_OK, "standard-library or third-party name"
    if file is not None:
        defs = defs_by_file.get(file, {})
        if name in defs:
            return _STATUS_OK, f"defined at {file}:{defs[name]}"
        if name in attrs_by_file.get(file, set()):
            return _STATUS_OK, f"attribute on self in {file}"
        if re.fullmatch(r"[a-z][a-z0-9_]*", name) and not _symbol_exists_anywhere(
            name, defs_by_file, attrs_by_file
        ):
            return _STATUS_SKIP, "prose-like identifier (not found anywhere)"
        return _STATUS_FLAGGED, f"symbol not found in {file}"
    hits = []
    for f, defs in defs_by_file.items():
        for name_in, ln in defs.items():
            if name_in == name:
                hits.append((f, ln))
    if hits:
        f, ln = sorted(hits)[0]
        return _STATUS_OK, f"defined at {f}:{ln}"
    attr_files = sorted(f for f, attrs in attrs_by_file.items() if name in attrs)
    if attr_files:
        return _STATUS_OK, f"attribute on self in {attr_files[0]}"
    if re.fullmatch(r"[a-z][a-z0-9_]*", name):
        return _STATUS_SKIP, "prose-like identifier (not found anywhere)"
    return _STATUS_FLAGGED, "symbol not found anywhere in workspace"


def _verify_line(
    line_no: int,
    file: str | None,
    line_counts: dict[str, int],
    rel_files: list[str],
) -> tuple[str, str]:
    if file is None:
        return _STATUS_SKIP, "no file context"
    status, reason = _verify_existence(file, rel_files)
    if status != _STATUS_OK:
        return status, reason
    total = line_counts.get(file, 0)
    if 1 <= line_no <= total:
        return _STATUS_OK, f"within range ({file} has {total} lines)"
    return _STATUS_FLAGGED, f"line {line_no} out of range ({file} has {total} lines)"


def _verify_snippet(snippet: str, file: str | None, contents: dict[str, str]) -> tuple[str, str]:
    if file is None:
        return _STATUS_SKIP, "no file context"
    if not any(marker in snippet for marker in _SNIPPET_MARKERS):
        return _STATUS_SKIP, "prose fragment without code markers"
    if re.fullmatch(r"[^A-Za-z0-9_]+", snippet):
        return _STATUS_SKIP, "shell metacharacters or prose fragment"
    content = contents.get(file, "")
    if not content:
        return _STATUS_FLAGGED, "file not found in workspace"
    if snippet in content:
        return _STATUS_OK, "code pattern found in file"
    tokens = [t for t in re.findall(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*|\w+=\w+", snippet)
              if "." in t or "=" in t]
    if any(t in content for t in tokens):
        return _STATUS_OK, "code pattern found in file"
    return _STATUS_FLAGGED, "code pattern not found in file"


def _pair_symbol_with_line(claims: list[_Claim]) -> None:
    """Flag a line claim when a symbol on the same line is defined elsewhere."""
    lines = [c for c in claims if c.kind == "line" and c.status == _STATUS_OK and c.file]
    for c in claims:
        if c.kind != "symbol" or c.status != _STATUS_OK:
            continue
        if c.file is None:
            # Global hit: reason already stores "defined at file:line"
            m = re.search(r"defined at (\S+):(\d+)", c.reason)
            if not m:
                continue
            file, def_line = m.group(1), int(m.group(2))
        else:
            m = re.search(r":(\d+)", c.reason)
            if not m:
                continue
            file, def_line = c.file, int(m.group(1))
        best: _Claim | None = None
        best_dist = 1_000_000
        for lc in lines:
            if lc.file != file:
                continue
            dist = abs(lc.offset - c.offset)
            if dist < best_dist:
                best, best_dist = lc, dist
        if best is None or best_dist > 200:
            continue
        tol = _LINE_TOLERANCE_APPROX if best.approx else _LINE_TOLERANCE_EXACT
        if abs((best.line or 0) - def_line) > tol:
            best.status = _STATUS_FLAGGED
            best.reason = f"{c.text} is defined at {file}:{def_line} (claimed line {best.line})"


def _iter_lines(analysis: str) -> list[tuple[str, int]]:
    """Split analysis into lines, each with its absolute start offset.

    Lines are the claim-context unit: a ``*File: agent.py*`` mention scopes
    only the claims on the same line, avoiding bleed between bullet points.
    """
    output: list[tuple[str, int]] = []
    start = 0
    for line in analysis.splitlines(keepends=True):
        output.append((line.rstrip("\r\n"), start))
        start += len(line)
    return output


async def verify_analysis_claims(analysis: str, ws_path: Path) -> VerificationResult:
    """Verify code claims in *analysis* against the workspace tree at *ws_path*.

    Returns the original text with a ``## Verification Report`` appended when
    claims were found. Claim text is never rewritten — unverifiable claims are
    flagged in the report.
    """
    rel_files = _list_files(ws_path)
    if not rel_files:
        return VerificationResult(analysis, 0, 0)

    contents = await _read_files_async(ws_path, rel_files)
    defs_by_file: dict[str, dict[str, int]] = {}
    attrs_by_file: dict[str, set[str]] = {}
    line_counts: dict[str, int] = {}
    for rel, content in contents.items():
        defs, attrs = _index_file(content)
        defs_by_file[rel] = defs
        attrs_by_file[rel] = attrs
        line_counts[rel] = content.count("\n") + (1 if content else 0)

    claim_list: list[_Claim] = []
    for segment, seg_start in _iter_lines(analysis):
        claims = _extract_claims(segment, seg_start)
        for c in claims:
            if c.kind == "file":
                status, reason = _verify_existence(c.file, rel_files)
            elif c.kind == "line":
                _attach_context(c, claims)
                status, reason = _verify_line(c.line or 0, c.file, line_counts, rel_files)
            elif c.kind == "snippet":
                _attach_context(c, claims)
                status, reason = _verify_snippet(c.text, c.file, contents)
            else:
                _attach_context(c, claims)
                status, reason = _verify_symbol(c.text, c.file, defs_by_file, attrs_by_file)
                if c.check_absence:
                    if status == _STATUS_FLAGGED:
                        status, reason = _STATUS_OK, "confirmed absent (no definition found)"
                    elif status == _STATUS_OK and reason.startswith(("defined at", "attribute on self")):
                        status, reason = _STATUS_FLAGGED, f"claim says it is absent, but it is {reason}"
            c.status, c.reason = status, reason
        claim_list.extend(claims)

    _pair_symbol_with_line(claim_list)

    checked = [c for c in claim_list if c.status in (_STATUS_OK, _STATUS_FLAGGED)]
    flagged = [c for c in claim_list if c.status == _STATUS_FLAGGED]
    if not checked:
        return VerificationResult(analysis, 0, 0)

    lines = [
        f"- Code claims checked: {len(checked)} — {len(checked) - len(flagged)} verified, {len(flagged)} flagged."
    ]
    for c in flagged:
        lines.append(f"- [UNVERIFIED] `{c.text}` — {c.reason}")
    report = "\n\n---\n\n## Verification Report\n\n" + "\n".join(lines) + "\n"
    return VerificationResult(analysis + report, len(checked), len(flagged))