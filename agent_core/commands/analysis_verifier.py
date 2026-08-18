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

_ATTR_RE = re.compile(r"self\.([A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*[^=]+)?\s*=")
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
    segment_start: int = 0


def _list_files(ws_path: Path) -> list[str]:
    """Return workspace-relative file paths, skipping cache and venv dirs.

    Only ``.py`` files are returned — non-Python artifacts (``.json``,
    ``.jsonl``, ``.db``, ``.html``, etc.) are runtime/session state that can
    contain arbitrary strings matching symbol names, causing false positives in
    the global symbol search. Their existence is still verifiable via the raw
    filesystem fallback in ``_verify_existence``.
    """
    rel_files: list[str] = []
    if not ws_path.is_dir():
        return rel_files
    for root, dirs, files in os.walk(ws_path):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fn in files:
            if not fn.endswith(".py"):
                continue
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


def _shadows_stdlib(file_path: str) -> str | None:
    """Return the conflicting stdlib name if *file_path* shadows a stdlib module."""
    parts = file_path.replace("\\", "/").rstrip("/").split("/")
    for part in parts[:-1]:
        if part.lower() in _STDLIB_NAMES:
            return part
    return None


def _extract_claims(segment: str, seg_start: int) -> list[_Claim]:
    """Extract file, symbol, snippet, and line claims from one source line."""
    claims: list[_Claim] = []

    # Backticked regions are masked so _FILE_RE only sees bare file refs
    # (backticked file paths become dedicated file claims below).
    masked = _mask_backticks(segment)
    for m in _FILE_RE.finditer(masked):
        rel = m.group(1).replace("\\", "/")
        offset = seg_start + m.start()
        claims.append(_Claim(kind="file", text=rel, offset=offset, file=rel, segment_start=seg_start))
        line_m = re.match(r":(\d+)", segment[m.end():])
        if line_m:
            claims.append(_Claim(
                kind="line",
                text=f"{rel}:{line_m.group(1)}",
                offset=offset + m.end() - m.start(),
                file=rel,
                line=int(line_m.group(1)),
                segment_start=seg_start,
            ))

    for m in _BACKTICK_RE.finditer(segment):
        content = m.group(1)
        kind = _classify_backtick(content)
        offset = seg_start + m.start()
        if kind == "file":
            rel = content.replace("\\", "/")
            claims.append(_Claim(kind="file", text=rel, offset=offset, file=rel, segment_start=seg_start))
        elif kind == "symbol":
            claims.append(_Claim(kind="symbol", text=content, offset=offset, segment_start=seg_start))
        else:
            claims.append(_Claim(kind="snippet", text=content, offset=offset, segment_start=seg_start))

    for m in _LINE_RE.finditer(segment):
        if not _is_inside_backticks(segment, m.start()):
            claims.append(_Claim(
                kind="line",
                text=f"line {'~' if m.group(1) else ''}{m.group(2)}",
                offset=seg_start + m.start(),
                line=int(m.group(2)),
                approx=bool(m.group(1)),
                segment_start=seg_start,
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
    # Directory claims (paths ending with /) — check directory existence.
    if file.endswith("/"):
        dir_path = file.rstrip("/")
        for rel in rel_files:
            if os.path.dirname(rel).replace("\\", "/") == dir_path or rel.startswith(dir_path + "/"):
                return _STATUS_OK, "directory exists in workspace"
        # Also check the raw filesystem — .docs/ etc. may not have files listed
        # but still exist as directories.
        ws_dir = Path(file.rstrip("/")) if file else None
        if ws_dir is not None and ws_dir.is_dir():
            return _STATUS_OK, "directory exists in workspace"
        return _STATUS_FLAGGED, "directory not found in workspace"
    # Empty __init__.py files are valid package markers — presence matters.
    if file.endswith("__init__.py"):
        for rel in rel_files:
            if rel == file or os.path.basename(rel) == "__init__.py":
                return _STATUS_OK, "file exists in workspace"
        # Check raw filesystem too (empty files may not be indexed).
        ws_file = Path(file) if file else None
        if ws_file is not None and ws_file.is_file():
            return _STATUS_OK, "file exists in workspace"
    if file in rel_files:
        return _STATUS_OK, "file exists in workspace"
    # Partial path matching: the LLM may cite a sub-path without the full prefix
    # (e.g. `evolution/metrics.py` when the actual file is `agent1/evolution/metrics.py`).
    # Match by basename + directory suffix so near-duplicate paths resolve correctly.
    base = os.path.basename(file)
    for rel in rel_files:
        if os.path.basename(rel) == base and (rel.endswith(file) or file.endswith(os.path.dirname(rel).replace("\\", "/"))):
            return _STATUS_OK, f"file exists at {rel} (path prefix differs)"
    # Wildcard patterns (e.g. benchmark_*.json) — match against any existing file.
    if "*" in file or "?" in file:
        import fnmatch
        for rel in rel_files:
            if fnmatch.fnmatch(os.path.basename(rel), os.path.basename(file)):
                return _STATUS_OK, "file matching pattern exists in workspace"
        # Check raw filesystem too.
        ws_dir = Path(".")  # relative to cwd; verifier runs from workspace root
        for found in ws_dir.rglob("*"):
            if fnmatch.fnmatch(found.name, os.path.basename(file)):
                return _STATUS_OK, "file matching pattern exists in workspace"
        return _STATUS_FLAGGED, f"no file matches pattern {file}"
    # Check raw filesystem as fallback (files may exist but not be indexed).
    ws_file = Path(file) if file else None
    if ws_file is not None and ws_file.is_file():
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
    scope_files: list[str] | None = None,
) -> tuple[str, str]:
    if name in _STDLIB_NAMES or name in _EXTRA_MODULES:
        return _STATUS_OK, "standard-library or third-party name"
    if file is not None:
        defs = defs_by_file.get(file, {})
        if name in defs:
            return _STATUS_OK, f"defined at {file}:{defs[name]}"
        if name in attrs_by_file.get(file, set()):
            return _STATUS_OK, f"attribute on self in {file}"
        # A claim spanning several .py files on the same segment (e.g. "agent.py
        # and tool_router.py duplicate path helpers — «_normalize_path», ...")
        # is verified when the symbol exists in ANY .py file the segment names:
        # the prose scope is the whole segment, not just the nearest file mention.
        for sf in scope_files or ():
            if sf == file:
                continue
            if name in defs_by_file.get(sf, {}):
                return _STATUS_OK, f"defined at {sf}:{defs_by_file[sf][name]} (mentioned in claim segment)"
            if name in attrs_by_file.get(sf, set()):
                return _STATUS_OK, f"attribute on self in {sf} (mentioned in claim segment)"
        # Scoped-file miss: the LLM often cites the wrong file for a symbol that
        # exists elsewhere (e.g. "conftest.py defines Agent" when Agent is in agent.py).
        # A global hit means the claim is real — just mis-scoped, not fabricated.
        # Flag with an informative reason so the analysis author can correct the scope.
        hits = []
        for f, defs in defs_by_file.items():
            for name_in, ln in defs.items():
                if name_in == name:
                    hits.append((f, ln))
        if hits:
            f, ln = sorted(hits)[0]
            return _STATUS_FLAGGED, f"defined at {f}:{ln} but not in claimed file {file}"
        attr_files = sorted(f for f, attrs in attrs_by_file.items() if name in attrs)
        if attr_files:
            return _STATUS_FLAGGED, f"attribute on self in {attr_files[0]} but not in claimed file {file}"
        if re.fullmatch(r"[a-z][a-z0-9_]*", name):
            return _STATUS_SKIP, "prose-like identifier (not found anywhere)"
        return _STATUS_FLAGGED, f"symbol not found in {file} or anywhere"
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


def _verify_snippet(
    snippet: str,
    file: str | None,
    contents: dict[str, str],
    defs_by_file: dict[str, dict[str, int]] | None = None,
    attrs_by_file: dict[str, set[str]] | None = None,
    rel_files: list[str] | None = None,
) -> tuple[str, str]:
    if file is None:
        return _STATUS_SKIP, "no file context"
    # Wildcard patterns — two kinds: filename globs (benchmark_*.json) and module
    # path references with wildcards (commands.*, agent_core.logging_config).
    if "*" in snippet or "?" in snippet:
        import fnmatch
        base = os.path.basename(snippet)
        # Filename glob: match against existing file basenames.
        for rel, content in contents.items():
            if fnmatch.fnmatch(os.path.basename(rel), base):
                return _STATUS_OK, "file matching pattern exists in workspace"
        # Module path reference (e.g. commands.*): check if any import statement
        # references the module prefix — this is a valid code pattern even though
        # no single file literally contains the dotted wildcard string.
        prefix = snippet.rstrip(".*").rstrip(".")  # e.g. "commands" from "commands.*"
        for rel, content in contents.items():
            if re.search(r'from\s+[\w.]+\s+import\s+', content) and (prefix + ".") in content or prefix in content:
                return _STATUS_OK, f"module path reference found in {rel}"
        # Check raw filesystem for filename globs too.
        ws_dir = Path(".")  # verifier runs from workspace root
        for found in ws_dir.rglob("*"):
            if fnmatch.fnmatch(found.name, base):
                return _STATUS_OK, "file matching pattern exists in workspace"
        return _STATUS_FLAGGED, f"no file matches pattern {snippet}"
    # Bare function-call patterns like print() — the LLM may cite the zero-arg
    # form while the code uses print(...) with arguments. Recognize the call
    # prefix as a valid match even when the exact literal isn't present.
    if re.fullmatch(r"[A-Za-z_]\w*\(\)", snippet):
        func_name = snippet[:-2]  # strip ()
        for rel, content in contents.items():
            if f"{func_name}(" in content:
                return _STATUS_OK, "code pattern found in file"
        return _STATUS_FLAGGED, "code pattern not found in file"
    if not any(marker in snippet for marker in _SNIPPET_MARKERS):
        return _STATUS_SKIP, "prose fragment without code markers"
    if re.fullmatch(r"[^A-Za-z0-9_]+", snippet):
        return _STATUS_SKIP, "shell metacharacters or prose fragment"
    content = contents.get(file, "")
    if not content:
        # The claim may cite a short path ("tool_executor.py") while the index
        # key carries the full prefix ("agent_core/tool_executor.py") — resolve
        # like _verify_existence's partial matching before giving up.
        if file is not None:
            for rel in contents:
                if rel == file or rel.endswith("/" + file) or os.path.basename(rel) == os.path.basename(file):
                    content = contents[rel]
                    break
    if not content:
        # File may exist but be empty/unreadable — check raw filesystem.
        ws_file = Path(file) if file else None
        if ws_file is not None and ws_file.is_file():
            return _STATUS_SKIP, "file exists but no readable content"
        return _STATUS_FLAGGED, "file not found in workspace"
    if snippet in content:
        return _STATUS_OK, "code pattern found in file"
    tokens = [t for t in re.findall(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*|\w+=\w+", snippet)
              if "." in t or "=" in t]
    if any(t in content for t in tokens):
        return _STATUS_OK, "code pattern found in file"
    # Fallback: check all files globally (snippet may be scoped to wrong file).
    for rel, content in contents.items():
        if snippet in content or any(t in content for t in tokens):
            return _STATUS_OK, f"code pattern found in {rel}"
    # Dotted module/method references (e.g. agent_core.logging_config,
    # FileSystem.normalize_path, FileSearcher._safe_path) — the LLM cites a
    # dotted path that no single file literally contains as one string, but each
    # component exists as a class/def/import/module in some file. Verify by
    # checking if all components resolve to real definitions/imports anywhere.
    parts = snippet.split(".")
    if len(parts) >= 2 and all(p for p in parts):
        if defs_by_file is None or attrs_by_file is None or rel_files is None:
            return _STATUS_FLAGGED, "code pattern not found in file"
        resolved: list[str] = []
        for part in parts:
            found_in = None
            for f, defs in defs_by_file.items():
                if part in defs:
                    found_in = f
                    break
            if not found_in:
                for f, attrs in attrs_by_file.items():
                    if part in attrs:
                        found_in = f
                        break
            # Module path components (e.g. "agent_core", "security") may be
            # directory/file names rather than definitions — a package directory
            # (part appears as a path component of some file) or a module file
            # (basename 'part.py') both count as the component existing.
            if not found_in and any(
                part in rel.split("/") for rel in rel_files
            ):
                found_in = f"{part}/"
            if not found_in and any(
                os.path.basename(rel).rsplit(".", 1)[0] == part for rel in rel_files
            ):
                found_in = next(
                    rel for rel in rel_files
                    if os.path.basename(rel).rsplit(".", 1)[0] == part
                )
            resolved.append(found_in or "")
        if all(r for r in resolved):
            return _STATUS_OK, f"all components of dotted reference resolve ({snippet})"
    return _STATUS_FLAGGED, "code pattern not found in file"


def _pair_symbol_with_line(claims: list[_Claim]) -> None:
    """Flag a line claim when a symbol on the same source segment is defined elsewhere.

    Pairing is restricted to claims sharing the same analysis source-line (segment_start)
    so a ``line N`` mention in one bullet point cannot bleed into a symbol verified in an
    adjacent bullet — matching _iter_lines' "avoiding bleed between bullet points" design.
    """
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
        for lc in lines:
            if lc.file != file or lc.segment_start != c.segment_start:
                continue
            dist = abs(lc.offset - c.offset)
            if dist < 200 and (best is None or dist < abs(best.offset - c.offset)):
                best = lc
        if best is None:
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
                if status == _STATUS_OK and c.file:
                    shadow = _shadows_stdlib(c.file)
                    if shadow:
                        status, reason = _STATUS_FLAGGED, (
                            f"shadows stdlib module '{shadow}' — rename directory"
                        )
            elif c.kind == "line":
                _attach_context(c, claims)
                status, reason = _verify_line(c.line or 0, c.file, line_counts, rel_files)
            elif c.kind == "snippet":
                _attach_context(c, claims)
                status, reason = _verify_snippet(
                    c.text, c.file, contents, defs_by_file, attrs_by_file, rel_files
                )
            else:
                _attach_context(c, claims)
                scope_files = [
                    fc.file for fc in claims
                    if fc.kind == "file" and fc.file and fc.file.endswith(".py")
                ]
                status, reason = _verify_symbol(
                    c.text, c.file, defs_by_file, attrs_by_file, scope_files
                )
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