#!/usr/bin/env python3
"""Rewrap over-long source lines (ruff E501) without changing behaviour.

The rewriter is deliberately conservative.  A candidate edit is only accepted
when it passes three gates:

1. VALUE GATE - the parsed AST of the file must be identical before and
   after the edit (``ast.dump`` equality).  Any change to a literal, name,
   call or expression is rejected, so no runtime value can change.  This
   also covers docstrings: the exact ``__doc__`` text (including trailing
   whitespace) is part of the AST.
2. FORMAT GATE - the rewritten file must be a fixed point of ``ruff format``,
   the project's formatter.  Otherwise the next format run would undo the
   edit and the repository would gain churn.
3. LINT GATE - the E501 violation on the target line must be gone and the
   total number of violations must strictly decrease.

Candidates are generated loosely and then normalised by ``ruff format``
before the gates are applied, so the fixer never has to guess how the
formatter would lay a construct out.

Strategies:

``rebalance``
    A group of adjacent string literals (an implicit concatenation) is
    merged, re-split at word boundaries and rewritten so that every part
    fits.  The concatenated value is unchanged, and the result reads like
    text that was written to fit.
``split``
    A single literal is split into several adjacent literals at word
    boundaries.
``comment``
    An over-long comment is split into two comments at a word boundary.
``continuation``
    One long physical line of a literal is cut at a word boundary and the
    two halves are joined by a backslash continuation.  The literal stays a
    single token, so a docstring stays a docstring and its value does not
    change.  This is the only strategy that can shorten a line without
    changing the shape of the code, so it is the fallback for docstrings.

Usage::

    python scripts/fix_e501.py                # report only
    python scripts/fix_e501.py --apply        # rewrite the files
    python scripts/fix_e501.py --apply --jobs 8
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import tokenize
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

LINE_LIMIT = 88
RUFF = [sys.executable, "-m", "ruff"]
#: Total number of candidates tried for one row.
MAX_CANDIDATES = 32
#: Candidates kept per strategy, so a chatty strategy cannot starve the
#: others (a truncated list would hide the only candidate that works).
MAX_PER_STRATEGY = 6
MAX_PARTS = 12

#: An escape sequence inside a string literal.  A split may not fall inside
#: one, because that would change the value of the literal.
_ESCAPE_RE = re.compile(
    r"\\(?:x[0-9a-fA-F]{2}|[0-7]{1,3}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|N\{[^}]*\}|.)",
    re.S,
)

#: A doubled brace inside an f-string.  It stands for one literal brace,
#: so a split may not fall between the two braces.
_BRACE_ESCAPE_RE = re.compile(r"\{\{|\}\}")

#: Punctuation that ends a sentence in prose.  A split right after one of
#: these keeps the sentence inside one part, so the phrase stays whole.
_SENTENCE_ENDERS = ".!?:"

#: Comment text that carries a tool directive.  Splitting such a comment
#: would change what the directive means, so it is left alone.
_DIRECTIVE_MARKERS = (
    "noqa",
    "type:",
    "pragma",
    "fmt:",
    "ruff:",
    "mypy",
    "pylint",
    "flake8",
    "eslint",
    "isort:",
    "coverage:",
    "pyright:",
    "codespell:",
)


# ---------------------------------------------------------------------------
# ruff plumbing
# ---------------------------------------------------------------------------


def _run(args: list[str], text: Optional[str] = None) -> subprocess.CompletedProcess:
    # encoding/errors are pinned: ruff emits UTF-8 and the locale codepage
    # would otherwise mojibake (or crash) the decoded output.
    return subprocess.run(
        RUFF + args,
        input=text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _check_files(files: list[Path], config: Optional[Path]) -> dict[Path, list[int]]:
    """E501 rows per file, for a batch of files."""
    if not files:
        return {}
    args = ["check", "--select", "E501", "--no-cache", "--output-format", "json"]
    if config is not None:
        args += ["--config", str(config)]
    args += [str(f) for f in files]
    proc = _run(args)
    try:
        data = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return {}
    out: dict[Path, list[int]] = defaultdict(list)
    for item in data:
        out[Path(item["filename"]).resolve()].append(int(item["location"]["row"]))
    return {k: sorted(v) for k, v in out.items()}


def _format_files(files: list[Path], config: Optional[Path]) -> None:
    """Format a batch of files in place."""
    if not files:
        return
    args = ["format", "--no-cache"]
    if config is not None:
        args += ["--config", str(config)]
    args += [str(f) for f in files]
    _run(args)


def _find_config(path: Path) -> Optional[Path]:
    """The nearest pyproject.toml that configures ruff, for ``path``."""
    for parent in [path.parent, *path.parent.parents]:
        candidate = parent / "pyproject.toml"
        if candidate.is_file():
            try:
                if "[tool.ruff]" in candidate.read_text(encoding="utf-8"):
                    return candidate
            except OSError:
                return None
    return None


# ---------------------------------------------------------------------------
# source model
# ---------------------------------------------------------------------------


@dataclass
class Source:
    """A source file plus a line -> absolute offset index."""

    path: Path
    text: str
    line_starts: list[int] = field(default_factory=list)

    @classmethod
    def make(cls, path: Path, text: str) -> "Source":
        starts = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                starts.append(i + 1)
        return cls(path, text, starts)

    def offset(self, pos: tuple[int, int]) -> int:
        """Absolute offset of a (1-based row, 0-based column) position."""
        return self.line_starts[pos[0] - 1] + pos[1]

    def line_end(self, row: int) -> int:
        start = self.line_starts[row - 1]
        end = self.text.find("\n", start)
        return len(self.text) if end == -1 else end

    def line_text(self, row: int) -> str:
        return self.text[self.line_starts[row - 1] : self.line_end(row)]

    def indent_of(self, row: int) -> str:
        line = self.line_text(row)
        return line[: len(line) - len(line.lstrip())]


# ---------------------------------------------------------------------------
# string tokens
# ---------------------------------------------------------------------------


@dataclass
class StrTok:
    start: tuple[int, int]
    end: tuple[int, int]
    text: str
    prefix: str
    quote: str
    content: str
    spans: list[tuple[int, int]]
    depth: int = 0
    adjacent: bool = False

    @property
    def is_raw(self) -> bool:
        return "r" in self.prefix.lower()

    @property
    def is_fstring(self) -> bool:
        return "f" in self.prefix.lower()

    @property
    def multiline(self) -> bool:
        return self.start[0] != self.end[0]


def _fstring_spans(content: str) -> list[tuple[int, int]]:
    """Ranges of ``content`` that are literal text (outside f-string expressions)."""
    spans: list[tuple[int, int]] = []
    depth = 0
    start = 0
    i = 0
    n = len(content)
    while i < n:
        ch = content[i]
        if depth == 0:
            if ch == "{":
                if content[i : i + 2] == "{{":
                    i += 2
                    continue
                spans.append((start, i))
                depth = 1
                i += 1
                continue
            if ch == "}" and content[i : i + 2] == "}}":
                i += 2
                continue
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                start = i + 1
        elif ch in "\"'":
            quote = ch
            i += 1
            while i < n and content[i] != quote:
                if content[i] == "\\":
                    i += 1
                i += 1
        i += 1
    if depth == 0:
        spans.append((start, n))
    return spans


def _scan_string(text: str) -> tuple[str, str, str, list[tuple[int, int]]]:
    """Split a string token's source into (prefix, quote, content, literal spans)."""
    i = 0
    while i < len(text) and text[i] not in "\"'":
        i += 1
    prefix = text[:i]
    rest = text[i:]
    quote = rest[:3] if rest[:3] in ('"""', "'''") else rest[:1]
    content = text[len(prefix) + len(quote) : len(text) - len(quote)]
    if "f" in prefix.lower():
        spans = _fstring_spans(content)
    else:
        spans = [(0, len(content))]
    return prefix, quote, content, spans


def _make_token(src: Source, start, end, depth: int, adjacent: bool) -> StrTok:
    a = src.offset(start)
    b = src.offset(end)
    text = src.text[a:b]
    prefix, quote, content, spans = _scan_string(text)
    return StrTok(
        start=start,
        end=end,
        text=text,
        prefix=prefix,
        quote=quote,
        content=content,
        spans=spans,
        depth=depth,
        adjacent=adjacent,
    )


def logical_strings(src: Source) -> list[StrTok]:
    """Every string literal of the file, flagged with adjacency and depth.

    ``adjacent`` is true when the literal directly follows another literal,
    i.e. the two form an implicit concatenation.
    """
    try:
        raw = list(tokenize.generate_tokens(io.StringIO(src.text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []
    out: list[StrTok] = []
    depth = 0
    prev_sig: Optional[str] = None
    fstart = getattr(tokenize, "FSTRING_START", None)
    fend = getattr(tokenize, "FSTRING_END", None)
    i = 0
    while i < len(raw):
        tok = raw[i]
        if tok.type in (
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.COMMENT,
            tokenize.ENDMARKER,
        ):
            i += 1
            continue
        if tok.type == tokenize.OP:
            if tok.string in "([{":
                depth += 1
            elif tok.string in ")]}":
                depth = max(0, depth - 1)
            prev_sig = "op"
            i += 1
            continue
        if tok.type == tokenize.STRING:
            out.append(_make_token(src, tok.start, tok.end, depth, prev_sig == "str"))
            prev_sig = "str"
            i += 1
            continue
        if fstart is not None and tok.type == fstart:
            # An f-string may hold another f-string, so the matching end
            # is the one that closes every nested f-string.
            j = i
            nested = 0
            while j < len(raw):
                if raw[j].type == fstart:
                    nested += 1
                elif raw[j].type == fend:
                    nested -= 1
                    if nested == 0:
                        break
                j += 1
            if j < len(raw):
                out.append(
                    _make_token(src, tok.start, raw[j].end, depth, prev_sig == "str")
                )
                prev_sig = "str"
                i = j + 1
                continue
        prev_sig = "other"
        i += 1
    return out


def string_groups(src: Source) -> list[list[StrTok]]:
    """Maximal runs of adjacent string literals (implicit concatenations)."""
    groups: list[list[StrTok]] = []
    current: list[StrTok] = []
    for tok in logical_strings(src):
        if tok.adjacent and current:
            current.append(tok)
        else:
            if current:
                groups.append(current)
            current = [tok]
    if current:
        groups.append(current)
    return groups


# ---------------------------------------------------------------------------
# split points
# ---------------------------------------------------------------------------


def _expr_regions(tok: StrTok) -> list[tuple[int, int]]:
    """The regions of ``tok.content`` that hold f-string expressions."""
    if not tok.is_fstring:
        return []
    regions: list[tuple[int, int]] = []
    pos = 0
    for lo, hi in tok.spans:
        if lo > pos:
            regions.append((pos, lo))
        pos = max(pos, hi)
    if pos < len(tok.content):
        regions.append((pos, len(tok.content)))
    return regions


def _blocked_regions(tok: StrTok) -> list[tuple[int, int]]:
    """Regions of ``tok.content`` a split may not fall inside.

    Escape sequences are blocked because cutting one changes the value.
    For f-strings the expression parts are blocked because cutting one
    would split an expression, and the doubled braces are blocked because
    they stand for a single literal brace.
    """
    content = tok.content
    blocked: list[tuple[int, int]] = []
    if not tok.is_raw:
        blocked += [(m.start(), m.end()) for m in _ESCAPE_RE.finditer(content)]
    blocked += _expr_regions(tok)
    if tok.is_fstring:
        for m in _BRACE_ESCAPE_RE.finditer(content):
            blocked.append((m.start(), m.end() + 1))
    return blocked


def _safe_offsets(tok: StrTok) -> list[int]:
    """Offsets in ``tok.content`` where a split cannot break anything."""
    content = tok.content
    blocked = _blocked_regions(tok)
    return [
        k for k in range(len(content) + 1) if not any(lo < k < hi for lo, hi in blocked)
    ]


def _sentence_points(content: str, safe: list[int]) -> list[int]:
    """Split offsets that fall just after a sentence boundary.

    Two shapes are recognised:

    * ``"punctuation "`` - the cut falls after the space, so the part
      ends with ``"...punctuation "``.
    * ``"punctuation\\n"`` - the newline is an escape sequence, so the
      cut falls before it and the part ends with ``"...punctuation"``.

    Either way the whole sentence stays inside one part.  This keeps
    prose phrases contiguous in the source text.  A test that looks for
    a phrase in a file reads the source, not the parsed value, so a
    split in the middle of a sentence hides the phrase from it.
    """
    points: list[int] = []
    for k in safe:
        if k <= 0 or k >= len(content):
            continue
        if content[k - 1] == " ":
            if k >= 2 and content[k - 2] in _SENTENCE_ENDERS:
                points.append(k)
        elif content[k - 1] in _SENTENCE_ENDERS and content[k : k + 2] == "\\n":
            points.append(k)
    return points


def _merged_offsets(group: list[StrTok]) -> tuple[str, list[int]]:
    """The concatenated content of a group plus its safe split offsets.

    Escape sequences are tracked per literal, because each literal has its
    own prefix and may be raw while the others are not.
    """
    merged = "".join(t.content for t in group)
    safe: set[int] = set()
    base = 0
    for tok in group:
        blocked = _blocked_regions(tok)
        for k in range(len(tok.content) + 1):
            if any(lo < k < hi for lo, hi in blocked):
                continue
            safe.add(base + k)
        base += len(tok.content)
    safe.add(len(merged))
    return merged, sorted(safe)


def _quote_part(tok: StrTok, content: str) -> str:
    return tok.prefix + tok.quote + content + tok.quote


def _fits(*parts: str) -> bool:
    return all(len(p) <= LINE_LIMIT for p in parts)


def _splice_rows(src: Source, first: int, last: int, lines: list[str]) -> str:
    """Replace physical lines ``first``..``last`` of the file with ``lines``."""
    start = src.line_starts[first - 1]
    end = src.line_end(last)
    return src.text[:start] + "\n".join(lines) + src.text[end:]


# ---------------------------------------------------------------------------
# chunk planning
# ---------------------------------------------------------------------------


def _search_parts(
    points: list[int],
    limit: int,
    overhead: int,
    budgets: list[int],
    chunk: int,
    start: int,
    memo: dict,
) -> Optional[list[int]]:
    """Find split points that make every chunk fit its budget.

    ``budgets`` holds one budget per chunk.  The last chunk must consume
    the content exactly; the others may end at any candidate point.
    """
    key = (chunk, start)
    if key in memo:
        return memo[key]
    memo[key] = None
    if chunk == len(budgets) - 1:
        if limit - start + overhead <= budgets[chunk]:
            memo[key] = [limit]
            return memo[key]
        return None

    ideal = start + (limit - start) / (len(budgets) - chunk)
    rest_capacity = sum(budgets[chunk + 1 :]) - (len(budgets) - chunk - 1) * overhead
    options = [
        p for p in points if p > start and p - start + overhead <= budgets[chunk]
    ]
    options.sort(key=lambda p: abs(p - ideal))
    for end in options:
        if limit - end > rest_capacity:
            continue
        rest = _search_parts(points, limit, overhead, budgets, chunk + 1, end, memo)
        if rest is not None:
            memo[key] = [end] + rest
            return memo[key]
    return None


def _plan_parts(
    content: str,
    points: list[int],
    overhead: int,
    budget_first: int,
    budget_mid: int,
    budget_last: int,
    min_parts: int,
    count: Optional[int] = None,
) -> Optional[list[int]]:
    """Choose the chunk boundaries of a literal.

    Returns the end offsets of the chunks, or None when no layout fits.
    With ``count`` the number of chunks is fixed, otherwise the smallest
    feasible number of chunks is used.
    """
    limit = len(content)
    if limit == 0:
        return None
    counts: range = range(min_parts, MAX_PARTS + 1)
    if count is not None:
        counts = range(count, count + 1)
    for n in counts:
        if n == 1:
            budgets = [budget_first]
        else:
            budgets = [budget_first] + [budget_mid] * (n - 2) + [budget_last]
        if sum(budgets) - n * overhead < limit:
            continue
        found = _search_parts(points, limit, overhead, budgets, 0, 0, {})
        if found is not None:
            return found
    return None


# ---------------------------------------------------------------------------
# candidate generation
# ---------------------------------------------------------------------------


def _group_candidates(src: Source, group: list[StrTok]) -> list[str]:
    """Candidates that re-balance a group of adjacent literals."""
    head = group[0]
    if any(t.prefix != head.prefix or t.quote != head.quote for t in group):
        return []
    if any(t.multiline for t in group):
        return []

    first_row = group[0].start[0]
    last_row = group[-1].end[0]
    prefix_text = src.text[src.line_starts[first_row - 1] : src.offset(group[0].start)]
    suffix_text = src.text[src.offset(group[-1].end) : src.line_end(last_row)]
    overhead = len(head.prefix) + 2 * len(head.quote)

    content, safe = _merged_offsets(group)
    word = [k for k in safe if 0 < k <= len(content) and content[k - 1] == " "]
    # Sentence boundaries come first: they keep a prose phrase inside one
    # part, so a test that searches the source text still finds it.
    sentence = _sentence_points(content, safe)
    point_sets: list[list[int]] = []
    if sentence:
        point_sets.append(sentence)
    if word:
        point_sets.append(word)
    point_sets.append(safe)

    close_indent = src.indent_of(first_row)
    body_indent = (
        src.indent_of(last_row) if group[0].depth > 0 else close_indent + "    "
    )

    layouts: list[tuple[bool, int, int, int]] = []
    if group[0].depth > 0:
        # Already inside brackets: keep them, no new parentheses needed.
        layouts.append(
            (
                False,
                LINE_LIMIT - len(prefix_text),
                LINE_LIMIT - len(body_indent),
                LINE_LIMIT - len(body_indent) - len(suffix_text),
            )
        )
    layouts.append(
        (
            True,
            LINE_LIMIT - len(body_indent),
            LINE_LIMIT - len(body_indent),
            LINE_LIMIT - len(body_indent),
        )
    )

    out: list[str] = []
    for wrap, budget_first, budget_mid, budget_last in layouts:
        if wrap and len(prefix_text) + 1 > LINE_LIMIT:
            continue
        min_parts = 1 if wrap else 2
        for points in point_sets:
            parts = _plan_parts(
                content,
                points,
                overhead,
                budget_first,
                budget_mid,
                budget_last,
                min_parts,
            )
            if not parts or len(parts) < min_parts:
                continue
            chunks = []
            start = 0
            for end in parts:
                chunks.append(content[start:end])
                start = end
            if wrap:
                lines = [prefix_text + "("]
                lines += [body_indent + _quote_part(head, c) for c in chunks]
                lines += [close_indent + ")" + suffix_text]
            else:
                lines = [prefix_text + _quote_part(head, chunks[0])]
                lines += [body_indent + _quote_part(head, c) for c in chunks[1:-1]]
                lines += [body_indent + _quote_part(head, chunks[-1]) + suffix_text]
            if _fits(*lines):
                out.append(_splice_rows(src, first_row, last_row, lines))
                break
    return out


def _single_token_candidates(src: Source, tok: StrTok) -> list[str]:
    """Candidates that split one literal into several adjacent literals.

    Used when a group of adjacent literals cannot be re-balanced as a
    whole (mixed prefixes or quotes).  The literal keeps its place, so
    the implicit concatenation keeps its value.

    An f-string keeps its ``f`` prefix in every part.  The parts are
    concatenated before the compiler looks at them, so the result is one
    f-string again and its value does not change.
    """
    if tok.multiline:
        return []
    row = tok.start[0]
    indent = src.indent_of(row)
    inner = indent + "    "
    overhead = len(tok.prefix) + 2 * len(tok.quote)

    tok_start = src.offset(tok.start)
    tok_end = src.offset(tok.end)
    prefix_text = src.text[src.line_starts[row - 1] : tok_start]
    suffix_text = src.text[tok_end : src.line_end(row)]

    content, safe = _merged_offsets([tok])
    word = [k for k in safe if 0 < k <= len(content) and content[k - 1] == " "]
    # Sentence boundaries come first: they keep a prose phrase inside one
    # part, so a test that searches the source text still finds it.
    sentence = _sentence_points(content, safe)
    point_sets: list[list[int]] = []
    if sentence:
        point_sets.append(sentence)
    if word:
        point_sets.append(word)
    point_sets.append(safe)

    out: list[str] = []
    for points in point_sets:
        if tok.depth > 0:
            budgets = (
                LINE_LIMIT - len(prefix_text),
                LINE_LIMIT - len(inner),
                LINE_LIMIT - len(inner) - len(suffix_text),
            )
        else:
            budget = LINE_LIMIT - len(inner)
            budgets = (budget, budget, budget)
        ends = _plan_parts(content, points, overhead, *budgets, 2)
        if not ends:
            continue
        chunks: list[str] = []
        start = 0
        for end in ends:
            chunks.append(content[start:end])
            start = end
        if tok.depth > 0:
            lines = [prefix_text + _quote_part(tok, chunks[0])]
            lines += [inner + _quote_part(tok, c) for c in chunks[1:-1]]
            lines += [inner + _quote_part(tok, chunks[-1]) + suffix_text]
        else:
            lines = [prefix_text + "("]
            lines += [inner + _quote_part(tok, c) for c in chunks]
            lines += [indent + ")" + suffix_text]
        if _fits(*lines):
            out.append(_splice_rows(src, row, row, lines))
            break
    return out


def _comment_candidates(src: Source, row: int) -> list[str]:
    """Candidates that split an over-long comment line."""
    line = src.line_text(row)
    hash_at = line.find("#")
    if hash_at < 0:
        return []
    code = line[:hash_at].rstrip()
    body = line[hash_at + 1 :].strip()
    if not body:
        return []
    if any(marker in body for marker in _DIRECTIVE_MARKERS):
        # Splitting a directive would change what the directive means.
        return []
    indent = line[: len(line) - len(line.lstrip())]
    out: list[str] = []
    for k in range(len(body) - 1, 0, -1):
        if body[k] != " ":
            continue
        if code:
            head = code + "  # " + body[:k].rstrip()
        else:
            head = indent + "# " + body[:k].rstrip()
        tail = indent + "# " + body[k + 1 :].lstrip()
        if _fits(head, tail):
            out.append(_splice_rows(src, row, row, [head, tail]))
    return out


def _continuation_candidates(src: Source, tok: StrTok, row: int) -> list[str]:
    """Candidates that shorten one over-long physical line of a literal.

    The line is cut at a word boundary and the two halves are joined by a
    backslash continuation, so the newline leaves the value of the literal
    while the literal itself stays a single token: a docstring stays a
    docstring and its exact value is preserved.

    The continuation starts at column 0.  A backslash continuation swallows
    the newline *and* the indentation of the next line, so any indentation
    there would leak into the value of the literal.
    """
    if tok.is_raw:
        # In a raw literal the backslash stays in the value.
        return []
    if tok.is_fstring:
        # A backslash inside an f-string is version dependent, so the
        # strategy does not touch f-strings.
        return []
    if not tok.multiline or len(tok.quote) != 3:
        # A backslash continuation is only legal inside a triple-quoted
        # literal.  In a single-quoted literal it is a syntax error.
        return []
    content_start = src.offset(tok.start) + len(tok.prefix) + len(tok.quote)
    line_start = src.line_starts[row - 1]
    line_end = src.line_end(row)
    lo = max(0, line_start - content_start)
    hi = min(len(tok.content), line_end - content_start)
    if hi <= lo:
        return []
    prefix_on_row = (
        src.text[line_start:content_start] if content_start > line_start else ""
    )
    suffix_on_row = (
        src.text[src.offset(tok.end) : line_end] if tok.end[0] == row else ""
    )
    # An f-string is a list of literal chunks in the AST, so the cut may
    # only fall on a chunk boundary, i.e. right before an expression.
    expr_starts = {hi_ for hi_ in (s[1] for s in tok.spans) if hi_ < len(tok.content)}
    safe = _safe_offsets(tok)
    out: list[str] = []
    for k in safe:
        if not (lo < k <= hi):
            continue
        if tok.content[k - 1] != " ":
            continue
        if tok.is_fstring and k not in expr_starts:
            continue
        head = prefix_on_row + tok.content[lo:k] + "\\"
        body = tok.content[k:hi] + suffix_on_row
        if not _fits(head, body):
            continue
        out.append(src.text[:line_start] + head + "\n" + body + src.text[line_end:])
    return out[:MAX_CANDIDATES]


def candidates_for_row(src: Source, row: int, groups: list[list[StrTok]]) -> list[str]:
    """All candidate rewrites for the E501 reported on ``row``."""
    out: list[str] = []
    toks = logical_strings(src)
    buckets: list[list[str]] = []

    # Every group that covers the row is a candidate target.  A line can
    # hold several groups, for example a dict key and its long value:
    #   {"content": "a very long string ..."}
    # The key is a group of its own, so only the first group of the line
    # must not be allowed to hide the others.
    covering = [g for g in groups if g[0].start[0] <= row <= g[-1].end[0]]

    # Re-balancing a group of adjacent literals.
    for group in covering:
        buckets.append(_group_candidates(src, group))

    # Splitting a single literal into several adjacent literals.
    for group in covering:
        for tok in group:
            if tok.start[0] <= row <= tok.end[0]:
                buckets.append(_single_token_candidates(src, tok))
                break

    # A comment that makes the line too long: split it into two comments.
    buckets.append(_comment_candidates(src, row))

    # A long physical line inside a multi-line literal: cut it with a
    # backslash continuation.  This keeps the literal a single token, so
    # it is the only strategy that can shorten a docstring without
    # changing its value.
    for tok in toks:
        if tok.multiline and tok.start[0] <= row <= tok.end[0]:
            buckets.append(_continuation_candidates(src, tok, row))
            break

    for bucket in buckets:
        out.extend(bucket[:MAX_PER_STRATEGY])

    seen: set[str] = set()
    unique: list[str] = []
    for text in out:
        if text not in seen:
            seen.add(text)
            unique.append(text)
    return unique[:MAX_CANDIDATES]


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def _ast_dump(text: str) -> Optional[str]:
    try:
        return ast.dump(ast.parse(text))
    except (SyntaxError, ValueError):
        return None


def _try_fix(src: Source, row: int, config: Optional[Path], tmp: Path) -> Optional[str]:
    """Try to fix the violation on ``row``; return the accepted text or None."""
    groups = string_groups(src)
    candidates = candidates_for_row(src, row, groups)
    if not candidates:
        return None
    base_ast = _ast_dump(src.text)
    if base_ast is None:
        return None

    staged: list[Path] = []
    for cand in candidates:
        if _ast_dump(cand) != base_ast:
            continue
        path = tmp / f"c{len(staged)}.py"
        path.write_text(cand, encoding="utf-8")
        staged.append(path)
    if not staged:
        return None

    # Normalise every survivor with the formatter.
    _format_files(staged, config)
    survivors = [
        p for p in staged if _ast_dump(p.read_text(encoding="utf-8")) == base_ast
    ]
    if not survivors:
        return None

    # FORMAT GATE: ruff format is idempotent, so a formatted file is a fixed
    # point.  Re-format to be sure.
    second = tmp / "second"
    second.mkdir(exist_ok=True)
    copies: list[tuple[Path, Path]] = []
    for i, p in enumerate(survivors):
        copy = second / f"s{i}.py"
        copy.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        copies.append((p, copy))
    _format_files([c for _, c in copies], config)
    stable = [
        p
        for p, c in copies
        if c.read_text(encoding="utf-8") == p.read_text(encoding="utf-8")
    ]
    if not stable:
        return None

    # LINT GATE: the target row must be fixed and the count must drop.
    counts = _check_files(stable, config)
    base_count = len(_check_files([src.path], config).get(src.path.resolve(), []))
    for p in stable:
        new_rows = counts.get(p.resolve(), [])
        if row in new_rows:
            continue
        if len(new_rows) >= base_count:
            continue
        return p.read_text(encoding="utf-8")
    return None


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def fix_text(
    path: Path, text: str, config: Optional[Path]
) -> tuple[str, list[int], list[int]]:
    """Rewrap the E501 violations of ``text``.

    Returns (new_text, fixed_rows, unfixed_rows).
    """
    src = Source.make(path, text)
    fixed: list[int] = []
    unfixed: list[int] = []
    failed: set[tuple[str, int]] = set()
    with tempfile.TemporaryDirectory(prefix="e501_") as tmpdir:
        tmp = Path(tmpdir)
        current = tmp / "current.py"
        current.write_text(src.text, encoding="utf-8")
        for _ in range(1000):
            rows = _check_files([current], config).get(current.resolve(), [])
            if not rows:
                break
            seen: dict[str, int] = defaultdict(int)
            pending: list[tuple[int, tuple[str, int]]] = []
            for row in rows:
                sig = src.line_text(row) if row <= len(src.line_starts) else ""
                key = (sig, seen[sig])
                seen[sig] += 1
                if key in failed:
                    continue
                pending.append((row, key))
            if not pending:
                break
            row, key = pending[-1]
            accepted = _try_fix(src, row, config, tmp)
            if accepted is None:
                failed.add(key)
                unfixed.append(row)
                continue
            src = Source.make(path, accepted)
            current.write_text(accepted, encoding="utf-8")
            fixed.append(row)
    return src.text, fixed, unfixed


def _e501_rows_of_text(path: Path, text: str, config: Optional[Path]) -> list[int]:
    """E501 rows of ``text`` (staged in a temporary file)."""
    with tempfile.TemporaryDirectory(prefix="e501r_") as tmpdir:
        tmp = Path(tmpdir) / "probe.py"
        tmp.write_text(text, encoding="utf-8")
        return _check_files([tmp], config).get(tmp.resolve(), [])


def fix_file(path: Path, apply: bool = False) -> tuple[int, int, list[int], bool]:
    """Rewrap one file.  Returns (fixed, remaining, unfixed rows, skipped)."""
    text = path.read_text(encoding="utf-8")
    config = _find_config(path)
    # Safety: never touch a file the formatter would rewrite anyway.
    probe = _run(
        ["format", "--no-cache", "--stdin-filename", str(path), "-"], text=text
    )
    if probe.returncode != 0 or probe.stdout != text:
        rows = _check_files([path], config).get(path.resolve(), [])
        return 0, len(rows), [], True

    new_text, fixed, unfixed = fix_text(path, text, config)
    if apply and new_text != text:
        path.write_text(new_text, encoding="utf-8")
    remaining = len(_e501_rows_of_text(path, new_text, config))
    return len(fixed), remaining, unfixed, False


def collect_files(root: Path) -> list[Path]:
    proc = _run(
        [
            "check",
            "--select",
            "E501",
            "--no-cache",
            "--output-format",
            "json",
            str(root),
        ]
    )
    try:
        data = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return []
    return sorted({Path(d["filename"]).resolve() for d in data})


def _worker(payload: tuple[str, bool]) -> tuple[str, int, int, list[int], bool]:
    path_str, apply = payload
    path = Path(path_str)
    fixed, remaining, unfixed, skipped = fix_file(path, apply=apply)
    return str(path), fixed, remaining, unfixed, skipped


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rewrap over-long lines (E501).")
    parser.add_argument(
        "--apply", action="store_true", help="rewrite the files (default: report only)"
    )
    parser.add_argument("--path", default=".", help="path to scan (default: .)")
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(12, (os.cpu_count() or 2)),
        help="number of parallel workers",
    )
    args = parser.parse_args(argv)

    root = Path(args.path).resolve()
    files = collect_files(root)
    print(f"files with E501: {len(files)}")

    paths = [(str(f), args.apply) for f in files]
    if args.jobs > 1 and len(paths) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(_worker, paths))
    else:
        results = [_worker(p) for p in paths]

    total_fixed = 0
    total_left = 0
    todo: list[str] = []
    for path_str, fixed, remaining, unfixed, skipped in results:
        total_fixed += fixed
        total_left += remaining
        rel = path_str.replace(str(root) + os.sep, "")
        if fixed or remaining:
            print(f"{rel}: fixed {fixed}, remaining {remaining}")
        if skipped:
            print("    SKIPPED: file is not format-clean")
        if unfixed:
            todo.append(rel)
            print(f"    unfixed rows: {unfixed}")

    print(f"\nfixed={total_fixed} remaining={total_left}")
    if todo:
        print(f"files with unfixed rows: {len(todo)}")
        for t in todo:
            print("   ", t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
