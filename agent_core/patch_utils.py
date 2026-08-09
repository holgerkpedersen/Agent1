"""Shared unified-diff (``@@`` hunks) parsing and application for LLM patches.

Used by ``implement_cmd`` and ``optimize_cmd`` so patch-mode rewrites share
the exact same tolerant hunk semantics (whitespace-tolerant matching,
padding-space detection, broken-hunk filtering, syntax check after apply).

``apply_anchored_patch`` is the content-based fallback for flaky LLM hunks:
LLMs routinely emit wrong ``@@`` line numbers, merge several headers together,
and wrap the diff in markdown fences — so instead of trusting positions it
anchors each hunk by matching the patch lines against the file text, using
the claimed position only as a search window hint.
"""
import difflib
import os
import re
import subprocess
import tempfile


def normalize_patch_block(patch_text: str) -> str:
    """Clean a raw LLM response containing hunks: drop markdown fences,
    discard everything (prose, git ``--- a/... +++ b/...`` preamble) before
    the first ``@@`` header.  Returns '' when no hunk header exists."""
    text = patch_text.strip()
    fence = re.match(r"^```[a-zA-Z0-9_-]*\s*(\n)", text)
    if fence:
        text = text[fence.end():]
    if text.rstrip().endswith("```"):
        text = text[: text.rfind("```")]
    first = text.find("@@")
    if first == -1:
        return ""
    return text[first:].strip("\n")


def _strip_numbered_prefix(line: str) -> str:
    """Strip ``46 |`` artifacts that LLMs copy from numbered context."""
    m = re.match(r'^(\s*)([\-+ ]?)\s*\d+\s*\|\s*', line)
    if m:
        marker = m.group(2) or ' '
        return m.group(1) + marker + line[m.end():]
    return line


def split_patch_hunks(patch_text: str) -> list[tuple[int, list[tuple[str, str]]]]:
    """Parse *patch_text* into ``(start_line, [(op, text), ...])`` hunks.

    Handles fence/prose prefixes (via ``normalize_patch_block``) AND fused
    headers: some models emit several ``@@`` headers with no body lines in
    between — ``---/+++`` text then belongs to the header of the previous
    header.  Re-splitting on every ``@@`` boundary keeps each hunk intact.
    """
    text = normalize_patch_block(patch_text)
    if not text:
        return []
    parts = re.split(r"(?=@@\s*-\d+)", text)
    hunks: list[tuple[int, list[tuple[str, str]]]] = []
    for part in parts:
        part = part.strip("\n")
        if not part:
            continue
        m = re.match(r"@@\s*-(\d+)(?:,(\d+))?(?:\s*\+(\d+)(?:,\d+)?)?\s*@@(.*)", part, re.DOTALL)
        if not m:
            continue
        start = int(m.group(1))
        body = (m.group(4) or "").rstrip("\n")
        chunks: list[tuple[str, str]] = []
        for line in body.split("\n"):
            line = line.rstrip("\r")
            line = _strip_numbered_prefix(line)
            if line.startswith("-"):
                chunks.append(("-", line[1:]))
            elif line.startswith("+"):
                chunks.append(("+", line[1:]))
            elif line.startswith(" "):
                chunks.append((" ", line[1:]))
        if chunks:
            hunks.append((start, chunks))
    return hunks


def apply_patch(patch_text: str, original_lines: list[str]) -> tuple[bool, str]:
    """Apply a unified-diff *patch_text* to *original_lines* (a list of lines
    without trailing newlines).  Returns ``(success, result or error)``.

    Whitespace-tolerant: strips leading/trailing whitespace when comparing
    lines.  Filters broken hunks with incomplete replacements.

    Line convention: git-style patches put the whole line (indentation
    included) right after the +/- marker; LLM-style patches add one padding
    space after the marker.  We detect per-hunk which convention was used by
    comparing a '-' line's leading whitespace against the matching file line,
    and strip that single padding space only when present.
    """
    # Parse hunks: @@ -start,count +start,count @@ ... @@
    # Also lenient: @@ -start @@ (missing +start part)
    hunks = []
    for m in re.finditer(r'@@\s*-(\d+)(?:,\d+)?(?:\s*\+(\d+)(?:,\d+)?)?\s*@@(.*?)(?=@@|\Z)', patch_text, re.DOTALL):
        start = int(m.group(1))
        body = m.group(3).strip('\n')
        chunks: list[tuple[str, str | None]] = []
        for line in body.split('\n'):
            line = line.rstrip('\r')
            line = _strip_numbered_prefix(line)
            if line.startswith('-'): chunks.append(('-', line[1:]))
            elif line.startswith('+'): chunks.append(('+', line[1:]))
            elif line.startswith(' '): chunks.append((' ', line[1:]))
        if chunks:
            hunks.append((start, chunks))
    if not hunks:
        return False, "Could not parse patch"

    # Filter out broken hunks (incomplete: has - but no +, empty + lines, or incomplete lines)
    incomplete_ops = ('=', '+', '-', '*', '/')
    valid_hunks = []
    for start, chunks in hunks:
        has_minus = any(op == '-' for op, _ in chunks)
        has_plus = any(op == '+' for op, _ in chunks)
        if not has_minus and not has_plus:
            continue
        if any(op == '+' and not text.strip() for op, text in chunks):
            continue  # Empty replacement — skip
        # Filter incomplete lines (trailing operators like =, +, -, etc.)
        if any(op == '+' and text.rstrip().endswith(incomplete_ops) for op, text in chunks):
            continue  # Incomplete replacement — skip
        valid_hunks.append((start, chunks))

    if not valid_hunks:
        return False, "No valid hunks in patch"

    # Verify old lines match (whitespace-tolerant)
    # For '-' lines: must match (removed lines)
    # For ' ' context lines: skip if mismatched (LLM often hallucinates context)
    # Record, per hunk, whether its +/- lines carry an LLM padding space.
    hunk_padding: dict[int, bool] = {}
    for start, chunks in valid_hunks:
        idx = start - 1
        filtered_chunks = []
        padded = False
        for op, text in chunks:
            if op == '-':
                if idx < 0 or idx >= len(original_lines):
                    return False, f"Patch mismatch at line {idx+1}: line out of range"
                actual = original_lines[idx].rstrip('\r\n')
                patch_lead = len(text) - len(text.lstrip())
                file_lead = len(actual) - len(actual.lstrip())
                if not padded and patch_lead != file_lead and text.strip() == actual.strip():
                    padded = True
                if actual.strip() != text.strip():
                    return False, f"Patch mismatch at line {idx+1}: expected '{text[:60]}'"
                filtered_chunks.append((op, text))
                idx += 1
            elif op == ' ':
                if idx < 0 or idx >= len(original_lines):
                    idx += 1
                    continue  # Skip out-of-range context
                actual = original_lines[idx].rstrip('\r\n')
                if actual.strip() != text.strip():
                    # Skip mismatched context line — LLM hallucinated it
                    idx += 1
                    continue
                filtered_chunks.append((op, text))
                idx += 1
            else:
                filtered_chunks.append((op, text))
        chunks[:] = filtered_chunks
        hunk_padding[start] = padded

    def _render_plus(start: int, text: str) -> str:
        if hunk_padding.get(start, False) and text.startswith(' '):
            return text[1:]
        return text

    # Apply hunks in reverse order.  Content is applied verbatim (only the
    # marker padding is stripped) — indentation is never rewritten here; the
    # syntax check below rejects patches with broken indentation.
    result = [l + '\n' for l in original_lines]
    for start, chunks in reversed(valid_hunks):
        old_lines = [text for op, text in chunks if op in ('-', ' ')]
        new_lines = []
        i = 0
        while i < len(chunks):
            op, text = chunks[i]
            if op == '-':
                if i + 1 < len(chunks) and chunks[i + 1][0] == '+':
                    # Paired removal + addition: the + line is applied verbatim.
                    new_lines.append(_render_plus(start, chunks[i + 1][1]))
                    i += 2  # Skip both - and + lines
                else:
                    # Unpaired removal (a later hunk has the +): the old line
                    # is dropped, not kept — keeping it would resurrect the
                    # removed line right next to its replacement.
                    i += 1
            elif op == '+':
                new_lines.append(_render_plus(start, text))
                i += 1
            elif op == ' ':
                new_lines.append(text)
                i += 1

        idx = start - 1
        if idx + len(old_lines) <= len(result):
            del result[idx:idx + len(old_lines)]
            for i, text in enumerate(new_lines):
                result.insert(idx + i, text + '\n')

    # Syntax check
    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    tf.write(''.join(result))
    tf.close()
    r = subprocess.run(["python", "-m", "py_compile", tf.name], capture_output=True, text=True)
    os.unlink(tf.name)
    if r.returncode != 0:
        return False, f"Patch breaks syntax: {r.stderr[:200]}"

    return True, ''.join(result)


def _match_score(a: str, b: str) -> float:
    """Similarity in [0, 1] — 1.0 on whitespace-normalized equality."""
    if a.strip() == b.strip():
        return 1.0
    return difflib.SequenceMatcher(None, a.strip(), b.strip()).ratio()


def _find_hunk_anchor(lines: list[str], chunks: list[tuple[str, str]],
                      claimed_start: int) -> int | None:
    """Locate the ('-'/' ') line that the first old line of *chunks* anchors
    to in *lines*, searching a ±60-line window around *claimed_start*.

    Requires an unambiguous best match >= 0.8 that also holds for the whole
    old-line run that follows.  Returns the 0-based index or None.
    """
    old_lines = [text for op, text in chunks if op in ('-', ' ')]
    if not old_lines:
        return None
    center = max(0, (claimed_start or 1) - 1)
    lo = max(0, min(center - 60, len(lines)))
    hi = min(len(lines), center + 60)
    if lo >= hi:
        lo, hi = max(0, len(lines) - 120), len(lines)
    best, best_score = -1, 0.0
    for i in range(lo, hi):
        score = _match_score(lines[i], old_lines[0])
        if score > best_score:
            best, best_score = i, score
    if best_score < 0.8:
        return None
    run = 1
    while run < len(old_lines) and best + run < len(lines):
        if _match_score(lines[best + run], old_lines[run]) < 0.8:
            break
        run += 1
    if run < len(old_lines):
        return None
    return best


def apply_anchored_patch(patch_text: str, original_lines: list[str]) -> tuple[bool, str]:
    """Content-anchored fallback for ``apply_patch``.

    Same safety rules (no removal-only hunks, no empty replacements, final
    syntax gate) but line numbers are only a search-window hint: each hunk's
    old lines are matched against the file text and the edit is applied at
    the anchor, which absorbs wrong ``@@`` numbers, fence-wrapped diffs and
    glued headers.  Falls cheap when the content cannot be anchored
    unambiguously rather than risk a wrong edit.
    """
    incomplete_ops = ('=', '+', '-', '*', '/')
    valid: list[tuple[int, list[tuple[str, str]], bool]] = []
    for claimed, chunks in split_patch_hunks(patch_text):
        has_minus = any(op == '-' for op, _ in chunks)
        has_plus = any(op == '+' for op, _ in chunks)
        if not has_minus and not has_plus:
            continue  # context-only hunk — no edit; do not let it delete
        if any(op == '+' and not text.strip() for op, text in chunks):
            continue
        if any(op == '+' and text.rstrip().endswith(incomplete_ops) for op, text in chunks):
            continue
        valid.append((claimed, chunks))
    if not valid:
        return False, "No valid hunks in patch"

    edits: list[tuple[int, int, list[str]]] = []
    for claimed, chunks in valid:
        anchor = _find_hunk_anchor(original_lines, chunks, claimed)
        if anchor is None:
            return False, (f"Cannot anchor patch content in the file around "
                           f"line {claimed}")
        # Verify the whole old run verbatim (strip-compare) at the anchor and
        # detect the LLM padding-space convention from the first '-'/' ' line.
        idx = anchor
        padded = False
        for op, text in chunks:
            if op == '-':
                actual = original_lines[idx].rstrip('\r\n')
                patch_lead = len(text) - len(text.lstrip())
                file_lead = len(actual) - len(actual.lstrip())
                if not padded and patch_lead != file_lead and actual.strip() == text.strip():
                    padded = True
                if actual.strip() != text.strip():
                    return False, (f"Patch mismatch at line {idx + 1}: "
                                   f"expected '{text[:60]}'")
                idx += 1
            elif op == ' ':
                if idx < len(original_lines) and original_lines[idx].rstrip('\r\n').strip() == text.strip():
                    idx += 1
        old_len = idx - anchor
        # Build the replacement: context (' ') lines are RE-EMITTED FROM THE
        # FILE (they were strip-verified above — the patch text may have lost
        # their indentation, e.g. inlined right after a fused @@ header) and
        # '+' lines come from the patch verbatim (padding stripped).  Removed
        # '-' lines advance the file pointer (they occupied a slot in the source)
        # but emit nothing, so a trailing context line is re-emitted from the
        # correct original line rather than shifting onto a removed line.
        new_lines: list[str] = []
        k = anchor
        for op, text in chunks:
            if op == ' ':
                if k < len(original_lines):
                    new_lines.append(original_lines[k])
                k += 1
            elif op == '-':
                k += 1
            elif op == '+':
                new_lines.append(
                    text[1:] if (padded and text.startswith(' ')) else text
                )
        edits.append((anchor, old_len, new_lines))

    # Apply bottom-up so earlier anchors are never disturbed.
    result = [l + '\n' for l in original_lines]
    for anchor, old_len, new_lines in sorted(edits, key=lambda e: e[0], reverse=True):
        del result[anchor:anchor + old_len]
        for i, text in enumerate(new_lines):
            result.insert(anchor + i, text + '\n')

    # Syntax check.
    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    tf.write(''.join(result))
    tf.close()
    r = subprocess.run(["python", "-m", "py_compile", tf.name], capture_output=True, text=True)
    os.unlink(tf.name)
    if r.returncode != 0:
        return False, f"Patch breaks syntax: {r.stderr[:200]}"

    return True, ''.join(result)