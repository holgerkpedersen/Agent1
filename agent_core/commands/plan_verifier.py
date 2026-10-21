"""Deterministic regression testing for generated plan documents.

The workflow verifies ``analyze`` output claim-by-claim
(:mod:`agent_core.commands.analysis_verifier`) before planning continues —
but ``plan``, ``entities``, and ``taskplan`` docs are written straight from
LLM text.  A plan that references files which do not exist, or a taskplan
whose tasks define colliding symbols, silently poisons every downstream
``implement`` run.

This module runs three deterministic checks over freshly generated docs:

* :func:`verify_plan_doc`     — every backticked path exists in the workspace
  or is explicitly marked/planned as new (``[NEW]`` tag or "create/add"
  wording on the same line); a ``[MODIFY]`` target must already exist.
* :func:`verify_taskplan_doc` — the same path check per numbered task, plus a
  duplicate-definition scan over all *existing* referenced modules in the
  same directory.
* :func:`verify_entities_doc` — every ```python fence parses (``ast.parse``)
  and no top-level name is defined twice across blocks.

:func:`check_doc` additionally runs a structural validity pass over every
doc: a hallucinated tool-result JSON envelope, fabricated tool-call XML,
absolute paths outside the workspace, or a document with no markdown
structure for its kind is flagged.  Observed live: a 1.5B branch model
answered the plan prompt with a ``{"status": ..., "filename": "/home/...",
"content": ...}`` envelope from its training data, and the old gate scored
it clean (no backticked paths) and wrote it as ``project_plan.md``.

Nothing is ever rewritten: mirroring ``analysis_verifier``, unverifiable
claims are listed in an appended ``## Verification Report``.  An existing
report section is stripped before re-checking so repeated gates stay
idempotent and never verify their own findings as claims.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_BACKTICK_RE = re.compile(r"`([^`\n]+)`")
_PY_FENCE_RE = re.compile(r"```python[ \t]*\r?\n(.*?)```", re.DOTALL)
_REPORT_MARKER = "## Verification Report"

#: Extensions treated as file-path claims (plans also mention `.md` outputs).
_CLAIM_EXTS = (".py", ".md", ".toml", ".json", ".yaml", ".yml", ".txt", ".cfg")

#: Line wording that marks a missing path as an intentional new file.
_NEW_INTENT_WORDS = (
    "create",
    "new file",
    "add ",
    "scaffold",
    "introduce",
    "generate",
    "implement the new",
    "boot strap",
    "bootstrap",
)
_MODIFY_INTENT_WORDS = (
    "modify",
    "edit",
    "update",
    "refactor",
    "extend",
    "change",
    "fix",
)

_MAX_REPORT_FINDINGS = 20


@dataclass
class PlanCheckResult:
    """Outcome of one regression check over a generated doc."""

    doc_kind: str
    checked: int = 0
    flagged: int = 0
    findings: list[str] = field(default_factory=list)
    #: Workflow file name of the checked doc (set by :func:`check_doc`).
    doc_name: str = ""

    @property
    def clean(self) -> bool:
        return self.flagged == 0

    def located(self, finding: str) -> str:
        """*finding* prefixed with the doc file name it refers to."""
        return f"{self.doc_name}: {finding}" if self.doc_name else finding


#: Workflow file each doc kind is written to (see ``doc_paths.WORKFLOW_DOC_NAMES``).
_DOC_NAMES = {
    "plan": "project_plan.md",
    "taskplan": "project_tasks.md",
    "entities": "project_entities.md",
}


def _split_report(text: str) -> str:
    """Return *text* without a previously appended Verification Report."""
    idx = text.find(_REPORT_MARKER)
    if idx == -1:
        return text
    base = text[:idx]
    # Drop the trailing ``---`` separator the report itself added.
    return re.sub(r"\n-{3,}\s*$", "", base.rstrip()) + "\n"


def _looks_like_path(token: str) -> bool:
    """True when a backticked token is a file-path claim worth checking."""
    token = token.strip()
    if not token or "*" in token or "?" in token:
        return False
    if token.startswith(("#", "http://", "https://", "./.")):
        return False
    lowered = token.lower()
    return lowered.endswith(_CLAIM_EXTS)


def _norm(path: str) -> str:
    # Strip only a leading "./" prefix. lstrip("./") would also eat the root
    # "/" of an absolute POSIX path and turn it into a workspace-relative one.
    normalized = path.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _resolve(ws: Path, rel: str) -> Path:
    candidate = Path(rel)
    return candidate if candidate.is_absolute() else ws / candidate


@dataclass
class _PathClaim:
    path: str
    line_no: int
    line_text: str


def _extract_path_claims(text: str) -> list[_PathClaim]:
    """Backticked path claims with their line context (reports excluded)."""
    claims: list[_PathClaim] = []
    for line_no, line in enumerate(_split_report(text).splitlines(), 1):
        for m in _BACKTICK_RE.finditer(line):
            token = m.group(1).strip()
            if _looks_like_path(token):
                claims.append(_PathClaim(_norm(token), line_no, line))
    return claims


def _check_path_claim(claim: _PathClaim, ws: Path) -> tuple[bool, str]:
    """Classify one claim: (ok, reason). Missing modify-targets are errors."""
    line = claim.line_text.lower()
    if _resolve(ws, claim.path).exists():
        return True, ""
    has_new_tag = "[new]" in line or "(new)" in line
    has_modify_tag = "[modify]" in line or "(modify)" in line
    if has_modify_tag or any(w in line for w in _MODIFY_INTENT_WORDS):
        return False, "modify target does not exist yet"
    if has_new_tag or any(w in line for w in _NEW_INTENT_WORDS):
        return True, ""  # intentional new file — verified at implement time
    return False, "not found in workspace and not marked as new"


def _top_level_names(py_file: Path) -> tuple[list[str], str | None]:
    """Top-level class/def names of *py_file*, or an error string."""
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError) as exc:
        return [], f"unreadable ({exc.__class__.__name__})"
    names = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return names, None


def _duplicate_definitions(paths: list[str], ws: Path) -> list[str]:
    """Names defined at top level by two+ referenced files in the same dir."""
    seen: dict[tuple[str, str], list[str]] = {}
    problems: list[str] = []
    for rel in dict.fromkeys(paths):
        if not rel.endswith(".py"):
            continue
        abs_path = _resolve(ws, rel)
        if not abs_path.is_file():
            continue
        names, err = _top_level_names(abs_path)
        if err:
            problems.append(f"`{rel}` — {err}")
            continue
        key_dir = rel.rsplit("/", 1)[0] if "/" in rel else ""
        for name in names:
            seen.setdefault((key_dir, name), []).append(rel)
    for (directory, name), rels in sorted(seen.items()):
        if len(rels) > 1:
            problems.append(
                f"`{name}` defined in {' and '.join(f'`{r}`' for r in rels)}"
                f" (directory `{directory or '.'}`) — merge or rename"
            )
    return problems


class PlanVerifier:
    """Comprehensive plan verification class with schema and safety checks."""

    def __init__(self, workspace: Path):
        self.workspace = workspace

    def verify_schema_compliance(self, content: str) -> tuple[bool, list[str]]:
        """Validate the markdown plan against required schema structure."""
        from .plan_schema import validate_plan_markdown

        return validate_plan_markdown(content)

    def verify_affected_files_exist(self, content: str) -> PlanCheckResult:
        """Check that all referenced files in the plan exist or are marked [NEW]."""
        return verify_plan_doc(content, self.workspace)

    def verify_no_unsupported_operations(self, content: str) -> tuple[bool, list[str]]:
        """Block plans containing dangerous operations like 'rm -rf' or 'git reset --hard'."""
        dangerous_patterns = [
            r"rm\s+-rf",
            r"git\s+reset\s+--hard",
            r"git\s+clean\s+-f",
            r"del\s+/[sq]",
            r"format\s+[a-zA-Z]:",
        ]
        violations = []
        for line in content.splitlines():
            for pattern in dangerous_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append(f"Unsupported operation found: {line.strip()}")
        return len(violations) == 0, violations


def verify_plan_doc(text: str, ws: str | Path) -> PlanCheckResult:
    """Regression-check backticked path references in a generated plan."""
    ws_path = Path(ws)
    result = PlanCheckResult("plan")
    for claim in _extract_path_claims(text):
        ok, reason = _check_path_claim(claim, ws_path)
        result.checked += 1
        if not ok:
            result.flagged += 1
            result.findings.append(f"`{claim.path}` — {reason} (line {claim.line_no})")
    return result


def verify_taskplan_doc(text: str, ws: str | Path) -> PlanCheckResult:
    """Regression-check a generated taskplan: paths + duplicate definitions."""
    ws_path = Path(ws)
    result = PlanCheckResult("taskplan")
    claims = _extract_path_claims(text)
    for claim in claims:
        ok, reason = _check_path_claim(claim, ws_path)
        result.checked += 1
        if not ok:
            result.flagged += 1
            result.findings.append(f"`{claim.path}` — {reason} (line {claim.line_no})")
    existing = [c.path for c in claims if c.path.endswith(".py")]
    for problem in _duplicate_definitions(existing, ws_path):
        result.flagged += 1
        result.findings.append(problem)
    if existing:
        result.checked += len(existing)
    return result


def verify_entities_doc(text: str, ws: str | Path) -> PlanCheckResult:
    """Regression-check a generated entities doc: fences parse, names unique.

    Every finding's ``(line N)`` is the line number in the doc itself, not
    relative to the fenced block, so it can be opened directly.
    """
    del ws  # entities are checked syntactically, not against the workspace
    result = PlanCheckResult("entities")
    body = _split_report(text)
    seen_names: dict[str, list[int]] = {}
    for idx, m in enumerate(_PY_FENCE_RE.finditer(body), 1):
        block = m.group(1)
        # Doc line of the block's first body line (1-based).
        block_line = body.count("\n", 0, m.start(1)) + 1
        result.checked += 1
        try:
            tree = ast.parse(block)
        except SyntaxError as exc:
            result.flagged += 1
            at = block_line + (exc.lineno or 1) - 1
            result.findings.append(
                f"python block #{idx} — SyntaxError: {exc.msg} (line {at})"
            )
            continue
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names = [(node.name, node.lineno)]
            elif isinstance(node, ast.Assign):
                names = [
                    (t.id, node.lineno) for t in node.targets if isinstance(t, ast.Name)
                ]
            else:
                names = []
            for name, lineno in names:
                seen_names.setdefault(name, []).append(block_line + lineno - 1)
    for name, where in sorted(seen_names.items()):
        if len(where) > 1:
            result.flagged += 1
            spots = ", ".join(f"line {n}" for n in where)
            result.findings.append(
                f"`{name}` defined {len(where)}× across entity blocks ({spots}) — keep one"
            )
    return result


_CHECKERS = {
    "plan": verify_plan_doc,
    "taskplan": verify_taskplan_doc,
    "entities": verify_entities_doc,
}


#: Envelope keys of a hallucinated tool-result JSON document.
_JSON_ENVELOPE_KEYS = ("filename", "start_line", "total_lines", "truncated", "type")

#: Pseudo-tool XML a weak model writes INSTEAD of producing the document
#: (same shape as ``speculate_cmd._FAKE_TOOL_XML_RE``).
_FABRICATED_XML_RE = re.compile(
    r"<\s*/?\s*(?:definitions|references|search|web_search|read|read_skill|"
    r"list_files|diff|tests|file|result|tool_result)\b[^>]*\b"
    r"(?:path|query|symbol|file|max_results|line)\s*=",
    re.IGNORECASE,
)

#: Absolute filesystem paths (POSIX system dirs or Windows drives).
_ABS_PATH_RE = re.compile(
    r"(?<![\w])((?:[A-Za-z]:[\\/]|/(?:home|Users|usr|var|etc|opt|tmp)/)"
    r"[^\s`\"'<>\])|,]+)"
)

_HEADING_RE = re.compile(r"^#{1,6} ", re.MULTILINE)
_LIST_RE = re.compile(r"^(?:[-*+] |\d+[.)] )", re.MULTILINE)


def _json_tool_envelope(text: str) -> bool:
    """True when *text* is a whole-document tool-result JSON envelope."""
    stripped = text.strip()
    if not stripped.startswith("{") or '"content"' not in stripped:
        return False
    try:
        obj = json.loads(stripped)
    except ValueError:
        return any(k in stripped for k in _JSON_ENVELOPE_KEYS)
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("content"), str)
        and any(k in obj for k in _JSON_ENVELOPE_KEYS)
    )


def _foreign_abs_paths(text: str, ws: Path) -> list[str]:
    """Absolute paths in *text* that exist neither on disk nor in *ws*."""
    bad: list[str] = []
    ws_resolved = ws.resolve()
    for m in _ABS_PATH_RE.finditer(text):
        raw = m.group(1)
        try:
            candidate = Path(raw)
            if candidate.exists():
                continue
            candidate.resolve().relative_to(ws_resolved)
            continue  # inside the workspace (new or existing)
        except (OSError, ValueError):
            pass
        bad.append(raw)
    return bad


def _structure_findings(doc_kind: str, text: str, ws: Path) -> list[str]:
    """Deterministic validity findings shared by every doc kind."""
    findings: list[str] = []
    body = _split_report(text)
    if _json_tool_envelope(text):
        findings.append(
            f"document is a hallucinated tool-result JSON envelope, not a "
            f"{doc_kind} document — regenerate"
        )
    if _FABRICATED_XML_RE.search(body):
        findings.append("fabricated tool-call XML found — not a real document")
    for bad in _foreign_abs_paths(body, ws):
        findings.append(f"absolute path outside the workspace: `{bad}`")
    stripped = body.strip()
    if doc_kind == "entities":
        if not any(m.group(1).strip() for m in _PY_FENCE_RE.finditer(stripped)):
            findings.append("no ```python entity blocks found")
    elif not (_HEADING_RE.search(stripped) or _LIST_RE.search(stripped)):
        findings.append(
            f"no markdown headings or list items found — not a {doc_kind} document"
        )
    return findings


def check_doc(doc_kind: str, text: str, ws: str | Path) -> PlanCheckResult:
    """Dispatch to the checker registered for *doc_kind*.

    The structural validity pass (:func:`_structure_findings`) runs for
    every kind, so a fabricated or malformed document can never score
    clean just because it contains no path claims to check.
    """
    try:
        checker = _CHECKERS[doc_kind]
    except KeyError:
        raise ValueError(f"unknown doc kind: {doc_kind!r}") from None
    ws_path = Path(ws)
    result = checker(text, ws_path)
    for finding in _structure_findings(doc_kind, text, ws_path):
        result.flagged += 1
        result.findings.insert(0, finding)
    result.doc_name = _DOC_NAMES.get(doc_kind, doc_kind)
    return result


def report_section(result: PlanCheckResult) -> str:
    """Render the appended Verification Report for a flagged result."""
    verified = max(0, result.checked - result.flagged)
    lines = [
        f"- {result.doc_kind.capitalize()} claims checked: {result.checked}"
        f" — {verified} verified, {result.flagged} flagged."
    ]
    shown = result.findings[:_MAX_REPORT_FINDINGS]
    lines += [f"- [UNVERIFIED] {result.located(f)}" for f in shown]
    hidden = len(result.findings) - len(shown)
    if hidden > 0:
        lines.append(f"- ... {hidden} more finding(s)")
    return f"\n\n---\n\n{_REPORT_MARKER}\n\n" + "\n".join(lines) + "\n"


def apply_report(text: str, result: PlanCheckResult) -> str:
    """Append the report for a flagged *result* (replacing any stale one)."""
    if result.clean:
        return text
    return _split_report(text) + report_section(result)


def summarize(result: PlanCheckResult, label: str) -> None:
    """Print the one-line summary (and findings) for a finished check."""
    if result.checked or result.flagged:
        status = "clean" if result.clean else f"{result.flagged} flagged"
        print(f"  [{label}] Regression-checked {result.checked} claims ({status})")
    for finding in result.findings[:10]:
        print(f"    - [UNVERIFIED] {result.located(finding)}")
