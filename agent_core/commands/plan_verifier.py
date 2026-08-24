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

Nothing is ever rewritten: mirroring ``analysis_verifier``, unverifiable
claims are listed in an appended ``## Verification Report``.  An existing
report section is stripped before re-checking so repeated gates stay
idempotent and never verify their own findings as claims.
"""
from __future__ import annotations

import ast
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
    "create", "new file", "add ", "scaffold", "introduce", "generate",
    "implement the new", "boot strap", "bootstrap",
)
_MODIFY_INTENT_WORDS = ("modify", "edit", "update", "refactor", "extend",
                        "change", "fix")

_MAX_REPORT_FINDINGS = 20


@dataclass
class PlanCheckResult:
    """Outcome of one regression check over a generated doc."""

    doc_kind: str
    checked: int = 0
    flagged: int = 0
    findings: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.flagged == 0


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
    return path.strip().replace("\\", "/").lstrip("./")


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
        node.name for node in tree.body
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
    """Regression-check a generated entities doc: fences parse, names unique."""
    del ws  # entities are checked syntactically, not against the workspace
    result = PlanCheckResult("entities")
    blocks = _PY_FENCE_RE.findall(_split_report(text))
    seen_names: dict[str, int] = {}
    for idx, block in enumerate(blocks, 1):
        result.checked += 1
        try:
            tree = ast.parse(block)
        except SyntaxError as exc:
            result.flagged += 1
            result.findings.append(
                f"python block #{idx} — SyntaxError: {exc.msg} (line {exc.lineno})"
            )
            continue
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                seen_names[node.name] = seen_names.get(node.name, 0) + 1
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        seen_names[tgt.id] = seen_names.get(tgt.id, 0) + 1
    for name, count in sorted(seen_names.items()):
        if count > 1:
            result.flagged += 1
            result.findings.append(
                f"`{name}` defined {count}× across entity blocks — keep one"
            )
    return result


_CHECKERS = {
    "plan": verify_plan_doc,
    "taskplan": verify_taskplan_doc,
    "entities": verify_entities_doc,
}


def check_doc(doc_kind: str, text: str, ws: str | Path) -> PlanCheckResult:
    """Dispatch to the checker registered for *doc_kind*."""
    try:
        checker = _CHECKERS[doc_kind]
    except KeyError:
        raise ValueError(f"unknown doc kind: {doc_kind!r}") from None
    return checker(text, ws)


def report_section(result: PlanCheckResult) -> str:
    """Render the appended Verification Report for a flagged result."""
    verified = result.checked - result.flagged
    lines = [
        f"- {result.doc_kind.capitalize()} claims checked: {result.checked}"
        f" — {verified} verified, {result.flagged} flagged."
    ]
    shown = result.findings[:_MAX_REPORT_FINDINGS]
    lines += [f"- [UNVERIFIED] {f}" for f in shown]
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
    if result.checked:
        status = "clean" if result.clean else f"{result.flagged} flagged"
        print(f"  [{label}] Regression-checked {result.checked} claims ({status})")
    for finding in result.findings[:10]:
        print(f"    - [UNVERIFIED] {finding}")
