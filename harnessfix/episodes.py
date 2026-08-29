"""Phase 5 - episodic memory: successful runs as retrievable episodes.

Compiles the trace corpus into compact ``Episode`` records that capture what
a successful run did, so a later fix/implement prompt can be shown analogous
past resolutions ("few successful examples").

Success definition is reused verbatim from ``corpus._is_failed_trace``: a run
is successful when it is NOT failed, i.e. no ``tool_error`` kind, a ``loop_end``
present (or <3 events), and either ``outcome == "completed"`` or it delivered
a substantive final answer (guard-terminated runs that still answered count
as success).  This is the corpus's own notion of "did the task" - we do NOT
add a separate "tests green" gate, because the trace has no first-class
verification signal (a ``run`` that prints a Traceback is a ``tool_result``,
not a ``tool_error`` kind).

Episodes are intentionally concise: we store a short problem snippet, the file
stems and error classes touched, an action summary (tool + target, never full
file contents), the outcome and the final answer excerpt.  Embedding the full
episode would be token-bloated and dominated by boilerplate.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .corpus import MIN_ACTIVITY_EVENTS, _is_failed_trace
from .htir import TraceGraph, compile_trace
from .reader import TraceValidationError

_PROBLEM_CAP = 400
_ACTION_CAP = 600
_ANSWER_CAP = 400
_ACTION_KINDS = ("tool_call", "tool_result", "tool_error")
_MUTATING_TOOLS = frozenset({"write", "edit", "apply_patch", "fix", "create_file"})


@dataclass(frozen=True)
class Episode:
    """One successful past run, summarised for retrieval + display."""

    task_id: str
    problem: str
    file_stems: tuple[str, ...]
    error_classes: tuple[str, ...]
    actions_summary: str
    outcome: str
    final_answer: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def index_text(self) -> str:
        """Concise string used for embedding / similarity matching."""
        parts = [self.problem[:_PROBLEM_CAP]]
        if self.error_classes:
            parts.append("errors: " + " ".join(self.error_classes))
        if self.file_stems:
            parts.append("files: " + " ".join(self.file_stems))
        parts.append("outcome: " + self.outcome)
        return "\n".join(p for p in parts if p)

    def render(self, cap: int = 300) -> str:
        """Human-readable episode block for prompt injection."""
        lines = [f"[{self.task_id[:8]}] problem: {self.problem[:cap]}"]
        if self.error_classes:
            lines.append(f"  errors: {', '.join(self.error_classes)}")
        if self.file_stems:
            lines.append(f"  files: {', '.join(self.file_stems)}")
        lines.append(f"  actions: {self.actions_summary[:cap]}")
        lines.append(f"  outcome: {self.outcome}")
        if self.final_answer:
            lines.append(f"  answer: {self.final_answer[:cap]}")
        return "\n".join(lines)


def _norm_path(p: str) -> str:
    return str(p).replace("\\", "/").lower()


def _paths_from_args(args: Any) -> list[str]:
    if not isinstance(args, dict):
        return []
    out: list[str] = []
    for key in ("path", "file", "filename", "target", "fpath"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            out.append(value)
        elif isinstance(value, list):
            out.extend(v for v in value if isinstance(v, str) and v.strip())
    return out


def _stems_from(paths: list[str]) -> list[str]:
    stems: list[str] = []
    for p in paths:
        name = os.path.basename(_norm_path(p))
        if "." in name:
            name = name.rsplit(".", 1)[0]
        if name and name not in stems:
            stems.append(name)
    return stems


_ERROR_RE = __import__("re").compile(
    r"(ImportError|ModuleNotFoundError|NameError|AttributeError|TypeError|"
    r"ValueError|KeyError|IndexError|SyntaxError|FileNotFoundError|"
    r"PermissionError|RuntimeError|AssertionError|ZeroDivisionError|"
    r"RecursionError|UnicodeDecodeError|OverflowError|NotImplementedError)"
)


def _error_classes_from(text: str) -> list[str]:
    if not text:
        return []
    found = _ERROR_RE.findall(text)
    # de-duplicate, preserve order
    out: list[str] = []
    for e in found:
        if e not in out:
            out.append(e)
    return out


def extract_episode(graph: TraceGraph) -> Episode | None:
    """Summarise a *successful* trace graph into an Episode, or None.

    Returns None for failed traces (delegated to the caller via
    ``_is_failed_trace``) or for traces with no usable problem text.
    """
    task_id = graph.task_id

    problem = ""
    for s in graph.steps:
        if s.kind == "task_begin":
            problem = str(s.payload.get("user_input", "")).strip()
            break
    if not problem:
        problem = f"(task {task_id[:8]})"

    outcome = ""
    for s in graph.steps:
        if s.kind == "loop_end":
            outcome = str(s.payload.get("outcome", "completed"))
    if not outcome:
        outcome = "completed" if len(graph.steps) < MIN_ACTIVITY_EVENTS else "interrupted"

    # file stems + error classes + action summary
    paths: list[str] = []
    action_lines: list[str] = []
    error_classes: list[str] = []
    seen_actions: set[str] = set()
    for s in graph.steps:
        if s.kind not in _ACTION_KINDS:
            continue
        try:
            args = json.loads(s.payload.get("args_hash", "{}"))
        except (ValueError, TypeError):
            args = {}
        if not isinstance(args, dict):
            args = {}
        step_paths = _paths_from_args(args)
        if s.payload.get("affected_files"):
            step_paths = step_paths + [str(f) for f in s.payload["affected_files"]]
        paths.extend(step_paths)

        tool = str(s.payload.get("tool", "?"))
        note = ""
        if s.kind == "tool_error":
            msg = f"{s.payload.get('exception', '')}: {s.payload.get('message', '')}"
            error_classes.extend(_error_classes_from(msg))
            note = " (error)"
        elif s.kind == "tool_result" and s.payload.get("result"):
            error_classes.extend(_error_classes_from(str(s.payload["result"])))
        target = os.path.basename(_norm_path(step_paths[0])) if step_paths else ""
        verb = "edit" if tool == "edit" else tool
        line = f"{verb} {target}{note}" if target else verb
        if line not in seen_actions:
            seen_actions.add(line)
            action_lines.append(line)

    # final answer = last llm_response with substantive text
    final_answer = ""
    for s in graph.steps:
        if s.kind == "llm_response":
            txt = str(s.payload.get("text", "")).strip()
            if len(txt) >= 80:
                final_answer = txt

    file_stems = tuple(_stems_from(paths))
    error_classes = tuple(dict.fromkeys(error_classes))

    return Episode(
        task_id=task_id,
        problem=problem[:_PROBLEM_CAP],
        file_stems=file_stems,
        error_classes=error_classes,
        actions_summary="; ".join(action_lines)[:_ACTION_CAP],
        outcome=outcome,
        final_answer=final_answer[:_ANSWER_CAP],
        meta={"n_steps": len(graph.steps)},
    )


_EPISODE_CACHE: tuple[tuple[tuple[str, int, int], ...], list[Episode]] | None = None


def successful_episodes(workspace: str) -> list[Episode]:
    """Compile every successful trace under ``workspace/reports/traces``.

    Cached per-process, keyed on a signature of the trace directory so new
    runs after process start are picked up on the next call (mirrors
    harnessfix/history._trace_events).  Fail-open: a corrupt trace is skipped.
    """
    global _EPISODE_CACHE
    trace_dir = Path(workspace) / "reports" / "traces"
    if trace_dir.is_dir():
        sig = tuple(
            sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in trace_dir.glob("*.jsonl"))
        )
    else:
        sig = ()

    if _EPISODE_CACHE is not None and _EPISODE_CACHE[0] == sig:
        return _EPISODE_CACHE[1]

    episodes: list[Episode] = []
    if trace_dir.is_dir():
        for path in sorted(trace_dir.glob("*.jsonl")):
            try:
                graph = compile_trace(path)
            except TraceValidationError:
                continue
            if _is_failed_trace(graph):
                continue
            ep = extract_episode(graph)
            if ep is not None:
                episodes.append(ep)
    _EPISODE_CACHE = (sig, episodes)
    return episodes


def clear_episode_cache() -> None:
    """Drop the process-level episode index (used by tests)."""
    global _EPISODE_CACHE
    _EPISODE_CACHE = None


__all__ = [
    "Episode",
    "extract_episode",
    "successful_episodes",
    "clear_episode_cache",
]
