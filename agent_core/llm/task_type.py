"""Task-type inference — keyword/heuristic classifier for the meta-policy loop.

Bridges free-text user input to :class:`agent_core.llm.llm_types.TaskType`,
the key of ``TASK_PROFILE_MAP`` (agent_core/llm/config.py:8), which maps each
task type to a sampling profile.  This is the **only** consumer of that map in
production (before this module, ``TASK_PROFILE_MAP`` was built but never
called).

Rule sources (docstring contract per task B1): ordered keyword buckets matched
against the lowercased user input —

* ``FIX``      — fix / error / bug / traceback / regression / failing
* ``ANALYZE``  — test / pytest / coverage, and analyze / investigate / explain
  / why / read
* ``OPTIMIZE`` — optimize / slow / latency
* ``IMPLEMENT``— implement / add / create / feature / build
* ``WORKFLOW`` — workflow / pipeline / chain
* ``PERF_TUNING`` — tune / profile / benchmark
* default      — ``ANALYZE``

The function is **total over the keys of ``TASK_PROFILE_MAP``**: its return
value is always a ``TaskType`` that the map defines (the map covers every
member of the enum today, but the default ANCHOR is chosen independently of
that so a newly added enum member cannot KeyError the caller).
"""

from __future__ import annotations

from agent_core.llm.llm_types import TaskType

__all__ = ["TaskType", "infer_task_type"]


#: Ordered (keywords, task type) rules — first match wins, so more specific
#: buckets (fix/test) precede the generic ones (analyze/implement).
_RULES: tuple[tuple[tuple[str, ...], TaskType], ...] = (
    # Fixes beat everything: a traceback is a fix request even if it also
    # mentions tests ("the test suite fails") or performance.
    (("fix", "error", "bug", "traceback", "regression", "failing", "crash"),
     TaskType.FIX),
    # Test/coverage work is analysis of the suite, checked BEFORE the generic
    # analyze bucket only in ordering terms (both map differently: see map).
    (("test", "pytest", "coverage"), TaskType.ANALYZE),
    (("optimize", "slow", "latency", "speed up", "performance"),
     TaskType.OPTIMIZE),
    (("implement", "add ", "create", "feature", "build", "write "),
     TaskType.IMPLEMENT),
    (("workflow", "pipeline", "chain"), TaskType.WORKFLOW),
    (("tune", "profile", "benchmark"), TaskType.PERF_TUNING),
    (("analyze", "investigate", "explain", "why", "read"), TaskType.ANALYZE),
)


def infer_task_type(
    user_input: str,
    *,
    mutated_files: list[str] | None = None,
    command: str | None = None,
) -> TaskType:
    """Classify *user_input* into a :class:`TaskType` (never raises).

    Keyword/heuristic rules as documented in the module docstring; matched
    against the lowercased input.  *mutated_files* and *command* are accepted
    for signature stability with the call site (a turn that already mutated
    files under a known REPL command is a strong hint) but the keyword rules
    alone are sufficient — they are applied first and decide the default.

    Total over ``TASK_PROFILE_MAP`` keys: the fallback is ``ANALYZE``, which
    the map defines.
    """
    text = (user_input or "").lower()
    for keywords, task_type in _RULES:
        if any(keyword in text for keyword in keywords):
            return task_type
    return TaskType.ANALYZE
