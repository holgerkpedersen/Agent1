"""Tool calling loop orchestrator for LLM conversations."""
import enum
import json
import sys
import time
from typing import Any, Awaitable, Callable

from harnessfix.tracing import (
    GUARD_BUDGET,
    GUARD_DEADLINE,
    GUARD_NO_MUTATION,
    GUARD_STUCK,
    KIND_GUARD_TRIGGERED,
    KIND_LLM_RESPONSE,
    KIND_LOOP_END,
    KIND_STEP_START,
    KIND_TOOL_CALL,
    KIND_TOOL_ERROR,
    KIND_TOOL_RESULT,
    LAYER_LIFECYCLE,
    LAYER_OBSERVABILITY,
    LAYER_TOOL_INTERFACE,
    RESULT_CAP,
    TEXT_CAP,
    TraceSink,
    truncate,
)

try:
    from agent_core.colors import cyan, green, red, yellow, magenta, gray, colorize_result
except Exception:  # pragma: no cover - colors degrade gracefully if unavailable
    def _identity(text: str, *, bold: bool = False) -> str:  # noqa: ARG001
        return text

    cyan = green = red = yellow = magenta = gray = _identity

    def colorize_result(result: str) -> str:
        return result

#: How much tool-call activity is printed to the end user during NLP turns.
class DisplayMode(str, enum.Enum):
    VERBOSE = "verbose"   # every call + full (truncated) result shown
    CLEAN = "clean"      # reason line per call; results summarized for display only
    QUIET = "quiet"      # tool calls/results hidden; only final answer printed

#: Injected into the history when only a few iterations remain, so the model
#: starts wrapping up instead of discovering the cap when it is too late.
_DEADLINE_NOTE = (
    "BUDGET WARNING: you have only {remaining} tool call(s) left before the "
    "loop stops. Make each call count — prefer a single high-value action "
    "(read the key file, run the verification) over further exploration. "
    "After that you MUST give your final answer in text."
)

#: Appended after the cap is hit while tool calls were still pending, so the
#: final answer is always produced from the gathered tool results.
_FORCED_SYNTHESIS_NOTE = (
    "The tool budget is exhausted and no more tools are available. "
    "Based on everything you have already read and gathered, give your final "
    "answer to the user's request NOW. Do not call any tools. If the task is "
    "incomplete, report exactly what is missing and what you would need."
)

#: Used when the model repeats the same tool call three times in a row: the
#: loop stops immediately and forces a text answer instead of burning budget.
_STUCK_SYNTHESIS_NOTE = (
    "You have now repeated the same tool call without making progress, so the "
    "tool loop is stopping. Do NOT call any more tools. Using only what you "
    "have already read in this conversation, give your final answer to the "
    "user's request now — report what you found, and if you lack information, "
    "state exactly what is missing and what you would need."
)

#: Used when a forced synthesis call still comes back empty (large contexts
#: make models occasionally return nothing): one explicit second chance.
_FORCED_SYNTHESIS_RETRY = (
    "Your previous response was empty. You MUST answer in text now — no tools "
    "are available. State the conclusion of your analysis in a few sentences, "
    "even if it is partial or only a summary of what you could not do."
)

#: Tools that count as making progress (they change the workspace).
MUTATING_TOOLS = frozenset({"write", "edit", "fix"})

_NO_PROGRESS_NUDGE = (
    "NOTE: You have now made {count} calls without discovering anything new "
    "or changing any file — you are repeating or circling. Either explore "
    "something NEW (a different file, search, directory, or command), take "
    "action with write/edit/fix, or give your final answer now."
)

_NO_PROGRESS_FORCE = (
    "You have made {count} calls without discovering anything new or changing "
    "any file, so the tool loop is stopping. Do NOT call any more tools. "
    "Give your final answer now — report what you found and what you would "
    "change."
)

#: Result prefixes that signal a path does not exist in the workspace (the agent's
#: read_file/write_file/apply_patch return these verbatim).  These are exactly what
#: laguna-s-2.1-style models see when they cite an invented path instead of checking
#: it exists — without this, the model loops retrying the same dead path forever.
_PATH_MISS_PREFIXES = ("File not found:", "Error reading file:")

#: Steering note injected with a parent-directory listing when a tool result shows
#: a missing path — supplies the simple existence check the model lacks so it can
#: progress instead of getting stuck on the same non-existent path (decision #035).
_PATH_RECOVERY_NOTE = (
    "PATH RECOVERY: The requested path does not exist. Parent directory listing:\n"
    "{listing}\n\nRead or use a path from this listing instead — do NOT retry the\n"
    "same non-existent path."
)

#: Tool-level consecutive-failure guard: when the same tool returns empty or
#: error results N times in a row (even with different arguments), the model
#: is stuck in a variant-repeat loop (e.g. grep with slightly different
#: patterns that all fail).  This note breaks the cycle by forcing a
#: strategy change.  Decision #061 — laguna-s-2.1 repeated grep variants.
_TOOL_CONSECUTIVE_FAILURE_LIMIT = 4
_TOOL_CONSECUTIVE_FAILURE_NOTE = (
    "You have now called {tool} {count} times in a row with no useful results "
    "(all returned empty or errors). The current approach is not working. "
    "STOP calling {tool} — try a fundamentally different strategy: use a "
    "different tool (read, bash, list_files), change the search scope, or "
    "give your final answer with what you already know."
)


#: Console display cap for [result] lines. The model ALWAYS receives the full
#: result; this only bounds what the human observer sees in the REPL (was 200,
#: which made read/search output look cut off).
_RESULT_DISPLAY_LIMIT = 1500

#: Tools whose result a missing-path recovery can meaningfully follow up on.
_PATH_SENSITIVE_TOOLS = frozenset({"read", "edit", "write"})

#: Trace guard label for the path-existence recovery (decision #035). Defined here
#: because harnessfix.tracing does not expose a dedicated enum value.
GUARD_PATH_MISS = "path_miss"


def _is_path_miss(result_str: str) -> bool:
    """True when *result_str* reports that the requested path does not exist."""
    return any(result_str.startswith(p) for p in _PATH_MISS_PREFIXES)


def _parent_dir_of(args: dict[str, Any]) -> str:
    """Derive a parent directory to list from tool args (``path``/``file``)."""
    import pathlib

    raw = ""
    if isinstance(args, dict):
        for key in ("path", "file"):
            val = args.get(key)
            if isinstance(val, str) and val:
                raw = val
                break
    parent = str(pathlib.PurePath(raw).parent) if raw else "."
    return parent


async def _recover_path_miss(
    execute_tool_fn: Callable[[str, dict[str, Any]], Awaitable[str]],
    args: dict[str, Any],
    result_str: str,
) -> tuple[str, bool]:
    """Attach a PATH RECOVERY note with the parent directory listing.

    Returns ``(augmented_result, discovered)`` — ``discovered`` is True when a
    fresh directory listing was obtained (counts as progress so the stuck guard
    does not fire on retrying an invented path).  Any failure still yields a
    textual recovery hint without the listing rather than leaving the model dead.
    """
    parent = _parent_dir_of(args)
    listing = ""
    discovered = False
    try:
        listing = await execute_tool_fn("list_files", {"path": str(parent)})
        if not _is_path_miss(listing):
            discovered = True
    except Exception as exc:  # pragma: no cover - defensive, never raises
        listing = f"(listing failed: {exc})"
    note = _PATH_RECOVERY_NOTE.format(
        listing=truncate(listing or "(empty)", RESULT_CAP) if listing else "(empty)"
    )
    return f"{result_str}\n\n{note}", discovered


class ToolLoopRunner:
    """Orchestrates tool calling loop with LLM.

    Extracted from LLMClient.chat_with_tool_loop to separate tool orchestration
    from LLM communication. This enables testing tool logic without a running LLM.
    """

    def __init__(
        self,
        max_iterations: int = 150,
        deadline_window: int = 2,
        no_mutation_limit: int = 30,
        force_after_no_mutation: int = 50,
        display_mode: DisplayMode | str | None = None,
        trace: TraceSink | None = None,
        effects_fn: Callable[[str, dict[str, Any]], list[str]] | None = None,
    ):
        self.max_iterations = max_iterations
        #: Number of final iterations where the model is warned (and
        #: steered) toward producing a text answer before the cap hits.
        self.deadline_window = max(1, min(deadline_window, max_iterations))
        #: Progress guard: after this many consecutive non-mutating tool calls
        #: a steering note is injected; the count resets on write/edit/fix.
        self.no_mutation_limit = max(1, no_mutation_limit)
        self.force_after_no_mutation = max(self.no_mutation_limit + 1, force_after_no_mutation)
        #: Console display mode for this run. Defaults to VERBOSE so existing
        #: behaviour (and tests that assert on history/final_text, not stdout)
        #: is unchanged when the caller does not specify a mode.
        if isinstance(display_mode, DisplayMode):
            self.display_mode = display_mode
        elif isinstance(display_mode, str):
            try:
                self.display_mode = DisplayMode(display_mode.lower())
            except ValueError:
                self.display_mode = DisplayMode.VERBOSE
        else:
            self.display_mode = DisplayMode.VERBOSE
        #: Optional trace sink (harnessfix.tracing.TraceWriter).  When None the
        #: loop emits nothing and behaves byte-identically to before tracing.
        self._trace = trace
        #: Optional callback returning the files a tool call affected
        #: (self-improvement: files-affected recording).  Only consulted when
        #: a trace sink is attached — invisible unless tracing (decision #048).
        self._effects_fn = effects_fn
        #: How this run ended: "answer" (model answered in text), "cap"
        #: (iteration cap hit), "stuck" (repeated identical calls), or
        #: "no_progress" (too many calls without modifying any file).
        self.termination_reason: str = "answer"
        #: Observability stats for the final-answer fallback: what the run
        #: actually did when the model produced no usable answer.
        self.tool_calls_made: int = 0
        self.tools_used: dict[str, int] = {}
        self.last_tool_call: str = ""
        self.iterations_used: int = 0
        #: Keys of calls that already counted as discovery this run — a
        #: repeated key is a stuck signal, not progress.
        self._seen_progress_keys: set[Any] = set()
        #: Files mutated (write/edit/fix) this run, for the abandonment-resume
        #: protocol (repair abandonment-resume-protocol): when the run ends
        #: non-completed after changes, the next turn is told what was touched
        #: so it resumes instead of restarting (decision #052).  Populated from
        #: the trace effects callback; empty unless a sink is attached.
        self._mutated_files: set[str] = set()

    async def run(
        self,
        messages: list[dict[str, Any]],
        llm_chat_fn: Callable[[list[dict[str, Any]], list[dict[str, Any]]], Awaitable[tuple[str, list[dict[str, Any]]]]],
        execute_tool_fn: Callable[[str, dict[str, Any]], Awaitable[str]],
        tools: list[dict[str, Any]] | None = None,
        seen_calls: dict[tuple[str, str], int] | None = None,
        trace: TraceSink | None = None,
        effects_fn: Callable[[str, dict[str, Any]], list[str]] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Run conversation with automatic tool calling loop.

        Args:
            messages: Initial conversation messages
            llm_chat_fn: Async function that sends messages to LLM and returns
                        (response_text, updated_messages_with_reasoning)
            execute_tool_fn: Async function that executes a tool and returns result
            tools: Optional tool schemas (uses default if None)
            seen_calls: Optional shared registry (call_key -> executions) across
                        chained runs, so a repeated probe cannot hide behind a
                        fresh run's duplicate counter.
            trace: Optional trace sink (harnessfix.tracing.TraceWriter) that
                        receives one JSON event per loop step.  None = untraced.
            effects_fn: Optional callback ``(tool_name, args) -> [paths]`` listing
                        the files a tool call affected, attached to the
                        tool_result/tool_error events.  Only invoked when a
                        trace sink is attached.

        Returns:
            Tuple of (final_text, updated_messages)

        The loop never ends without a text answer: with ``deadline_window``
        iterations left a budget warning is injected, and if the iteration cap
        is hit while tool calls are still pending, one final tool-less call
        forces the model to synthesize the answer from the gathered results.
        """
        self._trace = self._trace if trace is None else trace
        self._effects_fn = self._effects_fn if effects_fn is None else effects_fn
        try:
            return await self._run_traced(
                messages, llm_chat_fn, execute_tool_fn, tools, seen_calls
            )
        finally:
            # Trace completeness: a loop_end event is ALWAYS written, even when
            # an exception escapes the loop (outcome becomes "error").
            self._emit_loop_end()

    def _emit(self, kind: str, layer: str, **fields: Any) -> None:
        """Emit one trace event when a sink is attached; never raises."""
        if self._trace is not None:
            self._trace.emit({"kind": kind, "layer": layer, **fields})

    def _collect_effects(
        self, tool_name: str, args: dict[str, Any]
    ) -> list[str]:
        """Ask the caller which files this tool call affected.

        Only invoked when a trace sink is attached and an ``effects_fn`` was
        supplied (decision #048: instrumentation invisible unless tracing).
        Failures degrade to an empty list — never raise into the loop.
        """
        if self._trace is None or self._effects_fn is None:
            return []
        try:
            return [str(p) for p in (self._effects_fn(tool_name, args) or [])]
        except Exception:
            return []

    def _emit_loop_end(self) -> None:
        if self._trace is None:
            return
        outcome = {
            "answer": "completed",
            "cap": "budget_exhausted",
            "stuck": "stuck",
            "no_progress": "no_progress",
        }.get(self.termination_reason, "error")
        if sys.exc_info()[0] is not None:
            outcome = "error"
        self._emit(
            KIND_LOOP_END,
            LAYER_LIFECYCLE,
            outcome=outcome,
            termination_reason=self.termination_reason,
        )

    async def _run_traced(
        self,
        messages: list[dict[str, Any]],
        llm_chat_fn: Callable[[list[dict[str, Any]], list[dict[str, Any]]], Awaitable[tuple[str, list[dict[str, Any]]]]],
        execute_tool_fn: Callable[[str, dict[str, Any]], Awaitable[str]],
        tools: list[dict[str, Any]] | None,
        seen_calls: dict[tuple[str, str], int] | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Body of run(): the tool-calling loop with trace event emission."""
        if not tools:
            tools = []
        # Discovery tracking is per-run: a fresh run has a fresh "known" set.
        self._seen_progress_keys.clear()

        all_text_parts = []
        current_messages = [dict(m) for m in messages]
        deadline_injected = False
        hit_cap = False
        stuck = False
        no_progress_forced = False
        #: Contents of the steering notes injected during this run.  They are
        #: stripped from the returned history so a follow-up turn is not
        #: confused by a stale "budget exhausted / no more tools" instruction.
        injected_notes: list[str] = []
        #: Consecutive-duplicate detection: the exact same call twice in a row
        #: means the model is stuck (e.g. re-searching a symbol it already
        #: searched).  Duplicates are not re-executed; they get a note instead.
        #: A third consecutive duplicate stops the loop and forces synthesis.
        prev_call_key: tuple[str, str] | None = None
        prev_result: str = ""
        #: Seen-call sets for parallel-repeat detection (decision #062):
        #: ``_seen_this_iter`` accumulates every call key of the CURRENT batch
        #: (catches identical calls inside one batch); ``_prev_batch_keys``
        #: holds the FULL key set of the PREVIOUS batch so a model repeating
        #: an entire parallel batch gets every call flagged — not just the one
        #: adjacent to ``prev_call_key``.
        _seen_this_iter: set[tuple[str, str]] = set()
        _prev_batch_keys: set[tuple[str, str]] = set()
        #: Total executions per call_key across ALL batches in this run —
        #: shared with the cross-run ``seen_calls`` registry for consistency.
        _run_seen_calls: dict[tuple[str, str], int] = {}
        #: Per-call-key consecutive-duplicate memory: a key whose PREVIOUS
        #: occurrence was already flagged as a duplicate triggers stuck-
        #: synthesis on its next repeat (third time).  Keyed — not a global
        #: bool — so unrelated calls in a mixed batch never inherit another
        #: call's duplicate history.
        _dup_streak: dict[tuple[str, str], bool] = {}
        #: Progress guard: calls since the last MUTATION or DISCOVERY of new
        #: information.  Read-only tasks (audits, reviews) never mutate, so
        #: mutation alone was a wrong progress proxy that stopped them
        #: mid-work; a call that reads a new file, searches a new query,
        #: lists a new directory, or runs a new command counts as progress.
        #: A long streak of repeats means genuine stuck behaviour — nudge,
        #: then force synthesis.
        calls_without_progress = 0
        nudge_injected = False
        #: Tool-level consecutive-failure counter: tracks how many times each
        #: tool has returned empty/error results in a row (even with different
        #: arguments).  Breaks variant-repeat loops like grep with slightly
        #: different patterns that all fail (decision #061).
        _tool_consec_failures: dict[str, int] = {}

        for iteration in range(self.max_iterations):
            self._emit(
                KIND_STEP_START,
                LAYER_LIFECYCLE,
                iteration=iteration,
                budget_remaining=self.max_iterations - iteration,
            )
            # Progress guard: too many calls without any new progress.
            if calls_without_progress >= self.force_after_no_mutation:
                no_progress_forced = True
                break
            if not nudge_injected and calls_without_progress == self.no_mutation_limit:
                note = _NO_PROGRESS_NUDGE.format(count=calls_without_progress)
                # User role, not system: strict chat templates (qwen Jinja)
                # reject system messages anywhere but the leading block.
                current_messages.append({"role": "user", "content": note})
                injected_notes.append(note)
                nudge_injected = True
                self._emit(
                    KIND_GUARD_TRIGGERED,
                    LAYER_LIFECYCLE,
                    guard=GUARD_NO_MUTATION,
                    iteration=iteration,
                    note=note,
                )

            # Steer toward wrapping up before the cap is actually reached.
            if not deadline_injected and (
                self.max_iterations - iteration <= self.deadline_window
            ):
                remaining = self.max_iterations - iteration
                note = _DEADLINE_NOTE.format(remaining=remaining)
                current_messages.append({"role": "user", "content": note})
                injected_notes.append(note)
                deadline_injected = True
                self._emit(
                    KIND_GUARD_TRIGGERED,
                    LAYER_LIFECYCLE,
                    guard=GUARD_DEADLINE,
                    iteration=iteration,
                    note=note,
                )

            # Call LLM
            response_text, updated_messages = await llm_chat_fn(current_messages, tools)
            current_messages = updated_messages

            if response_text:
                all_text_parts.append(response_text)

            # Check for tool calls in the last assistant message
            last_msg = current_messages[-1] if current_messages else {}
            tool_calls = last_msg.get("tool_calls", [])
            self._emit(
                KIND_LLM_RESPONSE,
                LAYER_OBSERVABILITY,
                iteration=iteration,
                text=truncate(response_text, TEXT_CAP),
                tool_calls_requested=len(tool_calls) if isinstance(tool_calls, list) else 0,
            )

            # No tool calls - we're done
            if not tool_calls:
                break

            #: The model's own narration preceding these tool calls — used as the
            #: human-readable reason in CLEAN mode so bare calls are explained.
            prev_text = str(last_msg.get("content") or "")

            # Seed this batch's duplicate detection with the FULL key set of
            # the previous batch: any key re-issued in consecutive batches is
            # a parallel repeat, even when batch N+1 reorders or adds calls.
            _seen_this_iter = set(_prev_batch_keys)
            _run_seen_calls.clear()
            #: Keys actually EXECUTED in this batch (duplicates excluded).
            _executed_this_batch: set[tuple[str, str]] = set()

            # Execute each tool call
            for tc in tool_calls:
                tc_id = tc.get("id", "")
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                self.tool_calls_made += 1
                self.tools_used[tool_name] = self.tools_used.get(tool_name, 0) + 1
                self.last_tool_call = tool_name
                try:
                    args = json.loads(func.get("arguments", "{}"))
                    if not isinstance(args, dict):
                        args = {}
                except json.JSONDecodeError:
                    args = {}

                call_key = (
                    tool_name,
                    json.dumps(args, sort_keys=True, default=str),
                )
                args_hash = call_key[1]
                # Progress = mutation or discovering something NEW; repeats
                # of known calls (same read window / query / dir / command)
                # increment the stuck counter instead.
                if self._is_progress(tool_name, args):
                    calls_without_progress = 0
                else:
                    calls_without_progress += 1
                # Per-iteration duplicate check: catches parallel repeats where
                # two different calls in batch N are identical to two different
                # calls in batch N+1 (prev_call_key alone misses this).
                is_dup_this_iter = call_key in _seen_this_iter
                _seen_this_iter.add(call_key)
                total_runs = _run_seen_calls.get(call_key, 0) + 1
                _run_seen_calls[call_key] = total_runs
                if call_key == prev_call_key or is_dup_this_iter:
                    #: Total executions of this exact call across ALL chained
                    #: runs (shared registry), so a repeated probe is flagged as
                    #: a known dead-end even after an auto-continue restart.
                    total_runs = max(total_runs, seen_calls.get(call_key, 1) if seen_calls is not None else 1)
                    # If the repeated call's prior result was a missing path and this
                    # tool can follow up with a listing, recover before declaring stuck —
                    # an invented-path retry gets un-stuck instead of dead-ending (decision #035).
                    if _is_path_miss(prev_result) and tool_name in _PATH_SENSITIVE_TOOLS:
                        result_str, discovered = await _recover_path_miss(
                            execute_tool_fn, args, prev_result
                        )
                        calls_without_progress = 0 if discovered else calls_without_progress + 1
                        self._emit(
                            KIND_GUARD_TRIGGERED,
                            LAYER_LIFECYCLE,
                            guard=GUARD_PATH_MISS,
                            iteration=iteration,
                            tool=tool_name,
                            note=_PATH_RECOVERY_NOTE.format(listing="..."),
                        )
                    elif _dup_streak.get(call_key):
                        # Third consecutive identical call: the model is stuck.
                        # Stop the loop right here and force a text answer.
                        result_str = (
                            "NOTE: This identical call has now been made three times in a "
                            "row with the same result. Stop repeating it — take a different "
                            "action or give your final answer now."
                        )
                        stuck = True
                        self._emit(
                            KIND_GUARD_TRIGGERED,
                            LAYER_LIFECYCLE,
                            guard=GUARD_STUCK,
                            iteration=iteration,
                            tool=tool_name,
                            note=_STUCK_SYNTHESIS_NOTE,
                        )
                    else:
                        result_str = (
                            f"NOTE: This exact call has now been executed {total_runs} "
                            f"time(s) in this conversation (result: {prev_result[:160]}). "
                            "It is not re-executed — unless the file changed, the result is "
                            "identical. Take a different action or answer in text."
                        )
                    _dup_streak[call_key] = True
                    self._emit(
                        KIND_TOOL_CALL,
                        LAYER_TOOL_INTERFACE,
                        iteration=iteration,
                        tool=tool_name,
                        args_hash=args_hash,
                        tc_id=tc_id,
                        duplicate=True,
                    )
                    if self.display_mode != DisplayMode.QUIET:
                        print(f"  {yellow('[tool]')} {cyan(tool_name)}({gray(_fmt_args(args))}) (duplicate, not re-executed)")
                        if self.display_mode == DisplayMode.CLEAN:
                            print(f"  {yellow('[reason]')} {_derive_reason(tool_name, args, prev_text)}")
                            summary = _summarize_result(tool_name, result_str)[:_RESULT_DISPLAY_LIMIT]
                            print(f"  {yellow('[result]')} {colorize_result(summary)}")
                        else:
                            shown = colorize_result(result_str[:_RESULT_DISPLAY_LIMIT])
                            if "note:" in shown.lower():
                                shown = magenta(shown)
                            print(f"  {yellow('[result]')} {shown}")
                    current_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result_str,
                    })
                    self._emit(
                        KIND_TOOL_RESULT,
                        LAYER_TOOL_INTERFACE,
                        iteration=iteration,
                        tool=tool_name,
                        args_hash=args_hash,
                        tc_id=tc_id,
                        duplicate=True,
                        result=truncate(result_str, RESULT_CAP),
                    )
                    if stuck:
                        break
                    continue

                if self.display_mode != DisplayMode.QUIET:
                    tool_label = cyan(tool_name) + "(" + gray(_fmt_args(args)) + ")"
                    print(f"  {yellow('[tool]')} {tool_label}")
                self._emit(
                    KIND_TOOL_CALL,
                    LAYER_TOOL_INTERFACE,
                    iteration=iteration,
                    tool=tool_name,
                    args_hash=args_hash,
                    tc_id=tc_id,
                    duplicate=False,
                )
                t_call = time.monotonic()
                affected: list[str] = []
                try:
                    result_str = await execute_tool_fn(tool_name, args)
                except Exception as exc:
                    affected = self._collect_effects(tool_name, args)
                    self._emit(
                        KIND_TOOL_ERROR,
                        LAYER_TOOL_INTERFACE,
                        iteration=iteration,
                        tool=tool_name,
                        args_hash=args_hash,
                        exception=type(exc).__name__,
                        message=str(exc)[:500],
                        affected_files=affected,
                    )
                    result_str = f"Tool error: {exc}"
                else:
                    affected = self._collect_effects(tool_name, args)
                #: Record mutated files for the abandonment-resume protocol
                #: (repair abandonment-resume-protocol): the reconnect note
                #: names exactly what this run changed (decision #052).
                if affected:
                    self._mutated_files.update(str(f) for f in affected)

                #: Path-existence recovery (decision #035): when a read/edit/write
                #: reports the path does not exist, augment the result with a parent-
                #: directory listing so the model can discover the real path instead of
                #: looping on an invented one — this is the simple existence check laguna-s-2.1 lacked.
                if _is_path_miss(result_str) and tool_name in _PATH_SENSITIVE_TOOLS:
                    result_str, discovered = await _recover_path_miss(
                        execute_tool_fn, args, result_str
                    )
                    if discovered:
                        calls_without_progress = 0
                    self._emit(
                        KIND_GUARD_TRIGGERED,
                        LAYER_LIFECYCLE,
                        guard=GUARD_PATH_MISS,
                        iteration=iteration,
                        tool=tool_name,
                        note=_PATH_RECOVERY_NOTE.format(listing="..."),
                    )
                self._emit(
                    KIND_TOOL_RESULT,
                    LAYER_TOOL_INTERFACE,
                    iteration=iteration,
                    tool=tool_name,
                    args_hash=args_hash,
                    tc_id=tc_id,
                    duplicate=False,
                    duration_s=time.monotonic() - t_call,
                    result=truncate(result_str, RESULT_CAP),
                    affected_files=affected,
                )
                if self.display_mode != DisplayMode.QUIET:
                    if self.display_mode == DisplayMode.CLEAN:
                        print(f"  {yellow('[reason]')} {_derive_reason(tool_name, args, prev_text)}")
                        summary = _summarize_result(tool_name, result_str)[:_RESULT_DISPLAY_LIMIT]
                        print(f"  {yellow('[result]')} {colorize_result(summary)}")
                    else:
                        shown = colorize_result(result_str[:_RESULT_DISPLAY_LIMIT])
                        if "tool error:" in shown.lower() or "error" in shown.lower():
                            shown = red(shown)
                        print(f"  {yellow('[result]')} {shown}")
                prev_call_key = call_key
                _dup_streak[call_key] = False
                prev_result = result_str
                _executed_this_batch.add(call_key)
                if seen_calls is not None:
                    seen_calls[call_key] = seen_calls.get(call_key, 0) + 1
                # Tool-level consecutive-failure guard (decision #061):
                # detect variant-repeat loops where the same tool is called
                # with slightly different args that all produce empty/error
                # results (e.g. grep with regex variations).
                _result_stripped = result_str.strip()
                _is_empty_or_error = (
                    not _result_stripped
                    or _result_stripped.startswith("No files found")
                    or _result_stripped.startswith("Tool error:")
                    or _result_stripped.startswith("Error")
                    or "returned no output" in _result_stripped.lower()
                )
                if _is_empty_or_error:
                    _tool_consec_failures[tool_name] = _tool_consec_failures.get(tool_name, 0) + 1
                else:
                    _tool_consec_failures[tool_name] = 0
                if _tool_consec_failures.get(tool_name, 0) >= _TOOL_CONSECUTIVE_FAILURE_LIMIT:
                    note = _TOOL_CONSECUTIVE_FAILURE_NOTE.format(
                        tool=tool_name, count=_tool_consec_failures[tool_name],
                    )
                    current_messages.append({"role": "user", "content": note})
                    injected_notes.append(note)
                    _tool_consec_failures[tool_name] = 0
                    self._emit(
                        KIND_GUARD_TRIGGERED,
                        LAYER_LIFECYCLE,
                        guard=GUARD_STUCK,
                        iteration=iteration,
                        tool=tool_name,
                        note=note,
                    )
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result_str,
                })
            _prev_batch_keys = _executed_this_batch

            if stuck:
                break
        else:
            # The cap was hit while tool calls were still pending: the model
            # never produced a text answer, so force a final synthesis from
            # the tool results already gathered.
            hit_cap = True

        if hit_cap or stuck or no_progress_forced:
            if no_progress_forced:
                self.termination_reason = "no_progress"
                note = _NO_PROGRESS_FORCE.format(count=calls_without_progress)
                guard = GUARD_NO_MUTATION
            elif stuck:
                self.termination_reason = "stuck"
                note = _STUCK_SYNTHESIS_NOTE
                guard = GUARD_STUCK
            else:
                self.termination_reason = "cap"
                note = _FORCED_SYNTHESIS_NOTE
                guard = GUARD_BUDGET
            self._emit(
                KIND_GUARD_TRIGGERED,
                LAYER_LIFECYCLE,
                guard=guard,
                iteration=iteration,
                note=note,
            )
            current_messages.append({"role": "user", "content": note})
            injected_notes.append(note)
            response_text, updated_messages = await llm_chat_fn(current_messages, [])
            current_messages = updated_messages
            self._emit_synthesis_response(iteration, response_text)
            # Large contexts occasionally make the model return nothing even
            # when forced; one explicit second chance keeps the guarantee that
            # the loop never ends without an answer (decision #034).
            if not response_text or response_text.strip() == "(no output)":
                current_messages.append({"role": "user", "content": _FORCED_SYNTHESIS_RETRY})
                response_text, updated_messages = await llm_chat_fn(current_messages, [])
                current_messages = updated_messages
                self._emit_synthesis_response(iteration, response_text)
            if response_text:
                all_text_parts.append(response_text)
        else:
            self.termination_reason = "answer"

        # Only the LAST non-empty text matters: earlier texts are the model's
        # intermediate narration ("I will read the file...") and would clutter
        # the final answer the user sees.
        final_text = ""
        for part in reversed(all_text_parts):
            if part and part.strip():
                final_text = part
                break
        # Steering notes were only meant for the current loop; a fresh turn has
        # a fresh budget, so they must not leak into the persisted history.
        # Matched on content (any role): the notes travel as "user" messages
        # because strict chat templates reject mid-conversation system roles.
        if injected_notes:
            current_messages = [
                m for m in current_messages
                if m.get("content") not in injected_notes
            ]
        self.iterations_used = min(iteration + 1, self.max_iterations)
        return final_text, current_messages

    def _emit_synthesis_response(self, iteration: int, response_text: str) -> None:
        """Trace the forced-synthesis LLM call (decision #034: every loop
        event, including the final tool-less call, is recorded)."""
        self._emit(
            KIND_LLM_RESPONSE,
            LAYER_OBSERVABILITY,
            iteration=iteration,
            text=truncate(response_text or "", TEXT_CAP),
            tool_calls_requested=0,
        )

    def _is_progress(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Progress = mutation or discovering something NEW.

        Read-only tasks (audits, reviews, analysis) never mutate, so the old
        mutation-only proxy stopped them mid-work while they were still
        learning.  A call counts as progress when it reads a new file window,
        searches a new query, lists a new directory, or runs a new command;
        repeating a known call does not — that is the genuine stuck signal.
        """
        if tool_name in MUTATING_TOOLS:
            return True
        if tool_name == "read":
            key: Any = (str(args.get("path", "")), int(args.get("offset") or 0))
        elif tool_name == "search":
            key = (str(args.get("query", "")), str(args.get("path", "")))
        elif tool_name == "list_files":
            key = str(args.get("path", ""))
        else:
            key = json.dumps(args, sort_keys=True, default=str)
        if key in self._seen_progress_keys:
            return False
        self._seen_progress_keys.add(key)
        return True


def _fmt_args(args: dict[str, Any]) -> str:
    """Short one-line rendering of tool arguments for the console."""
    pieces = []
    for key, value in list(args.items())[:4]:
        text = str(value)
        if len(text) > 60:
            text = text[:57] + "..."
        pieces.append(f"{key}={text}")
    return ", ".join(pieces)


def _derive_reason(tool_name: str, args: dict[str, Any], prior_text: str | None) -> str:
    """Build a short human-readable reason for why a tool call was made.

    Uses the model's own reasoning when it supplied one (via ``assistant``
    message content), otherwise falls back to a small heuristic keyed on the
    tool name and its arguments so bare calls are never unexplained."""
    if prior_text:
        first = " ".join(prior_text.split())[:140]
        return f"{tool_name}({_fmt_args(args)}) — {first}"
    # Heuristic fallback keyed on the tool name + args.
    arg_summary = _fmt_args(args) or ""
    if tool_name == "read":
        path = str(args.get("path", ""))[:60]
        return f"read({arg_summary}) — to inspect {path}"
    if tool_name == "search":
        query = str(args.get("query", ""))[:60]
        return f"search({arg_summary}) — to locate '{query}' in the workspace"
    if tool_name == "list_files":
        path = str(args.get("path", "."))[:60]
        return f"list_files({arg_summary}) — to enumerate {path}"
    if tool_name == "write":
        path = str(args.get("path", ""))[:60]
        return f"write({arg_summary}) — to create/update {path}"
    if tool_name == "edit":
        path = str(args.get("path", ""))[:60]
        return f"edit({arg_summary}) — to patch {path}"
    if tool_name in ("run", "git", "tests"):
        target = arg_summary or "(command)"
        return f"{tool_name}({arg_summary}) — to execute/verify {target}"
    if tool_name == "diff":
        file1 = str(args.get("file1", ""))[:60]
        return f"diff({arg_summary}) — to compare changes in {file1}"
    if tool_name == "analyze":
        target = arg_summary or "(code)"
        return f"analyze({arg_summary}) — to review {target}"
    return f"{tool_name}({arg_summary}) — model action"


def _summarize_result(tool_name: str, result_str: str) -> str:
    """Display-only summary of a tool result (the model always receives the
    full payload; this only shapes what the human sees in CLEAN mode)."""
    if not result_str:
        return "(no output)"
    lines = result_str.splitlines()
    n = len(lines)
    head = "\n".join(lines[:6])
    tail = "\n".join(lines[-3:]) if n > 9 else ""

    # Verification line from write/edit/fix is end-user evidence — keep it.
    verify_match = None
    for ln in lines:
        if ln.startswith("[verify] py_compile"):
            verify_match = ln.strip()
            break

    summary = f"{n} line(s) returned"
    if head:
        summary += f":\n{head}"
    if tail and n > 9:
        summary += f"\n... [truncated — see full result in model context] ...\n{tail}"
    if verify_match:
        summary = f"{verify_match}\n{summary}"
    return summary
