"""Tool calling loop orchestrator for LLM conversations."""
import enum
import json
from typing import Callable, Awaitable, Any

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

#: Tools that count as making progress (they change the workspace).
MUTATING_TOOLS = frozenset({"write", "edit", "fix"})

_NO_PROGRESS_NUDGE = (
    "NOTE: You have now made {count} tool calls without modifying any file. "
    "Exploration alone does not finish tasks — either take action with "
    "write/edit/fix, or give your final answer now."
)

_NO_PROGRESS_FORCE = (
    "You have made {count} tool calls without modifying any file, so the tool "
    "loop is stopping. Do NOT call any more tools. Give your final answer now "
    "— report what you found and what you would change."
)


#: Console display cap for [result] lines. The model ALWAYS receives the full
#: result; this only bounds what the human observer sees in the REPL (was 200,
#: which made read/search output look cut off).
_RESULT_DISPLAY_LIMIT = 1500


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
        #: How this run ended: "answer" (model answered in text), "cap"
        #: (iteration cap hit), "stuck" (repeated identical calls), or
        #: "no_progress" (too many calls without modifying any file).
        self.termination_reason: str = "answer"

    async def run(
        self,
        messages: list[dict[str, Any]],
        llm_chat_fn: Callable[[list[dict[str, Any]], list[dict[str, Any]]], Awaitable[tuple[str, list[dict[str, Any]]]]],
        execute_tool_fn: Callable[[str, dict[str, Any]], Awaitable[str]],
        tools: list[dict[str, Any]] | None = None,
        seen_calls: dict[tuple[str, str], int] | None = None,
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

        Returns:
            Tuple of (final_text, updated_messages)

        The loop never ends without a text answer: with ``deadline_window``
        iterations left a budget warning is injected, and if the iteration cap
        is hit while tool calls are still pending, one final tool-less call
        forces the model to synthesize the answer from the gathered results.
        """
        if not tools:
            tools = []

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
        prev_was_duplicate = False
        prev_result: str = ""
        #: Progress guard: non-mutating calls (read/search/list/run/git/diff/
        #: tests/analyze) since the last write/edit/fix.  A long streak means
        #: exploration without converging — nudge, then force synthesis.
        calls_since_mutation = 0
        nudge_injected = False

        for iteration in range(self.max_iterations):
            # Progress guard: too many calls without changing anything.
            if calls_since_mutation >= self.force_after_no_mutation:
                no_progress_forced = True
                break
            if not nudge_injected and calls_since_mutation == self.no_mutation_limit:
                note = _NO_PROGRESS_NUDGE.format(count=calls_since_mutation)
                current_messages.append({"role": "system", "content": note})
                injected_notes.append(note)
                nudge_injected = True

            # Steer toward wrapping up before the cap is actually reached.
            if not deadline_injected and (
                self.max_iterations - iteration <= self.deadline_window
            ):
                remaining = self.max_iterations - iteration
                note = _DEADLINE_NOTE.format(remaining=remaining)
                current_messages.append({"role": "system", "content": note})
                injected_notes.append(note)
                deadline_injected = True

            # Call LLM
            response_text, updated_messages = await llm_chat_fn(current_messages, tools)
            current_messages = updated_messages

            if response_text:
                all_text_parts.append(response_text)

            # Check for tool calls in the last assistant message
            last_msg = current_messages[-1] if current_messages else {}
            tool_calls = last_msg.get("tool_calls", [])

            # No tool calls - we're done
            if not tool_calls:
                break

            #: The model's own narration preceding these tool calls — used as the
            #: human-readable reason in CLEAN mode so bare calls are explained.
            prev_text = str(last_msg.get("content") or "")

            # Execute each tool call
            for tc in tool_calls:
                tc_id = tc.get("id", "")
                func = tc.get("function", {})
                tool_name = func.get("name", "")
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
                if call_key == prev_call_key:
                    #: Total executions of this exact call across ALL chained
                    #: runs (shared registry), so a repeated probe is flagged as
                    #: a known dead-end even after an auto-continue restart.
                    total_runs = (
                        seen_calls.get(call_key, 1) if seen_calls is not None else 1
                    )
                    if prev_was_duplicate:
                        # Third consecutive identical call: the model is stuck.
                        # Stop the loop right here and force a text answer.
                        result_str = (
                            "NOTE: This identical call has now been made three times in a "
                            "row with the same result. Stop repeating it — take a different "
                            "action or give your final answer now."
                        )
                        stuck = True
                    else:
                        result_str = (
                            f"NOTE: This exact call has now been executed {total_runs} "
                            f"time(s) in this conversation (result: {prev_result[:160]}). "
                            "It is not re-executed — unless the file changed, the result is "
                            "identical. Take a different action or answer in text."
                        )
                    prev_was_duplicate = True
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
                    if stuck:
                        break
                    continue

                if self.display_mode != DisplayMode.QUIET:
                    tool_label = cyan(tool_name) + "(" + gray(_fmt_args(args)) + ")"
                    print(f"  {yellow('[tool]')} {tool_label}")
                try:
                    result_str = await execute_tool_fn(tool_name, args)
                except Exception as exc:
                    result_str = f"Tool error: {exc}"
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
                prev_was_duplicate = False
                prev_result = result_str
                if seen_calls is not None:
                    seen_calls[call_key] = seen_calls.get(call_key, 0) + 1
                calls_since_mutation = (
                    0 if tool_name in MUTATING_TOOLS else calls_since_mutation + 1
                )
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result_str,
                })

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
                note = _NO_PROGRESS_FORCE.format(count=calls_since_mutation)
            elif stuck:
                self.termination_reason = "stuck"
                note = _STUCK_SYNTHESIS_NOTE
            else:
                self.termination_reason = "cap"
                note = _FORCED_SYNTHESIS_NOTE
            current_messages.append({"role": "system", "content": note})
            injected_notes.append(note)
            response_text, updated_messages = await llm_chat_fn(current_messages, [])
            current_messages = updated_messages
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
        if injected_notes:
            current_messages = [
                m for m in current_messages
                if not (m.get("role") == "system" and m.get("content") in injected_notes)
            ]
        return final_text, current_messages


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
