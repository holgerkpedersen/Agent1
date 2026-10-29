# Steering Plan (2026-08-12)

**Status:** proposed
**Goal:** Pressing Escape twice at the REPL prompt interrupts the current input read, opens a second prompt for directing text, and injects that text into the active conversation as a user message at the next tool-loop turn boundary. Windows-first (this machine is Windows), with graceful degradation where ESC delivery is impossible.

## Tasks

- [T1] Create agent_core/steering.py: dependency-free steering queue with push/drain/pending/clear API, per-message length cap, queue cap, module docstring explaining the design (role: implementer)
- [T2] Add escape-aware input wrapper to agent_core/signal_handlers.py: escape_input() delegating to _input_executor when set (preserves all existing tests), plain input() fallback on Windows/non-TTY stdin, POSIX raw-mode termios reader detecting double-Esc and raising EscSteerSignal with the typed text (role: implementer) deps: T1
- [T3] Wire the REPL in agent_core/cli/interactive.py: call the new wrapper at the single read site, catch EscSteerSignal in the loop (push to steering queue, print gray notice, continue without registry dispatch), extend banner tip (role: implementer) deps: T1, T2
- [T4] Add the poll point in agent_core/llm/tool_loop.py: opt-in runner flag steering=False default; when enabled drain queue at top-of-iteration and append user-role notes with a [steering] prefix, tracked in injected_notes and stripped from persisted history; emit lifecycle event (role: implementer) deps: T1
- [T5] Thread the opt-in flag from CLI construction into agent runs so interactive sessions enable steering (role: implementer) deps: T3, T4
- [T6] Write tests/test_steering.py: queue unit behaviour, wrapper executor/non-TTY/ESC paths (scripted stdin), tool-loop injection at iteration boundary after completed tool batch, REPL EscSteerSignal routing (role: tester) deps: T1, T2, T3, T4
- [T7] Update docs/README REPL help text with double-Esc usage and the documented Windows limitation (role: documenter) deps: T1, T2, T3, T4, T5, T6

## Non-goals

No third-party TUI library. No mid-tool-batch injection. No changes to signal cleanup semantics.
