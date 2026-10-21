"""agent.py — thin compatibility shim (refactor step r08).

The implementation lives in :mod:`agent_core.agent` (class ``AgentCore``,
exposed here as ``Agent``) and :mod:`agent_core.cli.interactive`.  This
module re-exports the frozen public/private surface so ``import agent``,
``python -m agent``, the REPL banner and every test monkeypatch on the
``agent`` module keep working unchanged.
"""

from agent_core.agent import *  # noqa: F401,F403 — frozen surface via __all__

from agent_dashboard import (  # noqa: F401
    _build_dashboard,
    _dashboard_port,
    _default_alert_rules,
    _emit_command_metrics,
    _emit_tool_metrics,
    _shared_metrics_collector,
    get_metrics_collector,
    record_command_metrics,
    run_dashboard_server,
    start_dashboard_thread,
)

from agent_core.cli.interactive import (  # noqa: F401
    _INCOMPLETE_MARKERS,
    _COMPLETE_MARKERS,
    _ACTION_VERB_RE,
    _QUESTION_LEAD_RE,
    _looks_incomplete,
    _unverified_action_claim,
    _register_commands,
    _build_registry,
    _warn_uncommitted,
    _build_chat_prompt,
    CHAT_INPUT_PROMPT,
    run_interactive,
    main,
)

if __name__ == "__main__":
    import asyncio

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
