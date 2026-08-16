"""Root pytest configuration.

Trace capture in agent.py chat_nlp is opt-out (AGENT_NO_TRACE=1 disables it)
so a real agent session produces a trace corpus by default.  Test runs must
not write reports/traces/ artifacts, so the whole suite runs with tracing
disabled unless a test explicitly enables it.
"""

import os

os.environ.setdefault("AGENT_NO_TRACE", "1")
