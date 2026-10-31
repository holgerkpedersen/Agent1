"""Interactive CLI package: REPL loop, commands, and session management.

``interactive`` holds the REPL extracted from agent.py; ``commands`` and
``session`` are subpackages. Submodules import this package lazily (agent.py
imports them), so no heavy re-exports happen at package-import time.
"""
