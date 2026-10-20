"""Regression: module docstrings must sit above ``from __future__`` imports.

When a string literal follows ``from __future__ import annotations`` it is
just an expression, so ``module.__doc__`` stays ``None``.
"""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    ["agent_core.llm.orchestrator", "agent_core.llm.prompt_cache"],
)
def test_module_has_docstring(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module.__doc__ and module.__doc__.strip()
