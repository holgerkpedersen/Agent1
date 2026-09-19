"""Regression test for Decision #035 — consolidate duplicate LLMClient.

Decision #035 states: remove the redundant ``LLMClient`` in
``agent_core/llm_client.py``; all LLM access must go through the canonical
``LLMProvider`` Protocol defined in ``agent_core/llm/provider.py``.

This test guards against the divergent httpx-based duplicate silently
reappearing. It asserts:

* The canonical ``LLMProvider`` Protocol and ``build_provider`` factory exist.
* No module-level ``LLMClient`` class is importable from
  ``agent_core.llm_client`` (the removed file must not be reintroduced as a
  standalone client).
* Nothing in the codebase imports symbols from that path.

The repo-root ``agent.py`` retains a *pass-through wrapper* named ``LLMClient``
that delegates to ``build_provider`` — this is intentional and allowed by #035;
only the divergent provider-bypassing copy must stay gone.
"""

from __future__ import annotations

import os
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _module_exists(name: str) -> bool:
    spec = importlib.util.find_spec(name)
    return spec is not None and spec.origin is not None


class TestDecision035Consolidation:
    def test_canonical_provider_protocol_present(self) -> None:
        """LLMProvider Protocol + build_provider factory must exist."""
        assert _module_exists("agent_core.llm.provider"), (
            "Canonical provider module agent_core/llm/provider.py is missing"
        )
        from agent_core.llm import provider as prov

        assert hasattr(prov, "build_provider"), (
            "build_provider factory missing from agent_core/llm/provider.py — "
            "the canonical LLM access surface required by Decision #035"
        )
        assert isinstance(getattr(prov, "LLMProvider", None), type) or hasattr(
            prov, "LLMProvider"
        ), (
            "LLMProvider Protocol missing from agent_core/llm/provider.py — "
            "the canonical LLM access surface required by Decision #035"
        )

    def test_divergent_llm_client_module_removed(self) -> None:
        """agent_core/llm_client.py must not reintroduce a standalone client.

        If the file is absent, find_spec returns None — that's the desired state.
        If someone recreates it as an importable module with its own LLMClient
        class that does NOT delegate to build_provider, this test fails.
        """
        if not _module_exists("agent_core.llm_client"):
            return  # file removed / not on path — desired state

        # If the module still exists, it must NOT define a divergent client
        # that bypasses the provider protocol. We import lazily so an absent
        # module never raises ImportError here (handled above).
        try:
            from agent_core import llm_client as legacy  # noqa: F401
        except ImportError:
            return

        has_own_client = hasattr(legacy, "LLMClient")
        if not has_own_client:
            return  # module exists but no divergent class — acceptable shim

        client_cls = getattr(legacy, "LLMClient")
        src_path = getattr(client_cls, "__module__", "") + ":LLMClient"
        # A proper wrapper delegates to build_provider; a divergent duplicate
        # does not. Inspect the source for delegation evidence.
        try:
            import inspect

            body = inspect.getsource(client_cls)
        except (OSError, TypeError):
            body = ""

        assert "build_provider" in body or "_provider" in body, (
            f"Divergent LLMClient at {src_path} must delegate to the canonical "
            "LLMProvider/build_provider per Decision #035; it does not. Remove "
            "the duplicate."
        )

    def test_no_imports_of_removed_module(self) -> None:
        """No .py file in the repo should import from agent_core.llm_client."""
        offenders = []
        for dp, _, fns in os.walk(ROOT):
            if any(s in dp for s in ["__pycache__", ".pytest_cache", "node_modules"]):
                continue
            # Skip this test file itself and the legacy module under test.
            rel = Path(dp).relative_to(ROOT)
            if str(rel) == "tests/unit":
                pass  # still scan, but we filter by content below
            for fn in fns:
                if not fn.endswith(".py"):
                    continue
                p = os.path.join(dp, fn)
                try:
                    src = open(p, encoding="utf-8").read()
                except Exception:
                    continue
                for i, line in enumerate(src.split("\n"), 1):
                    if "llm_client" in line and (
                        "import" in line or "from agent_core.llm_client" in line
                    ):
                        # Allow references that are NOT imports — e.g. a string
                        # filename list in workflow_cmd.py ("llm_client.py").
                        stripped = line.strip()
                        if stripped.startswith(("import ", "from ")) and (
                            "agent_core.llm_client" in stripped or ".llm_client import" in stripped
                        ):
                            offenders.append(f"{p}:{i}: {stripped}")

        assert not offenders, (
            "Decision #035 requires no imports of the removed agent_core/llm_client "
            f"module; found: {offenders}"
        )
