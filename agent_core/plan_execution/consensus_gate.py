"""Consensus review gate for plan execution (plan item #3, second half).

The consensus machinery in :mod:`agent_core.llm.parallel` — structured
``VERDICT:`` lines, ``ParallelRun.auto_agree``, ``quorum_reached()`` — has
existed since 2026-08, but nothing outside tests ever let it gate a real
action.  This module gives it one: when the workspace preference
``consensus_gate`` is on, ``plan_start`` first asks every configured model
for a structured verdict on the proposed plan, and only starts the plan when
the approval ratio reaches quorum (default 0.6).

Semantics (deliberate, pinned by tests/test_consensus_gate.py):

* **Opt-in.**  The gate is off unless the workspace pref ``consensus_gate``
  is explicitly true — with it off, no provider call happens and
  ``plan_start`` behaves exactly as before.
* **Model disagreement blocks.**  A quorum of structured APPROVE verdicts
  auto-approves the plan; anything the models honestly reject stays
  ``proposed`` (no lifecycle transition).
* **Infrastructure failure fails OPEN.**  If no model answers at all (dead
  servers, fewer than two usable models), the gate *skips* with a visible
  notice — a network outage must not silently block the workflow.  Only an
  honest "no" from models that answered can stop a plan.
* **Abstentions skipped.**  Answers with no parseable ``VERDICT:`` line are
  skipped, never counted as rejects (``auto_agree`` semantics).

The preference lives in the workspace prefs file alongside ``habits_enabled``
and ``profile_auto`` (:mod:`agent_core.llm.workspace_prefs`):
``.workspace/agent_llm_prefs.json``::

    {"consensus_gate": true, "consensus_quorum": 0.75}
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

#: Workspace pref that enables the gate (default off — opt-in).
CONSUSUS_GATE_PREF = "consensus_gate"

#: Optional workspace pref overriding the approval-ratio quorum.
QUORUM_PREF = "consensus_quorum"

#: Default approval ratio required to auto-approve a plan.
DEFAULT_QUORUM = 0.6

#: Cap on the plan text sent to the reviewers (small NPU windows are real).
MAX_PLAN_CHARS = 8000

#: Template id prefix under which votes land in the consensus ledger.
TEMPLATE_ID = "plan-review"


@dataclass(frozen=True)
class ConsensusGateOutcome:
    """What the gate decided about one proposed plan."""

    enabled: bool
    approved: bool
    consensus: str = ""
    recorded: int = 0
    skipped_reason: str = ""


def gate_enabled(workspace: Path | str) -> bool:
    """True only when the workspace explicitly enables ``consensus_gate``."""
    from agent_core.llm.workspace_prefs import get_pref

    try:
        return get_pref(Path(str(workspace)), CONSUSUS_GATE_PREF) is True
    except Exception:  # unreadable prefs file: keep behaviour unchanged
        logger.debug("consensus_gate pref unreadable, continuing", exc_info=True)
        return False


def _quorum(workspace: Path | str) -> float:
    """Quorum threshold: explicit pref when numeric, else DEFAULT_QUORUM."""
    from agent_core.llm.workspace_prefs import get_pref

    try:
        raw = get_pref(Path(str(workspace)), QUORUM_PREF)
    except Exception:
        logger.debug("consensus_quorum pref unreadable, continuing", exc_info=True)
        return DEFAULT_QUORUM
    if isinstance(raw, (int, float)) and 0.0 <= float(raw) <= 1.0:
        return float(raw)
    return DEFAULT_QUORUM


def default_models(agent: Any, settings: Any) -> list[str]:
    """Models for the review: agent's current model + configured opencode one.

    Same canonical pairing ``multillm`` uses (one local, one hosted): deduped,
    empties dropped.  With fewer than two models the gate skips — a single
    "model" is not a consensus.
    """
    out: list[str] = []
    current = str(getattr(getattr(agent, "llm", None), "model_name", "") or "")
    opencode_model = str(getattr(settings, "opencode_model", "") or "")
    for m in (current, opencode_model):
        if m and m not in out:
            out.append(m)
    return out


def review_prompt(plan_text: str) -> str:
    """The reviewer question every model answers with its own verdict."""
    text = plan_text.strip()
    if len(text) > MAX_PLAN_CHARS:
        text = text[:MAX_PLAN_CHARS] + "\n...[plan truncated]"
    return (
        "You are reviewing a plan proposed for execution. Judge whether it "
        "is coherent, safe and worth executing as written.\n\n"
        f"PLAN UNDER REVIEW:\n{text}\n\n"
        "Decide: should this plan be approved for execution?"
    )


async def review_plan(
    agent: Any,
    plan_text: str,
    *,
    models: Optional[Sequence[str]] = None,
    settings: Any = None,
    threshold: Optional[float] = None,
    template_id: str = TEMPLATE_ID,
) -> ConsensusGateOutcome:
    """Run the parallel multi-model verdict review of *plan_text*.

    Imports :func:`run_parallel` lazily so this module imports cleanly in
    environments without provider deps (and so tests can patch it at its
    source module).
    """
    workspace = Path(str(getattr(agent, "workspace", ".")))
    if not gate_enabled(workspace):
        return ConsensusGateOutcome(
            enabled=False, approved=True, skipped_reason="gate disabled",
        )

    quorum = DEFAULT_QUORUM if threshold is None else float(threshold)

    resolved_models = [str(m) for m in (models or []) if str(m).strip()]
    if not resolved_models:
        if settings is None:
            try:
                from agent_core.config import load_agent_settings

                settings = load_agent_settings()
            except Exception as exc:  # infra failure -> fail open, visibly
                return ConsensusGateOutcome(
                    enabled=True, approved=True,
                    skipped_reason=f"settings unavailable ({exc})",
                )
        resolved_models = default_models(agent, settings)

    if len(resolved_models) < 2:
        return ConsensusGateOutcome(
            enabled=True, approved=True,
            skipped_reason=(
                "needs at least two models — got "
                f"{len(resolved_models)}; gate skipped (fail-open)"
            ),
        )

    from agent_core.llm.parallel import run_parallel

    messages = [{"role": "user", "content": review_prompt(plan_text)}]
    try:
        run = await run_parallel(
            messages,
            resolved_models,
            settings,
            template_id=template_id,
            disable_thinking=True,
            verdict_instruction=True,
        )
    except Exception as exc:  # provider dispatch failure -> fail open, visibly
        logger.debug("consensus gate run_parallel failed", exc_info=True)
        return ConsensusGateOutcome(
            enabled=True, approved=True,
            skipped_reason=f"review dispatch failed ({exc})",
        )

    recorded = run.auto_agree()
    # Quote the REAL threshold in the summary line (ParallelRun.consensus
    # defaults to 50%; the gate's quorum pref is what actually decides).
    consensus = run.consensus(quorum)
    if recorded == 0:
        return ConsensusGateOutcome(
            enabled=True, approved=True, consensus=consensus, recorded=0,
            skipped_reason="no usable answers — gate skipped (fail-open)",
        )
    approved = bool(run.quorum_reached(quorum))
    return ConsensusGateOutcome(
        enabled=True, approved=approved, consensus=consensus, recorded=recorded,
    )


__all__ = [
    "CONSUSUS_GATE_PREF",
    "QUORUM_PREF",
    "DEFAULT_QUORUM",
    "MAX_PLAN_CHARS",
    "TEMPLATE_ID",
    "ConsensusGateOutcome",
    "gate_enabled",
    "default_models",
    "review_prompt",
    "review_plan",
]
