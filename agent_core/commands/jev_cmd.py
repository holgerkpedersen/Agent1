"""jev command — typed, probabilistic answers from a dedicated small model.

Usage::

    jev yesno  "<statement>"                [state/threshold/... flags]
    jev choice "A|B|C" "<question>"         [state/threshold/... flags]
    jev score  --rubric "<rubric>" "<subject>" [state/threshold/... flags]

``jev choice`` accepts the option list positionally (``"A|B|C"``) or via
``--options "A|B|C"`` — the positional form is the natural first guess.

Shared flags::

    --state "<text>"   inline context the question is judged against
    --file <path>      read context from a workspace file (read-only)
    --samples N        vote samples (default from settings, 5)
    --threshold X      decision threshold in [0,1] (default 0.7)
    --mechanism M      auto | vote | logprobs (default auto)
    --model <name>     override the configured Jev model for this call
    --json             print the machine-readable result instead of the block

Unlike ``speculate`` (which deliberates with the agent's MAIN model), ``jev``
runs on the dedicated small model configured by ``AGENT_JEV_MODEL`` /
``model_catalog.json`` ``_defaults.jev_model`` — it never touches ``agent.llm``.
Output is a typed result (decision + probabilities) the caller can branch on.
The command is READ-ONLY: ``--file`` only reads; no tool can mutate anything.

REPL convention: ``shlex.split(posix=False)`` keeps literal quotes on quoted
values, stripped here with ``.strip('"')`` (same as ``speculate``/``multillm``).
"""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from .base import Command

if TYPE_CHECKING:
    from agent import Agent

_KINDS = ("yesno", "choice", "score")
_MECHANISMS = ("auto", "vote", "logprobs")
_VALUE_FLAGS = {
    "--options", "--state", "--file", "--rubric", "--samples",
    "--threshold", "--mechanism", "--model",
}


class JevCommand(Command):
    """Ask the dedicated small Jev model a typed question."""

    @property
    def name(self) -> str:
        return "jev"

    @property
    def help_text(self) -> str:
        return (
            'jev yesno "<statement>" | jev choice "A|B|C" "<q>" | '
            'jev score --rubric "<rubric>" "<subject>" | jev stats '
            '[--last N] [--json] | jev label <id> correct|incorrect '
            "[--note \"...\"] [--state \"<text>\" | --file <path>] "
            "[--samples N] [--threshold X] [--mechanism auto|vote|logprobs] "
            "[--model m] [--json] - typed probabilistic decision from the "
            "dedicated small Jev model; 'jev stats' reports decision counts, "
            "calibration and a suggested threshold; 'jev label' records an "
            "outcome so the threshold can be calibrated"
        )

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        from agent_core.jev_engine import (
            KIND_CHOICE,
            JevQuestion,
            build_jev_engine,
        )

        parts = list(args)
        if parts and parts[0].strip('"').lower() == "stats":
            return self._stats(parts[1:], agent)
        if parts and parts[0].strip('"').lower() == "label":
            return self._label(parts[1:], agent)
        if not parts or parts[0].strip('"').lower() not in _KINDS:
            self.error(
                'Usage: jev yesno|choice|score "<question>" '
                '[--options "A|B|C" (or a positional "A|B|C")] '
                '[--rubric "<r>"] [--state "<text>"] '
                "[--file <path>] [--samples N] [--threshold X] [--json]"
            )
            return True
        kind = parts[0].strip('"').lower()

        options: list[str] = []
        state_text = ""
        file_path = ""
        rubric = ""
        samples: int | None = None
        threshold = 0.7
        mechanism = "auto"
        model: str | None = None
        as_json = False

        i = 1
        while i < len(parts):
            p = parts[i]
            nxt = parts[i + 1] if i + 1 < len(parts) else None
            if p == "--options" and nxt is not None:
                options = [
                    o.strip() for o in nxt.strip('"').split("|") if o.strip()
                ]
                i += 2
                continue
            if p == "--state" and nxt is not None:
                state_text = nxt.strip('"')
                i += 2
                continue
            if p == "--file" and nxt is not None:
                file_path = nxt.strip('"')
                i += 2
                continue
            if p == "--rubric" and nxt is not None:
                rubric = nxt.strip('"')
                i += 2
                continue
            if p == "--samples" and nxt is not None:
                try:
                    samples = max(1, int(nxt))
                except ValueError:
                    self.error("--samples expects a number.")
                    return True
                i += 2
                continue
            if p == "--threshold" and nxt is not None:
                try:
                    threshold = float(nxt)
                except ValueError:
                    self.error("--threshold expects a number between 0 and 1.")
                    return True
                if not 0.0 <= threshold <= 1.0:
                    self.error("threshold must be within [0.0, 1.0]")
                    return True
                i += 2
                continue
            if p == "--mechanism" and nxt is not None:
                mechanism = nxt.strip('"').lower()
                if mechanism not in _MECHANISMS:
                    self.error(
                        "--mechanism must be one of "
                        + ", ".join(_MECHANISMS)
                    )
                    return True
                i += 2
                continue
            if p == "--model" and nxt is not None:
                model = nxt.strip('"')
                i += 2
                continue
            if p == "--json":
                as_json = True
                i += 1
                continue
            i += 1

        # Everything not a flag or a flag value is the question.
        flag_values: set[str] = set()
        for j, token in enumerate(parts):
            if token in _VALUE_FLAGS and j + 1 < len(parts):
                flag_values.add(parts[j + 1])
        question_words = [
            token for token in parts[1:]
            if token != "--json"
            and not token.startswith("--")
            and token not in flag_values
        ]
        # Convenience: `jev choice "A|B|C" "<question>"` — the first positional
        # token carrying pipes IS the option list (the documented form is
        # `--options "A|B|C"`, but the positional list is the natural guess).
        if kind == KIND_CHOICE and not options and question_words:
            first = question_words[0].strip('"').strip("'")
            if "|" in first:
                options = [o.strip() for o in first.split("|") if o.strip()]
                question_words = question_words[1:]
        question_text = " ".join(question_words).strip().strip('"')

        if not question_text:
            self.error("jev needs a question/proposition.")
            return True
        if kind == KIND_CHOICE and len(options) < 2:
            self.error(
                'a choice question needs options: --options "A|B|C" or a '
                'positional "A|B|C".'
            )
            return True

        state = state_text
        if file_path:
            read = self._read_state_file(file_path, agent)
            if read is None:
                return True
            state = (state + "\n" + read).strip() if state else read

        try:
            question = JevQuestion(
                kind=kind,
                text=question_text,
                options=tuple(options),
                rubric=rubric,
            )
        except ValueError as exc:
            self.error(str(exc))
            return True

        try:
            engine = build_jev_engine(
                model_name=model, samples=samples, mechanism=mechanism,
                threshold=threshold, source="repl",
                log_workspace=getattr(agent, "workspace", None),
            )
        except Exception as exc:  # noqa: BLE001 - surface a clear setup error
            self.error(f"could not build the Jev engine: {exc}")
            return True

        print(
            f"\n  [jev] kind={kind} model={engine.model_name} "
            f"mechanism={engine.mechanism} samples={engine.samples}"
        )
        try:
            result = await engine.decide(question, state)
        except Exception as exc:  # noqa: BLE001 - a failed decision must not kill the REPL
            self.error(f"jev decision failed: {exc}")
            return True

        if as_json:
            print(json.dumps(result.as_dict(), indent=2))
            return True
        for line in result.summary().splitlines():
            print(f"  {line}")
        return True

    def _read_state_file(self, file_path: str, agent: "Agent") -> str | None:
        """Read a workspace file for context (read-only); None on error."""
        path = file_path
        if not os.path.isabs(path):
            workspace = str(getattr(agent, "workspace", "") or os.getcwd())
            path = os.path.join(workspace, path)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except FileNotFoundError:
            self.error(f"state file not found: {path}")
            return None
        except OSError as exc:
            self.error(f"could not read state file: {exc}")
            return None
        cap = 8000
        return text[:cap] if len(text) > cap else text

    def _stats(self, args: list[str], agent: "Agent") -> bool:
        """Report decision counts + calibration from the telemetry ledger.

        ``jev stats [--last N] [--json]`` — reads
        ``reports/history/jev.jsonl`` (written by every ``decide``), prints
        per-kind/mechanism/source/model counts, predicted-vs-observed
        calibration for labeled yesno decisions, and the threshold that best
        separates correct from incorrect decisions so far.  With no labeled
        outcomes it explains how to label them.
        """
        from harnessfix.jev_telemetry import (
            format_report,
            load_decisions,
            summarize,
        )

        last: int | None = None
        as_json = False
        i = 0
        while i < len(args):
            token = args[i]
            nxt = args[i + 1] if i + 1 < len(args) else None
            if token == "--last" and nxt is not None:
                try:
                    last = max(1, int(nxt))
                except ValueError:
                    self.error("--last expects a number.")
                    return True
                i += 2
                continue
            if token == "--json":
                as_json = True
                i += 1
                continue
            i += 1

        workspace = getattr(agent, "workspace", None)
        records = load_decisions(workspace=workspace, last=last)
        report = summarize(records)
        if as_json:
            print(json.dumps(report, indent=2))
            return True
        for line in format_report(report).splitlines():
            print(f"  {line}")
        return True

    def _label(self, args: list[str], agent: "Agent") -> bool:
        """``jev label <id> correct|incorrect [--note "..."]``.

        The labeling half of the calibration loop: ``jev stats`` suggests a
        threshold from labeled decisions, and this is how decisions get their
        outcome (the id comes from the telemetry ledger /
        ``report["..."]``).
        """
        from harnessfix.jev_telemetry import record_outcome

        if len(args) < 2:
            self.error("Usage: jev label <decision-id> correct|incorrect")
            return True
        decision_id = args[0].strip('"').strip("'")
        outcome = args[1].strip('"').strip("'").lower()
        note = ""
        i = 2
        while i < len(args):
            if args[i] == "--note" and i + 1 < len(args):
                note = args[i + 1].strip('"')
                i += 2
                continue
            i += 1
        try:
            ok = record_outcome(
                decision_id, outcome,
                workspace=getattr(agent, "workspace", None), note=note,
            )
        except ValueError as exc:
            self.error(str(exc))
            return True
        if ok:
            print(f"  [jev] labeled {decision_id} as {outcome}.")
        else:
            self.error(f"decision '{decision_id}' not found in the ledger.")
        return True
