"""Jev decision engine — typed, probabilistic answers from a dedicated small model..

Jev (named after TypeSafe AI's System One decision model) turns a *state* plus a
*typed question* into a typed, probabilistic answer a caller can branch on:

* ``yesno``  -> ``P(yes)`` / ``P(no)`` for a statement (decision TRUE/FALSE/UNKNOWN)
* ``choice`` -> a distribution over fixed options (decision = argmax or UNDECIDED)
* ``score``  -> a number on a 0-100 rubric plus spread (decision = the value)

It is deliberately NOT a chat model: every call is a short, constrained
micro-decision answered with a single token.  That is what makes a small local
model (``qwen2.5-coder-1.5b-instruct`` by default — non-thinking and terse) the
right engine, and why the engine builds its OWN provider from
``settings.jev_model`` and never touches the agent's main LLM.
``settings.jev_provider`` (or the model-name prefix) selects the provider —
``qwen`` routes to LM Studio via ``model_catalog.json``.

Two mechanisms, with a safe ``auto`` default:

* ``logprobs`` — ONE call; read the model's own probability mass over the
  candidate answer tokens (requires a backend that returns logprobs, e.g.
  :meth:`agent_core.llm.lmstudio.LMStudioProvider.chat_logprobs`).
* ``vote`` — N independent samples at a non-zero temperature; the empirical
  agreement IS the probability (self-consistency).  Works on ANY provider.

``auto`` tries logprobs first and falls back to voting (also when a response
carries no usable logprobs).  Every failure — provider error, timeout, empty
output, an unparseable token, a leaked tool call — is an ABSTENTION, never a
fabricated vote; a majority of abstentions yields UNKNOWN/UNDECIDED.

``agent_core`` namespace rule: this module never imports ``agent``.
"""
from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agent_core.constants import DEFAULT_JEV_MODEL

KIND_YESNO = "yesno"
KIND_CHOICE = "choice"
KIND_SCORE = "score"
_VALID_KINDS: frozenset[str] = frozenset({KIND_YESNO, KIND_CHOICE, KIND_SCORE})

MECHANISM_AUTO = "auto"
MECHANISM_VOTE = "vote"
MECHANISM_LOGPROBS = "logprobs"
_VALID_MECHANISMS: frozenset[str] = frozenset(
    {MECHANISM_AUTO, MECHANISM_VOTE, MECHANISM_LOGPROBS}
)

#: Answer token families for the ``yesno`` primitive.
_YES_WORDS: frozenset[str] = frozenset(
    {"yes", "y", "true", "t", "correct", "pass", "1"}
)
_NO_WORDS: frozenset[str] = frozenset(
    {"no", "n", "false", "f", "incorrect", "fail", "0"}
)

#: Provider error prefixes (providers return ``[Error: ...]`` as plain text).
_ERROR_PREFIXES = ("[Error", "[LM Studio")
#: Leaked tool-call syntax (never a valid answer).
_TOOL_MARKERS = ("<|tool_call", "<tool_call", "<|tool_response", "<tool_response")
_INLINE_TOOL_RE = re.compile(r"^call[:_]\w+\s*[{(]", re.IGNORECASE)

_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")
_WORD_RE = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class JevQuestion:
    """A typed question for the Jev engine.

    Args:
        kind: one of ``yesno`` / ``choice`` / ``score``.
        text: the proposition to judge / question to answer.
        options: the fixed choices (``choice`` only; at least two).
        rubric: the scoring rubric (``score`` only, optional).
        max_value: the score scale (default 100, i.e. the model answers 0-100).
    """

    kind: str
    text: str
    options: tuple[str, ...] = ()
    rubric: str = ""
    max_value: float = 100.0

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"JevQuestion.kind must be one of {', '.join(sorted(_VALID_KINDS))}, "
                f"got {self.kind!r}"
            )
        if not str(self.text).strip():
            raise ValueError("JevQuestion.text must not be empty")
        if self.kind == KIND_CHOICE:
            if len(self.options) < 2:
                raise ValueError("a 'choice' question needs at least two options")
            cleaned = tuple(str(o).strip() for o in self.options)
            if any(not o for o in cleaned):
                raise ValueError("choice options must not be empty")
        if self.kind == KIND_SCORE and self.max_value <= 0:
            raise ValueError("score max_value must be positive")


@dataclass
class JevResult:
    """A typed, probabilistic Jev answer.

    ``probabilities`` maps a label to a probability in ``[0, 1]`` (for
    ``yesno`` the labels are ``yes``/``no``; for ``choice`` the option strings;
    empty for ``score``).  ``confidence`` is the winning probability
    (yesno/choice) or the normalised score value.  ``decision`` is the typed
    branch the caller acts on: ``TRUE`` / ``FALSE`` / ``UNKNOWN`` for yesno,
    the winning option / ``UNDECIDED`` for choice, or the numeric score for
    ``score``.  ``abstentions`` counts samples that produced no usable answer.
    """

    kind: str
    model: str
    mechanism: str
    probabilities: dict[str, float] = field(default_factory=dict)
    decision: str = "UNKNOWN"
    confidence: float = 0.0
    threshold: float = 0.7
    n: int = 0
    abstentions: int = 0
    value: float | None = None
    spread: float | None = None
    samples: list[str] = field(default_factory=list)
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Machine-readable form (the Jev "function call" contract)."""
        return {
            "kind": self.kind,
            "model": self.model,
            "mechanism": self.mechanism,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
            "threshold": self.threshold,
            "n": self.n,
            "abstentions": self.abstentions,
            "value": None if self.value is None else round(self.value, 4),
            "spread": None if self.spread is None else round(self.spread, 4),
            "error": self.error,
        }

    def summary(self) -> str:
        """One compact, plain-text block for the REPL / tool output."""
        head = (
            f"[jev] model={self.model or '?'} mechanism={self.mechanism} "
            f"kind={self.kind} samples={self.n} abstain={self.abstentions}"
        )
        if self.error:
            return f"{head}\n[jev] ERROR: {self.error}"
        if self.kind == KIND_YESNO:
            p_yes = self.probabilities.get("yes", 0.0)
            p_no = self.probabilities.get("no", 0.0)
            return (
                f"{head}\n[jev] P(yes)={p_yes:.2f} P(no)={p_no:.2f} -> "
                f"{self.decision} (confidence {self.confidence:.2f}, "
                f"threshold {self.threshold:g})"
            )
        if self.kind == KIND_CHOICE:
            dist = " ".join(
                f"P({opt})={p:.2f}" for opt, p in self.probabilities.items()
            )
            return (
                f"{head}\n[jev] {dist} -> {self.decision} "
                f"(confidence {self.confidence:.2f}, threshold {self.threshold:g})"
            )
        spread = "" if self.spread is None else f" (+/- {self.spread:.2f})"
        return f"{head}\n[jev] score={self.confidence:.2f}{spread}"


# ---------------------------------------------------------------------------
#  Parsers (pure, unit-testable)
# ---------------------------------------------------------------------------

def is_failure(text: str) -> bool:
    """True when *text* is a provider error or leaked tool-call syntax.

    Such a reply is an ABSTENTION, not an answer — treating it as a vote is how
    a fake ``P=1.0`` gets committed.
    """
    stripped = str(text).strip()
    if not stripped:
        return False
    if stripped.startswith(_ERROR_PREFIXES):
        return True
    low = stripped.lower()
    if any(marker in low for marker in _TOOL_MARKERS):
        return True
    return bool(_INLINE_TOOL_RE.match(stripped))


def _clean_token(token: str) -> str:
    """Normalise one answer token (strip quotes/punctuation, lowercase)."""
    return str(token).strip().strip("\"'`.,:;()[]{} ").lower()


def parse_yesno(reply: str) -> str | None:
    """Extract ``"yes"``/``"no"`` from a reply, or ``None`` (abstention)."""
    stripped = str(reply).strip()
    if not stripped or is_failure(stripped):
        return None
    words = _WORD_RE.findall(stripped.lower())
    for word in words:
        if word in _YES_WORDS:
            return "yes"
        if word in _NO_WORDS:
            return "no"
    # A bare digit answer ("1"/"0") has no alpha word.
    digits = _NUMBER_RE.findall(stripped)
    if digits and not words:
        try:
            return "yes" if float(digits[0]) >= 0.5 else "no"
        except ValueError:
            return None
    return None


def parse_choice(reply: str, options: tuple[str, ...]) -> str | None:
    """Map a reply onto one of *options*, or ``None`` (abstention)."""
    stripped = str(reply).strip()
    if not stripped or is_failure(stripped):
        return None
    low = _clean_token(stripped)
    # Letter ("A"/"a)") or 1-based digit answer.
    first = low[:1]
    if len(first) == 1:
        if first.isalpha():
            idx = ord(first) - ord("a")
            if 0 <= idx < len(options):
                return options[idx]
        if first.isdigit():
            idx = int(first) - 1
            if 0 <= idx < len(options):
                return options[idx]
    # A standalone option label anywhere ("Option B" -> "B").
    for token in re.findall(r"[A-Za-z0-9]+", stripped):
        if len(token) != 1:
            continue
        option_idx = _letter_index(token)
        if option_idx is not None and 0 <= option_idx < len(options):
            return options[option_idx]
    # Option text (longest match first so "red" wins over "r").
    for option in sorted(options, key=len, reverse=True):
        if option.strip().lower() in low:
            return option
    return None


def parse_score(reply: str, max_value: float = 100.0) -> float | None:
    """Extract a ``[0, max_value]`` score, or ``None`` (abstention).

    A model answering on the fractional scale (``0.85``) on a larger rubric is
    interpreted as a fraction of the scale; anything else is the raw value.
    """
    stripped = str(reply).strip()
    if not stripped or is_failure(stripped):
        return None
    match = _NUMBER_RE.search(stripped)
    if not match:
        return None
    try:
        value = float(match.group())
    except ValueError:
        return None
    scale = max_value if max_value > 0 else 100.0
    if scale > 1.0 and 0.0 <= value <= 1.0 and "%" not in stripped:
        value *= scale
    return max(0.0, min(scale, value))


# ---------------------------------------------------------------------------
#  Engine
# ---------------------------------------------------------------------------

def _time_of_day(now: datetime) -> str:
    """Human time-of-day bucket (morning/afternoon/evening/night).

    Named explicitly in the prompt so a small model does not have to derive
    "evening" from a 24-hour clock — the fact is handed to it.
    """
    hour = now.hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def _now_context() -> str:
    """Local date/time line injected into every Jev prompt.

    A Jev question can be time-dependent ("is it evening?", "is the release
    overdue?"); without the current time the small model can only guess or
    abstain.  The agent already exposes ``get_current_datetime`` — this makes
    the same fact available to every Jev call, automatically, including the
    named time of day.
    """
    now = datetime.now().astimezone()
    return (
        f"{now:%Y-%m-%d %H:%M} ({now:%A}, UTC{now:%z}); "
        f"time of day: {_time_of_day(now)}"
    )


def _build_messages(question: JevQuestion, state: str = "") -> list[dict[str, str]]:
    """The short, strict prompt for one Jev call.

    Always leads with the current local date/time as CONTEXT, then any caller
    STATE, then the typed question.
    """
    system = (
        "You are a strict classifier. Answer with EXACTLY one token and "
        "nothing else. No explanation, no punctuation, no tool calls."
    )
    lines: list[str] = [
        f"CONTEXT: current local date and time is {_now_context()}.", "",
    ]
    state = str(state or "").strip()
    if state:
        lines.extend(["STATE:", state, ""])
    if question.kind == KIND_YESNO:
        lines.append(f"QUESTION: {question.text}")
        lines.append("Answer with exactly one word: yes or no.")
    elif question.kind == KIND_CHOICE:
        lines.append(f"QUESTION: {question.text}")
        lines.append("OPTIONS:")
        for i, option in enumerate(question.options):
            lines.append(f"{chr(ord('A') + i)}) {option}")
        lines.append("Answer with exactly one option letter.")
    else:
        if question.rubric.strip():
            lines.append(f"RUBRIC: {question.rubric.strip()}")
        lines.append(f"QUESTION: {question.text}")
        lines.append(
            f"Answer with exactly one integer from 0 to {int(question.max_value)}."
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _majority_abstained(n: int, abstentions: int) -> bool:
    """True when at least half of the *n* samples were abstentions."""
    return n <= 0 or abstentions * 2 >= n


class JevEngine:
    """Ask a typed question to a dedicated small model and return probabilities.

    The engine owns a provider built from ``settings.jev_model``; it never reads
    or writes the agent's main LLM.  ``decide`` is async and safe to call from
    the REPL event loop.
    """

    def __init__(
        self,
        provider: Any,
        *,
        model_name: str = "",
        samples: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 512,
        timeout: float = 60.0,
        mechanism: str = MECHANISM_AUTO,
        threshold: float = 0.7,
        source: str = "engine",
        log_workspace: str | None = None,
    ) -> None:
        self.provider = provider
        self.model_name = model_name or str(getattr(provider, "model_name", "") or "")
        self.samples = max(1, int(samples))
        self.temperature = float(temperature)
        self.max_tokens = max(1, int(max_tokens))
        self.timeout = max(0.1, float(timeout))
        self.mechanism = mechanism if mechanism in _VALID_MECHANISMS else MECHANISM_AUTO
        self.threshold = min(1.0, max(0.0, float(threshold)))
        #: Telemetry labels: where the decision came from (repl/tool/speculate)
        #: and which workspace's reports/history/jev.jsonl to append to.
        self.source = str(source or "engine")
        self.log_workspace = log_workspace
        #: One-shot auto-select guard: the dedicated model is ensured loaded
        #: before the first request (see :meth:`_ensure_ready`).
        self._ready = False
        # Dedicated provider: safe to set sampling via the sanctioned hook.
        apply_profile = getattr(provider, "apply_profile", None)
        if callable(apply_profile):
            try:
                apply_profile("jev", self.temperature, self.max_tokens)
            except Exception:  # pragma: no cover - provider may not support it
                pass

    # -- sampling -----------------------------------------------------------

    async def _one_sample(self, messages: list[dict[str, str]]) -> str:
        """One sampled reply, or ``""`` on timeout/error (an abstention)."""
        try:
            reply = await asyncio.wait_for(
                self.provider.chat(
                    messages, max_tokens=self.max_tokens, disable_thinking=True,
                ),
                timeout=self.timeout,
            )
        except Exception:  # noqa: BLE001 - a bad sample is an abstention
            return ""
        return str(reply or "")

    async def _vote(self, messages: list[dict[str, str]]) -> list[str]:
        return list(await asyncio.gather(
            *[self._one_sample(messages) for _ in range(self.samples)]
        ))

    # -- mechanisms ---------------------------------------------------------

    async def _via_logprobs(
        self, question: JevQuestion, messages: list[dict[str, str]],
    ) -> JevResult | None:
        """Read the model's own token probabilities (one call), or None."""
        chat_logprobs = getattr(self.provider, "chat_logprobs", None)
        if not callable(chat_logprobs):
            return None
        try:
            content, top = await asyncio.wait_for(
                chat_logprobs(messages, max_tokens=self.max_tokens),
                timeout=self.timeout,
            )
        except Exception:  # noqa: BLE001 - fall back to voting
            return None
        if not top:
            return None
        dist = _top_logprobs_to_dist(question, top)
        if not dist:
            return None
        return self._result_from_probs(
            question, dist, mechanism=MECHANISM_LOGPROBS,
            n=1, abstentions=0, samples=[str(content)],
        )

    async def _via_vote(
        self, question: JevQuestion, messages: list[dict[str, str]],
    ) -> JevResult:
        replies = await self._vote(messages)
        return self._result_from_replies(question, replies)

    async def decide(self, question: JevQuestion, state: str = "") -> JevResult:
        """Ask *question* and return the typed, probabilistic answer.

        Order for ``auto``/``logprobs``: try logprobs (one call); when the
        provider does not expose it or returned nothing usable, fall back to
        sampling votes so the engine always answers with a real distribution.
        """
        if not isinstance(question, JevQuestion):
            raise TypeError("decide() expects a JevQuestion")
        await self._ensure_ready()
        messages = _build_messages(question, state)
        if self.mechanism in (MECHANISM_AUTO, MECHANISM_LOGPROBS):
            result = await self._via_logprobs(question, messages)
            if result is not None:
                self._record(question, result)
                return result
        result = await self._via_vote(question, messages)
        self._record(question, result)
        return result

    def _record(self, question: JevQuestion, result: JevResult) -> None:
        """Append the decision to the calibration ledger (best-effort).

        Telemetry must never break a decision: the import and the write are
        both guarded.  ``AGENT_NO_JEV_LOG=1`` opts out entirely.
        """
        try:
            from harnessfix.jev_telemetry import record_decision

            record_decision(
                result, question,
                workspace=self.log_workspace, source=self.source,
            )
        except Exception:  # noqa: BLE001 - telemetry is never fatal
            pass

    async def _ensure_ready(self) -> None:
        """Auto-select the dedicated model before the first request.

        Providers that can manage their own model (LM Studio) expose
        ``ensure_model_loaded``; a ``model`` switch or another shell may have
        evicted the small Jev model from VRAM, so the Jev commands re-select it
        automatically instead of failing on a "model not loaded" error.  Runs
        once per engine; best-effort (a provider without the hook, or a failed
        load, falls through to the normal request path).
        """
        if self._ready:
            return
        self._ready = True
        ensure = getattr(self.provider, "ensure_model_loaded", None)
        if not callable(ensure):
            return
        try:
            ok, message = await asyncio.to_thread(ensure)
        except Exception:  # noqa: BLE001 - availability must not break decide
            return
        if not ok:
            print(f"  [jev] could not select model {self.model_name}: {message}")

    async def ensure_ready(self) -> None:
        """Public wrapper for :meth:`_ensure_ready`.

        Consumers that use the engine's provider for their own work (e.g.
        ``speculate`` branches) call this so the dedicated model is selected
        before they run, not only when the first ``decide`` fires.
        """
        await self._ensure_ready()

    # -- aggregation --------------------------------------------------------

    def _result_from_replies(
        self, question: JevQuestion, replies: list[str],
    ) -> JevResult:
        if question.kind == KIND_YESNO:
            counts: dict[str, int] = {"yes": 0, "no": 0}
            abstentions = 0
            for reply in replies:
                label = parse_yesno(reply)
                if label is None:
                    abstentions += 1
                else:
                    counts[label] += 1
            total = counts["yes"] + counts["no"]
            if total == 0 or _majority_abstained(len(replies), abstentions):
                return JevResult(
                    kind=question.kind, model=self.model_name, mechanism=MECHANISM_VOTE,
                    decision="UNKNOWN", confidence=0.0, threshold=self.threshold,
                    n=len(replies), abstentions=abstentions, samples=replies,
                    error="" if total else "no parseable answer (all abstentions)",
                )
            probs = {"yes": counts["yes"] / total, "no": counts["no"] / total}
            return self._result_from_probs(
                question, probs, mechanism=MECHANISM_VOTE,
                n=total, abstentions=abstentions, samples=replies,
            )

        if question.kind == KIND_CHOICE:
            counts = {opt: 0 for opt in question.options}
            abstentions = 0
            for reply in replies:
                label = parse_choice(reply, question.options)
                if label is None:
                    abstentions += 1
                else:
                    counts[label] += 1
            total = sum(counts.values())
            if total == 0 or _majority_abstained(len(replies), abstentions):
                return JevResult(
                    kind=question.kind, model=self.model_name, mechanism=MECHANISM_VOTE,
                    probabilities={opt: 0.0 for opt in question.options},
                    decision="UNDECIDED", confidence=0.0, threshold=self.threshold,
                    n=len(replies), abstentions=abstentions, samples=replies,
                    error="" if total else "no parseable answer (all abstentions)",
                )
            probs = {opt: counts[opt] / total for opt in question.options}
            return self._result_from_probs(
                question, probs, mechanism=MECHANISM_VOTE,
                n=total, abstentions=abstentions, samples=replies,
            )

        # score
        values: list[float] = []
        abstentions = 0
        for reply in replies:
            value = parse_score(reply, question.max_value)
            if value is None:
                abstentions += 1
            else:
                values.append(value / question.max_value)
        if not values or _majority_abstained(len(replies), abstentions):
            return JevResult(
                kind=question.kind, model=self.model_name, mechanism=MECHANISM_VOTE,
                decision="UNKNOWN", confidence=0.0, threshold=self.threshold,
                n=len(replies), abstentions=abstentions, samples=replies,
                error="" if values else "no parseable answer (all abstentions)",
            )
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        spread = math.sqrt(var)
        return JevResult(
            kind=question.kind, model=self.model_name, mechanism=MECHANISM_VOTE,
            decision=f"{mean:.2f}", confidence=mean, threshold=self.threshold,
            n=len(values), abstentions=abstentions, value=mean, spread=spread,
            samples=replies,
        )

    def _result_from_probs(
        self,
        question: JevQuestion,
        probs: dict[str, float],
        *,
        mechanism: str,
        n: int,
        abstentions: int,
        samples: list[str],
    ) -> JevResult:
        if question.kind == KIND_YESNO:
            p_yes = probs.get("yes", 0.0)
            decision = "TRUE" if p_yes >= self.threshold else "FALSE"
            return JevResult(
                kind=question.kind, model=self.model_name, mechanism=mechanism,
                probabilities=probs, decision=decision,
                confidence=max(p_yes, 1.0 - p_yes), threshold=self.threshold,
                n=n, abstentions=abstentions, samples=samples,
            )
        # choice — the labels preserve the option order from the question.
        ordered = {opt: probs.get(opt, 0.0) for opt in question.options}
        top = max(ordered, key=lambda o: ordered[o])
        decision = top if ordered[top] >= self.threshold else "UNDECIDED"
        return JevResult(
            kind=question.kind, model=self.model_name, mechanism=mechanism,
            probabilities=ordered, decision=decision,
            confidence=ordered[top], threshold=self.threshold,
            n=n, abstentions=abstentions, samples=samples,
        )


def _letter_index(token: str) -> int | None:
    """Map an answer token to a 0-based option index, or ``None``."""
    cleaned = _clean_token(token)
    if not cleaned:
        return None
    first = cleaned[0]
    if first.isalpha():
        return ord(first) - ord("a")
    if first.isdigit():
        return int(first) - 1
    return None


def _top_logprobs_to_dist(
    question: JevQuestion, top: list[dict[str, Any]],
) -> dict[str, float] | None:
    """Collapse a first-token ``top_logprobs`` list into a label distribution.

    Returns ``None`` for the continuous ``score`` primitive (logprobs over an
    arbitrary number token is not a usable distribution) and when no token maps
    to a label.
    """
    if question.kind == KIND_SCORE:
        return None
    mass: dict[str, float] = {}
    for entry in top:
        token = entry.get("token")
        logprob = entry.get("logprob")
        if token is None or logprob is None:
            continue
        try:
            weight = math.exp(float(logprob))
        except (OverflowError, ValueError):
            continue
        if question.kind == KIND_YESNO:
            word = _clean_token(str(token))
            if word in _YES_WORDS:
                mass["yes"] = mass.get("yes", 0.0) + weight
            elif word in _NO_WORDS:
                mass["no"] = mass.get("no", 0.0) + weight
        else:
            idx = _letter_index(str(token))
            if idx is not None and 0 <= idx < len(question.options):
                option = question.options[idx]
                mass[option] = mass.get(option, 0.0) + weight
    total = sum(mass.values())
    if total <= 0:
        return None
    return {label: value / total for label, value in mass.items()}


def build_jev_engine(
    *,
    settings: Any = None,
    provider: Any = None,
    model_name: str | None = None,
    samples: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    mechanism: str = MECHANISM_AUTO,
    threshold: float = 0.7,
    source: str = "engine",
    log_workspace: str | None = None,
) -> JevEngine:
    """Build a :class:`JevEngine` bound to the configured small model.

    The provider is built from ``settings.jev_model`` (never the agent's main
    model), so Jev uses the dedicated small model exclusively.  Raises
    ``ValueError`` when no Jev model is configured, so a misconfiguration is a
    clear error instead of a silent fallback to the main model.
    """
    if settings is None:
        from agent_core.config import load_agent_settings

        settings = load_agent_settings()
    model = (
        model_name
        or str(getattr(settings, "jev_model", "") or "")
        or DEFAULT_JEV_MODEL
    )
    if not model:
        raise ValueError(
            "No Jev model configured: set AGENT_JEV_MODEL or "
            "model_catalog.json _defaults.jev_model."
        )
    if provider is None:
        from agent_core.llm.provider import build_provider

        override = str(getattr(settings, "jev_provider", "") or "") or None
        # single=True: one pinned provider, NO failover.  A failover chain
        # would silently answer with the main/reasoning model when the small
        # Jev model is unreachable — the opposite of "small model for Jev only".
        provider = build_provider(
            settings, model, provider_override=override, single=True,
        )
    return JevEngine(
        provider,
        model_name=model,
        samples=(
            samples if samples is not None
            else int(getattr(settings, "jev_samples", 5))
        ),
        temperature=(
            temperature if temperature is not None
            else float(getattr(settings, "jev_temperature", 0.7))
        ),
        max_tokens=(
            max_tokens if max_tokens is not None
            else int(getattr(settings, "jev_max_tokens", 512))
        ),
        timeout=(
            timeout if timeout is not None
            else float(getattr(settings, "jev_timeout", 60.0))
        ),
        mechanism=mechanism,
        threshold=threshold,
        source=source,
        log_workspace=log_workspace,
    )


__all__: list[str] = [
    "JevEngine",
    "JevQuestion",
    "JevResult",
    "KIND_CHOICE",
    "KIND_SCORE",
    "KIND_YESNO",
    "MECHANISM_AUTO",
    "MECHANISM_LOGPROBS",
    "MECHANISM_VOTE",
    "build_jev_engine",
    "is_failure",
    "parse_choice",
    "parse_score",
    "parse_yesno",
]
