"""OpenRouter LLM provider — OpenAI-compatible hosted multi-model gateway.

Talks to the OpenRouter API (``https://openrouter.ai/api/v1``) over the
standard ``/chat/completions`` protocol with NATIVE tool calling: the agent's
tool schemas are passed through unchanged and ``tool_calls`` come back in the
OpenAI message shape, so no protocol translation is needed.

Model ids are namespaced ``openrouter/<vendor>/<model>`` (e.g.
``openrouter/anthropic/claude-3.5-haiku``, free variants carry a ``:free``
suffix such as ``deepseek/deepseek-chat-v3-0324:free``).  The routing prefix
is kept on :attr:`model_name` so provider resolution / persistence /
``model list`` agree; it is stripped at the HTTP boundary (see
:func:`_http_model_id`).

The API key resolves from ``OPENROUTER_API_KEY`` (env or ``.env``) first, then
from the secure store — the same three-tier secret resolution as opencode.
Errors are returned as ``[Error: ...]`` strings; transport-level failures
carry a ``(connection error)`` marker so :func:`is_connection_failure` in
:mod:`agent_core.llm.provider` fails over to the next provider correctly.

Unlike :class:`LMStudioProvider` and :class:`LlamaProvider`, OpenRouter hosts
the models — there is NO model management (no load/unload, no auto-reload).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time as _time
import urllib.error
import urllib.request
from typing import Any, Callable

from .provider import ResponseMetrics
from .pricing import estimate_cost
from agent_core.constants import DEFAULT_OPENROUTER_API_BASE, DEFAULT_OPENROUTER_MODEL

logger = logging.getLogger(__name__)

#: HTTP statuses that are safe to retry — 429 (rate limit / quota) plus the
#: classic transient gateway failures.  OpenRouter routes through many upstream
#: providers whose 5xx blips should not abort a workflow run.
_TRANSIENT_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})

#: OpenRouter HTTP 400 phrase returned when an endpoint *mandates* reasoning
#: and the request tried to disable it. We detect this to transparently retry
#: without the disable knob (the model reasons anyway) so the turn survives.
_REASONING_MANDATORY_400 = "reasoning is mandatory"

#: OpenRouter 402 phrase when the requested ``max_tokens`` exceeds what the
#: account's remaining credits can afford. We parse the "can only afford N"
#: number out of it to transparently retry with a smaller budget instead of
#: surfacing a hard failure (the turn still completes with fewer tokens).
_CREDITS_AFFORD_RE = re.compile(
    r"can only afford\s+(\d+)", re.IGNORECASE
)

DEFAULT_API_BASE = DEFAULT_OPENROUTER_API_BASE


def _http_model_id(model_name: str) -> str:
    """Strip the ``openrouter/`` routing prefix for the request payload.

    OpenRouter model ids are already namespaced (``vendor/model``); internally
    we keep the extra ``openrouter/`` prefix so provider_for() / persistence /
    ``model list`` can tell this provider apart from LM Studio models that
    happen to contain a slash.  The HTTP request must use the bare id.
    """
    if model_name.startswith("openrouter/"):
        return model_name[len("openrouter/"):]
    return model_name


def _is_free_model(meta: dict[str, Any]) -> bool:
    """Return True when an OpenRouter /models entry is on the free tier.

    OpenRouter marks free models with zero pricing (``prompt`` and
    ``completion`` both ``"0"``); they also conventionally carry a ``:free``
    suffix.  A model is free only when its pricing is entirely zero — paid
    models that merely *look* free (or have no pricing info) are excluded so
    `model list` does not surface models that need credits.
    """
    if not isinstance(meta, dict):
        return False
    pricing = meta.get("pricing")
    if isinstance(pricing, dict):

        def _zero(v: Any) -> bool:
            try:
                return float(v or 0) == 0.0
            except (TypeError, ValueError):
                return str(v) == "0"

        return _zero(pricing.get("prompt")) and _zero(pricing.get("completion"))
    # No pricing info — fall back to the :free suffix convention.
    return str(meta.get("id", "")).endswith(":free")


def _reasoning_exhausted(
    content: str, reasoning: str, finish_reason: str | None = None
) -> str | None:
    """Detect when a reasoning model used its budget thinking and left no usable
    output.  Returns an actionable ``[Error: ...]`` string instead of silently
    collapsing to ``(no output)`` — callers retry with thinking disabled rather
    than burning more reasoning tokens (mirrors lmstudio/opencode behaviour).

    NOTE: a model that legitimately returns only a ``reasoning`` block with no
    ``content`` is NOT "exhausted" — that is a pure-reasoning response and is
    recovered by :func:`_postprocess_content`.  We only flag exhaustion when the
    turn was actually truncated (``finish_reason`` is ``"length"`` or absent),
    never on a clean ``stop`` — otherwise a valid reasoning-only answer would be
    misreported as a failure and the caller would uselessly retry.
    """
    truncated = finish_reason in (None, "length")
    if (
        truncated
        and not content.strip()
        and reasoning
        and len(reasoning) > 300
    ):
        return (
            f"[Error: model consumed {len(reasoning)} reasoning bytes with "
            "no output — reasoning models burn their budget thinking. Retry "
            "with disable_thinking=True (sends reasoning.enabled=false to "
            "OpenRouter).]"
        )
    if (
        finish_reason == "length"
        and reasoning
        and len(reasoning) > 300
        and len(content) < len(reasoning)
    ):
        return (
            f"[Error: model hit the output limit ({len(reasoning)} reasoning "
            "bytes) and the response was truncated before the output completed. "
            "Retry with disable_thinking=True or a larger output budget.]"
        )
    return None


def _postprocess_content(
    content: str, reasoning: str, finish_reason: str | None = None
) -> str:
    """Return usable text for a completed turn.

    Some OpenRouter endpoints are *pure reasoning* models (e.g. the hosted
    ``deepseek/deepseek-r1``): they return ``content=None`` and only a
    ``reasoning`` block.  For those we fall back to the reasoning text so the
    turn is not silently collapsed to ``(no output)``.  We only do this when
    there is no real content and the reasoning is substantial — a tiny
    reasoning stub is not a usable answer.
    """
    if content and content.strip():
        return content
    if reasoning and reasoning.strip() and len(reasoning.strip()) > 1:
        return reasoning
    return ""


def _affordable_tokens(detail: str) -> int | None:
    """Parse the affordable-token count out of a 402 detail string.

    OpenRouter returns e.g. ``"...can only afford 34782..."`` when the
    requested ``max_tokens`` exceeds the account's remaining credit budget.
    Returns that integer (minus a small safety margin) or ``None`` when the
    phrase is absent.
    """
    m = _CREDITS_AFFORD_RE.search(detail or "")
    if not m:
        return None
    try:
        return max(1, int(m.group(1)) - 256)
    except (TypeError, ValueError):
        return None


def _read_http_error_detail(exc: "urllib.error.HTTPError") -> str:
    """Return the gateway's error detail string from an ``HTTPError``."""
    try:
        body = exc.read().decode("utf-8", "replace").strip()
    except Exception:  # noqa: BLE001 - the fp may already be consumed
        body = ""
    return body or str(exc.reason)


def _format_http_error(code: int, detail: str) -> str:
    """Turn an OpenRouter HTTP error into an actionable ``[Error: ...]`` string.

    Free-tier models fail in two ways the user cannot fix by retrying the same
    id: a **403** means the model is "only available on agentic harnesses"
    (restricted — it cannot be called via the raw API), and a **429** means the
    free-tier model is "temporarily rate-limited upstream".  Both are surfaced
    with a concrete suggestion to switch to a different free model instead of a
    raw upstream JSON blob (which is what previously reached the user).
    """
    lowered = (detail or "").lower()
    if code == 403 and "agentic harness" in lowered:
        return (
            "[Error: openrouter model is restricted — \"only available on "
            "agentic harnesses\", so it cannot be called via the API. Pick a "
            "different free model, e.g. "
            f"'{OpenRouterProvider.suggest_free_model()}'.]"
        )
    if code == 429 or "rate-limit" in lowered or "rate limited" in lowered or "too many requests" in lowered:
        return (
            "[Error: openrouter free-tier model is rate-limited upstream "
            "(HTTP 429). The free tier is throttled — wait a moment or switch to "
            f"a different free model, e.g. '{OpenRouterProvider.suggest_free_model()}'.]"
        )
    return f"[Error: openrouter API request failed: HTTP Error {code}: {detail}]"


class OpenRouterProvider:
    """Concrete LLM provider for the hosted OpenRouter gateway."""

    def __init__(
        self,
        model_name: str | None = None,
        api_url: str | None = None,
        api_key: str | None = None,
        max_retries: int | None = None,
        retry_base_delay: float = 2.0,
    ) -> None:
        name = model_name or DEFAULT_OPENROUTER_MODEL
        # Keep the routing prefix so provider detection / persistence agree
        # (the bare id is only added back at the HTTP boundary).
        self.model_name = (
            name if name.startswith("openrouter/") else f"openrouter/{name}"
        )
        self.api_url = (api_url or os.environ.get("OPENROUTER_API_URL") or DEFAULT_API_BASE).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.temperature: float = 0.7
        self.max_tokens: int = 50000
        self._profile_name: str | None = None
        self.last_response_metrics: ResponseMetrics | None = None
        self._last_label: str | None = None
        #: Long reads are normal for big prompts (workflow plan steps); 600s
        #: matches the other hosted providers. Override via OPENROUTER_TIMEOUT.
        self._api_timeout = float(os.environ.get("OPENROUTER_TIMEOUT", "600"))
        #: Transient-failure retry (default 3 attempts, exponential backoff).
        #: A single intermittent gateway/upstream 5xx must not abort a run.
        self._max_retries = (
            max(0, int(os.environ.get("OPENROUTER_MAX_RETRIES", "3")))
            if max_retries is None
            else max(0, int(max_retries))
        )
        self._retry_base_delay = max(0.0, float(retry_base_delay))

    def apply_profile(self, name: str, temperature: float, max_tokens: int) -> None:
        """Activate *name* with its sampling parameters in one step."""
        self._profile_name = name
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Labeling (once per session, like the other providers)
    # ------------------------------------------------------------------

    def _label(self) -> str:
        label = f"[model: {self.model_name} | provider=openrouter]"
        if self._profile_name:
            label += f" profile={self._profile_name}"
        if label != self._last_label:
            print(f"  {label}", end="", flush=True)
            self._last_label = label
        return label

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _headers(self, json_body: bool = True) -> dict[str, str]:
        headers: dict[str, str] = {
            #: The gateway rejects the default "Python-urllib" user agent.
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        if json_body:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # Optional attribution headers (OpenRouter ranks referring apps in its
        # rankings; harmless elsewhere).
        headers["HTTP-Referer"] = "http://localhost/agent1"
        headers["X-Title"] = "Agent1"
        return headers

    def _request(
        self, method: str, url: str, body: Any = None, timeout: float | None = None
    ) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            url, data=data, headers=self._headers(body is not None), method=method
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else None

    async def _with_retry(
        self, factory: Callable[[], Any], *, label: str
    ) -> Any:
        """Run *factory* with exponential backoff on transient failures.

        Retries HTTP 429/5xx responses and network timeouts — the gateway sits
        in front of many upstream providers that intermittently return 5xx on
        healthy requests. Permanent client errors (4xx other than 429)
        propagate immediately. Raises the last error after retries are
        exhausted so callers can format it as an ``[Error: ...]`` string.

        The factory performs a BLOCKING urllib call — it is dispatched to a
        worker thread so it never stalls the event loop (a sync HTTP round-trip
        on the loop would serialize every concurrent model in run_parallel()).
        """
        last_error: BaseException | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await asyncio.to_thread(factory)
            except urllib.error.HTTPError as exc:
                if exc.code not in _TRANSIENT_HTTP_STATUSES:
                    raise
                last_error = exc
            except (TimeoutError, OSError) as exc:
                last_error = exc
            if attempt < self._max_retries:
                wait = self._retry_base_delay * (2 ** attempt)
                print(
                    f"  [retry {attempt + 1}/{self._max_retries}] {label}: "
                    f"{last_error}, waiting {wait:.0f}s...",
                    flush=True,
                )
                await asyncio.sleep(wait)
        assert last_error is not None
        raise last_error

    # ------------------------------------------------------------------
    # Chat (OpenAI-compatible, NATIVE tool calling)
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
    ) -> str:
        """Send chat request to OpenRouter.

        *disable_thinking* sends ``reasoning: {"enabled": false}`` — the
        unified OpenRouter knob that stops reasoning models from burning their
        whole output budget on thinking tokens and returning empty content.
        """
        self._label()
        if not self.api_key:
            return (
                "[Error: no OPENROUTER_API_KEY set — add it to .env or the "
                "secure store to use the openrouter provider]"
            )
        from .lmstudio import sanitize_message_roles

        payload: dict[str, Any] = {
            "model": _http_model_id(self.model_name),
            "messages": sanitize_message_roles(messages),
            "temperature": self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
        if disable_thinking:
            # OpenRouter expects a DICT here ({"enabled": false}); a plain
            # string or the LM Studio "reasoning": "off" knob is rejected.
            payload["reasoning"] = {"enabled": False}

        start = _time.monotonic()
        try:
            result = await self._with_retry(
                lambda: self._request(
                    "POST", f"{self.api_url}/chat/completions", payload,
                    timeout=self._api_timeout,
                ),
                label="openrouter chat/completions",
            )
        except urllib.error.HTTPError as exc:
            detail = _read_http_error_detail(exc)

            repaired = False

            # --- Fix A: endpoint mandates reasoning and rejects {"enabled": false}. ---
            if (
                exc.code == 400
                and disable_thinking
                and _REASONING_MANDATORY_400 in detail.lower()
            ):
                payload["reasoning"] = {"enabled": True}
                try:
                    result = await self._with_retry(
                        lambda: self._request(
                            "POST", f"{self.api_url}/chat/completions", payload,
                            timeout=self._api_timeout,
                        ),
                        label="openrouter chat/completions (reasoning enabled)",
                    )
                    repaired = True
                except urllib.error.HTTPError as exc2:
                    detail = _read_http_error_detail(exc2)
                    if (
                        exc2.code == 400
                        and disable_thinking
                        and _REASONING_MANDATORY_400 in detail.lower()
                    ):
                        payload.pop("reasoning", None)
                        try:
                            result = await self._with_retry(
                                lambda: self._request(
                                    "POST", f"{self.api_url}/chat/completions", payload,
                                    timeout=self._api_timeout,
                                ),
                                label="openrouter chat/completions (no reasoning knob)",
                            )
                            repaired = True
                        except urllib.error.HTTPError as exc3:
                            detail = _read_http_error_detail(exc3)
                            return _format_http_error(exc3.code, detail)
                        except (urllib.error.URLError, TimeoutError, OSError) as exc3:
                            return (
                                f"[Error: openrouter API request failed (connection "
                                f"error): {exc3}]"
                            )
                        except Exception as exc3:  # noqa: BLE001 - defensive per-turn
                            return f"[Error: openrouter API request failed: {exc3}]"
                    else:
                        return _format_http_error(exc2.code, detail)
                except (urllib.error.URLError, TimeoutError, OSError) as exc2:
                    return (
                        f"[Error: openrouter API request failed (connection "
                        f"error): {exc2}]"
                    )
                except Exception as exc2:  # noqa: BLE001 - defensive per-turn
                    return f"[Error: openrouter API request failed: {exc2}]"

            # --- Fix B: credits exhausted — transparently retry with the largest max_tokens we can afford. ---
            if not repaired and exc.code == 402:
                affordable = _affordable_tokens(detail)
                if affordable is not None:
                    new_max = max(1, min(self.max_tokens, affordable))
                    payload["max_tokens"] = new_max
                    logger.info(
                        "OpenRouter credits exhausted (requested=%s); retrying with max_tokens=%d",
                        self.max_tokens,
                        new_max,
                    )
                    try:
                        result = await self._with_retry(
                            lambda: self._request(
                                "POST", f"{self.api_url}/chat/completions", payload,
                                timeout=self._api_timeout,
                            ),
                            label=f"openrouter chat/completions (reduced max_tokens={new_max})",
                        )
                        repaired = True
                    except urllib.error.HTTPError as exc2:
                        detail = _read_http_error_detail(exc2)
                        return _format_http_error(exc2.code, detail)
                    except (urllib.error.URLError, TimeoutError, OSError) as exc2:
                        return (
                            f"[Error: openrouter API request failed (connection "
                            f"error): {exc2}]"
                        )
                    except Exception as exc2:  # noqa: BLE001 - defensive per-turn
                        return f"[Error: openrouter API request failed: {exc2}]"

            # --- Fix C: model doesn't support tool use — strip tools and retry. ---
            if not repaired and exc.code == 404 and "support tool" in (detail or "").lower():
                payload.pop("tools", None)
                try:
                    result = await self._with_retry(
                        lambda: self._request(
                            "POST", f"{self.api_url}/chat/completions", payload,
                            timeout=self._api_timeout,
                        ),
                        label="openrouter chat/completions (no tools)",
                    )
                    repaired = True
                except urllib.error.HTTPError as exc2:
                    detail = _read_http_error_detail(exc2)
                    return _format_http_error(exc2.code, detail)
                except (urllib.error.URLError, TimeoutError, OSError) as exc2:
                    return (
                        f"[Error: openrouter API request failed (connection "
                        f"error): {exc2}]"
                    )
                except Exception as exc2:  # noqa: BLE001 - defensive per-turn
                    return f"[Error: openrouter API request failed: {exc2}]"

            if not repaired:
                return _format_http_error(exc.code, detail)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # Transport-level failure — the "(connection error)" marker makes
            # is_connection_failure() fail over to the next provider.
            return f"[Error: openrouter API request failed (connection error): {exc}]"
        except Exception as exc:  # noqa: BLE001 - defensive per-turn
            return f"[Error: openrouter API request failed: {exc}]"
        elapsed_ms = (_time.monotonic() - start) * 1000.0

        choices = result.get("choices") if isinstance(result, dict) else None
        if not choices:
            return "[Error: openrouter API returned no choices]"
        first = choices[0] if isinstance(choices, list) else choices
        message = first.get("message") or {}
        content_raw = message.get("content")
        reasoning_raw = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or ""
        )
        finish_reason = first.get("finish_reason")

        tool_calls = message.get("tool_calls")
        # Per-turn token/latency accounting (plan ARCH item 17).
        usage = result.get("usage") if isinstance(result, dict) else None
        prompt_tokens = int(usage.get("prompt_tokens") or 0) if isinstance(usage, dict) else 0
        completion_tokens = int(usage.get("completion_tokens") or 0) if isinstance(usage, dict) else 0
        self.last_response_metrics = ResponseMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=elapsed_ms,
            cost=estimate_cost(prompt_tokens, completion_tokens, self.model_name, "openrouter"),
        )
        if tool_calls:
            return json.dumps({"content": content_raw, "tool_calls": tool_calls})
        thinking_err = _reasoning_exhausted(
            str(content_raw or ""), str(reasoning_raw or ""), finish_reason
        )
        if thinking_err:
            return thinking_err
        content = _postprocess_content(
            str(content_raw or ""), str(reasoning_raw or ""), finish_reason
        )
        return content if content else "(no output)"

    async def chat_stream(self, messages: list[dict[str, Any]]) -> str:
        """No streaming support — return the full text response."""
        return await self.chat(messages)

    async def analyze_code(self, code: str) -> str:
        """Analyze code using the OpenRouter model."""
        return await self.chat([
            {"role": "system", "content": "Analyze the following code for bugs and improvements."},
            {"role": "user", "content": code},
        ])

    # ------------------------------------------------------------------
    # model command support
    # ------------------------------------------------------------------

    def list_models(self, free_only: bool = True) -> list[str]:
        """Live OpenRouter catalog (GET /models), namespaced openrouter/<id>.

        Returns ``[]`` on any failure so `model list` never crashes — the rest
        of the listing still works.

        By default only the **free tier** is returned: the agent's owner wants
        to use OpenRouter's no-cost models, and the full catalog is dominated by
        paid models that would otherwise clutter the list.  A model is free when
        its pricing is entirely zero (or, failing that, its id carries a ``:free``
        suffix).  Pass ``free_only=False`` to fetch the complete catalog (paid
        + free) — used by ``model openrouter --all`` / ``--paid``.
        """
        try:
            data = self._request("GET", f"{self.api_url}/models", timeout=15)
        except Exception:  # noqa: BLE001 - listing must never crash
            return []
        items = data.get("data") if isinstance(data, dict) else []
        out: list[str] = []
        for m in (items or []):
            if not isinstance(m, dict) or not m.get("id"):
                continue
            if free_only and not _is_free_model(m):
                continue
            out.append(f"openrouter/{m['id']}")
        return sorted(out)

    @staticmethod
    def suggest_free_model() -> str | None:
        """Return a stable, broadly-available free model id (``openrouter/<id>``).

        Used as a friendly fallback when the user picks a free model that turns
        out to be restricted (403) or rate-limited (429).  ``None`` when no
        suggestion is known (callers then just print the raw error).
        """
        return DEFAULT_OPENROUTER_MODEL
