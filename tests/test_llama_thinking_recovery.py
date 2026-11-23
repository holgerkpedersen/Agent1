"""Regression tests: llama-server provider recovers from a spent thinking budget.

User scenario (live): a reasoning model served by ``llama-server`` spent its
whole output budget inside the thinking block and emitted nothing.  The turn
ended with::

    [llm-error] The model did not produce a usable response:
    [Error: model consumed 1656 reasoning bytes with no output. ...]

and the only way forward was to switch models by hand, even though llama-server
honours ``reasoning: "off"`` + ``chat_template_kwargs.enable_thinking=False``
(both already set by ``_build_payload(disable_thinking=True)``).

Contract under test (mirrors LMStudioProvider's one-shot recovery):
1. non-streaming ``chat`` retries ONCE with thinking disabled and a larger
   ``max_tokens``, and returns the recovered answer instead of the error;
2. streaming ``chat_stream`` does the same when the stream produced reasoning
   but no visible content;
3. the recovery is one-shot only — an explicit ``disable_thinking=True`` call
   never retries (the caller already asked for the weak path);
4. a normal answer is returned untouched (no needless second request).
"""

import asyncio
import json
from io import BytesIO

from agent_core.llm.llama_provider import LlamaProvider
from agent_core.llm.lmstudio import _escalated_max_tokens

#: Long enough to trip ``_check_thinking_error`` (> 500 reasoning bytes).
_REASONING = "thinking " * 80


def _provider(max_tokens: int = 250) -> LlamaProvider:
    prov = LlamaProvider(model_name="llama/x", api_url="http://h/v1")
    prov.max_tokens = max_tokens
    # Skip the fail-soft GET /v1/models context probe: this test is offline.
    prov._context_probed = True
    return prov


def _exhausted_response() -> dict:
    return {
        "choices": [
            {
                "message": {"content": "", "reasoning_content": _REASONING},
                "finish_reason": "length",
            }
        ]
    }


def _answer_response(text: str = "final answer") -> dict:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
    }


class _FakeHTTPResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _sse_bytes(*chunks: dict) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


class TestChatThinkingRecovery:
    def test_chat_retries_with_thinking_disabled_and_returns_answer(self):
        prov = _provider(max_tokens=250)
        sent: list[dict] = []
        answers = [_exhausted_response(), _answer_response()]

        def fake_make_request(payload, timeout=None):
            sent.append(payload)
            return answers[len(sent) - 1]

        prov._make_request = fake_make_request  # type: ignore[method-assign]
        result = asyncio.run(prov.chat([{"role": "user", "content": "hi"}]))

        assert result == "final answer"
        assert len(sent) == 2, "exactly one recovery retry"
        assert sent[0].get("reasoning") is None
        # The retry disables thinking AND grows the budget (the exhaustion
        # cannot be cured by the reasoning switch alone for every model).
        assert sent[1]["reasoning"] == "off"
        assert sent[1]["chat_template_kwargs"] == {"enable_thinking": False}
        assert sent[1]["max_tokens"] == _escalated_max_tokens(250) == 4096

    def test_chat_does_not_retry_when_thinking_already_disabled(self):
        prov = _provider()
        sent: list[dict] = []

        def fake_make_request(payload, timeout=None):
            sent.append(payload)
            return _exhausted_response()

        prov._make_request = fake_make_request  # type: ignore[method-assign]
        result = asyncio.run(
            prov.chat([{"role": "user", "content": "hi"}], disable_thinking=True)
        )

        assert len(sent) == 1, "caller asked for thinking off — no retry loop"
        assert "reasoning bytes" in result

    def test_chat_normal_answer_is_not_retried(self):
        prov = _provider()
        sent: list[dict] = []

        def fake_make_request(payload, timeout=None):
            sent.append(payload)
            return _answer_response("straight answer")

        prov._make_request = fake_make_request  # type: ignore[method-assign]
        result = asyncio.run(prov.chat([{"role": "user", "content": "hi"}]))

        assert result == "straight answer"
        assert len(sent) == 1


class TestChatStreamThinkingRecovery:
    def _patch_urlopen(self, monkeypatch, payloads: list[dict]):
        """Serve canned SSE responses, recording the payload of every request."""
        exhausted = _sse_bytes(
            {"choices": [{"delta": {"reasoning_content": _REASONING}}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}]},
        )
        answered = _sse_bytes(
            {"choices": [{"delta": {"content": "hej"}}]},
            {"choices": [{"delta": {"content": " there"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        )
        bodies = [exhausted, answered]

        def fake_urlopen(req, timeout=None):
            payloads.append(json.loads(req.data.decode("utf-8")))
            return _FakeHTTPResponse(bodies[len(payloads) - 1])

        monkeypatch.setattr(
            "agent_core.llm.llama_provider.urllib.request.urlopen", fake_urlopen
        )

    def test_stream_retries_with_thinking_disabled_and_returns_answer(
        self, monkeypatch
    ):
        prov = _provider(max_tokens=250)
        payloads: list[dict] = []
        self._patch_urlopen(monkeypatch, payloads)

        result = asyncio.run(prov.chat_stream([{"role": "user", "content": "hi"}]))

        assert result == "hej there"
        assert len(payloads) == 2, "exactly one recovery retry"
        assert payloads[0].get("reasoning") is None
        assert payloads[1]["reasoning"] == "off"
        assert payloads[1]["chat_template_kwargs"] == {"enable_thinking": False}
        assert payloads[1]["max_tokens"] == _escalated_max_tokens(250) == 4096
        # stream_options must survive the rebuild — TPS needs real token totals.
        assert payloads[1]["stream_options"] == {"include_usage": True}

    def test_stream_normal_answer_is_not_retried(self, monkeypatch):
        prov = _provider()
        payloads: list[dict] = []
        answered = _sse_bytes(
            {"choices": [{"delta": {"content": "ok"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        )

        def fake_urlopen(req, timeout=None):
            payloads.append(json.loads(req.data.decode("utf-8")))
            return _FakeHTTPResponse(answered)

        monkeypatch.setattr(
            "agent_core.llm.llama_provider.urllib.request.urlopen", fake_urlopen
        )

        result = asyncio.run(prov.chat_stream([{"role": "user", "content": "hi"}]))

        assert result == "ok"
        assert len(payloads) == 1
