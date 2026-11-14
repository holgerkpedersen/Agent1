"""Measure TTFT and decode TPS against any OpenAI-compatible endpoint.

One command, one table row per run. This is the anchor for any performance
claim about local inference: without it, "the NPU is fast/slow" is an opinion;
with it, a change either moves the number or it does not.

(It was originally written to check the baseline in
``docs/plans/HYBRID_NPU_IGPU_PERFORMANCE_PLAN.md``. That plan is **abandoned**
as of 2026-11-14; this tool is independent of it and remains useful for
comparing any local backend.)

Measures a *streaming* call, because the interesting numbers only exist on a
stream:

* **TTFT** - wall time from request start to the first content delta (what
  the user waits for before any text appears).
* **TPS** - completion tokens divided by the post-TTFT window, i.e. actual
  decode speed with prefill excluded.

Usage::

    # NPU via Lemonade
    python tools/perf_probe.py --base-url http://localhost:8000/api/v1 \
        --model qwen3.5-9b-FLM --mode npu

    # iGPU via LM Studio
    python tools/perf_probe.py --base-url http://localhost:1234/v1 \
        --model qwen2.5-coder-1.5b-instruct --mode igpu --repeat 3

    # longer prompt, JSON out (for scripts)
    python tools/perf_probe.py --model ... --prompt-file big.txt --json

Read-only: it sends a completion request and prints numbers.  Nothing is
written, loaded or evicted.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

DEFAULT_PROMPT = (
    "Write a Python function that merges two sorted lists into one sorted "
    "list, then explain its time complexity in two sentences."
)

#: `/chat/completions` path appended to a base URL.
CHAT_SUFFIX = "/chat/completions"


class ProbeError(RuntimeError):
    """Raised when the endpoint cannot be probed (unreachable, bad reply)."""


def _chat_url(base_url: str) -> str:
    """Normalise *base_url* into a chat-completions URL.

    Accepts the shapes servers actually advertise — ``http://host:port``,
    ``http://host:port/v1``, ``http://host:port/api/v1`` — so callers do not
    have to know which one this particular backend uses.
    """
    base = base_url.rstrip("/")
    if base.endswith(CHAT_SUFFIX):
        return base
    return base + CHAT_SUFFIX


def stream_once(
    base_url: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.1,
    timeout: float = 600.0,
    api_key: str = "not-needed",
) -> dict[str, float | int | str]:
    """Run one streamed completion and return its timing numbers.

    ``ttft_s`` is measured to the first content delta; ``tps`` excludes that
    prefill window so a fast decoder behind a long prompt is not penalised.
    Token counts come from the server's final ``usage`` chunk when it honours
    ``stream_options.include_usage``, otherwise from a ``len/4`` estimate
    (flagged in the returned ``tokens_source``).
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        _chat_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    started = time.monotonic()
    first_token_at: float | None = None
    chunks: list[str] = []
    completion_tokens = 0
    prompt_tokens = 0

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    if data == "[DONE]":
                        break
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    if usage.get("prompt_tokens"):
                        prompt_tokens = int(usage["prompt_tokens"])
                    if usage.get("completion_tokens"):
                        completion_tokens = int(usage["completion_tokens"])
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                token = delta.get("content") or ""
                if token:
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                    chunks.append(token)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        target = _chat_url(base_url)
        raise ProbeError(f"HTTP {exc.code} from {target}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ProbeError(
            f"cannot reach {_chat_url(base_url)}: {exc.reason} "
            "(is the backend running?)"
        ) from exc

    elapsed = time.monotonic() - started
    text = "".join(chunks)
    tokens_source = "usage"
    if not completion_tokens:
        completion_tokens = max(1, len(text) // 4) if text else 0
        tokens_source = "estimated(len/4)"

    ttft = (first_token_at - started) if first_token_at is not None else 0.0
    decode_window = elapsed - ttft
    tps = 0.0
    if completion_tokens and decode_window > 0:
        tps = completion_tokens / decode_window

    return {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_source": tokens_source,
        "ttft_s": round(ttft, 3),
        "total_s": round(elapsed, 3),
        "tps": round(tps, 1),
        "chars": len(text),
    }


def _aggregate(
    runs: list[dict[str, float | int | str]],
) -> dict[str, float | int | str]:
    """Median-across-runs summary (median resists a single cold-load outlier)."""
    def med(key: str) -> float:
        return round(statistics.median(float(r[key]) for r in runs), 3)

    first = dict(runs[0])
    first["ttft_s"] = med("ttft_s")
    first["total_s"] = med("total_s")
    first["tps"] = med("tps")
    first["runs"] = len(runs)
    return first


def format_table(rows: list[dict[str, float | int | str]]) -> str:
    """Render results as a fixed-width table (mode/model columns first)."""
    header = (
        f"{'mode':<8} {'model':<38} {'ttft_s':>7} {'tps':>7} "
        f"{'tok':>5} {'total_s':>8}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{str(row.get('mode', '-')):<8} "
            f"{str(row['model'])[:38]:<38} "
            f"{float(row['ttft_s']):>7.3f} "
            f"{float(row['tps']):>7.1f} "
            f"{int(row['completion_tokens']):>5} "
            f"{float(row['total_s']):>8.2f}"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure TTFT and decode TPS on an OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/api/v1",
        help="endpoint base URL (default: %(default)s, Lemonade)",
    )
    parser.add_argument("--model", required=True, help="model id to probe")
    parser.add_argument(
        "--mode",
        default="",
        help="label for the table row, e.g. npu / igpu / hybrid",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="prompt text (default: a short code-generation task)",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="read the prompt from a file (overrides --prompt)",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run N times and report the median (use >1 to warm the model)",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--json", action="store_true", help="emit JSON, not a table")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prompt = args.prompt
    if args.prompt_file:
        try:
            with open(args.prompt_file, encoding="utf-8") as handle:
                prompt = handle.read()
        except OSError as exc:
            print(f"error: cannot read prompt file: {exc}", file=sys.stderr)
            return 2

    runs: list[dict[str, float | int | str]] = []
    for attempt in range(1, max(1, args.repeat) + 1):
        try:
            result = stream_once(
                args.base_url,
                args.model,
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
            )
        except ProbeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.repeat > 1:
            print(
                f"  run {attempt}/{args.repeat}: ttft={result['ttft_s']}s "
                f"tps={result['tps']}",
                file=sys.stderr,
            )
        runs.append(result)

    summary = _aggregate(runs)
    summary["mode"] = args.mode or "-"

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(format_table([summary]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
