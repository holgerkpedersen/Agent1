# Plan — Stop relying on opencode-zen free models (quota-exhausted dead end)

Status: **Tasks 1-9 implemented** (2026-11-27)
Date: 2026-11-27

>> Implementation note: ALL tasks are DONE. Tasks 1-3 (classify the quota 429
>> once, stop retrying it, open the failover gate) live in
>> `agent_core/llm/opencode_provider.py` + `agent_core/llm/provider.py`, pinned
>> by `tests/test_zen_quota_classification.py` (14 tests). Tasks 4-7 (the cached
>> tier-level `QuotaGuard` in `agent_core/llm/quota_guard.py`, the 0 s fast path,
>> the suppressed rotation and the honest keyless catalog) plus the Task 8
>> regression suite live in `tests/unit/test_zen_quota_guard.py` (21 tests).
>> Task 9 documentation landed in `agent_core/config.py` (caveat next to the zen
>> settings), `README.md` ("Zen free tier (not recommended)") and `CHANGES.md`.
>>
>> One correction found while implementing Task 6: `chat()` must NOT gate the
>> rotation on re-matching the returned text. The quota message is a friendly
>> rewrite that no longer carries the raw `FreeUsageLimitError` body, so the
>> guard itself is the authoritative signal (the body primes it inside
>> `_chat_api`). Re-matching text silently re-enabled the rotation.

## Problem

The zen free tier does not work, and the current code makes that failure slow and
opaque rather than routing around it. Measured against the live endpoint:

```
opencode-zen/beam-free                -> 429 FreeUsageLimitError
opencode-zen/deepseek-v4.1-flash-free -> 429 FreeUsageLimitError
opencode-zen/flint-40b-free           -> 429 FreeUsageLimitError
opencode-zen/hy4-flash-free           -> 429 FreeUsageLimitError
```

The gateway body is explicit:

> `{"type":"error","error":{"type":"FreeUsageLimitError","message":"Free usage limit
> reached. Free models work in OpenCode (https://opencode.ai); subscribe to OpenCode Go
> to keep going: https://opencode.ai/go"}}`

Two facts follow from that evidence:

1. **The quota is account-wide, not per-model.** Every free model returns the same
   429 at the same moment. Rotating between free models therefore cannot succeed — it
   only multiplies latency.
2. **The free tier is gated to the OpenCode app**, not to a third-party harness. This
   is a product restriction, not an outage, so it will not "come back later" for us.

### Scope check — the default path is already healthy

Verified in `.env`:

```
AGENT_MODEL=opencode-go/deepseek-v4.1-flash
AGENT_LLM_PROVIDERS=opencode:opencode-go/deepseek-v4.1-flash,openrouter,lmstudio,llama
```

The first chain entry is the **go** (keyed) tier, so nothing is broken by default.
The failure only bites once a user selects a zen/free model — `llm/provider.py`
then promotes that tier's slot to the front (`provider.py:525-556`) and the turn
stalls on a tier that cannot answer. So this plan is about the *selection* path,
not the shipped default.

## Root cause: the 429 is classified as retryable but not as failover-worthy

Four separate classifiers disagree about the same error string. Three treat it as
something we should keep hammering; only the correct one (provider failover) is never
reached.

| Site | Verdict on the 429 | Consequence |
|---|---|---|
| `llm/opencode_provider.py:48` `_TRANSIENT_HTTP_STATUSES` | retryable | 3 retries, 2+4+8s = **14 s wasted per turn** |
| `opencode_provider.py:339` `_BACKEND_DOWN_MARKERS` | not down | `_zen_free_fallback` never triggers |
| `llm/provider.py:604` `_CONNECTION_FAILURE_RE` | **not** a connection failure | `FailoverProvider` never tries the next provider |

Concrete evidence (probe over the real code path):

```
is_provider_error         = True
is_connection_failure     = False   <- failover gate: closed
provider._is_backend_down = False   <- zen fallback gate: closed
429 in _TRANSIENT         = True    <- retried 3x
backend-down markers hit  = []
is_free_model(zen id)     = True
```

So the turn does exactly the worst thing available: it waits 14 s, gives up, and
returns `[Error: ...]` **as if it were the model's answer**, without ever letting the
healthy providers in `AGENT_LLM_PROVIDERS` (openrouter, lmstudio, llama) answer.

### Secondary defect: the "free" catalog advertises 100 paid models

`list_models()` on the zen tier returns the whole gateway catalog:

```
zen catalog total     = 110
  suffixed '-free'    = 10
  NOT free (paid ids) = 100   e.g. opencode-zen/claude-opus-4-6, opencode-zen/big-pickle
```

Picking any of those 100 with a keyless client yields an error the user cannot
distinguish from a bug in this repo. This is why "they don't seem to work anyway"
is the subjective experience: most of what the catalog offers under that prefix is
not usable on the free tier at all.

## Goal

Never let a zen free-tier quota error cost a turn, and never present it as an answer:

- a known quota-exhausted tier is skipped **before** the request (0 s, not 14 s);
- if one slips through, it is classified as **not retryable** and **not an answer**,
  so `FailoverProvider` moves to the next configured provider immediately;
- the user gets one clear line explaining that zen needs a subscription, not a stack
  of retries.

## Non-goals

- Deleting zen support. A user with a Go subscription, or with quota restored, must
  keep working exactly as today.
- Guessing quota state from timing. Only an observed 429 changes the cached state.

## Tasks

### Task 1 — classify the quota error once, as a single source of truth

Add to `agent_core/llm/opencode_provider.py` a body-marker set and a predicate:

```python
#: Gateway bodies that mean "this ACCOUNT is out of free quota" — permanent for
#: the tier until the reset window, not a transient outage.
_QUOTA_BODY_MARKERS = ("freeusagelimiterror", "free usage limit reached")

def _is_quota_exhausted(text: str) -> bool: ...
```

Match case-insensitively on a whitespace-stripped body, mirroring the existing
`_SERVER_ERROR_BODY_MARKERS` style at `opencode_provider.py:55`.

### Task 2 — stop retrying a permanent quota error

`_is_transient_http_error` (`opencode_provider.py:79`) currently returns `True` for
every 429 before it looks at the body. Check the body **first**:

```python
body = re.sub(r"\s+", "", _http_error_body(exc)).lower()
if any(m in body for m in _QUOTA_BODY_MARKERS):
    return False          # permanent for this tier — do not burn retries
if exc.code in _TRANSIENT_HTTP_STATUSES:
    return True
```

A plain 429 with no quota marker stays retryable (genuine rate limiting still
deserves backoff). This alone removes the 14 s stall.

### Task 3 — open the provider failover gate

`is_connection_failure` (`llm/provider.py:625`) is what lets `FailoverProvider` try the
next provider; the regex at `provider.py:604-622` has no quota alternative. Add:

```
r"free usage limit reached|freeusagelimiterror|"
```

so quota exhaustion is treated like the existing
`openrouter free-tier model is rate-limited` alternative — the closest existing
precedent in that regex, and the same product shape.

Care: the regex is anchored on `startswith("[Error:")` and matched with `re.search`,
so the existing guard against successful tool-call payloads keeps holding.

### Task 4 — trip a cached, tier-level quota state (the actual "better way")

New module `agent_core/llm/quota_guard.py`, deliberately tiny and dependency-free:

```python
class QuotaGuard:
    """Remembers which provider tiers are out of free quota, with an expiry."""
    def is_exhausted(self, tier: str) -> bool: ...
    def mark_exhausted(self, tier: str, retry_after: float | None) -> None: ...
    def reset(self, tier: str | None = None) -> None: ...
```

- Process-local (no new files, no cross-session state): quota is account-wide and
  per-run, and a restart is the natural reset.
- Honours `Retry-After` when the gateway sends one, else a default cooldown
  (`AGENT_QUOTA_COOLDOWN_SECONDS`, default 900).
- Keyed by **tier** (`"opencode-zen"`), not by model — that is the measured reality.

### Task 5 — consult the guard before spending a request

In `OpencodeProvider.chat()` and `_chat_api()`, when `self.zen_mode` and the guard
reports the tier exhausted, return the actionable message **without any HTTP call**:

```
[Error: opencode-zen free tier quota exhausted (shared across all free models).
Subscribe to OpenCode Go (https://opencode.ai/go) or switch provider: `model list`.]
```

On receiving a quota body, call `mark_exhausted("opencode-zen", retry_after)`. This
makes the second and every later turn in a session cost 0 s instead of 14 s + a
retry cascade.

### Task 6 — collapse the pointless free-model rotation

`_zen_free_fallback` (`opencode_provider.py:1038`) tries each free model on the
assumption the backend marked *one model* down. For quota that premise is false. Once
Task 1's predicate fires, skip the rotation and return the Task 5 message directly.
Keep the rotation for its original purpose (`_BACKEND_DOWN_MARKERS`: 5xx, timeouts).

### Task 7 — make the zen catalog honest

In `list_models()` (`opencode_provider.py:1104`) the keyless tier must list only what
the keyless tier can actually call:

- keep ids ending `-free`;
- drop paid ids for the keyless/zen tier (they still appear for the keyed go tier);
- when the guard says the tier is exhausted, append one note instead of failing
  silently:
  `(10 free models — free-tier quota currently exhausted)`.

Not hiding models that a Go subscriber can use; only the keyless listing changes.

### Task 8 — regression tests

New `tests/unit/test_zen_quota_guard.py` (the shell is pure, so no network):

1. `_is_transient_http_error` returns **False** for a 429 whose body carries
   `FreeUsageLimitError` (build a real `urllib.error.HTTPError` with a `BytesIO` fp —
   the same trick the existing suite uses for body-carrying errors).
2. It still returns **True** for a bare 429 and for 503.
3. `is_connection_failure(<the recorded live 429 string>)` is **True**, and stays
   **False** for a successful tool-call payload that merely mentions `[Error:`.
4. `QuotaGuard`: mark → `is_exhausted` True; expiry by an injected clock → False;
   `reset()` clears; the key is the tier, so a different free *model* is still blocked.
5. `OpencodeProvider("opencode-zen/beam-free")` with an already-exhausted guard raises
   **no** network call — assert by monkeypatching `_request` to fail the test if
   touched (proves the 0 s fast path).
6. `list_models()` on the keyless tier returns no id lacking the `-free` suffix,
   using a stubbed `/models` payload (no live HTTP in the suite).

Use a frozen/offset timestamp rather than `sleep` for the expiry test.

### Task 9 — document the constraint where a user will hit it

- `.env` comment block next to `AGENT_ZEN_FREE_DEFAULT` / `AGENT_ZEN_FREE_FALLBACKS`
  (`agent_core/config.py:221-237`) stating the quota is account-wide and gated to the
  OpenCode app.
- `README`/docs model section: recommend a real provider
  (`AGENT_MODEL`, currently `opencode-go/deepseek-v4.1-flash`) as the default and keep
  zen as an opt-in for subscribers.
- One line in `CHANGES.md`.

## What to recommend instead (the "better way")

Ordered by cost to the user, and each already present in this repo:

1. **Stay on the keyed go tier** — `AGENT_MODEL=opencode-go/deepseek-v4.1-flash` is
   already set in `.env` and shares the code path, so nothing new is needed. This is
   the default recommendation.
2. **Local models via LM Studio / llama-server** — no quota at all, already in
   `AGENT_LLM_PROVIDERS`, and the repo already treats them as first-class
   (`is_connection_failure`, `lmstudio.py` tool-capability failover).
3. **OpenRouter free-tier ids** — the failover regex already handles their rate-limit
   message, so they slot in as a cloud fallback that is *already* wired for this exact
   failure mode.
4. **Zen, only for subscribers** — keep the tier for `opencode.ai/go` accounts, now
   honest in the catalog and no longer able to stall a turn.

The structural point: the chain should be *go or local first, zen last*, so a dead
free tier can never sit in front of a working provider.

## Verification

- `python -m pytest tests/unit/test_zen_quota_guard.py` — the new suite.
- `python -m pytest tests/test_opencode_provider.py` — existing provider contract,
  including the User-Agent/1010 test at `tests/test_opencode_provider.py:752`.
- Targeted re-run of the live probe used for this diagnosis: with the guard primed,
  a zen chat must return the quota message in < 1 s and make zero HTTP requests.
- One full-suite run at the end, per the repo's pytest budget rules.

## Risks

- **Over-broad body matching** — a non-quota error whose text contains
  "free usage limit reached" is implausible, but the marker match is restricted to
  full-body substrings and covered by Task 8.2 (bare 429 stays transient).
- **Stale exhaustion flag** — a user who subscribes mid-session stays blocked until
  the cooldown expires. Mitigation: `reset()` (and `model` re-selection) clears it;
  default cooldown is short (900 s).
- **Failover churn** — treating quota as a connection failure could skip zen quickly
  in a chain where the user *wants* zen first. Accepted: it cannot answer anyway, and
  the ordering recommendation above addresses intent.
