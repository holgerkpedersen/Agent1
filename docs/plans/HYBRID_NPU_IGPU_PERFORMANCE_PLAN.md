# Hybrid NPU+iGPU performance plan — Ryzen AI Max+ 395 (Strix Halo)

**Status:** ❌ **ABANDONED 2026-11-14** — dropped by maintainer decision ("not doing
any good for me"). No code was changed by this plan and none will be. Kept as an
evidence record only; **do not schedule Phases 0–4**. See §0 below for what was
learned and what, if anything, is worth salvaging separately.
**Measured on:** this machine, 2026 build — Ryzen AI Max+ 395, 128 GB unified LPCAMM,
Radeon 8060S iGPU (driver 32.0.32015.7029), Lemonade 2026.39.1, flm 1.0.5.

---

## 0. Abandonment record (2026-11-14)

**Decision:** abandoned before execution. Phase 1 had no work done; the 1.3 A/B
gate (same model across `*-NPU` / `*-iGPU` / `*-Hybrid`) was never run, so the
plan's headline claim was never tested.

Why it was dropped — the plan was not going to pay off:

1. **The prize was small and narrow.** §6 already capped it: better TTFT on a
   4B–14B model, *not* a step change — and it cannot make a 30B model fast.
   The measured baseline showed no prefill win to chase (NPU-only TTFT 2.19 s vs
   iGPU 2.14 s, §3).
2. **It did not answer the real need.** Large-model local use is Track B
   (Phase 3, iGPU/llama.cpp + CPU offload), which needs neither the hybrid
   runtime nor the NPU. The hybrid track was orthogonal to the actual goal.
3. **Cost and risk were front-loaded.** Multi-GB pulls, conflicting
   XDNA drivers (§6 "two backends, one NPU"), and Phase 0 instrumentation —
   all before a single verified gain.
4. **Step 4.1 was a misdiagnosis.** `agent_core/llm/prompt_cache.py` is a
   versioned prompt **template** store keyed by `(task_type, profile_type)`
   (see its docstring, `:48-101`). It caches template text, **not** KV/prefix
   state, so it cannot "prefill a long system prompt once". Any future prefill
   work must come from the backend (llama.cpp / LM Studio KV-prefix reuse).

**Salvage note:** `tools/perf_probe.py` (TTFT + decode TPS against any
OpenAI-compatible endpoint) is independent of this plan and remains useful —
it is the cheapest way to compare any local backend. Keep it. Everything else
here is closed.

---

## 1. Question asked

> AMD splits one LLM across the NPU (prefill → low TTFT) and the iGPU
> (decode → high TPS). Is Agent1 using that strategy? Make a plan to get
> better performance on this machine.

## 2. Answer: no, Agent1 does not use it — and it cannot, as configured

The split described in the request is real: it is **ONNX Runtime GenAI (OGA)
"hybrid execution mode"**, part of Ryzen AI Software. The NPU runs prefill
(compute-bound, ~50 TOPS), the iGPU runs the memory-bound decode loop. AMD
documents it as an *execution mode of the OGA runtime*
(ryzenai.docs.amd.com/en/latest/hybrid_oga.html; Ryzen AI 1.9 LLM Deployment
Overview).

Three independent blockers on this machine, each verified:

| # | Blocker | Evidence |
|---|---|---|
| 1 | **Wrong recipe is installed.** Every Lemonade model here is `recipe: flm` (FastFlowLM), which is NPU-**only**. Hybrid lives in the `ryzenai-llm` recipe (models named `*-Hybrid`). | `lemonade list` → downloaded: `gpt-oss-20b-FLM`, `qwen3.5-9b-FLM`, `Moonshine-Medium-Streaming`. `/api/v1/models` → `"recipe": "flm"` for all three. |
| 2 | **The hybrid backend is not installed.** | `lemonade backends` → `ryzenai-llm  npu  installable` ("Backend is supported but not installed"); only `flm npu  installed  v1.0.5`. |
| 3 | **The provider is not in Agent1's chain.** | `.env:21` → `AGENT_LLM_PROVIDERS=opencode:…,openrouter,lmstudio,llama` — no `lemonade` entry, so the NPU is never reached by default. |

Agent1 *does* already have the right architecture for multi-device work —
`agent_core/llm/engines.py` maps `npu → lemonade`, `igpu → lmstudio`, and
`multillm --engines npu,igpu` runs **two separate models on two devices in
parallel**. That is genuine NPU+iGPU utilisation, but it is **not** AMD's
single-model hybrid split. The requested strategy needs the *runtime*, not the
client.

## 3. Baseline measured today (the number to beat)

Method: streaming `/chat/completions`, `max_tokens=256`, temp 0.1, fixed
code-generation prompt, TTFT = first content delta, TPS = completion tokens ÷
(total − TTFT). Scratch probe deleted after use.

| Path | Model | TTFT | decode TPS |
|---|---|---|---|
| NPU-only (flm) | `qwen3.5-9b-FLM` (9B, ctx 65 536) | **2.19 s / 2.25 s** | **5.3 tok/s** (twice, ±0) |
| iGPU (LM Studio/Vulkan) | `qwen2.5-coder-1.5b-instruct` (1.5B) | 2.14 s | **114.1 tok/s** |

Reading:
- NPU decode of **5.3 tok/s** on a 9B model over Vulkan-accelerated iGPU decode
  of ~114 tok/s on a 1.5B model is *not* an apples-to-apples comparison (model
  size differs by 6×). It is an **indicative** signal only, and step 2 of the
  plan fixes exactly that by measuring one family in all three modes.
- Cold NPU load of the 8.7 GB model cost **~57 s** TTFT on the first call.
  Keep-alive / pinning matters more than raw throughput for an interactive CLI.

## 4. Strategy

Two tracks, in priority order. Track A is what the request literally asks for.
Track B is where the larger, more certain wins are on a 128 GB unified-memory
box — and it should not wait for Track A.

- **Track A — single-model hybrid (OGA `ryzenai-llm`, `*-Hybrid` models).**
  Genuine NPU prefill + iGPU decode for one model. Cost: one backend install
  plus a multi-GB model pull. Gain: lower TTFT and higher TPS on *that one
  model*; it does **not** let a 27B model run fast, because OGA hybrid supports
  a limited, pre-optimised model set.
- **Track B — device-split parallelism (already built, untuned).**
  `multillm --engines npu,igpu` puts two models on two accelerators. On Strix
  Halo the decisive win is instead **large MoE models on the iGPU with heavy
  layers offloaded to the CPU**: 128 GB of unified memory means a 60–70 GB
  MoE GGUF is *residency-feasible* at ~4-bit, and MoE decode touches only the
  active experts, so CPU offload costs far less than it would for a dense
  model. This is the highest-ceiling option on this hardware.

## 5. Plan

### Phase 0 — Make it measurable (blocking; nothing else is verifiable without it)

Agent1 records only `latency_ms` (`agent_core/llm/provider.py:113`); it has no
TTFT/TPS instrumentation, so no optimisation can be proved.

- **0.1** Add TTFT + decode-TPS capture to `ResponseMetrics` (first-delta
  timestamp on the streaming path, completion tokens from `usage`).
- **0.2** Surface both in `benchmark.py` (it already records `latency_ms` and
  `tokens_used` per question) so a run produces TTFT/TPS per model.
- **0.3** Promote the scratch probe to `tools/perf_probe.py` — one command that
  prints `model, mode, ttft_s, tps` for any OpenAI-compatible endpoint.
- **0.4** Record the baseline table from §3 into `reports/` as the comparison
  anchor.
- **Acceptance:** `python -m pytest` green; `tools/perf_probe.py` reproduces
  §3 within ±10 %.

### Phase 1 — Track A: enable real hybrid

- **1.1** `lemonade backends install ryzenai-llm:npu`. Verify:
  `lemonade backends` shows `ryzenai-llm  npu  installed`.
- **1.2** Pull **one** hybrid model and measure before pulling more. Entry
  point: `Qwen3-4B-Hybrid` (5.17 GB) — smallest credible chat model with
  `*-Hybrid`; escalate to `Qwen2.5-14B-instruct-Hybrid` (16.5 GB) only if 1.3
  shows a real gain.
- **1.3** A/B the same model family in all three modes — `*-NPU`, `*-iGPU`
  (LM Studio GGUF), `*-Hybrid` — with `ctx_size` held constant. Report
  TTFT and TPS deltas against §3. **This is the go/no-go gate for the whole
  track.**
- **1.4** If the gate passes: decide whether hybrid becomes the default for the
  CLI, or stays an opt-in engine (`--engines hybrid`). Add a `hybrid` engine to
  `engines.py` mapping to a `lemonade`-routed `*-Hybrid` model so
  `multillm --engines hybrid,igpu` works.
- **1.5** Raise `ctx_size`. The current flm models run at 16 384 / 65 536 while
  supporting 131 072 / 262 144. `lemonade load <model> --ctx-size 65536
  --save-options`. Context budgeting already exists
  (`agent_core/llm/context_budget.py`) but the NPU window caps it — freeing
  the window is a direct quality win.
- **Acceptance:** a table of TTFT/TPS for the three modes on one model, plus an
  explicit written go/no-go decision.

### Phase 2 — Wire Lemonade into Agent1 properly

- **2.1** Add `lemonade` to `AGENT_LLM_PROVIDERS` in `.env` (it is absent today),
  positioned so local inference is actually attempted before cloud failover.
- **2.2** Verify `build_provider` puts the `lemonade/…` model first (the
  synthetic-front-entry fix in AGENTS.md:677 is the precedent — that bug
  silently ran cloud instead of the NPU).
- **2.3** Set `AGENT_LEMONADE_MODEL` to the chosen model (currently unset; the
  provider falls back to an empty default).
- **2.4** Pin the model (`lemonade pin`) and enable `max_loaded_models=2` +
  `auto_evict` so a Jev/NPU model and a chat/iGPU model can coexist without a
  ~57 s cold reload between them.
- **Acceptance:** `model list` shows the `[lemonade]` section; a turn reports
  the NPU as the live provider; no cold-reload stall on the second turn.

### Phase 3 — Track B: Strix Halo large-MoE tuning (highest ceiling)

- **3.1** Install `llamacpp:vulkan` (`lemonade backends install llamacpp:vulkan`)
  and confirm `--llamacpp-device Vulkan0` binds the 8060S.
- **3.2** Benchmark a 30B-A3B MoE with progressively larger `-ngl` offload
  (`Qwen3-30B-A3B-GGUF` 17.4 GB, then `Qwen3-Coder-30B-A3B-Instruct-GGUF`
  18.6 GB) — measure TPS vs the 114 tok/s 1.5B figure and vs 5.3 tok/s NPU.
- **3.3** Only then consider a 60–70 GB MoE (`GLM-4.5-Air`, `Qwen3.5-122B-A10B`
  77.9 GB) to test whether 128 GB unified memory sustains a resident large MoE.
  Disk and load time are the real constraints here, not RAM.
- **3.4** Tune context/KV: KV cache at 128 k context on a 30B model is the
  memory hog, not the weights. Measure TPS at 8 k / 32 k / 128 k.
- **Acceptance:** a TPS-vs-model-size curve, and a documented "best model for
  interactive coding on this box" recommendation.

### Phase 4 — Interaction-level wins (cheap, orthogonal)

- **4.1** ~~Prompt-cache reuse — `agent_core/llm/prompt_cache.py` exists; verify it
  is active for the local providers so a long system prompt is prefilled once.~~
  **Void — misdiagnosis (see §0.4).** That module stores versioned prompt
  *templates*, not KV/prefix state, so it cannot avoid re-prefill.
- **4.2** Keep-warm: avoid reloading between turns (`warm=` path in
  `run_parallel`).
- **4.3** Batch/parallel `multillm` for speculative+judge work — offload the Jev
  judge to the NPU while branches run on the iGPU (already supported:
  AGENTS.md:669).

## 6. Risks and honest expectations

- **Hybrid is a narrow win.** OGA hybrid supports a *fixed set of pre-optimised
  models* on STX. It will not make a 30B model fast. Expect a better TTFT on a
  4B–14B model, not a step change in agent capability.
- **Two backends, one NPU.** flm and ryzenai-llm both drive the XDNA NPU and may
  not coexist cleanly; installing ryzenai-llm could disturb the working flm
  path. Step 1.1 must be verified against the §3 baseline before anything else
  is built on it.
- **Model pulls are large** (2–17 GB per experiment). Each step names its size
  so the cost is explicit before it is paid.
- **NPU throughput is the wrong knob for agent work.** An agent turn is many
  short calls with long prompts; TTFT and prompt-cache reuse dominate the felt
  latency far more than decode TPS.
- **No code has been changed by this plan.** Phases 0–4 are proposals.

## 7. Recommended order

1. **Phase 0** — without instrumentation, every later claim is a guess.
2. **Phase 3.1–3.2** — highest expected gain per hour; no new concept, reuses
   the existing LM Studio/llama.cpp path.
3. **Phase 2** — makes local inference actually reachable (currently it is not,
   by default).
4. **Phase 1** — only after the 1.3 gate; it is the request as literally stated,
   but the narrowest win.
