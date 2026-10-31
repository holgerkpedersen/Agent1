# Prospector strategy for the Agent1 harness - plan and safety argument

**Date:** 2026-10-30 - **Source:** arXiv:2610.37121 (LAMM/MIT; PI Markus J. Buehler), HF: `lamm-mit/Prospector` (LoRA adapters on `Qwen/Qwen3.5-4B`, Apache-2.0), code: github.com/lamm-mit/Prospector - **Target model:** `google/gemma-4-E4B-it` (native function calling).

**Verification status (read first):** The paper, the HF repository and the GitHub repository are confirmed, and the gate rules in section 0 are verified against both the paper abstract, the HF model card and the GitHub repository (the upstream repository's `agent/gate.py`, that is, in `lamm-mit/Prospector` - not in this workspace: "cites at least one edge, every cited edge was seen, and it names a node of the cited chain: set membership, with no judge"). The exact model name and model card quote are therefore not fully verified (the web search was re-run successfully and confirms that `google/gemma-4-E4B-it` is a real Gemma 4 model in the new Gemma family, that Gemma 4 has native function calling and structured JSON output, and that E4B is an edge-optimized variant with 128K context). The first author's name is not verified. The plan is therefore written so that it does not depend on the model name: phase 0 measures on the locally available function-calling model, and phase 4's model profile is parameterized. Confirm the model card before phase 4 goes live.

## 0. What the paper actually says (verified, see status above)

1. Prospector is a **traversal policy**: it calls three tools one at a time -
   `find(name)`, `expand(node, relation?, offset?)`, `connect(a, b, max_hops?)` -
   and builds context from what it already found.
   Budget: 20 tool calls, 1,024 tokens per turn.
2. **The gate** is strict set membership without a judge: a response is accepted
   only if (a) at least one edge is cited, (b) every cited edge was actually
   returned by a tool ("seen" set), (c) the response names a node in the cited
   chain. The goal is "gated success" (gate OK + response names the gold label).
3. **Key numbers (model card, 8 seeds):** Prospector-4B semiconductor 0.839 + 0.01;
   zero-shot transfer: bio-inspired materials 0.803, PFAS 0.766,
   materials properties 0.809. 2B: 0.755; 0.8B: 0.694.
4. **Training recipe (optional):** SFT on teacher demo, then GRPO over full
   episodes, then blended SFT, then two decision-level GRPO rounds + verifier.
5. Latent state reads the next move (follow/backtrack/answer) and can be
   controlled.

**Important distinction:** the paper's core gain (selection under reasoning + verifiable evidence) is a **strategy, not a weight**. It can be implemented untrained in a harness like Agent1. Fine-tuning is step 5 and optional.

## 1. Why it fits Agent1 (code-verified)

| Prospector principle | Agent1 component (existing) |
|---|---|
| Episode loop with sequential tool calls | `agent_core/llm/tool_loop.py::ToolLoopRunner` (max_iterations, deadline, compaction, tracing) |
| Tool schemas for the model | `agent_core/tool_schemas.py::NLP_TOOL_SCHEMAS`, native OpenAI format |
| Local model with function calling | `agent_core/llm/llama_provider.py::LlamaProvider` (LM Studio/llama.cpp, `_build_payload(messages, tools, ...)`), `agent_core/llm/model_profiles.py` |
| "Seen" set / observability | `harnessfix/tracing.py` - JSONL per task with `tool_call`/`tool_result`/`llm_response` events (decision #033: `reports/traces/`, gitignored) |
| External data facades without core change | `agent_core/mcp/manager.py` - MCP servers expose tools to the main loop |
| Measurement on fixed scenarios | `agentic_bench.py` - runs real main-loop tools, scores per run |

Conclusion: the harness only lacks (a) a graph facade and (b) a gate. Everything
else is reused.

## 2. Plan (6 phases, no core breakage)

### Phase 0 - Baseline establishment (measure before changing)
- Choose a local KG dataset (for example the paper's `tsmc_5b10p.graphml` from
  the HF repository, Apache-2.0, or a custom GraphML/JSON).
- Run 3 conditions on ~30 questions via the `agentic_bench` pattern:
  (A) Gemma-4-E4B closed book, (B) E4B + selected subgraph (RAG),
  (C) E4B + traversal tools. Record the share of answers with cited edges and
  the share of gate passes.
- **Artifacts:** `benchmarks/prospector_baseline.md` (the report; the dataset
  itself downloads locally and stays out of version control).

### Phase 1 - Graph facade as MCP server (`graphkg`)
- Implement `find`/`expand`/`connect` as **MCP tools** via the existing
  `McpManager` infrastructure. No change in ToolLoopRunner, no new native tools
  -> interaction with Kanban/dashboard/MCP status is inspected (phase 5).
- Same limits as the paper: find <=10 nodes, expand <=40 edges (>150 connections
  -> relations summary), connect <=3 paths.
- The facade returns **edges with stable IDs** (`e:<hash>`), so the gate and the
  "seen" set work on set membership, not string matching.

### Phase 2 - Episode loop + "seen" registration
- Reuse `ToolLoopRunner` unchanged as the loop; the run already has a TraceSink
  to `reports/traces/`.
- New small component `agent_core/prospector/seen.py`: derive the "seen"
  amount from the **tool results in the trace** (readably masked: the facade
  emits edges as JSON in the tool response). Only tool results are the source of
  truth - the model's own claim about an edge does not count.

### Phase 3 — The gate (deterministic, no judge)
- `agent_core/prospector/gate.py`: parse the final response format
  (`ANSWER: ... CHAIN: e1,e2,...`), verify: ≥1 edge; all edges ∈ seen; the
  response names a node in the chain. Normalization like the paper (Unicode,
  case, whitespace).
- A gate rejection is **not** an error: the loop gets one retry with the exact
  feedback about which citations failed (budget: max 2 attempts), otherwise the
  response is delivered marked `GATE: FAIL` — never as verified.

### Phase 4 — Prompt and profile for Gemma-4-E4B
- System prompt with the three tool signatures + example episode (from the
  paper's prompt wording, the HF repository `agent/prompts.py` can be studied as
  a reference — Apache-2.0).
- Model profile via `agent_core/llm/model_profiles.py`: E4B, low temperature,
  max 20 tool calls per episode (the harness' `max_iterations` is set per
  scenario, not globally).

### Phase 5 — Measurement and report
- Re-run A/B/C plus the gate metric; compare with the paper's numbers as a
  reference frame (not as a target: different dataset/questions).
- Check integrations: MCP status, dashboard, `agentic_bench`, Kanban runs
  (dead routes must not change existing tools' behavior).

### Phase 6 (optional) — Fine-tuning of E4B (SFT/GRPO recipe)
- Training data is generated naturally: **the traces from phases 0–5 are
  training sequences** (episodes with gate-passed answers = positive demos;
  gate failures = hard negatives). Format: ChatML turns with tool calls — saved
  as JSONL in the `reports/` tree (gitignored, see decision #033).
- Run outside the harness (llama-factory/TRL-like, LoRA on E4B,
  quantized afterwards to GGUF). **Always new adapter, never overwrite
  base weights.**
- Evaluate A/B against the untrained E4B on a holdout; adopt the adapter only
  if the gate metric improves.

## 3. Why the plan is safe (argument per pillar)

**A. No existing behavior changes.** All new files are additive:
MCP server `graphkg` delivers tools via the existing manager; gate and seen are
new modules in `agent_core/prospector/`, which only activate in dedicated
prospector runs (opt-in command/profile). ToolLoopRunner, NLP_TOOL_SCHEMAS and
existing tool routes stay bit-identical — verified by the existing test suite in
`tests/` running green before and after (CI requirement: the repository's
standard CI checks run before every commit/push).

**B. Deterministic verification, no judge.** The gate is set membership against
the tools' own returned edges. A model cannot "make up" a verified answer: it
cites an edge that was never returned, the gate fails. This is the paper's own
design (no LLM judge) — we copy the part that *removes* trust in the model's own
reporting.

**C. Loaded data is untrusted content.** Graph nodes/edges are external content
like all other data files; they appear as tool results, never as instructions.
The facade serializes edges structurally (fields, not free text in the system
role), and prospector episodes run with the same tool gating/plan mode as the
rest of the harness.

**D. Resource limits are inherited.** ToolLoopRunner already has an iteration
limit, deadline window, no-mutation detection and forced final response; episodes
get the paper's 20 calls. No infinite loops, no uncontrolled token growth
(compaction is already built in).

**E. Observability comes for free.** Every episode is already a JSONL trace with tool calls/results; the gate decision is written as one extra event in the same trace. Reproducibility: task ID + correlation ID exist.

**F. Model safety.** Gemma-4-E4B is local (LM Studio/llama.cpp) — data does not
leave the host; this matches the paper's privacy-first premise. Fine-tuning
(phase 6) is optional, reversible (adapter) and only adopted after A/B.

**G. Integration check (habits).** Before push: documentation updated (README/USAGE section on prospector state), CI green, MCP/Kanban inspection (phase 5). Emojis are avoided in all text files (decision #079 - use `[GATE OK]`/`[GATE FAIL]`).

## 3.1 What "fully safe" can and cannot mean

"Fully safe" does not exist for a plan that involves a language model. What can
be done safely, and what was done safely here, is **error states**. In concrete
terms, with verified mechanisms:

| Error state | Why it cannot cause harm | Verified in |
|---|---|---|
| Model invents an edge | The gate is set membership against "seen"; an edge that was not returned cannot pass | Upstream `lamm-mit/Prospector` `agent/gate.py` rule + phase 3 of the plan (the gate does not exist here yet) |
| Model loops | `max_iterations`, `deadline_window`, `no_mutation_limit`, `force_after_no_mutation` exist already as constructor parameters and are enforced by the loop | `agent_core/llm/tool_loop.py:685` (`ToolLoopRunner.__init__`) |
| Model mutates the repository unintentionally | Prospector episodes run read-only over the graph; no write tools in the tool set | Phase 1 (only `find`/`expand`/`connect`) |
| Context explodes | Compaction and character budget are built in (`_ensure_char_budget`, `_compact_messages`) | `tool_loop.py:1710`, `:416` |
| Silent error without trace | Every episode is a JSONL trace; the gate decision becomes an extra event | `harnessfix/tracing.py` (`KIND_TOOL_CALL`/`KIND_TOOL_RESULT`) |
| Data content instructs the model | Graph nodes are tool results, never system instructions; tool gating/plan mode applies | Same model as the rest of the harness |
| Data leaks out | Local model (LM Studio/llama.cpp); nothing leaves the host | `agent_core/llm/llama_provider.py:151` |
| Fine-tuning breaks the base model | LoRA adapter, new adapter per attempt, never overwritten base weights, A/B before adoption | Phase 6 |

**What is NOT guaranteed** (and the plan therefore measures it in phase 0 rather than assuming): that E4B's tool-calling quality is high enough on large graphs, and that the gate metric will be usable. Both are open questions answered by measurement - not by argument. That is why the safest part of the plan is the sequence: **baseline first (phase 0), gate before model ambition (phase 3), fine-tuning last and optional (phase 6).**

## 4. Risks and mitigations

| Risk | Mitigation |
|---|---|
| E4B's tool-calling quality is lower than Qwen3.5 on large graphs | Measurement in phase 0 (A/B/C) reveals it before any investment; the recipe (phase 6) is designed to close the gap |
| The facade's edges change between calls (hash drift) | Edges are content-hashed and the graph facade is read-only per episode |
| Graphs larger than memory | Expand pagination + relations summary (paper limits), GraphML → NetworkX/SQLite index |
| Gate too strict (true answer without node mention) | Follow the paper's metric 1:1; adjust only with documented exceptions |

## 5. Estimated scope

- Phases 0–2: ~2–3 working days (MCP server + seen, reuse loop).
- Phases 3–5: ~1–2 days (gate + profile + bench).
- Phase 6: separate project (GPU experiments), outside the harness.

## 6. References (see verification status at top for what was re-verified)

- arXiv:2610.37121 - abstract confirmed via arxiv.org.
- HF `lamm-mit/Prospector` - model card with result table, LoRA on Qwen3.5,
  Apache-2.0, data (GraphML) and trained checkpoints follow the repository.
- github.com/lamm-mit/Prospector - README: tool signatures, gate rules,
  `gated_acc` metric, training recipe (SFT → GRPO → blended SFT → 2× decision-GRPO).
- HF `google/gemma-4-E4B-it` - Gemma 4 model card: native function calling and
  structured JSON output for agentic workflows; E4B is an edge-optimized variant
  with 128K context. The web search confirmed the availability of the new Gemma 4
  sources, but the exact model card quote must still be verified directly before
  phase 4 goes live.
