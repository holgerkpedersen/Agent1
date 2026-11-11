# Kanban Card Work Visibility — Final Plan [DONE]

Status: **implemented 2026-11-10** — Agent1 side (`update_issue`,
`issue note`/`issue update` + child-card emission gating) and Kanban side
(idempotent `child_card_create` apply under the parent card) are committed;
regression tests in `tests/test_kanban_child_card_emission.py`.
Supersedes: `docs/plans/2026-11-10-kanban-card-work-visibility-plan.md`,
`docs/plans/2026-11-10-kanban-card-work-visibility-reconsidered.md`.

## 1. User need (restated)

On a Kanban board, a human must be able to see what Agent1 did to solve a
synchronized card. Agent1 must also be able to retrieve that same record later.

Two surfaces:

1. **Board**: work appears as child Kanban cards attached to the parent card
   via `parent_id`.
2. **Agent1**: `kanban card <id>` shows a Work log of what was done.

## 2. Current state (verified)

- Agent1 emits child cards on `issue_create` (intake) and `issue_resolve`
  (resolution transition), gated by `emit_child_card=True`.
- Kanban reader creates/updates the child under the parent (`parent_id`,
  `[source:<issue>]` idempotency).
- Agent1 `kanban card <id>` already includes a Work log from `.issues.json`.
- Gaps: no intermediate work (notes/updates), no dedupe on retries for all
  paths, no `issue note` command, flat-view badge only via tags.

## 3. Proposed change (smallest coherent)

Agent1 side:
- Add `issue_store.update_issue(issue_id, ..., emit_child_card=...)` (or reuse
  resolve-style) that mutates evidence/approach/status and enqueues
  `issue_update` + conditional child card.
- Add `issue note <id> "..."` command (`agent_core/commands/issue_cmd.py`) that
  records intermediate work with `emit_child_card=True`.
- Thread `parent_card_id` through update path.
- `_emit_child_card_for`: `[Agent1 work]` title prefix, `[card:<parent>]` tag,
  op in payload, best-effort idempotency key.
- Keep `work_log_for_card` query (already present) as retrieval.

Kanban side:
- `_apply_child_card_create`: dedupe by `(parent_id, issue_id, op)` in addition
  to source; keep best-effort no-op.
- Keep `[Agent1 work]` prefix readable in flat view; no JS change first.

## 4. Decision points (need your answer)

1. Intermediate work: per note/update child card, or only create + resolve +
   one summary? I lean per note.
2. Idempotency: skip if same `(parent, issue, op)` even if text changed, or
   allow re-emission? I lean skip-if-same-op.
3. Flat view: prefix+tag only, or also renderer change? I lean prefix+tag only.
4. Retrieval: Work log in `kanban card` (already) — confirm.

## 5. Verification plan

- Extend `test_kanban_child_card_emission.py` for update/note path and dedup.
- Extend Kanban `test_kanban_sync.py` for dedup and tag assertions.
- Run Agent1 and Kanban suites before commit.
- Manual smoke: sync card, resolve with note, confirm child + Work log.

## 6. What I will NOT do without your "yes"

No code changes. Only this plan document until you confirm.