---
name: kanban-card-work
description: Work a Kanban card by ID in Agent1 — resolve the card↔issue keys, read the full description (card body), check queue state, and close the loop back to the board
when_to_use: the user names a Kanban card id, card title, board column, or issue id that came from a Kanban card ("work on card 3f9a…", "what's the status of the card about X?", "why didn't my description show up?")
tags: [kanban, integration, bridge, queue]
---

# Kanban Card Work (Agent1 ↔ Kanban)

Use this whenever a request references a **Kanban card**. The two repos share
only files on disk; nothing is queried live. A card id is NOT an issue id, and
the description lives in a place the bridge does not copy verbatim. Read the
keys before touching code.

## The key map (memorize this)

| Key | Lives in | Travels as |
|---|---|---|
| Kanban **card id** (uuid4 hex, 32 chars) | `Kanban/data/board-<board>.json` | inbound `source_id`, and `locations: ["kanban:<card-id>"]` on the issue |
| Agent1 **issue id** (`iss-…`) | `Agent1/.issues.json` | outbound `source_id` |
| `.card_id_map.json` | `Kanban/data/` | cache of BOTH directions (card→issue and issue→card); it drifts and goes stale |
| `locations["kanban:<id>"]` | the issue record | the **durable** copy of the link; a hit here heals the map |
| frame id / frame_title | board file | column. Status is keyed by **title**, never by uuid |
| `seq.txt` / `offset.txt` | each queue dir | writer counter / reader cursor — separate files on purpose |

Resolution order used by the bridge (and by `kanban card <key>`):
`.card_id_map.json` → issue `locations` (heals the map) → active-board title match.
A card created directly on the board, never bridged, resolves to **nothing**:
its update/move/delete messages are quiet no-ops, permanently.

Status mapping:

```
Issues            → open          open        → open
Planned           → planned       planned     → planned
On-going          → in-progress   in-progress → in-progress
Quality & Assurance → reviewing   reviewing   → reviewing
Finished          → resolved      done        → resolved / wontfix
```

## Step 1 — resolve the key before doing anything

```
kanban card <card-id>            # REPL
python -m harnessfix.kanban_inspect card <card-id>
```

Accepts a card id, an issue id, a card title, or a unique title prefix. It
prints: which hop found the link, the ledger record, the board card (column +
status), the **composed description**, and any queued messages about that key.
If it resolves to nothing, say so — do not invent an issue.

## Step 2 — read the FULL description, not one field

The card body (`text`) is split into `evidence` / `suggested_approach` **only**
when the author used the `---` separator or `[component:]`/`[note:]`/`[status:]`
tags. A plain paragraph on the board has no tags, so a single field read shows
an empty issue and looks like "the description didn't move".

- `kanban card <id>` prints `description:` — card text plus every ledger field,
  de-duplicated. Use that as the card's content.
- For issues created before the bridge carried the body:
  `kanban sync-descriptions --dry-run`, review, then run it without the flag.
  It fills only EMPTY fields and never overwrites human text.

## Step 3 — check the queues when state looks stale

```
kanban status          # switches, board, map health, queue depths, dead letters
kanban queue           # unprocessed messages both ways, each resolved to an issue id
```

- Sync needs `KANBAN_SYNC_ENABLED=1` **and** the board's `sync.agent1` flag.
  Both are fail-closed: no flag, no sync.
- A message sitting in `queue-kanban-to-agent1` means Agent1 has not drained it
  (process not running, or retries climbing — `retry_count > 0`).
- Dead letters mean repeated failures; read the quarantined payload before
  re-applying anything by hand.

## Step 4 — do the work, then close the loop

Work the issue with the normal tools (`issue show`, `fix`, `implement`). When
it is done:

```
issue resolve <id> resolved --note "…"
kanban card <id>          # confirm what will travel back to the board
```

Resolving an issue enqueues `issue_resolve` outbound; the Kanban side applies it
on its own poll. Do not hand-edit queue files, `.card_id_map.json`, or the
board JSON — the queues are append-only ledgers with reader cursors, and a
manual edit desynchronises both sides.

## Never do this

- Treat a card id as an issue id (or vice versa) without resolving it.
- Conclude "the bridge is broken" from one empty field — read the composed
  description first.
- Edit queue JSONL, offsets, or the id map to "fix" a stuck message.
- Move status by writing the board file; statuses move through messages.
