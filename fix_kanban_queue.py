import json
from pathlib import Path

from harnessfix.kanban_bridge import resolve_working_dir

# Resolve the queue root through the bridge so this repair touches the same
# directory the bridge reads (see resolve_working_dir for precedence).
QUEUE_DIR = Path(resolve_working_dir()) / "queue-kanban-to-agent1"
KANBAN_QUEUE_FILE = QUEUE_DIR / "messages.jsonl"
OFFSET_FILE = KANBAN_QUEUE_FILE.parent / "offset.txt"
TARGET_ID = "f01a7cb8fb514e7d8e48f8d224b66d24"


def fix_queue():
    if not KANBAN_QUEUE_FILE.exists():
        print(f"Error: {KANBAN_QUEUE_FILE} not found")
        return

    messages = []
    with open(KANBAN_QUEUE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))

    print(f"Found {len(messages)} existing messages.")

    # Find the first card_update for our target ID to use as a template for creation
    template_msg = None
    for msg in messages:
        if msg.get("op") == "card_update" and msg.get("source_id") == TARGET_ID:
            template_msg = msg
            break

    if not template_msg:
        print(f"No card_update found for {TARGET_ID}. Cannot create missing card.")
        return

    # Create the missing card_create message at the beginning of the queue
    new_create_msg = {
        "op": "card_create",
        "source_id": TARGET_ID,
        "payload": template_msg["payload"],
        "ts": template_msg["ts"],
        "retry_count": 0,
        "processed": False,
    }

    print(f"Creating missing card_create message for {TARGET_ID}")
    messages = [new_create_msg] + messages

    # seq is the line's 1-based position, so renumber every line after the
    # insert: a hand-written seq would desynchronise the tail from its position.
    for position, msg in enumerate(messages, start=1):
        msg["seq"] = position

    # Write back the fixed queue, then reset the reader cursor through the
    # bridge so everything is re-processed.
    with open(KANBAN_QUEUE_FILE, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    # Write the cursor to OFFSET_FILE (the path next to KANBAN_QUEUE_FILE) so the
    # repair stays confined to the queue it just rewrote.
    OFFSET_FILE.write_text("0", encoding="utf-8")
    print("Queue fixed and offset reset to 0.")


if __name__ == "__main__":
    fix_queue()
