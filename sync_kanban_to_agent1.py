import json
from pathlib import Path
from typing import Any

# Import bridge components
from harnessfix.kanban_bridge import (
    _apply_card_create,
    _apply_card_update,
    _apply_card_move,
    _apply_card_delete,
)

KANBAN_QUEUE_FILE = Path(r"C:\Dev\Kanban\data\queue-kanban-to-agent1\messages.jsonl")

def sync():
    if not KANBAN_QUEUE_FILE.exists():
        print(f"Error: Kanban queue file not found at {KANBAN_QUEUE_FILE}")
        return

    print(f"Reading messages from {KANBAN_QUEUE_FILE}...")
    messages = []
    with open(KANBAN_QUEUE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))

    print(f"Found {len(messages)} messages.")

    processed_count = 0
    success_count = 0
    fail_count = 0

    for msg in messages:
        processed_count += 1
        op = msg.get("op")
        payload = msg  # The bridge functions seem to take the whole message or the payload part depending on implementation, but usually they expect the dict that contains source_id etc.
        
        # Based on previous reads of kanban_bridge.py:
        # _apply_card_create(payload) uses payload.get('id')
        # _apply_card_update(payload) uses payload.get('source_id')
        # _apply_card_move(payload) uses payload.get('source_id')
        # _apply_card_delete(payload) uses payload.get('source_id')
        
        print(f"[{processed_count}/{len(messages)}] Processing op='{op}'")
        
        result = False
        try:
            if op == "card_create":
                # _apply_card_create expects the card data (the 'payload' part of the message)
                card_data = msg.get("payload")
                if not card_data:
                    print(f"  Error: No payload found in card_create message")
                    fail_count += 1
                    continue
                result = _apply_card_create(card_data)
            elif op == "card_update":
                # _apply_card_update expects the whole message to get source_id
                result = _apply_card_update(msg)
            elif op == "card_move":
                # _apply_card_move expects the whole message to get source_id
                result = _apply_card_move(msg)
            elif op == "card_delete":
                # _apply_card_delete expects the whole message to get source_id
                result = _apply_card_delete(msg)
            else:
                print(f"  Skipping unknown operation: {op}")
                continue

            if result:
                success_count += 1
                print(f"  Success")
            else:
                fail_count += 1
                print(f"  Failed (returned False)")
        except Exception as e:
            fail_count += 1
            print(f"  Error: {e}")

    print(f"\nSync complete.")
    print(f"Processed: {processed_count}")
    print(f"Successes: {success_count}")
    print(f"Failures:  {fail_count}")

if __name__ == "__main__":
    sync()
