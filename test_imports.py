try:
    from harnessfix.kanban_bridge import (
        _apply_card_create,
        _apply_card_update,
        _apply_card_move,
        _apply_card_delete,
        _resolve_frame_status
    )
    print("Successfully imported functions.")
except ImportError as e:
    print(f"Import failed: {e}")
