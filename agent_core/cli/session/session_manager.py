"""In-process session manager backing the CLI clear command."""

from datetime import datetime, timezone
from typing import Any, Dict, List


class SessionManager:
    """Tracks conversation messages and exposes stats used by CLI commands."""

    def __init__(self) -> None:
        self._messages: List[Dict[str, Any]] = []
        self._last_activity: datetime = datetime.now(timezone.utc)

    def add_message(self, message: Dict[str, Any]) -> None:
        self._messages.append(message)
        self._last_activity = datetime.now(timezone.utc)

    def get_session_stats(self) -> Dict[str, Any]:
        token_usage = sum(
            len(str(m.get("content", ""))) // 4 for m in self._messages
        )
        return {
            "message_count": len(self._messages),
            "token_usage": token_usage,
            "last_activity": self._last_activity.isoformat(),
        }

    def clear_session(self) -> None:
        self._messages.clear()
        self._last_activity = datetime.now(timezone.utc)