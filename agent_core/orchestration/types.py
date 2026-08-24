"""Task/message data structures for the orchestration subsystem.

Moved verbatim from the retired ``src/agent1.core`` namespace.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class MessageType(Enum):
    TASK_REQUEST = "task_request"
    STATUS_UPDATE = "status_update"
    RESULT_SHARE = "result_share"
    QUERY = "query"


@dataclass
class AgentMessage:
    sender_id: str
    receiver_id: Optional[str]
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float
    message_id: str = ""

    def __post_init__(self) -> None:
        if not self.message_id:
            self.message_id = str(uuid.uuid4())


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class TaskNode:
    task_id: str
    name: str
    description: str
    dependencies: List[str]
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    priority: int = 0


__all__ = [
    "MessageType",
    "AgentMessage",
    "TaskStatus",
    "TaskNode",
]
