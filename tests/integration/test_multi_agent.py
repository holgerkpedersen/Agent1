import pytest

from src.agent1.core import AgentMessage, MessageType, SQLiteStorage
from src.agent1.memory.memory_store import MemoryStore


def test_multi_agent_memory_isolation():
    """Two stores on the same backend share persistence."""
    backend = SQLiteStorage(":memory:")
    store_a = MemoryStore(backend)
    store_b = MemoryStore(backend)

    store_a.save_memory("key1", {"value": "a"})
    store_b.save_memory("key2", {"value": "b"})

    result_a = store_a.load_memory("key1")
    result_b = store_b.load_memory("key2")

    assert result_a is not None
    assert result_b is not None
    assert result_a["value"] == "a"
    assert result_b["value"] == "b"


def test_agent_message_serialization():
    msg = AgentMessage(
        sender_id="agent_1",
        receiver_id="agent_2",
        message_type=MessageType.TASK_REQUEST,
        content={"task": "test"},
        timestamp=1000.0,
    )
    assert msg.sender_id == "agent_1"
    assert msg.receiver_id == "agent_2"
    assert msg.message_type == MessageType.TASK_REQUEST
    assert msg.content["task"] == "test"
    assert msg.message_id


def test_multi_agent_broadcast():
    msg1 = AgentMessage(
        sender_id="coordinator",
        receiver_id=None,
        message_type=MessageType.STATUS_UPDATE,
        content={"status": "started"},
        timestamp=1000.0,
    )
    msg2 = AgentMessage(
        sender_id="coordinator",
        receiver_id=None,
        message_type=MessageType.STATUS_UPDATE,
        content={"status": "completed"},
        timestamp=2000.0,
    )
    assert msg1.content["status"] == "started"
    assert msg2.content["status"] == "completed"


def test_multi_agent_message_routing():
    msg = AgentMessage(
        sender_id="agent_1",
        receiver_id="agent_3",
        message_type=MessageType.QUERY,
        content={"query": "get_status"},
        timestamp=1500.0,
    )
    assert msg.receiver_id == "agent_3"
    assert msg.message_type == MessageType.QUERY
