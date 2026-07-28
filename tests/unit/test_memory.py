import pytest

from src.agent1.core import AgentMessage, MessageType, SQLiteStorage
from src.agent1.memory.memory_store import MemoryStore


def test_memory_store_save_load():
    backend = SQLiteStorage(":memory:")
    store = MemoryStore(backend)
    store.save_memory("key1", {"value": 42})
    result = store.load_memory("key1")
    assert result is not None
    assert result["value"] == 42


def test_memory_store_forget():
    backend = SQLiteStorage(":memory:")
    store = MemoryStore(backend)
    store.save_memory("key2", {"value": "hello"})
    ok = store.forget_memory("key2")
    assert ok is True


def test_memory_store_overwrite():
    backend = SQLiteStorage(":memory:")
    store = MemoryStore(backend)
    store.save_memory("key3", {"value": 1})
    store.save_memory("key3", {"value": 2})
    result = store.load_memory("key3")
    assert result is not None
    assert result["value"] == 2


def test_memory_store_clear_cache():
    backend = SQLiteStorage(":memory:")
    store = MemoryStore(backend)
    store.save_memory("key4", {"value": 99})
    store.clear_cache()
    result = store.load_memory("key4")
    assert result is not None
    assert result["value"] == 99
