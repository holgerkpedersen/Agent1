from typing import Dict, Any, Optional, List, Protocol, Type, Callable, Awaitable, Set
import asyncio
import uuid
import json
import sqlite3
import numpy as np
import networkx as nx
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


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


MessageBusHandler = Callable[[AgentMessage], Awaitable[None]]


class StorageBackend(Protocol):
    def store(self, key: str, data: Dict[str, Any]) -> None: ...
    def retrieve(self, key: str) -> Optional[Dict[str, Any]]: ...
    def delete(self, key: str) -> bool: ...

class SQLiteStorage:
    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn: sqlite3.Connection = sqlite3.connect(db_path, check_same_thread=False)
        self._setup_schema()

    def _setup_schema(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_entries (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data TEXT NOT NULL,
                tags TEXT
            )
        """)
        self._conn.commit()

    def store(self, key: str, data: Dict[str, Any]) -> None:
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO memory_entries (id, agent_id, timestamp, data, tags)
            VALUES (?, ?, ?, ?, ?)
        """, (key, str(data.get("agent_id", "unknown")),
              time.strftime("%Y-%m-%dT%H:%M:%S"), json.dumps(data), ""))
        self._conn.commit()

    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        cursor = self._conn.cursor()
        cursor.execute("SELECT data FROM memory_entries WHERE id = ?", (key,))
        result = cursor.fetchone()
        if result is None:
            return None
        return json.loads(result[0])

    def delete(self, key: str) -> bool:
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM memory_entries WHERE id = ?", (key,))
        deleted = cursor.rowcount > 0
        self._conn.commit()
        return deleted


class VectorEmbeddingModel(Protocol):
    def encode(self, texts: List[str]) -> np.ndarray: ...

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name: str = model_name

    def embed_text(self, texts: List[str]) -> np.ndarray:
        return np.zeros((len(texts), 384))


class VectorDatabase:
    def __init__(self, dimension: int = 384) -> None:
        self._dimension: int = dimension
        self._vectors: Dict[int, np.ndarray] = {}
        self._id_to_metadata: Dict[int, Dict[str, Any]] = {}
        self._next_id: int = 0

    def add_vector(self, vector: np.ndarray, metadata: Dict[str, Any]) -> int:
        if len(vector) != self._dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self._dimension}, got {len(vector)}")
        normalized_vector = vector / (np.linalg.norm(vector) + 1e-8)
        vector_id = self._next_id
        self._vectors[vector_id] = normalized_vector
        self._id_to_metadata[vector_id] = metadata
        self._next_id += 1
        return vector_id

    def search_similar(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        normalized_query = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        results: List[tuple[float, int]] = []
        for vid, vec in self._vectors.items():
            similarity = float(np.dot(normalized_query, vec))
            results.append((similarity, vid))
        results.sort(key=lambda x: x[0], reverse=True)
        top_results: List[Dict[str, Any]] = []
        for i in range(min(k, len(results))):
            similarity, idx = results[i]
            if idx >= 0 and idx in self._id_to_metadata:
                result_item = {
                    "metadata": self._id_to_metadata[idx],
                    "similarity_score": similarity,
                    "vector_id": idx
                }
                top_results.append(result_item)
        return top_results


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


class PluginInterface(Protocol):
    def initialize(self, config: Dict[str, Any]) -> None: ...
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]: ...
    def cleanup(self) -> None: ...

@dataclass
class PluginMetadata:
    name: str
    version: str
    description: str
    author: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class MetricData:
    name: str
    value: float
    timestamp: float
    metric_type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertRule:
    name: str
    metric_name: str
    threshold: float
    comparison_operator: str
    severity: str
    cooldown_seconds: int = 60

@dataclass
class AlertEvent:
    rule_name: str
    triggered_at: float
    current_value: float
    threshold: float
    severity: str
    message: str


__all__: List[str] = [
    "MessageType",
    "AgentMessage",
    "MessageBusHandler",
    "StorageBackend",
    "SQLiteStorage",
    "VectorEmbeddingModel",
    "EmbeddingService",
    "VectorDatabase",
    "TaskStatus",
    "TaskNode",
    "PluginInterface",
    "PluginMetadata",
    "MetricType",
    "MetricData",
    "AlertRule",
    "AlertEvent",
]