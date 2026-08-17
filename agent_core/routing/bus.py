from __future__ import annotations
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class MessageType(Enum):
    EVENT = "event"
    COMMAND = "command"
    RESPONSE = "response"
    TASK_SUCCESS = "task_success"
    TASK_FAILURE = "task_failure"

class RoutingError(Exception):
    """Base exception for routing operations."""
    pass

class Message:
    def __init__(self, message: str, cause: Optional[Exception] = None) -> None:
        self.message = message
        self.cause = cause


class TaskNode:
    def __init__(self, name: str, dependencies: Optional[List[str]] = None) -> None:
        self.name = name
        self.dependencies = dependencies or []

class RoutingBus:
    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[[Message], Any]] = {}
        self._subscriptions: Dict[MessageType, List[str]] = {
            t: [] for t in MessageType
        }
        self._tasks: Dict[str, TaskNode] = {}

    def register_handler(self, name: str, handler: Callable[[Message], Any]) -> None:
        self._handlers[name] = handler
        logger.debug("Registered handler '%s'", name)

    def unregister_handler(self, name: str) -> bool:
        if name in self._handlers:
            del self._handlers[name]
            logger.debug("Unregistered handler '%s'", name)
            return True
        return False

    def subscribe(self, msg_type: MessageType, destination: str) -> None:
        if destination not in self._subscriptions[msg_type]:
            self._subscriptions[msg_type].append(destination)
            logger.debug("Subscribed '%s' to %s", destination, msg_type.value)

    def unsubscribe(self, msg_type: MessageType, destination: str) -> bool:
        if destination in self._subscriptions[msg_type]:
            self._subscriptions[msg_type].remove(destination)
            logger.debug("Unsubscribed '%s' from %s", destination, msg_type.value)
            return True
        return False

    def send(self, destination: str, payload: Dict[str, Any], **kwargs: Any) -> Optional[Any]:
        handler = self._handlers.get(destination)
        if not handler:
            logger.warning("No handler registered for destination '%s'", destination)
            return None
        msg = Message(str(payload))
        try:
            result = handler(msg)
            logger.debug("Sent message to '%s', result type=%s", destination, type(result).__name__)
            return result
        except RoutingError as exc:
            logger.error("Routing error in handler '%s': %s", destination, exc)
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled exception in handler '%s'", destination)
            self.broadcast(
                {"error": str(exc), "destination": destination},
                msg_type=MessageType.TASK_FAILURE,
            )
            return None

    def broadcast(self, payload: Dict[str, Any], msg_type: MessageType = MessageType.EVENT, **kwargs: Any) -> List[Any]:
        results: List[Any] = []
        for dest in self._subscriptions.get(msg_type, []):
            try:
                res = self.send(dest, payload)
                if res is not None:
                    results.append(res)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Broadcast failed for subscriber '%s' on %s", dest, msg_type.value)
        return results

    def add_task(self, task_node: TaskNode) -> None:
        self._tasks[task_node.name] = task_node
        logger.debug("Added task node '%s'", task_node.name)

    def remove_task(self, name: str) -> bool:
        if name in self._tasks:
            del self._tasks[name]
            logger.debug("Removed task node '%s'", name)
            return True
        return False

    def execute_task(self, name: str, message: Optional[Message] = None) -> Any:
        if name not in self._tasks:
            err_msg = f"Task {name} not found"
            logger.error(err_msg)
            raise RoutingError(err_msg)
        try:
            result = {"status": "executed", "task": name}
            logger.info("Executed task '%s'", name)
            return result
        except Exception as exc:  # noqa: BLE001
            logger.exception("Task execution failed for '%s'", name)
            self.broadcast(
                {"error": str(exc), "task": name},
                msg_type=MessageType.TASK_FAILURE,
            )
            raise RoutingError(f"Task {name} failed: {exc}") from exc

    def execute_graph(self, start_node: str, message: Optional[Message] = None) -> Dict[str, Any]:
        if start_node not in self._tasks:
            err_msg = f"Start node {start_node} not found"
            logger.error(err_msg)
            raise RoutingError(err_msg)

        results: Dict[str, Any] = {}
        visited: set[str] = set()

        def walk(node_name: str) -> None:
            if node_name in visited:
                return
            try:
                node = self._tasks[node_name]
            except KeyError as exc:
                logger.error("Missing task node '%s' during graph walk", node_name)
                raise RoutingError(f"Dependency {node_name} not found") from exc

            # Mark BEFORE recursing: dependency cycles (a -> b -> a) must
            # terminate instead of overflowing the stack.
            visited.add(node_name)
            for dep in node.dependencies:
                walk(dep)

            results[node_name] = self.execute_task(node_name, message)

        try:
            logger.info("Starting graph execution from '%s'", start_node)
            walk(start_node)
            logger.info("Graph execution completed with %d nodes", len(results))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Graph execution failed starting at '%s'", start_node)
            self.broadcast(
                {"error": str(exc), "start_node": start_node, "partial_results": results},
                msg_type=MessageType.TASK_FAILURE,
            )
            raise RoutingError(f"Graph execution from {start_node} failed: {exc}") from exc

        return results