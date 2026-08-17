"""RoutingBus tests (plan task 34): message routing, MessageType handling,
subscriptions, tasks, and error propagation."""
import pytest

from agent_core.routing.bus import Message, MessageType, RoutingBus, RoutingError, TaskNode


class TestMessageType:
    def test_members(self):
        assert MessageType.EVENT.value == "event"
        assert MessageType.COMMAND.value == "command"
        assert MessageType.RESPONSE.value == "response"
        assert MessageType.TASK_SUCCESS.value == "task_success"
        assert MessageType.TASK_FAILURE.value == "task_failure"


class TestHandlerRouting:
    def test_register_and_send(self):
        bus = RoutingBus()
        received = {}

        def handler(msg: Message):
            received["msg"] = msg.message
            return "handled"

        bus.register_handler("worker", handler)
        result = bus.send("worker", {"job": 1})
        assert result == "handled"
        assert received["msg"] == str({"job": 1})

    def test_unregister_handler(self):
        bus = RoutingBus()
        bus.register_handler("worker", lambda m: "x")
        assert bus.unregister_handler("worker") is True
        assert bus.unregister_handler("worker") is False
        assert bus.send("worker", {}) is None

    def test_send_to_missing_handler_returns_none(self):
        bus = RoutingBus()
        assert bus.send("nobody", {}) is None

    def test_send_re_raises_routing_error(self):
        bus = RoutingBus()

        def handler(msg):
            raise RoutingError("boom")

        bus.register_handler("worker", handler)
        with pytest.raises(RoutingError):
            bus.send("worker", {})

    def test_handler_exception_broadcasts_failure(self):
        bus = RoutingBus()
        failures = []
        bus.subscribe(MessageType.TASK_FAILURE, "notifier")
        bus.register_handler("notifier", lambda m: failures.append(m.message))

        def handler(msg):
            raise ValueError("kaboom")

        bus.register_handler("worker", handler)
        assert bus.send("worker", {}) is None
        assert len(failures) == 1
        assert "kaboom" in failures[0]


class TestSubscribeBroadcast:
    def test_broadcast_to_subscribers(self):
        bus = RoutingBus()
        got = []
        bus.subscribe(MessageType.EVENT, "a")
        bus.subscribe(MessageType.EVENT, "b")
        bus.register_handler("a", lambda m: "A")
        bus.register_handler("b", lambda m: "B")
        results = bus.broadcast({"tick": 1}, msg_type=MessageType.EVENT)
        assert results == ["A", "B"]

    def test_broadcast_only_matching_type(self):
        bus = RoutingBus()
        got = []
        bus.subscribe(MessageType.COMMAND, "a")
        bus.register_handler("a", lambda m: got.append(1))
        bus.broadcast({"x": 1}, msg_type=MessageType.EVENT)
        assert got == []

    def test_unsubscribe(self):
        bus = RoutingBus()
        bus.subscribe(MessageType.EVENT, "a")
        assert bus.unsubscribe(MessageType.EVENT, "a") is True
        assert bus.unsubscribe(MessageType.EVENT, "a") is False

    def test_broadcast_survives_subscriber_failure(self):
        bus = RoutingBus()
        bus.subscribe(MessageType.EVENT, "bad")
        bus.subscribe(MessageType.EVENT, "good")
        bus.register_handler("bad", lambda m: (_ for _ in ()).throw(RuntimeError("x")))
        bus.register_handler("good", lambda m: "OK")
        assert bus.broadcast({"x": 1}) == ["OK"]


class TestTasks:
    def test_add_remove_task(self):
        bus = RoutingBus()
        bus.add_task(TaskNode("t1", ["t0"]))
        assert bus.remove_task("t1") is True
        assert bus.remove_task("t1") is False

    def test_execute_task_missing_raises(self):
        bus = RoutingBus()
        with pytest.raises(RoutingError):
            bus.execute_task("ghost")

    def test_execute_task_returns_execution(self):
        bus = RoutingBus()
        bus.add_task(TaskNode("t1"))
        result = bus.execute_task("t1")
        assert result == {"status": "executed", "task": "t1"}

    def test_execute_graph_respects_dependencies(self):
        bus = RoutingBus()
        bus.add_task(TaskNode("a", ["b", "c"]))
        bus.add_task(TaskNode("b", ["c"]))
        bus.add_task(TaskNode("c"))
        results = bus.execute_graph("a")
        assert list(results.keys()) == ["c", "b", "a"]

    def test_execute_graph_cycle_safe(self):
        bus = RoutingBus()
        bus.add_task(TaskNode("a", ["b"]))
        bus.add_task(TaskNode("b", ["a"]))
        results = bus.execute_graph("a")
        assert set(results) == {"a", "b"}

    def test_execute_graph_missing_dependency_raises(self):
        bus = RoutingBus()
        bus.add_task(TaskNode("a", ["ghost"]))
        with pytest.raises(RoutingError):
            bus.execute_graph("a")

    def test_execute_graph_missing_start_raises(self):
        bus = RoutingBus()
        with pytest.raises(RoutingError):
            bus.execute_graph("nowhere")
