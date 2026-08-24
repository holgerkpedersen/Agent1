
from agent_core.orchestration.dependency_graph import DependencyGraph
from agent_core.orchestration.task_scheduler import TaskScheduler


def test_dependency_graph_init():
    dg = DependencyGraph()
    assert dg is not None


def test_dependency_graph_add_node():
    dg = DependencyGraph()
    node = dg.add_node("task1", task_type="test", priority=1)
    assert node is not None
    assert node.node_id == "task1"


def test_dependency_graph_add_edge():
    dg = DependencyGraph()
    dg.add_node("task1")
    dg.add_node("task2")
    dg.add_edge("task1", "task2")
    assert dg.has_node("task2")


def test_task_scheduler_init():
    scheduler = TaskScheduler()
    assert scheduler is not None
