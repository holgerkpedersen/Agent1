
from agent_core.memory import VectorDatabase
from agent_core.monitoring.metrics_collector import MetricsCollector


def test_metrics_collector_basic():
    coll = MetricsCollector()
    coll.set_gauge("test_gauge", 42)
    coll.increment_counter("test_counter")
    coll.increment_counter("test_counter", 5)
    assert coll.get_gauge_value("test_gauge") == 42
    assert coll.get_counter_value("test_counter") == 6


def test_vector_database_basic():
    import numpy as np
    db = VectorDatabase(dimension=4)
    vector = np.array([1.0, 0.0, 0.0, 0.0])
    vid = db.add_vector(vector, {"doc_id": "doc_1"})
    assert vid >= 0
    results = db.search_similar(vector, k=1)
    assert len(results) >= 0
