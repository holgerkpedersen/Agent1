import pytest
from agent_core.swarm_orchestrator import Orchestrator

def test_dispatch_speculative():
    """Test that speculative branches are dispatched and return results."""
    with Orchestrator(agents=[], max_workers=4) as orch:
        def reasoning_func(branch_id, context):
            # Simulate some work
            return {"branch": branch_id, "context_val": context["value"]}

        context = {"value": 100}
        num_branches = 3
        task_ids = orch.dispatch_speculative(reasoning_func, context, num_branches=num_branches)

        assert len(task_ids) == num_branches
        assert orch.wait_for_completion(timeout=5) is True

        results = []
        for tid in task_ids:
            res = orch.get_result(tid)
            results.append(res)

        # Verify all results are correct
        for i in range(num_branches):
            assert results[i] == {"branch": i, "context_val": 100}

def test_dispatch_speculative_exception():
    """Test that an exception in one speculative branch doesn't crash the orchestrator."""
    with Orchestrator(agents=[], max_workers=4) as orch:
        def reasoning_func(branch_id, context):
            if branch_id == 1:
                raise ValueError("Branch 1 failed")
            return {"branch": branch_id}

        context = {}
        task_ids = orch.dispatch_speculative(reasoning_func, context, num_branches=3)
        assert orch.wait_for_completion(timeout=5) is True

        results = []
        for tid in task_ids:
            results.append(orch.get_result(tid))

        # Branch 0 and 2 should be fine
        assert results[0] == {"branch": 0}
        assert results[2] == {"branch": 2}
        # Branch 1 should have the error
        assert "error" in results[1]
        assert results[1]["error"] == "Branch 1 failed"

def test_dispatch_speculative_empty():
    """Test dispatching zero branches."""
    with Orchestrator(agents=[], max_workers=4) as orch:
        def reasoning_func(branch_id, context):
            return {"branch": branch_id}
        
        task_ids = orch.dispatch_speculative(reasoning_func, {}, num_branches=0)
        assert len(task_ids) == 0
        assert orch.wait_for_completion(timeout=5) is True
