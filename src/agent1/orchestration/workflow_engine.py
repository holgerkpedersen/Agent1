"""Workflow engine module providing orchestration capabilities."""

import asyncio
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class WorkflowStatus(Enum):
    """Enumeration of possible workflow statuses."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepResult:
    """Represents the result of a single workflow step execution."""

    def __init__(self, success: bool, output: Any = None, error: Optional[str] = None):
        self.success = success
        self.output = output
        self.error = error


class WorkflowStep:
    """Represents an individual step within a workflow."""

    def __init__(self, name: str, func: Callable[[], asyncio.Future], timeout: float = 30.0):
        self.name = name
        self.func = func
        self.timeout = timeout
        self.status: WorkflowStatus = WorkflowStatus.PENDING
        self.result: Optional[StepResult] = None

    async def execute(self) -> StepResult:
        """Execute the step's function with a timeout."""
        self.status = WorkflowStatus.RUNNING
        try:
            output = await asyncio.wait_for(self.func(), timeout=self.timeout)
            self.result = StepResult(success=True, output=output)
            self.status = WorkflowStatus.COMPLETED
        except asyncio.TimeoutError:
            self.result = StepResult(success=False, error=f"Step '{self.name}' timed out after {self.timeout}s")
            self.status = WorkflowStatus.FAILED
        except Exception as exc:  # noqa: BLE001
            self.result = StepResult(success=False, error=str(exc))
            self.status = WorkflowStatus.FAILED
        return self.result


class WorkflowEngine:
    """Orchestrates execution of workflows composed of multiple steps."""

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self.workflows: Dict[str, List[WorkflowStep]] = {}
        self.workflow_statuses: Dict[str, WorkflowStatus] = {}

    def register_workflow(self, name: str, steps: List[WorkflowStep]) -> None:
        """Register a new workflow with its constituent steps."""
        if not steps:
            raise ValueError("A workflow must contain at least one step")
        self.workflows[name] = steps
        self.workflow_statuses[name] = WorkflowStatus.PENDING

    def get_workflow(self, name: str) -> Optional[List[WorkflowStep]]:
        """Retrieve a registered workflow by name."""
        return self.workflows.get(name)

    async def run_workflow(self, name: str) -> Dict[str, Any]:
        """Run an entire workflow sequentially and collect results.

        Returns a dictionary containing the overall status and per-step results.
        A cancellation only takes effect after the currently-running step has
        reached a terminal (COMPLETED/FAILED) status — no RUNNING step is left
        orphaned when control returns to the caller.
        """
        if name not in self.workflows:
            raise KeyError(f"Workflow '{name}' is not registered")

        steps = self.workflows[name]
        self.workflow_statuses[name] = WorkflowStatus.RUNNING
        step_results: List[Dict[str, Any]] = []

        for step in steps:
            if self.workflow_statuses[name] == WorkflowStatus.CANCELLED and step.status != WorkflowStatus.RUNNING:
                break
            # If a cancellation arrived while this step is RUNNING, let it finish
            # its bounded execution (step.execute() enforces the timeout) so that
            # every RUNNING task reaches a terminal status before we stop.
            result = await step.execute()
            step_results.append({
                "name": step.name,
                "status": step.status.value,
                "success": result.success,
                "output": result.output,
                "error": result.error,
            })

        if self.workflow_statuses[name] == WorkflowStatus.CANCELLED:
            final_status = WorkflowStatus.CANCELLED
        elif all(r["success"] for r in step_results):
            final_status = WorkflowStatus.COMPLETED
        else:
            final_status = WorkflowStatus.FAILED

        self.workflow_statuses[name] = final_status
        return {
            "workflow": name,
            "status": final_status.value,
            "steps": step_results,
        }

    def cancel_workflow(self, name: str) -> None:
        """Cancel a running workflow."""
        if name not in self.workflow_statuses:
            raise KeyError(f"Workflow '{name}' is not registered")
        current = self.workflow_statuses[name]
        if current == WorkflowStatus.RUNNING:
            self.workflow_statuses[name] = WorkflowStatus.CANCELLED
        else:
            self._logger.warning(
                "Cannot cancel workflow '%s' with status %s", name, current.value
            )

    def get_workflow_status(self, name: str) -> Optional[WorkflowStatus]:
        """Return the current status of a registered workflow."""
        return self.workflow_statuses.get(name)


__all__ = [
    "WorkflowEngine",
    "WorkflowStep",
    "StepResult",
    "WorkflowStatus",
]