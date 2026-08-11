import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future, wait
from typing import List, Any, Callable, Dict, Optional


class Orchestrator:
    """
    Manages a swarm of agents and coordinates task execution across them.
    """

    def __init__(self, agents: List[Any], max_workers: int = 20) -> None:
        self.agents = agents
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: Dict[int, Future[Any]] = {}
        self._task_counter: int = 0
        self._lock: threading.Lock = threading.Lock()
        self._logger: logging.Logger = logging.getLogger(__name__)

    def __enter__(self) -> "Orchestrator":
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[Any]) -> None:
        if exc_val is not None:
            self._logger.error("Orchestrator exited with exception: %s", exc_val)
        self.shutdown(wait=True)

    def dispatch(self, task_func: Callable[..., Any], *args: Any, **kwargs: Any) -> int:
        """
        Dispatches a task to the swarm and returns a unique task identifier.
        """
        with self._lock:
            task_id = self._task_counter
            self._task_counter += 1

        future = self._executor.submit(task_func, *args, **kwargs)
        self._tasks[task_id] = future
        self._logger.debug("Dispatched task %d", task_id)
        return task_id

    def get_result(self, task_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves the result of a completed task by its ID.
        """
        future = self._tasks.get(task_id)
        if future is not None and future.done():
            try:
                result = future.result()
                return result if isinstance(result, dict) else {"data": result}
            except Exception as exc:
                self._logger.error("Task %d failed with exception: %s", task_id, exc)
                return {"error": str(exc), "task_id": task_id}
        return None

    def shutdown(self, wait: bool = True) -> None:
        """
        Shuts down the executor and clears all pending tasks.
        """
        self._executor.shutdown(wait=wait)
        with self._lock:
            self._tasks.clear()
        self._logger.info("Orchestrator shutdown complete")

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Blocks until all dispatched tasks are completed or the timeout is reached.
        """
        with self._lock:
            futures = list(self._tasks.values())

        if not futures:
            return True

        done, _ = wait(futures, timeout=timeout)
        completed = len(done) == len(futures)
        if not completed and timeout is not None:
            pending = len(futures) - len(done)
            self._logger.warning("Timeout waiting for %d tasks", pending)
        return completed