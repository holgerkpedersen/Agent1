from typing import Any

class ExecutionMetric:
    def __init__(self, score: float, success: bool, latency: float) -> None:
        self.score = score
        self.success = success
        self.latency = latency

class EvolutionMetrics:
    def __init__(self, window_size: int = 10, threshold: float = 0.75) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self._history: list[ExecutionMetric] = []

    def record(self, metric: ExecutionMetric) -> None:
        """Records a new execution metric and maintains the sliding window."""
        self._history.append(metric)
        if len(self._history) > self.window_size:
            self._history.pop(0)

    def average_score(self) -> float:
        """Calculates the mean score of all metrics in the current window."""
        if not self._history:
            return 0.0
        return sum(m.score for m in self._history) / len(self._history)

    def recent_metrics(self) -> list[ExecutionMetric]:
        """Returns the history of execution metrics within the window."""
        return self._history

    def score(self) -> float:
        """Returns the score of the most recent execution."""
        if not self._history:
            return 0.0
        return self._history[-1].score

    def should_evolve(self) -> bool:
        """Determines if evolution is required based on average performance."""
        # Evolution is triggered when quality falls below the defined threshold
        return self.average_score() < self.threshold

    def summary(self) -> dict[str, Any]:
        """Provides a snapshot of current evolution metrics and status."""
        return {
            "window_size": self.window_size,
            "threshold": self.threshold,
            "average_score": self.average_score(),
            "history_count": len(self._history),
            "should_evolve": self.should_evolve()
        }