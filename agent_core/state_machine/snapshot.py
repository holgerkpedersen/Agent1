from typing import Any, Optional, List


class StateMachineError(Exception):
    """Base exception for all state machine related errors."""
    pass


class InvalidTransitionError(StateMachineError):
    """Raised when a requested transition is not permitted by the rules."""
    pass


class MissingStateError(StateMachineError):
    """Raised when a transition refers to a state that does not exist."""
    pass


class Snapshot:
    """Represents a point-in-time capture of the StateMachine's status."""

    def __init__(self, current_state: str, history: List[str], metadata: dict[str, Any]) -> None:
        self.current_state = current_state
        self.history = history
        self.metadata = metadata


class StateMachine:
    """
    A deterministic state machine supporting snapshots and rollbacks 
    to facilitate safe mutation testing and state recovery.
    """

    def __init__(self, initial_state: str) -> None:
        self._initial_state = initial_state
        self._current_state = initial_state
        self._history: List[str] = [initial_state]
        self._metadata: dict[str, Any] = {}

    def get_current_state(self) -> str:
        return self._current_state

    def get_history(self) -> List[str]:
        return list(self._history)

    def get_metadata(self, key: Optional[str] = None) -> Any:
        if key is None:
            return self._metadata
        return self._metadata.get(key)

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def take_snapshot(self) -> Snapshot:
        return Snapshot(
            current_state=self._current_state,
            history=list(self._history),
            metadata=dict(self._metadata)
        )

    def rollback_latest(self) -> Snapshot:
        if len(self._history) > 1:
            self._history.pop()
            self._current_state = self._history[-1]
        return self.take_snapshot()

    def rollback_to(self, snapshot: Snapshot) -> None:
        self._current_state = snapshot.current_state
        self._history = list(snapshot.history)
        self._metadata = dict(snapshot.metadata)

    def reset(self) -> None:
        self._current_state = self._initial_state
        self._history = [self._initial_state]
        self._metadata = {}

    def transition(self, target_state: str, **kwargs: Any) -> Optional[Any]:
        if not target_state:
            raise InvalidTransitionError("Target state cannot be empty.")
        
        # In this implementation, any non-empty string is a valid state 
        # unless specific transition constraints are added.
        self._current_state = target_state
        self._history.append(target_state)
        return self._current_state

    def __hash__(self) -> int:
        # Return identity hash to ensure the machine remains hashable 
        # while allowing internal state mutations.
        return hash(id(self))