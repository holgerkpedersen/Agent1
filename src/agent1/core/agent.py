"""Core agent implementation for Agent1.

This module provides the base Agent class that handles state management,
decision making, and interaction with external systems through a pluggable
architecture of brains, sensors, actuators, and memory stores.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class Agent:
    """A modular agent that coordinates brain, sensors, actuators, and memory.

    The agent runs a perception-action loop where it periodically reads from
    its sensors, feeds observations to its brain for decision making, stores
    important information in memory, and executes actions through actuators.

    Attributes:
        name (str): Human-readable identifier for the agent.
        brain (Any): Decision-making component responsible for producing actions.
        sensors (List[Any]): Input components that gather environmental data.
        actuators (List[Any]): Output components that execute agent decisions.
        memory (Optional[Any]): Persistent storage for learned information.
        config (Dict[str, Any]): Configuration parameters for the agent loop.
    """

    def __init__(
        self,
        name: str = "agent1",
        brain: Optional[Any] = None,
        sensors: Optional[List[Any]] = None,
        actuators: Optional[List[Any]] = None,
        memory: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the agent with its core components.

        Args:
            name: Identifier used in logs and telemetry.
            brain: Component implementing a ``decide`` method that accepts
                observations and returns an action or list of actions.
            sensors: List of components each implementing a ``read`` coroutine
                returning observation data.
            actuators: List of components each implementing an ``act`` coroutine
                accepting an action produced by the brain.
            memory: Optional component for storing/retrieving agent knowledge,
                expected to expose ``store`` and ``retrieve`` methods.
            config: Dictionary with optional keys ``loop_interval``,
                ``max_iterations``, and ``stop_event``.
        """
        self.name = name
        self.brain = brain
        self.sensors = sensors or []
        self.actuators = actuators or []
        self.memory = memory

        default_config: Dict[str, Any] = {
            "loop_interval": 1.0,
            "max_iterations": None,
            "stop_event": None,
        }
        if config is not None:
            default_config.update(config)
        self.config = default_config

        logger.info("Agent '%s' initialized with %d sensor(s), %d actuator(s)", name, len(self.sensors), len(self.actuators))

    async def _read_sensors(self) -> Dict[str, Any]:
        """Read all sensors concurrently and aggregate their observations.

        Returns:
            A dictionary mapping sensor names (or indices) to observation data.
        """
        if not self.sensors:
            return {}

        tasks = []
        for index, sensor in enumerate(self.sensors):
            label = getattr(sensor, "name", None) or f"sensor_{index}"
            tasks.append((label, sensor.read()))

        results: Dict[str, Any] = {}
        try:
            gathered = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Failed to gather sensor readings: %s", exc)
            return results

        for (label, _), data in zip(tasks, gathered):
            if isinstance(data, Exception):
                logger.warning("Sensor '%s' raised an error: %s", label, data)
            else:
                results[label] = data

        return results

    def _decide(self, observations: Dict[str, Any]) -> Union[Any, List[Any]]:
        """Delegate decision making to the brain.

        Args:
            observations: Aggregated sensor readings.

        Returns:
            An action or list of actions produced by the brain. If no brain is
            configured an empty list is returned.
        """
        if self.brain is None:
            logger.debug("No brain configured; returning default (no-op) action")
            return []

        try:
            decision = self.brain.decide(observations)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Brain '%s' raised an error during decide(): %s", getattr(self.brain, "name", None), exc)
            return []

        if decision is None:
            return []
        return decision

    async def _execute_actions(self, actions: Union[Any, List[Any]]) -> List[Any]:
        """Execute each action through the configured actuators.

        Args:
            actions: Single action or list of actions from the brain.

        Returns:
            A list of results returned by each actuator invocation.
        """
        if not self.actuators:
            logger.debug("No actuators configured; skipping action execution")
            return []

        if isinstance(actions, (list, tuple)):
            action_list = list(actions)
        else:
            action_list = [actions]

        results: List[Any] = []
        for index, actuator in enumerate(self.actuators):
            label = getattr(actuator, "name", None) or f"actuator_{index}"
            try:
                if hasattr(actuator, "act") and asyncio.iscoroutinefunction(actuator.act):
                    result = await actuator.act(action_list[index] if index < len(action_list) else action_list[0])
                elif hasattr(actuator, "act"):
                    result = actuator.act(action_list[index] if index < len(action_list) else action_list[0])
                else:
                    logger.warning("Actuator '%s' has no callable 'act' method", label)
                    result = None
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.error("Actuator '%s' raised an error during act(): %s", label, exc)
                result = None

            results.append(result)

        return results

    async def _store_memory(self, observations: Dict[str, Any], actions: Union[Any, List[Any]], actuator_results: List[Any]) -> None:
        """Persist the current cycle's data into memory if available.

        Args:
            observations: Sensor readings from this cycle.
            actions: Decisions produced by the brain.
            actuator_results: Outcomes of executing those decisions.
        """
        if self.memory is None:
            return

        try:
            record = {
                "observations": observations,
                "actions": actions,
                "actuator_results": actuator_results,
            }
            if hasattr(self.memory, "store"):
                if asyncio.iscoroutinefunction(self.memory.store):
                    await self.memory.store(record)
                else:
                    self.memory.store(record)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Memory store raised an error: %s", exc)

    async def step(self) -> Dict[str, Any]:
        """Execute one complete perception-action cycle.

        Returns:
            A dictionary summarizing observations, actions, and actuator results
            for this cycle.
        """
        logger.debug("Agent '%s' starting a new step", self.name)

        observations = await self._read_sensors()
        actions = self._decide(observations)
        actuator_results = await self._execute_actions(actions)

        await self._store_memory(observations, actions, actuator_results)

        cycle_summary: Dict[str, Any] = {
            "agent": self.name,
            "observations": observations,
            "actions": actions,
            "actuator_results": actuator_results,
        }
        logger.debug("Agent '%s' step complete", self.name)
        return cycle_summary

    async def run(self) -> None:
        """Run the perception-action loop until stopped or iteration limit hit.

        The loop respects ``config['stop_event']`` (an asyncio.Event that, when
        set, halts execution) and ``config['max_iterations']``.
        """
        stop_event = self.config.get("stop_event")
        max_iterations = self.config.get("max_iterations")
        interval = float(self.config.get("loop_interval", 1.0))

        iteration = 0
        logger.info("Agent '%s' entering run loop (interval=%ss)", self.name, interval)

        while True:
            if stop_event is not None and stop_event.is_set():
                logger.info("Stop event triggered; exiting agent loop")
                break

            if max_iterations is not None and iteration >= max_iterations:
                logger.info("Max iterations (%d) reached; exiting agent loop", max_iterations)
                break

            await self.step()
            iteration += 1

            if interval > 0:
                try:
                    if stop_event is not None:
                        # Wait for the interval or until stopped, whichever comes first.
                        wait_task = asyncio.ensure_future(asyncio.sleep(interval))
                        stop_task = asyncio.ensure_future(stop_event.wait())
                        done, pending = await asyncio.wait(
                            {wait_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
                        )
                        for task in pending:
                            task.cancel()
                        if stop_task in done and stop_event.is_set():
                            logger.info("Stop event triggered during sleep; exiting agent loop")
                            break
                    else:
                        await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    logger.info("Agent run cancelled during sleep")
                    raise

    async def async_shutdown(self) -> None:
        """Cleanly shut down the agent and its components asynchronously."""
        logger.info("Shutting down agent '%s'", self.name)

        for index, component in enumerate(list(self.sensors) + list(self.actuators)):
            label = getattr(component, "name", None) or f"component_{index}"
            if hasattr(component, "shutdown"):
                try:
                    shutdown_method = component.shutdown()
                    if asyncio.iscoroutine(shutdown_method):
                        await shutdown_method
                    else:
                        pass  # synchronous shutdown completed implicitly
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.error("Shutdown of component '%s' raised an error: %s", label, exc)

        if self.memory is not None and hasattr(self.memory, "shutdown"):
            try:
                method = self.memory.shutdown()
                if asyncio.iscoroutine(method):
                    await method
                else:
                    pass  # synchronous shutdown completed implicitly
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.error("Shutdown of memory raised an error: %s", exc)

        logger.info("Agent '%s' shutdown complete", self.name)