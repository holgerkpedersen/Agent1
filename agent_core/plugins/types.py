"""Plugin data structures: the structural interface and metadata record.

Moved verbatim from the retired ``src/agent1.core`` namespace.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


class PluginInterface(Protocol):
    def initialize(self, config: Dict[str, Any]) -> None: ...
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]: ...
    def cleanup(self) -> None: ...


@dataclass
class PluginMetadata:
    name: str
    version: str
    description: str
    author: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


__all__ = [
    "PluginInterface",
    "PluginMetadata",
]
