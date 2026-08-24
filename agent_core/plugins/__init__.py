"""Plugin subsystem: lifecycle, registry, and dynamic loading.

Moved from the retired ``src/agent1.plugins`` namespace into ``agent_core``.
"""

from typing import List

from .base_plugin import (
    BasePlugin,
    load_plugin_class_from_path,
    verify_plugin_interface,
)
from .plugin_manager import PluginManager
from .registry import PluginRegistry
from .types import PluginInterface, PluginMetadata

__all__: List[str] = [
    "PluginInterface",
    "PluginMetadata",
    "BasePlugin",
    "verify_plugin_interface",
    "load_plugin_class_from_path",
    "PluginRegistry",
    "PluginManager",
]
