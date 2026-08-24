from typing import Any, Dict, List, Optional

from .base_plugin import BasePlugin
from .registry import PluginRegistry


class PluginManager:
    """Manages plugin lifecycle and execution."""

    def __init__(self, registry: PluginRegistry) -> None:
        self._registry: PluginRegistry = registry
        self._loaded_plugins: Dict[str, BasePlugin] = {}
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}

    def load_plugin(self, name: str) -> bool:
        plugin_class = self._registry.get_plugin_class(name)
        if plugin_class is None:
            return False
        instance: BasePlugin = plugin_class()
        config: Dict[str, Any] = {}
        instance.initialize(config)
        self._loaded_plugins[name] = instance
        self._plugin_configs[name] = dict(config)
        return True

    def unload_plugin(self, name: str) -> bool:
        plugin_instance: Optional[BasePlugin] = self._loaded_plugins.pop(name, None)
        config_removed: Optional[Dict[str, Any]] = self._plugin_configs.pop(name, None)
        if plugin_instance is not None and config_removed is not None:
            plugin_instance.cleanup()
            return True
        return False

    def execute_plugin(self, name: str, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        plugin_instance: Optional[BasePlugin] = self._loaded_plugins.get(name)
        if plugin_instance is None:
            return None
        return plugin_instance.execute(input_data)

    def list_loaded_plugins(self) -> List[str]:
        return list(self._loaded_plugins.keys())
