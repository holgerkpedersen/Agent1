from typing import Dict, Any, List, Optional, Type
from ..core import PluginInterface, PluginMetadata


class BasePlugin:
    """Concrete base implementation of the plugin interface."""

    def __init__(self) -> None:
        self._metadata: PluginMetadata = PluginMetadata(
            name=self.__class__.__name__, version="1.0", description=""
        )
        self._initialized: bool = False
        self._config: Dict[str, Any] = {}

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    def initialize(self, config: Dict[str, Any]) -> None:
        self._config = dict(config)
        self._initialized = True

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Plugin must be initialized before execution")
        return {"result": "not implemented"}

    def cleanup(self) -> None:
        self._initialized = False


class PluginRegistry:
    """Central registry for available plugin classes."""

    def __init__(self) -> None:
        self._plugins: Dict[str, Type[BasePlugin]] = {}
        self._metadata_cache: Dict[str, PluginMetadata] = {}

    def register(self, plugin_class: Type[BasePlugin]) -> str:
        instance: BasePlugin = plugin_class()
        name: str = instance.metadata.name
        if not name:
            raise ValueError("Plugin must have a valid non-empty name")
        self._plugins[name] = plugin_class
        self._metadata_cache[name] = instance.metadata
        return name

    def unregister(self, name: str) -> bool:
        removed_plugin: Optional[Type[BasePlugin]] = self._plugins.pop(name, None)
        removed_meta: Optional[PluginMetadata] = self._metadata_cache.pop(name, None)
        return removed_plugin is not None and removed_meta is not None

    def get_plugin_class(self, name: str) -> Optional[Type[BasePlugin]]:
        return self._plugins.get(name)

    def list_plugins(self) -> List[str]:
        return list(self._plugins.keys())

    def get_metadata(self, name: str) -> Optional[PluginMetadata]:
        return self._metadata_cache.get(name)


class PluginManager:
    """Manages plugin lifecycle and execution."""

    def __init__(self, registry: PluginRegistry) -> None:
        self._registry: PluginRegistry = registry
        self._loaded_plugins: Dict[str, BasePlugin] = {}
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}

    def load_plugin(self, name: str) -> bool:
        plugin_class: Optional[Type[BasePlugin]] = self._registry.get_plugin_class(name)
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


__all__: List[str] = [
    "PluginInterface",
    "PluginMetadata",
    "BasePlugin",
    "PluginRegistry",
    "PluginManager",
]