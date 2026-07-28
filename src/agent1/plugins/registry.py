from typing import Dict, List, Optional, Type

from ..core import PluginMetadata
from .base_plugin import BasePlugin


class PluginRegistry:
    """Central registry for available plugin classes and their metadata."""

    def __init__(self) -> None:
        self._plugins: Dict[str, Type[BasePlugin]] = {}
        self._metadata_cache: Dict[str, PluginMetadata] = {}

    def register(self, plugin_class: Type[BasePlugin]) -> str:
        instance = plugin_class()
        name = instance.metadata.name
        if not name:
            raise ValueError("Plugin must expose a non-empty metadata name")
        self._plugins[name] = plugin_class
        self._metadata_cache[name] = instance.metadata
        return name

    def unregister(self, name: str) -> bool:
        existed = name in self._plugins
        if existed:
            del self._plugins[name]
            del self._metadata_cache[name]
        return existed

    def get_plugin_class(self, name: str) -> Optional[Type[BasePlugin]]:
        return self._plugins.get(name)

    def list_plugins(self) -> List[str]:
        return list(self._plugins.keys())

    def get_metadata(self, name: str) -> Optional[PluginMetadata]:
        return self._metadata_cache.get(name)


__all__: List[str] = ["PluginRegistry"]