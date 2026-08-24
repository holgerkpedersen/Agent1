from typing import Dict, Any, Optional, Type, List
import importlib.util
from pathlib import Path

from .types import PluginInterface, PluginMetadata


class BasePlugin:
    """Base class providing the common plugin lifecycle and metadata.

    Concrete plugins subclass ``BasePlugin`` (or any object satisfying the
    :class:`PluginInterface` protocol) to implement custom behaviour via
    ``initialize``, ``execute`` and ``cleanup``.
    """

    def __init__(self) -> None:
        self._metadata: Optional[PluginMetadata] = None
        self._initialized: bool = False
        self._config: Dict[str, Any] = {}

    @property
    def metadata(self) -> PluginMetadata:
        if self._metadata is None:
            return PluginMetadata(
                name=self.__class__.__name__,
                version="1.0.0",
                description="",
            )
        return self._metadata

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with a configuration dictionary."""
        self._config = dict(config)
        self._initialized = True

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the plugin against ``input_data`` and return results.

        Subclasses should override this to provide meaningful behaviour; the
        default implementation echoes receipt of the input payload.
        """
        if not self._initialized:
            raise RuntimeError("Plugin must be initialized before execution")
        return {"status": "ok", "input_received": input_data}

    def cleanup(self) -> None:
        """Release any resources held by the plugin."""
        self._initialized = False


def verify_plugin_interface(plugin: PluginInterface) -> bool:
    """Verify an object conforms to the structural ``PluginInterface`` protocol."""
    return (
        callable(getattr(plugin, "initialize", None))
        and callable(getattr(plugin, "execute", None))
        and callable(getattr(plugin, "cleanup", None))
    )


def load_plugin_class_from_path(path: Path, class_name: str) -> Optional[Type[BasePlugin]]:
    """Dynamically discover and load a ``BasePlugin`` subclass from a file path."""
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("plugin_module", str(path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        return None
    cls = getattr(module, class_name, None)
    if isinstance(cls, type) and issubclass(cls, BasePlugin):
        return cls
    return None


__all__: List[str] = ["BasePlugin", "verify_plugin_interface", "load_plugin_class_from_path"]
