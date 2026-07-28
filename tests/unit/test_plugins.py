import pytest

from src.agent1.plugins import BasePlugin, PluginRegistry, PluginManager


class SamplePlugin(BasePlugin):
    """Test plugin that counts executions."""

    def __init__(self) -> None:
        super().__init__()
        self.executed = False

    def execute(self, input_data):
        self.executed = True
        return {"result": "ok", "input": input_data}


def test_plugin_registry_register():
    """Test registering a plugin."""
    registry = PluginRegistry()
    name = registry.register(SamplePlugin)
    assert name == "SamplePlugin"
    assert "SamplePlugin" in registry.list_plugins()


def test_plugin_registry_unregister():
    """Test unregistering a plugin."""
    registry = PluginRegistry()
    registry.register(SamplePlugin)
    assert registry.unregister("SamplePlugin")
    assert "SamplePlugin" not in registry.list_plugins()


def test_plugin_registry_get_class():
    """Test getting a registered plugin class."""
    registry = PluginRegistry()
    registry.register(SamplePlugin)
    cls = registry.get_plugin_class("SamplePlugin")
    assert cls is SamplePlugin


def test_plugin_manager_load():
    """Test loading a plugin."""
    registry = PluginRegistry()
    registry.register(SamplePlugin)
    manager = PluginManager(registry)
    assert manager.load_plugin("SamplePlugin")
    assert "SamplePlugin" in manager.list_loaded_plugins()


def test_plugin_manager_execute():
    """Test executing a loaded plugin."""
    registry = PluginRegistry()
    registry.register(SamplePlugin)
    manager = PluginManager(registry)
    manager.load_plugin("SamplePlugin")
    result = manager.execute_plugin("SamplePlugin", {"task": "test"})
    assert result is not None
    assert result["result"] == "ok"


def test_plugin_manager_unload():
    """Test unloading a plugin."""
    registry = PluginRegistry()
    registry.register(SamplePlugin)
    manager = PluginManager(registry)
    manager.load_plugin("SamplePlugin")
    assert manager.unload_plugin("SamplePlugin")
    assert "SamplePlugin" not in manager.list_loaded_plugins()
