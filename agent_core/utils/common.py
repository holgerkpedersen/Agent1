from typing import Any, Optional

import json

def convert_value(value: str) -> Any:
    """Converts a string value to its appropriate Python type."""

    val = value.strip()
    lowered = val.lower()
    
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("none", "null"):
        return None
    
    try:
        # Try to parse as JSON for lists, dicts, and numbers
        return json.loads(val)
    except (json.JSONDecodeError, ValueError):
        return val

def get_nested(data: dict[str, Any], key_path: str | list[str]) -> Any:
    """Retrieves a value from a nested dictionary using a dot-separated string or list of keys."""
    keys = key_path.split(".") if isinstance(key_path, str) else key_path
    
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current

def set_nested(data: dict[str, Any], key_path: str | list[str], value: Any) -> None:
    """Sets a value in a nested dictionary, creating intermediate dictionaries if necessary."""
    keys = key_path.split(".") if isinstance(key_path, str) else key_path
    
    current = data
    for i in range(len(keys) - 1):
        key = keys[i]
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value

def setup_logging(config_manager: Optional[Any] = None) -> None:
    """Configures the global logging system based on the provided configuration manager."""
    # This is a placeholder for shared logging initialization logic
    pass
