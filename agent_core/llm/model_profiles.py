from __future__ import annotations

import dataclasses
import json
import os
from typing import Dict, Final, List, Optional


@dataclasses.dataclass
class ProfileMetadata:
    name: str
    description: str = ""
    model: str = ""              # model key to use with this profile
    temperature: float = 0.7
    max_tokens: int = 50000
    thinking: Optional[bool] = None  # None = use model default

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ProfileMetadata:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


_BUILTIN_PROFILES: Final[Dict[str, ProfileMetadata]] = {
    "fast-codegen": ProfileMetadata(
        name="fast-codegen",
        description="Low temperature, low tokens — fast code generation",
        temperature=0.1,
        max_tokens=16000,
    ),
    "deep-analysis": ProfileMetadata(
        name="deep-analysis",
        description="High temperature, max tokens — deep analysis and brainstorming",
        temperature=0.7,
        max_tokens=50000,
    ),
    "precise": ProfileMetadata(
        name="precise",
        description="Medium temperature, high tokens — careful code fixes",
        temperature=0.3,
        max_tokens=50000,
    ),
}

_PROFILES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "profiles.json")

_user_profiles: Dict[str, ProfileMetadata] = {}


def _load_user_profiles() -> None:
    global _user_profiles
    try:
        with open(_PROFILES_FILE, "r") as f:
            data = json.load(f)
        _user_profiles = {k: ProfileMetadata(**v) for k, v in data.items()}
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        _user_profiles = {}


def _save_user_profiles() -> None:
    try:
        with open(_PROFILES_FILE, "w") as f:
            json.dump({k: v.as_dict() for k, v in _user_profiles.items()}, f, indent=2)
    except OSError:
        pass


def list_profiles() -> List[ProfileMetadata]:
    """Return all profiles (built-in + user, user overrides built-in)."""
    _load_user_profiles()
    merged = dict(_BUILTIN_PROFILES)
    merged.update(_user_profiles)
    return list(merged.values())


def get_profile(name: str) -> ProfileMetadata:
    """Get a profile by name. Raises KeyError if not found."""
    _load_user_profiles()
    if name in _user_profiles:
        return _user_profiles[name]
    if name in _BUILTIN_PROFILES:
        return _BUILTIN_PROFILES[name]
    raise KeyError(f"Unknown profile: {name!r}")


def save_profile(profile: ProfileMetadata) -> None:
    """Save a user profile (persisted to profiles.json)."""
    _load_user_profiles()
    _user_profiles[profile.name] = profile
    _save_user_profiles()


def delete_profile(name: str) -> bool:
    """Delete a user profile. Returns False if it's a built-in."""
    _load_user_profiles()
    if name in _BUILTIN_PROFILES:
        return False  # can't delete built-ins
    if name in _user_profiles:
        del _user_profiles[name]
        _save_user_profiles()
        return True
    return False


def apply_profile(payload: dict, profile: ProfileMetadata) -> dict:
    """Apply profile overrides to an LLM request payload."""
    if profile.temperature is not None:
        payload["temperature"] = profile.temperature
    if profile.max_tokens is not None:
        payload["max_tokens"] = profile.max_tokens
    if profile.thinking is not None:
        payload["thinking"] = {"type": "enabled" if profile.thinking else "disabled"}
    return payload
