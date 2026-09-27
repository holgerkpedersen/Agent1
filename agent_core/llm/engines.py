"""Engine / accelerator abstraction for multi-device parallelism.

A *provider* is the transport (how the agent talks to a backend); an *engine*
is WHERE the compute happens.  Mapping engines to the agent's providers lets a
single prompt run on two accelerators at once — e.g. the AMD XDNA NPU and the
Radeon iGPU — instead of only failing over between them:

    npu    -> lemonade    (AMD XDNA via Lemonade Server / FastFlowLM)
    igpu   -> lmstudio    (Radeon iGPU via LM Studio / llama.cpp)
    llama  -> llama       (llama.cpp llama-server)
    cloud  -> opencode    (hosted opencode gateway)
    cloud  -> openrouter  (hosted OpenRouter gateway)

``device_for_provider`` labels telemetry/scheduling by accelerator; ``ENGINES``
and :func:`engine_model` let ``multillm --engines npu,igpu`` pick one model per
accelerator from the live catalogs (falling back to configured defaults).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_core.constants import (
    DEFAULT_MODEL,
    DEFAULT_OPENCODE_MODEL,
    DEFAULT_OPENROUTER_MODEL,
)

DEVICE_NPU = "npu"
DEVICE_IGPU = "igpu"
DEVICE_CPU = "cpu"
DEVICE_CLOUD = "cloud"

#: Accelerator class for each provider (agent-side default; a local provider
#: can run on the CPU when no accelerator is available, but the device label is
#: the intended accelerator — good enough to group parallel work).
_PROVIDER_DEVICE: dict[str, str] = {
    "lemonade": DEVICE_NPU,
    "lmstudio": DEVICE_IGPU,
    "llama": DEVICE_IGPU,
    "opencode": DEVICE_CLOUD,
    "openrouter": DEVICE_CLOUD,
}


def device_for_provider(provider: str) -> str:
    """Accelerator device for a provider name (``cloud`` when unknown)."""
    return _PROVIDER_DEVICE.get(str(provider or "").lower(), DEVICE_CLOUD)


def device_for_model(
    model: str,
    provider_setting: str = "lmstudio",
    persisted_provider: str | None = None,
) -> str:
    """Accelerator device the given *model* will run on."""
    from .provider import provider_for

    return device_for_provider(
        provider_for(model, provider_setting, persisted_provider)
    )


@dataclass(frozen=True)
class Engine:
    """A named accelerator the agent can target."""

    name: str
    provider: str
    device: str
    description: str


ENGINES: dict[str, Engine] = {
    "npu": Engine("npu", "lemonade", DEVICE_NPU,
                  "AMD XDNA NPU via Lemonade (FastFlowLM/Ryzen AI LLM)"),
    "igpu": Engine("igpu", "lmstudio", DEVICE_IGPU,
                   "Radeon iGPU via LM Studio"),
    "llama": Engine("llama", "llama", DEVICE_IGPU,
                    "llama.cpp llama-server"),
    "cloud": Engine("cloud", "opencode", DEVICE_CLOUD,
                    "hosted opencode gateway"),
    "openrouter": Engine("openrouter", "openrouter", DEVICE_CLOUD,
                         "hosted OpenRouter gateway"),
}


def resolve_engine(name: str) -> Engine | None:
    """Look up an engine by name (case-insensitive), or ``None``."""
    return ENGINES.get(str(name or "").strip().lower())


def engine_names() -> list[str]:
    return sorted(ENGINES)


def engine_model(
    engine: Engine, settings: Any = None, *, probe: bool = True,
) -> str | None:
    """Pick a concrete model for *engine*.

    Prefers the live catalog (Lemonade NPU list / loaded LM Studio model) so
    the engine targets something that actually exists, then falls back to the
    configured default (opencode/openrouter/Lemonade default).  ``probe=False``
    skips the catalog query (deterministic, used in tests).  Returns ``None``
    when no model can be determined.
    """
    provider = engine.provider
    if provider == "lemonade":
        if probe:
            try:
                from .lemonade_provider import LemonadeProvider

                models = LemonadeProvider(
                    api_url=getattr(settings, "lemonade_api_url", None)
                ).list_models()
                if models:
                    return str(models[0])
            except Exception:  # noqa: BLE001 - discovery must never break
                pass
        default = str(getattr(settings, "lemonade_model", "") or "").strip()
        return f"lemonade/{default}" if default else None
    if provider == "lmstudio":
        if probe:
            try:
                from .lmstudio import get_models_status

                loaded = [m["key"] for m in get_models_status() if m.get("loaded")]
                if loaded:
                    return str(loaded[0])
            except Exception:  # noqa: BLE001
                pass
        return DEFAULT_MODEL
    if provider == "opencode":
        return str(getattr(settings, "opencode_model", "") or DEFAULT_OPENCODE_MODEL)
    if provider == "openrouter":
        return str(
            getattr(settings, "openrouter_model", "") or DEFAULT_OPENROUTER_MODEL
        )
    # llama.cpp needs the running server to report its model id; not resolvable
    # from a static catalog.
    return None


def resolve_engines(
    names: list[str], settings: Any = None, *, probe: bool = True,
) -> tuple[list[str], dict[str, str], list[str]]:
    """Expand engine names to ``(models, provider_overrides, errors)``.

    The returned ``provider_overrides`` pins each model to its engine's provider
    so routing cannot be hijacked by a persisted provider of a different type
    (e.g. an LM Studio model name while ``model.json`` says ``lemonade``).
    """
    models: list[str] = []
    overrides: dict[str, str] = {}
    errors: list[str] = []
    for raw in names:
        name = str(raw or "").strip().lower()
        if not name:
            continue
        engine = resolve_engine(name)
        if engine is None:
            known = ", ".join(engine_names())
            errors.append(f"unknown engine '{name}' (known: {known})")
            continue
        model = engine_model(engine, settings, probe=probe)
        if not model:
            errors.append(f"no model available for engine '{name}'")
            continue
        if model in overrides:
            continue
        models.append(model)
        overrides[model] = engine.provider
    return models, overrides, errors


__all__: list[str] = [
    "DEVICE_CLOUD",
    "DEVICE_CPU",
    "DEVICE_IGPU",
    "DEVICE_NPU",
    "ENGINES",
    "Engine",
    "device_for_model",
    "device_for_provider",
    "engine_model",
    "engine_names",
    "resolve_engine",
    "resolve_engines",
]
