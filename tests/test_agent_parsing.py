from __future__ import annotations

from typing import Any


class BaseConfig:
    """Minimal dict-backed configuration base class."""

    def __init__(self, **kwargs: Any) -> None:
        self._fields: dict[str, Any] = dict(kwargs)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._fields)


class CoreConfig(BaseConfig):
    pass


class AgentConfig(BaseConfig):
    _known_fields: tuple[str, ...] = ("workspace", "model_name")


def split_kv_pairs(kv_str: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if not kv_str:
        return pairs
    for token in kv_str.split(","):
        token = token.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        pairs.append((key.strip(), value.strip()))
    return pairs


def _fields_from_source(source: str | dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if isinstance(source, dict):
        fields.update(source)
    else:
        for key, value in split_kv_pairs(str(source)):
            fields[key] = value
    return fields


def parse_base_config(source: str | dict[str, Any]) -> BaseConfig:
    return BaseConfig(**_fields_from_source(source))


def parse_core_config(source: str | dict[str, Any]) -> CoreConfig:
    return CoreConfig(**_fields_from_source(source))


def parse_agent_config(source: str | dict[str, Any]) -> AgentConfig:
    return AgentConfig(**_fields_from_source(source))


def merge_configs(*configs: BaseConfig) -> BaseConfig:
    merged: dict[str, Any] = {}
    for config in configs:
        merged.update(config.as_dict())
    # Later configs override earlier ones; keys absent from a later config are preserved.
    return CoreConfig(**merged)


def test_split_kv_pairs_basic() -> None:
    pairs = split_kv_pairs("workspace=/tmp,model_name=gpt-4")
    assert pairs == [("workspace", "/tmp"), ("model_name", "gpt-4")]


def test_split_kv_pairs_empty_string() -> None:
    assert split_kv_pairs("") == []


def test_parse_base_config_is_baseconfig() -> None:
    result = parse_base_config("a=1,b=2")
    assert isinstance(result, BaseConfig)
    assert not isinstance(result, CoreConfig)
    assert not isinstance(result, AgentConfig)


def test_parse_core_config_is_coreconfig() -> None:
    result = parse_core_config({"x": "v"})
    assert isinstance(result, CoreConfig)
    assert issubclass(CoreConfig, BaseConfig)
    assert not isinstance(result, AgentConfig)


def test_agent_config_is_baseconfig_subtype() -> None:
    assert issubclass(AgentConfig, BaseConfig)
    instance = parse_agent_config("workspace=/tmp")
    assert isinstance(instance, AgentConfig)
    assert isinstance(instance, BaseConfig)


def test_parse_agent_config_returns_agentconfig_instance() -> None:
    result = parse_agent_config({"model_name": "gpt-4"})
    assert isinstance(result, AgentConfig)


def test_parse_agent_config_empty_dict_still_instance() -> None:
    result = parse_agent_config({})
    assert isinstance(result, AgentConfig)
    assert result.as_dict() == {}


def test_parse_agent_config_from_dict_preserves_known_fields() -> None:
    data = {"workspace": "/tmp", "model_name": "gpt-4"}
    result = parse_agent_config(data)
    fields = result.as_dict()
    for field in AgentConfig._known_fields:
        assert field in fields
    assert fields["workspace"] == "/tmp"
    assert fields["model_name"] == "gpt-4"


def test_parse_agent_config_from_string_matches_split_kv_pairs() -> None:
    source = "workspace=/tmp,model_name=gpt-4"
    expected_fields = dict(split_kv_pairs(source))
    result = parse_agent_config(source)
    assert result.as_dict() == expected_fields


def test_parse_agent_config_string_then_dict_consistency() -> None:
    string_source = "workspace=/tmp,model_name=gpt-4"
    dict_source = {"workspace": "/tmp", "model_name": "gpt-4"}
    from_string = parse_agent_config(string_source).as_dict()
    from_dict = parse_agent_config(dict_source).as_dict()
    assert from_string == from_dict


def test_merge_configs_combines_distinct_keys() -> None:
    first = AgentConfig(a="x")
    second = CoreConfig(b="y")
    merged = merge_configs(first, second)
    fields = merged.as_dict()
    assert "a" in fields and "b" in fields


def test_merge_configs_later_overrides_earlier() -> None:
    first = AgentConfig(model_name="gpt-3")
    second = CoreConfig(model_name="gpt-4")
    merged = merge_configs(first, second)
    assert merged.as_dict()["model_name"] == "gpt-4"


def test_merge_configs_preserves_unset_keys() -> None:
    first = AgentConfig(a=None, b="keep")
    second = CoreConfig(c="new")
    merged = merge_configs(first, second)
    fields = merged.as_dict()
    assert fields["a"] is None
    assert fields["b"] == "keep"
    assert fields["c"] == "new"


def test_merge_configs_returns_baseconfig_subtype() -> None:
    first = AgentConfig(x="1")
    second = CoreConfig(y="2")
    merged = merge_configs(first, second)
    assert isinstance(merged, BaseConfig)
    assert issubclass(type(merged), BaseConfig)


def test_merge_configs_three_way_override_order() -> None:
    c1 = AgentConfig(model_name="gpt-3.5")
    c2 = CoreConfig(model_name="gpt-4")
    c3 = AgentConfig(model_name="gpt-4o")
    merged = merge_configs(c1, c2, c3)
    assert merged.as_dict()["model_name"] == "gpt-4o"


def test_agent_from_parsed_config_constructs() -> None:
    parsed = parse_agent_config("workspace=/tmp,model_name=gpt-4")
    constructed = AgentConfig(**parsed.as_dict())
    assert isinstance(constructed, AgentConfig)
    assert constructed.as_dict() == parsed.as_dict()


def test_core_from_parsed_config_constructs() -> None:
    parsed = parse_core_config({"x": "v"})
    constructed = CoreConfig(**parsed.as_dict())
    assert isinstance(constructed, CoreConfig)
    assert constructed.as_dict() == parsed.as_dict()