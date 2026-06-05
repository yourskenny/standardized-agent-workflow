from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import read_schema_version, warn_unknown_fields


@dataclass(frozen=True)
class ToolDefinition:
    id: str
    description: str = ""
    enabled: bool = False
    provider: str = "disabled"
    adapter: str = "disabled"
    allowed_intents: list[str] = field(default_factory=list)
    risk_level: str = "low"
    timeout_seconds: int = 10
    max_output_bytes: int = 20000
    requires_approval: bool = False
    input_mapping: dict[str, Any] = field(default_factory=dict)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    schema: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = ""


@dataclass(frozen=True)
class ToolResult:
    tool_id: str
    ok: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: str = ""


def load_tools(path: Path | None) -> list[ToolDefinition]:
    if path is None or not path.exists():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    schema_version = read_schema_version(data, "tool", path)
    warn_unknown_fields(data, {"schema_version", "tools"}, path)
    records = data.get("tools", [])
    if not isinstance(records, list):
        raise ValueError(f"Invalid tool config: {path}")
    return [_tool_from_record(record, path, schema_version) for record in records if isinstance(record, dict)]


def _tool_from_record(record: dict[str, object], path: Path, schema_version: str) -> ToolDefinition:
    known_fields = {
        "id",
        "description",
        "enabled",
        "provider",
        "adapter",
        "allowed_intents",
        "risk_level",
        "timeout_seconds",
        "max_output_bytes",
        "requires_approval",
        "input_mapping",
        "input_schema",
        "output_schema",
        "schema",
        "mock_output",
    }
    warn_unknown_fields(record, known_fields, path, "tools")
    tool_id = str(record.get("id", "")).strip()
    if not tool_id:
        raise ValueError(f"Tool missing id in {path}")
    enabled = record.get("enabled", False)
    adapter = str(record.get("adapter", record.get("provider", "disabled"))).strip() or "disabled"
    provider = str(record.get("provider", adapter)).strip() or adapter
    schema = record.get("schema", {})
    input_mapping = record.get("input_mapping", {})
    input_schema = record.get("input_schema", {})
    output_schema = record.get("output_schema", {})
    requires_approval = record.get("requires_approval", False)
    return ToolDefinition(
        id=tool_id,
        description=str(record.get("description", "")),
        enabled=enabled if isinstance(enabled, bool) else False,
        provider=provider,
        adapter=adapter,
        allowed_intents=_string_list(record.get("allowed_intents", [])),
        risk_level=str(record.get("risk_level", "low")),
        timeout_seconds=int(record.get("timeout_seconds", 10)),
        max_output_bytes=int(record.get("max_output_bytes", 20000)),
        requires_approval=requires_approval if isinstance(requires_approval, bool) else False,
        input_mapping=input_mapping if isinstance(input_mapping, dict) else {},
        input_schema=input_schema if isinstance(input_schema, dict) else {},
        output_schema=output_schema if isinstance(output_schema, dict) else {},
        schema=schema if isinstance(schema, dict) else {},
        metadata={
            key: value
            for key, value in record.items()
            if key
            not in {
                "id",
                "description",
                "enabled",
                "provider",
                "adapter",
                "allowed_intents",
                "risk_level",
                "timeout_seconds",
                "max_output_bytes",
                "requires_approval",
                "input_mapping",
                "input_schema",
                "output_schema",
                "schema",
            }
        },
        schema_version=schema_version,
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def __getattr__(name: str) -> object:
    if name in {"ToolProvider", "ConfiguredToolProvider"}:
        from .adapters.tools import ConfiguredToolProvider

        return ConfiguredToolProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ConfiguredToolProvider",
    "ToolDefinition",
    "ToolProvider",
    "ToolResult",
    "load_tools",
]
