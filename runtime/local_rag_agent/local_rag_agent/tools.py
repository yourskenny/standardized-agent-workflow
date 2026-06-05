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
    allowed_intents: list[str] = field(default_factory=list)
    risk_level: str = "low"
    timeout_seconds: int = 10
    max_output_bytes: int = 20000
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


class ToolProvider:
    def __init__(self, tools: list[ToolDefinition]):
        self.tools = {tool.id: tool for tool in tools}

    @classmethod
    def disabled(cls) -> "ToolProvider":
        return cls([])

    @classmethod
    def from_config(cls, path: Path | None) -> "ToolProvider":
        return cls(load_tools(path))

    def call(
        self,
        tool_id: str,
        arguments: dict[str, object],
        intent_id: str = "",
    ) -> ToolResult:
        tool = self.tools.get(tool_id)
        if tool is None:
            return ToolResult(tool_id=tool_id, ok=False, error=f"Tool is disabled or unavailable: {tool_id}")
        if not tool.enabled:
            return ToolResult(tool_id=tool_id, ok=False, error=f"Tool is disabled: {tool_id}")
        if tool.allowed_intents and intent_id not in tool.allowed_intents:
            return ToolResult(tool_id=tool_id, ok=False, error=f"Tool is not allowed for intent: {intent_id}")
        if tool.provider == "mock":
            output = tool.metadata.get("mock_output", {})
            if not isinstance(output, dict):
                output = {}
            return ToolResult(tool_id=tool_id, ok=True, output=_bounded_output(output, tool.max_output_bytes))
        return ToolResult(tool_id=tool_id, ok=False, error=f"Tool provider is not implemented: {tool_id}")


def _tool_from_record(record: dict[str, object], path: Path, schema_version: str) -> ToolDefinition:
    known_fields = {
        "id",
        "description",
        "enabled",
        "provider",
        "allowed_intents",
        "risk_level",
        "timeout_seconds",
        "max_output_bytes",
        "schema",
        "mock_output",
    }
    warn_unknown_fields(record, known_fields, path, "tools")
    tool_id = str(record.get("id", "")).strip()
    if not tool_id:
        raise ValueError(f"Tool missing id in {path}")
    enabled = record.get("enabled", False)
    schema = record.get("schema", {})
    return ToolDefinition(
        id=tool_id,
        description=str(record.get("description", "")),
        enabled=enabled if isinstance(enabled, bool) else False,
        provider=str(record.get("provider", "disabled")),
        allowed_intents=_string_list(record.get("allowed_intents", [])),
        risk_level=str(record.get("risk_level", "low")),
        timeout_seconds=int(record.get("timeout_seconds", 10)),
        max_output_bytes=int(record.get("max_output_bytes", 20000)),
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
                "allowed_intents",
                "risk_level",
                "timeout_seconds",
                "max_output_bytes",
                "schema",
            }
        },
        schema_version=schema_version,
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _bounded_output(output: dict[str, Any], max_output_bytes: int) -> dict[str, Any]:
    text = str(output)
    if len(text.encode("utf-8")) <= max_output_bytes:
        return dict(output)
    return {"truncated": True, "text": text[:max_output_bytes]}
