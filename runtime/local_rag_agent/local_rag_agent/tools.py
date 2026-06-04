from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    id: str
    description: str = ""
    enabled: bool = False
    provider: str = "disabled"
    metadata: dict[str, Any] = field(default_factory=dict)


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
    records = data.get("tools", [])
    if not isinstance(records, list):
        raise ValueError(f"Invalid tool config: {path}")
    return [_tool_from_record(record, path) for record in records if isinstance(record, dict)]


class ToolProvider:
    def __init__(self, tools: list[ToolDefinition]):
        self.tools = {tool.id: tool for tool in tools}

    @classmethod
    def disabled(cls) -> "ToolProvider":
        return cls([])

    @classmethod
    def from_config(cls, path: Path | None) -> "ToolProvider":
        return cls(load_tools(path))

    def call(self, tool_id: str, arguments: dict[str, object]) -> ToolResult:
        tool = self.tools.get(tool_id)
        if tool is None:
            return ToolResult(tool_id=tool_id, ok=False, error=f"Tool is disabled or unavailable: {tool_id}")
        if not tool.enabled:
            return ToolResult(tool_id=tool_id, ok=False, error=f"Tool is disabled: {tool_id}")
        return ToolResult(tool_id=tool_id, ok=False, error=f"Tool provider is not implemented: {tool_id}")


def _tool_from_record(record: dict[str, object], path: Path) -> ToolDefinition:
    tool_id = str(record.get("id", "")).strip()
    if not tool_id:
        raise ValueError(f"Tool missing id in {path}")
    enabled = record.get("enabled", False)
    return ToolDefinition(
        id=tool_id,
        description=str(record.get("description", "")),
        enabled=enabled if isinstance(enabled, bool) else False,
        provider=str(record.get("provider", "disabled")),
        metadata={key: value for key, value in record.items() if key not in {"id", "description", "enabled", "provider"}},
    )
