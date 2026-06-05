from __future__ import annotations

from pathlib import Path
from typing import Any

from ..ports import ToolPort
from ..tools import ToolDefinition, ToolResult, load_tools


class ConfiguredToolProvider(ToolPort):
    def __init__(self, tools: list[ToolDefinition]):
        self.tools = {tool.id: tool for tool in tools}

    @classmethod
    def disabled(cls) -> "ConfiguredToolProvider":
        return cls([])

    @classmethod
    def from_config(cls, path: Path | None) -> "ConfiguredToolProvider":
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
        adapter = tool.adapter or tool.provider
        if adapter == "mock":
            output = tool.metadata.get("mock_output", {})
            if not isinstance(output, dict):
                output = {}
            return ToolResult(tool_id=tool_id, ok=True, output=_bounded_output(output, tool.max_output_bytes))
        return ToolResult(tool_id=tool_id, ok=False, error=f"Tool provider is not implemented: {tool_id}")


ToolProvider = ConfiguredToolProvider


def _bounded_output(output: dict[str, Any], max_output_bytes: int) -> dict[str, Any]:
    text = str(output)
    if len(text.encode("utf-8")) <= max_output_bytes:
        return dict(output)
    return {"truncated": True, "text": text[:max_output_bytes]}
