from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .tools import ToolDefinition, ToolResult
from .workflows.runner import WorkflowContext


@dataclass(frozen=True)
class ToolInvocation:
    tool_id: str
    arguments: dict[str, object]
    intent_id: str


class ToolRuntime:
    def __init__(self, provider: object, audit_sink: object | None = None):
        self.provider = provider
        self.audit_sink = audit_sink

    def select(self, context: WorkflowContext) -> str:
        tools = getattr(self.provider, "tools", {})
        return sorted(tools)[0] if tools else ""

    def prepare_input(self, definition: ToolDefinition | None, context: WorkflowContext) -> dict[str, object]:
        mapping = definition.input_mapping if definition is not None else {}
        if not mapping:
            return {"query": context.request.message}
        return {
            str(name): self._resolve_mapping(str(selector), context)
            for name, selector in mapping.items()
        }

    def call(self, tool_id: str, context: WorkflowContext) -> tuple[ToolResult, dict[str, object]]:
        definition = getattr(self.provider, "tools", {}).get(tool_id)
        arguments = self.prepare_input(definition, context)
        authorization_error = self.authorize(definition, tool_id, context)
        if authorization_error:
            if definition is not None and definition.requires_approval:
                self._audit(
                    "tool.approval_required",
                    definition,
                    context,
                    arguments,
                )
            return ToolResult(tool_id=tool_id, ok=False, error=authorization_error), arguments
        self._audit("tool.call", definition, context, arguments)
        result = self.provider.call(
            tool_id,
            arguments,
            intent_id=context.intent_decision.intent.id,
        )
        return result, arguments

    def authorize(
        self,
        definition: ToolDefinition | None,
        tool_id: str,
        context: WorkflowContext,
    ) -> str:
        if definition is None:
            return f"Tool is disabled or unavailable: {tool_id}"
        if not definition.enabled:
            return f"Tool is disabled: {tool_id}"
        if definition.allowed_intents and context.intent_decision.intent.id not in definition.allowed_intents:
            return f"Tool is not allowed for intent: {context.intent_decision.intent.id}"
        if definition.requires_approval:
            return f"Tool requires approval: {tool_id}"
        return ""

    def _audit(
        self,
        event: str,
        definition: ToolDefinition | None,
        context: WorkflowContext,
        arguments: dict[str, object],
    ) -> None:
        if self.audit_sink is None or definition is None:
            return
        payload = {
            "event": event,
            "tool_id": definition.id,
            "intent": context.intent_decision.intent.id,
            "arguments": arguments,
        }
        if callable(self.audit_sink):
            self.audit_sink(payload)

    def _resolve_mapping(self, selector: str, context: WorkflowContext) -> object:
        if selector == "$message":
            return context.request.message
        if selector.startswith("$metadata."):
            return self._lookup_path(context.request.metadata, selector.removeprefix("$metadata."))
        if selector.startswith("$state."):
            return self._lookup_path(context.result, selector.removeprefix("$state."))
        return selector

    @staticmethod
    def _lookup_path(payload: dict[str, Any], path: str) -> object:
        current: object = payload
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return ""
            current = current[part]
        return current
