from __future__ import annotations

from ..ports import GeneratorProvider, RetrieverProvider
from ..tool_runtime import ToolRuntime
from ..types import AgentResponse, GenerationRecord, SourceReference
from .runner import WorkflowContext, WorkflowStep


class StepRegistry:
    def __init__(self, steps: dict[str, WorkflowStep]):
        self.steps = steps

    @classmethod
    def builtins(cls) -> "StepRegistry":
        return cls(
            {
                "prepare_retrieval_query": prepare_retrieval_query,
                "run_retrieval": run_retrieval,
                "apply_policy": apply_policy,
                "build_policy_response": build_policy_response,
                "generate_answer": generate_answer,
                "build_response": build_response,
                "build_retrieval_debug_response": build_retrieval_debug_response,
                "build_refusal_response": build_refusal_response,
                "tool.select": select_tool,
                "tool.call": call_tool,
                "tool.validate_output": validate_tool_output,
                "tool.call_first": call_first_tool,
                "response.tool_result": build_tool_response,
            }
        )

    def has(self, step_id: str) -> bool:
        return step_id in self.steps

    def get(self, step_id: str) -> WorkflowStep:
        if step_id not in self.steps:
            raise KeyError(f"Unknown workflow step: {step_id}")
        return self.steps[step_id]


def build_retrieval_query(question: str, history: list[dict[str, object]] | None = None) -> str:
    user_turns: list[str] = []
    for item in history or []:
        if item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            user_turns.append(content.strip())
    recent_context = "\n".join(user_turns[-3:])
    return f"{recent_context}\n{question}".strip()


def prepare_retrieval_query(context: WorkflowContext) -> None:
    context.retrieval_query = build_retrieval_query(context.request.message, context.request.history)
    context.trace.add_step("prepare_retrieval_query", {"query": context.retrieval_query})


def run_retrieval(context: WorkflowContext) -> None:
    provider = context.retriever_provider or RetrieverProvider.from_settings(context.settings)
    context.retrieved_chunks = provider.retrieve(context.settings, context.retrieval_query)
    context.trace.add_step(
        "run_retrieval",
        {
            "provider": context.settings.retrieval_provider,
            "top_k": context.settings.top_k,
            "source_count": len(context.retrieved_chunks),
            "top_source": str(context.retrieved_chunks[0].get("source", "")) if context.retrieved_chunks else "",
        },
    )


def apply_policy(context: WorkflowContext) -> None:
    context.policy_decision = context.policy_guard.evaluate(
        message=context.request.message,
        intent_decision=context.intent_decision,
        retrieved_chunks=context.retrieved_chunks,
        require_sources=_requires_sources(context),
    )
    context.trace.add_step(
        "apply_policy",
        {
            "allowed": context.policy_decision.allowed,
            "policy_id": context.policy_decision.policy_id,
            "action": context.policy_decision.action,
            "reason": context.policy_decision.reason,
        },
    )


def _requires_sources(context: WorkflowContext) -> bool:
    intent_value = context.intent_decision.intent.requires_sources
    if intent_value is not None:
        return intent_value
    if context.workflow_requires_sources is not None:
        return context.workflow_requires_sources
    return context.intent_decision.intent.workflow == "rag_qa"


def build_policy_response(context: WorkflowContext) -> None:
    decision = context.policy_decision
    if decision is None or decision.allowed:
        return
    mode = "refusal" if decision.action == "refuse" else decision.action
    context.response = AgentResponse(
        answer=decision.message,
        mode=mode,
        intent=context.intent_decision.intent.id,
        workflow=context.intent_decision.intent.workflow,
        sources=[],
        trace=context.trace,
        generation=GenerationRecord(mode=mode, provider="policy", source_count=0),
    )


def generate_answer(context: WorkflowContext) -> None:
    provider = context.generator_provider or GeneratorProvider.from_settings(context.settings)
    generated = provider.generate(
        context.settings,
        context.request.message,
        context.retrieved_chunks,
        model_client=context.model_client,
        history=context.request.history,
    )
    generation = dict(generated.metadata)
    if "prompt_blocks" in generation and "input_blocks" not in generation:
        generation["input_blocks"] = generation.pop("prompt_blocks")
    generation.setdefault("mode", generated.mode)
    generation.setdefault("source_count", len(generated.sources))
    context.result = {
        "answer": generated.answer,
        "mode": generated.mode,
        "sources": generated.sources,
        "generation": generation,
    }
    detail = {"provider": context.settings.generation_provider, "mode": generated.mode}
    detail.update(generated.metadata)
    context.trace.add_step("generate_answer", detail)


def call_first_tool(context: WorkflowContext) -> None:
    select_tool(context)
    call_tool(context)


def select_tool(context: WorkflowContext) -> None:
    context.selected_tool_id = ToolRuntime(context.tool_provider).select(context)
    context.trace.add_step("tool.select", {"tool_id": context.selected_tool_id})


def call_tool(context: WorkflowContext) -> None:
    result, arguments = ToolRuntime(context.tool_provider).call(context.selected_tool_id, context)
    payload = {
        "tool_id": result.tool_id,
        "ok": result.ok,
        "output": result.output,
        "error": result.error,
        "arguments": arguments,
    }
    context.tool_results.append(payload)
    context.trace.add_step(
        "tool.call",
        {"tool_id": result.tool_id, "ok": result.ok, "error": result.error, "arguments": arguments},
    )


def validate_tool_output(context: WorkflowContext) -> None:
    result = context.tool_results[-1] if context.tool_results else None
    if result is None:
        context.trace.add_step(
            "tool.validate_output",
            {"tool_id": "", "ok": False, "error": "no tool result to validate"},
        )
        return
    tool_id = str(result.get("tool_id", ""))
    tool = context.tool_provider.tools.get(tool_id)
    schema = (tool.output_schema or tool.schema) if tool is not None else {}
    if not isinstance(schema, dict) or not schema:
        context.trace.add_step("tool.validate_output", {"tool_id": tool_id, "ok": True, "error": ""})
        return
    output = result.get("output", {})
    if not isinstance(output, dict):
        _mark_tool_validation_error(result, "tool output must be an object")
        context.trace.add_step("tool.validate_output", {"tool_id": tool_id, "ok": False, "error": result["error"]})
        return
    error = _tool_schema_error(output, schema)
    if error:
        _mark_tool_validation_error(result, error)
        context.trace.add_step("tool.validate_output", {"tool_id": tool_id, "ok": False, "error": result["error"]})
        return
    result["output"] = _sanitize_tool_output(output, schema)
    context.trace.add_step("tool.validate_output", {"tool_id": tool_id, "ok": True, "error": ""})


def _tool_schema_error(output: dict[str, object], schema: dict[str, object]) -> str:
    required = schema.get("required", [])
    if isinstance(required, list):
        for field in required:
            field_name = str(field)
            if field_name not in output:
                return f"missing required field {field_name}"
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return ""
    for field_name, spec in properties.items():
        if field_name not in output or not isinstance(spec, dict):
            continue
        expected_type = str(spec.get("type", "")).strip()
        if expected_type and not _matches_schema_type(output[field_name], expected_type):
            return f"field {field_name} must be {expected_type}"
    return ""


def _sanitize_tool_output(output: dict[str, object], schema: dict[str, object]) -> dict[str, object]:
    properties = schema.get("properties", {})
    if not isinstance(properties, dict) or not properties:
        return dict(output)
    return {str(field_name): output[str(field_name)] for field_name in properties if str(field_name) in output}


def _matches_schema_type(value: object, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    return True


def _mark_tool_validation_error(result: dict[str, object], error: str) -> None:
    result["ok"] = False
    result["output"] = {}
    result["error"] = f"Tool output failed schema validation: {error}"


def build_tool_response(context: WorkflowContext) -> None:
    result = context.tool_results[0] if context.tool_results else {}
    output = result.get("output", {})
    answer = ""
    if isinstance(output, dict) and output:
        answer = str(output.get("answer") or output.get("text") or output)
    if not answer:
        answer = str(result.get("error", "Tool did not produce output."))
    context.response = AgentResponse(
        answer=answer,
        mode="tool" if result.get("ok") else "tool_error",
        intent=context.intent_decision.intent.id,
        workflow=context.intent_decision.intent.workflow,
        sources=[],
        trace=context.trace,
        generation=GenerationRecord(
            mode="tool" if result.get("ok") else "tool_error",
            provider="tool",
            source_count=0,
        ),
        metadata={"tool_results": context.tool_results},
    )


def build_response(context: WorkflowContext) -> None:
    sources = [
        SourceReference.from_mapping(source)
        for source in context.result.get("sources", [])
        if isinstance(source, dict)
    ]
    context.response = AgentResponse(
        answer=str(context.result.get("answer", "")),
        mode=str(context.result.get("mode", "")),
        intent=context.intent_decision.intent.id,
        workflow=context.intent_decision.intent.workflow,
        sources=sources,
        trace=context.trace,
        generation=GenerationRecord.from_mapping(
            context.result.get("generation", {}) if isinstance(context.result.get("generation"), dict) else {}
        ),
    )


def build_retrieval_debug_response(context: WorkflowContext) -> None:
    lines = ["Local retrieval debug results:"]
    if not context.retrieved_chunks:
        lines.append("No matching chunks.")
    for index, chunk in enumerate(context.retrieved_chunks, start=1):
        lines.append(
            f"{index}. {chunk.get('source')} ({chunk.get('chunk_id')}, score={chunk.get('score', 0)})"
        )
    context.response = AgentResponse(
        answer="\n".join(lines),
        mode="retrieval_debug",
        intent=context.intent_decision.intent.id,
        workflow=context.intent_decision.intent.workflow,
        sources=[SourceReference.from_mapping(chunk) for chunk in context.retrieved_chunks],
        trace=context.trace,
        generation=GenerationRecord(
            mode="not_called",
            provider="retrieval_debug",
            source_count=len(context.retrieved_chunks),
        ),
    )


def build_refusal_response(context: WorkflowContext) -> None:
    answer = context.policy_decision.message if context.policy_decision else ""
    if not answer:
        answer = "This request is outside the current agent boundary. Ask for allowed steps, sources, methods, or review checks instead."
    context.trace.add_step("build_refusal", {"policy": context.intent_decision.intent.policy})
    context.response = AgentResponse(
        answer=answer,
        mode="refusal",
        intent=context.intent_decision.intent.id,
        workflow=context.intent_decision.intent.workflow,
        sources=[],
        trace=context.trace,
        generation=GenerationRecord(mode="refusal", provider="policy", source_count=0),
    )
