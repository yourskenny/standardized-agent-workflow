from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import Settings
from .intent import IntentDecision
from .policy import PolicyDecision, PolicyGuard
from .ports import GeneratorProvider, RetrieverProvider
from .tools import ToolProvider
from .types import AgentRequest, AgentResponse, AgentTrace, SourceReference

WorkflowStep = Callable[["WorkflowContext"], None]
SUPPORTED_WORKFLOW_SCHEMA_VERSIONS = {"workflow.v1"}


@dataclass(frozen=True)
class WorkflowDefinition:
    id: str
    steps: list[str]
    description: str = ""
    schema_version: str = ""


@dataclass
class WorkflowContext:
    settings: Settings
    request: AgentRequest
    intent_decision: IntentDecision
    trace: AgentTrace
    model_client: object | None = None
    policy_guard: PolicyGuard = field(default_factory=PolicyGuard.builtins)
    tool_provider: ToolProvider = field(default_factory=ToolProvider.disabled)
    retriever_provider: RetrieverProvider | None = None
    generator_provider: GeneratorProvider | None = None
    retrieval_query: str = ""
    retrieved_chunks: list[dict[str, object]] = field(default_factory=list)
    policy_decision: PolicyDecision | None = None
    tool_results: list[dict[str, object]] = field(default_factory=list)
    result: dict[str, object] = field(default_factory=dict)
    response: AgentResponse | None = None


class WorkflowPipeline:
    def __init__(self, workflow_id: str, steps: list[WorkflowStep]):
        self.workflow_id = workflow_id
        self.steps = steps

    def run(self, context: WorkflowContext) -> AgentResponse:
        context.trace.add_step(
            "start_workflow",
            {"workflow": self.workflow_id, "steps": [_step_name(step) for step in self.steps]},
        )
        for step in self.steps:
            step(context)
            if context.response is not None:
                break
        if context.response is None:
            raise RuntimeError(f"Workflow did not produce a response: {self.workflow_id}")
        return context.response


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


class WorkflowRegistry:
    def __init__(
        self,
        workflows: dict[str, WorkflowPipeline],
        config_versions: dict[str, str] | None = None,
    ):
        self.workflows = workflows
        self.config_versions = config_versions or {}

    @classmethod
    def builtins(cls) -> "WorkflowRegistry":
        return cls(
            {
                "rag_qa": WorkflowPipeline(
                    "rag_qa",
                    [
                        prepare_retrieval_query,
                        run_retrieval,
                        apply_policy,
                        build_policy_response,
                        generate_answer,
                        build_response,
                    ],
                ),
                "retrieval_debug": WorkflowPipeline(
                    "retrieval_debug",
                    [prepare_retrieval_query, run_retrieval, build_retrieval_debug_response],
                ),
                "refusal_with_guidance": WorkflowPipeline(
                    "refusal_with_guidance",
                    [apply_policy, build_policy_response, build_refusal_response],
                ),
            }
        )

    @classmethod
    def from_config(
        cls,
        path: Path | None,
        step_registry: StepRegistry | None = None,
    ) -> "WorkflowRegistry":
        if path is None or not path.exists():
            return cls.builtins()
        steps = step_registry or StepRegistry.builtins()
        pipelines: dict[str, WorkflowPipeline] = {}
        definitions = load_workflows(path)
        for definition in definitions:
            workflow_steps: list[WorkflowStep] = []
            for step_id in definition.steps:
                if not steps.has(step_id):
                    raise ValueError(f"Unknown workflow step in {path}: {step_id}")
                workflow_steps.append(steps.get(step_id))
            pipelines[definition.id] = WorkflowPipeline(definition.id, workflow_steps)
        builtins = cls.builtins().workflows
        builtins.update(pipelines)
        schema_version = next((definition.schema_version for definition in definitions if definition.schema_version), "")
        return cls(builtins, {"workflow": schema_version} if schema_version else {})

    def has(self, workflow_id: str) -> bool:
        return workflow_id in self.workflows

    def get(self, workflow_id: str) -> WorkflowPipeline:
        if workflow_id not in self.workflows:
            return self.workflows["rag_qa"]
        return self.workflows[workflow_id]


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


def load_workflows(path: Path | None) -> list[WorkflowDefinition]:
    if path is None or not path.exists():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    schema_version = str(data.get("schema_version", "")).strip()
    if schema_version and schema_version not in SUPPORTED_WORKFLOW_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported schema_version in {path}: {schema_version}")
    records = data.get("workflows", [])
    if not isinstance(records, list):
        raise ValueError(f"Invalid workflow config: {path}")
    return [_workflow_from_record(record, path, schema_version) for record in records if isinstance(record, dict)]


def _workflow_from_record(record: dict[str, object], path: Path, schema_version: str) -> WorkflowDefinition:
    workflow_id = str(record.get("id", "")).strip()
    if not workflow_id:
        raise ValueError(f"Workflow missing id in {path}")
    steps = _string_list(record.get("steps", []))
    if not steps:
        raise ValueError(f"Workflow missing steps in {path}: {workflow_id}")
    return WorkflowDefinition(
        id=workflow_id,
        steps=steps,
        description=str(record.get("description", "")),
        schema_version=schema_version,
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _step_name(step: WorkflowStep) -> str:
    return getattr(step, "__name__", str(step))


def apply_policy(context: WorkflowContext) -> None:
    context.policy_decision = context.policy_guard.evaluate(
        message=context.request.message,
        intent_decision=context.intent_decision,
        retrieved_chunks=context.retrieved_chunks,
        require_sources=context.intent_decision.intent.workflow == "rag_qa",
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
    context.result = {"answer": generated.answer, "mode": generated.mode, "sources": generated.sources}
    context.trace.add_step(
        "generate_answer",
        {"provider": context.settings.generation_provider, "mode": generated.mode},
    )


def call_first_tool(context: WorkflowContext) -> None:
    tool_ids = sorted(context.tool_provider.tools)
    if not tool_ids:
        result = context.tool_provider.call("", {"query": context.request.message}, intent_id=context.intent_decision.intent.id)
    else:
        result = context.tool_provider.call(
            tool_ids[0],
            {"query": context.request.message},
            intent_id=context.intent_decision.intent.id,
        )
    payload = {
        "tool_id": result.tool_id,
        "ok": result.ok,
        "output": result.output,
        "error": result.error,
    }
    context.tool_results.append(payload)
    context.trace.add_step(
        "tool.call",
        {"tool_id": result.tool_id, "ok": result.ok, "error": result.error},
    )


def build_tool_response(context: WorkflowContext) -> None:
    result = context.tool_results[0] if context.tool_results else {}
    output = result.get("output", {})
    answer = ""
    if isinstance(output, dict):
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
    )
