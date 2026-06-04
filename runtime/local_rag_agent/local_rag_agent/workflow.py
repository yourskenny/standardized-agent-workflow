from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .agent import answer_question, build_extractive_answer
from .config import Settings
from .index_store import read_index
from .intent import IntentDecision
from .llm import OpenAICompatibleClient
from .policy import PolicyDecision, PolicyGuard
from .retrieval import rank_chunks
from .tools import ToolProvider
from .types import AgentRequest, AgentResponse, AgentTrace, SourceReference

WorkflowStep = Callable[["WorkflowContext"], None]


@dataclass
class WorkflowContext:
    settings: Settings
    request: AgentRequest
    intent_decision: IntentDecision
    trace: AgentTrace
    model_client: object | None = None
    policy_guard: PolicyGuard = field(default_factory=PolicyGuard.builtins)
    tool_provider: ToolProvider = field(default_factory=ToolProvider.disabled)
    retrieval_query: str = ""
    retrieved_chunks: list[dict[str, object]] = field(default_factory=list)
    policy_decision: PolicyDecision | None = None
    result: dict[str, object] = field(default_factory=dict)
    response: AgentResponse | None = None


class WorkflowPipeline:
    def __init__(self, workflow_id: str, steps: list[WorkflowStep]):
        self.workflow_id = workflow_id
        self.steps = steps

    def run(self, context: WorkflowContext) -> AgentResponse:
        context.trace.add_step("start_workflow", {"workflow": self.workflow_id})
        for step in self.steps:
            step(context)
            if context.response is not None:
                break
        if context.response is None:
            raise RuntimeError(f"Workflow did not produce a response: {self.workflow_id}")
        return context.response


class WorkflowRegistry:
    def __init__(self, workflows: dict[str, WorkflowPipeline]):
        self.workflows = workflows

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
    payload = read_index(context.settings)
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError(f"Invalid index format: {context.settings.index_path}")
    context.retrieved_chunks = rank_chunks(context.retrieval_query, chunks, context.settings.top_k)
    context.trace.add_step(
        "run_retrieval",
        {
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
    client = context.model_client if context.model_client is not None else OpenAICompatibleClient.from_env()
    context.result = answer_question(
        context.settings,
        context.request.message,
        context.retrieved_chunks,
        client,
        history=context.request.history,
    )
    context.trace.add_step("generate_answer", {"mode": str(context.result.get("mode", ""))})


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
    answer = build_extractive_answer(context.request.message, [])
    context.trace.add_step("build_refusal", {"policy": context.intent_decision.intent.policy})
    context.response = AgentResponse(
        answer=answer,
        mode="refusal",
        intent=context.intent_decision.intent.id,
        workflow=context.intent_decision.intent.workflow,
        sources=[],
        trace=context.trace,
    )
