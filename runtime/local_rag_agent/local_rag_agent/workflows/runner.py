from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..config import Settings
from ..intent import IntentDecision
from ..policy import PolicyDecision, PolicyGuard
from ..ports import GeneratorProvider, RetrieverProvider
from ..tools import ToolProvider
from ..types import AgentRequest, AgentResponse, AgentTrace

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
    retriever_provider: RetrieverProvider | None = None
    generator_provider: GeneratorProvider | None = None
    retrieval_query: str = ""
    retrieved_chunks: list[dict[str, object]] = field(default_factory=list)
    workflow_requires_sources: bool | None = None
    policy_decision: PolicyDecision | None = None
    selected_tool_id: str = ""
    tool_results: list[dict[str, object]] = field(default_factory=list)
    result: dict[str, object] = field(default_factory=dict)
    response: AgentResponse | None = None


class WorkflowPipeline:
    def __init__(
        self,
        workflow_id: str,
        steps: list[WorkflowStep],
        requires_sources: bool | None = None,
    ):
        self.workflow_id = workflow_id
        self.steps = steps
        self.requires_sources = requires_sources

    def run(self, context: WorkflowContext) -> AgentResponse:
        context.workflow_requires_sources = self.requires_sources
        context.trace.add_step(
            "start_workflow",
            {
                "workflow": self.workflow_id,
                "steps": [_step_name(step) for step in self.steps],
                "requires_sources": self.requires_sources,
            },
        )
        for step in self.steps:
            step(context)
            if context.response is not None:
                break
        if context.response is None:
            raise RuntimeError(f"Workflow did not produce a response: {self.workflow_id}")
        return context.response


def _step_name(step: WorkflowStep) -> str:
    return getattr(step, "__name__", str(step))
