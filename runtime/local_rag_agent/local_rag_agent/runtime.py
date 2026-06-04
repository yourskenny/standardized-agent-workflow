from __future__ import annotations

from pathlib import Path

from .config import Settings, load_settings
from .intent import IntentRouter, load_intents
from .policy import PolicyGuard
from .tools import ToolProvider
from .types import AgentRequest, AgentResponse, AgentTrace
from .workflow import WorkflowContext, WorkflowRegistry, build_retrieval_query as build_workflow_retrieval_query

DEFAULT_INTENT = "knowledge_qa"
DEFAULT_WORKFLOW = "rag_qa"


class AgentRuntime:
    def __init__(self, settings: Settings, model_client: object | None = None):
        self.settings = settings
        self.model_client = model_client
        self.intent_router = IntentRouter(
            load_intents(settings.intent_config_path),
            default_intent=settings.default_intent,
            default_workflow=settings.default_workflow,
        )
        self.workflow_registry = WorkflowRegistry.builtins()
        self.policy_guard = PolicyGuard.from_config(settings.policy_config_path)
        self.tool_provider = ToolProvider.from_config(settings.tool_config_path)

    @classmethod
    def from_project(
        cls,
        project_root: Path,
        config_path: Path,
        model_client: object | None = None,
    ) -> "AgentRuntime":
        return cls(load_settings(project_root, config_path), model_client=model_client)

    def run(self, request: AgentRequest) -> AgentResponse:
        intent_decision = self.intent_router.route(request.message)
        trace = AgentTrace(intent=intent_decision.intent.id, workflow=intent_decision.intent.workflow)
        trace.add_step(
            "route_intent",
            {
                "source": intent_decision.source,
                "confidence": intent_decision.confidence,
                "matched_terms": intent_decision.matched_terms,
            },
        )
        workflow = self.workflow_registry.get(intent_decision.intent.workflow)
        context = WorkflowContext(
            settings=self.settings,
            request=request,
            intent_decision=intent_decision,
            trace=trace,
            model_client=self.model_client,
            policy_guard=self.policy_guard,
            tool_provider=self.tool_provider,
        )
        return workflow.run(context)

    @staticmethod
    def build_retrieval_query(question: str, history: list[dict[str, object]] | None = None) -> str:
        return build_workflow_retrieval_query(question, history)
