from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from .components import ComponentRegistry
from .config import Settings, load_settings
from .intent import IntentRouter, load_intents
from .policy import PolicyGuard
from .ports import GeneratorProvider, RetrieverProvider
from .tools import ToolProvider
from .types import AgentRequest, AgentResponse, AgentTrace
from .workflow import WorkflowContext, WorkflowRegistry, build_retrieval_query as build_workflow_retrieval_query

DEFAULT_INTENT = "knowledge_qa"
DEFAULT_WORKFLOW = "rag_qa"
LOGGER = logging.getLogger(__name__)


class AgentRuntime:
    def __init__(
        self,
        settings: Settings,
        model_client: object | None = None,
        components: ComponentRegistry | None = None,
        run_store: object | None = None,
    ):
        self.settings = settings
        self.model_client = model_client
        self.run_store = run_store
        self.components = components or ComponentRegistry.from_settings(settings)
        intents = load_intents(settings.intent_config_path)
        self.intent_router = IntentRouter(
            intents,
            default_intent=settings.default_intent,
            default_workflow=settings.default_workflow,
        )
        self.workflow_registry = WorkflowRegistry.from_config(
            settings.workflow_config_path,
            step_registry=self.components.step_registry(),
        )
        self.policy_guard = self.components.build_policy_guard(settings)
        self.tool_provider = self.components.build_tool_provider(settings)
        self.retriever_provider = self.components.build_retriever_provider(settings)
        self.generator_provider = self.components.build_generator_provider(settings)
        self.config_versions = _merge_config_versions(
            settings.config_schema_versions or {},
            _first_schema_version("intent", intents),
            self.workflow_registry.config_versions,
            _first_schema_version("policy", self.policy_guard.policies.values()),
            _first_schema_version("tool", self.tool_provider.tools.values()),
        )

    @classmethod
    def from_project(
        cls,
        project_root: Path,
        config_path: Path,
        model_client: object | None = None,
        components: ComponentRegistry | None = None,
        run_store: object | None = None,
    ) -> "AgentRuntime":
        return cls(
            load_settings(project_root, config_path),
            model_client=model_client,
            components=components,
            run_store=run_store,
        )

    def run(self, request: AgentRequest) -> AgentResponse:
        intent_decision = self.intent_router.route(request.message)
        request_id = _request_id(request)
        run_id = _run_id(request)
        trace = AgentTrace(
            intent=intent_decision.intent.id,
            workflow=intent_decision.intent.workflow,
            request_id=request_id,
            run_id=run_id,
            config_versions=self.config_versions,
        )
        self._create_run(run_id, request_id, request, intent_decision.intent.id, intent_decision.intent.workflow)
        trace.add_step(
            "route_intent",
            {
                "source": intent_decision.source,
                "confidence": intent_decision.confidence,
                "matched_terms": intent_decision.matched_terms,
            },
        )
        workflow = self.workflow_registry.get(
            intent_decision.intent.workflow,
            allow_fallback=self.settings.allow_workflow_fallback,
        )
        context = WorkflowContext(
            settings=self.settings,
            request=request,
            intent_decision=intent_decision,
            trace=trace,
            model_client=self.model_client,
            policy_guard=self.policy_guard,
            tool_provider=self.tool_provider,
            retriever_provider=self.retriever_provider,
            generator_provider=self.generator_provider,
            run_store=self.run_store,
            run_id=run_id,
        )
        response = workflow.run(context)
        event = _runtime_trace_event(request_id, response)
        self.components.emit_trace(event)
        LOGGER.info("runtime_trace %s", json.dumps(event, ensure_ascii=False, sort_keys=True))
        return response

    def _create_run(
        self,
        run_id: str,
        request_id: str,
        request: AgentRequest,
        intent: str,
        workflow: str,
    ) -> None:
        if self.run_store is None:
            return
        create_run = getattr(self.run_store, "create_run", None)
        if callable(create_run):
            create_run(
                run_id=run_id,
                thread_id=str(request.metadata.get("thread_id", "")),
                intent=intent,
                workflow=workflow,
                status="running",
                metadata={"request_id": request_id},
            )

    @staticmethod
    def build_retrieval_query(question: str, history: list[dict[str, object]] | None = None) -> str:
        return build_workflow_retrieval_query(question, history)


def _first_schema_version(config_name: str, definitions: object) -> dict[str, str]:
    for definition in definitions:
        schema_version = str(getattr(definition, "schema_version", "") or "")
        if schema_version:
            return {config_name: schema_version}
    return {}


def _merge_config_versions(*items: dict[str, str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for item in items:
        versions.update(item)
    return versions


def _request_id(request: AgentRequest) -> str:
    raw_request_id = request.metadata.get("request_id")
    if raw_request_id is not None and str(raw_request_id).strip():
        return str(raw_request_id)
    return uuid.uuid4().hex


def _run_id(request: AgentRequest) -> str:
    raw_run_id = request.metadata.get("run_id")
    if raw_run_id is not None and str(raw_run_id).strip():
        return str(raw_run_id)
    return uuid.uuid4().hex


def _runtime_trace_event(request_id: str, response: AgentResponse) -> dict[str, object]:
    trace = response.trace.to_dict() if response.trace else {}
    return {
        "event": "runtime_trace",
        "request_id": request_id,
        "run_id": trace.get("run_id", ""),
        "intent": response.intent,
        "workflow": response.workflow,
        "mode": response.mode,
        "config_versions": trace.get("config_versions", {}),
        "steps": trace.get("steps", []),
    }
