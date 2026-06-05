from __future__ import annotations

from pathlib import Path

from .definitions import load_workflows
from .runner import WorkflowPipeline, WorkflowStep
from .steps import (
    StepRegistry,
    apply_policy,
    build_policy_response,
    build_refusal_response,
    build_response,
    build_retrieval_debug_response,
    generate_answer,
    prepare_retrieval_query,
    run_retrieval,
)


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
                    requires_sources=True,
                ),
                "retrieval_debug": WorkflowPipeline(
                    "retrieval_debug",
                    [prepare_retrieval_query, run_retrieval, build_retrieval_debug_response],
                    requires_sources=False,
                ),
                "refusal_with_guidance": WorkflowPipeline(
                    "refusal_with_guidance",
                    [apply_policy, build_policy_response, build_refusal_response],
                    requires_sources=False,
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
            pipelines[definition.id] = WorkflowPipeline(
                definition.id,
                workflow_steps,
                requires_sources=definition.requires_sources,
            )
        builtins = cls.builtins().workflows
        builtins.update(pipelines)
        schema_version = next((definition.schema_version for definition in definitions if definition.schema_version), "")
        return cls(builtins, {"workflow": schema_version} if schema_version else {})

    def has(self, workflow_id: str) -> bool:
        return workflow_id in self.workflows

    def get(self, workflow_id: str, allow_fallback: bool = False) -> WorkflowPipeline:
        if workflow_id not in self.workflows:
            if allow_fallback:
                return self.workflows["rag_qa"]
            raise KeyError(f"Unknown workflow: {workflow_id}")
        return self.workflows[workflow_id]
