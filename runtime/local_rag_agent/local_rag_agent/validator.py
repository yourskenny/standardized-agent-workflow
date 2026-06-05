from __future__ import annotations

import tomllib
import warnings as warning_module
from dataclasses import dataclass, field
from pathlib import Path

from .components import ComponentRegistry
from .config import Settings, load_settings
from .index_store import read_index
from .intent import IntentRouter, load_intent_tests, load_intents
from .manifest import expand_manifest_entries
from .policy import PolicyGuard, load_policies
from .schema import warn_unknown_fields
from .tools import load_tools
from .workflow import WorkflowRegistry, load_workflows

KNOWN_RETRIEVER_PROVIDERS = {"lexical"}
KNOWN_GENERATOR_PROVIDERS = {"extractive", "openai_compatible"}
KNOWN_GENERATION_FALLBACKS = {"extractive"}
KNOWN_TOOL_PROVIDERS = {"disabled", "mock"}
KNOWN_GRAPH_CONDITIONS = {"default", "policy.blocked", "intent.requires_tool"}


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    path: Path | str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "path": str(self.path).replace("\\", "/"),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ValidationResult:
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


def validate_project_contract(settings: Settings) -> ValidationResult:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    intents = load_intents(settings.intent_config_path)
    workflow_definitions = load_workflows(settings.workflow_config_path)
    component_registry = ComponentRegistry.from_settings(settings)
    errors.extend(_validate_workflow_definitions(settings, workflow_definitions, component_registry))
    errors.extend(_validate_runtime_providers(settings))
    errors.extend(_validate_tool_providers(settings))
    errors.extend(_validate_manifest_and_index(settings))
    if errors:
        return ValidationResult(errors=errors, warnings=warnings)
    workflow_registry = WorkflowRegistry.from_config(
        settings.workflow_config_path,
        step_registry=component_registry.step_registry(),
    )
    policy_guard = PolicyGuard.from_config(settings.policy_config_path)

    for intent in intents:
        if not workflow_registry.has(intent.workflow):
            errors.append(
                ValidationIssue(
                    code="UNKNOWN_WORKFLOW",
                    path=settings.intent_config_path or "",
                    detail=f"intent {intent.id} references unknown workflow {intent.workflow}",
                )
            )
        if intent.policy and intent.policy not in policy_guard.policies:
            errors.append(
                ValidationIssue(
                    code="UNKNOWN_POLICY",
                    path=settings.intent_config_path or "",
                    detail=f"intent {intent.id} references unknown policy {intent.policy}",
                )
            )
    errors.extend(_validate_intent_tests(settings, intents))

    return ValidationResult(errors=errors, warnings=warnings)


def _validate_intent_tests(settings: Settings, intents: object) -> list[ValidationIssue]:
    errors: list[ValidationIssue] = []
    tests = load_intent_tests(settings.intent_config_path)
    if not tests:
        return errors
    router = IntentRouter(
        list(intents),
        default_intent=settings.default_intent,
        default_workflow=settings.default_workflow,
    )
    for test in tests:
        decision = router.route(test.input)
        if decision.intent.id != test.expected_intent:
            errors.append(
                ValidationIssue(
                    code="INTENT_TEST_FAILED",
                    path=settings.intent_config_path or "",
                    detail=(
                        f"intent test for {test.intent_id} expected {test.expected_intent} "
                        f"but routed to {decision.intent.id}: {test.input}"
                    ),
                )
            )
    return errors


def _validate_runtime_providers(settings: Settings) -> list[ValidationIssue]:
    errors: list[ValidationIssue] = []
    if settings.retrieval_provider not in KNOWN_RETRIEVER_PROVIDERS:
        errors.append(
            ValidationIssue(
                code="UNKNOWN_RETRIEVER_PROVIDER",
                path="runtime.toml",
                detail=f"unknown retriever provider {settings.retrieval_provider}",
            )
        )
    if settings.generation_provider not in KNOWN_GENERATOR_PROVIDERS:
        errors.append(
            ValidationIssue(
                code="UNKNOWN_GENERATOR_PROVIDER",
                path="runtime.toml",
                detail=f"unknown generator provider {settings.generation_provider}",
            )
        )
    if settings.generation_fallback not in KNOWN_GENERATION_FALLBACKS:
        errors.append(
            ValidationIssue(
                code="UNKNOWN_GENERATION_FALLBACK",
                path="runtime.toml",
                detail=f"unknown generation fallback {settings.generation_fallback}",
            )
        )
    return errors


def _validate_tool_providers(settings: Settings) -> list[ValidationIssue]:
    errors: list[ValidationIssue] = []
    for tool in load_tools(settings.tool_config_path):
        if not tool.enabled:
            continue
        adapter = tool.adapter or tool.provider
        if adapter not in KNOWN_TOOL_PROVIDERS:
            errors.append(
                ValidationIssue(
                    code="UNKNOWN_TOOL_PROVIDER",
                    path=settings.tool_config_path or "",
                    detail=f"tool {tool.id} references unknown provider {adapter}",
                )
            )
    return errors


def _validate_workflow_definitions(
    settings: Settings,
    workflow_definitions: object,
    component_registry: ComponentRegistry,
) -> list[ValidationIssue]:
    errors: list[ValidationIssue] = []
    steps = component_registry.step_registry()
    terminal_steps = component_registry.terminal_steps()
    for definition in workflow_definitions:
        if getattr(definition, "type", "pipeline") == "graph":
            errors.extend(_validate_graph_workflow(settings, definition))
        for step_id in definition.steps:
            if not steps.has(step_id):
                errors.append(
                    ValidationIssue(
                        code="UNKNOWN_WORKFLOW_STEP",
                        path=settings.workflow_config_path or "",
                        detail=f"workflow {definition.id} references unknown step {step_id}",
                    )
                )
        for terminal_step in getattr(definition, "terminal_steps", []):
            if terminal_step not in definition.steps:
                errors.append(
                    ValidationIssue(
                        code="UNKNOWN_TERMINAL_STEP",
                        path=settings.workflow_config_path or "",
                        detail=(
                            f"workflow {definition.id} declares terminal step "
                            f"{terminal_step} that is not in workflow steps"
                        ),
                    )
                )
        terminal_candidates = definition.terminal_steps or definition.steps
        if not any(step_id in terminal_steps for step_id in terminal_candidates):
            errors.append(
                ValidationIssue(
                    code="NO_TERMINAL_RESPONSE_PATH",
                    path=settings.workflow_config_path or "",
                    detail=f"workflow {definition.id} has no terminal response step",
                )
            )
    return errors


def _validate_graph_workflow(settings: Settings, definition: object) -> list[ValidationIssue]:
    errors: list[ValidationIssue] = []
    node_ids = {str(node.get("id", "")) for node in getattr(definition, "nodes", []) if isinstance(node, dict)}
    start = str(getattr(definition, "start", ""))
    if not start or start not in node_ids:
        errors.append(
            ValidationIssue(
                code="UNKNOWN_GRAPH_START",
                path=settings.workflow_config_path or "",
                detail=f"workflow {definition.id} references unknown graph start {start}",
            )
        )
    for edge in getattr(definition, "edges", []):
        if not isinstance(edge, dict):
            continue
        from_node = str(edge.get("from", ""))
        to_node = str(edge.get("to", ""))
        if from_node not in node_ids:
            errors.append(
                ValidationIssue(
                    code="UNKNOWN_GRAPH_NODE",
                    path=settings.workflow_config_path or "",
                    detail=f"workflow {definition.id} edge references unknown source {from_node}",
                )
            )
        if to_node not in node_ids:
            errors.append(
                ValidationIssue(
                    code="UNKNOWN_GRAPH_EDGE_TARGET",
                    path=settings.workflow_config_path or "",
                    detail=f"workflow {definition.id} edge references unknown target {to_node}",
                )
            )
        condition = str(edge.get("condition", "default")) or "default"
        if condition not in KNOWN_GRAPH_CONDITIONS:
            errors.append(
                ValidationIssue(
                    code="UNSUPPORTED_GRAPH_CONDITION",
                    path=settings.workflow_config_path or "",
                    detail=f"workflow {definition.id} edge uses unsupported condition {condition}",
                )
            )
    return errors


def validate_project_config(project_root: Path, config_path: Path) -> ValidationResult:
    errors: list[ValidationIssue] = []
    collected_warnings: list[ValidationIssue] = []
    settings: Settings | None = None

    with warning_module.catch_warnings(record=True) as warning_records:
        warning_module.simplefilter("always", UserWarning)
        try:
            settings = load_settings(project_root, config_path)
        except ValueError as error:
            errors.append(_issue_from_exception(config_path, error))
        collected_warnings.extend(_issues_from_warnings(warning_records, config_path))

    if settings is None:
        return ValidationResult(errors=errors, warnings=collected_warnings)

    for path, loader in (
        (settings.intent_config_path, load_intents),
        (settings.workflow_config_path, load_workflows),
        (settings.policy_config_path, load_policies),
        (settings.tool_config_path, load_tools),
    ):
        if path is None or not path.exists():
            continue
        with warning_module.catch_warnings(record=True) as warning_records:
            warning_module.simplefilter("always", UserWarning)
            try:
                loader(path)
            except ValueError as error:
                errors.append(_issue_from_exception(path, error))
            collected_warnings.extend(_issues_from_warnings(warning_records, path))

    if settings.ui_config_path and settings.ui_config_path.exists():
        collected_warnings.extend(_validate_ui_config(settings.ui_config_path))

    if errors:
        return ValidationResult(errors=errors, warnings=collected_warnings)

    contract = validate_project_contract(settings)
    return ValidationResult(
        errors=[*errors, *contract.errors],
        warnings=[*collected_warnings, *contract.warnings],
    )


def _issue_from_exception(path: Path, error: ValueError) -> ValidationIssue:
    detail = str(error)
    code = "INVALID_CONFIG"
    if "Unsupported schema_version" in detail:
        code = "UNSUPPORTED_SCHEMA_VERSION"
    elif "escapes project root" in detail:
        code = "PATH_OUTSIDE_PROJECT"
    return ValidationIssue(code=code, path=path, detail=detail)


def _issues_from_warnings(
    warning_records: list[warning_module.WarningMessage],
    fallback_path: Path,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for record in warning_records:
        detail = str(record.message)
        issues.append(
            ValidationIssue(
                code="UNKNOWN_FIELD" if "Unknown field" in detail else "CONFIG_WARNING",
                path=_warning_path(detail, fallback_path),
                detail=detail,
            )
        )
    return issues


def _warning_path(detail: str, fallback_path: Path) -> Path:
    prefix = "Unknown field in "
    marker = ": "
    if detail.startswith(prefix) and marker in detail:
        raw_path = detail[len(prefix) : detail.index(marker)]
        return Path(raw_path)
    return fallback_path


def _validate_ui_config(path: Path) -> list[ValidationIssue]:
    data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    with warning_module.catch_warnings(record=True) as warning_records:
        warning_module.simplefilter("always", UserWarning)
        warn_unknown_fields(
            data,
            {
                "title",
                "home_title",
                "home_heading",
                "home_lead",
                "placeholder",
                "status_text",
                "welcome_intro",
                "welcome_items",
                "demo_sources",
            },
            path,
        )
    return _issues_from_warnings(warning_records, path)


def _validate_manifest_and_index(settings: Settings) -> list[ValidationIssue]:
    if not settings.manifest_path.exists():
        return []
    try:
        manifest_files = expand_manifest_entries(settings)
    except FileNotFoundError as error:
        return [
            ValidationIssue(
                code="MANIFEST_NOT_FOUND",
                path=settings.manifest_path,
                detail=str(error),
            )
        ]
    except ValueError as error:
        detail = str(error)
        code = "MANIFEST_EMPTY"
        if "escapes project root" in detail:
            code = "PATH_OUTSIDE_PROJECT"
        return [ValidationIssue(code=code, path=settings.manifest_path, detail=detail)]

    if not settings.index_path.exists():
        return []

    try:
        payload = read_index(settings)
    except (FileNotFoundError, ValueError) as error:
        return [ValidationIssue(code="INVALID_INDEX", path=settings.index_path, detail=str(error))]

    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        return [
            ValidationIssue(
                code="INVALID_INDEX",
                path=settings.index_path,
                detail=f"index chunks must be a list: {settings.index_path}",
            )
        ]

    indexed_sources = {
        str(chunk.get("source", "")).replace("\\", "/")
        for chunk in chunks
        if isinstance(chunk, dict) and str(chunk.get("source", "")).strip()
    }
    manifest_sources = {
        path.resolve().relative_to(settings.project_root).as_posix()
        for path in manifest_files
    }
    missing_sources = sorted(manifest_sources - indexed_sources)
    extra_sources = sorted(indexed_sources - manifest_sources)
    if not missing_sources and not extra_sources:
        return []
    detail_parts = []
    if missing_sources:
        detail_parts.append(f"missing indexed sources: {', '.join(missing_sources)}")
    if extra_sources:
        detail_parts.append(f"stale indexed sources: {', '.join(extra_sources)}")
    return [
        ValidationIssue(
            code="INDEX_MANIFEST_MISMATCH",
            path=settings.index_path,
            detail="; ".join(detail_parts),
        )
    ]
