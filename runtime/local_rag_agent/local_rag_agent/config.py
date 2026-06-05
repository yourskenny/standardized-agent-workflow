from __future__ import annotations

import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path


SUPPORTED_SCHEMA_VERSIONS = {
    "runtime": {"runtime.v1"},
}


@dataclass(frozen=True)
class Settings:
    project_root: Path
    prompt_path: Path
    manifest_path: Path
    knowledge_root: Path
    index_path: Path
    config_path: Path | None = None
    chunk_size: int = 1200
    chunk_overlap: int = 160
    top_k: int = 5
    retrieval_provider: str = "lexical"
    generation_provider: str = "openai_compatible"
    generation_fallback: str = "extractive"
    regression_output_dir: Path | None = None
    intent_config_path: Path | None = None
    workflow_config_path: Path | None = None
    policy_config_path: Path | None = None
    tool_config_path: Path | None = None
    ui_config_path: Path | None = None
    model_config_path: Path | None = None
    default_intent: str = "knowledge_qa"
    default_workflow: str = "rag_qa"
    allow_workflow_fallback: bool = False
    plugin_modules: list[str] = field(default_factory=list)
    server_request_body_limit_bytes: int = 1_000_000
    server_timeout_seconds: float = 30
    server_auth_token: str = ""
    server_basic_auth_username: str = ""
    server_basic_auth_password: str = ""
    server_cors_allowlist: list[str] = field(default_factory=list)
    retrieval_source_boosts: list[tuple[str, float]] | None = None
    config_schema_versions: dict[str, str] | None = None


def load_settings(project_root: Path, config_path: Path) -> Settings:
    root = project_root.resolve()
    config_file = config_path.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Project root not found: {root}")
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    data = tomllib.loads(config_file.read_text(encoding="utf-8-sig"))
    schema_versions = _schema_versions(data, "runtime", config_file)
    _warn_unknown_fields(
        data,
        {
            "schema_version",
            "project",
            "runtime",
            "retrieval",
            "generation",
            "plugins",
            "server",
            "regression",
        },
        config_file,
    )
    project = data.get("project", {})
    runtime = data.get("runtime", {})
    retrieval = data.get("retrieval", {})
    regression = data.get("regression", {})
    generation = data.get("generation", {})
    plugins = data.get("plugins", {})
    server = data.get("server", {})
    _warn_unknown_fields(project, {"prompt_path", "manifest_path", "knowledge_root", "index_path"}, config_file, "project")
    _warn_unknown_fields(
        runtime,
        {
            "default_intent",
            "default_workflow",
            "intent_config",
            "workflow_config",
            "policy_config",
            "tool_config",
            "ui_config",
            "model_config",
            "allow_workflow_fallback",
        },
        config_file,
        "runtime",
    )
    _warn_unknown_fields(
        retrieval,
        {"provider", "chunk_size", "chunk_overlap", "top_k", "source_boosts"},
        config_file,
        "retrieval",
    )
    _warn_unknown_fields(generation, {"provider", "fallback"}, config_file, "generation")
    _warn_unknown_fields(plugins, {"modules"}, config_file, "plugins")
    _warn_unknown_fields(
        server,
        {
            "request_body_limit_bytes",
            "timeout_seconds",
            "auth_token",
            "basic_auth_username",
            "basic_auth_password",
            "cors_allowlist",
        },
        config_file,
        "server",
    )
    _warn_unknown_fields(regression, {"output_dir"}, config_file, "regression")

    prompt_path = _resolve_inside(root, project.get("prompt_path", "agent/system-prompt.md"))
    manifest_path = _resolve_inside(root, project.get("manifest_path", "knowledge_base/_manifests/current-upload-manifest.md"))
    knowledge_root = _resolve_inside(root, project.get("knowledge_root", "knowledge_base"))
    index_path = _resolve_inside(root, project.get("index_path", ".local_rag_agent/index.json"))
    regression_output_dir = None
    if regression.get("output_dir"):
        regression_output_dir = _resolve_inside(root, regression["output_dir"])
    intent_config_path = None
    if runtime.get("intent_config"):
        intent_config_path = _resolve_inside(root, runtime["intent_config"])
    workflow_config_path = None
    if runtime.get("workflow_config"):
        workflow_config_path = _resolve_inside(root, runtime["workflow_config"])
    policy_config_path = None
    if runtime.get("policy_config"):
        policy_config_path = _resolve_inside(root, runtime["policy_config"])
    tool_config_path = None
    if runtime.get("tool_config"):
        tool_config_path = _resolve_inside(root, runtime["tool_config"])
    ui_config_path = None
    if runtime.get("ui_config"):
        ui_config_path = _resolve_inside(root, runtime["ui_config"])
    model_config_path = None
    if runtime.get("model_config"):
        model_config_path = _resolve_inside(root, runtime["model_config"])

    return Settings(
        project_root=root,
        config_path=config_file,
        prompt_path=prompt_path,
        manifest_path=manifest_path,
        knowledge_root=knowledge_root,
        index_path=index_path,
        chunk_size=int(retrieval.get("chunk_size", 1200)),
        chunk_overlap=int(retrieval.get("chunk_overlap", 160)),
        top_k=int(retrieval.get("top_k", 5)),
        retrieval_provider=str(retrieval.get("provider", "lexical")),
        generation_provider=str(generation.get("provider", "openai_compatible")),
        generation_fallback=str(generation.get("fallback", "extractive")),
        regression_output_dir=regression_output_dir,
        intent_config_path=intent_config_path,
        workflow_config_path=workflow_config_path,
        policy_config_path=policy_config_path,
        tool_config_path=tool_config_path,
        ui_config_path=ui_config_path,
        model_config_path=model_config_path,
        default_intent=str(runtime.get("default_intent", "knowledge_qa")),
        default_workflow=str(runtime.get("default_workflow", "rag_qa")),
        allow_workflow_fallback=bool(runtime.get("allow_workflow_fallback", False)),
        plugin_modules=_string_list(plugins.get("modules", [])),
        server_request_body_limit_bytes=int(server.get("request_body_limit_bytes", 1_000_000)),
        server_timeout_seconds=float(server.get("timeout_seconds", 30)),
        server_auth_token=str(server.get("auth_token", "")),
        server_basic_auth_username=str(server.get("basic_auth_username", "")),
        server_basic_auth_password=str(server.get("basic_auth_password", "")),
        server_cors_allowlist=_string_list(server.get("cors_allowlist", [])),
        retrieval_source_boosts=_source_boosts(retrieval.get("source_boosts", []), config_file),
        config_schema_versions=schema_versions,
    )


def _resolve_inside(root: Path, value: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Configured path escapes project root: {value}")
    return resolved


def _schema_versions(data: dict[str, object], config_name: str, path: Path) -> dict[str, str]:
    version = data.get("schema_version")
    if version is None:
        return {}
    schema_version = str(version).strip()
    supported = SUPPORTED_SCHEMA_VERSIONS.get(config_name, set())
    if schema_version not in supported:
        raise ValueError(f"Unsupported schema_version in {path}: {schema_version}")
    return {config_name: schema_version}


def _warn_unknown_fields(
    data: object,
    allowed: set[str],
    path: Path,
    section: str | None = None,
) -> None:
    if not isinstance(data, dict):
        return
    for key in sorted(str(item) for item in data):
        if key not in allowed:
            label = f"[{section}].{key}" if section else key
            warnings.warn(f"Unknown field in {path}: {label}", UserWarning, stacklevel=2)


def _source_boosts(value: object, path: Path) -> list[tuple[str, float]] | None:
    if value in (None, []):
        return None
    if not isinstance(value, list):
        raise ValueError(f"Invalid retrieval.source_boosts in {path}")
    boosts: list[tuple[str, float]] = []
    for record in value:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid retrieval.source_boosts entry in {path}")
        pattern = str(record.get("pattern", "")).strip()
        if not pattern:
            raise ValueError(f"retrieval.source_boosts entry missing pattern in {path}")
        boosts.append((pattern, float(record.get("boost", 0))))
    return boosts


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
