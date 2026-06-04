from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    prompt_path: Path
    manifest_path: Path
    knowledge_root: Path
    index_path: Path
    chunk_size: int = 1200
    chunk_overlap: int = 160
    top_k: int = 5
    regression_output_dir: Path | None = None
    intent_config_path: Path | None = None
    workflow_config_path: Path | None = None
    policy_config_path: Path | None = None
    tool_config_path: Path | None = None
    default_intent: str = "knowledge_qa"
    default_workflow: str = "rag_qa"


def load_settings(project_root: Path, config_path: Path) -> Settings:
    root = project_root.resolve()
    config_file = config_path.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Project root not found: {root}")
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    data = tomllib.loads(config_file.read_text(encoding="utf-8-sig"))
    project = data.get("project", {})
    runtime = data.get("runtime", {})
    retrieval = data.get("retrieval", {})
    regression = data.get("regression", {})

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

    return Settings(
        project_root=root,
        prompt_path=prompt_path,
        manifest_path=manifest_path,
        knowledge_root=knowledge_root,
        index_path=index_path,
        chunk_size=int(retrieval.get("chunk_size", 1200)),
        chunk_overlap=int(retrieval.get("chunk_overlap", 160)),
        top_k=int(retrieval.get("top_k", 5)),
        regression_output_dir=regression_output_dir,
        intent_config_path=intent_config_path,
        workflow_config_path=workflow_config_path,
        policy_config_path=policy_config_path,
        tool_config_path=tool_config_path,
        default_intent=str(runtime.get("default_intent", "knowledge_qa")),
        default_workflow=str(runtime.get("default_workflow", "rag_qa")),
    )


def _resolve_inside(root: Path, value: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Configured path escapes project root: {value}")
    return resolved
