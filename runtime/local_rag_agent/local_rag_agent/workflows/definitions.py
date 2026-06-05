from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ..schema import warn_unknown_fields

SUPPORTED_WORKFLOW_SCHEMA_VERSIONS = {"workflow.v1", "workflow.v2", "workflow.v3"}


@dataclass(frozen=True)
class WorkflowDefinition:
    id: str
    steps: list[str]
    description: str = ""
    type: str = "pipeline"
    start: str = ""
    requires_sources: bool | None = None
    terminal_steps: list[str] = field(default_factory=list)
    nodes: list[dict[str, object]] = field(default_factory=list)
    edges: list[dict[str, object]] = field(default_factory=list)
    schema_version: str = ""


def load_workflows(path: Path | None) -> list[WorkflowDefinition]:
    if path is None or not path.exists():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    schema_version = str(data.get("schema_version", "")).strip()
    if schema_version and schema_version not in SUPPORTED_WORKFLOW_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported schema_version in {path}: {schema_version}")
    records = data.get("workflows", [])
    warn_unknown_fields(data, {"schema_version", "workflows"}, path)
    if not isinstance(records, list):
        raise ValueError(f"Invalid workflow config: {path}")
    return [_workflow_from_record(record, path, schema_version) for record in records if isinstance(record, dict)]


def _workflow_from_record(record: dict[str, object], path: Path, schema_version: str) -> WorkflowDefinition:
    warn_unknown_fields(
        record,
        {
            "id",
            "description",
            "type",
            "start",
            "steps",
            "nodes",
            "edges",
            "requires_sources",
            "terminal_steps",
        },
        path,
        "workflows",
    )
    workflow_id = str(record.get("id", "")).strip()
    if not workflow_id:
        raise ValueError(f"Workflow missing id in {path}")
    workflow_type = str(record.get("type", "pipeline")).strip() or "pipeline"
    if workflow_type == "graph":
        return _graph_workflow_from_record(record, path, schema_version, workflow_id)
    steps = _string_list(record.get("steps", []))
    if not steps:
        raise ValueError(f"Workflow missing steps in {path}: {workflow_id}")
    return WorkflowDefinition(
        id=workflow_id,
        steps=steps,
        description=str(record.get("description", "")),
        type=workflow_type,
        requires_sources=_optional_bool(record.get("requires_sources")),
        terminal_steps=_string_list(record.get("terminal_steps", [])),
        schema_version=schema_version,
    )


def _graph_workflow_from_record(
    record: dict[str, object],
    path: Path,
    schema_version: str,
    workflow_id: str,
) -> WorkflowDefinition:
    nodes = _graph_nodes(record.get("nodes", []), path, workflow_id)
    edges = _graph_edges(record.get("edges", []), path, workflow_id)
    steps = [str(node["step"]) for node in nodes if str(node.get("step", "")).strip()]
    if not steps:
        raise ValueError(f"Workflow missing graph nodes in {path}: {workflow_id}")
    terminal_steps = [
        str(node["step"])
        for node in nodes
        if bool(node.get("terminal")) and str(node.get("step", "")).strip()
    ]
    return WorkflowDefinition(
        id=workflow_id,
        steps=steps,
        description=str(record.get("description", "")),
        type="graph",
        start=str(record.get("start", "")).strip(),
        requires_sources=_optional_bool(record.get("requires_sources")),
        terminal_steps=terminal_steps,
        nodes=nodes,
        edges=edges,
        schema_version=schema_version,
    )


def _graph_nodes(value: object, path: Path, workflow_id: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"Workflow graph nodes must be a list in {path}: {workflow_id}")
    nodes: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("id", "")).strip()
        step = str(item.get("step", "")).strip()
        if not node_id or not step:
            raise ValueError(f"Workflow graph node missing id or step in {path}: {workflow_id}")
        nodes.append(
            {
                "id": node_id,
                "step": step,
                "terminal": bool(item.get("terminal", False)),
                "checkpoint_after": bool(item.get("checkpoint_after", False)),
            }
        )
    return nodes


def _graph_edges(value: object, path: Path, workflow_id: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"Workflow graph edges must be a list in {path}: {workflow_id}")
    edges: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        from_node = str(item.get("from", "")).strip()
        to_node = str(item.get("to", "")).strip()
        if not from_node or not to_node:
            raise ValueError(f"Workflow graph edge missing from or to in {path}: {workflow_id}")
        edges.append(
            {
                "from": from_node,
                "to": to_node,
                "condition": str(item.get("condition", "default")).strip() or "default",
            }
        )
    return edges


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None
