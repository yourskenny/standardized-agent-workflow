from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ..schema import warn_unknown_fields

SUPPORTED_WORKFLOW_SCHEMA_VERSIONS = {"workflow.v1", "workflow.v2"}


@dataclass(frozen=True)
class WorkflowDefinition:
    id: str
    steps: list[str]
    description: str = ""
    requires_sources: bool | None = None
    terminal_steps: list[str] = field(default_factory=list)
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
        {"id", "description", "steps", "requires_sources", "terminal_steps"},
        path,
        "workflows",
    )
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
        requires_sources=_optional_bool(record.get("requires_sources")),
        terminal_steps=_string_list(record.get("terminal_steps", [])),
        schema_version=schema_version,
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None
