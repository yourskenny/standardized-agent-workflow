from __future__ import annotations

import warnings
from pathlib import Path


SUPPORTED_SCHEMA_VERSIONS = {
    "intent": {"intent.v1"},
    "policy": {"policy.v1"},
    "tool": {"tool.v1"},
}


def read_schema_version(data: dict[str, object], config_name: str, path: Path) -> str:
    version = data.get("schema_version")
    if version is None:
        return ""
    schema_version = str(version).strip()
    supported = SUPPORTED_SCHEMA_VERSIONS.get(config_name, set())
    if schema_version not in supported:
        raise ValueError(f"Unsupported schema_version in {path}: {schema_version}")
    return schema_version


def warn_unknown_fields(
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
