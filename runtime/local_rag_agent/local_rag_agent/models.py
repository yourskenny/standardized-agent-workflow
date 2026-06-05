from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .schema import read_schema_version, warn_unknown_fields


@dataclass(frozen=True)
class ModelDefinition:
    id: str
    provider: str = "openai_compatible"
    model: str = "gpt-4.1-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "LOCAL_AGENT_API_KEY"
    fallback: str = "extractive"
    timeout_seconds: int = 60
    schema_version: str = ""


def load_models(path: Path | None) -> list[ModelDefinition]:
    if path is None or not path.exists():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    schema_version = read_schema_version(data, "models", path)
    warn_unknown_fields(data, {"schema_version", "models"}, path)
    records = data.get("models", [])
    if not isinstance(records, list):
        raise ValueError(f"Invalid model config: {path}")
    return [_model_from_record(record, path, schema_version) for record in records if isinstance(record, dict)]


def _model_from_record(record: dict[str, object], path: Path, schema_version: str) -> ModelDefinition:
    warn_unknown_fields(
        record,
        {"id", "provider", "model", "base_url", "api_key_env", "fallback", "timeout_seconds"},
        path,
        "models",
    )
    model_id = str(record.get("id", "")).strip()
    if not model_id:
        raise ValueError(f"Model missing id in {path}")
    return ModelDefinition(
        id=model_id,
        provider=str(record.get("provider", "openai_compatible")).strip() or "openai_compatible",
        model=str(record.get("model", "gpt-4.1-mini")).strip() or "gpt-4.1-mini",
        base_url=str(record.get("base_url", "https://api.openai.com/v1")).strip() or "https://api.openai.com/v1",
        api_key_env=str(record.get("api_key_env", "LOCAL_AGENT_API_KEY")).strip() or "LOCAL_AGENT_API_KEY",
        fallback=str(record.get("fallback", "extractive")).strip() or "extractive",
        timeout_seconds=int(record.get("timeout_seconds", 60)),
        schema_version=schema_version,
    )
