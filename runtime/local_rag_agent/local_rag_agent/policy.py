from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .schema import read_schema_version, warn_unknown_fields


@dataclass(frozen=True)
class PolicyDefinition:
    id: str
    action: str
    reason: str = ""
    message: str = ""
    keywords: list[str] = field(default_factory=list)
    schema_version: str = ""


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    policy_id: str = ""
    action: str = "allow"
    reason: str = ""
    message: str = ""


def load_policies(path: Path | None) -> list[PolicyDefinition]:
    if path is None or not path.exists():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    schema_version = read_schema_version(data, "policy", path)
    warn_unknown_fields(data, {"schema_version", "policies"}, path)
    records = data.get("policies", [])
    if not isinstance(records, list):
        raise ValueError(f"Invalid policy config: {path}")
    return [_policy_from_record(record, path, schema_version) for record in records if isinstance(record, dict)]


def _policy_from_record(record: dict[str, object], path: Path, schema_version: str) -> PolicyDefinition:
    warn_unknown_fields(
        record,
        {"id", "action", "reason", "message", "keywords"},
        path,
        "policies",
    )
    policy_id = str(record.get("id", "")).strip()
    if not policy_id:
        raise ValueError(f"Policy missing id in {path}")
    return PolicyDefinition(
        id=policy_id,
        action=str(record.get("action", "allow")).strip() or "allow",
        reason=str(record.get("reason", "")),
        message=str(record.get("message", "")),
        keywords=_string_list(record.get("keywords", [])),
        schema_version=schema_version,
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def __getattr__(name: str) -> object:
    if name in {"PolicyGuard", "KeywordPolicyGuard"}:
        from .adapters.policies import KeywordPolicyGuard

        return KeywordPolicyGuard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "KeywordPolicyGuard",
    "PolicyDecision",
    "PolicyDefinition",
    "PolicyGuard",
    "load_policies",
]
