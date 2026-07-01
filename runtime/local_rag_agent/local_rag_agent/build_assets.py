from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BuildSource:
    id: str
    kind: str
    location: str
    rows: int | None = None
    fields: list[str] = field(default_factory=list)
    private_fields_excluded: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "BuildSource":
        return cls(
            id=str(value.get("id", "")),
            kind=str(value.get("kind", "")),
            location=str(value.get("location", "")),
            rows=_optional_int(value.get("rows")),
            fields=_string_list(value.get("fields")),
            private_fields_excluded=_string_list(value.get("private_fields_excluded")),
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "kind": self.kind,
            "location": self.location,
            "fields": self.fields,
            "private_fields_excluded": self.private_fields_excluded,
        }
        if self.rows is not None:
            payload["rows"] = self.rows
        return payload


@dataclass(frozen=True)
class DerivedArtifact:
    path: str
    kind: str
    description: str = ""
    record_count: int | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "DerivedArtifact":
        return cls(
            path=str(value.get("path", "")),
            kind=str(value.get("kind", "")),
            description=str(value.get("description", "")),
            record_count=_optional_int(value.get("record_count")),
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "path": self.path,
            "kind": self.kind,
            "description": self.description,
        }
        if self.record_count is not None:
            payload["record_count"] = self.record_count
        return payload


@dataclass(frozen=True)
class PrivacyControl:
    policy: str
    excluded_fields: list[str] = field(default_factory=list)
    aggregation_level: str = ""
    notes: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "PrivacyControl":
        return cls(
            policy=str(value.get("policy", "")),
            excluded_fields=_string_list(value.get("excluded_fields")),
            aggregation_level=str(value.get("aggregation_level", "")),
            notes=str(value.get("notes", "")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "excluded_fields": self.excluded_fields,
            "aggregation_level": self.aggregation_level,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class BuildManifest:
    name: str
    version: str
    sources: list[BuildSource] = field(default_factory=list)
    derived_artifacts: list[DerivedArtifact] = field(default_factory=list)
    privacy: PrivacyControl | None = None
    notes: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "BuildManifest":
        privacy_value = value.get("privacy")
        return cls(
            name=str(value.get("name", "")),
            version=str(value.get("version", "")),
            sources=[BuildSource.from_mapping(item) for item in _mapping_list(value.get("sources"))],
            derived_artifacts=[
                DerivedArtifact.from_mapping(item)
                for item in _mapping_list(value.get("derived_artifacts"))
            ],
            privacy=PrivacyControl.from_mapping(privacy_value) if isinstance(privacy_value, dict) else None,
            notes=str(value.get("notes", "")),
        )

    @classmethod
    def read_json(cls, path: Path) -> "BuildManifest":
        return cls.from_mapping(json.loads(path.read_text(encoding="utf-8")))

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "version": self.version,
            "sources": [source.to_dict() for source in self.sources],
            "derived_artifacts": [artifact.to_dict() for artifact in self.derived_artifacts],
            "notes": self.notes,
        }
        if self.privacy is not None:
            payload["privacy"] = self.privacy.to_dict()
        return payload

    def audit_summary(self) -> dict[str, object]:
        excluded_fields = self.privacy.excluded_fields if self.privacy is not None else []
        source_private_fields = sorted(
            {
                field
                for source in self.sources
                for field in source.private_fields_excluded
            }
        )
        return {
            "name": self.name,
            "version": self.version,
            "source_count": len(self.sources),
            "derived_artifact_count": len(self.derived_artifacts),
            "privacy_policy": self.privacy.policy if self.privacy is not None else "",
            "excluded_field_count": len(set([*excluded_fields, *source_private_fields])),
            "aggregation_level": self.privacy.aggregation_level if self.privacy is not None else "",
        }


def _optional_int(value: object) -> int | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except ValueError:
        return None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _mapping_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]

