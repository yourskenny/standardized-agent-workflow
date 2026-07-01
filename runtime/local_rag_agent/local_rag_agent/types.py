from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentRequest:
    message: str
    history: list[dict[str, object]] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class SourceReference:
    source: str = ""
    title: str = ""
    chunk_id: str = ""
    score: float | int = 0
    content: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "SourceReference":
        score = value.get("score", 0)
        return cls(
            source=str(value.get("source", "")),
            title=str(value.get("title", "")),
            chunk_id=str(value.get("chunk_id", "")),
            score=score if isinstance(score, (int, float)) else 0,
            content=str(value.get("content", "")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "title": self.title,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "content": self.content,
        }


@dataclass
class AgentTrace:
    intent: str
    workflow: str
    request_id: str = ""
    run_id: str = ""
    steps: list[dict[str, object]] = field(default_factory=list)
    config_versions: dict[str, str] = field(default_factory=dict)

    def add_step(self, name: str, detail: dict[str, object] | None = None, status: str = "ok") -> None:
        self.steps.append({"name": name, "status": status, "detail": detail or {}})

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "run_id": self.run_id,
            "intent": self.intent,
            "workflow": self.workflow,
            "config_versions": self.config_versions,
            "steps": self.steps,
        }


@dataclass(frozen=True)
class GenerationRecord:
    mode: str = ""
    provider: str = ""
    model: str = ""
    input_blocks: list[dict[str, object]] = field(default_factory=list)
    source_count: int = 0
    credential_status: str = ""
    fallback: str = ""
    error: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "GenerationRecord":
        return cls(
            mode=str(value.get("mode", "")),
            provider=str(value.get("provider", "")),
            model=str(value.get("model", "")),
            input_blocks=[
                item for item in value.get("input_blocks", [])
                if isinstance(item, dict)
            ] if isinstance(value.get("input_blocks"), list) else [],
            source_count=_safe_int(value.get("source_count", 0)),
            credential_status=str(value.get("credential_status", "")),
            fallback=str(value.get("fallback", "")),
            error=str(value.get("error", "")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "provider": self.provider,
            "model": self.model,
            "input_blocks": self.input_blocks,
            "source_count": self.source_count,
            "credential_status": self.credential_status,
            "fallback": self.fallback,
            "error": self.error,
        }


@dataclass
class AgentResponse:
    answer: str
    mode: str
    intent: str
    workflow: str
    sources: list[SourceReference] = field(default_factory=list)
    trace: AgentTrace | None = None
    generation: GenerationRecord | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "mode": self.mode,
            "intent": self.intent,
            "workflow": self.workflow,
            "sources": [source.to_dict() for source in self.sources],
            "trace": self.trace.to_dict() if self.trace else {},
            "generation": self.generation.to_dict() if self.generation else {},
            "metadata": self.metadata,
        }


def _safe_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0
