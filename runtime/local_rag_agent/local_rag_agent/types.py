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
    steps: list[dict[str, object]] = field(default_factory=list)
    config_versions: dict[str, str] = field(default_factory=dict)

    def add_step(self, name: str, detail: dict[str, object] | None = None, status: str = "ok") -> None:
        self.steps.append({"name": name, "status": status, "detail": detail or {}})

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "intent": self.intent,
            "workflow": self.workflow,
            "config_versions": self.config_versions,
            "steps": self.steps,
        }


@dataclass
class AgentResponse:
    answer: str
    mode: str
    intent: str
    workflow: str
    sources: list[SourceReference] = field(default_factory=list)
    trace: AgentTrace | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "mode": self.mode,
            "intent": self.intent,
            "workflow": self.workflow,
            "sources": [source.to_dict() for source in self.sources],
            "trace": self.trace.to_dict() if self.trace else {},
            "metadata": self.metadata,
        }
