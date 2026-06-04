from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class IntentDefinition:
    id: str
    workflow: str
    description: str = ""
    examples: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    risk_level: str = "medium"
    policy: str = ""
    knowledge_scopes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class IntentDecision:
    intent: IntentDefinition
    confidence: float
    matched_terms: list[str] = field(default_factory=list)
    source: str = "fallback"


def load_intents(path: Path | None) -> list[IntentDefinition]:
    if path is None or not path.exists():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    records = data.get("intents", [])
    if not isinstance(records, list):
        raise ValueError(f"Invalid intent config: {path}")
    return [_intent_from_record(record, path) for record in records if isinstance(record, dict)]


def _intent_from_record(record: dict[str, object], path: Path) -> IntentDefinition:
    intent_id = str(record.get("id", "")).strip()
    if not intent_id:
        raise ValueError(f"Intent missing id in {path}")
    workflow = str(record.get("workflow", "rag_qa")).strip() or "rag_qa"
    return IntentDefinition(
        id=intent_id,
        workflow=workflow,
        description=str(record.get("description", "")),
        examples=_string_list(record.get("examples", [])),
        keywords=_string_list(record.get("keywords", [])),
        risk_level=str(record.get("risk_level", "medium")),
        policy=str(record.get("policy", "")),
        knowledge_scopes=_string_list(record.get("knowledge_scopes", [])),
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


class IntentRouter:
    def __init__(
        self,
        intents: list[IntentDefinition],
        default_intent: str = "knowledge_qa",
        default_workflow: str = "rag_qa",
    ):
        self.intents = intents
        self.default_intent = default_intent
        self.default_workflow = default_workflow

    def route(self, message: str) -> IntentDecision:
        fallback = self._fallback_intent()
        if not self.intents:
            return IntentDecision(intent=fallback, confidence=0.0, source="fallback")

        compact = message.replace(" ", "").lower()
        best: IntentDecision | None = None
        for intent in self.intents:
            matched = [term for term in intent.keywords if term and term.replace(" ", "").lower() in compact]
            if not matched:
                matched = [
                    example
                    for example in intent.examples
                    if example and example.replace(" ", "").lower() in compact
                ]
            if not matched:
                continue
            confidence = min(1.0, 0.55 + 0.15 * len(matched))
            decision = IntentDecision(intent=intent, confidence=confidence, matched_terms=matched, source="config")
            if best is None or decision.confidence > best.confidence:
                best = decision
        return best or IntentDecision(intent=fallback, confidence=0.0, source="fallback")

    def _fallback_intent(self) -> IntentDefinition:
        for intent in self.intents:
            if intent.id == self.default_intent:
                return intent
        return IntentDefinition(
            id=self.default_intent,
            workflow=self.default_workflow,
            description="Default knowledge QA intent",
        )
