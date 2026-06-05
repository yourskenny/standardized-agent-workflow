from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .schema import read_schema_version, warn_unknown_fields


@dataclass(frozen=True)
class IntentDefinition:
    id: str
    workflow: str
    description: str = ""
    examples: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    negative_keywords: list[str] = field(default_factory=list)
    priority: int = 0
    confidence_threshold: float = 0.0
    risk_level: str = "medium"
    policy: str = ""
    knowledge_scopes: list[str] = field(default_factory=list)
    requires_sources: bool | None = None
    schema_version: str = ""


@dataclass(frozen=True)
class IntentDecision:
    intent: IntentDefinition
    confidence: float
    matched_terms: list[str] = field(default_factory=list)
    source: str = "fallback"


@dataclass(frozen=True)
class IntentTestCase:
    intent_id: str
    input: str
    expected_intent: str


def load_intents(path: Path | None) -> list[IntentDefinition]:
    if path is None or not path.exists():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    schema_version = read_schema_version(data, "intent", path)
    warn_unknown_fields(data, {"schema_version", "intents"}, path)
    records = data.get("intents", [])
    if not isinstance(records, list):
        raise ValueError(f"Invalid intent config: {path}")
    return [_intent_from_record(record, path, schema_version) for record in records if isinstance(record, dict)]


def load_intent_tests(path: Path | None) -> list[IntentTestCase]:
    if path is None or not path.exists():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    read_schema_version(data, "intent", path)
    records = data.get("intents", [])
    if not isinstance(records, list):
        raise ValueError(f"Invalid intent config: {path}")
    tests: list[IntentTestCase] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        intent_id = str(record.get("id", "")).strip()
        for test_record in record.get("tests", []):
            if not isinstance(test_record, dict):
                continue
            input_text = str(test_record.get("input", "")).strip()
            expected = str(test_record.get("expected_intent", "")).strip()
            if input_text and expected:
                tests.append(
                    IntentTestCase(
                        intent_id=intent_id,
                        input=input_text,
                        expected_intent=expected,
                    )
                )
    return tests


def _intent_from_record(record: dict[str, object], path: Path, schema_version: str) -> IntentDefinition:
    warn_unknown_fields(
        record,
        {
            "id",
            "workflow",
            "description",
            "examples",
            "keywords",
            "negative_keywords",
            "priority",
            "confidence_threshold",
            "risk_level",
            "policy",
            "knowledge_scopes",
            "requires_sources",
            "tests",
        },
        path,
        "intents",
    )
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
        negative_keywords=_string_list(record.get("negative_keywords", [])),
        priority=int(record.get("priority", 0)),
        confidence_threshold=float(record.get("confidence_threshold", 0.0)),
        risk_level=str(record.get("risk_level", "medium")),
        policy=str(record.get("policy", "")),
        knowledge_scopes=_string_list(record.get("knowledge_scopes", [])),
        requires_sources=_optional_bool(record.get("requires_sources")),
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
            negative_matched = [
                term for term in intent.negative_keywords if term and term.replace(" ", "").lower() in compact
            ]
            if negative_matched:
                continue
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
            if confidence < intent.confidence_threshold:
                continue
            decision = IntentDecision(intent=intent, confidence=confidence, matched_terms=matched, source="config")
            if (
                best is None
                or decision.intent.priority > best.intent.priority
                or (
                    decision.intent.priority == best.intent.priority
                    and decision.confidence > best.confidence
                )
            ):
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
