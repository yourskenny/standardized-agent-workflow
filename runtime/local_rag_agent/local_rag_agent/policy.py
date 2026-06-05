from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .intent import IntentDecision
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


class PolicyGuard:
    def __init__(self, policies: list[PolicyDefinition]):
        self.policies = {policy.id: policy for policy in policies}

    @classmethod
    def builtins(cls, project_policies: list[PolicyDefinition] | None = None) -> "PolicyGuard":
        policies = {policy.id: policy for policy in _builtin_policies()}
        for policy in project_policies or []:
            policies[policy.id] = policy
        return cls(list(policies.values()))

    @classmethod
    def from_config(cls, path: Path | None) -> "PolicyGuard":
        return cls.builtins(load_policies(path))

    def evaluate(
        self,
        message: str,
        intent_decision: IntentDecision,
        retrieved_chunks: list[dict[str, object]] | None = None,
        require_sources: bool = False,
    ) -> PolicyDecision:
        intent_policy = intent_decision.intent.policy
        if intent_policy and intent_policy in self.policies:
            return _decision_from_policy(self.policies[intent_policy])

        compact = message.replace(" ", "").lower()
        for policy in self.policies.values():
            matched = [term for term in policy.keywords if term and term.replace(" ", "").lower() in compact]
            if matched:
                return _decision_from_policy(policy)

        if require_sources and not retrieved_chunks and "source_required" in self.policies:
            return _decision_from_policy(self.policies["source_required"])

        return PolicyDecision(allowed=True)


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


def _builtin_policies() -> list[PolicyDefinition]:
    return [
        PolicyDefinition(
            id="academic_integrity",
            action="refuse",
            reason="complete_submission_request",
            message=(
                "我不能直接替你完成完整论文、完整作业或可直接提交的报告，"
                "但可以帮你拆成研究问题、资料来源、分析方法、代码步骤、结构和检查清单。"
            ),
            keywords=["完整论文", "完整作业", "完整报告", "直接提交", "代写", "complete paper"],
        ),
        PolicyDefinition(
            id="source_required",
            action="no_evidence",
            reason="no_retrieval_evidence",
            message="根据当前知识库资料，未找到明确说明。建议你向负责人或维护者确认。",
        ),
    ]


def _decision_from_policy(policy: PolicyDefinition) -> PolicyDecision:
    allowed = policy.action in {"allow", ""}
    return PolicyDecision(
        allowed=allowed,
        policy_id=policy.id,
        action=policy.action,
        reason=policy.reason,
        message=policy.message,
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
