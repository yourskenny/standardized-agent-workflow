from __future__ import annotations

from pathlib import Path

from ..intent import IntentDecision
from ..policy import PolicyDecision, PolicyDefinition, load_policies
from ..ports import PolicyPort


class KeywordPolicyGuard(PolicyPort):
    def __init__(self, policies: list[PolicyDefinition]):
        self.policies = {policy.id: policy for policy in policies}

    @classmethod
    def builtins(cls, project_policies: list[PolicyDefinition] | None = None) -> "KeywordPolicyGuard":
        policies = {policy.id: policy for policy in _builtin_policies()}
        for policy in project_policies or []:
            policies[policy.id] = policy
        return cls(list(policies.values()))

    @classmethod
    def from_config(cls, path: Path | None) -> "KeywordPolicyGuard":
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


PolicyGuard = KeywordPolicyGuard


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
