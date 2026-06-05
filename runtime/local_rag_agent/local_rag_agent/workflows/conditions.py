from __future__ import annotations

from .runner import WorkflowContext


def condition_matches(condition: str, context: WorkflowContext) -> bool:
    if condition == "default":
        return True
    if condition == "policy.blocked":
        return bool(context.policy_decision is not None and not context.policy_decision.allowed)
    if condition == "intent.requires_tool":
        return bool(getattr(context.intent_decision.intent, "requires_tool", False))
    return False
