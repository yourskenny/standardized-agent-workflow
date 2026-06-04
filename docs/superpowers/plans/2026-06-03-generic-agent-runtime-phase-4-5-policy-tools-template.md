# Generic Agent Runtime Phase 4-5 Policy, Tools, And Template Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class policy/tool ports and upgrade the project template so new agent projects declare runtime, workflow, policy, and tool configuration in files.

**Architecture:** Keep the existing ports-and-pipeline runtime. Extend settings with runtime config paths, add `PolicyGuard` and disabled-by-default `ToolProvider` ports, wire policy decisions into workflow steps, and add template TOML stubs as executable project configuration. Workflow TOML remains declarative for this phase; built-in workflow execution stays in Python.

**Tech Stack:** Python standard library, `tomllib`, dataclasses, unittest, existing `local_rag_agent` runtime modules.

---

## File Structure

- Modify `runtime/local_rag_agent/local_rag_agent/config.py`
  - Resolve optional `workflow_config`, `policy_config`, and `tool_config` paths from `[runtime]`.
  - Add configurable `default_intent` and `default_workflow`.
- Create `runtime/local_rag_agent/local_rag_agent/policy.py`
  - Owns `PolicyDefinition`, `PolicyDecision`, `PolicyGuard`, and `load_policies()`.
- Create `runtime/local_rag_agent/local_rag_agent/tools.py`
  - Owns `ToolDefinition`, `ToolResult`, `ToolProvider`, and `load_tools()`.
- Modify `runtime/local_rag_agent/local_rag_agent/runtime.py`
  - Build `PolicyGuard` and `ToolProvider` once and pass them into workflow context.
  - Pass configured default intent/workflow into `IntentRouter`.
- Modify `runtime/local_rag_agent/local_rag_agent/workflow.py`
  - Carry policy/tool ports in `WorkflowContext`.
  - Add `apply_policy` and `build_policy_response` workflow steps.
- Modify `runtime/local_rag_agent/tests/test_local_rag_agent.py`
  - Add tests for config paths, policies, tools, workflow policy trace, and template TOML stubs.
- Create template files:
  - `templates/agent-project/runtime.toml`
  - `templates/agent-project/agent/workflows.toml`
  - `templates/agent-project/agent/policies.toml`
  - `templates/agent-project/agent/tools.toml`
- Modify `templates/agent-project/README.md`
  - Document the structured runtime config files.

## Task 1: Runtime Settings For Config Paths

- [ ] Write a failing test proving `load_settings()` resolves `workflow_config`, `policy_config`, `tool_config`, `default_intent`, and `default_workflow`.
- [ ] Run the focused test and confirm it fails on missing attributes.
- [ ] Add fields to `Settings` and resolve paths from `[runtime]`.
- [ ] Pass the focused test.

## Task 2: Policy Guard Port

- [ ] Write failing tests for loading policy TOML and returning an `academic_integrity` refusal decision.
- [ ] Run the focused tests and confirm import failure for `local_rag_agent.policy`.
- [ ] Implement `policy.py` with built-in fallback policies and project TOML override support.
- [ ] Pass policy tests.

## Task 3: Tool Provider Port

- [ ] Write failing tests for loading tool TOML and disabled tool calls returning a structured `ToolResult`.
- [ ] Run the focused tests and confirm import failure for `local_rag_agent.tools`.
- [ ] Implement `tools.py` with disabled-by-default behavior.
- [ ] Pass tool tests.

## Task 4: Wire Ports Into Workflow Runtime

- [ ] Write a failing runtime test proving `refusal_with_guidance` records an `apply_policy` trace step.
- [ ] Write a failing runtime test proving a no-evidence RAG answer records a `source_required` policy decision.
- [ ] Wire `PolicyGuard` and `ToolProvider` through `AgentRuntime` into `WorkflowContext`.
- [ ] Add `apply_policy` and `build_policy_response` to built-in workflows.
- [ ] Pass focused runtime tests and the full unit suite.

## Task 5: Template Config Upgrade

- [ ] Write a failing test proving the template includes `runtime.toml`, `agent/workflows.toml`, `agent/policies.toml`, and `agent/tools.toml`.
- [ ] Add the template files with maintainable stubs.
- [ ] Update `templates/agent-project/README.md` setup flow.
- [ ] Pass the template test.

## Task 6: Verification

- [ ] Run the full unit suite:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest discover -s runtime\local_rag_agent\tests
```

- [ ] Smoke a copied template project using `runtime.toml`, ingest it, run regression, and confirm trace steps include `route_intent`, `start_workflow`, and policy-related metadata when applicable.

## Completion Criteria

- Runtime settings expose all structured config paths needed by the template.
- `PolicyGuard` and `ToolProvider` are explicit ports, not hidden ad hoc logic.
- Built-in workflows trace policy decisions.
- Template projects contain structured runtime, workflow, policy, and tool config stubs.
- Existing tests and template smoke path pass.
