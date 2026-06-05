# Agent Runtime Kernel Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the current configurable local agent runtime into a plugin-aware, strongly validated, state-ready agent kernel without expanding into a full visual platform.

**Architecture:** Keep `AgentRuntime` as the single core entrypoint and harden the contracts around it. The first release fixes the validator/runtime registry split with `StepDefinition`; later releases add tool runtime, run/checkpoint storage, graph workflow, provider resolution, prompt compilation, interface separation, retrieval v2, and skills/memory.

**Tech Stack:** Python standard library, `tomllib`, dataclasses, sqlite3, unittest, existing `local_rag_agent` runtime modules, existing template project TOML files.

---

## Source Package Summary

This plan is based on `agent_architecture_market_learning_bundle.zip`. The package rates the current repo at roughly 82/100:

- Strong enough as a configurable RAG/agent runtime kernel draft.
- Not yet a LangGraph-style state graph runtime.
- Not yet a Dify-style app platform.
- Not yet a long-running assistant system with memory, sessions, checkpointing, and gateway separation.

The package identifies one immediate P0:

```text
Runtime plugins can register workflow steps.
Validator and release gate still validate against builtin steps only.
Result: plugin workflows can run, but cannot become a stable project contract.
```

The recommended implementation order is:

```text
S31 -> S32 -> S34 -> S33 -> S36 -> S35 -> S38 -> S37 -> S39
```

## Planning Boundary

This is a hardening plan, not a platform rewrite.

In scope:

- Plugin-aware validation and release gate.
- Step metadata contract.
- Tool runtime contract and audit path.
- SQLite-backed run/checkpoint primitives.
- Graph workflow v1 as an additive workflow type.
- Provider and prompt contracts.
- Interface separation around the existing runtime.
- Retrieval v2 and skills/memory as later-stage extensions.

Out of scope for this plan:

- Visual workflow builder.
- Marketplace.
- Multi-tenant SaaS workspace model.
- Full multi-agent framework.
- Automatic self-modifying skills.
- Plugin sandbox implementation beyond declaring the contract and safe validation boundaries.

## File Structure

### S31 Plugin-Aware Validate

- Create `runtime/local_rag_agent/local_rag_agent/components/definitions.py`
  - Owns `StepDefinition` and related validation metadata.
- Create `runtime/local_rag_agent/local_rag_agent/components/registry.py`
  - New focused registry implementation if the existing flat `components.py` becomes too broad.
- Modify `runtime/local_rag_agent/local_rag_agent/components.py`
  - Keep compatibility exports.
  - Allow `register_step()` to accept either a callable or `StepDefinition`.
  - Expose `step_definitions()`, `terminal_steps()`, and a callable `step_registry()`.
- Modify `runtime/local_rag_agent/local_rag_agent/workflows/steps.py`
  - Register builtins with metadata.
  - Mark response-producing steps as terminal.
- Modify `runtime/local_rag_agent/local_rag_agent/validator.py`
  - Build `ComponentRegistry.from_settings(settings)` during contract validation.
  - Validate workflow steps and terminal response paths from registry definitions.
  - Remove or downgrade hard-coded `TERMINAL_RESPONSE_STEPS`.
- Modify `runtime/local_rag_agent/local_rag_agent/regression.py`
  - Ensure smoke/release-gate calls validate with the same plugin-aware contract.
- Modify `runtime/local_rag_agent/tests/test_local_rag_agent.py`
  - Add plugin-aware validate and release-gate tests.

### S32 Tool Runtime v3

- Create `runtime/local_rag_agent/local_rag_agent/tools/runtime.py`
  - Owns `ToolRuntime`, selection, input preparation, authorization, adapter dispatch, output validation, redaction, and audit.
- Create `runtime/local_rag_agent/local_rag_agent/ports/tool.py`
  - Owns `ToolAdapterPort` once `ports.py` is split.
- Modify `runtime/local_rag_agent/local_rag_agent/tools.py`
  - Keep `ToolDefinition` and loader compatibility.
  - Add permissions and v3 validation support.
- Modify `runtime/local_rag_agent/local_rag_agent/adapters/tools.py`
  - Convert current `ConfiguredToolProvider.call()` path into a runtime adapter implementation or delegate.
- Modify `runtime/local_rag_agent/local_rag_agent/workflows/steps.py`
  - Make `tool.select`, `tool.call`, `tool.validate_output`, and `response.tool_result` use `ToolRuntime`.
- Modify `templates/agent-project/agent/tools.toml`
  - Add v3 examples only after loader compatibility is implemented.

### S34 SQLite Run And Checkpoint Store

- Create `runtime/local_rag_agent/local_rag_agent/stores/schema.sql`
  - Tables: `threads`, `runs`, `checkpoints`, `messages`, `tool_calls`, `memories`, `approvals`.
- Create `runtime/local_rag_agent/local_rag_agent/stores/sqlite.py`
  - Minimal `SQLiteRunStore`.
- Create `runtime/local_rag_agent/local_rag_agent/ports/store.py`
  - `RunStorePort`, checkpoint APIs, message APIs, tool-call audit APIs.
- Modify `runtime/local_rag_agent/local_rag_agent/runtime.py`
  - Add `run_id` creation and optional store injection.
- Modify `runtime/local_rag_agent/local_rag_agent/workflows/runner.py`
  - Add checkpoint hooks without changing current pipeline behavior.

### S33 Graph Workflow v1

- Modify `runtime/local_rag_agent/local_rag_agent/workflows/definitions.py`
  - Add `workflow.v3`, `type`, `start`, `nodes`, `edges`, `condition`, and `checkpoint_after`.
- Create `runtime/local_rag_agent/local_rag_agent/workflows/graph.py`
  - Deterministic graph runner.
- Create `runtime/local_rag_agent/local_rag_agent/workflows/conditions.py`
  - `default`, `policy.blocked`, `intent.requires_tool`, and simple state predicates.
- Modify `runtime/local_rag_agent/local_rag_agent/workflows/registry.py`
  - Load both pipeline and graph definitions.
  - Preserve pipeline compatibility.

### S36 Provider Resolver

- Create `runtime/local_rag_agent/local_rag_agent/models.py`
  - `ModelDefinition`, `load_models()`, schema version handling.
- Create `runtime/local_rag_agent/local_rag_agent/providers/resolver.py`
  - `ModelProviderResolver`, `CredentialResolver`, fallback trace.
- Modify `runtime/local_rag_agent/local_rag_agent/adapters/generators.py`
  - Delegate provider choice to resolver where possible.
- Modify `templates/agent-project/runtime.toml`
  - Point to optional `agent/models.toml`.
- Create `templates/agent-project/agent/models.toml`
  - Minimal extractive and openai-compatible examples.

### S35 Prompt Compiler

- Create `runtime/local_rag_agent/local_rag_agent/prompt/blocks.py`
  - `PromptBlock` with source, type, text, and approximate token count.
- Create `runtime/local_rag_agent/local_rag_agent/prompt/compiler.py`
  - Stable/context/volatile assembly.
- Create `runtime/local_rag_agent/local_rag_agent/prompt/budget.py`
  - Deterministic budget trimming.
- Modify `runtime/local_rag_agent/local_rag_agent/adapters/generators.py`
  - Accept compiled prompt context when model-backed generation is used.

### S38 Interface And Network Split

- Create `runtime/local_rag_agent/local_rag_agent/interfaces/http/`
  - `server.py`, `routes.py`, `schemas.py`, `errors.py`.
- Create `runtime/local_rag_agent/local_rag_agent/interfaces/cli/`
  - Move command-specific code out of the top-level `cli.py` gradually.
- Create `runtime/local_rag_agent/local_rag_agent/services/`
  - `runtime_service.py`, `validation_service.py`, `regression_service.py`.
- Modify `runtime/local_rag_agent/local_rag_agent/server.py`
  - Keep compatibility import/path while delegating to interface modules.

### S37 Retrieval v2

- Create `runtime/local_rag_agent/local_rag_agent/adapters/retrieval/sqlite_fts.py`
  - SQLite FTS adapter first.
- Create `runtime/local_rag_agent/local_rag_agent/adapters/retrieval/hybrid.py`
  - Lexical plus FTS merge path.
- Modify `runtime/local_rag_agent/local_rag_agent/ports.py` or split retrieval port.
  - Keep current `RetrieverProvider` compatibility.
- Modify `templates/agent-project/runtime.toml`
  - Add optional retrieval provider examples only after validate support exists.

### S39 Skills And Memory v1

- Create `runtime/local_rag_agent/local_rag_agent/memory/`
  - Markdown memory loader and optional SQLite-backed search.
- Create `runtime/local_rag_agent/local_rag_agent/skills/`
  - `SkillDefinition`, manifest loader, selection contract.
- Modify `templates/agent-project/`
  - Add `skills/` and `memory/` only after validation and prompt injection are ready.

## Phase 1: S31 Plugin-Aware Validate And StepDefinition

### Task 1: Add StepDefinition Without Breaking Existing StepRegistry

**Files:**

- Create: `runtime/local_rag_agent/local_rag_agent/components/definitions.py`
- Modify: `runtime/local_rag_agent/local_rag_agent/components.py`
- Modify: `runtime/local_rag_agent/local_rag_agent/workflows/steps.py`
- Test: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] Write a failing test proving `ComponentRegistry.register_step_definition()` stores metadata and rejects duplicate ids.

Expected test shape:

```python
def test_component_registry_registers_step_definition_metadata(self):
    from local_rag_agent.components import ComponentRegistry, StepDefinition

    def custom_response(context):
        context.response = None

    registry = ComponentRegistry()
    definition = StepDefinition(
        id="custom.response",
        fn=custom_response,
        terminal=True,
        risk_level="low",
        timeout_seconds=5,
    )

    registry.register_step_definition(definition)

    self.assertTrue(registry.has_step("custom.response"))
    self.assertEqual(registry.get_step_definition("custom.response").id, "custom.response")
    self.assertEqual(registry.terminal_steps(), {"custom.response"})
    with self.assertRaisesRegex(ValueError, "Duplicate component registration: step custom.response"):
        registry.register_step_definition(definition)
```

- [ ] Run the focused test and confirm it fails before implementation.

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.ComponentRegistryTests.test_component_registry_registers_step_definition_metadata
```

Expected: failure because `StepDefinition` or `register_step_definition()` does not exist.

- [ ] Implement `StepDefinition`.

Minimum implementation:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..workflows.runner import WorkflowStep


@dataclass(frozen=True)
class StepDefinition:
    id: str
    fn: WorkflowStep
    description: str = ""
    terminal: bool = False
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"
    timeout_seconds: float | None = None
    checkpoint_after: bool = False
```

- [ ] Extend `ComponentRegistry` so callable registration remains compatible.

Required behavior:

```text
registry.register_step("x", fn) still works.
registry.register_step_definition(StepDefinition(...)) works.
registry.step_registry() still returns workflow callable registry.
registry.terminal_steps() returns ids whose definitions are terminal.
```

- [ ] Convert builtin step registration to definitions.

Terminal builtin ids:

```text
build_policy_response
build_response
build_retrieval_debug_response
build_refusal_response
response.tool_result
```

- [ ] Run focused registry tests.

Expected: pass.

### Task 2: Make Validator Use ComponentRegistry.from_settings()

**Files:**

- Modify: `runtime/local_rag_agent/local_rag_agent/validator.py`
- Test: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] Write a failing test proving a plugin terminal step validates.

Expected test shape:

```python
def test_validate_project_contract_accepts_plugin_terminal_step(self):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        # Create runtime.toml, agent/workflows.toml, agent/intents.toml, plugin module.
        # Plugin registers StepDefinition(id="plugin.response", terminal=True).
        settings = load_settings(root, root / "runtime.toml")

        result = validate_project_contract(settings)

        self.assertTrue(result.ok, result.to_dict())
```

- [ ] Run the focused test and confirm it fails with `UNKNOWN_WORKFLOW_STEP` or `NO_TERMINAL_RESPONSE_PATH`.

- [ ] Change `_validate_workflow_definitions()` to accept a `ComponentRegistry`.

Required logic:

```text
registry = ComponentRegistry.from_settings(settings)
step_registry = registry.step_registry()
terminal_steps = registry.terminal_steps()
for workflow step: validate step_registry.has(step_id)
for explicit terminal_steps: validate step is in workflow and known
terminal path passes if explicit terminal step is terminal or if any candidate is terminal
```

- [ ] Ensure `validate_project_contract()` passes the plugin-aware registry into workflow validation and `WorkflowRegistry.from_config()`.

- [ ] Run plugin-aware validation tests.

Expected: plugin terminal workflow validates.

### Task 3: Release Gate Covers Plugin Workflow

**Files:**

- Modify: `runtime/local_rag_agent/local_rag_agent/regression.py`
- Test: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] Write a failing smoke/release-gate test using a plugin response workflow.

Acceptance:

```text
validate ok true
regression question_count > 0
release_gate ok true
response mode plugin
```

- [ ] Run the focused test and confirm current failure comes from validate.

- [ ] Wire the plugin-aware validation path through smoke/release-gate if it does not already use `validate_project_contract(settings)`.

- [ ] Run the focused plugin release-gate test.

Expected: pass.

### Task 4: Fix CWD-Independent Tests

**Files:**

- Modify: `runtime/local_rag_agent/tests/test_local_rag_agent.py`
- Possible create: `runtime/local_rag_agent/tests/conftest.py` only if switching to pytest fixtures; otherwise keep unittest helpers.

- [ ] Add repo-root helpers.

Required helper:

```python
REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = REPO_ROOT / "runtime" / "local_rag_agent"
TEMPLATE_ROOT = REPO_ROOT / "templates" / "agent-project"
```

- [ ] Replace fragile relative paths in tests with these helpers.

- [ ] Verify from repo root:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m pytest -q runtime/local_rag_agent/tests
```

Expected: all tests pass.

- [ ] Verify from runtime subdir:

```powershell
Set-Location C:\coding\standardized-agent-workflow\runtime\local_rag_agent
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m pytest -q
```

Expected: all tests pass.

### Task 5: Phase 1 Verification And Commit

- [ ] Run full verification.

```powershell
Set-Location C:\coding\standardized-agent-workflow
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m pytest -q runtime/local_rag_agent/tests
python -m local_rag_agent validate --project templates/agent-project --config templates/agent-project/runtime.toml
python -m local_rag_agent smoke --project templates/agent-project --config templates/agent-project/runtime.toml --questions templates/agent-project/examples/core-regression-questions.md
git diff --check
```

Expected:

```text
pytest: all tests passed
validate: {"ok": true, "errors": [], "warnings": []}
smoke: ok true and release_gate ok true
git diff --check: no output
```

- [ ] Commit.

```powershell
git add runtime/local_rag_agent/local_rag_agent runtime/local_rag_agent/tests/test_local_rag_agent.py
git commit -m "feat: add plugin-aware workflow validation"
```

## Phase 2: S32 Tool Runtime v3

### Task 6: Introduce ToolRuntime Around Existing ConfiguredToolProvider

**Files:**

- Create: `runtime/local_rag_agent/local_rag_agent/tools/runtime.py`
- Modify: `runtime/local_rag_agent/local_rag_agent/adapters/tools.py`
- Modify: `runtime/local_rag_agent/local_rag_agent/workflows/steps.py`
- Test: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] Write failing tests for input mapping.

Required cases:

```text
"$message" -> request.message
"$metadata.user_id" -> request.metadata["user_id"]
"$state.foo" -> context/result state value once state exists, or context.result["foo"] during this phase
missing required mapped value -> structured error
```

- [ ] Implement `ToolRuntime.prepare_input()` with only the mapping forms above.

- [ ] Write failing tests for disabled tool, allowed intents, and max output bytes.

- [ ] Delegate current mock tool call through `ToolRuntime.call()`.

- [ ] Keep existing `tool.call_first` workflow behavior passing.

### Task 7: Add Approval And Audit Contract

**Files:**

- Modify: `runtime/local_rag_agent/local_rag_agent/tools/runtime.py`
- Modify: `runtime/local_rag_agent/local_rag_agent/types.py`
- Test: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] Add tests proving `requires_approval=true` returns a pending approval response instead of calling the adapter.

- [ ] Add an in-memory audit sink for tests.

- [ ] Record audit events for selected, blocked, approved, called, failed, and redacted tool calls.

- [ ] Commit.

```powershell
git add runtime/local_rag_agent/local_rag_agent runtime/local_rag_agent/tests/test_local_rag_agent.py
git commit -m "feat: add tool runtime v3 core"
```

## Phase 3: S34 SQLite Run And Checkpoint Store

### Task 8: Add Store Schema And SQLite Adapter

**Files:**

- Create: `runtime/local_rag_agent/local_rag_agent/stores/schema.sql`
- Create: `runtime/local_rag_agent/local_rag_agent/stores/sqlite.py`
- Create: `runtime/local_rag_agent/local_rag_agent/ports/store.py`
- Test: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] Write tests that initialize an in-memory SQLite store and create a run.

- [ ] Add tables from the package scaffold:

```text
threads
runs
checkpoints
messages
tool_calls
memories
approvals
```

- [ ] Implement create/read methods for runs and checkpoints only.

- [ ] Add tool-call audit persistence after Task 7.

### Task 9: Add run_id And Checkpoint Hooks

**Files:**

- Modify: `runtime/local_rag_agent/local_rag_agent/runtime.py`
- Modify: `runtime/local_rag_agent/local_rag_agent/workflows/runner.py`
- Modify: `runtime/local_rag_agent/local_rag_agent/types.py`
- Test: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] Write a failing test proving every response trace has `run_id`.

- [ ] Write a failing test proving a checkpoint is written after a configured step.

- [ ] Implement optional store injection into `AgentRuntime`.

- [ ] Commit.

```powershell
git add runtime/local_rag_agent/local_rag_agent runtime/local_rag_agent/tests/test_local_rag_agent.py
git commit -m "feat: add sqlite run checkpoint store"
```

## Phase 4: S33 Graph Workflow v1

### Task 10: Parse And Validate workflow.v3 Graph Config

**Files:**

- Modify: `runtime/local_rag_agent/local_rag_agent/workflows/definitions.py`
- Modify: `runtime/local_rag_agent/local_rag_agent/validator.py`
- Test: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] Add tests for graph config parsing.

Required validation errors:

```text
UNKNOWN_GRAPH_START
UNKNOWN_GRAPH_NODE
UNKNOWN_GRAPH_EDGE_TARGET
NO_TERMINAL_RESPONSE_PATH
UNSUPPORTED_GRAPH_CONDITION
```

- [ ] Implement schema-compatible parser while keeping workflow.v1/v2 pipeline config unchanged.

### Task 11: Execute Deterministic Graph Workflow

**Files:**

- Create: `runtime/local_rag_agent/local_rag_agent/workflows/graph.py`
- Create: `runtime/local_rag_agent/local_rag_agent/workflows/conditions.py`
- Modify: `runtime/local_rag_agent/local_rag_agent/workflows/registry.py`
- Test: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] Add tests for `policy.blocked`, `intent.requires_tool`, and `default` edge behavior.

- [ ] Implement a simple max-step guard to prevent loops.

- [ ] Ensure pipeline workflows still pass unchanged.

- [ ] Commit.

```powershell
git add runtime/local_rag_agent/local_rag_agent runtime/local_rag_agent/tests/test_local_rag_agent.py
git commit -m "feat: add graph workflow v1"
```

## Phase 5: S36 Provider Resolver

### Task 12: Add models.toml And Resolver

**Files:**

- Create: `runtime/local_rag_agent/local_rag_agent/models.py`
- Create: `runtime/local_rag_agent/local_rag_agent/providers/resolver.py`
- Modify: `runtime/local_rag_agent/local_rag_agent/config.py`
- Modify: `runtime/local_rag_agent/local_rag_agent/adapters/generators.py`
- Create: `templates/agent-project/agent/models.toml`
- Test: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] Add tests for no-key fallback to extractive.

- [ ] Add tests that provider/model/base_url/api_key_env are recorded in trace without exposing secret values.

- [ ] Keep current generation provider config as a compatibility path.

- [ ] Commit.

```powershell
git add runtime/local_rag_agent/local_rag_agent templates/agent-project runtime/local_rag_agent/tests/test_local_rag_agent.py
git commit -m "feat: add model provider resolver"
```

## Phase 6: S35 Prompt Compiler

### Task 13: Add Prompt Blocks And Compiler Trace

**Files:**

- Create: `runtime/local_rag_agent/local_rag_agent/prompt/blocks.py`
- Create: `runtime/local_rag_agent/local_rag_agent/prompt/compiler.py`
- Create: `runtime/local_rag_agent/local_rag_agent/prompt/budget.py`
- Modify: `runtime/local_rag_agent/local_rag_agent/adapters/generators.py`
- Test: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] Add tests for stable/context/volatile block ordering.

- [ ] Add tests for deterministic budget trimming.

- [ ] Record prompt block source and type in trace.

- [ ] Commit.

```powershell
git add runtime/local_rag_agent/local_rag_agent runtime/local_rag_agent/tests/test_local_rag_agent.py
git commit -m "feat: add prompt compiler"
```

## Phase 7: S38 Interface Split

### Task 14: Split HTTP Server Without Changing API

**Files:**

- Create: `runtime/local_rag_agent/local_rag_agent/interfaces/http/server.py`
- Create: `runtime/local_rag_agent/local_rag_agent/interfaces/http/routes.py`
- Create: `runtime/local_rag_agent/local_rag_agent/interfaces/http/errors.py`
- Create: `runtime/local_rag_agent/local_rag_agent/services/runtime_service.py`
- Modify: `runtime/local_rag_agent/local_rag_agent/server.py`
- Test: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] Add tests proving `/healthz`, `/version`, `/api/v1/chat`, and `/api/v1/validate` responses remain compatible.

- [ ] Move HTTP routing out of top-level `server.py`.

- [ ] Add request_id to API response trace and structured error envelopes.

- [ ] Commit.

```powershell
git add runtime/local_rag_agent/local_rag_agent runtime/local_rag_agent/tests/test_local_rag_agent.py
git commit -m "refactor: split http interface"
```

## Phase 8: S37 Retrieval v2

### Task 15: Add SQLite FTS Retriever

**Files:**

- Create: `runtime/local_rag_agent/local_rag_agent/adapters/retrieval/sqlite_fts.py`
- Create: `runtime/local_rag_agent/local_rag_agent/adapters/retrieval/hybrid.py`
- Modify: `runtime/local_rag_agent/local_rag_agent/components.py`
- Modify: `runtime/local_rag_agent/local_rag_agent/validator.py`
- Test: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] Add tests for FTS retrieval against a tiny local SQLite index.

- [ ] Add tests for provider validation.

- [ ] Register `sqlite_fts` and `hybrid` retrievers.

- [ ] Commit.

```powershell
git add runtime/local_rag_agent/local_rag_agent runtime/local_rag_agent/tests/test_local_rag_agent.py
git commit -m "feat: add sqlite fts retrieval"
```

## Phase 9: S39 Skills And Memory v1

### Task 16: Add Readable Skills And Memory Contracts

**Files:**

- Create: `runtime/local_rag_agent/local_rag_agent/skills/definitions.py`
- Create: `runtime/local_rag_agent/local_rag_agent/skills/registry.py`
- Create: `runtime/local_rag_agent/local_rag_agent/memory/markdown.py`
- Modify: `runtime/local_rag_agent/local_rag_agent/prompt/compiler.py`
- Modify: `templates/agent-project/README.md`
- Test: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] Add tests for loading `skills/*/manifest.toml` and `SKILL.md`.

- [ ] Add tests that selected skill text enters prompt context and trace.

- [ ] Add tests that memory files are read-only unless an explicit proposal mode is used.

- [ ] Commit.

```powershell
git add runtime/local_rag_agent/local_rag_agent templates/agent-project runtime/local_rag_agent/tests/test_local_rag_agent.py
git commit -m "feat: add skills memory v1"
```

## Release Gate Upgrade Checklist

Each phase must keep the existing template path working:

```powershell
Set-Location C:\coding\standardized-agent-workflow
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m pytest -q runtime/local_rag_agent/tests
python -m local_rag_agent validate --project templates/agent-project --config templates/agent-project/runtime.toml
python -m local_rag_agent smoke --project templates/agent-project --config templates/agent-project/runtime.toml --questions templates/agent-project/examples/core-regression-questions.md
git diff --check
```

By the end of all phases, release gate coverage should include:

- Plugin step validate.
- Plugin workflow release gate.
- Tool input mapping.
- Tool disabled, approval, timeout, and output validation.
- Graph workflow validate and execution.
- Checkpoint written test.
- Resume/replay smoke path.
- Model provider fallback trace.
- Prompt block trace.
- HTTP request_id and structured errors.
- CWD-independent tests.
- Config schema version compatibility tests.

## Competitor Guidance From The Package

The package discusses Dify, LangGraph, CrewAI, OpenClaw, and Hermes as learning references. It explicitly does not recommend copying them wholesale.

What to learn:

- Dify: productized separation of app config, workflow, model abstraction, RAG pipeline, plugin lifecycle, observability, and provider registry.
- LangGraph: state, node, edge, checkpoint, thread, interrupt/resume, replay, and time-travel debugging.
- CrewAI: deterministic flow control separated from more autonomous agent/team execution.
- OpenClaw: gateway/session separation, channel adapters, readable Markdown memory, strict config validation, and last-known-good config ideas.
- Hermes: one core loop across surfaces, prompt compiler, provider runtime resolution, tool registry, session storage, skills as procedural memory, and human-approved learning loop.

What not to copy now:

- Dify's full platform, marketplace, and visual workflow builder.
- LangGraph's full distributed graph runtime.
- CrewAI's full multi-agent framework.
- OpenClaw's entire multi-channel ecosystem.
- Hermes-style automatic self-improvement or automatic skill mutation.

The resulting recommendation is to build a small, hard runtime kernel first.

## Execution Recommendation

Execute Phase 1 first and stop for review after the plugin-aware validation release gate passes. Phase 1 has the clearest evidence, fixes the current P0, and gives later phases a safer contract surface.

After Phase 1, choose between:

- Continue linearly into Tool Runtime v3 if tool use is the next product need.
- Jump to SQLite Run Store if resume/replay/human approval is the next product need.

The package recommends Tool Runtime before Graph Workflow, but Graph Workflow should not begin until the store/checkpoint path is at least minimally real.
