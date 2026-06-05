# M001 Roadmap: Evaluation Package Driven Agent Runtime Upgrade

## Vision

Move `runtime/local_rag_agent` from a configurable local RAG runtime prototype toward the evaluation package target: a standardized generic agent base where intent, workflow, policy, tools, retrieval, generation, interfaces, validation, and release gates can evolve without rewriting the runtime core.

The roadmap follows the assessment bundle's migration guidance: validate first, then make hidden runtime contracts explicit, then split ports/adapters and workflows, then add plugin/tool/interface/deployment maturity. The scaffold in the bundle is treated as an architectural reference, not a direct replacement.

## Success Criteria

- `python -m local_rag_agent validate --project ... --config ...` reports structured errors and warnings for runtime contract problems.
- Unknown intent/workflow/policy/tool/step/provider references fail in validate or strict runtime mode instead of silently falling back.
- Source-required behavior is declared in config rather than inferred from the `rag_qa` workflow name.
- Generation provider wiring supports explicit extractive fallback and OpenAI-compatible model paths.
- `ports.py` and `workflow.py` are split into maintainable modules without breaking the public `AgentRuntime.run()` API.
- Intent/workflow/tool configuration can consume the v2 fields shown in the evaluation bundle while preserving v1 compatibility.
- A plugin module can register at least one workflow step or tool adapter without editing core runtime code.
- HTTP, release-gate, template project, and deployment smoke paths prove the assembled runtime works end-to-end.

## Slices

- [x] **S01: Introduce Structured Validation Results** `risk:high` `depends:[]`
  > After this: `local_rag_agent.validator.validate_project_contract(settings)` returns machine-readable errors and warnings with stable codes, paths, and details, and existing tests still pass.

- [x] **S02: Validate Config Schema Versions And Unknown Fields** `risk:medium` `depends:[S01]`
  > After this: runtime, intent, workflow, policy, tool, and UI config files are checked through one validate path for supported schema versions and unknown-field warnings.

- [x] **S03: Validate Intent To Workflow And Policy References** `risk:high` `depends:[S01]`
  > After this: an intent that points to a missing workflow or missing policy produces a validation error before any chat request is run.

- [x] **S04: Validate Workflow Steps And Terminal Response Paths** `risk:high` `depends:[S01]`
  > After this: every configured workflow must reference known steps and at least one terminal response step, with a failing test for an unknown step and for a pipeline that cannot produce a response.

- [x] **S05: Validate Tool Providers And Configured Provider Names** `risk:high` `depends:[S01]`
  > After this: enabled tools, retriever provider, and generator provider names are checked against known adapters/providers, and invalid names fail validate with specific error codes.

- [x] **S06: Add Validate CLI Command** `risk:medium` `depends:[S02,S03,S04,S05]`
  > After this: `python -m local_rag_agent validate --project templates/agent-project --config runtime.toml` prints structured JSON or a clear human summary and exits nonzero on contract errors.

- [x] **S07: Validate Manifest And Index Readiness** `risk:medium` `depends:[S06]`
  > After this: validate detects a manifest that expands to zero public knowledge files, configured paths escaping the project root, and index/manifest mismatches where the project is expected to be ingested.

- [x] **S08: Make Unknown Workflow Fallback Explicit** `risk:high` `depends:[S03,S04,S06]`
  > After this: strict runtime and validate mode raise on unknown workflow ids, while any demo fallback to `rag_qa` is controlled by an explicit setting and covered by tests.

- [x] **S09: Add Explicit Requires-Sources Contract** `risk:high` `depends:[S04,S08]`
  > After this: `requires_sources` can be declared on workflow or intent config, policy evaluation uses that contract, and no-evidence behavior works for a source-required workflow not named `rag_qa`.

- [x] **S10: Add Workflow v2 Compatibility** `risk:medium` `depends:[S04,S09]`
  > After this: `workflow.v1` remains supported, `workflow.v2` accepts `requires_sources` and `terminal_steps`, and templates/tests cover both versions.

- [x] **S11: Add Intent v2 Routing Fields** `risk:medium` `depends:[S03,S09]`
  > After this: intent config accepts `priority`, `confidence_threshold`, `negative_keywords`, `requires_sources`, and `knowledge_scopes` while preserving v1 behavior.

- [x] **S12: Add Intent Contract Tests** `risk:medium` `depends:[S11]`
  > After this: `[[intents.tests]]` cases can be loaded and validated so conflicting or missing intent behavior is caught before release.

- [x] **S13: Repair Generation Provider Wiring** `risk:medium` `depends:[S05,S06]`
  > After this: `extractive`, `openai_compatible`, and configured fallback behavior are explicit, fake model-client tests prove `mode=model`, and no-key fallback tests prove `mode=extractive`.

- [x] **S14: Split Retriever And Generator Ports From Adapters** `risk:medium` `depends:[S13]`
  > After this: retriever/generator Protocols live under ports, lexical/extractive/openai-compatible implementations live under adapters, and imports keep the old public API compatible.

- [x] **S15: Split Tool And Policy Ports From Implementations** `risk:medium` `depends:[S05,S14]`
  > After this: tool and policy interfaces are separate from mock/keyword implementations, and current tool/policy tests still exercise behavior through the runtime.

- [x] **S16: Split Workflow Definitions, Registry, Runner, And Steps** `risk:medium` `depends:[S10,S14,S15]`
  > After this: workflow loading, step registration, pipeline execution, and built-in steps live in separate modules while `WorkflowRegistry.from_config()` and `AgentRuntime.run()` remain stable.

- [x] **S17: Add Component Registry Facade** `risk:medium` `depends:[S16]`
  > After this: runtime construction uses a single registry object for steps, retrievers, generators, policies, tools, and trace sinks, with duplicate and missing registration errors tested.

- [x] **S18: Add Minimal Plugin Loader** `risk:high` `depends:[S17]`
  > After this: `[plugins].modules` can import a local module exposing `register(registry)`, and a test plugin can add a step without editing workflow core code.

- [x] **S19: Replace Tool call_first With Select And Call Steps** `risk:high` `depends:[S16,S18]`
  > After this: workflows can use `tool.select` and `tool.call` instead of `tool.call_first`, with trace showing selected tool id and call result.

- [x] **S20: Add Tool Output Validation And Response Step** `risk:high` `depends:[S19]`
  > After this: `tool.validate_output` checks configured output schema and `response.tool_result` returns sanitized output or tool errors through the standard `AgentResponse`.

- [x] **S21: Upgrade Tool v2 Config Contract** `risk:medium` `depends:[S20]`
  > After this: tool config supports `adapter`, `allowed_intents`, `risk_level`, `timeout_seconds`, `max_output_bytes`, `requires_approval`, `input_mapping`, `input_schema`, and `output_schema`.

- [x] **S22: Add Trace Sink And Structured Runtime Logs** `risk:medium` `depends:[S17,S21]`
  > After this: each response trace can be sent to a pluggable sink and structured logs include request id, intent, workflow, step statuses, and config versions.

- [x] **S23: Add HTTP Health Version And Validate Endpoints** `risk:medium` `depends:[S06,S16]`
  > After this: the server exposes `/healthz`, `/version`, and `/api/v1/validate` without duplicating runtime orchestration outside `AgentRuntime`.

- [x] **S24: Add HTTP Chat v1 Contract And Error Envelope** `risk:medium` `depends:[S22,S23]`
  > After this: `/api/v1/chat` accepts message/history/metadata, returns answer/mode/intent/workflow/sources/trace, and uses a consistent JSON error envelope.

- [x] **S25: Add HTTP Safety Hooks** `risk:medium` `depends:[S24]`
  > After this: server configuration supports request body limit, timeout, token/basic auth hook, CORS allowlist, and rate-limit hook stubs without hard-coding project behavior.

- [x] **S26: Wire Release Gate To Validate** `risk:medium` `depends:[S06,S09,S12]`
  > After this: release-gate fails if validate fails before reading regression results.

- [x] **S27: Extend Release Gate Trace And Policy Checks** `risk:medium` `depends:[S21,S22,S26]`
  > After this: release-gate checks route_intent, start_workflow, source-required retrieval trace, policy trace, tool trace for tool workflows, and config_versions.

- [x] **S28: Upgrade Template Project To v2 Contracts** `risk:low` `depends:[S10,S11,S21,S23]`
  > After this: `templates/agent-project` uses v2 intent/workflow/policy/tool examples and still passes ingest, chat, validate, and regression smoke paths.

- [x] **S29: Add Packaging And Environment Files** `risk:medium` `depends:[S24,S28]`
  > After this: the repo includes package metadata, `.env.example`, and either Dockerfile or equivalent run commands for local service startup.

- [x] **S30: Add End-To-End Template Runtime Smoke Gate** `risk:high` `depends:[S27,S28,S29]`
  > After this: one command validates, ingests, runs regression, runs release-gate, and starts or probes the HTTP service for the template project.

## Key Risks

- Validation can become too shallow if it only checks file parsing; slices S03-S07 force cross-file and project-readiness checks early.
- Refactors can break imports for existing users; slices S14-S16 require compatibility through current public imports and tests.
- Tool and plugin work can over-expand scope; slices S18-S21 keep it to one local plugin path and one mock-style tool contract before external tools.
- HTTP/server work can duplicate orchestration; slices S23-S25 require interfaces to call `AgentRuntime` rather than rebuilding the workflow path.
- v2 config support can break v1 templates; slices S10-S12 and S28 require backward compatibility evidence.

## Proof Strategy

- Unit tests cover validators, schema handling, registries, provider selection, intent routing, workflow loading, policy decisions, and tool contracts.
- Integration tests run `AgentRuntime.from_project(...).run(...)` against temporary projects and `templates/agent-project`.
- CLI tests cover `validate`, `ingest`, `chat`, `regression`, and `release-gate` exit codes.
- HTTP tests cover health/version/validate/chat endpoints and error envelopes.
- A final smoke gate exercises the template project through validate, ingest, chat/regression, release-gate, and HTTP probe.

## Verification Classes

- `contract`: config cross-file validation, v1/v2 compatibility, registry lookup behavior.
- `runtime`: request routing, workflow pipeline execution, policy fallback, tool execution, generation fallback.
- `interface`: CLI output/exit codes and HTTP endpoint contracts.
- `template`: `templates/agent-project` remains a runnable example.
- `release`: regression summary, trace contract checks, package/deployment smoke commands.

## Definition Of Done

- Every slice checkbox above has a corresponding passing test or documented command evidence.
- `AgentRuntime.run()` remains the single orchestration boundary for CLI, HTTP, tests, and future interfaces.
- New config-driven features fail fast on invalid references and unsupported providers.
- The template project is validated as the canonical demonstration project.
- The roadmap's final smoke gate proves the 85/100+ target shape is not only documented but executable.

## Requirement Coverage

- Assessment Phase 1, config contract validation: S01-S07, S26.
- Unknown workflow fail-fast and source contract cleanup: S08-S10.
- Generation provider wiring: S13.
- Ports/adapters split: S14-S15.
- Workflow split: S16-S17.
- Plugin registration: S18.
- Intent router v2: S11-S12.
- Tool runtime v1/v2: S19-S21.
- Network and deployment standardization: S23-S25, S29-S30.
- Observability and release-gate upgrade: S22, S26-S27.
- Template migration: S28.

## Horizontal Checklist

- [x] Preserve the public Python boundary: `AgentRuntime.from_project(...)` and `AgentRuntime.run(...)`.
- [x] Preserve Windows PowerShell-friendly commands.
- [x] Keep v1 config compatibility until v2 template smoke is proven.
- [x] Keep trace fields serializable through `AgentResponse.to_dict()`.
- [x] Avoid moving unrelated docs or generated temp artifacts into the milestone.
- [x] Update README/runtime docs only when behavior is implemented, not merely planned.

## Boundary Map

### S01-S07 Produce

`local_rag_agent.validator.ValidationResult`, stable validation error codes, validate CLI behavior, and project-readiness checks.

### S08-S12 Consume

Validation primitives from S01-S07 to enforce workflow, source, and intent contracts at config load time and runtime selection time.

### S13-S15 Consume

Provider validation from S05 and produce adapter boundaries for retrieval, generation, policy, and tools.

### S16-S18 Consume

Adapter boundaries from S14-S15 and produce workflow modules plus a component registry that plugin loading can extend.

### S19-S21 Consume

Workflow step registration and plugin registry from S16-S18 and produce tool runtime steps with schema-validated outputs.

### S22-S25 Consume

Runtime trace, registry, and validation contracts and produce interface-grade observability plus HTTP endpoints.

### S26-S28 Consume

Validation, v2 intent/workflow/tool contracts, and trace contracts and produce release-gate enforcement plus a migrated template project.

### S29-S30 Consume

Template, HTTP, release-gate, and packaging work and produce the final deployable smoke path.
