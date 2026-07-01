# Trusted Agent Build Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic trusted-build and generation-boundary contract to the upstream standardized agent runtime.

**Architecture:** Keep project-specific data processing outside the runtime, but standardize how projects describe source inventories, derived artifacts, privacy controls, generation inputs, and regression evidence. Extend existing dataclasses rather than replacing the runtime flow.

**Tech Stack:** Python 3.11 stdlib, dataclasses, unittest, Markdown templates.

---

### Task 1: Build Manifest Contract

**Files:**
- Create: `runtime/local_rag_agent/local_rag_agent/build_assets.py`
- Test: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [x] Add dataclasses for `BuildSource`, `DerivedArtifact`, `PrivacyControl`, and `BuildManifest`.
- [x] Add JSON round-trip helpers.
- [x] Add an audit summary helper for quick validation.
- [x] Add a unit test that verifies source, artifact, and privacy fields survive serialization.

### Task 2: Generation Boundary

**Files:**
- Modify: `runtime/local_rag_agent/local_rag_agent/types.py`
- Modify: `runtime/local_rag_agent/local_rag_agent/workflows/steps.py`
- Test: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [x] Add a `GenerationRecord` dataclass.
- [x] Add a `generation` field to `AgentResponse.to_dict()`.
- [x] Populate generation metadata in answer, refusal, debug, and tool responses.
- [x] Add a unit test proving legacy response fields remain present while the generation boundary is serialized.

### Task 3: Regression Evidence

**Files:**
- Modify: `runtime/local_rag_agent/local_rag_agent/regression.py`
- Test: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [x] Persist `generation` into each regression JSONL record.
- [x] Summarize generation modes and count missing generation records.
- [x] Add a unit test proving regression output carries the generation boundary.

### Task 4: Templates and Docs

**Files:**
- Create: `templates/agent-project/knowledge_base/_manifests/build-manifest.example.json`
- Create: `templates/agent-project/knowledge_base/_templates/build-artifact-contract.md`
- Modify: `templates/agent-project/knowledge_base/_pre_ingestion/README.md`
- Create: `docs/10-trusted-build-layer.md`
- Create: `docs/11-architecture-explanation-template.md`
- Modify: `README.md`

- [x] Add a generic build manifest example with no marketing data.
- [x] Add a template explaining how projects should document source processing.
- [x] Update pre-ingestion guidance to emphasize privacy downgrade and auditability.
- [x] Add upstream docs explaining trusted build layer and explainable architecture demos.

### Task 5: Verification

**Files:**
- Test: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [x] Run the targeted new tests.
- [x] Run the full upstream unittest file.
- [x] Review git diff for accidental downstream data leakage.

