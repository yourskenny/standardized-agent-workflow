# Generic Agent Runtime Base Design

## Goal

Evolve this repository from a runnable local RAG reference implementation into a standardized, reusable agent runtime base. The runtime should behave like a flexible foundation that different domain agents can shape through configuration, content, and optional components rather than by editing core runtime code.

The target direction has three required layers:

1. Agent scheduling base: a low-coupling runtime that can route requests through configurable workflows and components.
2. RAG workflow base: retrieval-augmented question answering remains the first built-in workflow, but it becomes one workflow among others.
3. Project template base: new agent projects should declare intents, workflows, tools, policies, and knowledge scopes in project files.

## Current State

The repository currently has two useful foundations:

- `templates/agent-project/` defines a maintainable content-project shape with a prompt, knowledge base, manifests, maintenance records, and regression questions.
- `runtime/local_rag_agent/` can ingest Markdown knowledge files, build a local JSON index, retrieve chunks, answer questions, serve a local UI/API, and run regression evidence.

The current runtime is still mostly a linear RAG path:

```text
CLI/server
  -> load settings
  -> read index
  -> rank chunks
  -> answer with model or extractive fallback
  -> return sources
```

That shape is good for a reference implementation, but it is not yet a general agent base. Intent recognition is documented in `agent/intent-map.md`, but it is not executable configuration. Workflow choice is hard-coded. Adding tools, evaluators, policy checks, or new task modes would require editing central code paths.

## Design Principles

1. Keep one simple external entry point.
   CLI, HTTP server, tests, and future integrations should call the same runtime boundary instead of rebuilding the orchestration path.

2. Make internal capabilities explicit ports.
   Intent recognition, retrieval, generation, policy checks, tool execution, evaluation, and trace recording should each have a narrow interface.

3. Prefer configuration for project-specific behavior.
   A course agent, HR assistant, marketing assistant, or research assistant should differ primarily by project files, not runtime forks.

4. Keep the first implementation dependency-light.
   The existing standard-library runtime is valuable. The first generalized runtime should preserve that property where possible.

5. Make observability a first-class output.
   A generic base must show which intent was selected, which workflow ran, which components contributed, which sources were used, and why a request was refused or downgraded.

6. Avoid premature graph complexity.
   The architecture should support future graph-style workflows, but the first implementation should use a readable pipeline abstraction.

## Chosen Interface Shape

Use a ports-and-pipeline runtime shape.

External callers use one primary entry point:

```python
runtime = AgentRuntime.from_project(project_root, config_path)
response = runtime.run(AgentRequest(message="...", history=[]))
```

Internally, the runtime routes requests through explicit ports:

```text
AgentRuntime
  -> IntentRouter
  -> WorkflowRegistry
  -> WorkflowPipeline
  -> ComponentRegistry
       -> Retriever
       -> PolicyGuard
       -> Generator
       -> ToolProvider
       -> Evaluator
       -> TraceSink
```

This is the recommended first-stage design because it gives a simple caller experience while keeping extension points visible and independently testable. It avoids the current linear script coupling without forcing a full graph engine before the repository needs one.

## Core Data Types

### `AgentRequest`

Represents one user turn.

Required fields:

- `message`: user input.
- `history`: previous user/assistant turns.
- `metadata`: optional caller metadata, such as channel, locale, user role, or project-specific flags.

### `AgentResponse`

Represents a completed runtime result.

Required fields:

- `answer`: final text shown to the user.
- `mode`: such as `extractive`, `model`, `tool`, `refusal`, or `error`.
- `intent`: selected intent id.
- `workflow`: selected workflow id.
- `sources`: source chunks or tool evidence shown to the user.
- `trace`: structured runtime trace for maintainers.

### `AgentTrace`

Records runtime decisions.

Recommended fields:

- selected intent and confidence.
- selected workflow.
- component steps executed.
- retrieval query and top source ids.
- policy decisions.
- tool calls and sanitized outputs.
- model configuration name, without secrets.
- errors or fallbacks.

### `RuntimeContext`

Shared per-request context passed between components.

Recommended fields:

- loaded project settings.
- component registry.
- request metadata.
- mutable state produced by workflow steps.
- trace collector.

## Intent Recognition

Intent recognition should move from a Markdown-only table to executable project configuration.

Add a project-level intent config, for example:

```toml
[[intents]]
id = "knowledge_qa"
description = "Answer source-backed questions from the project knowledge base."
examples = ["这门课什么时候上课？", "迟交政策是什么？"]
keywords = ["时间", "地点", "政策", "要求"]
workflow = "rag_qa"
risk_level = "medium"
knowledge_scopes = ["current", "stable", "policy"]

[[intents]]
id = "complete_submission_request"
description = "Requests to produce a complete assignment, paper, report, or directly submittable work."
examples = ["直接帮我写完整论文", "帮我写完作业"]
keywords = ["完整论文", "完整作业", "直接提交", "代写"]
workflow = "refusal_with_guidance"
risk_level = "high"
policy = "academic_integrity"
```

The first implementation can use deterministic matching:

1. exact keyword and phrase matching.
2. example similarity using the existing lexical term logic.
3. fallback to a default intent.

Later implementations can add model-based classification or embeddings behind the same `IntentRouter` port.

The existing `agent/intent-map.md` can remain as human-readable documentation, but the runtime should prefer the structured config when present.

## Workflow Model

A workflow is an ordered pipeline of named steps. Each step reads and writes `RuntimeContext`.

Example workflow config:

```toml
[[workflows]]
id = "rag_qa"
steps = [
  "prepare_retrieval_query",
  "retrieve",
  "apply_policy",
  "generate_answer",
  "attach_sources",
  "record_trace"
]

[[workflows]]
id = "refusal_with_guidance"
steps = [
  "apply_policy",
  "build_refusal",
  "record_trace"
]
```

The runtime should ship with a small set of built-in workflows:

- `rag_qa`: the current source-backed question-answer path.
- `retrieval_debug`: retrieve only and return ranked chunks.
- `refusal_with_guidance`: refuse unsafe or out-of-bound requests while offering allowed alternatives.
- `regression`: run the same runtime path and emit evidence records.

Future workflows can add tool use, multi-step planning, form filling, report drafting, or human approval.

## Component Ports

### `IntentRouter`

Responsibility:

- turn an `AgentRequest` into an intent decision.

Minimal interface:

```python
class IntentRouter:
    def route(self, request: AgentRequest, context: RuntimeContext) -> IntentDecision:
        ...
```

### `Retriever`

Responsibility:

- retrieve source evidence for a query and optional knowledge scopes.

Minimal interface:

```python
class Retriever:
    def retrieve(self, query: str, context: RuntimeContext, top_k: int) -> list[SourceChunk]:
        ...
```

The current lexical retriever becomes the first adapter.

### `PolicyGuard`

Responsibility:

- enforce configured boundaries before and after retrieval or generation.

Minimal interface:

```python
class PolicyGuard:
    def evaluate(self, context: RuntimeContext) -> PolicyDecision:
        ...
```

The first policies should cover:

- no basis in knowledge base.
- complete submission or代写 requests.
- high-risk facts require source-backed answers.
- private, maintenance, or pre-ingestion content must not be exposed.

### `Generator`

Responsibility:

- produce the final answer using either extractive fallback or an LLM.

Minimal interface:

```python
class Generator:
    def generate(self, context: RuntimeContext) -> GeneratedAnswer:
        ...
```

The current `answer_question`, `build_messages`, and extractive fallback become the first generator adapter.

### `ToolProvider`

Responsibility:

- expose optional tools to workflows.

Minimal interface:

```python
class ToolProvider:
    def call(self, tool_name: str, arguments: dict[str, object], context: RuntimeContext) -> ToolResult:
        ...
```

The first milestone does not need many tools. It only needs the port and a no-op or disabled provider so workflows can grow without rewiring the runtime.

### `Evaluator`

Responsibility:

- inspect responses for regression evidence and quality checks.

Minimal interface:

```python
class Evaluator:
    def evaluate(self, request: AgentRequest, response: AgentResponse, context: RuntimeContext) -> EvaluationResult:
        ...
```

The current regression JSONL writer can stay simple while using the same runtime output.

## Project Configuration

The project template should grow from document-only guidance into executable runtime configuration.

Recommended files:

```text
agent/
  system-prompt.md
  answer-policies.md
  intent-map.md
  intents.toml
  workflows.toml
  policies.toml
  tools.toml

runtime.toml
```

The top-level runtime config should keep project-specific paths and default component choices:

```toml
[project]
prompt_path = "agent/system-prompt.md"
knowledge_root = "knowledge_base"
manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"
index_path = ".local_agent/index.json"

[runtime]
default_intent = "knowledge_qa"
default_workflow = "rag_qa"
intent_config = "agent/intents.toml"
workflow_config = "agent/workflows.toml"
policy_config = "agent/policies.toml"
tool_config = "agent/tools.toml"

[retrieval]
provider = "lexical"
chunk_size = 1200
chunk_overlap = 160
top_k = 5

[generation]
provider = "openai_compatible"
fallback = "extractive"
```

The existing `runtime/local_rag_agent/examples/r-course-agent.toml` can continue to work. If a project lacks structured intent or workflow config, the runtime should default to the current RAG behavior.

## Data Flow

### Chat Request

```text
AgentRequest
  -> load runtime context
  -> IntentRouter.route()
  -> WorkflowRegistry.select(intent.workflow)
  -> WorkflowPipeline.run()
       -> prepare retrieval query
       -> Retriever.retrieve()
       -> PolicyGuard.evaluate()
       -> Generator.generate()
       -> attach sources
       -> TraceSink.record()
  -> AgentResponse
```

### Ingest

Ingest remains a separate command because it builds durable retrieval state:

```text
project config
  -> manifest entries
  -> knowledge files
  -> chunker
  -> index store
```

However, ingest should use component ports too:

- `ManifestProvider`
- `Chunker`
- `IndexStore`

This keeps future embedding indexes or database stores replaceable.

### Regression

Regression should call `AgentRuntime.run()` for each test question and save:

- request.
- answer.
- intent.
- workflow.
- sources.
- trace summary.
- optional evaluation result.

This makes regression evidence cover the same path users hit through CLI or HTTP.

## Error Handling

The runtime should use structured failures inside `AgentResponse` where possible, and hard failures only when the project cannot run.

Hard failures:

- project root missing.
- config missing or invalid.
- configured path escapes project root.
- manifest has no ingestible knowledge files during ingest.

Response-level failures:

- index missing during chat.
- model credentials missing when extractive fallback is disabled.
- tool disabled or unavailable.
- policy refusal.
- no source-backed answer for a high-risk question.

Every response-level failure should include trace data.

## Backward Compatibility

The existing commands should continue to work:

```text
ingest
retrieve
chat
serve
regression
demo-check
```

Their internals should gradually move behind `AgentRuntime`:

- `chat` calls `AgentRuntime.run()`.
- `retrieve` calls the configured retriever through the runtime registry.
- `regression` calls the same runtime as chat.
- `server` uses the same request and response data classes.

Existing simple TOML configs should keep working by creating implicit defaults:

- default intent: `knowledge_qa`.
- default workflow: `rag_qa`.
- default retriever: lexical.
- default generator: OpenAI-compatible with extractive fallback.

## Implementation Phases

### Phase 1: Runtime Types And Orchestrator

Add the core data types and a simple `AgentRuntime` that preserves current behavior:

- `AgentRequest`
- `AgentResponse`
- `AgentTrace`
- `RuntimeContext`
- `AgentRuntime.run()`

Move chat orchestration out of `cli.py` into the runtime boundary while keeping CLI behavior unchanged.

### Phase 2: Intent Configuration

Add structured intent config loading:

- parse `agent/intents.toml` when present.
- route with deterministic keyword/example matching.
- fall back to `knowledge_qa`.
- include selected intent in responses and regression records.

### Phase 3: Workflow Pipeline

Add a minimal workflow registry:

- built-in `rag_qa`.
- built-in `retrieval_debug`.
- built-in `refusal_with_guidance`.
- workflow trace step recording.

Keep implementation simple: ordered Python callables, not a graph engine.

### Phase 4: Policy And Tool Ports

Add first-class ports:

- `PolicyGuard`
- `ToolProvider`

Move complete-submission refusal and no-evidence responses out of ad hoc answer code into policy/generator boundaries.

### Phase 5: Template Upgrade

Update `templates/agent-project/` with:

- `runtime.toml`
- `agent/intents.toml`
- `agent/workflows.toml`
- `agent/policies.toml`
- `agent/tools.toml`

Keep the Markdown files as maintainer-readable documentation, but make TOML the runtime source of truth.

### Phase 6: Regression And Observability

Extend regression output with:

- selected intent.
- workflow.
- policy result.
- trace summary.

Add a demo-check mode that reports whether required configs and workflow pieces are present.

## Testing Strategy

Each phase needs tests that prove low coupling rather than only preserving behavior.

Required test categories:

- Runtime request/response shape is stable.
- CLI `chat` and HTTP `/api/chat` use the same runtime path.
- Missing structured configs fall back to existing RAG behavior.
- Intent config can add a new intent without editing runtime code.
- Workflow config can select a different built-in workflow.
- Policy refusal is traceable and source-safe.
- Regression records include intent, workflow, sources, and mode.
- Existing ingest/retrieve/chat/regression smoke path remains green.

## Non-Goals For The First Generalization

Do not build these in the first implementation pass:

- a full graph engine.
- a visual workflow editor.
- user accounts or permissions.
- production deployment automation.
- a large plugin marketplace.
- mandatory embedding/vector database dependencies.

The first win is a flexible, inspectable runtime base that can grow without core rewrites.

## Success Criteria

The first generalized version is successful when:

1. Existing local RAG commands still run.
2. `chat` goes through `AgentRuntime.run()`.
3. A project can add or modify an intent in configuration without changing runtime code.
4. The selected intent and workflow appear in the response trace and regression output.
5. The default RAG workflow is represented as a workflow, not as CLI-only procedural code.
6. The template includes structured runtime, intent, workflow, policy, and tool config stubs.
