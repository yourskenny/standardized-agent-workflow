# Decisions

## 2026-06-03: Generic Agent Runtime Uses Ports And Pipeline First

Decision: the first generalization of `runtime/local_rag_agent` will use a ports-and-pipeline runtime shape.

The public caller interface should remain simple:

```python
runtime = AgentRuntime.from_project(project_root, config_path)
response = runtime.run(AgentRequest(message="...", history=[]))
```

Internally, the runtime should route through explicit ports:

```text
IntentRouter
WorkflowRegistry
WorkflowPipeline
Retriever
PolicyGuard
Generator
ToolProvider
Evaluator
TraceSink
```

Reason: this shape covers the user's required directions together: a generic agent scheduling base, RAG as the first built-in workflow, and structured project templates for intents, workflows, tools, and policies. It is more flexible than the current linear RAG path, but avoids prematurely committing the project to a full graph engine before the component boundaries are stable.
