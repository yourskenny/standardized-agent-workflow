# Generic Agent Runtime Phase 3 Workflow Pipeline Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert workflow from response metadata into an actual built-in pipeline dispatcher while preserving current RAG behavior.

**Architecture:** Add a focused `workflow.py` module with `WorkflowContext`, `WorkflowPipeline`, and `WorkflowRegistry`. `AgentRuntime.run()` routes intent first, then dispatches to a named built-in workflow. Phase 3 includes built-in `rag_qa`, `retrieval_debug`, and `refusal_with_guidance`; configurable workflow TOML is intentionally deferred.

**Tech Stack:** Python standard library, dataclasses, unittest, existing `local_rag_agent` modules.

---

## File Structure

- Create `runtime/local_rag_agent/local_rag_agent/workflow.py`
  - Owns workflow context, registry, built-in workflow pipelines, and pipeline steps.
- Modify `runtime/local_rag_agent/local_rag_agent/runtime.py`
  - Delegate workflow execution to `WorkflowRegistry`.
  - Keep `AgentRuntime.build_retrieval_query()` as a compatibility wrapper.
- Modify `runtime/local_rag_agent/tests/test_local_rag_agent.py`
  - Add workflow registry tests.
  - Update runtime tests to assert workflow dispatch behavior.

## Task 1: Workflow Registry And Pipeline Core

**Files:**
- Create: `runtime/local_rag_agent/local_rag_agent/workflow.py`
- Modify: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] **Step 1: Write failing workflow registry tests**

Add import:

```python
from local_rag_agent.workflow import WorkflowContext, WorkflowRegistry, build_retrieval_query
```

Add this test class after `IntentConfigTests`:

```python
class WorkflowPipelineTests(unittest.TestCase):
    def test_registry_contains_required_builtin_workflows(self):
        registry = WorkflowRegistry.builtins()

        self.assertTrue(registry.has("rag_qa"))
        self.assertTrue(registry.has("retrieval_debug"))
        self.assertTrue(registry.has("refusal_with_guidance"))

    def test_build_retrieval_query_keeps_recent_user_turns(self):
        query = build_retrieval_query(
            "那地点呢？",
            [
                {"role": "user", "content": "这门课的上课时间是什么？"},
                {"role": "assistant", "content": "星期三上午。"},
                {"role": "user", "content": "老师什么时候答疑？"},
            ],
        )

        self.assertIn("这门课的上课时间是什么？", query)
        self.assertIn("老师什么时候答疑？", query)
        self.assertIn("那地点呢？", query)
        self.assertNotIn("星期三上午", query)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.WorkflowPipelineTests -v
```

Expected: import failure for `local_rag_agent.workflow`.

- [ ] **Step 3: Implement minimal workflow module**

Create `runtime/local_rag_agent/local_rag_agent/workflow.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .config import Settings
from .intent import IntentDecision
from .types import AgentRequest, AgentResponse, AgentTrace

WorkflowStep = Callable[["WorkflowContext"], None]


@dataclass
class WorkflowContext:
    settings: Settings
    request: AgentRequest
    intent_decision: IntentDecision
    trace: AgentTrace
    model_client: object | None = None
    retrieval_query: str = ""
    retrieved_chunks: list[dict[str, object]] = field(default_factory=list)
    result: dict[str, object] = field(default_factory=dict)
    response: AgentResponse | None = None


class WorkflowPipeline:
    def __init__(self, workflow_id: str, steps: list[WorkflowStep]):
        self.workflow_id = workflow_id
        self.steps = steps

    def run(self, context: WorkflowContext) -> AgentResponse:
        context.trace.add_step("start_workflow", {"workflow": self.workflow_id})
        for step in self.steps:
            step(context)
        if context.response is None:
            raise RuntimeError(f"Workflow did not produce a response: {self.workflow_id}")
        return context.response


class WorkflowRegistry:
    def __init__(self, workflows: dict[str, WorkflowPipeline]):
        self.workflows = workflows

    @classmethod
    def builtins(cls) -> "WorkflowRegistry":
        return cls(
            {
                "rag_qa": WorkflowPipeline("rag_qa", []),
                "retrieval_debug": WorkflowPipeline("retrieval_debug", []),
                "refusal_with_guidance": WorkflowPipeline("refusal_with_guidance", []),
            }
        )

    def has(self, workflow_id: str) -> bool:
        return workflow_id in self.workflows

    def get(self, workflow_id: str) -> WorkflowPipeline:
        if workflow_id not in self.workflows:
            return self.workflows["rag_qa"]
        return self.workflows[workflow_id]


def build_retrieval_query(question: str, history: list[dict[str, object]] | None = None) -> str:
    user_turns: list[str] = []
    for item in history or []:
        if item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            user_turns.append(content.strip())
    recent_context = "\n".join(user_turns[-3:])
    return f"{recent_context}\n{question}".strip()
```

- [ ] **Step 4: Run workflow core tests**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.WorkflowPipelineTests -v
```

Expected: tests pass.

## Task 2: Built-In `rag_qa` Workflow Produces Current Answer Shape

**Files:**
- Modify: `runtime/local_rag_agent/local_rag_agent/workflow.py`
- Modify: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] **Step 1: Write failing `rag_qa` workflow test**

Add this test to `WorkflowPipelineTests`:

```python
    def test_rag_qa_workflow_produces_answer_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("系统提示词", encoding="utf-8")
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
            )
            settings.index_path.parent.mkdir()
            settings.index_path.write_text(
                '{"chunks":[{"chunk_id":"course.md#0","source":"course.md","title":"课程事实","content":"答：上课时间是星期三上午。"}]}',
                encoding="utf-8",
            )
            decision = IntentRouter([]).route("上课时间？")
            trace = AgentTrace(intent=decision.intent.id, workflow=decision.intent.workflow)
            context = WorkflowContext(settings, AgentRequest("上课时间？"), decision, trace)

            response = WorkflowRegistry.builtins().get("rag_qa").run(context)
            payload = response.to_dict()

            self.assertEqual(payload["workflow"], "rag_qa")
            self.assertIn("上课时间是星期三上午", payload["answer"])
            self.assertEqual(payload["sources"][0]["source"], "course.md")
            self.assertIn("run_retrieval", [step["name"] for step in payload["trace"]["steps"]])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.WorkflowPipelineTests.test_rag_qa_workflow_produces_answer_response -v
```

Expected: workflow has empty steps and does not produce response.

- [ ] **Step 3: Implement built-in workflow steps**

In `workflow.py`, import:

```python
from .agent import answer_question, build_extractive_answer
from .index_store import read_index
from .llm import OpenAICompatibleClient
from .retrieval import rank_chunks
from .types import SourceReference
```

Add step functions:

```python
def prepare_retrieval_query(context: WorkflowContext) -> None:
    context.retrieval_query = build_retrieval_query(context.request.message, context.request.history)
    context.trace.add_step("prepare_retrieval_query", {"query": context.retrieval_query})


def run_retrieval(context: WorkflowContext) -> None:
    payload = read_index(context.settings)
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError(f"Invalid index format: {context.settings.index_path}")
    context.retrieved_chunks = rank_chunks(context.retrieval_query, chunks, context.settings.top_k)
    context.trace.add_step(
        "run_retrieval",
        {
            "top_k": context.settings.top_k,
            "source_count": len(context.retrieved_chunks),
            "top_source": str(context.retrieved_chunks[0].get("source", "")) if context.retrieved_chunks else "",
        },
    )


def generate_answer(context: WorkflowContext) -> None:
    client = context.model_client if context.model_client is not None else OpenAICompatibleClient.from_env()
    context.result = answer_question(
        context.settings,
        context.request.message,
        context.retrieved_chunks,
        client,
        history=context.request.history,
    )
    context.trace.add_step("generate_answer", {"mode": str(context.result.get("mode", ""))})


def build_response(context: WorkflowContext) -> None:
    sources = [
        SourceReference.from_mapping(source)
        for source in context.result.get("sources", [])
        if isinstance(source, dict)
    ]
    context.response = AgentResponse(
        answer=str(context.result.get("answer", "")),
        mode=str(context.result.get("mode", "")),
        intent=context.intent_decision.intent.id,
        workflow=context.intent_decision.intent.workflow,
        sources=sources,
        trace=context.trace,
    )
```

Update `WorkflowRegistry.builtins()`:

```python
"rag_qa": WorkflowPipeline("rag_qa", [prepare_retrieval_query, run_retrieval, generate_answer, build_response])
```

- [ ] **Step 4: Run workflow test and full suite**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.WorkflowPipelineTests.test_rag_qa_workflow_produces_answer_response -v
python -m unittest discover -s runtime\local_rag_agent\tests -v
```

Expected: all tests pass after updating trace assertions as needed.

## Task 3: AgentRuntime Delegates To WorkflowRegistry

**Files:**
- Modify: `runtime/local_rag_agent/local_rag_agent/runtime.py`
- Modify: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] **Step 1: Write failing runtime workflow test**

Add this assertion to `AgentTests.test_runtime_run_preserves_current_rag_answer_shape`:

```python
self.assertIn("start_workflow", [step["name"] for step in payload["trace"]["steps"]])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.AgentTests.test_runtime_run_preserves_current_rag_answer_shape -v
```

Expected: trace does not include `start_workflow`.

- [ ] **Step 3: Modify runtime to dispatch workflow**

In `runtime.py`, remove direct imports no longer needed:

```python
from .agent import answer_question
from .index_store import read_index
from .llm import OpenAICompatibleClient
from .retrieval import rank_chunks
from .types import SourceReference
```

Add:

```python
from .workflow import WorkflowContext, WorkflowRegistry, build_retrieval_query
```

In `__init__`, add:

```python
self.workflow_registry = WorkflowRegistry.builtins()
```

Replace the body after `route_intent` trace setup with:

```python
        workflow = self.workflow_registry.get(intent_decision.intent.workflow)
        context = WorkflowContext(
            settings=self.settings,
            request=request,
            intent_decision=intent_decision,
            trace=trace,
            model_client=self.model_client,
        )
        return workflow.run(context)
```

Update static wrapper:

```python
    @staticmethod
    def build_retrieval_query(question: str, history: list[dict[str, object]] | None = None) -> str:
        return build_retrieval_query(question, history)
```

- [ ] **Step 4: Run runtime test and full suite**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.AgentTests.test_runtime_run_preserves_current_rag_answer_shape -v
python -m unittest discover -s runtime\local_rag_agent\tests -v
```

Expected: all tests pass after updating old trace step assertions to account for `start_workflow`.

## Task 4: `retrieval_debug` Workflow

**Files:**
- Modify: `runtime/local_rag_agent/local_rag_agent/workflow.py`
- Modify: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] **Step 1: Write failing retrieval_debug runtime test**

Add this test to `AgentTests`:

```python
    def test_runtime_runs_retrieval_debug_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("系统提示词", encoding="utf-8")
            intent_config = root / "intents.toml"
            intent_config.write_text(
                '[[intents]]\n'
                'id = "debug_retrieval"\n'
                'workflow = "retrieval_debug"\n'
                'keywords = ["调试检索"]\n',
                encoding="utf-8",
            )
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
                intent_config_path=intent_config,
            )
            settings.index_path.parent.mkdir()
            settings.index_path.write_text(
                '{"chunks":[{"chunk_id":"course.md#0","source":"course.md","title":"课程事实","content":"上课时间是星期三上午。"}]}',
                encoding="utf-8",
            )

            response = AgentRuntime(settings).run(AgentRequest("调试检索：上课时间？")).to_dict()

            self.assertEqual(response["workflow"], "retrieval_debug")
            self.assertEqual(response["mode"], "retrieval_debug")
            self.assertIn("course.md", response["answer"])
            self.assertEqual(response["sources"][0]["source"], "course.md")
```

- [ ] **Step 2: Run test to verify it fails**

Expected: empty retrieval_debug workflow produces no response.

- [ ] **Step 3: Implement debug answer step**

Add:

```python
def build_retrieval_debug_response(context: WorkflowContext) -> None:
    lines = ["本地检索调试结果："]
    for index, chunk in enumerate(context.retrieved_chunks, start=1):
        lines.append(f"{index}. {chunk.get('source')} ({chunk.get('chunk_id')}, score={chunk.get('score', 0)})")
    context.response = AgentResponse(
        answer="\n".join(lines),
        mode="retrieval_debug",
        intent=context.intent_decision.intent.id,
        workflow=context.intent_decision.intent.workflow,
        sources=[SourceReference.from_mapping(chunk) for chunk in context.retrieved_chunks],
        trace=context.trace,
    )
```

Update registry:

```python
"retrieval_debug": WorkflowPipeline("retrieval_debug", [prepare_retrieval_query, run_retrieval, build_retrieval_debug_response])
```

- [ ] **Step 4: Run target and full suite**

Expected: target and full suite pass.

## Task 5: `refusal_with_guidance` Workflow

**Files:**
- Modify: `runtime/local_rag_agent/local_rag_agent/workflow.py`
- Modify: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] **Step 1: Write failing refusal runtime test**

Add this test to `AgentTests`:

```python
    def test_runtime_runs_refusal_workflow_without_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("系统提示词", encoding="utf-8")
            intent_config = root / "intents.toml"
            intent_config.write_text(
                '[[intents]]\n'
                'id = "complete_submission_request"\n'
                'workflow = "refusal_with_guidance"\n'
                'keywords = ["完整论文"]\n',
                encoding="utf-8",
            )
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "missing-index.json",
                intent_config_path=intent_config,
            )

            response = AgentRuntime(settings).run(AgentRequest("请直接帮我写完整论文。")).to_dict()

            self.assertEqual(response["workflow"], "refusal_with_guidance")
            self.assertEqual(response["mode"], "refusal")
            self.assertIn("不能直接替你完成", response["answer"])
            self.assertEqual(response["sources"], [])
```

- [ ] **Step 2: Run test to verify it fails**

Expected: refusal workflow is empty or tries to read missing index.

- [ ] **Step 3: Implement refusal response step**

Add:

```python
def build_refusal_response(context: WorkflowContext) -> None:
    answer = build_extractive_answer(context.request.message, [])
    context.trace.add_step("build_refusal", {"policy": context.intent_decision.intent.policy})
    context.response = AgentResponse(
        answer=answer,
        mode="refusal",
        intent=context.intent_decision.intent.id,
        workflow=context.intent_decision.intent.workflow,
        sources=[],
        trace=context.trace,
    )
```

Update registry:

```python
"refusal_with_guidance": WorkflowPipeline("refusal_with_guidance", [build_refusal_response])
```

- [ ] **Step 4: Run target and full suite**

Expected: target and full suite pass.

## Task 6: Smoke Verification

**Files:**
- No source edits.

- [ ] **Step 1: Run full unit suite**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest discover -s runtime\local_rag_agent\tests
```

Expected: all tests pass.

- [ ] **Step 2: Smoke a template project with workflow dispatch**

Create a temporary project from `templates/agent-project`, configure `[runtime].intent_config = "agent/intents.toml"`, run ingest and regression.

Inspect the first JSONL record and confirm:

- `trace.steps` includes `route_intent`.
- `trace.steps` includes `start_workflow`.
- `trace.steps` includes `run_retrieval`.

## Completion Criteria

Phase 3 is complete when:

- `workflow.py` owns built-in workflow registry and pipeline execution.
- `AgentRuntime.run()` dispatches by selected workflow id instead of hard-coding the RAG chain.
- `rag_qa` preserves current chat behavior.
- `retrieval_debug` returns ranked source evidence without generation.
- `refusal_with_guidance` can return a refusal without requiring an index.
- Full unit suite and workflow smoke path pass.
