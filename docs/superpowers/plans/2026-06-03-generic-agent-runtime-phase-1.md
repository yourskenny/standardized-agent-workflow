# Generic Agent Runtime Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stable `AgentRuntime.run()` boundary with request, response, and trace types while preserving the current local RAG CLI/server behavior.

**Architecture:** Introduce a small runtime orchestration layer over the existing retrieval and answer functions. `cli.py` and `server.py` should call this runtime path for chat, while ingest and retrieve commands remain compatible. This phase does not add configurable intents or workflow files yet; it creates default `knowledge_qa` and `rag_qa` metadata so later phases have a place to attach configuration.

**Tech Stack:** Python standard library, dataclasses, unittest, existing `local_rag_agent` package.

---

## File Structure

- Create `runtime/local_rag_agent/local_rag_agent/types.py`
  - Owns runtime dataclasses: `AgentRequest`, `AgentResponse`, `AgentTrace`, and `SourceReference`.
- Create `runtime/local_rag_agent/local_rag_agent/runtime.py`
  - Owns `AgentRuntime`, the default intent/workflow constants, and chat orchestration.
- Modify `runtime/local_rag_agent/local_rag_agent/cli.py`
  - Replace `chat_question()` internals with `AgentRuntime.run()`.
  - Keep public CLI output unchanged.
- Modify `runtime/local_rag_agent/local_rag_agent/server.py`
  - Keep calling `chat_question()` so server behavior uses the runtime indirectly.
  - No direct server rewrite in this phase.
- Modify `runtime/local_rag_agent/local_rag_agent/regression.py`
  - Record `intent`, `workflow`, and `trace` when present in runtime responses.
- Modify `runtime/local_rag_agent/tests/test_local_rag_agent.py`
  - Add tests for runtime request/response shape, CLI path compatibility, and regression metadata.

## Task 1: Runtime Types

**Files:**
- Create: `runtime/local_rag_agent/local_rag_agent/types.py`
- Test: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] **Step 1: Write the failing test**

Add imports near the top of `runtime/local_rag_agent/tests/test_local_rag_agent.py`:

```python
from local_rag_agent.types import AgentRequest, AgentResponse, AgentTrace, SourceReference
```

Add this test class after `AgentTests`:

```python
class RuntimeTypeTests(unittest.TestCase):
    def test_agent_response_converts_to_legacy_payload(self):
        trace = AgentTrace(intent="knowledge_qa", workflow="rag_qa")
        trace.add_step("retrieve", {"top_k": 1})
        response = AgentResponse(
            answer="上课时间是星期三上午。",
            mode="extractive",
            intent="knowledge_qa",
            workflow="rag_qa",
            sources=[
                SourceReference(
                    source="course.md",
                    title="课程事实",
                    chunk_id="course.md#0",
                    score=3.0,
                    content="上课时间是星期三上午。",
                )
            ],
            trace=trace,
        )

        payload = response.to_dict()

        self.assertEqual(payload["answer"], "上课时间是星期三上午。")
        self.assertEqual(payload["mode"], "extractive")
        self.assertEqual(payload["intent"], "knowledge_qa")
        self.assertEqual(payload["workflow"], "rag_qa")
        self.assertEqual(payload["sources"][0]["source"], "course.md")
        self.assertEqual(payload["trace"]["steps"][0]["name"], "retrieve")

    def test_agent_request_keeps_history_and_metadata_defaults(self):
        request = AgentRequest(message="那地点呢？")

        self.assertEqual(request.message, "那地点呢？")
        self.assertEqual(request.history, [])
        self.assertEqual(request.metadata, {})
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.RuntimeTypeTests -v
```

Expected: import failure for `local_rag_agent.types`.

- [ ] **Step 3: Implement minimal runtime types**

Create `runtime/local_rag_agent/local_rag_agent/types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentRequest:
    message: str
    history: list[dict[str, object]] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class SourceReference:
    source: str = ""
    title: str = ""
    chunk_id: str = ""
    score: float | int = 0
    content: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "SourceReference":
        return cls(
            source=str(value.get("source", "")),
            title=str(value.get("title", "")),
            chunk_id=str(value.get("chunk_id", "")),
            score=value.get("score", 0) if isinstance(value.get("score", 0), (int, float)) else 0,
            content=str(value.get("content", "")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "title": self.title,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "content": self.content,
        }


@dataclass
class AgentTrace:
    intent: str
    workflow: str
    steps: list[dict[str, object]] = field(default_factory=list)

    def add_step(self, name: str, detail: dict[str, object] | None = None) -> None:
        self.steps.append({"name": name, "detail": detail or {}})

    def to_dict(self) -> dict[str, object]:
        return {
            "intent": self.intent,
            "workflow": self.workflow,
            "steps": self.steps,
        }


@dataclass
class AgentResponse:
    answer: str
    mode: str
    intent: str
    workflow: str
    sources: list[SourceReference] = field(default_factory=list)
    trace: AgentTrace | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "mode": self.mode,
            "intent": self.intent,
            "workflow": self.workflow,
            "sources": [source.to_dict() for source in self.sources],
            "trace": self.trace.to_dict() if self.trace else {},
            "metadata": self.metadata,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.RuntimeTypeTests -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Run full suite**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest discover -s runtime\local_rag_agent\tests -v
```

Expected: all existing tests still pass.

## Task 2: AgentRuntime Orchestrator

**Files:**
- Create: `runtime/local_rag_agent/local_rag_agent/runtime.py`
- Modify: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] **Step 1: Write the failing test**

Add import:

```python
from local_rag_agent.runtime import AgentRuntime
```

Add this test to `AgentTests`:

```python
    def test_runtime_run_preserves_current_rag_answer_shape(self):
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
            runtime = AgentRuntime(settings)

            response = runtime.run(AgentRequest(message="上课时间？"))
            payload = response.to_dict()

            self.assertEqual(payload["intent"], "knowledge_qa")
            self.assertEqual(payload["workflow"], "rag_qa")
            self.assertEqual(payload["mode"], "extractive")
            self.assertIn("上课时间是星期三上午", payload["answer"])
            self.assertEqual(payload["sources"][0]["source"], "course.md")
            self.assertEqual(payload["trace"]["steps"][0]["name"], "build_retrieval_query")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.AgentTests.test_runtime_run_preserves_current_rag_answer_shape -v
```

Expected: import failure for `local_rag_agent.runtime`.

- [ ] **Step 3: Implement `AgentRuntime`**

Create `runtime/local_rag_agent/local_rag_agent/runtime.py`:

```python
from __future__ import annotations

from .agent import answer_question
from .config import Settings
from .index_store import read_index
from .llm import OpenAICompatibleClient
from .retrieval import rank_chunks
from .types import AgentRequest, AgentResponse, AgentTrace, SourceReference

DEFAULT_INTENT = "knowledge_qa"
DEFAULT_WORKFLOW = "rag_qa"


class AgentRuntime:
    def __init__(self, settings: Settings, model_client: object | None = None):
        self.settings = settings
        self.model_client = model_client

    def run(self, request: AgentRequest) -> AgentResponse:
        trace = AgentTrace(intent=DEFAULT_INTENT, workflow=DEFAULT_WORKFLOW)
        retrieval_query = self.build_retrieval_query(request.message, request.history)
        trace.add_step("build_retrieval_query", {"query": retrieval_query})

        payload = read_index(self.settings)
        chunks = payload.get("chunks", [])
        if not isinstance(chunks, list):
            raise ValueError(f"Invalid index format: {self.settings.index_path}")

        retrieved = rank_chunks(retrieval_query, chunks, self.settings.top_k)
        trace.add_step(
            "retrieve",
            {
                "top_k": self.settings.top_k,
                "source_count": len(retrieved),
                "top_source": str(retrieved[0].get("source", "")) if retrieved else "",
            },
        )

        client = self.model_client if self.model_client is not None else OpenAICompatibleClient.from_env()
        result = answer_question(self.settings, request.message, retrieved, client, history=request.history)
        trace.add_step("generate_answer", {"mode": str(result.get("mode", ""))})

        sources = [
            SourceReference.from_mapping(source)
            for source in result.get("sources", [])
            if isinstance(source, dict)
        ]
        return AgentResponse(
            answer=str(result.get("answer", "")),
            mode=str(result.get("mode", "")),
            intent=DEFAULT_INTENT,
            workflow=DEFAULT_WORKFLOW,
            sources=sources,
            trace=trace,
        )

    @staticmethod
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

- [ ] **Step 4: Run runtime test**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.AgentTests.test_runtime_run_preserves_current_rag_answer_shape -v
```

Expected: the test passes.

- [ ] **Step 5: Run full suite**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest discover -s runtime\local_rag_agent\tests -v
```

Expected: all tests pass.

## Task 3: Route CLI Chat Through AgentRuntime

**Files:**
- Modify: `runtime/local_rag_agent/local_rag_agent/cli.py`
- Modify: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] **Step 1: Write failing compatibility test**

Add this test to `CliWorkflowTests`:

```python
    def test_chat_question_returns_runtime_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("prompt", encoding="utf-8")
            settings = Settings(
                project_root=root,
                prompt_path=prompt,
                manifest_path=root / "manifest.md",
                knowledge_root=root / "knowledge_base",
                index_path=root / ".local_rag_agent" / "index.json",
            )
            settings.index_path.parent.mkdir()
            settings.index_path.write_text(
                '{"chunks":[{"chunk_id":"course.md#0","source":"course.md","title":"课程事实","content":"答：星期三上午。"}]}',
                encoding="utf-8",
            )

            response = chat_question(settings, "上课时间？", model_client=None)

            self.assertEqual(response["intent"], "knowledge_qa")
            self.assertEqual(response["workflow"], "rag_qa")
            self.assertEqual(response["trace"]["steps"][0]["name"], "build_retrieval_query")
            self.assertIn("星期三上午", response["answer"])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.CliWorkflowTests.test_chat_question_returns_runtime_metadata -v
```

Expected: failure because `chat_question()` does not yet include `intent`, `workflow`, and `trace`.

- [ ] **Step 3: Modify `cli.py` imports**

In `runtime/local_rag_agent/local_rag_agent/cli.py`, add:

```python
from .runtime import AgentRuntime
from .types import AgentRequest
```

Remove direct imports that become unused after Step 4:

```python
from .agent import answer_question
from .llm import OpenAICompatibleClient
```

Keep `rank_chunks` because `retrieve_question()` still uses it.

- [ ] **Step 4: Modify `chat_question()`**

Replace `chat_question()` with:

```python
def chat_question(
    settings: Settings,
    question: str,
    model_client: object | None = None,
    history: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    runtime = AgentRuntime(settings, model_client=model_client)
    request = AgentRequest(message=question, history=history or [])
    return runtime.run(request).to_dict()
```

- [ ] **Step 5: Keep `build_retrieval_query()` as compatibility wrapper**

Replace `build_retrieval_query()` with:

```python
def build_retrieval_query(question: str, history: list[dict[str, object]] | None = None) -> str:
    return AgentRuntime.build_retrieval_query(question, history)
```

- [ ] **Step 6: Run compatibility test**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.CliWorkflowTests.test_chat_question_returns_runtime_metadata -v
```

Expected: the test passes.

- [ ] **Step 7: Run existing CLI retrieval-query test**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.CliWorkflowTests.test_build_retrieval_query_includes_recent_user_turns_for_followups -v
```

Expected: the test still passes.

## Task 4: Regression Records Runtime Metadata

**Files:**
- Modify: `runtime/local_rag_agent/local_rag_agent/regression.py`
- Modify: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] **Step 1: Write failing regression test**

Add this test to `RegressionTests`:

```python
    def test_run_regression_records_runtime_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            questions = root / "questions.md"
            output = root / "results.jsonl"
            questions.write_text(
                "| 编号 | 问题 | 预期要点 |\n"
                "| --- | --- | --- |\n"
                "| C01 | 上课时间？ | 星期三 |\n",
                encoding="utf-8",
            )

            count = run_regression(
                questions,
                output,
                lambda question: {
                    "answer": "星期三上午。",
                    "sources": [],
                    "mode": "extractive",
                    "intent": "knowledge_qa",
                    "workflow": "rag_qa",
                    "trace": {"steps": [{"name": "retrieve", "detail": {"source_count": 1}}]},
                },
            )

            record = json.loads(output.read_text(encoding="utf-8").strip())

            self.assertEqual(count, 1)
            self.assertEqual(record["intent"], "knowledge_qa")
            self.assertEqual(record["workflow"], "rag_qa")
            self.assertEqual(record["trace"]["steps"][0]["name"], "retrieve")
```

Add `import json` only if it is not already present. It is already present in the current test file, so no import change should be needed.

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.RegressionTests.test_run_regression_records_runtime_metadata -v
```

Expected: failure because regression records do not include `intent`, `workflow`, or `trace`.

- [ ] **Step 3: Modify regression record**

In `runtime/local_rag_agent/local_rag_agent/regression.py`, update the `record` dict inside `run_regression()`:

```python
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "id": item["id"],
                "question": item["question"],
                "expected": item["expected"],
                "answer": response.get("answer", ""),
                "sources": response.get("sources", []),
                "mode": response.get("mode", ""),
                "intent": response.get("intent", ""),
                "workflow": response.get("workflow", ""),
                "trace": response.get("trace", {}),
            }
```

- [ ] **Step 4: Run regression metadata test**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.RegressionTests.test_run_regression_records_runtime_metadata -v
```

Expected: the test passes.

- [ ] **Step 5: Run full suite**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest discover -s runtime\local_rag_agent\tests -v
```

Expected: all tests pass.

## Task 5: Runtime Smoke Verification

**Files:**
- No source files required.

- [ ] **Step 1: Run full unit suite**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest discover -s runtime\local_rag_agent\tests
```

Expected: all tests pass.

- [ ] **Step 2: Create a temporary template project**

Run:

```powershell
$repo = "C:\coding\standardized-agent-workflow"
$stamp = Get-Date -Format 'yyyyMMddHHmmss'
$smokeRoot = Join-Path $repo "tmp\phase1-$stamp"
$project = Join-Path $smokeRoot "agent-project"
$config = Join-Path $smokeRoot "agent.toml"
Copy-Item -Recurse -Path (Join-Path $repo "templates\agent-project") -Destination $project
@'
[project]
prompt_path = "agent/system-prompt.md"
knowledge_root = "knowledge_base"
manifest_path = "knowledge_base/_manifests/current-upload-manifest.md"
index_path = ".local_rag_agent/index.json"

[retrieval]
chunk_size = 1200
chunk_overlap = 160
top_k = 5

[regression]
output_dir = ".local_rag_agent/regression"
'@ | Set-Content -Path $config -Encoding UTF8
```

Expected: no command error.

- [ ] **Step 3: Run ingest**

Run:

```powershell
$env:PYTHONPATH = Join-Path $repo "runtime\local_rag_agent"
python -m local_rag_agent ingest --project $project --config $config
```

Expected JSON includes:

```json
{
  "file_count": 1,
  "chunk_count": 4
}
```

- [ ] **Step 4: Run chat and confirm runtime metadata is preserved internally**

Run:

```powershell
python -m local_rag_agent chat --project $project --config $config "入库前处理区里的原始资料可以直接给用户吗？"
```

Expected: output still prints the answer and sources in the old CLI shape. It does not need to print `intent` or `trace` for CLI users in Phase 1.

- [ ] **Step 5: Run regression and inspect JSONL metadata**

Run:

```powershell
python -m local_rag_agent regression --project $project --config $config --questions (Join-Path $project "examples\core-regression-questions.md")
$recordPath = Join-Path $project ".local_rag_agent\regression\core-regression-questions.jsonl"
Get-Content -Path $recordPath -Encoding UTF8 -TotalCount 1
```

Expected: the first JSONL record includes keys:

```json
{
  "intent": "knowledge_qa",
  "workflow": "rag_qa",
  "trace": {}
}
```

The `trace` object should contain a `steps` array when the answer path succeeds.

## Completion Criteria

Phase 1 is complete when:

- `AgentRequest`, `AgentResponse`, `AgentTrace`, and `SourceReference` exist and are tested.
- `AgentRuntime.run()` preserves the current RAG answer behavior.
- `chat_question()` uses `AgentRuntime.run()`.
- Existing CLI chat output remains compatible.
- Regression JSONL records include intent, workflow, and trace.
- Full unit suite passes.
- Template-project smoke path passes ingest, chat, and regression.
