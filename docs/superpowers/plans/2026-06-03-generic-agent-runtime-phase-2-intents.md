# Generic Agent Runtime Phase 2 Intent Configuration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add structured project intent configuration so new intents can be added through `agent/intents.toml` without changing runtime orchestration code.

**Architecture:** Extend settings with an optional `intent_config_path`, add deterministic intent loading/routing in a focused `intent.py` module, then have `AgentRuntime.run()` select intent/workflow from that router. Missing config falls back to the Phase 1 defaults: `knowledge_qa` and `rag_qa`.

**Tech Stack:** Python standard library, `tomllib`, dataclasses, unittest, existing lexical term behavior.

---

## File Structure

- Create `runtime/local_rag_agent/local_rag_agent/intent.py`
  - Owns `IntentDefinition`, `IntentDecision`, `IntentRouter`, and `load_intents()`.
- Modify `runtime/local_rag_agent/local_rag_agent/config.py`
  - Add optional `intent_config_path` loaded from `[runtime].intent_config`.
- Modify `runtime/local_rag_agent/local_rag_agent/runtime.py`
  - Route each request through `IntentRouter`.
  - Use selected `intent.id` and `intent.workflow` in `AgentTrace` and `AgentResponse`.
- Modify `runtime/local_rag_agent/tests/test_local_rag_agent.py`
  - Add tests for settings path loading, intent TOML parsing, deterministic routing, runtime fallback, and configured runtime intent selection.
- Create `templates/agent-project/agent/intents.toml`
  - Provide maintainable default intent examples for new projects.
- Modify `templates/agent-project/README.md`
  - Mention structured intent configuration.

## Task 1: Settings Reads Optional Intent Config Path

**Files:**
- Modify: `runtime/local_rag_agent/local_rag_agent/config.py`
- Modify: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] **Step 1: Write failing settings test**

Add this test to `ConfigAndManifestTests`:

```python
    def test_load_config_resolves_optional_intent_config_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent").mkdir()
            (root / "agent" / "system-prompt.md").write_text("prompt", encoding="utf-8")
            (root / "agent" / "intents.toml").write_text("[[intents]]\nid = \"knowledge_qa\"\n", encoding="utf-8")
            (root / "knowledge_base").mkdir()
            (root / "manifest.md").write_text("- `knowledge_base/`\n", encoding="utf-8")
            config_path = root / "agent.toml"
            config_path.write_text(
                '[project]\n'
                'prompt_path = "agent/system-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "manifest.md"\n'
                '[runtime]\n'
                'intent_config = "agent/intents.toml"\n',
                encoding="utf-8",
            )

            settings = load_settings(root, config_path)

            self.assertEqual(settings.intent_config_path, root.resolve() / "agent" / "intents.toml")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.ConfigAndManifestTests.test_load_config_resolves_optional_intent_config_path -v
```

Expected: `AttributeError` for `intent_config_path`.

- [ ] **Step 3: Add settings field**

In `Settings`, add:

```python
    intent_config_path: Path | None = None
```

In `load_settings()`, read the runtime section:

```python
    runtime = data.get("runtime", {})
```

Resolve the optional path:

```python
    intent_config_path = None
    if runtime.get("intent_config"):
        intent_config_path = _resolve_inside(root, runtime["intent_config"])
```

Pass it into `Settings(...)`:

```python
        intent_config_path=intent_config_path,
```

- [ ] **Step 4: Run settings test and full suite**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.ConfigAndManifestTests.test_load_config_resolves_optional_intent_config_path -v
python -m unittest discover -s runtime\local_rag_agent\tests -v
```

Expected: all tests pass.

## Task 2: Intent Definitions And Loader

**Files:**
- Create: `runtime/local_rag_agent/local_rag_agent/intent.py`
- Modify: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] **Step 1: Write failing loader test**

Add imports:

```python
from local_rag_agent.intent import IntentRouter, load_intents
```

Add this test class after `RuntimeTypeTests`:

```python
class IntentConfigTests(unittest.TestCase):
    def test_load_intents_reads_project_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intents.toml"
            path.write_text(
                '[[intents]]\n'
                'id = "policy_question"\n'
                'description = "Policy questions"\n'
                'examples = ["迟交政策是什么？"]\n'
                'keywords = ["迟交", "政策"]\n'
                'workflow = "rag_qa"\n'
                'risk_level = "high"\n',
                encoding="utf-8",
            )

            intents = load_intents(path)

            self.assertEqual(len(intents), 1)
            self.assertEqual(intents[0].id, "policy_question")
            self.assertEqual(intents[0].workflow, "rag_qa")
            self.assertEqual(intents[0].keywords, ["迟交", "政策"])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.IntentConfigTests.test_load_intents_reads_project_toml -v
```

Expected: import failure for `local_rag_agent.intent`.

- [ ] **Step 3: Implement loader**

Create `runtime/local_rag_agent/local_rag_agent/intent.py`:

```python
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class IntentDefinition:
    id: str
    workflow: str
    description: str = ""
    examples: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    risk_level: str = "medium"
    policy: str = ""
    knowledge_scopes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class IntentDecision:
    intent: IntentDefinition
    confidence: float
    matched_terms: list[str] = field(default_factory=list)
    source: str = "fallback"


def load_intents(path: Path | None) -> list[IntentDefinition]:
    if path is None or not path.exists():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    records = data.get("intents", [])
    if not isinstance(records, list):
        raise ValueError(f"Invalid intent config: {path}")
    return [_intent_from_record(record, path) for record in records if isinstance(record, dict)]


def _intent_from_record(record: dict[str, object], path: Path) -> IntentDefinition:
    intent_id = str(record.get("id", "")).strip()
    if not intent_id:
        raise ValueError(f"Intent missing id in {path}")
    workflow = str(record.get("workflow", "rag_qa")).strip() or "rag_qa"
    return IntentDefinition(
        id=intent_id,
        workflow=workflow,
        description=str(record.get("description", "")),
        examples=_string_list(record.get("examples", [])),
        keywords=_string_list(record.get("keywords", [])),
        risk_level=str(record.get("risk_level", "medium")),
        policy=str(record.get("policy", "")),
        knowledge_scopes=_string_list(record.get("knowledge_scopes", [])),
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


class IntentRouter:
    def __init__(self, intents: list[IntentDefinition], default_intent: str = "knowledge_qa", default_workflow: str = "rag_qa"):
        self.intents = intents
        self.default_intent = default_intent
        self.default_workflow = default_workflow

    def route(self, message: str) -> IntentDecision:
        fallback = self._fallback_intent()
        if not self.intents:
            return IntentDecision(intent=fallback, confidence=0.0, source="fallback")
        compact = message.replace(" ", "").lower()
        best: IntentDecision | None = None
        for intent in self.intents:
            matched = [term for term in intent.keywords if term and term.replace(" ", "").lower() in compact]
            if not matched:
                matched = [example for example in intent.examples if example and example.replace(" ", "").lower() in compact]
            if not matched:
                continue
            confidence = min(1.0, 0.55 + 0.15 * len(matched))
            decision = IntentDecision(intent=intent, confidence=confidence, matched_terms=matched, source="config")
            if best is None or decision.confidence > best.confidence:
                best = decision
        return best or IntentDecision(intent=fallback, confidence=0.0, source="fallback")

    def _fallback_intent(self) -> IntentDefinition:
        for intent in self.intents:
            if intent.id == self.default_intent:
                return intent
        return IntentDefinition(id=self.default_intent, workflow=self.default_workflow, description="Default knowledge QA intent")
```

- [ ] **Step 4: Run loader test and full suite**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.IntentConfigTests.test_load_intents_reads_project_toml -v
python -m unittest discover -s runtime\local_rag_agent\tests -v
```

Expected: all tests pass.

## Task 3: Deterministic Intent Routing

**Files:**
- Modify: `runtime/local_rag_agent/tests/test_local_rag_agent.py`
- Modify: `runtime/local_rag_agent/local_rag_agent/intent.py` only if the test exposes a real gap.

- [ ] **Step 1: Write routing tests**

Add these tests to `IntentConfigTests`:

```python
    def test_intent_router_selects_keyword_match(self):
        router = IntentRouter(
            [
                load_intents_from_inline(
                    '[[intents]]\n'
                    'id = "submission_boundary"\n'
                    'workflow = "refusal_with_guidance"\n'
                    'keywords = ["完整论文", "直接提交"]\n'
                )[0],
                load_intents_from_inline(
                    '[[intents]]\n'
                    'id = "knowledge_qa"\n'
                    'workflow = "rag_qa"\n'
                    'keywords = ["上课时间"]\n'
                )[0],
            ]
        )

        decision = router.route("请直接帮我写完整论文。")

        self.assertEqual(decision.intent.id, "submission_boundary")
        self.assertEqual(decision.intent.workflow, "refusal_with_guidance")
        self.assertEqual(decision.source, "config")

    def test_intent_router_falls_back_to_default(self):
        router = IntentRouter([])

        decision = router.route("普通问题")

        self.assertEqual(decision.intent.id, "knowledge_qa")
        self.assertEqual(decision.intent.workflow, "rag_qa")
        self.assertEqual(decision.source, "fallback")
```

Add this helper near the tests:

```python
def load_intents_from_inline(text: str):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "intents.toml"
        path.write_text(text, encoding="utf-8")
        return load_intents(path)
```

- [ ] **Step 2: Run routing tests**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.IntentConfigTests -v
```

Expected: routing tests pass with the implementation from Task 2. If not, fix only the router behavior needed by these tests.

## Task 4: AgentRuntime Uses Configured Intent

**Files:**
- Modify: `runtime/local_rag_agent/local_rag_agent/runtime.py`
- Modify: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] **Step 1: Write failing runtime test**

Add this test to `AgentTests`:

```python
    def test_runtime_uses_configured_intent_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prompt = root / "prompt.md"
            prompt.write_text("系统提示词", encoding="utf-8")
            intent_config = root / "intents.toml"
            intent_config.write_text(
                '[[intents]]\n'
                'id = "submission_boundary"\n'
                'workflow = "refusal_with_guidance"\n'
                'keywords = ["完整论文"]\n',
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
                '{"chunks":[{"chunk_id":"boundaries.md#0","source":"boundaries.md","title":"边界","content":"智能体不能提供可直接提交的完整论文。"}]}',
                encoding="utf-8",
            )
            runtime = AgentRuntime(settings)

            response = runtime.run(AgentRequest(message="请直接帮我写完整论文。"))
            payload = response.to_dict()

            self.assertEqual(payload["intent"], "submission_boundary")
            self.assertEqual(payload["workflow"], "refusal_with_guidance")
            self.assertEqual(payload["trace"]["intent"], "submission_boundary")
            self.assertEqual(payload["trace"]["steps"][0]["name"], "route_intent")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.AgentTests.test_runtime_uses_configured_intent_when_present -v
```

Expected: failure because runtime still hard-codes `knowledge_qa`.

- [ ] **Step 3: Modify runtime**

In `runtime.py`, import intent support:

```python
from .intent import IntentRouter, load_intents
```

In `AgentRuntime.__init__`, load the router:

```python
        self.intent_router = IntentRouter(load_intents(settings.intent_config_path))
```

At the start of `run()`, route the message before creating trace:

```python
        intent_decision = self.intent_router.route(request.message)
        trace = AgentTrace(intent=intent_decision.intent.id, workflow=intent_decision.intent.workflow)
        trace.add_step(
            "route_intent",
            {
                "source": intent_decision.source,
                "confidence": intent_decision.confidence,
                "matched_terms": intent_decision.matched_terms,
            },
        )
```

Return selected metadata:

```python
            intent=intent_decision.intent.id,
            workflow=intent_decision.intent.workflow,
```

Keep the rest of the current RAG behavior unchanged even if workflow is not `rag_qa`. Phase 3 will make workflow execution branch by workflow id.

- [ ] **Step 4: Update existing runtime trace assertions**

Existing tests that expected the first trace step to be `build_retrieval_query` should now expect:

```python
self.assertEqual(payload["trace"]["steps"][0]["name"], "route_intent")
self.assertEqual(payload["trace"]["steps"][1]["name"], "build_retrieval_query")
```

- [ ] **Step 5: Run runtime tests and full suite**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.AgentTests.test_runtime_uses_configured_intent_when_present -v
python -m unittest discover -s runtime\local_rag_agent\tests -v
```

Expected: all tests pass.

## Task 5: Template Intent Config Stub

**Files:**
- Create: `templates/agent-project/agent/intents.toml`
- Modify: `templates/agent-project/README.md`
- Modify: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] **Step 1: Write template existence test**

Add this test to `ConfigAndManifestTests`:

```python
    def test_template_agent_project_includes_structured_intents(self):
        template_root = Path("templates/agent-project")
        intent_config = template_root / "agent" / "intents.toml"

        intents = load_intents(intent_config)

        self.assertTrue(intent_config.exists())
        self.assertTrue(any(intent.id == "knowledge_qa" for intent in intents))
        self.assertTrue(any(intent.id == "complete_submission_request" for intent in intents))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.ConfigAndManifestTests.test_template_agent_project_includes_structured_intents -v
```

Expected: failure because `templates/agent-project/agent/intents.toml` does not exist.

- [ ] **Step 3: Add template intent config**

Create `templates/agent-project/agent/intents.toml`:

```toml
[[intents]]
id = "knowledge_qa"
description = "Answer source-backed questions from the project knowledge base."
examples = ["<高频事实问题>", "<领域专业问题>"]
keywords = ["<关键词1>", "<关键词2>"]
workflow = "rag_qa"
risk_level = "medium"
knowledge_scopes = ["current", "stable", "policy"]

[[intents]]
id = "complete_submission_request"
description = "Requests for complete assignments, papers, reports, or directly submittable work."
examples = ["请直接帮我写完整论文", "帮我写完整作业"]
keywords = ["完整论文", "完整作业", "完整报告", "直接提交", "代写"]
workflow = "refusal_with_guidance"
risk_level = "high"
policy = "academic_integrity"
knowledge_scopes = ["policy"]
```

- [ ] **Step 4: Update template README**

In `templates/agent-project/README.md`, add a short bullet under the project setup section:

```markdown
- `agent/intents.toml`：结构化意图配置。新增意图时优先改这里，再同步 `agent/intent-map.md` 作为维护者说明。
```

- [ ] **Step 5: Run template test and full suite**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest runtime.local_rag_agent.tests.test_local_rag_agent.ConfigAndManifestTests.test_template_agent_project_includes_structured_intents -v
python -m unittest discover -s runtime\local_rag_agent\tests -v
```

Expected: all tests pass.

## Task 6: Phase 2 Smoke Verification

**Files:**
- No source edits.

- [ ] **Step 1: Run full unit suite**

Run:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m unittest discover -s runtime\local_rag_agent\tests
```

Expected: all tests pass.

- [ ] **Step 2: Smoke a template project with structured intents enabled**

Create a temporary project from template, copy a TOML config with:

```toml
[runtime]
intent_config = "agent/intents.toml"
```

Run:

```powershell
python -m local_rag_agent ingest --project $project --config $config
python -m local_rag_agent regression --project $project --config $config --questions (Join-Path $project "examples\core-regression-questions.md")
```

Inspect the first JSONL record and confirm:

- `intent` is not empty.
- `workflow` is not empty.
- `trace.steps[0].name` is `route_intent`.

## Completion Criteria

Phase 2 is complete when:

- `Settings` supports optional `intent_config_path`.
- `intent.py` loads `agent/intents.toml`.
- `IntentRouter` can select a keyword-matched intent and fall back to `knowledge_qa`.
- `AgentRuntime.run()` records selected intent/workflow in response and trace.
- Existing projects without intent config still use `knowledge_qa`/`rag_qa`.
- Template projects include `agent/intents.toml`.
- Full unit suite and structured-intent smoke path pass.
