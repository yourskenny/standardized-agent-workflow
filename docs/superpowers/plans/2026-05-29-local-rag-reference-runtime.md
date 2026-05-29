# Local RAG Reference Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dependency-light local RAG runtime that can ingest an agent project, retrieve knowledge chunks, answer through an OpenAI-compatible model when configured, and run regression evidence locally.

**Architecture:** The runtime is a Python package under `runtime/local_rag_agent`. It reads TOML configuration, resolves a content project, expands manifest entries, chunks Markdown files, stores a JSON index, ranks chunks with lexical scoring, and exposes CLI plus a small local HTTP server.

**Tech Stack:** Python standard library, `unittest`, JSON index files, OpenAI-compatible chat-completions HTTP API.

---

## File Structure

- Create `runtime/local_rag_agent/README.md`: maintainer-facing runtime guide.
- Create `runtime/local_rag_agent/examples/r-course-agent.toml`: config for the existing R course agent.
- Create `runtime/local_rag_agent/local_rag_agent/__init__.py`: package marker and version.
- Create `runtime/local_rag_agent/local_rag_agent/__main__.py`: `python -m local_rag_agent` entry point.
- Create `runtime/local_rag_agent/local_rag_agent/config.py`: config loading and path resolution.
- Create `runtime/local_rag_agent/local_rag_agent/manifest.py`: manifest parsing and file expansion.
- Create `runtime/local_rag_agent/local_rag_agent/chunking.py`: Markdown chunking.
- Create `runtime/local_rag_agent/local_rag_agent/index_store.py`: JSON index read/write.
- Create `runtime/local_rag_agent/local_rag_agent/retrieval.py`: tokenization and ranking.
- Create `runtime/local_rag_agent/local_rag_agent/llm.py`: OpenAI-compatible HTTP client.
- Create `runtime/local_rag_agent/local_rag_agent/agent.py`: prompt assembly and answer orchestration.
- Create `runtime/local_rag_agent/local_rag_agent/regression.py`: Markdown table question parser and JSONL runner.
- Create `runtime/local_rag_agent/local_rag_agent/server.py`: local HTTP API and browser chat shell.
- Create `runtime/local_rag_agent/local_rag_agent/cli.py`: CLI commands.
- Create `runtime/local_rag_agent/tests/test_local_rag_agent.py`: unit tests.
- Modify `.gitignore`: ignore local generated indexes and test output.
- Create `docs/08-local-rag-reference-implementation.md`: workflow-level explanation.

## Task 1: Config And Manifest Foundation

**Files:**
- Create: `runtime/local_rag_agent/local_rag_agent/config.py`
- Create: `runtime/local_rag_agent/local_rag_agent/manifest.py`
- Create: `runtime/local_rag_agent/tests/test_local_rag_agent.py`
- Create: `runtime/local_rag_agent/examples/r-course-agent.toml`

- [ ] **Step 1: Write failing tests**

```python
class ConfigAndManifestTests(unittest.TestCase):
    def test_load_config_resolves_project_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "knowledge_base").mkdir()
            (root / "dify").mkdir()
            (root / "dify" / "app-prompt.md").write_text("prompt", encoding="utf-8")
            config_path = root / "agent.toml"
            config_path.write_text(
                '[project]\n'
                'prompt_path = "dify/app-prompt.md"\n'
                'knowledge_root = "knowledge_base"\n'
                'manifest_path = "manifest.md"\n',
                encoding="utf-8",
            )
            settings = load_settings(root, config_path)
            self.assertEqual(settings.prompt_path, root / "dify" / "app-prompt.md")

    def test_manifest_parser_reads_bulleted_knowledge_paths(self):
        text = "- `knowledge_base/a.md`\n- `knowledge_base/dir/`\n"
        self.assertEqual(parse_manifest_entries(text), ["knowledge_base/a.md", "knowledge_base/dir/"])
```

- [ ] **Step 2: Verify tests fail**

Run: `python -m unittest discover -s runtime/local_rag_agent/tests -v`

Expected: import errors for missing `local_rag_agent.config` and `local_rag_agent.manifest`.

- [ ] **Step 3: Implement minimal config and manifest code**

Implement:

```python
@dataclass(frozen=True)
class Settings:
    project_root: Path
    prompt_path: Path
    manifest_path: Path
    knowledge_root: Path
    index_path: Path
    chunk_size: int = 1200
    chunk_overlap: int = 160
    top_k: int = 5

def load_settings(project_root: Path, config_path: Path) -> Settings:
    ...

def parse_manifest_entries(text: str) -> list[str]:
    ...

def expand_manifest_entries(settings: Settings) -> list[Path]:
    ...
```

- [ ] **Step 4: Verify tests pass**

Run: `python -m unittest discover -s runtime/local_rag_agent/tests -v`

Expected: config and manifest tests pass.

## Task 2: Chunking, Indexing, And Retrieval

**Files:**
- Create: `runtime/local_rag_agent/local_rag_agent/chunking.py`
- Create: `runtime/local_rag_agent/local_rag_agent/index_store.py`
- Create: `runtime/local_rag_agent/local_rag_agent/retrieval.py`
- Modify: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] **Step 1: Write failing tests**

```python
class RetrievalTests(unittest.TestCase):
    def test_chunk_markdown_preserves_source_metadata(self):
        chunks = chunk_markdown(Path("knowledge_base/course.md"), "# 标题\n\n上课时间是星期三上午。", 20, 4)
        self.assertEqual(chunks[0]["source"], "knowledge_base/course.md")
        self.assertIn("上课时间", chunks[0]["content"])

    def test_retrieval_ranks_matching_course_fact_first(self):
        chunks = [
            {"content": "上课时间是星期三上午 1-2 节。", "source": "a.md", "title": "课程事实", "chunk_id": "a#0"},
            {"content": "R Markdown 可以生成 HTML。", "source": "b.md", "title": "R Markdown", "chunk_id": "b#0"},
        ]
        results = rank_chunks("这门课周几上课？", chunks, top_k=1)
        self.assertEqual(results[0]["source"], "a.md")
```

- [ ] **Step 2: Verify tests fail**

Run: `python -m unittest discover -s runtime/local_rag_agent/tests -v`

Expected: missing chunking and retrieval functions.

- [ ] **Step 3: Implement minimal chunking, JSON index, and ranking**

Implement source-aware chunks with fields:

```python
{
    "chunk_id": "knowledge_base/course.md#0001",
    "source": "knowledge_base/course.md",
    "title": "标题",
    "content": "标题\n\n上课时间是星期三上午。",
}
```

Ranking uses Chinese 2-character n-grams, English words, R identifiers, and small boosts for title/source matches.

- [ ] **Step 4: Verify tests pass**

Run: `python -m unittest discover -s runtime/local_rag_agent/tests -v`

Expected: retrieval tests pass.

## Task 3: Agent Answering And LLM Client

**Files:**
- Create: `runtime/local_rag_agent/local_rag_agent/llm.py`
- Create: `runtime/local_rag_agent/local_rag_agent/agent.py`
- Modify: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] **Step 1: Write failing tests**

```python
class AgentTests(unittest.TestCase):
    def test_agent_returns_retrieval_only_answer_without_model(self):
        settings = make_test_settings()
        chunks = [{"content": "上课时间是星期三上午。", "source": "course.md", "title": "课程事实", "chunk_id": "course.md#0"}]
        response = answer_question(settings, "上课时间？", chunks, model_client=None)
        self.assertIn("本地检索结果", response["answer"])
        self.assertEqual(response["sources"][0]["source"], "course.md")
```

- [ ] **Step 2: Verify tests fail**

Run: `python -m unittest discover -s runtime/local_rag_agent/tests -v`

Expected: missing agent module.

- [ ] **Step 3: Implement prompt assembly and retrieval-only fallback**

Implement `answer_question(settings, question, chunks, model_client=None)`. If `model_client` is absent, return a source-backed diagnostic answer. If present, call `model_client.chat(messages)`.

- [ ] **Step 4: Verify tests pass**

Run: `python -m unittest discover -s runtime/local_rag_agent/tests -v`

Expected: agent tests pass.

## Task 4: CLI, Server, And Regression Evidence

**Files:**
- Create: `runtime/local_rag_agent/local_rag_agent/cli.py`
- Create: `runtime/local_rag_agent/local_rag_agent/__main__.py`
- Create: `runtime/local_rag_agent/local_rag_agent/server.py`
- Create: `runtime/local_rag_agent/local_rag_agent/regression.py`
- Modify: `runtime/local_rag_agent/tests/test_local_rag_agent.py`

- [ ] **Step 1: Write failing tests**

```python
class RegressionTests(unittest.TestCase):
    def test_parse_regression_table_extracts_question_column(self):
        markdown = "| 编号 | 问题 | 预期要点 |\n| --- | --- | --- |\n| C01 | 上课时间？ | 星期三 |\n"
        questions = parse_regression_questions(markdown)
        self.assertEqual(questions[0]["id"], "C01")
        self.assertEqual(questions[0]["question"], "上课时间？")
```

- [ ] **Step 2: Verify tests fail**

Run: `python -m unittest discover -s runtime/local_rag_agent/tests -v`

Expected: missing regression parser.

- [ ] **Step 3: Implement CLI and local server**

Commands:

```text
python -m local_rag_agent ingest --project C:\coding\syllabus_R\course-agent-r --config C:\coding\standardized-agent-workflow\runtime\local_rag_agent\examples\r-course-agent.toml
python -m local_rag_agent retrieve --project C:\coding\syllabus_R\course-agent-r --config C:\coding\standardized-agent-workflow\runtime\local_rag_agent\examples\r-course-agent.toml "这门课周几上课？"
python -m local_rag_agent chat --project C:\coding\syllabus_R\course-agent-r --config C:\coding\standardized-agent-workflow\runtime\local_rag_agent\examples\r-course-agent.toml "这门课的上课时间和地点是什么？"
python -m local_rag_agent serve --project C:\coding\syllabus_R\course-agent-r --config C:\coding\standardized-agent-workflow\runtime\local_rag_agent\examples\r-course-agent.toml --port 8765
python -m local_rag_agent regression --project C:\coding\syllabus_R\course-agent-r --config C:\coding\standardized-agent-workflow\runtime\local_rag_agent\examples\r-course-agent.toml --questions C:\coding\syllabus_R\course-agent-r\examples\core-regression-questions.md
```

- [ ] **Step 4: Verify tests pass**

Run: `python -m unittest discover -s runtime/local_rag_agent/tests -v`

Expected: all tests pass.

## Task 5: Documentation And R Course Smoke Test

**Files:**
- Create: `runtime/local_rag_agent/README.md`
- Create: `docs/08-local-rag-reference-implementation.md`
- Modify: `README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Document local R course usage**

Add commands for:

```powershell
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m local_rag_agent ingest --project C:\coding\syllabus_R\course-agent-r --config C:\coding\standardized-agent-workflow\runtime\local_rag_agent\examples\r-course-agent.toml
python -m local_rag_agent chat --project C:\coding\syllabus_R\course-agent-r --config C:\coding\standardized-agent-workflow\runtime\local_rag_agent\examples\r-course-agent.toml "这门课的上课时间和地点是什么？"
```

- [ ] **Step 2: Run tests**

Run: `python -m unittest discover -s runtime/local_rag_agent/tests -v`

Expected: all tests pass.

- [ ] **Step 3: Run R course ingest smoke test**

Run with `PYTHONPATH` set:

```powershell
python -m local_rag_agent ingest --project C:\coding\syllabus_R\course-agent-r --config C:\coding\standardized-agent-workflow\runtime\local_rag_agent\examples\r-course-agent.toml
```

Expected: index is written under `C:\coding\syllabus_R\course-agent-r\.local_rag_agent\index.json`.

- [ ] **Step 4: Run R course chat smoke test**

Run:

```powershell
python -m local_rag_agent chat --project C:\coding\syllabus_R\course-agent-r --config C:\coding\standardized-agent-workflow\runtime\local_rag_agent\examples\r-course-agent.toml "这门课的上课时间和地点是什么？"
```

Expected: answer includes retrieved source references and course-time source snippets.
