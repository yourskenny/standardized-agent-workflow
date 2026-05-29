# Local RAG Reference Implementation Design

## Goal

Build a local, self-hosted reference implementation that can reproduce the core effect of the current Dify-based course agent without depending on Dify. The first validation target is the R course agent project, while the implementation must remain reusable for other course agents and later business agents.

## Scope

This first version replaces the core Dify path:

```text
knowledge files + system prompt
  -> ingest
  -> chunk
  -> local index
  -> retrieve
  -> prompt assembly
  -> OpenAI-compatible model call or retrieval-only fallback
  -> answer with sources
```

It does not replace Dify's visual admin console, account system, hosted web app publishing, workflow designer, plugin marketplace, or production deployment. Those can be added later after the local runtime proves that the architecture works.

## User-Facing Outcome

A maintainer can run the R course agent locally from the standardized workflow repository by pointing the runtime at the existing course project:

```powershell
python -m local_rag_agent ingest --project C:\coding\syllabus_R\course-agent-r --config runtime\local_rag_agent\examples\r-course-agent.toml
python -m local_rag_agent chat --project C:\coding\syllabus_R\course-agent-r --config runtime\local_rag_agent\examples\r-course-agent.toml "这门课的上课时间和地点是什么？"
python -m local_rag_agent serve --project C:\coding\syllabus_R\course-agent-r --config runtime\local_rag_agent\examples\r-course-agent.toml --port 8765
```

If model credentials are configured, the runtime generates a normal conversational answer. If credentials are absent, it returns a retrieval-first diagnostic answer with the top matching sources, so maintainers can still validate ingestion, chunking, and retrieval locally.

## Architecture

### Runtime Boundary

The reusable runtime lives under:

```text
runtime/local_rag_agent/
```

It is a small Python package with no required third-party dependencies in the first version. This keeps installation lightweight and makes the runtime easy to inspect, copy, and adapt.

### Content Project Boundary

The runtime does not own course content. It reads a content project through configuration:

- system prompt path, such as `dify/app-prompt.md`
- upload manifest path, such as `knowledge_base/_manifests/dify-upload-manifest-2026-spring.md`
- knowledge root, such as `knowledge_base/`
- examples or regression question paths

For the R course validation, the content project remains:

```text
C:\coding\syllabus_R\course-agent-r
```

### Main Components

1. `config`
   Loads TOML configuration and resolves paths relative to the selected project root.

2. `manifest`
   Reads the existing Dify upload manifest and expands file or directory entries into Markdown knowledge files. Directories such as `_templates`, `_manifests`, `_pre_ingestion`, and `archive` remain excluded unless explicitly listed by a future config.

3. `chunking`
   Splits Markdown files into retrieval-sized chunks. It keeps heading context, source path, title, and chunk index.

4. `index`
   Stores chunks in a local JSON index. This keeps the first implementation portable. A later version can add Chroma, SQLite FTS, pgvector, or a remote vector database behind the same interface.

5. `retrieval`
   Uses lightweight lexical scoring over Chinese character n-grams plus English/R tokens. It is not as powerful as semantic embeddings, but it is transparent, dependency-free, and sufficient for validating architecture, source boundaries, and regression workflows.

6. `llm`
   Calls an OpenAI-compatible `/chat/completions` endpoint using environment variables. This supports OpenAI-compatible providers without hard-coding one platform.

7. `agent`
   Combines the system prompt, retrieved chunks, answer policy, and user question into a model prompt. It returns an answer plus source metadata.

8. `server`
   Provides a minimal local HTTP chat UI and JSON API using Python's standard library.

9. `cli`
   Exposes `ingest`, `retrieve`, `chat`, `serve`, and `regression` commands.

## Configuration

The first reference config is:

```text
runtime/local_rag_agent/examples/r-course-agent.toml
```

It defines project-relative paths and retrieval defaults. Model credentials are not stored in the file. They are read from environment variables:

```text
LOCAL_AGENT_API_KEY
LOCAL_AGENT_BASE_URL
LOCAL_AGENT_MODEL
```

The runtime also accepts common OpenAI-compatible aliases:

```text
OPENAI_API_KEY
OPENAI_BASE_URL
OPENAI_MODEL
```

## Data Flow

### Ingest

1. Load config.
2. Resolve manifest.
3. Expand all Markdown files from upload entries.
4. Read files as UTF-8.
5. Split content into chunks with metadata.
6. Write `.local_rag_agent/index.json` under the content project, unless config overrides the path.

### Chat

1. Load config and local index.
2. Rank chunks for the user question.
3. Assemble prompt with:
   - original system prompt
   - local runtime instruction
   - retrieved source snippets
   - user question
4. If a model is configured, call the OpenAI-compatible endpoint.
5. If no model is configured, return retrieval-only output.
6. Always include source file paths and chunk identifiers.

### Regression

1. Parse Markdown tables from regression question files.
2. Run each question through the same chat path.
3. Save a JSONL result file containing question, answer, retrieved sources, and timestamp.
4. This version records evidence but does not automatically grade semantic correctness.

## Error Handling

- Missing project root: fail with a clear path error.
- Missing config or manifest: fail before ingest.
- Manifest entry resolves outside project root: reject it.
- No knowledge files found: fail with a manifest-oriented message.
- Missing index during chat: tell the user to run `ingest`.
- Missing model credentials: do not fail; return retrieval-only output.
- Model HTTP failure: return a clear error and include retrieved sources so debugging can continue.

## Testing Strategy

The implementation uses Python `unittest` so it can run without dependency installation.

Required tests:

- Config resolution keeps paths inside the project root.
- Manifest parsing includes required upload entries and excludes maintenance sections.
- Directory expansion only returns Markdown files.
- Chunking preserves source metadata and respects size limits.
- Retrieval ranks a chunk containing a course fact above unrelated chunks.
- Agent returns retrieval-only output when no model is configured.
- Regression parser extracts questions from Markdown tables.

## Migration Value

This design turns the current Dify project into a reusable architecture:

- Dify prompt becomes `agent/system-prompt` input.
- Dify upload manifest becomes ingest configuration.
- Dify knowledge chunks become local chunks.
- Dify retrieval becomes a swappable retriever.
- Dify Web App becomes a local server and API.
- Existing regression questions become automated evidence.

The result is a practical bridge from a single R course agent to a school-wide course-agent template and to later marketing-agent validation.
