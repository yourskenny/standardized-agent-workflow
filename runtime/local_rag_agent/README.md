# Local RAG Agent Runtime

This is a dependency-light reference implementation for running a standardized agent project without Dify. It is intentionally small: the first goal is to validate architecture, retrieval boundaries, source visibility, and regression evidence.

## What It Replaces

For local validation, this runtime replaces the core Dify path:

```text
upload manifest
  -> knowledge files
  -> chunking
  -> local JSON index
  -> retrieval
  -> system prompt + retrieved context
  -> model answer or retrieval-only answer
```

It does not replace Dify's visual admin console, user accounts, hosted Web App publishing, workflow designer, or production deployment.

## R Course Agent Smoke Test

From this repository:

```powershell
cd C:\coding\standardized-agent-workflow
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
```

Build a local index from the existing R course project:

```powershell
python -m local_rag_agent ingest `
  --project C:\coding\syllabus_R\course-agent-r `
  --config C:\coding\standardized-agent-workflow\runtime\local_rag_agent\examples\r-course-agent.toml
```

Ask a question:

```powershell
python -m local_rag_agent chat `
  --project C:\coding\syllabus_R\course-agent-r `
  --config C:\coding\standardized-agent-workflow\runtime\local_rag_agent\examples\r-course-agent.toml `
  "这门课的上课时间和地点是什么？"
```

Run regression evidence:

```powershell
python -m local_rag_agent regression `
  --project C:\coding\syllabus_R\course-agent-r `
  --config C:\coding\standardized-agent-workflow\runtime\local_rag_agent\examples\r-course-agent.toml `
  --questions C:\coding\syllabus_R\course-agent-r\examples\core-regression-questions.md
```

Start a local browser UI:

```powershell
python -m local_rag_agent serve `
  --project C:\coding\syllabus_R\course-agent-r `
  --config C:\coding\standardized-agent-workflow\runtime\local_rag_agent\examples\r-course-agent.toml `
  --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

## Model Configuration

If no model API key is set, `chat` returns retrieval-only output. This is useful for validating ingestion and source matching.

To use an OpenAI-compatible model:

```powershell
$env:LOCAL_AGENT_API_KEY="..."
$env:LOCAL_AGENT_BASE_URL="https://api.openai.com/v1"
$env:LOCAL_AGENT_MODEL="gpt-4.1-mini"
```

The runtime also accepts:

```text
OPENAI_API_KEY
OPENAI_BASE_URL
OPENAI_MODEL
```

## Generated Files

By default, generated local files are written inside the content project:

```text
.local_rag_agent/index.json
.local_rag_agent/regression/*.jsonl
```

These files are local evidence and should not be committed to content repositories unless a project explicitly wants to archive them.

## Design Notes

- The first retriever is lexical, transparent, and dependency-free.
- The index is JSON, so maintainers can inspect exactly what was ingested.
- The runtime reads the existing Dify upload manifest, which makes it a bridge rather than a separate content pipeline.
- Later versions can replace retrieval with embeddings, Chroma, SQLite FTS, pgvector, or a remote vector database behind the same CLI shape.
