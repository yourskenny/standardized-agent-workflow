# Local RAG Reference Implementation

This document explains how the standardized workflow can run without Dify for local validation.

## Why This Exists

Dify is useful for quickly building a knowledge-base agent, but it hides many runtime choices inside platform configuration: chunking, retrieval, model calling, source display, and publication. For reusable course agents and future business agents, those choices need to become explicit, testable, and portable.

The local reference implementation is the first step. It keeps the content workflow from this repository but adds a runnable path:

```text
agent project
  -> upload manifest
  -> Markdown knowledge files
  -> local index
  -> retrieval
  -> system prompt
  -> answer with sources
```

## What Stays The Same

The content project still owns:

- `PROJECT_BRIEF.md`
- `agent/system-prompt.md` or an equivalent prompt file
- `knowledge_base/`
- `knowledge_base/_manifests/`
- `examples/core-regression-questions.md`
- `maintenance/update-log.md`

For the R course project, the runtime uses the existing files directly:

- `dify/app-prompt.md`
- `knowledge_base/_manifests/dify-upload-manifest-2026-spring.md`
- `knowledge_base/semester_specific/2026-spring/`
- `knowledge_base/stable_materials/r_language/`
- `knowledge_base/policy_and_boundaries/`

## What Changes

Dify is no longer the only runtime. The local runtime can now:

- ingest the same knowledge files Dify would receive,
- create a local JSON index,
- retrieve relevant chunks,
- return a source-backed diagnostic answer without a model,
- call an OpenAI-compatible model when credentials are configured,
- write regression evidence as JSONL.

## Recommended First Validation

Use the R course project as the first validation target:

```powershell
cd C:\coding\standardized-agent-workflow
$env:PYTHONPATH="C:\coding\standardized-agent-workflow\runtime\local_rag_agent"
python -m local_rag_agent ingest --project C:\coding\syllabus_R\course-agent-r --config C:\coding\standardized-agent-workflow\runtime\local_rag_agent\examples\r-course-agent.toml
python -m local_rag_agent chat --project C:\coding\syllabus_R\course-agent-r --config C:\coding\standardized-agent-workflow\runtime\local_rag_agent\examples\r-course-agent.toml "这门课的上课时间和地点是什么？"
```

## How To Read The Results

If no model key is configured, the answer starts with:

```text
本地检索结果（未配置模型 API，因此没有生成式回答）
```

That output is still valuable. It shows whether the local runtime retrieved the right source files and snippets. Once retrieval quality is acceptable, configure a model API key to test generated answers.

## Migration Direction

The first version is deliberately modest. The valuable path is:

1. local lexical RAG for visibility and tests,
2. optional embedding retriever for better semantic recall,
3. shared runtime config for many courses,
4. a small frontend compatible with course websites,
5. production deployment only after local behavior is stable.

This keeps the system useful as both a Dify alternative and a template for future marketing-agent validation.
