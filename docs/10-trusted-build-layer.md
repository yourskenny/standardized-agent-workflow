# Trusted Build Layer

The runtime can only be trusted when the project can explain what data was allowed
to reach retrieval and generation. A trusted build layer is the contract between
messy project inputs and the generic agent runtime.

## Boundary

The upstream runtime does not own project-specific cleaning logic. A course
assistant, marketing assistant, legal intake assistant, and operations assistant
will all process different sources. The upstream responsibility is to standardize
the evidence that a project must leave behind.

## Required Artifacts

Each project that derives knowledge from raw inputs should keep:

- a source inventory;
- a list of sensitive or private fields that were excluded or downgraded;
- derived artifacts such as metrics, facts, summaries, or retrieval chunks;
- a build manifest that links sources to artifacts;
- regression questions covering normal answers, refusals, and source visibility.

## Generic Flow

```text
raw project inputs
  -> project build or review script
  -> privacy downgrade and aggregation
  -> derived artifacts
  -> build manifest
  -> retrieval index and runtime response
```

The LLM should never receive unreviewed raw inputs by default. It should receive
only compiled prompt blocks: stable instructions, selected skills, source-backed
context chunks, limited conversation history, and the current user request.

## Runtime Support

The runtime now exposes two generic audit surfaces:

- `BuildManifest` in `local_rag_agent.build_assets` for source/artifact/privacy records.
- `AgentResponse.generation` for the generation boundary: provider, model, mode,
  input block summaries, source count, credential status, and fallback.

These surfaces are intentionally small. Domain logic remains downstream, while
audit shape remains upstream.

