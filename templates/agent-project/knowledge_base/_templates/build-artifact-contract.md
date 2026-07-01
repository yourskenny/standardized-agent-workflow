# Build Artifact Contract

Use this file to explain how project-specific raw inputs become trusted agent assets.

## Source Inventory

| Source ID | Kind | Location | Rows / Items | Sensitive Fields |
| --- | --- | --- | --- | --- |
| source-1 | spreadsheet | data/raw/example.xlsx | 120 | customer_name, phone, full_address |

## Processing Steps

1. Clean field types so numeric, date, and category values are stable.
2. Remove or downgrade fields that should not reach retrieval, prompts, UI, or regression output.
3. Join lookup tables into readable labels when this improves source review.
4. Aggregate raw rows into domain metrics or reviewed knowledge chunks.
5. Write derived artifacts and update `knowledge_base/_manifests/build-manifest.example.json`.

## Produced Artifacts

| Artifact | Kind | Purpose |
| --- | --- | --- |
| data/processed/example_metrics.json | metrics | Structured metrics for project tools or UI |
| knowledge_base/generated/example_chunks.json | knowledge_chunks | Evidence chunks allowed in retrieval |
| knowledge_base/_manifests/build-manifest.example.json | build_manifest | Audit record for source, artifact, and privacy handling |

## Review Rules

- Raw private data must not be added to the upload manifest.
- LLM context should receive only reviewed prompts, selected history, and source-backed chunks.
- Each generated artifact should be reproducible from a project-owned script or documented manual process.
- Regression questions should cover normal answers, refusals, and source visibility.

