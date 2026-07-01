# Trusted Agent Build Layer Design

## Goal

Promote reusable lessons from a downstream domain-agent demo into the generic agent runtime without carrying over domain data, domain prompts, or demo-specific implementation details.

## What Moves Upstream

The upstream project should gain a general contract for trusted agent assets:

- raw inputs are inventoried, not exposed directly;
- private or sensitive fields are explicitly excluded or downgraded;
- derived metrics and knowledge chunks are versioned as build artifacts;
- generation records describe what the LLM received and what role it played;
- regression output keeps enough metadata to audit source use, generation mode, and trace shape.

## What Stays Downstream

The following must remain project-specific:

- raw files, processed domain metrics, and local data paths;
- domain-specific intents, answer templates, heuristics, and UI copy;
- provider-specific secrets or local key file paths;
- any business-specific privacy field list unless a project declares it in its own build manifest.

## Architecture

The generic runtime keeps its current request flow:

```text
request -> intent -> workflow -> retrieval -> policy -> generation -> response
```

This change adds a trusted-build contract beside that flow:

```text
raw inputs -> project build script -> BuildManifest + derived artifacts -> retrieval/generation
```

The runtime does not try to understand every business dataset. Instead, it gives projects a common manifest type and template so each domain can explain which sources were transformed, which artifacts were produced, and which privacy controls were applied.

## Interface Additions

`AgentResponse` gains a `generation` object. This object is intentionally small:

- `mode`: model, extractive, refusal, tool, or another runtime mode;
- `provider`: the generation provider or policy/tool owner;
- `model`: the model name when applicable;
- `input_blocks`: prompt block trace summaries, not raw secrets;
- `source_count`: number of evidence sources exposed to generation;
- `credential_status`: whether model credentials were present, missing, or unnecessary.

The existing `metadata` field remains available for project-specific extensions.

## Validation

The upstream tests should prove four things:

- build manifests round-trip through JSON and preserve privacy/audit fields;
- responses serialize the generation boundary without breaking legacy payload fields;
- regression records include generation metadata;
- the project template contains a build-manifest example and a build-artifact contract document.
