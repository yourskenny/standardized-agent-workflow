# Architecture Explanation Template

Use this outline when a new agent project needs to explain its architecture to
maintainers, reviewers, or non-engineering stakeholders.

## 1. Data Layer To Build Layer

Explain which inputs were processed, how they were cleaned, what sensitive fields
were removed or downgraded, which lookup tables made IDs readable, and which
derived artifacts were produced.

Example structure:

```text
source files -> cleaning -> joins/lookups -> aggregation -> privacy downgrade
  -> metrics / chunks / build manifest / regression questions
```

## 2. Build Layer To Memory Layer

Explain which produced artifacts become long-term project memory. Distinguish
versioned project memory from short-term chat history and runtime traces.

## 3. Memory Layer To Interface Layer

Explain what endpoints or UI surfaces expose. A good interface contract names
request fields, response fields, source references, trace objects, and generation
records.

## 4. Agent Flow

Explain the request path:

```text
message -> retrieval query -> intent -> retrieval -> policy -> generation -> response
```

For each step, name the input, output, failure mode, and trace entry.

## 5. LLM Boundary

State exactly what the LLM receives:

- stable system instructions;
- selected skill or policy text;
- source-backed context chunks;
- limited conversation history;
- the current user request.

State what the LLM does not receive:

- raw private data;
- fields excluded by the build manifest;
- secrets or API keys;
- unreviewed source files.

State what the LLM returns: generated text plus provider metadata. The runtime is
responsible for sources, trace, policy status, and final response shape.

