# PilgrimAI — RAG Multi-Agent Research System

## Current status

Phase 5 is in progress: the project now has conversation routing, structured
query understanding, deterministic entity resolution, and an evidence schema
for preserving retrieved chunk IDs through grounded answer generation. The
baseline retrieval artifacts remain validated through dense,
section-aware retrieval.

## Research direction

The research question is:

> Can domain-specific hybrid retrieval and explicit pilgrimage-constraint
> validation improve the factual grounding and itinerary feasibility of
> LLM-based pilgrimage planning compared with generic vector RAG and
> unconstrained LLM planning?

Research questions:

1. How does domain-specific hybrid retrieval affect factual accuracy and
   grounding compared with standard vector RAG?
2. Does agentic orchestration improve pilgrimage itinerary feasibility compared
   with a single-agent LLM planner?
3. Does explicit temporal, spatial, budget, and pilgrimage-constraint
   validation reduce itinerary violations?

## Planned experiments

### Experiment 1 — Retrieval and grounded answering

Compare:

1. Dense vector RAG.
2. Hybrid dense plus BM25 retrieval with fusion and reranking.
3. Domain-routed hybrid retrieval with reranking.

Retrieval metrics include Recall@K, Precision@K, MRR, nDCG@K, entity
accuracy, domain accuracy, and section accuracy. Answer metrics include
correctness, faithfulness, relevance, citation accuracy, claim support rate,
unsupported-claim rate, hallucination rate, and freshness correctness.

The batch evaluator is [`tests/evaluate_grounded_answers.py`](tests/evaluate_grounded_answers.py).
It runs every selected benchmark case through a retrieval mode and records
the evidence chunks, generated claims, citation validity, claim-support rate,
and latency. It supports `unfiltered`, `entity_filtered`, and
`section_filtered` modes. A single live query is only a smoke test and is not
reported as experimental evidence.

### Experiment 2 — Planning and agent architecture

Compare:

1. Single LLM planner.
2. Single tool-using agent.
3. Multi-agent planner with a supervisor.
4. Multi-agent planner with supervisor, deterministic validator, and replanning.

Measure itinerary feasibility, temporal/budget/route violations, missing
requirements, preference satisfaction, latency, token usage, and cost.

### Experiment 3 — Pilgrimage Planning Benchmark

Build a benchmark of approximately 300 queries:

- 100 factual temple questions
- 100 simple travel and service questions
- 100 multi-constraint itinerary questions

Each planning case will include ground truth for duration, group size, budget,
temple visits, food preferences, timing requirements, and route constraints.

## Repository structure

```text
multi-agent-pilgrim/
├── README.md
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
│
├── src/
│   ├── conversation/
│   │   ├── __init__.py
│   │   ├── router.py
│   │   └── schemas.py
│   ├── evidence/
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── chunker.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedder.py
│   ├── vector_db/
│   │   ├── __init__.py
│   │   └── vector_store.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py
│   ├── query_understanding/
│   │   ├── __init__.py
│   │   ├── parser.py
│   │   └── schemas.py
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── prompt_templates.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── llm_client.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
│
├── data/
│   ├── raw/
│   │   └── pilgrimai_cleaned.zip
│   ├── processed/
│   │   ├── phase2_documents.jsonl
│   │   └── phase3_chunks.jsonl
│   ├── chroma_db/
│   └── evaluation/
│       ├── phase4_evaluation_dataset.json
│       ├── phase4_evaluation_results.json
│       ├── phase4_filtered_evaluation_results.json
│       └── phase4_failure_analysis.json
│
├── tests/
│   ├── test_app.py
│   ├── retrieval_smoke_test.py
│   ├── build_evaluation_dataset.py
│   ├── evaluate_dense_retrieval.py
│   ├── evaluate_filtered_retrieval.py
│   └── analyze_retrieval_failures.py
│
├── logs/
│   ├── app.log
│   └── retrieval_smoke_test.txt
│
└── main.py
```

## What has been validated

- Cleaned dataset: 23 temples, 136 accommodation records, 120 restaurant records, 23 emergency records, and 1 registry.
- Phase 2 document generation: 419 logical documents.
- Phase 3 chunking: 1,805 chunks.
- Embedding model: `intfloat/multilingual-e5-base`.
- Chroma collection: `pilgrimai`.
- Chroma record count: 1,805.
- Metadata-aware retrieval supports optional `source_type`, `temple_id`, and `section` constraints.
- Section is optional: broad queries do not require a section filter.
- Live Groq routing and structured query parsing have been validated with
  `openai/gpt-oss-20b`.
- End-to-end routed retrieval has been validated for a temple-history query.
- Retrieved chunks can now be represented as `EvidenceChunk` objects with
  stable citation IDs.
- Grounded answer validation rejects claims that cite unavailable chunks.

## Current implementation plan

Phase 5 — grounded evidence and claim tracking:

1. Preserve retrieved chunk IDs, metadata, content, and scores.
2. Require generated claims to identify supporting chunk IDs.
3. Reject unsupported citation references during schema validation.
4. Evaluate claim support, citation accuracy, and unsupported-claim rate.

The Phase 5 experiment protocol uses the existing 322-case retrieval
benchmark for batch grounded-answer evaluation. Development runs should use
`--limit`; the full benchmark run should be executed only after the prompt
and schema behavior is stable. Results are written to
`data/evaluation/grounded_answer_evaluation_results.json`.

Development batch status:

- Dataset slice: 3 cases
- Retrieval mode: `section_filtered`
- Successful cases: 3/3
- Generation errors: 0
- Claims: 21
- Claim support rate: 1.00
- Citation validity: 1.00
- Average latency: approximately 6.35 seconds per case

These development metrics are pipeline validation only, not statistically
meaningful research results. The full benchmark must be run for conclusions.

Full-run execution status:

- The unfiltered mode traversed all 322 benchmark cases.
- 42 cases generated grounded answers successfully.
- 280 cases returned explicit Groq `RateLimitError` results after the
  organization's 200,000-token daily limit was reached.
- The unfiltered artifact is
  `data/evaluation/grounded_answer_unfiltered_results.json`.
- Its partial successful subset contained 252 claims, with a 0.992 claim
  support rate and 1.000 citation validity.
- These numbers are not a valid comparison result because the run was
  quota-limited. Entity-filtered and section-filtered full runs were not
  started.

## Cost-controlled research protocol

The grounded-answer experiment is being changed from an unrestricted
322-query, three-mode LLM run to a staged protocol. This avoids treating a
quota-limited run as a scientific comparison and keeps API usage measurable.

### Stage A — Full retrieval benchmark

Run all 322 benchmark queries without LLM calls. Compare:

1. Dense vector retrieval
2. Entity-filtered retrieval
3. Section-filtered retrieval
4. Hybrid dense plus BM25 retrieval
5. Domain-routed hybrid retrieval

Measure Recall@K, Precision@K, MRR, nDCG@K, domain accuracy, entity
accuracy, section accuracy, and retrieval latency. This stage is local and
does not consume Groq quota.

Verified Stage A baseline runs:

| Configuration | Cases | Recall@5 | MRR | Strict/answer-section metric |
|---|---:|---:|---:|---:|
| Unfiltered dense retrieval | 322 | entity 0.972 | entity 0.920 | strict recall 0.568 |
| Source/entity-filtered dense retrieval | 322 | 0.944 | 0.869 | temple recall 0.843 |

The unfiltered run also achieved temple-section recall@5 of 0.817. These are
retrieval metrics only; no LLM answer generation was used. The result files
are `data/evaluation/phase4_evaluation_results.json` and
`data/evaluation/phase4_filtered_evaluation_results.json`.

Hybrid dense/BM25 and domain-routed hybrid retrieval are planned comparison
conditions and have not yet been implemented or measured.

## Why move toward hybrid retrieval?

Dense retrieval is strong at semantic and multilingual matching, but it can
miss exact lexical signals that matter in pilgrimage information, including
ritual names, seva names, temple service labels, timings, place names, prices,
and facility names. The current unfiltered baseline illustrates this gap:
entity recall@5 is 0.972, while strict recall@5 is only 0.568.

BM25 complements dense retrieval by rewarding exact term overlap. Hybrid
retrieval combines both signals:

```text
query -> dense retrieval
      -> BM25 lexical retrieval
      -> rank fusion
      -> final evidence candidates
```

This is the next research step because RQ1 asks whether
domain-specific retrieval improves factual grounding over standard vector
RAG. Hybrid retrieval gives us a concrete, retrieval-only comparison before
we spend LLM quota on grounded-answer evaluation. It also lets us test
whether exact pilgrimage terminology improves domain, entity, and section
accuracy rather than assuming that a larger language model will correct
retrieval errors downstream.

Planned hybrid conditions:

1. Dense plus BM25 fusion without domain routing.
2. Domain-routed dense plus BM25 fusion using validated metadata constraints.

These conditions will be evaluated on the full 322-query benchmark using
Recall@K, Precision@K, MRR, nDCG@K, entity accuracy, domain accuracy,
section accuracy, and retrieval latency.

Verified hybrid retrieval comparison:

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Avg latency |
|---|---:|---:|---:|---:|---:|
| Dense strict retrieval | 0.335 | 0.509 | 0.568 | 0.423 | 238.7 ms |
| BM25 strict retrieval | 0.168 | 0.323 | 0.425 | 0.259 | 21.1 ms |
| Unfiltered hybrid RRF | 0.283 | 0.469 | 0.540 | 0.380 | 337.8 ms |
| Domain-routed hybrid RRF | 0.717 | 0.776 | 0.826 | 0.755 | 303.7 ms |

These strict metrics require the expected source type and entity, and for
temple cases the expected section. The domain-routed condition uses the
benchmark's expected source type and temple ID as an oracle filter; it
therefore measures the value of hybrid retrieval after correct routing, not
the complete accuracy of the LLM router. The full result artifact is
`data/evaluation/hybrid_retrieval_results.json`.

The initial result is itself a useful finding: unfiltered BM25 and
unfiltered hybrid do not beat dense retrieval on this benchmark, while
domain restriction substantially improves hybrid retrieval. This supports
evaluating routing and lexical fusion as separate factors rather than
assuming that hybrid retrieval alone improves every query.

### Stage B — Stratified grounded-answer evaluation

Use a fixed 60-query sample rather than generating answers for all 322 cases:

- 20 temple queries
- 10 accommodation queries
- 10 restaurant queries
- 10 emergency queries
- 10 complex or multi-section queries

Compare only the selected baseline and proposed retrieval conditions. Cache
every generated response, use fixed prompts/model/settings, and record
generation errors separately from answer-quality metrics.

Measure claim-support rate, citation accuracy, citation completeness,
unsupported-claim rate, factual correctness, answer relevance, latency,
token usage, and cost.

### Stage C — Initial planning benchmark

Create an initial 30-case planning benchmark:

- 10 general itinerary cases
- 10 temporal/spatial constraint cases
- 10 budget/preference cases

Expand this benchmark after the validator and planner interfaces stabilize.
Each case should define hard constraints and expected feasibility outcomes.

### Cost and reproducibility rules

- Use retrieval-only evaluation whenever an LLM is not required.
- Use fixed, stratified samples for expensive LLM comparisons.
- Cache generated outputs and do not regenerate identical cases.
- Keep model, prompt, temperature, top-K, and retrieval configuration fixed
  within an experiment.
- Record quota errors, latency, token usage, and cost as experiment outcomes.
- Do not use blind API-key rotation as the primary research methodology.
- Treat the previous quota-limited 322-case Groq run as an execution artifact,
  not as a valid comparative result.

Next phases:

1. Add BM25 retrieval and dense/BM25 fusion.
2. Add reranking and domain-routed hybrid retrieval.
3. Build the pilgrimage planning benchmark.
4. Implement deterministic itinerary validation.
5. Compare single-agent, multi-agent, and validator/replanning systems.

## Conversation routing

The application should route each incoming message before retrieval:

```text
general_chat            -> conversational response, no retrieval
knowledge_query         -> structured query understanding -> retrieval
trip_planning           -> structured query -> validation/planning
follow_up               -> use conversation history, then route the resolved request
clarification_required  -> ask the user for missing information
```

The LLM may interpret language and propose structured fields, but Pydantic
validation and deterministic entity resolution must run before any metadata
filter is passed to ChromaDB. LLM output must never directly control a
database query.

## Important separation

LLMs will be used for language understanding, reasoning, and candidate planning. Deterministic code will be used where possible for exact filtering and validation of temporal, spatial, budget, ordering, and other hard constraints.
