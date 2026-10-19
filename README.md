# PilgrimAI â€” RAG Multi-Agent Research System

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

### Experiment 1 â€” Retrieval and grounded answering

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

### Experiment 2 â€” Planning and agent architecture

Compare:

1. Single LLM planner.
2. Single tool-using agent.
3. Multi-agent planner with a supervisor.
4. Multi-agent planner with supervisor, deterministic validator, and replanning.

Measure itinerary feasibility, temporal/budget/route violations, missing
requirements, preference satisfaction, latency, token usage, and cost.

### Experiment 3 â€” Pilgrimage Planning Benchmark

Build a benchmark of approximately 300 queries:

- 100 factual temple questions
- 100 simple travel and service questions
- 100 multi-constraint itinerary questions

Each planning case will include ground truth for duration, group size, budget,
temple visits, food preferences, timing requirements, and route constraints.

## Repository structure

```text
multi-agent-pilgrim/
â”œâ”€â”€ README.md
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ pyproject.toml
â”œâ”€â”€ uv.lock
â”œâ”€â”€ .env.example
â”œâ”€â”€ .gitignore
â”‚
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ conversation/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â”œâ”€â”€ router.py
â”‚   â”‚   â””â”€â”€ schemas.py
â”‚   â”œâ”€â”€ evidence/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â””â”€â”€ schemas.py
â”‚   â”œâ”€â”€ ingestion/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â””â”€â”€ loader.py
â”‚   â”œâ”€â”€ chunking/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â””â”€â”€ chunker.py
â”‚   â”œâ”€â”€ embeddings/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â””â”€â”€ embedder.py
â”‚   â”œâ”€â”€ vector_db/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â””â”€â”€ vector_store.py
â”‚   â”œâ”€â”€ retrieval/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â””â”€â”€ retriever.py
â”‚   â”œâ”€â”€ query_understanding/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â”œâ”€â”€ parser.py
â”‚   â”‚   â””â”€â”€ schemas.py
â”‚   â”œâ”€â”€ prompts/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â””â”€â”€ prompt_templates.py
â”‚   â”œâ”€â”€ llm/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â””â”€â”€ llm_client.py
â”‚   â”œâ”€â”€ api/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â””â”€â”€ routes.py
â”‚   â””â”€â”€ utils/
â”‚       â”œâ”€â”€ __init__.py
â”‚       â””â”€â”€ helpers.py
â”‚
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/
â”‚   â”‚   â””â”€â”€ pilgrimai_cleaned.zip
â”‚   â”œâ”€â”€ processed/
â”‚   â”‚   â”œâ”€â”€ phase2_documents.jsonl
â”‚   â”‚   â””â”€â”€ phase3_chunks.jsonl
â”‚   â”œâ”€â”€ chroma_db/
â”‚   â””â”€â”€ evaluation/
â”‚       â”œâ”€â”€ phase4_evaluation_dataset.json
â”‚       â”œâ”€â”€ phase4_evaluation_results.json
â”‚       â”œâ”€â”€ phase4_filtered_evaluation_results.json
â”‚       â””â”€â”€ phase4_failure_analysis.json
â”‚
â”œâ”€â”€ tests/
â”‚   â”œâ”€â”€ test_app.py
â”‚   â”œâ”€â”€ retrieval_smoke_test.py
â”‚   â”œâ”€â”€ build_evaluation_dataset.py
â”‚   â”œâ”€â”€ evaluate_dense_retrieval.py
â”‚   â”œâ”€â”€ evaluate_filtered_retrieval.py
â”‚   â””â”€â”€ analyze_retrieval_failures.py
â”‚
â”œâ”€â”€ logs/
â”‚   â”œâ”€â”€ app.log
â”‚   â””â”€â”€ retrieval_smoke_test.txt
â”‚
â””â”€â”€ main.py
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
  `qwen/qwen3.8-27b`.
- End-to-end routed retrieval has been validated for a temple-history query.
- Retrieved chunks can now be represented as `EvidenceChunk` objects with
  stable citation IDs.
- Grounded answer validation rejects claims that cite unavailable chunks.

## Current implementation plan

Phase 5 â€” grounded evidence and claim tracking:

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

### Stage A â€” Full retrieval benchmark

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

### Stage A.1 â€” LLM-routed retrieval evaluation

The next retrieval experiment connects the existing LLM conversation router
and structured query parser to metadata-aware retrieval. It uses a fixed
60-case stratified sample (20 temple, 10 accommodation, 10 restaurant, 10
emergency, and 10 additional deterministic cases), caches every LLM response, and then
evaluates local retrieval without regenerating the LLM output.

The comparison is:

```text
unfiltered dense baseline
unfiltered hybrid baseline
LLM-routed dense retrieval
LLM-routed hybrid retrieval
oracle-routed hybrid reference
```

The LLM predicts the route, domain, entity, and optional section. Pydantic
validation and deterministic entity resolution run before any filters reach
ChromaDB or BM25. Route errors, unknown entities, malformed responses, and
rate-limit failures are recorded explicitly rather than treated as retrieval
failures.

Entity resolution now uses the versioned deterministic registry at
`data/registry/entity_registry.json`. Canonical names and aliases resolve to
registry IDs, while ambiguous aliases such as `Dakshina Kashi` are rejected
instead of selecting an arbitrary entity. The LLM supplies an entity mention;
the registry, not the LLM, supplies the final entity ID.

The routing report includes valid-route rate, validated-query rate, domain
accuracy, entity accuracy, and section accuracy. Retrieval
failures are reported separately from LLM routing or validation failures.

The evaluator is `tests/evaluate_llm_routed_retrieval.py`. Its generated
artifacts are `data/evaluation/llm_router_cache.json` and
`data/evaluation/llm_routed_retrieval_results.json`; these files must not
contain API keys. The experiment remains quota-sensitive because each new
query requires router and structured-query calls, so cached responses should
be reused for all subsequent retrieval comparisons.

The evaluator resumes from successful cache entries and writes each completed
record atomically after processing it, including records created after a
provider quota error. Groq requests use a 120-second timeout by default and
retry connection, timeout, rate-limit, and server errors with bounded
exponential backoff. These settings can be overridden with
`--request-delay`, `--request-timeout`, and `--max-retries`; the selected
values are stored in the result configuration.

This condition is the first realistic test of whether the oracle-filtered
hybrid result can be approached using predicted routing. It must not be
interpreted as an end-to-end planning result.

The first 60-case execution completed with the following measured results:

| Metric | Result |
|---|---:|
| Valid route rate | 1.000 |
| Validated-query rate | 0.500 |
| Domain accuracy over all cases | 0.500 |
| Entity accuracy over all cases | 0.500 |
| Section accuracy over all cases | 0.233 |
| LLM-routed dense Recall@5 | 0.333 |
| LLM-routed dense MRR | 0.303 |
| LLM-routed hybrid Recall@5 | 0.283 |
| LLM-routed hybrid MRR | 0.271 |

The run had 30 explicit validation or entity-resolution errors. The main
failure patterns were model outputs whose `constraints` field was a string or
list instead of an object, plus unresolved entity names. These are counted as
LLM/query-understanding failures, not silently converted into retrieval
failures. The result is therefore diagnostic rather than a final quality
claim: the current bottleneck is structured-output reliability and entity
resolution before ranking. The cache and complete per-query artifact are
`data/evaluation/llm_router_cache.json` and
`data/evaluation/llm_routed_retrieval_results.json`.

After integrating the supplied entity registry, the same fresh 60-case
sample was rerun with a separate cache. All 31 valid resolutions used an
exact registry alias match; 29 cases still failed before retrieval, primarily
because the LLM returned `constraints` as a string or list instead of an
object. The registry-backed results were:

| Metric | Registry-backed result |
|---|---:|
| Valid route rate | 1.000 |
| Validated-query rate | 0.517 |
| Domain accuracy over all cases | 0.500 |
| Entity accuracy over all cases | 0.517 |
| Section accuracy over all cases | 0.267 |
| LLM-routed dense Recall@5 | 0.333 |
| LLM-routed dense MRR | 0.300 |
| LLM-routed hybrid Recall@5 | 0.267 |
| LLM-routed hybrid MRR | 0.267 |

The fresh cache and result artifact are
`data/evaluation/llm_router_registry_cache.json` and
`data/evaluation/llm_routed_registry_retrieval_results.json`. Compared with
the prior run, the registry successfully resolves canonical/alias names, but
it cannot improve cases that fail structured-query validation before entity
resolution. The next improvement should therefore target the LLM
`constraints` schema contract, not replace the registry.

The parser intentionally rejects string/list constraint values and accepts
only a JSON object or `null` (normalized to `{}`). The evaluator now reports
both end-to-end coverage over all cases and accuracy among valid predictions,
so malformed LLM output is not confused with ranking quality.

The generic constraint contract was tightened without introducing a new
constraint vocabulary: arbitrary JSON objects are preserved, `null` becomes
`{}`, and strings/lists are rejected. A two-query live smoke test with
`qwen/qwen3.8-27b` confirmed one complete registry-backed parse, including
`constraints: {}`. The second response also returned an object for
`constraints`, but was rejected because the model omitted the required
`confidence` field; this remains a separate required-field reliability issue.
The requested `llama-3.1-8b-instant` model was not available to the Groq
organization, so the accessible Qwen model is now the configured default.

The contract-v3 rerun was blocked by the existing Groq organization quota:
the provider reported 199,597 of 200,000 daily tokens already used before a
new request. The evaluator now records that provider-specific rate-limit error
and stops issuing further calls instead of aborting or repeatedly retrying.

### OpenRouter structured-output experiment

The LLM boundary now includes an OpenRouter adapter that sends the actual
Pydantic response model through `response_format.type = "json_schema"` with
strict mode enabled. Routing and query parsing use separate schemas, so each
call is constrained to its own contract. The existing Groq JSON-mode adapter
remains unchanged.

The OpenRouter model catalog confirmed that `openrouter/free` advertises
structured-output support. However, the live two-query probe and the full
60-query run were both rate-limited with HTTP 429 before receiving model
responses. The 60-query artifact therefore contains 60 provider errors, zero
validated queries, and zero malformed structured-query responses. This is a
provider availability result, not evidence that the schema contract failed.
The artifact is `data/evaluation/openrouter_structured60_results.json`.

### Gemini structured-output experiment

The LLM boundary also supports Gemini's native JSON response schema using the
local `GEMINI_API_KEY` and `gemini-3-flash-preview`. Gemini requires a
provider-specific schema conversion because raw Pydantic keywords such as
`$defs`, `$ref`, and `additionalProperties` are not accepted by its REST
`responseSchema` field. The conversion resolves references and preserves
nullable fields without weakening Pydantic validation.

The first controlled 10-query run produced four valid structured queries with
100% domain and entity accuracy among those valid cases. Six requests received
HTTP 429, so the next Gemini run should use pacing and retry/backoff rather
than a full unthrottled batch.

### Stage B â€” Stratified grounded-answer evaluation

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

### Stage C â€” Initial planning benchmark

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

## Experimented Results and Research-Paper Evidence

Yes. We performed several experiments before the final 200-query benchmark. They should not be discarded. They should be organized into:

1. **Baseline experiments**
2. **Ablation and comparison experiments**
3. **Failure-analysis experiments**
4. **Provider and structured-output experiments**
5. **Final validated experiment**
6. **Incomplete or non-publishable diagnostic runs**

The important point is that we should not present every run as equal scientific evidence. Some runs are final results, some are ablations, and some only explain why the system was improved.

# 1. Dataset and retrieval baseline experiments

These were the initial non-LLM retrieval experiments over the full 322-case dataset.

## Dataset

The benchmark contains:

| Domain | Cases |
|---|---:|
| Temple | 115 |
| Accommodation | 69 |
| Restaurant | 69 |
| Emergency | 69 |
| **Total** | **322** |

The corpus contains:

- 23 temples
- 136 accommodation records
- 120 restaurant records
- 23 emergency records
- 1 entity registry
- 419 logical documents
- 1,805 chunks
- Chroma vector collection with 1,805 records
- Embedding model: `intfloat/multilingual-e5-base`

These corpus-construction details belong in the **Dataset and System Construction** section of the paper.

Relevant artifacts:

- `phase4_evaluation_results.json`
- `phase4_filtered_evaluation_results.json`

## Initial dense retrieval baseline

The original dense retrieval experiment measured both entity-level retrieval and strict answer retrieval.

| Metric | Result |
|---|---:|
| Entity Recall@1 | 0.885 |
| Entity Recall@3 | 0.950 |
| Entity Recall@5 | 0.972 |
| Entity MRR | 0.920 |
| Strict Recall@1 | 0.335 |
| Strict Recall@3 | 0.509 |
| Strict Recall@5 | 0.568 |
| Strict MRR | 0.423 |
| Temple-section Recall@1 | 0.461 |
| Temple-section Recall@3 | 0.730 |
| Temple-section Recall@5 | 0.817 |
| Temple-section MRR | 0.595 |

This is an important baseline result because it shows that retrieving the correct temple/entity is easier than retrieving the exact required section.

For example:

```text
Entity-level retrieval:
The correct temple may be retrieved.

Strict retrieval:
The correct temple and correct section must both be retrieved.
```

This baseline supports the argument that entity recognition alone is not enough for factual retrieval quality.

---

# 2. Metadata-filtered retrieval experiment

The next experiment introduced metadata filters such as:

- Source type
- Temple ID
- Section

The filtered dense retrieval result was:

| Metric | Result |
|---|---:|
| Recall@1 | 0.823 |
| Recall@3 | 0.916 |
| Recall@5 | 0.944 |
| MRR | 0.869 |

This experiment showed that metadata-aware filtering significantly improves exact retrieval compared with unfiltered dense retrieval.

However, the paper should explain that filtering changes the retrieval task. It is not purely a better embedding model. It is a combination of:

```text
Semantic retrieval
+
Correct metadata constraints
```

This is why the paper should report both:

- Unfiltered dense retrieval
- Metadata-filtered dense retrieval

rather than only reporting the filtered result.

---

# 3. Dense, BM25, and hybrid retrieval experiment

We also evaluated:

1. Dense retrieval
2. BM25 retrieval
3. Unfiltered hybrid retrieval
4. Domain-routed hybrid retrieval

The results were:

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Avg latency |
|---|---:|---:|---:|---:|---:|
| Dense strict retrieval | 0.335 | 0.509 | 0.568 | 0.423 | 238.7 ms |
| BM25 strict retrieval | 0.168 | 0.323 | 0.425 | 0.259 | 21.1 ms |
| Unfiltered hybrid RRF | 0.283 | 0.469 | 0.540 | 0.380 | 337.8 ms |
| Domain-routed hybrid RRF | 0.717 | 0.776 | 0.826 | 0.755 | 303.7 ms |

Artifact:

- `hybrid_retrieval_results.json`

## Research significance

This is a useful negative and positive result:

- BM25 alone was weaker than dense retrieval.
- Unfiltered hybrid retrieval did not beat dense retrieval.
- Domain-routed hybrid retrieval performed substantially better.
- Correct routing and metadata restriction were more important than simply combining retrieval methods.

This supports the research conclusion:

> Hybrid retrieval is most effective when combined with correct domain and entity restrictions. Hybrid fusion alone does not guarantee better retrieval.

## Important qualification

The domain-routed hybrid experiment used the expected benchmark domain/entity values as an oracle filter.

Therefore, it measures:

```text
Retrieval quality after correct routing
```

It does not measure:

```text
Complete end-to-end LLM routing performance
```

This distinction must be stated clearly in the paper.

---

# 4. Initial LLM-routed retrieval experiment

The first 60-query LLM-routed experiment used a less mature structured-output contract.

Artifact:

- `llm_routed_retrieval_results.json`

Results:

| Metric | Result |
|---|---:|
| Cases | 60 |
| Valid route rate | 1.000 |
| Validated-query rate | 0.500 |
| Domain accuracy over all cases | 0.500 |
| Entity accuracy over all cases | 0.500 |
| Section accuracy over all cases | 0.233 |
| LLM-routed dense Recall@5 | 0.333 |
| LLM-routed dense MRR | 0.303 |
| LLM-routed hybrid Recall@5 | 0.283 |
| LLM-routed hybrid MRR | 0.271 |

There were 30 validation or entity-resolution errors.

The common failures were:

- `constraints` returned as a string
- `constraints` returned as a list
- Missing required fields
- Unknown entity names
- Inconsistent section names

## How to use this in the paper

This should not be presented as the final system performance.

It should be presented as:

> Initial failure analysis showing that malformed structured output and entity resolution failures were the dominant bottlenecks before retrieval.

This experiment justifies the later schema and registry improvements.

---

# 5. Entity-registry experiment

We then introduced the deterministic entity registry.

Artifact:

- `llm_routed_registry_retrieval_results.json`

Results:

| Metric | Result |
|---|---:|
| Valid route rate | 1.000 |
| Validated-query rate | 0.517 |
| Domain accuracy over all cases | 0.500 |
| Entity accuracy over all cases | 0.517 |
| Section accuracy over all cases | 0.267 |
| LLM-routed dense Recall@5 | 0.333 |
| LLM-routed dense MRR | 0.300 |
| LLM-routed hybrid Recall@5 | 0.267 |
| LLM-routed hybrid MRR | 0.267 |

The registry successfully resolved valid canonical names and aliases, but 29 cases still failed before retrieval because of malformed structured output.

## Research significance

This experiment demonstrates an important architectural point:

> Deterministic entity resolution improves entity reliability, but it cannot repair malformed LLM outputs that fail schema validation before entity resolution.

Therefore, the final architecture required two separate controls:

```text
Structured schema validation
+
Deterministic entity registry
```

The registry experiment is useful as an ablation or incremental-development result.

---

# 6. Qwen model experiment

We also evaluated the Qwen model on a 60-query sample.

Artifact:

- `llm_routed_qwen60_results.json`

Results:

| Metric | Result |
|---|---:|
| Valid route rate | 1.000 |
| Validated-query rate | 0.500 |
| Domain accuracy | 0.500 |
| Entity accuracy | 0.500 |
| Section accuracy | 0.183 |
| Dense Recall@5 | 0.450 |
| Dense MRR | 0.378 |
| Hybrid Recall@5 | 0.317 |
| Hybrid MRR | 0.265 |

This experiment is useful for comparing model behavior, but it is not directly comparable with the final Groq GPT-OSS result unless all of the following are held constant:

- Same prompt
- Same schema version
- Same dataset sample
- Same registry
- Same retrieval code
- Same retry and pacing configuration

Therefore, it should be described as an early model-selection or diagnostic experiment, not as the main model comparison.

---

# 7. Five-query Groq experiment

Before running the 60-query and 200-query experiments, we performed a controlled five-query test.

## Before strict section enforcement

Artifact:

- `groq_gpt_oss_120b_5_results.json`

Results:

| Metric | Result |
|---|---:|
| Valid route rate | 1.000 |
| Validated-query rate | 1.000 |
| Domain accuracy | 1.000 |
| Entity accuracy | 1.000 |
| Section accuracy | 0.600 |
| Dense Recall@5 | 0.600 |
| Hybrid Recall@5 | 0.600 |
| Dense MRR | 0.600 |
| Hybrid MRR | 0.600 |

This was a smoke test and showed that the basic Groq pipeline worked, but section normalization was not yet reliable.

## After strict section enforcement

Artifact:

- `groq_gpt_oss_120b_5_strict_sections_results.json`

Results:

| Metric | Result |
|---|---:|
| Valid route rate | 1.000 |
| Validated-query rate | 1.000 |
| Domain accuracy | 1.000 |
| Entity accuracy | 1.000 |
| Section accuracy | 1.000 |
| Dense Recall@1/3/5 | 1.000 / 1.000 / 1.000 |
| Hybrid Recall@1/3/5 | 1.000 / 1.000 / 1.000 |
| Dense MRR | 1.000 |
| Hybrid MRR | 1.000 |

## Research use

These two five-query runs are not statistically meaningful, but they are useful as:

- Development validation
- Contract-ablation evidence
- Motivation for strict canonical section values
- Demonstration that the schema fix solved a measurable failure

They should be included in a development or ablation subsection, not in the main results table.

---

# 8. Strict-section 60-query experiment

Artifact:

- `groq_gpt_oss_120b_60_strict_sections_results.json`

This run used strict canonical temple sections but before the complete reference-entity fix.

Results:

| Metric | Result |
|---|---:|
| Valid route rate | 1.000 |
| Validated-query rate | 0.500 |
| Domain accuracy | 0.500 |
| Entity accuracy | 0.500 |
| Section accuracy | 0.500 |
| Dense Recall@5 | 0.500 |
| Hybrid Recall@5 | 0.500 |
| Errors | 30 |

The valid predictions themselves were accurate, but 30 cases failed validation.

## Why this matters

This run showed that strict section enforcement solved one class of problem but exposed another:

```text
Temple-only entity representation was insufficient for accommodation,
restaurant, and emergency queries referring to a temple.
```

For example:

```text
Accommodation near Temple X
```

was incorrectly represented as if Temple X itself were the requested accommodation entity.

This motivated the `reference_entity_name` and `reference_entity_type` schema fields.

This is a strong failure-analysis experiment and should be included in the paper as evidence of iterative schema refinement.

---

# 9. Reference-entity 60-query experiment

Artifact:

- `groq_gpt_oss_120b_60_reference_entities_results.json`

After adding:

- `reference_entity_name`
- `reference_entity_type`
- Reference entity resolution
- Cross-domain validation

the complete 60-query experiment achieved:

| Metric | Result |
|---|---:|
| Valid route rate | 1.000 |
| Validated-query rate | 1.000 |
| Domain accuracy | 1.000 |
| Entity accuracy | 1.000 |
| Section accuracy | 1.000 |
| Dense Recall@1/3/5 | 1.000 / 1.000 / 1.000 |
| Hybrid Recall@1/3/5 | 1.000 / 1.000 / 1.000 |
| Dense MRR | 1.000 |
| Hybrid MRR | 1.000 |

This is the first complete stable validation of the corrected LLM-to-retrieval contract.

It can be presented as:

> The reference-entity schema resolved the cross-domain failure mode and restored complete validated coverage on the 60-case stratified benchmark.

This is a strong intermediate milestone.

---

# 10. Final 200-query experiment

Artifact:

- `groq_gpt_oss_120b_200_paper_results.json`

This is the strongest current result.

Configuration:

- 200 cases
- 71 temple
- 43 accommodation
- 43 restaurant
- 43 emergency
- Groq
- `openai/gpt-oss-120b`
- Four-second request pacing
- 120-second timeout
- Five retries

Results:

| Metric | Result |
|---|---:|
| Valid route rate | 1.000 |
| Validated-query rate | 1.000 |
| Domain accuracy | 1.000 |
| Entity accuracy | 1.000 |
| Section accuracy | 1.000 |
| Dense Recall@1/3/5 | 1.000 / 1.000 / 1.000 |
| Hybrid Recall@1/3/5 | 1.000 / 1.000 / 1.000 |
| Dense MRR | 1.000 |
| Hybrid MRR | 1.000 |
| Dense nDCG@1/3/5 | 1.000 / 1.000 / 1.000 |
| Hybrid nDCG@1/3/5 | 1.000 / 1.000 / 1.000 |
| Dense latency | 230.69 ms |
| Hybrid latency | 226.93 ms |
| Evaluator errors | 0 |

This should be the main LLM-routed retrieval result.

However, it must be interpreted carefully:

- It proves excellent performance on this benchmark.
- It does not prove general performance on unrelated datasets.
- The validated metadata filters strongly constrain retrieval.
- It is not yet an end-to-end itinerary-planning result.

---

# 11. OpenRouter experiments

We attempted structured-output experiments with OpenRouter.

Artifacts include:

- `openrouter_structured1_results.json`
- `openrouter_structured60_results.json`
- `openrouter_paid10_results.json`

The OpenRouter adapter supported strict JSON schema output using:

```text
response_format.type = json_schema
strict = true
```

But the live experiments were affected by provider availability:

- HTTP 429 rate limits
- Free-tier restrictions
- Billing or credit limitations

The 60-query structured run produced:

| Metric | Result |
|---|---:|
| Cases | 60 |
| Valid route rate | 0.633 |
| Validated-query rate | 0.283 |
| Errors | 43 |
| Dense Recall@5 | 0.183 |
| Hybrid Recall@5 | 0.133 |

These results should **not** be presented as evidence that OpenRouter’s structured output is technically inferior. They primarily measure provider availability and quota conditions.

The correct paper classification is:

> Provider feasibility and reproducibility experiment; not a model-quality comparison.

---

# 12. Gemini experiments

Artifacts include:

- `gemini_structured10_results.json`
- `gemini_structured20_results.json`

Gemini required a custom schema conversion because its REST API did not accept all raw Pydantic JSON Schema keywords, including:

- `$defs`
- `$ref`
- `additionalProperties`

The first controlled 10-query run achieved:

| Metric | Result |
|---|---:|
| Cases | 10 |
| Validated queries | 4 |
| Domain accuracy among valid cases | 1.000 |
| Entity accuracy among valid cases | 1.000 |
| Section accuracy among valid cases | 0.250 |
| HTTP 429 requests | 6 |

The 20-query run was effectively quota-limited:

| Metric | Result |
|---|---:|
| Cases | 20 |
| Validated queries | 0 |
| Errors | 20 |

These results should be included only in a provider evaluation or reproducibility section.

They should not be used to claim that Gemini performs poorly because the experiments were dominated by rate limiting.

---

# 13. Grounded-answer experiments

## Development grounded-answer run

Artifact:

- `grounded_answer_dev_results.json`

Results:

| Metric | Result |
|---|---:|
| Cases | 3 |
| Successful cases | 3 |
| Claims | 21 |
| Supported claims | 21 |
| Claim support rate | 1.000 |
| Citation validity | 1.000 |
| Average latency | Approximately 6.35 seconds |

This validates the evidence and citation mechanism, but it is too small to represent final research evidence.

## Full grounded-answer attempt

Artifact:

- `grounded_answer_unfiltered_results.json`

This run was quota-limited:

- 322 cases were attempted
- 42 cases generated successfully
- 280 cases returned Groq rate-limit errors
- 252 claims were produced in the successful subset
- Claim support rate: 0.992
- Citation validity: 1.000

This must be classified as:

> Partial infrastructure and pipeline validation, not a valid full benchmark comparison.

It should not be combined with the 200-query retrieval metrics.

---

# 14. Provider and reliability experiments

We also tested the operational reliability of:

- OpenRouter
- Gemini
- Groq

The project encountered:

- HTTP 429 rate limits
- HTTP 402 billing restrictions
- Connection resets
- Provider timeouts
- Model availability errors
- SDK dependency issues
- Embedding environment issues

The evaluator was improved with:

- Four-second request pacing
- Retry and exponential backoff
- 120-second request timeout
- Connection retry
- Timeout retry
- Rate-limit retry
- Atomic cache writes
- Resume-from-cache behavior
- Explicit provider error records

This belongs in the paper’s **Reproducibility and Operational Constraints** section.

It is especially important because two LLM calls are made per query:

```text
Router call
+
Structured query parsing call
```

Therefore, a 200-case experiment can require approximately 400 provider requests.

---

# 15. How the experiments should be organized in the paper

## Main results

Use these as the primary results:

1. Full 322-case dense retrieval baseline
2. Full 322-case filtered retrieval
3. Full 322-case dense/BM25/hybrid comparison
4. 60-case reference-entity validation
5. 200-case final LLM-routed retrieval experiment

## Ablation and development results

Use these to explain system evolution:

1. Initial malformed-output 60-case run
2. Registry-backed 60-case run
3. Strict-section five-query run
4. Strict-section 60-query run
5. Reference-entity schema improvement
6. Qwen model experiment

## Provider feasibility results

Use these in a separate subsection:

1. OpenRouter structured-output experiment
2. Gemini structured-output experiment
3. Groq provider experiment
4. Rate-limit and quota observations
5. Retry and cache-resume design

## Pipeline validation only

Mention, but do not present as final comparative evidence:

1. Three-case grounded-answer development run
2. Quota-limited grounded-answer full-run attempt
3. Single-query smoke tests
4. Five-query model smoke tests

---

# 16. The overall experimental story

The experiments form a clear progression:

```text
Initial dense retrieval baseline
        |
        v
Metadata-filtered retrieval
        |
        v
BM25 and hybrid retrieval comparison
        |
        v
LLM-routed retrieval
        |
        v
Failure analysis:
malformed constraints and unknown entities
        |
        v
Deterministic entity registry
        |
        v
Strict canonical section contract
        |
        v
Cross-domain reference entity schema
        |
        v
Stable 60-query validation
        |
        v
Stable 200-query paper benchmark
```

The key research story is:

> Initial retrieval quality was limited by both exact-section retrieval and unreliable LLM-generated structured queries. Dense/BM25 fusion alone did not solve the problem. Deterministic entity resolution, strict schema validation, canonical section normalization, and explicit reference-entity modeling were then introduced. After these changes, the system achieved complete validated coverage and perfect retrieval metrics on the 200-query benchmark.

That progression is valuable because it shows not only the final result, but also **why each architectural component was necessary**.

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
