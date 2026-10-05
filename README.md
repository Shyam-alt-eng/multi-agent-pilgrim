# PilgrimAI — RAG Multi-Agent Research System

## Current status

Validated through the dense retrieval / section-aware retrieval stage. The repository is organized so that retrieval experiments can be added without mixing ingestion, embeddings, vector storage, prompting, LLM integration, and API code.

## Research direction

The project is being developed as a research system, not only as an application. Planned experiments compare: 

1. Standard dense/vector RAG.
2. Hybrid dense + BM25 retrieval with fusion and reranking.
3. Domain-routed hybrid retrieval.
4. Single-agent versus multi-agent pilgrimage planning.
5. Deterministic constraint validation and replanning.

The experimental comparisons will use the same benchmark queries and evaluation protocol for controlled comparisons.

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

## Next research implementation

The next component is natural-language query understanding: converting a user query into structured fields such as domain, entity, intent, section, and constraints. That layer will feed the generic retrieval engine rather than directly accessing ChromaDB.

## Important separation

LLMs will be used for language understanding, reasoning, and candidate planning. Deterministic code will be used where possible for exact filtering and validation of temporal, spatial, budget, ordering, and other hard constraints.
