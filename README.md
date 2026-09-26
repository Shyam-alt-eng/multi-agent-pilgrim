# PilgrimAI — RAG Multi-Agent System

## Progress Report, Decisions, Commands & Future Plan

> **Current status:** Phases 1–3 complete, ChromaDB ingestion complete, retrieval smoke testing complete.  
> **Current stopping point:** Phase 4 retrieval evaluation across all domains.  
> **Next implementation:** Complete retrieval evaluation, then build the generic retrieval engine.

---

# 1. Project Overview

PilgrimAI is a Telangana-focused pilgrimage assistance system using a multi-agent RAG architecture.

The planned system contains four main domain agents:

- **Travel Agent**
- **Accommodation Agent**
- **Food Agent**
- **Emergency Agent**

A central orchestrator/supervisor will understand the user's request, decide which agents are needed, coordinate retrieval, combine the results, and generate the final response.

The overall architecture is:

```text
1. Data Understanding & Normalization
2. Document + Metadata Preparation
3. Embedding + ChromaDB Knowledge Layer
4. Generic Retrieval Engine
5. Agent-Specific Retrieval Tools
6. Pydantic Search Schemas
7. Individual Agents
8. Orchestrator / Supervisor
9. Final Answer Generation
10. End-to-End Testing
```

---

# 2. Current Project Status

```text
Phase 1 — Data Understanding & Normalization
                    ✅ COMPLETE

Phase 2 — Document + Metadata Preparation
                    ✅ COMPLETE
                    └── 419 documents

Phase 3 — Chunking
                    ✅ COMPLETE
                    └── 1,805 chunks
                    └── 0 chunks > 450 tokens
                    └── 0 chunks > 512 tokens

Phase 3 — Embedding Model Selection
                    ✅ COMPLETE
                    └── intfloat/multilingual-e5-base
                    └── 768 dimensions

Phase 4 — ChromaDB Ingestion
                    ✅ COMPLETE
                    └── 1,805 records
                    └── collection: pilgrimai

Phase 4 — Retrieval Smoke Testing
                    ✅ COMPLETE
                    ├── semantic search
                    ├── metadata filtering
                    ├── temple_id filtering
                    └── controlled multi-query testing

Phase 4 — Full Retrieval Evaluation
                    🟡 CURRENT

Phase 5 — Generic Retrieval Engine
                    ⏳ NEXT

Phase 6 — Agent-Specific Retrieval Tools
                    ⏳

Phase 7 — Pydantic Search Schemas
                    ⏳

Phase 8 — Individual Agents
                    ⏳

Phase 9 — Orchestrator / Supervisor
                    ⏳

Phase 10 — End-to-End Testing
                    ⏳
```

---

# 3. Project Structure

Current important files:

```text
multi-agent-pilgrim/
│
├── .venv/
├── pyproject.toml
├── uv.lock
│
└── data/
    ├── phase2_document_builder.py
    ├── phase2_documents.jsonl
    │
    ├── phase3_chunk_builder_v2.py
    ├── phase3_chunks.jsonl
    │
    ├── phase4_chromadb_ingest.py
    ├── phase4_retrieval_test.py
    │
    └── chroma_db/
```

The project is managed using `uv`.

There is no need to manually maintain a `requirements.txt` for the current setup.

---

# 4. Phase 1 — Data Understanding & Normalization ✅

The cleaned PilgrimAI dataset was inspected and validated.

Dataset:

```text
23 temple JSON files
136 accommodation JSON files
120 restaurant JSON files
23 emergency JSON files
1 temple registry
```

Total:

```text
303 JSON files
302 actual entity JSON files
1 registry
```

Validation results:

```text
Duplicate accommodation IDs        : 0
Content-reference artifacts        : 0
Malformed JSON files               : 0
```

The central relationship identified in the data is:

```text
                    Temple
                      |
                  temple_id
          ┌───────────┼───────────┐
          ↓           ↓           ↓
   Accommodation   Restaurant   Emergency
```

`temple_id` is therefore one of the most important fields in the entire retrieval architecture.

It allows information from different domains to be connected to the same pilgrimage destination.

---

# 5. Phase 2 — Document + Metadata Preparation ✅

The cleaned JSON data was converted into LangChain `Document` objects.

Output:

```text
data/phase2_documents.jsonl
```

Generated documents:

```text
419
```

Each document contains:

```text
page_content
metadata
```

Document types:

```text
Temple
Accommodation
Restaurant
Emergency facility
```

Emergency information is represented at the individual facility level.

This allows later retrieval to distinguish facilities based on:

```text
ambulance availability
trauma care
ICU
pharmacy
24x7 availability
distance
verification status
```

---

# 6. Metadata Design

The design separates structured filtering information from semantic text.

## Metadata

Metadata contains values that may be used for exact or structured filtering.

Examples:

```text
source_type
entity_id
temple_id
hotel_id
restaurant_id
facility_id
location
religion
temple_type
category
deity
rating
distance_km
price-related fields
amenities
dietary options
facility information
```

## Page Content

`page_content` contains descriptive information used for semantic retrieval.

Examples:

```text
overview
history
religious significance
hotel descriptions
restaurant descriptions
food information
emergency facility descriptions
darshan information
travel information
FAQ information
```

The core principle is:

```text
Metadata
    ↓
Exact / structured filtering

page_content
    ↓
Semantic retrieval
```

This separation becomes important when the generic retrieval engine and agent tools are built.

---

# 7. Restaurant Price Decision

During Phase 2 inspection, restaurant prices were found in forms such as:

```text
averageCostForOne: ₹100–200
averageCostForTwo: ₹250–400
```

We identified that a generic:

```text
price_min
price_max
```

can become ambiguous when multiple pricing bases exist.

A future improvement could use:

```text
cost_per_person_min
cost_per_person_max
```

However:

> **The current Phase 2 mapping was intentionally left unchanged.**

This is recorded as a future refinement rather than a current implementation change.

---

# 8. Phase 3 — Embedding Model Selection ✅

Several model options were considered:

```text
intfloat/multilingual-e5-base
intfloat/multilingual-e5-large
BAAI/bge-m3
Indic/Telugu-specific sentence-transformer models
OpenAI embedding models
```

Selected model:

```text
intfloat/multilingual-e5-base
```

## Why this model was selected

The project requires:

- multilingual retrieval
- English and Telugu support
- local execution
- practical resource usage
- semantic search
- ChromaDB compatibility

The model produces:

```text
768-dimensional embeddings
```

---

# 9. E5 Prefix Rule

E5 requires different prefixes for documents and queries.

For documents:

```text
passage: <document text>
```

For queries:

```text
query: <search text>
```

The ingestion and retrieval scripts therefore use a custom embedding wrapper:

```python
class E5Embeddings(HuggingFaceEmbeddings):

    def embed_documents(self, texts):
        texts = [f"passage: {text}" for text in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        text = f"query: {text}"
        return super().embed_query(text)
```

This rule must remain consistent between ingestion and retrieval.

---

# 10. Important Embedding Model Decision

A question came up:

> Should we change the embedding model because some source documents are very large?

The conclusion was:

> **No. Long source documents should first be handled through appropriate chunking.**

A model change should only be considered if retrieval evaluation demonstrates an actual problem such as:

- poor Telugu retrieval
- poor mixed Telugu/English retrieval
- consistently irrelevant top-k results
- poor Recall@k
- poor MRR
- unacceptable embedding speed
- unacceptable resource usage

Changing the embedding model later would require rebuilding the vector collection because embeddings from different models are not interchangeable.

---

# 11. Phase 3 Dependencies ✅

Packages installed using `uv`:

```text
sentence-transformers
langchain-huggingface
langchain-chroma
chromadb
```

Important environment files:

```text
pyproject.toml
uv.lock
.venv/
```

Useful verification:

```powershell
uv run python -c "import chromadb; print(chromadb.__version__)"
```

The installed ChromaDB version used during this phase was:

```text
1.5.9
```

---

# 12. Embedding Model Test ✅

The embedding model was loaded and tested successfully.

Result:

```text
Embedding dimension: 768
```

This confirmed:

```text
Embedding package      → working
Hugging Face model     → loading
Embedding generation   → working
Vector dimension       → 768
```

---

# 13. Document Length Inspection

Initial Phase 2 document inspection:

```text
Documents:              419
Shortest:               6 words
Longest:                8119 words
Average:                438.65 words
Documents > 400 words:  23
Documents > 500 words:  23
```

Domain-level inspection showed that temple documents were much larger than the other domains.

Approximate characteristics:

```text
Temple
    23 documents
    Min: 3,220 words
    Median: 4,829
    Average: 5,126
    Max: 8,119

Accommodation
    136 documents
    Average: ~248 words

Restaurant
    120 documents
    Average: ~226 words

Emergency
    140 documents
    Average: ~36 words
```

This showed that embedding every logical entity as one vector would not be appropriate.

---

# 14. Chunking Strategy ✅

The final strategy is domain-aware and token-aware.

```text
                    Phase 2 Documents
                           │
              ┌────────────┴────────────┐
              │                         │
        source_type                 document size
              │                         │
              ▼                         ▼
        Domain-aware             token-limit check
         strategy
              │
    ┌─────────┼──────────┬────────────┐
    ▼         ▼          ▼            ▼
 Temple    Hotel      Restaurant   Emergency
    │         │          │            │
    ▼         ▼          ▼            ▼
 Semantic   Whole     Whole when   Whole when
 chunks     entity    small enough small enough
    │                    │            │
    │                    └────┬───────┘
    │                         │
    │                    If too large
    │                         ↓
    │                    split at field
    │                    boundaries
    │
    └── if a section is too large
             ↓
       split that section
       at field/line boundaries
```

---

# 15. Temple Chunking

Temple documents are large and highly structured.

They are divided into semantic sections:

```text
identity
location
overview
history
significance
architecture
worship
visitor
accommodation
travel
nearby
accessibility_contact
reviews
faq
quality
```

Example chunk IDs:

```text
T0001__identity__part_1
T0001__location__part_1
T0001__history__part_1
T0001__worship__part_1
T0001__worship__part_2
T0001__travel__part_1
```

This keeps related information together.

For example:

```text
darshan_timings
darshan_and_tickets
sevas
rituals
festivals
```

are grouped under the worship area instead of being separated using arbitrary character boundaries.

---

# 16. Accommodation, Restaurant and Emergency Chunking

The preferred rule is:

```text
One logical entity
        ↓
One chunk
```

**if the entity fits within the token budget.**

If the entity is too large:

```text
One logical entity
        ↓
Multiple field-based chunks
```

The splitting happens at field/line boundaries.

This is important because the actual dataset contained restaurant records much larger than initially expected.

---

# 17. First Chunking Attempt — What Went Wrong

The first Phase 3 implementation targeted approximately:

```text
380 tokens
```

for the body.

However, after adding the semantic header, some chunks became larger than the intended limit.

Initial result:

```text
Input documents : 419
Output documents: 1358

Temple chunks: 962
Max tokens: 569
Chunks over 450: 258
```

Examples:

```text
T0006__reviews__part_2 → 557 tokens
T0006__reviews__part_3 → 569 tokens
T0001__reviews__part_2 → 542 tokens
```

Some restaurant records were also much larger:

```text
restaurant__R0080 → 952 tokens
restaurant__R0113 → 1161 tokens
```

## Cause

The first implementation effectively did:

```text
body ≈ 380 tokens
        +
semantic header
        =
more than the intended limit
```

This was a chunking implementation issue, not proof that the embedding model was unsuitable.

---

# 18. Final Chunking Implementation ✅

The Phase 3 v2 chunker fixed the previous problems.

It:

- uses the actual E5 tokenizer
- counts tokens instead of relying only on word count
- reserves token space for semantic headers
- splits oversized sections at field/line boundaries
- can split an individual oversized field as a last resort
- keeps smaller non-temple entities whole
- splits oversized non-temple entities when necessary
- preserves stable identifiers
- performs final safety checks

Configuration:

```text
Embedding model: intfloat/multilingual-e5-base
Target tokens:   380
Maximum tokens:  450
```

Final output:

```text
Input documents : 419
Output chunks   : 1805
```

Breakdown:

```text
accommodation : 318
emergency     : 140
restaurant    : 381
temple        : 966
```

Temple chunk statistics:

```text
Min words  : 15
Max words  : 297
Avg words  : 132.73

Min tokens : 34
Max tokens : 405
Avg tokens : 298.11
```

Safety checks:

```text
Chunks > 450 tokens : 0
Chunks > 512 tokens : 0
```

Therefore:

> **Phase 3 chunking is complete.**

---

# 19. Token Limit Clarification

A key clarification:

```text
Embedding model input limit
            ≠
recommended retrieval chunk size
```

Even if another embedding model accepts a larger input, a huge heterogeneous entity can produce a less focused retrieval vector.

For example, a restaurant can contain:

```text
location
contact
working hours
food
dietary information
pricing
facilities
nearby places
reviews
```

A query about vegetarian food should retrieve information relevant to food/dietary requirements rather than relying on one giant embedding containing everything.

Therefore:

> **Chunking is a retrieval-design decision, not merely a workaround for a token limit.**

---

# 20. Stable IDs and Update/Delete Design

ChromaDB is the retrieval/index layer, not the master source of truth.

The intended flow is:

```text
Source JSON / structured database
          ↓
Document builder
          ↓
Chunker
          ↓
Embeddings
          ↓
ChromaDB
```

Stable chunk IDs were designed for update/delete operations.

Examples:

```text
T0001__identity__part_1
T0001__location__part_1
T0001__history__part_1
T0001__worship__part_1
T0001__worship__part_2
```

Each chunk also preserves information such as:

```text
entity_id
temple_id
section
chunk_id
section_part
section_parts
```

## Delete an entire temple

Conceptually:

```text
temple_id = T0001
        ↓
delete all matching chunks
```

## Update one section

For example:

```text
temple_id = T0001
section   = worship
        ↓
delete old worship chunks
        ↓
generate new worship chunks
        ↓
embed them
        ↓
insert new chunks
```

The important rule is:

> Remove old logical-section records before inserting regenerated chunks.

This prevents stale chunks from remaining when the number of chunks changes.

---

# 21. Phase 4 — ChromaDB Knowledge Layer ✅

The ChromaDB ingestion script is:

```text
data/phase4_chromadb_ingest.py
```

It:

1. Loads `phase3_chunks.jsonl`
2. Loads `intfloat/multilingual-e5-base`
3. Adds the E5 `passage:` prefix to documents
4. Generates 768-dimensional embeddings
5. Creates a persistent ChromaDB collection
6. Stores all chunks and metadata
7. Uses stable `chunk_id` values as Chroma IDs
8. Verifies the stored record count

Configuration:

```text
Collection:
pilgrimai
```

Database path:

```text
data/chroma_db/
```

Final ingestion result:

```text
Input chunks      : 1805
Embedding model   : intfloat/multilingual-e5-base
Collection        : pilgrimai
ChromaDB records  : 1805
Count verification: PASS
```

Therefore:

> **ChromaDB ingestion is complete.**

---

# 22. ChromaDB Metadata Filtering

During retrieval testing, the following filter initially caused an error:

```python
filter={
    "source_type": "temple",
    "temple_id": "T0001"
}
```

The ChromaDB version being used expects combined conditions using `$and`.

Correct form:

```python
filter={
    "$and": [
        {"source_type": "temple"},
        {"temple_id": "T0001"}
    ]
}
```

This is now used in the retrieval test script.

---

# 23. Phase 4 — Retrieval Smoke Testing ✅

Retrieval test script:

```text
data/phase4_retrieval_test.py
```

It loads the same:

```text
embedding model
ChromaDB collection
E5 query prefix
```

used by ingestion.

The helper function uses:

```python
vectorstore.similarity_search_with_score(
    query,
    k=k,
    filter=filter_dict
)
```

Important:

```text
k=5
```

means:

> Search the entire collection and return the top 5 results.

It does **not** mean that only 5 records exist.

Current ChromaDB contains:

```text
1805 records
```

For example, the test suite uses:

```text
Test 1 → k=5
Test 2 → k=5
Test 3 → k=5
Test 4 → k=5
Test 5 → k=5
Test 6 → k=10
```

If every test returns enough results, the script prints:

```text
5 + 5 + 5 + 5 + 5 + 10 = 35 result records
```

These are retrieval results, not the total database size.

---

# 24. Retrieval Smoke-Test Results

The current smoke tests demonstrate that retrieval is functioning.

Examples:

## Darshan query

Query:

```text
darshan timings and tickets
```

with:

```text
source_type = temple
temple_id = T0001
```

returned relevant T0001 worship chunks, including the official Sarva Darshan timing information.

## History query

Query:

```text
history of Yadagirigutta temple
```

returned:

```text
T0001__history__part_1
T0001__history__part_2
```

as the first two results.

## Deity query

Query:

```text
Lord Lakshmi Narasimha temple
```

retrieved both:

```text
T0017 — Dharmapuri
T0001 — Yadadri/Yadagirigutta
```

This is expected because both are Lakshmi Narasimha temples.

## Controlled T0001 retrieval

Filtering directly to T0001 returns relevant sections such as:

```text
significance
overview
reviews
worship
history
nearby
FAQ
```

This confirms that semantic retrieval and metadata filtering are functioning together.

---

# 25. Retrieval Score Clarification

The test uses:

```python
similarity_search_with_score()
```

The returned score is a **distance**, not an accuracy percentage.

Generally:

```text
Lower distance
      ↓
closer vector match
```

Therefore:

```text
0.25
```

does **not** mean:

```text
25% accuracy
```

and:

```text
0.90
```

does not directly mean:

```text
90% accuracy
```

Retrieval quality should instead be evaluated using metrics such as:

```text
Recall@1
Recall@3
Recall@5
MRR
```

and manual relevance checks where appropriate.

---

# 26. Important Retrieval Evaluation Decision

The current retrieval tests are **smoke tests**, not a complete evaluation.

We should not judge the embedding model from one temple or one query.

The evaluation needs to cover the complete dataset:

```text
Temple
Accommodation
Restaurant
Emergency
```

and multiple query types.

Example queries:

```text
restaurants near Basara
hotels near Yadadri
24x7 hospital near Vemulawada
history of Ramappa temple
darshan timings at Bhadrachalam
vegetarian restaurants near Basara
accommodation near Kondagattu
```

The evaluation should measure whether the expected entity/domain appears in:

```text
Top-1
Top-3
Top-5
```

before deciding whether the embedding model needs to change.

---

# 27. Why We Are Not Changing the Embedding Model Yet

The fact that one query may return another relevant temple does not automatically mean E5 is bad.

For example:

```text
"Lord Lakshmi Narasimha temple"
```

can legitimately match multiple Lakshmi Narasimha temples.

The correct evaluation question is:

```text
Given a query with a known expected answer,
does the expected relevant chunk/entity appear
within the required top-k results?
```

Therefore:

```text
Smoke testing
      ↓
Full retrieval evaluation
      ↓
Only then decide whether E5 needs changing
```

---

# 28. Phase 4 — Current Status

```text
Phase 4.1 — ChromaDB setup
                    ✅

Phase 4.2 — Ingest 1,805 chunks
                    ✅

Phase 4.3 — Verify collection count
                    ✅

Phase 4.4 — Semantic retrieval smoke test
                    ✅

Phase 4.5 — Metadata filtering
                    ✅

Phase 4.6 — Controlled retrieval tests
                    ✅

Phase 4.7 — Full multi-domain retrieval evaluation
                    🟡 CURRENT
```

---

# 29. Generic Retrieval Engine — Next Phase

After retrieval evaluation is complete, build a reusable generic retriever.

Conceptually:

```text
Search Request
      ↓
Generic Retriever
      ↓
ChromaDB
      ↓
Relevant Documents
```

The generic retriever should handle:

- query embedding
- semantic search
- metadata filters
- top-k
- result formatting
- score/distance
- stable metadata
- optional source/domain filtering

It should **not** contain Travel-specific or Food-specific business logic.

---

# 30. Agent-Specific Retrieval Tools

After the generic retriever:

```text
Travel Retrieval
Accommodation Retrieval
Food Retrieval
Emergency Retrieval
```

Each domain tool will convert a structured agent request into a retrieval operation.

The generic retriever remains reusable underneath.

---

# 31. Pydantic Search Schemas

The intended search contracts are already designed.

Example:

```python
{
    "agent": "accommodation",
    "query": "...",
    "filters": {
        "temple_id": "...",
        "location": "...",
        "price": "...",
        "distance": "...",
        "rating": "..."
    },
    "top_k": 5
}
```

Similar contracts will exist for:

```text
Travel
Accommodation
Food
Emergency
```

The architecture becomes:

```text
LLM / Agent
      ↓
Structured Search Request
      ↓
Retrieval Tool
      ↓
Generic Retriever
      ↓
ChromaDB
```

---

# 32. Individual Agents

Planned agents:

```text
Travel Agent
Accommodation Agent
Food Agent
Emergency Agent
```

Each agent will have:

- a clear responsibility
- its own retrieval tool
- its own search schema
- relevant context
- controlled output structure

---

# 33. Orchestrator / Supervisor

The orchestrator will handle the complete user request.

Responsibilities:

```text
Understand user query
        ↓
Identify required agents
        ↓
Break query into tasks
        ↓
Extract constraints
        ↓
Create structured search requests
        ↓
Run required agents
        ↓
Combine results
        ↓
Generate final response
```

The LLM does **not** directly execute ChromaDB operations.

Instead:

```text
LLM
 ↓
Search request
 ↓
Retrieval tool
 ↓
Generic retriever
 ↓
ChromaDB
 ↓
Results
```

This keeps retrieval deterministic and easier to test.

---

# 34. What the LLM Is Responsible For

The LLM can be responsible for:

1. Understanding the user's query
2. Identifying required agents
3. Breaking the query into agent-specific tasks
4. Extracting constraints
5. Converting natural language into structured search requests
6. Deciding semantic vs structured requirements
7. Creating filters
8. Deciding top-k
9. Understanding `temple_id` relationships
10. Deciding which tasks can run independently
11. Combining agent results
12. Generating the final response

The LLM should **not** directly:

```text
execute ChromaDB
generate database filters blindly
perform vector storage
manage embeddings
delete source data
```

Those operations belong to deterministic application code/tools.

---

# 35. Final Answer Generation

Example user request:

```text
I am visiting Yadadri. Find the temple,
a nearby hotel, a vegetarian restaurant,
and emergency facilities.
```

Potential flow:

```text
                    User
                      ↓
               Orchestrator
                      ↓
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
     Travel     Accommodation      Food
     Agent          Agent          Agent
        │             │             │
        └─────────────┼─────────────┘
                      ↓
                Emergency
                   Agent
                      ↓
                 Results
                      ↓
                Orchestrator
                      ↓
               Final Answer
```

`temple_id` helps connect results belonging to the same destination.

---

# 36. End-to-End Testing

The final system should be tested with:

## Temple discovery

```text
Find famous temples near Hyderabad.
```

## Accommodation

```text
Find hotels near Yadadri under a given budget.
```

## Food

```text
Find vegetarian restaurants near Basara.
```

## Emergency

```text
Find a 24x7 hospital near Vemulawada.
```

## Cross-domain

```text
I am visiting Yadadri.
Find the temple, a nearby hotel,
a vegetarian restaurant, and emergency facilities.
```

## Relationship tests

```text
Temple → Hotel
Temple → Restaurant
Temple → Emergency
```

## Retrieval tests

```text
semantic search
metadata filtering
combined search
top-k relevance
multilingual queries
```

## Data update tests

```text
update one temple section
delete one entity
regenerate changed chunks
verify stale chunks are removed
```

---

# 37. Useful Commands

## Create/activate environment

```powershell
uv venv
```

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Install dependencies

```powershell
uv add sentence-transformers langchain-huggingface langchain-chroma chromadb
```

## Run Phase 2

From the project root:

```powershell
python data\phase2_document_builder.py
```

## Run Phase 3 chunking

```powershell
python data\phase3_chunk_builder_v2.py
```

## Build ChromaDB

```powershell
python data\phase4_chromadb_ingest.py
```

## Test retrieval

```powershell
python data\phase4_retrieval_test.py
```

## Check ChromaDB version

```powershell
uv run python -c "import chromadb; print(chromadb.__version__)"
```

## Check installed packages

```powershell
uv pip list
```

## Check Python

```powershell
python --version
```

## Check project dependencies

```powershell
uv tree
```

---

# 38. Useful Verification Commands

Check whether Phase 3 output exists:

```powershell
Test-Path data\phase3_chunks.jsonl
```

Check ChromaDB directory:

```powershell
Test-Path data\chroma_db
```

Check Phase 3 file size:

```powershell
Get-Item data\phase3_chunks.jsonl | Select-Object Name,Length
```

Run the ingestion script and verify:

```text
Input chunks      : 1805
ChromaDB records  : 1805
Count verification: PASS
```

---

# 39. Important Files and Their Responsibilities

```text
phase2_document_builder.py
    ↓
Converts source JSON into normalized LangChain documents.

phase2_documents.jsonl
    ↓
419 normalized source documents.

phase3_chunk_builder_v2.py
    ↓
Creates semantic + size-aware retrieval chunks.

phase3_chunks.jsonl
    ↓
1,805 retrieval chunks.

phase4_chromadb_ingest.py
    ↓
Embeds and stores the 1,805 chunks in ChromaDB.

chroma_db/
    ↓
Persistent vector database.

phase4_retrieval_test.py
    ↓
Read-only retrieval smoke tests.
```

---

# 40. Current Pipeline

The actual pipeline now is:

```text
Source Dataset
      ↓
Phase 1
Data Understanding + Normalization
      ↓
Phase 2
419 LangChain Documents
      ↓
Phase 3
Semantic + Size-Aware Chunking
      ↓
1,805 Chunks
      ↓
E5 Embeddings
intfloat/multilingual-e5-base
      ↓
768-dimensional vectors
      ↓
ChromaDB
collection: pilgrimai
      ↓
1,805 stored records
      ↓
Retrieval
      ↓
CURRENT: Full Retrieval Evaluation
      ↓
Generic Retriever
      ↓
Agent-Specific Retrieval Tools
      ↓
Pydantic Search Schemas
      ↓
4 Domain Agents
      ↓
Orchestrator
      ↓
Final Answer
```

---

# 41. Exact Current Starting Point

The project is **not** starting from ChromaDB anymore.

ChromaDB has already been:

```text
created
loaded
populated
verified
tested
```

The exact current task is:

> **Evaluate retrieval across the complete PilgrimAI dataset before building the generic retrieval engine.**

The evaluation should cover:

```text
Temple
Accommodation
Restaurant
Emergency
```

and should measure:

```text
Top-1
Top-3
Top-5
```

using known expected entities/sections.

Only after this evaluation should we decide whether:

```text
E5 is sufficient
```

or whether the embedding/retrieval strategy needs improvement.

---

# 42. Next Session Plan

Start with:

```text
1. Build a multi-domain retrieval evaluation dataset
2. Run Top-1 / Top-3 / Top-5 evaluation
3. Inspect failures
4. Determine whether failures come from:
      - query wording
      - chunking
      - metadata
      - filtering
      - embedding model
5. Fix only the actual problem
6. Freeze the retrieval layer
7. Build the generic retriever
```

Do **not** change the embedding model simply because a single query produces an unexpected ranking.

---

# 43. Final Project Status

```text
                         PILGRIMAI
                            │
                            ▼
             ┌─────────────────────────┐
             │ Phase 1: Data           │
             │ Understanding            │
             │           ✅             │
             └────────────┬────────────┘
                          ▼
             ┌─────────────────────────┐
             │ Phase 2: Documents      │
             │ + Metadata              │
             │           ✅             │
             │       419 docs           │
             └────────────┬────────────┘
                          ▼
             ┌─────────────────────────┐
             │ Phase 3: Chunking       │
             │           ✅             │
             │      1,805 chunks        │
             └────────────┬────────────┘
                          ▼
             ┌─────────────────────────┐
             │ E5 Embeddings           │
             │           ✅             │
             │       768 dimensions     │
             └────────────┬────────────┘
                          ▼
             ┌─────────────────────────┐
             │ ChromaDB Ingestion      │
             │           ✅             │
             │      1,805 records       │
             └────────────┬────────────┘
                          ▼
             ┌─────────────────────────┐
             │ Retrieval Smoke Tests   │
             │           ✅             │
             └────────────┬────────────┘
                          ▼
             ┌─────────────────────────┐
             │ Full Retrieval          │
             │ Evaluation              │
             │          🟡 CURRENT      │
             └────────────┬────────────┘
                          ▼
             ┌─────────────────────────┐
             │ Generic Retriever       │
             │           ⏳             │
             └────────────┬────────────┘
                          ▼
             ┌─────────────────────────┐
             │ Agent Retrieval Tools   │
             │           ⏳             │
             └────────────┬────────────┘
                          ▼
             ┌─────────────────────────┐
             │ Pydantic Schemas        │
             │           ⏳             │
             └────────────┬────────────┘
                          ▼
             ┌─────────────────────────┐
             │ 4 Domain Agents         │
             │           ⏳             │
             └────────────┬────────────┘
                          ▼
             ┌─────────────────────────┐
             │ Orchestrator            │
             │           ⏳             │
             └────────────┬────────────┘
                          ▼
             ┌─────────────────────────┐
             │ End-to-End Testing      │
             │           ⏳             │
             └─────────────────────────┘
```

---

# 44. One-Line Project Status

> **PilgrimAI currently has 419 normalized source documents, 1,805 size-aware semantic retrieval chunks, multilingual E5 embeddings, a verified persistent ChromaDB collection with 1,805 records, and working semantic + metadata retrieval. The next task is full multi-domain retrieval evaluation before building the generic retrieval engine.**
