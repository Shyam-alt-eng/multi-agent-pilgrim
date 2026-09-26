# PilgrimAI --- RAG Multi-Agent System

## Progress Report & Future Plan

> **Status:** Phase 1 complete, Phase 2 completed, Phase 3 started\
> **Current stopping point:** Phase 3 --- document chunking strategy for
> long temple documents

------------------------------------------------------------------------

## 1. Project Overview

PilgrimAI is being developed as a Telangana-focused pilgrimage
assistance system using a multi-agent RAG architecture.

The planned system contains four main domain agents:

-   **Travel Agent**
-   **Accommodation Agent**
-   **Food Agent**
-   **Emergency Agent**

A central orchestrator/supervisor will understand the user's request,
decide which agents are needed, coordinate retrieval, combine the
results, and generate the final response.

The overall architecture we established is:

``` text
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

------------------------------------------------------------------------

# 2. What We Accomplished

## Phase 1 --- Data Understanding & Normalization ✅

The cleaned PilgrimAI dataset was inspected and validated.

The dataset contains:

-   23 temple JSON files
-   136 accommodation JSON files
-   120 restaurant JSON files
-   23 emergency JSON files
-   1 temple registry

There are **303 JSON files in total**, including the registry, and **302
actual entity JSON files**.

Important validation results:

-   No duplicate accommodation IDs
-   No `:contentReference[oaicite:0]{index=0}` artifacts
-   No malformed JSON files
-   `temple_id` was identified as the central relationship between the
    major domains

The relationship is:

``` text
                    Temple
                      |
                  temple_id
          ┌───────────┼───────────┐
          ↓           ↓           ↓
   Accommodation   Restaurant   Emergency
```

This relationship will later allow the agents and orchestrator to
connect information belonging to the same pilgrimage destination.

------------------------------------------------------------------------

# 3. Phase 2 --- Document + Metadata Preparation ✅

We created a document-building pipeline that converts the cleaned JSON
data into LangChain `Document` objects.

The output is:

``` text
data/
└── phase2_documents.jsonl
```

The generated JSONL contains **419 documents**.

Each document has:

``` text
page_content
metadata
```

### Document types

The generated documents represent:

-   Temples
-   Accommodation
-   Restaurants
-   Emergency facilities

Emergency information is represented at the **individual facility
level**, rather than keeping all emergency facilities for a temple
inside one document.

This is important because emergency searches may need to filter
facilities by properties such as:

-   ambulance availability
-   trauma care
-   ICU
-   pharmacy
-   24x7 availability
-   distance
-   verification status

------------------------------------------------------------------------

# 4. Metadata Design

We separated information into two conceptual areas.

### Metadata

Metadata contains structured information useful for filtering.

Examples:

``` text
source_type
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

### Page Content

`page_content` contains the descriptive information that should
participate in semantic search.

Examples include:

``` text
overview
history
religious significance
hotel description
restaurant description
food information
emergency facility details
```

The principle we established is:

``` text
Metadata
    ↓
Exact / structured filtering

page_content
    ↓
Semantic retrieval
```

This separation will become important when we build the retrieval layer.

------------------------------------------------------------------------

# 5. Restaurant Price Decision

During inspection of the generated documents, we noticed that restaurant
price information can contain separate values such as:

``` text
averageCostForOne: ₹100–200
averageCostForTwo: ₹250–400
```

The current Phase 2 output was intentionally **left unchanged for now**,
as decided today.

We identified that future refinement could make the restaurant price
metadata more explicit, for example:

``` text
cost_per_person_min
cost_per_person_max
```

instead of relying on an ambiguous generic `price_min` / `price_max`.

This is recorded as a possible improvement rather than a change made
today.

------------------------------------------------------------------------

# 6. Phase 3 --- Embeddings + ChromaDB ⏳

Phase 3 has now started.

The objective is to convert our prepared documents into vector
representations and store them in ChromaDB.

The planned flow is:

``` text
phase2_documents.jsonl
        ↓
Document preparation
        ↓
Embedding model
        ↓
Vector embeddings
        ↓
ChromaDB
        ↓
Semantic retrieval
```

------------------------------------------------------------------------

# 7. Embedding Model Selection ✅

We evaluated the following options:

-   `intfloat/multilingual-e5-base`
-   `intfloat/multilingual-e5-large`
-   `BAAI/bge-m3`
-   Indic/Telugu-specific sentence-transformer models
-   OpenAI embedding models

For the current implementation, we selected:

``` text
intfloat/multilingual-e5-base
```

### Reason for the selection

The project needs:

-   multilingual retrieval
-   English and Telugu support
-   local execution
-   practical resource usage
-   semantic search
-   compatibility with our ChromaDB architecture

The model produces **768-dimensional embeddings**.

E5 also uses different prefixes for passages and queries:

``` text
Documents:
passage: <document text>

Queries:
query: <search text>
```

This requirement will be respected in the embedding layer.

------------------------------------------------------------------------

# 8. Phase 3 Dependencies ✅

The following packages were installed using `uv`:

``` text
sentence-transformers
langchain-huggingface
langchain-chroma
chromadb
```

The project is using:

``` text
pyproject.toml
uv.lock
```

rather than manually maintaining a `requirements.txt`.

The environment is:

``` text
.venv
```

and the project is managed using `uv`.

------------------------------------------------------------------------

# 9. Embedding Model Test ✅

The selected model was successfully downloaded and tested.

The test produced:

``` text
Embedding dimension: 768
```

This confirms that:

-   the embedding package works
-   the Hugging Face model loads correctly
-   the model can generate embeddings
-   the expected vector dimension is 768

The embedding layer itself is therefore working.

------------------------------------------------------------------------

# 10. Document Length Inspection

Before embedding the complete dataset, we inspected the generated
documents.

Results:

``` text
Documents:          419
Shortest document:  6 words
Longest document:   8119 words
Average:            438.65 words
Over 400 words:     23
Over 500 words:     23
```

We then inspected the documents over 500 words.

The long documents were temple documents:

``` text
T0004 — Sri Gnana Saraswathi Devasthanam — 4468 words
T0001 — Sri Lakshmi Narasimha Swamy Temple — 4260 words
T0011 — Sri Ramalingeswara Swamy Temple — 5365 words
T0003 — Sri Raja Rajeshwara Swamy Devasthanam — 4395 words
T0005 — Sri Kondagattu Anjaneya Swamy Temple — 4441 words
T0002 — Sri Sita Ramachandra Swamy Temple — 4649 words
T0019 — Manyamkonda Lakshmi Venkateswara Swamy Temple — 4512 words
```

This revealed an important issue before ChromaDB creation:

**We should not embed these long documents as one single vector.**

Doing so could result in information being truncated by the embedding
model.

------------------------------------------------------------------------

# 11. Current Stopping Point

We have **not yet created the ChromaDB database**.

We stopped at the correct point because we discovered that the long
temple documents need a chunking strategy.

The current state is:

``` text
Phase 1 — Data Understanding & Normalization
                    ✅

Phase 2 — Document + Metadata Preparation
                    ✅

Phase 3 — Embeddings + ChromaDB
                    🟡

    Model selection              ✅
    Dependencies                 ✅
    Model test                   ✅
    Document length inspection   ✅
    Chunking strategy            ⏳ NEXT
    ChromaDB creation            ⏳
    Retrieval testing            ⏳
```

------------------------------------------------------------------------

# 12. What We Will Do Next

## Step 1 --- Inspect the structure of a long temple document

We will inspect a document such as:

``` text
T0001 — Sri Lakshmi Narasimha Swamy Temple
```

The purpose is to understand its actual section structure.

We do not want arbitrary character-based splitting.

------------------------------------------------------------------------

## Step 2 --- Design the chunking strategy

The likely strategy is conditional chunking:

``` text
Document
   |
   ├── Short document
   |       ↓
   |    1 vector
   |
   └── Long document
           ↓
        sections/chunks
           ↓
       multiple vectors
```

We will preserve the original identity through metadata.

For example:

``` json
{
  "source_type": "temple",
  "temple_id": "T0001",
  "chunk_id": 0
}
```

Therefore, even though ChromaDB searches individual chunks, the system
can still identify the original temple.

------------------------------------------------------------------------

# 13. Future Phase 3 Work

After chunking:

``` text
Chunks
   ↓
multilingual-e5-base
   ↓
768-dimensional vectors
   ↓
ChromaDB
```

Then we will test:

### Semantic search

Example:

``` text
"Tell me about temples associated with Lord Narasimha"
```

### Metadata filtering

Example:

``` text
source_type = temple
temple_id = T0001
```

### Combined retrieval

Example:

``` text
Semantic query
+
metadata filter
```

This will establish the basic knowledge layer.

------------------------------------------------------------------------

# 14. Phase 4 --- Generic Retrieval Engine

After ChromaDB works, we will create a reusable retrieval layer.

Conceptually:

``` text
Search Request
      ↓
Generic Retriever
      ↓
ChromaDB
      ↓
Relevant Documents
```

The retrieval layer should not contain Travel-specific or Food-specific
business logic.

It should be generic and reusable.

------------------------------------------------------------------------

# 15. Phase 5 --- Agent-Specific Retrieval Tools

After the generic retriever works, we will create tools for:

``` text
Travel Retrieval
Accommodation Retrieval
Food Retrieval
Emergency Retrieval
```

Each tool will translate an agent's structured request into retrieval
operations.

------------------------------------------------------------------------

# 16. Phase 6 --- Pydantic Search Schemas

We already designed the intended search contracts.

Example:

``` python
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

Similar contracts will exist for Travel, Food, and Emergency.

These schemas will provide a clean interface between:

``` text
LLM / Agent
      ↓
Structured Search Request
      ↓
Retrieval Tool
```

------------------------------------------------------------------------

# 17. Phase 7 --- Individual Agents

We will then build:

``` text
Travel Agent
Accommodation Agent
Food Agent
Emergency Agent
```

Each agent will have:

-   a clear responsibility
-   its own retrieval tool
-   its own search schema
-   relevant context
-   a controlled output structure

------------------------------------------------------------------------

# 18. Phase 8 --- Orchestrator / Supervisor

The orchestrator will handle the overall user request.

Its responsibilities include:

``` text
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

The LLM will **not directly execute ChromaDB operations**.

Instead:

``` text
LLM
 ↓
Search request
 ↓
Retrieval tool
 ↓
ChromaDB
 ↓
Results
```

This separation keeps retrieval deterministic and easier to test.

------------------------------------------------------------------------

# 19. Phase 9 --- Final Answer Generation

Once the agents return their results, the orchestrator will combine
them.

For example, a request such as:

``` text
"I am visiting Yadadri. Find the temple,
a nearby hotel, a vegetarian restaurant,
and emergency facilities."
```

could require:

``` text
Travel Agent
       ↓
Accommodation Agent
       ↓
Food Agent
       ↓
Emergency Agent
       ↓
       Orchestrator
       ↓
Final response
```

The central `temple_id` relationship will help connect the retrieved
information to the same pilgrimage destination.

------------------------------------------------------------------------

# 20. Phase 10 --- End-to-End Testing

Finally, we will test complete user queries covering:

-   temple discovery
-   accommodation search
-   restaurant search
-   emergency search
-   cross-agent queries
-   temple-to-hotel relationships
-   temple-to-restaurant relationships
-   temple-to-emergency relationships
-   semantic queries
-   structured filters
-   multilingual queries where supported

The final objective is:

``` text
User Query
    ↓
Orchestrator
    ↓
Required Agents
    ↓
Retrieval
    ↓
ChromaDB
    ↓
Agent Results
    ↓
Orchestrator
    ↓
Final Answer
```

------------------------------------------------------------------------

# 21. Project Status at the End of Today

``` text
                 PILGRIMAI
                     │
                     ▼
        ┌────────────────────────┐
        │ Phase 1: Data Cleaning │
        │          ✅            │
        └────────────┬───────────┘
                     ▼
        ┌────────────────────────┐
        │ Phase 2: Documents     │
        │ + Metadata             │
        │          ✅            │
        └────────────┬───────────┘
                     ▼
        ┌────────────────────────┐
        │ Phase 3: Embeddings    │
        │ + ChromaDB             │
        │          🟡            │
        └────────────┬───────────┘
                     │
                     ▼
             CURRENT STEP
          Inspect long temple
          document structure
                     │
                     ▼
              Chunking design
                     │
                     ▼
             ChromaDB creation
                     │
                     ▼
            Retrieval engine
                     │
                     ▼
          Agent-specific tools
                     │
                     ▼
             Pydantic schemas
                     │
                     ▼
               4 Agents
                     │
                     ▼
             Orchestrator
                     │
                     ▼
             Final responses
                     │
                     ▼
              End-to-end tests
```

------------------------------------------------------------------------

## Tomorrow's Starting Point

**We will start exactly here:**

> **Inspect the structure of the T0001 long temple document and design
> the chunking strategy.**

We will **not jump directly into ChromaDB** until the chunking strategy
is settled and tested.

Once the chunks are correct, the next sequence will be:

``` text
Chunking
   ↓
Embedding
   ↓
ChromaDB
   ↓
Semantic search
   ↓
Metadata filtering
   ↓
Generic retriever
```

That is the current project state and the planned path forward.
