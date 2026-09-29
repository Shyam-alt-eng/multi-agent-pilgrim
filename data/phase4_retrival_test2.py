
import sys

sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "pilgrimai"

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


# ============================================================
# E5 EMBEDDINGS
# ============================================================

class E5Embeddings(HuggingFaceEmbeddings):

    def embed_documents(self, texts):
        texts = [f"passage: {text}" for text in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")


# ============================================================
# LOAD CHROMADB
# ============================================================

embeddings = E5Embeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={
        "normalize_embeddings": True
    }
)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
)


# ============================================================
# HELPER
# ============================================================

def run_search(title, query, k=5, filter=None):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print(f"Query : {query}")
    print(f"Top-K : {k}")

    if filter:
        print(f"Filter: {filter}")

    results = vectorstore.similarity_search_with_score(
        query,
        k=k,
        filter=filter
    )

    if not results:
        print("NO RESULTS")
        return

    for i, (doc, score) in enumerate(results, start=1):

        print(f"\n--- Result {i} ---")

        print(f"Score      : {score}")
        print(f"Chunk ID   : {doc.metadata.get('chunk_id')}")
        print(f"Entity ID  : {doc.metadata.get('entity_id')}")
        print(f"Source     : {doc.metadata.get('source_type')}")
        print(f"Temple ID  : {doc.metadata.get('temple_id')}")
        print(f"Location   : {doc.metadata.get('location')}")

        print("\nContent:")
        print(doc.page_content[:700])


# ============================================================
# TEST 1 — TEMPLE SEMANTIC SEARCH
# ============================================================

run_search(
    title="TEST 1 — Temple Semantic Search",
    query="yadagirigutta darshan timings",
    k=5
)


# ============================================================
# TEST 2 — TEMPLE FILTERED SEARCH
# ============================================================

run_search(
    title="TEST 2 — T0001 Temple Filter",
    query="darshan timings",
    k=5,
    filter={
        "$and": [
            {"source_type": "temple"},
            {"temple_id": "T0001"},
        ]
    }
)


# ============================================================
# TEST 3 — ACCOMMODATION
# ============================================================

run_search(
    title="TEST 3 — Accommodation Search",
    query="hotels to stay near Yadagirigutta temple",
    k=5,
    filter={
        "source_type": "accommodation"
    }
)


# ============================================================
# TEST 4 — RESTAURANT
# ============================================================

run_search(
    title="TEST 4 — Restaurant Search",
    query="restaurants near Yadagirigutta temple",
    k=5,
    filter={
        "source_type": "restaurant"
    }
)


# ============================================================
# TEST 5 — EMERGENCY
# ============================================================

run_search(
    title="TEST 5 — Emergency Search",
    query="emergency hospital near Yadagirigutta temple",
    k=5,
    filter={
        "source_type": "emergency"
    }
)


# ============================================================
# TEST 6 — CROSS-DOMAIN SEMANTIC SEARCH
# ============================================================

run_search(
    title="TEST 6 — Cross Domain Search",
    query="places to stay and eat near Yadagirigutta temple",
    k=10
)


# ============================================================
# TEST 7 — TEMPLE + ACCOMMODATION RELATIONSHIP
# ============================================================

run_search(
    title="TEST 7 — T0001 Accommodation",
    query="accommodation",
    k=5,
    filter={
        "$and": [
            {"source_type": "accommodation"},
            {"temple_id": "T0001"},
        ]
    }
)


# ============================================================
# TEST 8 — DATABASE COUNT
# ============================================================

print("\n" + "=" * 80)
print("TEST 8 — CHROMADB COUNT")
print("=" * 80)

count = vectorstore._collection.count()

print(f"ChromaDB record count : {count}")

if count == 1805:
    print("COUNT VERIFICATION : PASS")
else:
    print("COUNT VERIFICATION : CHECK")