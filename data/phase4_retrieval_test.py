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
        text = f"query: {text}"
        return super().embed_query(text)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("=" * 70)
print("LOADING EMBEDDING MODEL")
print("=" * 70)

embeddings = E5Embeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={
        "normalize_embeddings": True
    }
)

print("Embedding model:", EMBEDDING_MODEL)


# ============================================================
# LOAD CHROMADB
# ============================================================

print("\n" + "=" * 70)
print("LOADING CHROMADB")
print("=" * 70)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings
)

count = vectorstore._collection.count()

print("Collection:", COLLECTION_NAME)
print("ChromaDB path:", CHROMA_DIR)
print("Record count:", count)


# ============================================================
# HELPER FUNCTION
# ============================================================

def run_search(title, query, k=5, filter_dict=None):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print("Query:", query)

    if filter_dict:
        print("Filter:", filter_dict)

    print("-" * 70)

    results = vectorstore.similarity_search_with_score(
        query,
        k=k,
        filter=filter_dict
    )

    if not results:
        print("No results found.")
        return

    for i, (doc, score) in enumerate(results, start=1):

        metadata = doc.metadata

        print(f"\nRESULT {i}")
        print("-" * 50)

        print("Distance:", score)

        print("Chunk ID:")
        print(metadata.get("chunk_id"))

        print("Entity ID:")
        print(metadata.get("entity_id"))

        print("Source Type:")
        print(metadata.get("source_type"))

        print("Temple ID:")
        print(metadata.get("temple_id"))

        print("Name:")
        print(
            metadata.get("name")
            or metadata.get("hotel_name")
            or metadata.get("restaurant_name")
            or metadata.get("facility_name")
        )

        print("\nCONTENT:")
        print(doc.page_content[:1000])


# ============================================================
# TEST 1
# PURE SEMANTIC SEARCH
# ============================================================

run_search(
    title="TEST 1 — Semantic Search",
    query="restaurants near basara",
    k=5
)


# ============================================================
# TEST 2
# FILTER BY TEMPLE ID
# ============================================================

run_search(
    title="TEST 2 — T0001 Darshan Search",
    query="darshan timings",
    k=5,
    filter_dict={
        "$and": [
            {
                "source_type": "temple"
            },
            {
                "temple_id": "T0001"
            }
        ]
    }
)


# ============================================================
# TEST 3
# FILTER BY TEMPLE ID + SPECIFIC QUERY
# ============================================================

run_search(
    title="TEST 3 — T0001 Darshan and Tickets",
    query="darshan timings and tickets",
    k=5,
    filter_dict={
        "$and": [
            {
                "source_type": "temple"
            },
            {
                "temple_id": "T0001"
            }
        ]
    }
)


# ============================================================
# TEST 4
# TEMPLE HISTORY
# ============================================================

run_search(
    title="TEST 4 — Temple History",
    query="history of Yadagirigutta temple",
    k=5
)


# ============================================================
# TEST 5
# DEITY SEARCH
# ============================================================

run_search(
    title="TEST 5 — Deity Search",
    query="Lord Lakshmi Narasimha temple",
    k=5
)


# ============================================================
# TEST 6
# EXACT T0001 FILTER
# ============================================================

run_search(
    title="TEST 6 — All T0001 Worship Results",
    query="worship temple",
    k=10,
    filter_dict={
        "$and": [
            {
                "source_type": "temple"
            },
            {
                "temple_id": "T0001"
            }
        ]
    }
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("RETRIEVAL TESTING COMPLETE")
print("=" * 70)

print("ChromaDB records:", count)
print("Embedding model:", EMBEDDING_MODEL)
print("Collection:", COLLECTION_NAME)

print("\nImportant:")
print("- Lower distance generally means a closer vector match.")
print("- Distance is NOT an accuracy percentage.")
print("- TEST 1 checks general semantic retrieval.")
print("- TEST 2/3/6 check metadata filtering.")
print("- TEST 4/5 check different semantic queries.")