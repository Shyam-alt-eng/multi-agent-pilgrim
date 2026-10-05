import sys

sys.stdout.reconfigure(encoding="utf-8")

from src.retrieval.retriever import EMBEDDING_MODEL, create_vectorstore, retrieve
from src.vector_db.vector_store import COLLECTION_NAME


print("=" * 70)
print("LOADING CHROMADB RETRIEVAL ENGINE")
print("=" * 70)

vectorstore = create_vectorstore()
count = vectorstore._collection.count()

print("Embedding model:", EMBEDDING_MODEL)
print("Collection:", COLLECTION_NAME)
print("ChromaDB records:", count)


def run_search(
    title,
    query,
    *,
    k=5,
    source_type=None,
    temple_id=None,
    section=None,
):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print("Query:", query)
    print("Source type:", source_type)
    print("Temple ID:", temple_id)
    print("Section:", section)
    print("-" * 70)

    results = retrieve(
        vectorstore,
        query,
        top_k=k,
        source_type=source_type,
        temple_id=temple_id,
        section=section,
    )

    if not results:
        print("No results found.")
        return

    for i, (doc, score) in enumerate(results, start=1):
        metadata = doc.metadata

        print(f"\nRESULT {i}")
        print("-" * 50)
        print("Distance:", score)
        print("Chunk ID:", metadata.get("chunk_id"))
        print("Entity ID:", metadata.get("entity_id"))
        print("Source Type:", metadata.get("source_type"))
        print("Temple ID:", metadata.get("temple_id"))
        print("Section:", metadata.get("section"))
        print(
            "Name:",
            metadata.get("name")
            or metadata.get("hotel_name")
            or metadata.get("restaurant_name")
            or metadata.get("facility_name"),
        )
        print("\nCONTENT:")
        print(doc.page_content[:1000])


# ------------------------------------------------------------------
# TEST 1 — SECTION-AWARE TEMPLE SEARCH
# ------------------------------------------------------------------
# Explicit section intent: only architecture chunks for T0002 are eligible.
run_search(
    title="TEST 1 — T0002 Architecture (Section Filter)",
    query="architecture of Sri Sita Ramachandra Swamy Temple",
    k=5,
    source_type="temple",
    temple_id="T0002",
    section="architecture",
)


# ------------------------------------------------------------------
# TEST 2 — SECTION-AWARE TEMPLE HISTORY
# ------------------------------------------------------------------
run_search(
    title="TEST 2 — T0002 History (Section Filter)",
    query="history of Sri Sita Ramachandra Swamy Temple",
    k=5,
    source_type="temple",
    temple_id="T0002",
    section="history",
)


# ------------------------------------------------------------------
# TEST 3 — BROAD TEMPLE QUERY WITHOUT SECTION
# ------------------------------------------------------------------
# No section intent: section=None means all T0002 temple sections remain eligible.
run_search(
    title="TEST 3 — T0002 Broad Query (No Section Filter)",
    query="Tell me about Sri Sita Ramachandra Swamy Temple",
    k=5,
    source_type="temple",
    temple_id="T0002",
)


# ------------------------------------------------------------------
# TEST 4 — EXISTING WORSHIP SEARCH
# ------------------------------------------------------------------
run_search(
    title="TEST 4 — T0001 Worship (Section Filter)",
    query="darshan timings",
    k=5,
    source_type="temple",
    temple_id="T0001",
    section="worship",
)


# ------------------------------------------------------------------
# TEST 5 — ACCOMMODATION WITHOUT SECTION
# ------------------------------------------------------------------
# Non-temple entities currently use section="entity" internally, but the
# retrieval contract does not need a section constraint for them.
run_search(
    title="TEST 5 — T0001 Accommodation (No Section Filter)",
    query="accommodation",
    k=5,
    source_type="accommodation",
    temple_id="T0001",
)


# ------------------------------------------------------------------
# TEST 6 — DATABASE COUNT
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 6 — CHROMADB COUNT")
print("=" * 70)
print("ChromaDB record count:", count)
print("COUNT VERIFICATION:", "PASS" if count == 1805 else "CHECK")
