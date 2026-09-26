import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_JSONL = BASE_DIR / "phase3_chunks.jsonl"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "pilgrimai"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


# ------------------------------------------------------------
# Load Phase 3 chunks
# ------------------------------------------------------------

def load_documents():
    documents = []

    with INPUT_JSONL.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)

                documents.append(
                    Document(
                        page_content=obj["page_content"],
                        metadata=obj["metadata"],
                    )
                )

            except Exception as exc:
                raise ValueError(
                    f"Invalid JSONL at line {line_number}: {exc}"
                ) from exc

    return documents


# ------------------------------------------------------------
# E5 embedding function
# ------------------------------------------------------------

class E5Embeddings(HuggingFaceEmbeddings):
    """
    E5 requires:
      documents -> passage: <text>
      queries   -> query: <text>
    """

    def embed_documents(self, texts):
        return super().embed_documents(
            [f"passage: {text}" for text in texts]
        )

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    if not INPUT_JSONL.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_JSONL}"
        )

    documents = load_documents()

    print("=" * 60)
    print("PHASE 4 - CHROMADB INGESTION")
    print("=" * 60)
    print(f"Input chunks     : {len(documents)}")
    print(f"Embedding model  : {EMBEDDING_MODEL}")
    print(f"Collection       : {COLLECTION_NAME}")
    print(f"ChromaDB path    : {CHROMA_DIR}")
    print()

    # Create embedding function.
    embeddings = E5Embeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    # Create/open persistent Chroma collection.
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # Use our deterministic chunk IDs.
    ids = [
        doc.metadata["chunk_id"]
        for doc in documents
    ]

    # Add documents and their metadata.
    vectorstore.add_documents(
        documents=documents,
        ids=ids,
    )

    # Verify count.
    collection = vectorstore._collection
    count = collection.count()

    print("INGESTION COMPLETE")
    print("-" * 60)
    print(f"ChromaDB record count: {count}")

    if count != len(documents):
        raise RuntimeError(
            f"Count mismatch: expected {len(documents)}, got {count}"
        )

    print("Count verification : PASS")
    print()
    print(f"Database location: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
