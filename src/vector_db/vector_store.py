"""Persistent ChromaDB storage and ingestion utilities."""

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.embeddings.embedder import MODEL_NAME, create_embeddings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_JSONL = PROJECT_ROOT / "data" / "processed" / "phase3_chunks.jsonl"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "pilgrimai"


def load_documents(path: Path = INPUT_JSONL):
    documents = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                documents.append(Document(page_content=obj["page_content"], metadata=obj["metadata"]))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
    return documents


def create_vectorstore():
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=create_embeddings(),
    )


def ingest_documents():
    documents = load_documents()
    vectorstore = create_vectorstore()
    ids = [doc.metadata["chunk_id"] for doc in documents]
    vectorstore.add_documents(documents=documents, ids=ids)
    count = vectorstore._collection.count()
    if count < len(documents):
        raise RuntimeError(f"ChromaDB count is unexpectedly low: {count} < {len(documents)}")
    return count


def main():
    documents = load_documents()
    print("=" * 60)
    print("CHROMADB INGESTION")
    print("=" * 60)
    print(f"Input chunks    : {len(documents)}")
    print(f"Embedding model : {MODEL_NAME}")
    print(f"Collection      : {COLLECTION_NAME}")
    print(f"ChromaDB path   : {CHROMA_DIR}")
    count = ingest_documents()
    print(f"ChromaDB records: {count}")
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
