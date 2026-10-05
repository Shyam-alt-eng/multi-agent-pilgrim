"""Generic metadata-aware dense retrieval engine."""

from typing import Optional

from langchain_chroma import Chroma

from src.embeddings.embedder import create_embeddings, MODEL_NAME
from src.vector_db.vector_store import CHROMA_DIR, COLLECTION_NAME


def build_metadata_filter(*, source_type: Optional[str] = None, temple_id: Optional[str] = None, section: Optional[str] = None):
    conditions = []
    if source_type is not None:
        conditions.append({"source_type": source_type})
    if temple_id is not None:
        conditions.append({"temple_id": temple_id})
    if section is not None:
        conditions.append({"section": section})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def create_vectorstore():
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=create_embeddings(),
    )


def retrieve(vectorstore: Chroma, query: str, *, top_k: int = 5, source_type: Optional[str] = None, temple_id: Optional[str] = None, section: Optional[str] = None):
    """Retrieve using optional structured constraints plus dense semantic ranking."""
    return vectorstore.similarity_search_with_score(
        query,
        k=top_k,
        filter=build_metadata_filter(source_type=source_type, temple_id=temple_id, section=section),
    )


EMBEDDING_MODEL = MODEL_NAME
