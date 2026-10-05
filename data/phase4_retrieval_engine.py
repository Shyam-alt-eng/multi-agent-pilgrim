"""Shared retrieval engine for PilgrimAI Phase 4.

This module keeps metadata filtering separate from semantic ranking.
The section filter is optional:

- section=None -> search all sections for the resolved entity/domain.
- section="architecture" -> search only that semantic section.

The LLM/query-understanding layer will eventually populate these values.
This module does not infer sections and does not call an LLM.
"""

from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "pilgrimai"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


class E5Embeddings(HuggingFaceEmbeddings):
    """E5 embedding wrapper using the required query/passage prefixes."""

    def embed_documents(self, texts):
        texts = [f"passage: {text}" for text in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")


def build_metadata_filter(
    *,
    source_type: Optional[str] = None,
    temple_id: Optional[str] = None,
    section: Optional[str] = None,
):
    """Build a Chroma filter from optional structured constraints.

    Chroma 1.5.x requires a logical operator when multiple metadata
    conditions are combined, so all supplied conditions are placed under
    ``$and``.

    ``section`` is intentionally optional. If it is None, no section
    constraint is applied and semantic retrieval can search across all
    sections belonging to the resolved entity/domain.
    """

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
    """Create the persistent Chroma connection using the project E5 setup."""

    embeddings = E5Embeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


def retrieve(
    vectorstore: Chroma,
    query: str,
    *,
    top_k: int = 5,
    source_type: Optional[str] = None,
    temple_id: Optional[str] = None,
    section: Optional[str] = None,
):
    """Retrieve documents using optional metadata constraints plus semantic ranking."""

    metadata_filter = build_metadata_filter(
        source_type=source_type,
        temple_id=temple_id,
        section=section,
    )

    return vectorstore.similarity_search_with_score(
        query,
        k=top_k,
        filter=metadata_filter,
    )
