"""Generic metadata-aware dense retrieval engine."""

import json
from typing import Optional
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.retrieval.bm25 import BM25Index
from src.embeddings.embedder import create_embeddings, MODEL_NAME
from src.evidence.schemas import EvidenceChunk
from src.query_understanding.schemas import ValidatedQuery
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


def retrieve(
    vectorstore: Chroma,
    query: str,
    *,
    top_k: int = 5,
    source_type: Optional[str] = None,
    temple_id: Optional[str] = None,
    section: Optional[str] = None,
    structured_query: ValidatedQuery | None = None,
):
    """Retrieve using optional structured constraints plus dense semantic ranking."""
    if structured_query is not None:
        source_type = structured_query.domain.value
        temple_id = structured_query.temple_id
        section = structured_query.section

    return vectorstore.similarity_search_with_score(
        query,
        k=top_k,
        filter=build_metadata_filter(source_type=source_type, temple_id=temple_id, section=section),
    )


def retrieve_with_evidence(
    vectorstore: Chroma,
    query: str,
    *,
    top_k: int = 5,
    source_type: Optional[str] = None,
    temple_id: Optional[str] = None,
    section: Optional[str] = None,
    structured_query: ValidatedQuery | None = None,
) -> list[EvidenceChunk]:
    """Retrieve chunks while preserving stable IDs for grounded citations."""

    results = retrieve(
        vectorstore,
        query,
        top_k=top_k,
        source_type=source_type,
        temple_id=temple_id,
        section=section,
        structured_query=structured_query,
    )
    evidence: list[EvidenceChunk] = []
    for document, score in results:
        chunk_id = document.metadata.get("chunk_id")
        if not chunk_id:
            raise ValueError("Retrieved document is missing required chunk_id metadata")
        evidence.append(
            EvidenceChunk(
                chunk_id=str(chunk_id),
                content=document.page_content,
                score=float(score),
                metadata=dict(document.metadata),
            )
        )
    return evidence


EMBEDDING_MODEL = MODEL_NAME

_BM25_INDEX: BM25Index | None = None


def create_bm25_index() -> BM25Index:
    """Build the lexical index from the canonical processed chunk artifact."""

    global _BM25_INDEX
    if _BM25_INDEX is None:
        project_root = Path(__file__).resolve().parents[2]
        input_path = project_root / "data" / "processed" / "phase3_chunks.jsonl"
        documents: list[Document] = []
        with input_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    documents.append(
                        Document(
                            page_content=record["page_content"],
                            metadata=record["metadata"],
                        )
                    )
                except (KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Invalid chunk at JSONL line {line_number}: {exc}"
                    ) from exc
        _BM25_INDEX = BM25Index(documents)
    return _BM25_INDEX


def _metadata_predicate(
    *,
    source_type: Optional[str],
    temple_id: Optional[str],
    section: Optional[str],
):
    return lambda document: (
        (source_type is None or document.metadata.get("source_type") == source_type)
        and (temple_id is None or document.metadata.get("temple_id") == temple_id)
        and (section is None or document.metadata.get("section") == section)
    )


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    top_k: int = 5,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked IDs using standard reciprocal rank fusion."""

    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, document_id in enumerate(ranked_list, start=1):
            scores[document_id] = scores.get(document_id, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def retrieve_hybrid(
    vectorstore: Chroma,
    query: str,
    *,
    top_k: int = 5,
    source_type: Optional[str] = None,
    temple_id: Optional[str] = None,
    section: Optional[str] = None,
    dense_candidate_k: int = 50,
    lexical_candidate_k: int = 50,
) -> list[tuple[Document, float]]:
    """Fuse dense and BM25 retrieval while preserving metadata constraints."""

    metadata_filter = build_metadata_filter(
        source_type=source_type,
        temple_id=temple_id,
        section=section,
    )
    dense_results = vectorstore.similarity_search_with_score(
        query,
        k=dense_candidate_k,
        filter=metadata_filter,
    )
    predicate = _metadata_predicate(
        source_type=source_type,
        temple_id=temple_id,
        section=section,
    )
    lexical_results = create_bm25_index().search(
        query,
        top_k=lexical_candidate_k,
        predicate=predicate,
    )

    dense_by_id = {
        str(document.metadata["chunk_id"]): document
        for document, _score in dense_results
    }
    lexical_by_id = {
        str(result.document.metadata["chunk_id"]): result.document
        for result in lexical_results
    }
    fused = reciprocal_rank_fusion(
        [
            list(dense_by_id),
            list(lexical_by_id),
        ],
        top_k=top_k,
    )
    documents = {**lexical_by_id, **dense_by_id}
    return [(documents[document_id], score) for document_id, score in fused]
