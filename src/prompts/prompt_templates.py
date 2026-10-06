"""Prompts for evidence-grounded answer generation."""

import json
from collections.abc import Sequence

from src.evidence.schemas import EvidenceChunk


def grounded_answer_prompt(
    query: str,
    evidence: Sequence[EvidenceChunk],
) -> str:
    """Require claims to cite only the supplied evidence chunk IDs."""

    serialized = [
        {
            "chunk_id": item.chunk_id,
            "content": item.content,
            "metadata": item.metadata,
        }
        for item in evidence
    ]
    return (
        "Answer the user using only the supplied evidence. Return exactly one "
        "JSON object with this shape: "
        '{"answer":"string","claims":[{"claim":"string",'
        '"supported_by":["chunk_id"],"supported":true}]}. '
        "Each claim must contain claim, supported_by, and supported. "
        "supported_by may contain only supplied chunk_id values. Mark a claim "
        "unsupported when the evidence does not establish it. Do not invent "
        "dates, timings, prices, or availability. Do not include markdown or "
        "additional keys. "
        f"User query: {query}\n"
        f"Evidence: {json.dumps(serialized, ensure_ascii=False)}"
    )
