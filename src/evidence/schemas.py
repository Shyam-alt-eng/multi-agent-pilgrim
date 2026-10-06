"""Structured evidence traces exchanged by retrieval and generation."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceChunk(BaseModel):
    """A retrieved chunk with the metadata needed for citation."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class GroundedClaim(BaseModel):
    """One answer claim and the evidence chunks that support it."""

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1)
    supported_by: list[str] = Field(default_factory=list)
    supported: bool


class GroundedAnswer(BaseModel):
    """LLM answer with an explicit claim-to-evidence trace."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    claims: list[GroundedClaim] = Field(default_factory=list)
    evidence: list[EvidenceChunk] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_citations(self) -> "GroundedAnswer":
        evidence_ids = {item.chunk_id for item in self.evidence}
        for claim in self.claims:
            missing = set(claim.supported_by) - evidence_ids
            if missing:
                raise ValueError(
                    f"Claim cites unavailable evidence chunks: {sorted(missing)}"
                )
            if claim.supported and not claim.supported_by:
                raise ValueError("Supported claims must cite at least one evidence chunk")
        return self

