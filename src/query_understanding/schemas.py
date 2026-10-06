"""Validated structures exchanged between query understanding and retrieval."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QueryDomain(str, Enum):
    TEMPLE = "temple"
    ACCOMMODATION = "accommodation"
    RESTAURANT = "restaurant"
    EMERGENCY = "emergency"


class QueryIntent(str, Enum):
    INFORMATION = "information"
    FIND = "find"
    PLAN = "plan"


class StructuredQuery(BaseModel):
    """LLM output after schema validation, before entity resolution."""

    model_config = ConfigDict(extra="forbid")

    domain: QueryDomain
    intent: QueryIntent
    entity_name: str | None = None
    temple_id: str | None = None
    section: str | None = None
    constraints: dict[str, Any] | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    def normalized_constraints(self) -> dict[str, Any]:
        return self.constraints or {}


class ValidatedQuery(BaseModel):
    """Query safe to pass to metadata-aware retrieval."""

    model_config = ConfigDict(extra="forbid")

    domain: QueryDomain
    intent: QueryIntent
    temple_id: str | None = None
    section: str | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
