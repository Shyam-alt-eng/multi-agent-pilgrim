"""Validated structures exchanged between query understanding and retrieval."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QueryDomain(str, Enum):
    TEMPLE = "temple"
    ACCOMMODATION = "accommodation"
    RESTAURANT = "restaurant"
    EMERGENCY = "emergency"


class QueryIntent(str, Enum):
    INFORMATION = "information"
    FIND = "find"
    PLAN = "plan"


TEMPLE_SECTIONS = frozenset(
    {"history", "worship", "significance", "architecture", "travel"}
)


class StructuredQuery(BaseModel):
    """LLM output after schema validation, before entity resolution."""

    model_config = ConfigDict(extra="forbid")

    domain: QueryDomain
    intent: QueryIntent
    entity_name: str | None = None
    reference_entity_name: str | None = None
    reference_entity_type: str | None = None
    temple_id: str | None = None
    section: str | None = None
    constraints: dict[str, Any] | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def normalize_and_validate(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        value = dict(value)
        constraints = value.get("constraints")
        if constraints is None:
            value["constraints"] = {}
        elif not isinstance(constraints, dict):
            raise TypeError("constraints must be a JSON object or null")

        if value.get("domain") == QueryDomain.TEMPLE.value:
            section = value.get("section")
            if section is not None and section not in TEMPLE_SECTIONS:
                allowed = ", ".join(sorted(TEMPLE_SECTIONS))
                raise ValueError(
                    f"temple section must be one of: {allowed}"
                )
        return value

    def normalized_constraints(self) -> dict[str, Any]:
        return self.constraints or {}


class ValidatedQuery(BaseModel):
    """Query safe to pass to metadata-aware retrieval."""

    model_config = ConfigDict(extra="forbid")

    domain: QueryDomain
    intent: QueryIntent
    temple_id: str | None = None
    entity_id: str | None = None
    canonical_name: str | None = None
    matched_alias: str | None = None
    match_type: str | None = None
    reference_entity_id: str | None = None
    reference_entity_type: str | None = None
    reference_canonical_name: str | None = None
    reference_matched_alias: str | None = None
    reference_match_type: str | None = None
    section: str | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
