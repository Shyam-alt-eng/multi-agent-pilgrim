"""Schemas for classifying incoming conversation turns."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RouteType(str, Enum):
    GENERAL_CHAT = "general_chat"
    KNOWLEDGE_QUERY = "knowledge_query"
    TRIP_PLANNING = "trip_planning"
    FOLLOW_UP = "follow_up"
    CLARIFICATION_REQUIRED = "clarification_required"


class ConversationRoute(BaseModel):
    """Validated router output; it contains no database-controlled values."""

    model_config = ConfigDict(extra="forbid")

    route: RouteType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    requires_retrieval: bool
    payload: dict[str, Any] = Field(default_factory=dict)

