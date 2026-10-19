"""Route conversation turns before invoking retrieval or planning."""

import json
from collections.abc import Sequence

from pydantic import ValidationError

from src.conversation.schemas import ConversationRoute
from src.llm.llm_client import StructuredLLM


def _router_prompt(message: str, history: Sequence[dict[str, str]]) -> str:
    return (
        "Classify this user message for a pilgrimage assistant. "
        "Use one route: general_chat, knowledge_query, trip_planning, "
        "follow_up, or clarification_required. "
        "Return JSON with route, confidence (0 to 1), reason, "
        "requires_retrieval, and payload. "
        f"Conversation history: {json.dumps(list(history), ensure_ascii=False)}\n"
        f"User message: {message}"
    )


def route_message(
    message: str,
    llm: StructuredLLM,
    *,
    history: Sequence[dict[str, str]] = (),
) -> ConversationRoute:
    """Classify a message and reject malformed model output explicitly."""

    if not message.strip():
        raise ValueError("message must not be empty")

    try:
        response = dict(llm.generate_json(_router_prompt(message, history)))
        if isinstance(response.get("payload"), str):
            response["payload"] = {"question": response["payload"]}
        elif response.get("payload") is None:
            response["payload"] = {}
        return ConversationRoute.model_validate(
            response
        )
    except (ValidationError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid conversation-router response: {exc}") from exc
