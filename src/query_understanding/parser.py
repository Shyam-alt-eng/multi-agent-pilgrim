"""Parse LLM query output and resolve entities against processed metadata."""

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from src.llm.llm_client import StructuredLLM
from src.query_understanding.schemas import StructuredQuery, ValidatedQuery


def _query_prompt(message: str) -> str:
    return (
        "Extract a pilgrimage request into JSON. Required fields are domain "
        "(temple, accommodation, restaurant, emergency), intent "
        "(information, find, plan), entity_name, temple_id, section, "
        "constraints, and confidence (0 to 1). Use null when unknown. "
        "Return only JSON. User message: "
        + message
    )


def load_entities(path: Path) -> list[dict[str, str]]:
    """Load the metadata index used for deterministic entity checks."""

    entities: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                metadata = record["metadata"]
                entity = {
                    "source_type": str(metadata["source_type"]),
                    "temple_id": str(metadata.get("temple_id", "")),
                    "name": str(metadata.get("name", "")),
                }
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid entity metadata at line {line_number}: {exc}"
                ) from exc
            if entity not in entities:
                entities.append(entity)
    return entities


def _resolve_temple_id(
    query: StructuredQuery,
    entities: Iterable[dict[str, str]],
) -> str | None:
    known = list(entities)
    if query.temple_id:
        if any(
            item["temple_id"] == query.temple_id
            and item["source_type"] == query.domain.value
            for item in known
        ):
            return query.temple_id
        raise ValueError(f"Unknown temple_id: {query.temple_id}")

    if not query.entity_name:
        return None

    needle = query.entity_name.casefold()
    matches = [
        item
        for item in known
        if item["source_type"] == query.domain.value
        and needle in item["name"].casefold()
    ]
    ids = {item["temple_id"] for item in matches if item["temple_id"]}
    if len(ids) > 1:
        raise ValueError(f"Ambiguous entity_name: {query.entity_name}")
    if not ids:
        raise ValueError(f"Unknown entity_name: {query.entity_name}")
    return ids.pop()


def understand_query(
    message: str,
    llm: StructuredLLM,
    *,
    entities: Iterable[dict[str, str]],
) -> ValidatedQuery:
    """Parse, validate, and resolve one LLM-generated retrieval query."""

    if not message.strip():
        raise ValueError("message must not be empty")

    try:
        parsed = StructuredQuery.model_validate(
            llm.generate_json(_query_prompt(message))
        )
    except (ValidationError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid structured-query response: {exc}") from exc

    return ValidatedQuery(
        domain=parsed.domain,
        intent=parsed.intent,
        temple_id=_resolve_temple_id(parsed, entities),
        section=parsed.section,
        constraints=parsed.normalized_constraints(),
        confidence=parsed.confidence,
    )
