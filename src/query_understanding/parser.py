"""Parse LLM query output and resolve entities against processed metadata."""

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from src.entity_registry.registry import EntityRegistry, EntityRegistryError
from src.llm.llm_client import StructuredLLM
from src.query_understanding.schemas import StructuredQuery, ValidatedQuery


def _query_prompt(message: str) -> str:
    return (
        "Extract a pilgrimage request into exactly one JSON object. Required "
        "fields are domain (temple, accommodation, restaurant, emergency), "
        "intent (information, find, plan), entity_name, temple_id, section, "
        "reference_entity_name, and reference_entity_type. Use entity_name "
        "only for the primary entity in the requested domain. For a service "
        "request near a temple, set entity_name to null and put the temple "
        "in reference_entity_name with reference_entity_type='temple'. "
        "A reference entity is context, not the requested domain. "
        "For domain=temple, section MUST be exactly one of history, worship, "
        "significance, architecture, or travel. Map darshan timings, pooja "
        "timings, and worship information to worship; map religious significance "
        "to significance; map travel or directions to travel. Never return a "
        "descriptive phrase as the section. "
        "constraints, and confidence (0 to 1). Use null when an optional "
        "value is unknown. The constraints field MUST be a JSON object or "
        "null; never return a string or an array for constraints. Use {} "
        "when there are no constraints. Preserve any constraints as a "
        "generic JSON object without inventing fields. Do not invent entity IDs; "
        "return the user-mentioned entity name and let the registry resolve "
        "it. Return only JSON. User message: "
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


def understand_query(
    message: str,
    llm: StructuredLLM,
    *,
    entities: Iterable[dict[str, str]] = (),
    registry: EntityRegistry | None = None,
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

    resolved = None
    resolved_reference = None
    active_registry = registry or EntityRegistry()
    if parsed.temple_id:
        entity = active_registry.get(parsed.temple_id)
        if not entity or entity.get("entity_type") != "temple":
            raise ValueError(f"Unknown temple_id: {parsed.temple_id}")
    if parsed.entity_name:
        try:
            resolved = active_registry.resolve(
                parsed.entity_name,
                entity_type=parsed.domain.value,
            )
        except EntityRegistryError as exc:
            raise ValueError(str(exc)) from exc
    if parsed.reference_entity_name:
        reference_type = parsed.reference_entity_type or "temple"
        try:
            resolved_reference = active_registry.resolve(
                parsed.reference_entity_name,
                entity_type=reference_type,
            )
        except EntityRegistryError as exc:
            raise ValueError(f"Invalid reference entity: {exc}") from exc
    temple_id = parsed.temple_id or (
        resolved.entity_id if resolved and parsed.domain.value == "temple" else None
    )
    if resolved_reference and resolved_reference.entity_type == "temple":
        if temple_id and temple_id != resolved_reference.entity_id:
            raise ValueError("temple_id does not match the reference entity")
        temple_id = resolved_reference.entity_id
    if parsed.domain.value == "temple" and parsed.temple_id and resolved and parsed.temple_id != resolved.entity_id:
        raise ValueError("temple_id does not match the resolved entity_name")
    return ValidatedQuery(
        domain=parsed.domain,
        intent=parsed.intent,
        temple_id=temple_id,
        entity_id=resolved.entity_id if resolved else parsed.temple_id,
        canonical_name=resolved.canonical_name if resolved else None,
        matched_alias=resolved.matched_alias if resolved else None,
        match_type=resolved.match_type if resolved else None,
        reference_entity_id=(
            resolved_reference.entity_id if resolved_reference else None
        ),
        reference_entity_type=(
            resolved_reference.entity_type if resolved_reference else None
        ),
        reference_canonical_name=(
            resolved_reference.canonical_name if resolved_reference else None
        ),
        reference_matched_alias=(
            resolved_reference.matched_alias if resolved_reference else None
        ),
        reference_match_type=(
            resolved_reference.match_type if resolved_reference else None
        ),
        section=parsed.section,
        constraints=parsed.normalized_constraints(),
        confidence=parsed.confidence,
    )
