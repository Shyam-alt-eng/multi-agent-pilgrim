"""Resolve user entity mentions against the versioned project registry."""

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

REGISTRY_FILE = Path(__file__).resolve().parents[2] / "data" / "registry" / "entity_registry.json"


class EntityRegistryError(ValueError):
    """Raised when an entity cannot be resolved safely."""


@dataclass(frozen=True)
class ResolvedEntity:
    entity_id: str
    entity_type: str
    canonical_name: str
    matched_alias: str
    match_type: str


class EntityRegistry:
    """Deterministic canonical-name, alias, and cautious fuzzy resolver."""

    fuzzy_threshold = 0.88

    def __init__(self, registry_file: Path = REGISTRY_FILE) -> None:
        self.registry_file = registry_file
        try:
            data = json.loads(registry_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EntityRegistryError(f"Unable to load entity registry: {registry_file}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("entities"), dict):
            raise EntityRegistryError("Entity registry must contain an entities object")
        aliases = data.get("aliases")
        if not isinstance(aliases, dict):
            raise EntityRegistryError("Entity registry must contain an aliases object")
        self.entities: dict[str, dict[str, Any]] = data["entities"]
        self.aliases: dict[str, list[str]] = aliases

    @staticmethod
    def normalize(value: str) -> str:
        if not isinstance(value, str):
            return ""
        replacements = "-_,./\\()'\"：:;"
        normalized = value.strip().casefold()
        for character in replacements:
            normalized = normalized.replace(character, " ")
        return " ".join(normalized.split())

    def _candidate_ids(self, mention: str, entity_type: str | None) -> list[tuple[str, str, str]]:
        normalized = self.normalize(mention)
        candidates: list[tuple[str, str, str]] = []
        for alias, entity_ids in self.aliases.items():
            if not isinstance(entity_ids, list):
                continue
            similarity = SequenceMatcher(None, normalized, self.normalize(str(alias))).ratio()
            if normalized == self.normalize(str(alias)):
                match_type = "exact_alias"
            elif similarity >= self.fuzzy_threshold:
                match_type = "fuzzy_alias"
            else:
                continue
            for entity_id in entity_ids:
                entity = self.entities.get(str(entity_id))
                if not entity or (entity_type and entity.get("entity_type") != entity_type):
                    continue
                candidates.append((str(entity_id), str(alias), match_type))
        return candidates

    def get(self, entity_id: str) -> dict[str, Any] | None:
        return self.entities.get(entity_id)

    def resolve(self, mention: str, *, entity_type: str | None = None) -> ResolvedEntity:
        if not mention or not mention.strip():
            raise EntityRegistryError("Entity mention must not be empty")
        candidates = self._candidate_ids(mention, entity_type)
        exact = [candidate for candidate in candidates if candidate[2] == "exact_alias"]
        candidates = exact or candidates
        entity_ids = {candidate[0] for candidate in candidates}
        if len(entity_ids) > 1:
            raise EntityRegistryError(f"Ambiguous entity_name: {mention}")
        if not candidates:
            raise EntityRegistryError(f"Unknown entity_name: {mention}")
        entity_id, matched_alias, match_type = candidates[0]
        entity = self.entities[entity_id]
        return ResolvedEntity(
            entity_id=entity_id,
            entity_type=str(entity["entity_type"]),
            canonical_name=str(entity["canonical_name"]),
            matched_alias=matched_alias,
            match_type=match_type,
        )
