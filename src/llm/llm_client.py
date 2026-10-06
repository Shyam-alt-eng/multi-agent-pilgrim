"""Provider-neutral structured LLM boundary."""

import json
import os
from collections.abc import Mapping
from typing import Any, Protocol

from src.evidence.schemas import EvidenceChunk, GroundedAnswer
from src.prompts.prompt_templates import grounded_answer_prompt


class StructuredLLM(Protocol):
    """Minimal interface required by routing and query understanding."""

    def generate_json(self, prompt: str) -> Mapping[str, Any]:
        """Return one JSON object generated from ``prompt``."""


class GroundedLLM(Protocol):
    """Interface for generating answers with an evidence trace."""

    def generate_grounded_answer(
        self,
        query: str,
        evidence: list[EvidenceChunk],
    ) -> GroundedAnswer:
        """Generate a validated answer from retrieved evidence."""


class GroqStructuredLLM:
    """Groq adapter using JSON mode; credentials are read at call time."""

    def __init__(self, *, model: str = "openai/gpt-oss-20b") -> None:
        self.model = model

    def generate_json(self, prompt: str) -> Mapping[str, Any]:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required for LLM-backed workflows")

        from groq import Groq

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Return only one valid JSON object. Do not include markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned an empty response")

        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("LLM response must be a JSON object")
        return value

    def generate_grounded_answer(
        self,
        query: str,
        evidence: list[EvidenceChunk],
    ) -> GroundedAnswer:
        """Generate and validate an evidence-grounded answer."""

        value = self.generate_json(grounded_answer_prompt(query, evidence))
        return GroundedAnswer.model_validate(
            {
                **value,
                "evidence": [item.model_dump() for item in evidence],
            }
        )
