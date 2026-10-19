"""Provider-neutral structured LLM boundary."""

import json
import os
import random
import time
import urllib.request
from collections.abc import Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

from pydantic import BaseModel

from src.evidence.schemas import EvidenceChunk, GroundedAnswer
from src.prompts.prompt_templates import grounded_answer_prompt

_LAST_REQUEST_AT = 0.0
DEFAULT_REQUEST_DELAY_SECONDS = 4.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0


class LLMRateLimitError(RuntimeError):
    """Raised when a provider still rate-limits after all retry attempts."""

    def __init__(self, message: str, *, retry_after: float | None, attempts: int) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.attempts = attempts


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _request_json(
    request: Any,
    *,
    provider: str,
    request_delay_seconds: float,
    max_retries: int,
    max_backoff_seconds: float,
) -> Any:
    """Make a JSON request with pacing and bounded provider-error retries."""

    global _LAST_REQUEST_AT
    import urllib.error
    import urllib.request

    for attempt in range(1, max_retries + 2):
        elapsed = time.monotonic() - _LAST_REQUEST_AT
        if elapsed < request_delay_seconds:
            time.sleep(request_delay_seconds - elapsed)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                _LAST_REQUEST_AT = time.monotonic()
                print(f"[{provider}] request succeeded (attempt {attempt})")
                return json.load(response)
        except urllib.error.HTTPError as exc:
            _LAST_REQUEST_AT = time.monotonic()
            retry_after = _retry_after_seconds(exc.headers.get("Retry-After"))
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt > max_retries:
                if exc.code == 429:
                    raise LLMRateLimitError(
                        f"{provider} rate limit persisted after {attempt} attempts",
                        retry_after=retry_after,
                        attempts=attempt,
                    ) from exc
                raise RuntimeError(
                    f"{provider} request failed with HTTP {exc.code}: {exc.reason}"
                ) from exc

            backoff = min(
                max_backoff_seconds,
                2 ** (attempt - 1) + random.uniform(0, 0.5),
            )
            delay = max(retry_after or 0.0, backoff)
            print(
                f"[{provider}] HTTP {exc.code}; retry {attempt}/{max_retries} "
                f"in {delay:.1f}s"
            )
            time.sleep(delay)
        except OSError as exc:
            raise RuntimeError(f"{provider} request failed: {exc}") from exc

    raise AssertionError("unreachable")


def _gemini_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert Pydantic JSON Schema to Gemini's supported schema subset."""

    source = model.model_json_schema()
    definitions = source.pop("$defs", {})

    def convert(value: Any) -> Any:
        if isinstance(value, list):
            return [convert(item) for item in value]
        if not isinstance(value, dict):
            return value

        if "$ref" in value:
            definition = definitions[value["$ref"].rsplit("/", 1)[-1]]
            return convert(definition)

        result = {
            key: convert(item)
            for key, item in value.items()
            if key
            not in {
                "$defs",
                "$ref",
                "additionalProperties",
                "default",
                "description",
                "title",
            }
        }
        alternatives = value.get("anyOf")
        if alternatives:
            non_null = [item for item in alternatives if item.get("type") != "null"]
            if len(non_null) == 1 and len(non_null) != len(alternatives):
                result = convert(non_null[0])
                result["nullable"] = True
        return result

    return convert(source)


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

    def __init__(
        self,
        *,
        model: str = "qwen/qwen3.8-27b",
        request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
        max_retries: int = 3,
        max_backoff_seconds: float = 30.0,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.model = model
        self.request_delay_seconds = request_delay_seconds
        self.max_retries = max_retries
        self.max_backoff_seconds = max_backoff_seconds
        self.timeout_seconds = timeout_seconds

    def generate_json(self, prompt: str) -> Mapping[str, Any]:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required for LLM-backed workflows")

        from groq import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            Groq,
            RateLimitError,
        )

        client = Groq(api_key=api_key, timeout=self.timeout_seconds)
        global _LAST_REQUEST_AT
        for attempt in range(1, self.max_retries + 2):
            elapsed = time.monotonic() - _LAST_REQUEST_AT
            if elapsed < self.request_delay_seconds:
                time.sleep(self.request_delay_seconds - elapsed)
            try:
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
                _LAST_REQUEST_AT = time.monotonic()
                print(f"[Groq] request succeeded (attempt {attempt})")
                break
            except (
                APIConnectionError,
                RateLimitError,
                APIStatusError,
                APITimeoutError,
            ) as exc:
                _LAST_REQUEST_AT = time.monotonic()
                status_code = getattr(exc, "status_code", None)
                retryable = isinstance(
                    exc, (APIConnectionError, RateLimitError, APITimeoutError)
                ) or (
                    status_code is not None and 500 <= status_code < 600
                )
                if not retryable or attempt > self.max_retries:
                    if isinstance(exc, RateLimitError):
                        raise LLMRateLimitError(
                            f"Groq rate limit persisted after {attempt} attempts",
                            retry_after=None,
                            attempts=attempt,
                        ) from exc
                    raise RuntimeError(f"Groq request failed: {exc}") from exc
                delay = min(
                    self.max_backoff_seconds,
                    2 ** (attempt - 1) + random.uniform(0, 0.5),
                )
                print(
                    f"[Groq] {type(exc).__name__}; retry "
                    f"{attempt}/{self.max_retries} in {delay:.1f}s"
                )
                time.sleep(delay)
        else:
            raise AssertionError("unreachable")
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


class OpenRouterStructuredLLM:
    """OpenRouter adapter using provider-enforced JSON Schema outputs."""

    def __init__(
        self,
        *,
        response_model: type[BaseModel],
        model: str = "openrouter/free",
        request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
        max_retries: int = 3,
        max_backoff_seconds: float = 30.0,
    ) -> None:
        self.model = model
        self.response_model = response_model
        self.request_delay_seconds = request_delay_seconds
        self.max_retries = max_retries
        self.max_backoff_seconds = max_backoff_seconds

    def generate_json(self, prompt: str) -> Mapping[str, Any]:
        api_key = os.getenv("OPEN_ROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPEN_ROUTER_API_KEY is required for LLM-backed workflows")

        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Return only the requested structured object.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.response_model.__name__,
                        "strict": True,
                        "schema": self.response_model.model_json_schema(),
                    },
                },
                "temperature": 0,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "multi-agent-pilgrim",
            },
            method="POST",
        )
        body = _request_json(
            request,
            provider="OpenRouter",
            request_delay_seconds=self.request_delay_seconds,
            max_retries=self.max_retries,
            max_backoff_seconds=self.max_backoff_seconds,
        )

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenRouter returned an invalid completion response") from exc
        if not content:
            raise RuntimeError("OpenRouter returned an empty structured response")
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("OpenRouter response must be a JSON object")
        return value


class GeminiStructuredLLM:
    """Google Gemini adapter using provider-enforced JSON Schema outputs."""

    def __init__(
        self,
        *,
        response_model: type[BaseModel],
        model: str = "gemini-3-flash-preview",
        request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
        max_retries: int = 3,
        max_backoff_seconds: float = 30.0,
    ) -> None:
        self.model = model
        self.response_model = response_model
        self.request_delay_seconds = request_delay_seconds
        self.max_retries = max_retries
        self.max_backoff_seconds = max_backoff_seconds

    def generate_json(self, prompt: str) -> Mapping[str, Any]:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for LLM-backed workflows")

        payload = json.dumps(
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}],
                    }
                ],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": _gemini_schema(self.response_model),
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            data=payload,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        body = _request_json(
            request,
            provider="Gemini",
            request_delay_seconds=self.request_delay_seconds,
            max_retries=self.max_retries,
            max_backoff_seconds=self.max_backoff_seconds,
        )

        try:
            content = body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Gemini returned an invalid completion response") from exc
        if not content:
            raise RuntimeError("Gemini returned an empty structured response")
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("Gemini response must be a JSON object")
        return value
