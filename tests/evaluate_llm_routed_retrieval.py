"""Evaluate real LLM routing before dense and hybrid retrieval."""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conversation.router import route_message
from src.conversation.schemas import ConversationRoute
from src.llm.llm_client import (
    GeminiStructuredLLM,
    GroqStructuredLLM,
    LLMRateLimitError,
    OpenRouterStructuredLLM,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
)
from src.query_understanding.parser import load_entities, understand_query
from src.query_understanding.schemas import StructuredQuery
from src.retrieval.retriever import create_vectorstore, retrieve, retrieve_hybrid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
DATASET = PROJECT_ROOT / "data" / "evaluation" / "phase4_evaluation_dataset.json"
ENTITIES = PROJECT_ROOT / "data" / "processed" / "phase3_chunks.jsonl"
DEFAULT_CACHE = PROJECT_ROOT / "data" / "evaluation" / "llm_router_cache.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "llm_routed_retrieval_results.json"
CACHE_VERSION = "llm-routed-retrieval-v3"
DEFAULT_REQUEST_DELAY_SECONDS = 4.0
DEFAULT_MAX_RETRIES = 3


def load_cases() -> list[dict[str, Any]]:
    with DATASET.open("r", encoding="utf-8") as source:
        cases = json.load(source)
    if not isinstance(cases, list):
        raise ValueError("Evaluation dataset must contain a JSON array")
    return cases


def stratified_sample(cases: list[dict[str, Any]], *, limit: int = 60) -> list[dict[str, Any]]:
    """Select a deterministic, proportionally stratified sample."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit >= len(cases):
        return cases

    domains = ["temple", "accommodation", "restaurant", "emergency"]
    domain_cases = {
        domain: [case for case in cases if case.get("domain") == domain]
        for domain in domains
    }
    total = sum(len(items) for items in domain_cases.values())
    raw_targets = {
        domain: limit * len(domain_cases[domain]) / total for domain in domains
    }
    targets = {domain: int(raw_targets[domain]) for domain in domains}
    remaining_slots = limit - sum(targets.values())
    for domain in sorted(
        domains,
        key=lambda item: raw_targets[item] - targets[item],
        reverse=True,
    )[:remaining_slots]:
        targets[domain] += 1

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for domain in domains:
        for case in domain_cases[domain][: targets[domain]]:
            selected.append(case)
            selected_ids.add(str(case["id"]))

    remaining = [case for case in cases if str(case["id"]) not in selected_ids]
    selected.extend(remaining[: max(0, limit - len(selected))])
    selected_ids = {str(case["id"]) for case in selected[:limit]}
    return [case for case in cases if str(case["id"]) in selected_ids][:limit]


def _cache_key(query: str, model: str, *, version: str = CACHE_VERSION) -> str:
    value = f"{version}|{model}|{query}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as source:
        cache = json.load(source)
    if not isinstance(cache, dict):
        raise ValueError("LLM cache must contain a JSON object")
    return cache


def save_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        json.dump(cache, destination, indent=2, ensure_ascii=False)
    temporary.replace(path)


def _expected_rank(results: list[tuple[Any, float]], expected: dict[str, Any]) -> int | None:
    for rank, (document, _score) in enumerate(results, start=1):
        metadata = document.metadata
        if (
            metadata.get("source_type") == expected.get("source_type")
            and metadata.get("temple_id") == expected.get("temple_id")
            and (
                expected.get("source_type") != "temple"
                or metadata.get("section") == expected.get("section")
            )
        ):
            return rank
    return None


def _retrieval_summary(
    cases: list[dict[str, Any]],
    vectorstore: Any,
    validated_queries: dict[str, Any],
) -> dict[str, Any]:
    summaries: dict[str, dict[str, Any]] = {}
    for mode in ("llm_routed_dense", "llm_routed_hybrid"):
        hits = {1: 0, 3: 0, 5: 0}
        precision_sums = {1: 0.0, 3: 0.0, 5: 0.0}
        ndcg_sums = {1: 0.0, 3: 0.0, 5: 0.0}
        reciprocal_rank_sum = 0.0
        failures = 0
        started = time.perf_counter()
        for case in cases:
            record = validated_queries[str(case["id"])]
            if record.get("error") or not record.get("validated_query"):
                failures += 1
                continue
            validated_query = record["validated_query"]
            filters = {
                key: validated_query.get(key)
                for key in ("source_type", "temple_id", "section")
            }
            if mode == "llm_routed_dense":
                results = retrieve(vectorstore, case["query"], top_k=5, **filters)
            else:
                results = retrieve_hybrid(vectorstore, case["query"], top_k=5, **filters)
            rank = _expected_rank(results, case["expected"])
            if rank is None:
                failures += 1
                continue
            reciprocal_rank_sum += 1 / rank
            for cutoff in hits:
                if rank <= cutoff:
                    hits[cutoff] += 1
                    precision_sums[cutoff] += 1 / cutoff
                    ndcg_sums[cutoff] += 1 / math.log2(rank + 1)
        elapsed_ms = (time.perf_counter() - started) * 1000
        count = len(cases)
        summaries[mode] = {
            "cases": count,
            "recall": {str(cutoff): hits[cutoff] / count for cutoff in hits},
            "precision": {
                str(cutoff): precision_sums[cutoff] / count for cutoff in hits
            },
            "mrr": reciprocal_rank_sum / count,
            "ndcg": {
                str(cutoff): ndcg_sums[cutoff] / count for cutoff in hits
            },
            "failures": failures,
            "average_latency_ms": elapsed_ms / count,
        }
    return summaries


METRIC_FORMULAS = {
    "valid_route_rate": "valid routes / total cases",
    "validated_query_rate": "validated queries / total cases",
    "domain_accuracy": "correct predicted domains / total cases",
    "entity_accuracy": "correct resolved entities / total cases",
    "section_accuracy": "correct sections / total cases",
    "recall_at_k": "queries with the expected document in top-k / total cases",
    "precision_at_k": "relevant documents in top-k / k, averaged over cases",
    "mrr": "mean(1 / rank of the first relevant document)",
    "ndcg_at_k": "mean(DCG@k / IDCG@k), with gain 1/log2(rank + 1)",
    "average_latency_ms": "total retrieval time in milliseconds / total cases",
}


def _routing_summary(
    cases: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    domain_correct = 0
    entity_correct = 0
    section_correct = 0
    validated = 0
    route_correct = 0
    for case in cases:
        record = records[str(case["id"])]
        route = record.get("route") or {}
        expected = case["expected"]
        if route.get("route") is not None:
            route_correct += 1
        predicted = record.get("validated_query")
        if not predicted:
            continue
        validated += 1
        domain_correct += predicted.get("source_type") == expected.get("source_type")
        entity_correct += predicted.get("temple_id") == expected.get("temple_id")
        if expected.get("source_type") != "temple" or predicted.get("section") == expected.get("section"):
            section_correct += 1
    count = len(cases)
    return {
        "cases": count,
        "valid_route_rate": route_correct / count,
        "validated_query_rate": validated / count,
        "domain_accuracy": domain_correct / count,
        "entity_accuracy": entity_correct / count,
        "section_accuracy": section_correct / count,
        "valid_prediction_metrics": {
            "cases": validated,
            "domain_accuracy": domain_correct / validated if validated else 0.0,
            "entity_accuracy": entity_correct / validated if validated else 0.0,
            "section_accuracy": section_correct / validated if validated else 0.0,
        },
    }


def evaluate(
    cases: list[dict[str, Any]],
    *,
    model: str = "qwen/qwen3.8-27b",
    provider: str = "groq",
    cache_path: Path = DEFAULT_CACHE,
    output_path: Path = DEFAULT_OUTPUT,
    request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run routing/query understanding once per case and evaluate local retrieval."""

    if provider == "openrouter":
        route_llm = OpenRouterStructuredLLM(
            model=model,
            response_model=ConversationRoute,
            request_delay_seconds=request_delay_seconds,
            max_retries=max_retries,
        )
        query_llm = OpenRouterStructuredLLM(
            model=model,
            response_model=StructuredQuery,
            request_delay_seconds=request_delay_seconds,
            max_retries=max_retries,
        )
    elif provider == "gemini":
        route_llm = GeminiStructuredLLM(
            model=model,
            response_model=ConversationRoute,
            request_delay_seconds=request_delay_seconds,
            max_retries=max_retries,
        )
        query_llm = GeminiStructuredLLM(
            model=model,
            response_model=StructuredQuery,
            request_delay_seconds=request_delay_seconds,
            max_retries=max_retries,
        )
    elif provider == "groq":
        route_llm = query_llm = GroqStructuredLLM(
            model=model,
            request_delay_seconds=request_delay_seconds,
            max_retries=max_retries,
            timeout_seconds=request_timeout_seconds,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    entities = load_entities(ENTITIES)
    cache = load_cache(cache_path)
    query_records: dict[str, dict[str, Any]] = {}
    quota_error: str | None = None
    for case in cases:
        case_id = str(case["id"])
        key = _cache_key(case["query"], model)
        cached = cache.get(key)
        legacy_cached = cache.get(
            _cache_key(case["query"], model, version="llm-routed-retrieval-v1")
        )
        if cached is None:
            cached = legacy_cached
        if cached is not None and cached.get("error") is None:
            print(f"[{case_id}] cache hit")
            query_records[case_id] = cached
            continue

        record: dict[str, Any] = {
            "query": case["query"],
            "model": model,
            "route": None,
            "validated_query": None,
            "error": None,
        }
        if quota_error is not None:
            record["error"] = quota_error
            cache[key] = record
            save_cache(cache_path, cache)
            query_records[case_id] = record
            continue
        try:
            print(f"[{case_id}] routing: {case['query']}")
            route = route_message(case["query"], route_llm)
            record["route"] = route.model_dump(mode="json")
            if not route.requires_retrieval:
                record["error"] = f"Route does not require retrieval: {route.route.value}"
            else:
                print(f"[{case_id}] query understanding")
                validated = understand_query(case["query"], query_llm, entities=entities)
                record["validated_query"] = {
                    "source_type": validated.domain.value,
                    "temple_id": validated.temple_id,
                    "entity_id": validated.entity_id,
                    "canonical_name": validated.canonical_name,
                    "matched_alias": validated.matched_alias,
                    "match_type": validated.match_type,
                    "reference_entity_id": validated.reference_entity_id,
                    "reference_entity_type": validated.reference_entity_type,
                    "reference_canonical_name": validated.reference_canonical_name,
                    "reference_matched_alias": validated.reference_matched_alias,
                    "reference_match_type": validated.reference_match_type,
                    "section": validated.section,
                }
        except LLMRateLimitError as exc:
            quota_error = f"LLMRateLimitError: {exc}"
            record["error"] = quota_error
            record["retry_metadata"] = {
                "attempts": exc.attempts,
                "retry_after_seconds": exc.retry_after,
                "next_allowed_at": (
                    time.time() + exc.retry_after if exc.retry_after is not None else None
                ),
            }
            print(f"[{case_id}] rate limit exhausted: {quota_error}")
        except RateLimitError as exc:
            quota_error = f"RateLimitError: {exc}"
            record["error"] = quota_error
            print(f"[{case_id}] rate limit: {quota_error}")
        except (RuntimeError, ValueError, TypeError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{case_id}] failed: {record['error']}")
        else:
            print(f"[{case_id}] validated")
        cache[key] = record
        save_cache(cache_path, cache)
        query_records[case_id] = record

    vectorstore = create_vectorstore()
    output = {
        "configuration": {
            "cache_version": CACHE_VERSION,
            "cases": len(cases),
            "model": model,
            "provider": provider,
            "request_delay_seconds": request_delay_seconds,
            "request_timeout_seconds": request_timeout_seconds,
            "max_retries": max_retries,
            "sample_ids": [str(case["id"]) for case in cases],
            "sample_composition": {
                domain: sum(case.get("domain") == domain for case in cases)
                for domain in ("temple", "accommodation", "restaurant", "emergency")
            },
            "metric_formulas": METRIC_FORMULAS,
        },
        "routing": {
            **_routing_summary(cases, query_records),
            "errors": sum(record["error"] is not None for record in query_records.values()),
        },
        "queries": query_records,
        "retrieval": _retrieval_summary(cases, vectorstore, query_records),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as destination:
        json.dump(output, destination, indent=2, ensure_ascii=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY_SECONDS,
        help="Minimum seconds between provider requests",
    )
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="Provider request timeout in seconds",
    )
    parser.add_argument("--model", default="qwen/qwen3.8-27b")
    parser.add_argument(
        "--provider", choices=["groq", "openrouter", "gemini"], default="groq"
    )
    args = parser.parse_args()
    cases = stratified_sample(load_cases(), limit=args.limit)
    print(
        json.dumps(
            evaluate(
                cases,
                model=args.model,
                provider=args.provider,
                cache_path=args.cache,
                output_path=args.output,
                request_delay_seconds=args.request_delay,
                max_retries=args.max_retries,
                request_timeout_seconds=args.request_timeout,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
