"""Evaluate dense, BM25, and hybrid retrieval on the 322-case benchmark."""

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.retriever import (  # noqa: E402
    create_bm25_index,
    create_vectorstore,
    retrieve,
    retrieve_hybrid,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "evaluation" / "phase4_evaluation_dataset.json"
OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "hybrid_retrieval_results.json"


def load_cases() -> list[dict[str, Any]]:
    with DATASET.open("r", encoding="utf-8") as source:
        cases = json.load(source)
    if not isinstance(cases, list):
        raise ValueError("Evaluation dataset must contain a JSON array")
    return cases


def parameters(expected: dict[str, Any], mode: str) -> dict[str, str | None]:
    if mode == "dense" or mode == "bm25" or mode == "hybrid":
        return {}
    if mode == "domain_routed_hybrid":
        return {
            "source_type": expected.get("source_type"),
            "temple_id": expected.get("temple_id"),
        }
    raise ValueError(f"Unknown retrieval mode: {mode}")


def retrieve_mode(vectorstore: Any, query: str, expected: dict[str, Any], mode: str):
    filters = parameters(expected, mode)
    if mode == "dense":
        return retrieve(vectorstore, query, top_k=5, **filters)
    if mode == "bm25":
        results = create_bm25_index().search(
            query,
            top_k=5,
            predicate=lambda document: True,
        )
        return [(result.document, result.score) for result in results]
    return retrieve_hybrid(vectorstore, query, top_k=5, **filters)


def rank_for_expected(results: list[tuple[Any, float]], expected: dict[str, Any]):
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


def evaluate_mode(
    cases: list[dict[str, Any]],
    vectorstore: Any,
    mode: str,
) -> dict[str, Any]:
    hits = {1: 0, 3: 0, 5: 0}
    reciprocal_rank_sum = 0.0
    total_latency = 0.0
    failures = 0
    for case in cases:
        started = time.perf_counter()
        results = retrieve_mode(vectorstore, case["query"], case["expected"], mode)
        total_latency += (time.perf_counter() - started) * 1000
        rank = rank_for_expected(results, case["expected"])
        if rank is None:
            failures += 1
        else:
            reciprocal_rank_sum += 1 / rank
            for cutoff in hits:
                hits[cutoff] += rank <= cutoff
    count = len(cases)
    return {
        "mode": mode,
        "cases": count,
        "recall": {str(cutoff): hits[cutoff] / count for cutoff in hits},
        "mrr": reciprocal_rank_sum / count,
        "failures": failures,
        "average_latency_ms": total_latency / count,
    }


def main() -> None:
    cases = load_cases()
    vectorstore = create_vectorstore()
    summaries = [
        evaluate_mode(cases, vectorstore, mode)
        for mode in ("dense", "bm25", "hybrid", "domain_routed_hybrid")
    ]
    output = {
        "configuration": {
            "dataset_cases": len(cases),
            "top_k": 5,
            "model": "intfloat/multilingual-e5-base",
            "fusion": "reciprocal_rank_fusion",
        },
        "summaries": summaries,
    }
    with OUTPUT.open("w", encoding="utf-8") as destination:
        json.dump(output, destination, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
