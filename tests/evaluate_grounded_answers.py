"""Batch evaluation of grounded answers over the Phase 4 benchmark."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.llm_client import GroqStructuredLLM
from src.retrieval.retriever import create_vectorstore, retrieve_with_evidence

DEFAULT_DATASET = PROJECT_ROOT / "data" / "evaluation" / "phase4_evaluation_dataset.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "evaluation" / "grounded_answer_evaluation_results.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=("unfiltered", "entity_filtered", "section_filtered"),
        default="section_filtered",
    )
    return parser.parse_args()


def load_cases(path: Path, limit: int | None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        cases = json.load(source)
    if not isinstance(cases, list):
        raise ValueError("Evaluation dataset must contain a JSON array")
    return cases[:limit] if limit is not None else cases


def retrieval_parameters(
    expected: dict[str, Any],
    mode: str,
) -> dict[str, str | None]:
    if mode == "unfiltered":
        return {}
    parameters: dict[str, str | None] = {
        "source_type": expected.get("source_type"),
        "temple_id": expected.get("temple_id"),
    }
    if mode == "section_filtered":
        parameters["section"] = expected.get("section")
    return parameters


def evaluate_case(
    case: dict[str, Any],
    *,
    vectorstore: Any,
    llm: GroqStructuredLLM,
    top_k: int,
    mode: str,
) -> dict[str, Any]:
    query = str(case["query"])
    expected = case["expected"]
    started = time.perf_counter()
    evidence = retrieve_with_evidence(
        vectorstore,
        query,
        top_k=top_k,
        **retrieval_parameters(expected, mode),
    )
    answer = llm.generate_grounded_answer(query, evidence)
    elapsed_ms = (time.perf_counter() - started) * 1000
    evidence_ids = {item.chunk_id for item in evidence}
    cited_ids = {
        chunk_id
        for claim in answer.claims
        for chunk_id in claim.supported_by
    }
    valid_citation_count = len(cited_ids & evidence_ids)
    citation_count = len(cited_ids)
    return {
        "case_id": case.get("case_id"),
        "query": query,
        "mode": mode,
        "expected": expected,
        "evidence": [item.model_dump() for item in evidence],
        "answer": answer.model_dump(),
        "metrics": {
            "claim_count": len(answer.claims),
            "supported_claim_count": sum(
                claim.supported for claim in answer.claims
            ),
            "citation_count": citation_count,
            "valid_citation_count": valid_citation_count,
            "citation_validity": (
                valid_citation_count / citation_count
                if citation_count
                else 1.0
            ),
            "latency_ms": round(elapsed_ms, 2),
        },
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [item for item in results if "metrics" in item]
    metrics = [item["metrics"] for item in successful]
    claim_count = sum(item["claim_count"] for item in metrics)
    supported_count = sum(item["supported_claim_count"] for item in metrics)
    citation_count = sum(item["citation_count"] for item in metrics)
    valid_citation_count = sum(item["valid_citation_count"] for item in metrics)
    return {
        "case_count": len(results),
        "successful_case_count": len(successful),
        "error_case_count": len(results) - len(successful),
        "claim_count": claim_count,
        "supported_claim_count": supported_count,
        "claim_support_rate": (
            supported_count / claim_count if claim_count else 0.0
        ),
        "citation_count": citation_count,
        "valid_citation_count": valid_citation_count,
        "citation_validity": (
            valid_citation_count / citation_count if citation_count else 1.0
        ),
        "average_latency_ms": (
            sum(item["latency_ms"] for item in metrics) / len(metrics)
            if metrics
            else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    cases = load_cases(args.dataset, args.limit)
    vectorstore = create_vectorstore()
    llm = GroqStructuredLLM()
    results: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        try:
            result = evaluate_case(
                case,
                vectorstore=vectorstore,
                llm=llm,
                top_k=args.top_k,
                mode=args.mode,
            )
        except Exception as exc:
            result = {
                "case_id": case.get("case_id"),
                "query": case.get("query"),
                "mode": args.mode,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        results.append(result)
        if "error" in result:
            print(
                f"[{index}/{len(cases)}] {result['case_id']} "
                f"ERROR={result['error']['type']}: {result['error']['message']}"
            )
        else:
            print(
                f"[{index}/{len(cases)}] {result['case_id']} "
                f"claims={result['metrics']['claim_count']} "
                f"latency_ms={result['metrics']['latency_ms']}"
            )

    output = {
        "configuration": {
            "dataset": str(args.dataset),
            "mode": args.mode,
            "top_k": args.top_k,
            "model": llm.model,
        },
        "summary": summarize(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(output, destination, ensure_ascii=False, indent=2)
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
