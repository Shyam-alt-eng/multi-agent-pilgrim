import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = PROJECT_ROOT / "data" / "evaluation" / "phase4_evaluation_results.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "evaluation" / "phase4_failure_analysis.json"


# =========================================================
# LOAD RESULTS
# =========================================================

print("=" * 70)
print("PHASE 4 — RETRIEVAL FAILURE ANALYSIS")
print("=" * 70)

print("\nLoading evaluation results...")

with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

failures = data["failures"]

print(f"Total strict failures : {len(failures)}")


# =========================================================
# FAILURE CLASSIFICATION
# =========================================================

classification_counts = Counter()

classified_failures = []


# =========================================================
# CLASSIFY EACH FAILURE
# =========================================================

for failure in failures:

    expected = failure["expected"]
    retrieved = failure["retrieved"]

    expected_source = expected.get("source_type")
    expected_temple = expected.get("temple_id")
    expected_section = expected.get("section")

    # -----------------------------------------------------
    # Find positions of relevant results
    # -----------------------------------------------------

    entity_rank = None
    domain_rank = None
    section_rank = None

    for result in retrieved:

        rank = result["rank"]

        same_entity = (
            result.get("temple_id")
            == expected_temple
        )

        same_domain = (
            result.get("source_type")
            == expected_source
        )

        same_section = (
            result.get("section")
            == expected_section
        )

        # Correct temple/entity
        if same_entity and entity_rank is None:
            entity_rank = rank

        # Correct domain + entity
        if (
            same_domain
            and same_entity
            and domain_rank is None
        ):
            domain_rank = rank

        # Correct domain + entity + section
        if (
            same_domain
            and same_entity
            and same_section
            and section_rank is None
        ):
            section_rank = rank

    # -----------------------------------------------------
    # Classification
    # -----------------------------------------------------

    if section_rank is not None:

        classification = (
            "correct_section_but_strict_failure"
        )

    elif entity_rank is not None:

        if expected_source == "temple":

            classification = (
                "correct_entity_wrong_section"
            )

        else:

            classification = (
                "correct_entity_wrong_domain"
            )

    else:

        # Check whether expected domain appeared
        # without the correct entity.

        domain_found = any(
            result.get("source_type")
            == expected_source
            for result in retrieved
        )

        if domain_found:

            classification = (
                "wrong_entity_same_domain"
            )

        else:

            classification = (
                "wrong_entity_wrong_domain"
            )

    classification_counts[classification] += 1

    # -----------------------------------------------------
    # Store detailed classification
    # -----------------------------------------------------

    classified_failures.append({

        "case_id": failure["case_id"],

        "query": failure["query"],

        "domain": failure["domain"],

        "expected": expected,

        "classification": classification,

        "entity_rank": entity_rank,

        "domain_entity_rank": domain_rank,

        "section_rank": section_rank,

        "retrieved": retrieved
    })


# =========================================================
# PRINT CLASSIFICATION SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("FAILURE CLASSIFICATION")
print("=" * 70)

for classification, count in (
    classification_counts.most_common()
):

    percentage = (
        count / len(failures) * 100
    )

    print(
        f"{classification:<40}"
        f"{count:>5} "
        f"({percentage:.2f}%)"
    )


# =========================================================
# DOMAIN BREAKDOWN
# =========================================================

domain_counts = Counter(
    failure["domain"]
    for failure in classified_failures
)

print("\n" + "=" * 70)
print("FAILURES BY DOMAIN")
print("=" * 70)

for domain, count in domain_counts.most_common():

    print(
        f"{domain:<25}"
        f"{count:>5}"
    )


# =========================================================
# CLASSIFICATION + DOMAIN BREAKDOWN
# =========================================================

combined_counts = Counter(
    (
        failure["domain"],
        failure["classification"]
    )
    for failure in classified_failures
)

print("\n" + "=" * 70)
print("CLASSIFICATION BY DOMAIN")
print("=" * 70)

for (domain, classification), count in (
    combined_counts.most_common()
):

    print(
        f"{domain:<18}"
        f"{classification:<40}"
        f"{count:>5}"
    )


# =========================================================
# EXAMPLES OF EACH FAILURE TYPE
# =========================================================

print("\n" + "=" * 70)
print("EXAMPLES BY FAILURE TYPE")
print("=" * 70)

grouped = defaultdict(list)

for failure in classified_failures:

    grouped[
        failure["classification"]
    ].append(failure)


for classification, items in grouped.items():

    print("\n" + "-" * 70)

    print(classification)

    print("-" * 70)

    for failure in items[:3]:

        print(
            f"\nCase ID : {failure['case_id']}"
        )

        print(
            f"Query   : {failure['query']}"
        )

        print(
            f"Expected: {failure['expected']}"
        )

        print(
            f"Entity rank       : "
            f"{failure['entity_rank']}"
        )

        print(
            f"Domain+entity rank: "
            f"{failure['domain_entity_rank']}"
        )

        print(
            f"Section rank      : "
            f"{failure['section_rank']}"
        )

        print("Top results:")

        for result in failure["retrieved"]:

            print(
                f"  #{result['rank']} "
                f"| {result.get('source_type')} "
                f"| {result.get('temple_id')} "
                f"| {result.get('section')} "
                f"| {result.get('chunk_id')}"
            )


# =========================================================
# SAVE ANALYSIS
# =========================================================

analysis_output = {

    "total_failures": len(failures),

    "classification_counts":
        dict(classification_counts),

    "domain_counts":
        dict(domain_counts),

    "classification_by_domain": {
        f"{domain}::{classification}": count
        for (
            domain,
            classification
        ), count in combined_counts.items()
    },

    "failures": classified_failures
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        analysis_output,
        f,
        indent=2,
        ensure_ascii=False
    )


# =========================================================
# DONE
# =========================================================

print("\n" + "=" * 70)
print("FAILURE ANALYSIS COMPLETE")
print("=" * 70)

print("\nAnalysis saved to:")
print(OUTPUT_FILE)