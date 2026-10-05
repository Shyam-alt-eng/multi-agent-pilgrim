import json
from collections import defaultdict
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "phase3_chunks.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "evaluation" / "phase4_evaluation_dataset.json"

NUM_TEMPLES = 23


# ============================================================
# LOAD CHUNKS
# ============================================================

rows = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        rows.append(json.loads(line))


print(f"Loaded chunks: {len(rows)}")


# ============================================================
# ORGANIZE DATA
# ============================================================

temples = defaultdict(list)
domains_by_temple = defaultdict(set)

for row in rows:

    metadata = row["metadata"]

    source_type = metadata.get("source_type")
    temple_id = metadata.get("temple_id")

    if not temple_id:
        continue

    domains_by_temple[temple_id].add(source_type)

    if source_type == "temple":
        temples[temple_id].append(metadata)


# ============================================================
# FIND TEMPLES THAT HAVE ALL FOUR DOMAINS
# ============================================================

required_domains = {
    "temple",
    "accommodation",
    "restaurant",
    "emergency",
}

eligible_temples = []

for temple_id, domains in domains_by_temple.items():

    if required_domains.issubset(domains):
        eligible_temples.append(temple_id)


eligible_temples = sorted(eligible_temples)

print("\nEligible temples with all four domains:")

for temple_id in eligible_temples:
    print(
        f"  {temple_id}: "
        f"{sorted(domains_by_temple[temple_id])}"
    )


# Use only the first NUM_TEMPLES for the initial evaluation set.
selected_temples = eligible_temples[:NUM_TEMPLES]

print("\nSelected temples:")

for temple_id in selected_temples:

    temple_metadata = temples[temple_id][0]

    print(
        f"  {temple_id} -> "
        f"{temple_metadata.get('name')}"
    )


# ============================================================
# BUILD EVALUATION CASES
# ============================================================

evaluation_cases = []

case_number = 1


# ------------------------------------------------------------
# TEMPLE EVALUATION
# ------------------------------------------------------------

temple_sections = [
    "history",
    "worship",
    "significance",
    "architecture",
    "travel",
]


for temple_id in selected_temples:

    temple_metadata = temples[temple_id][0]

    temple_name = temple_metadata.get("name")

    available_sections = {
        metadata.get("section")
        for metadata in temples[temple_id]
    }

    for section in temple_sections:

        if section not in available_sections:
            continue

        # Create natural-language query
        if section == "history":
            query = f"history of {temple_name}"

        elif section == "worship":
            query = f"darshan timings and worship information at {temple_name}"

        elif section == "significance":
            query = f"religious significance of {temple_name}"

        elif section == "architecture":
            query = f"architecture of {temple_name}"

        elif section == "travel":
            query = f"how to travel to {temple_name}"

        else:
            query = f"{section} information about {temple_name}"

        evaluation_cases.append(
            {
                "id": f"T{case_number:03d}",
                "query": query,
                "domain": "temple",
                "expected": {
                    "source_type": "temple",
                    "temple_id": temple_id,
                    "section": section,
                },
            }
        )

        case_number += 1


# ------------------------------------------------------------
# ACCOMMODATION EVALUATION
# ------------------------------------------------------------

for temple_id in selected_temples:

    temple_metadata = temples[temple_id][0]

    temple_name = temple_metadata.get("name")

    accommodation_queries = [
        f"accommodation near {temple_name}",
        f"hotels to stay near {temple_name}",
        f"places to stay near {temple_name}",
    ]

    for query in accommodation_queries:

        evaluation_cases.append(
            {
                "id": f"A{case_number:03d}",
                "query": query,
                "domain": "accommodation",
                "expected": {
                    "source_type": "accommodation",
                    "temple_id": temple_id,
                },
            }
        )

        case_number += 1


# ------------------------------------------------------------
# RESTAURANT / FOOD EVALUATION
# ------------------------------------------------------------

for temple_id in selected_temples:

    temple_metadata = temples[temple_id][0]

    temple_name = temple_metadata.get("name")

    restaurant_queries = [
        f"restaurants near {temple_name}",
        f"food near {temple_name}",
        f"vegetarian restaurants near {temple_name}",
    ]

    for query in restaurant_queries:

        evaluation_cases.append(
            {
                "id": f"F{case_number:03d}",
                "query": query,
                "domain": "restaurant",
                "expected": {
                    "source_type": "restaurant",
                    "temple_id": temple_id,
                },
            }
        )

        case_number += 1


# ------------------------------------------------------------
# EMERGENCY EVALUATION
# ------------------------------------------------------------

for temple_id in selected_temples:

    temple_metadata = temples[temple_id][0]

    temple_name = temple_metadata.get("name")

    emergency_queries = [
        f"emergency hospital near {temple_name}",
        f"emergency medical facilities near {temple_name}",
        f"hospital near {temple_name}",
    ]

    for query in emergency_queries:

        evaluation_cases.append(
            {
                "id": f"E{case_number:03d}",
                "query": query,
                "domain": "emergency",
                "expected": {
                    "source_type": "emergency",
                    "temple_id": temple_id,
                },
            }
        )

        case_number += 1


# ============================================================
# SAVE DATASET
# ============================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    json.dump(
        evaluation_cases,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("EVALUATION DATASET CREATED")
print("========================================")

print(f"Output file : {OUTPUT_FILE}")
print(f"Total cases : {len(evaluation_cases)}")

domain_counts = defaultdict(int)

for case in evaluation_cases:
    domain_counts[case["domain"]] += 1

print("\nCases by domain:")

for domain, count in sorted(domain_counts.items()):
    print(f"  {domain:<15}: {count}")

print("\nSelected temples:")

for temple_id in selected_temples:

    temple_name = temples[temple_id][0].get("name")

    print(f"  {temple_id} -> {temple_name}")

print("\nDONE")