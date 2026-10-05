import json
import sys
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# =========================================================
# CONFIG
# =========================================================

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent

EVALUATION_FILE = BASE_DIR / "phase4_evaluation_dataset.json"
CHROMA_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "intfloat/multilingual-e5-base"
COLLECTION_NAME = "pilgrimai"

K_VALUES = [1, 3, 5]
TOP_K = 5


# =========================================================
# E5 EMBEDDINGS
# =========================================================

class E5Embeddings(HuggingFaceEmbeddings):

    def embed_documents(self, texts):
        texts = [f"passage: {text}" for text in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 70)
print("PHASE 4 — FILTERED RETRIEVAL EVALUATION")
print("=" * 70)

with open(EVALUATION_FILE, "r", encoding="utf-8") as f:
    cases = json.load(f)

print(f"\nEvaluation cases : {len(cases)}")


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading embedding model...")

embeddings = E5Embeddings(
    model_name=MODEL_NAME,
    encode_kwargs={
        "normalize_embeddings": True
    }
)

print("Embedding model  :", MODEL_NAME)


# =========================================================
# CONNECT TO CHROMADB
# =========================================================

print("\nConnecting to ChromaDB...")

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)

print(
    "ChromaDB records :",
    vectorstore._collection.count()
)


# =========================================================
# BUILD CHROMA FILTER
# =========================================================

def build_filter(expected):

    source_type = expected["source_type"]
    temple_id = expected["temple_id"]

    return {
        "$and": [
            {"source_type": source_type},
            {"temple_id": temple_id}
        ]
    }


# =========================================================
# STRICT MATCH
# =========================================================

def strict_match(metadata, expected):

    if metadata.get("source_type") != expected["source_type"]:
        return False

    if metadata.get("temple_id") != expected["temple_id"]:
        return False

    # Temple queries additionally require section.
    if expected["source_type"] == "temple":

        return (
            metadata.get("section")
            == expected.get("section")
        )

    return True


# =========================================================
# RUN EVALUATION
# =========================================================

strict_hits = {
    1: 0,
    3: 0,
    5: 0
}

rr_sum = 0.0

failures = []
results_output = []

print("\n" + "=" * 70)
print("RUNNING FILTERED RETRIEVAL")
print("=" * 70)


for index, case in enumerate(cases, start=1):

    query = case["query"]
    expected = case["expected"]

    chroma_filter = build_filter(expected)

    results = vectorstore.similarity_search_with_score(
        query,
        k=TOP_K,
        filter=chroma_filter
    )

    retrieved = []

    for rank, (document, score) in enumerate(
        results,
        start=1
    ):

        metadata = document.metadata

        retrieved.append({
            "rank": rank,
            "score": float(score),
            "chunk_id": metadata.get("chunk_id"),
            "source_type": metadata.get("source_type"),
            "temple_id": metadata.get("temple_id"),
            "section": metadata.get("section"),
            "name": metadata.get("name")
        })

    # -----------------------------------------------------
    # Find first strict match
    # -----------------------------------------------------

    strict_rank = None

    for result in retrieved:

        metadata_like = {
            "source_type": result["source_type"],
            "temple_id": result["temple_id"],
            "section": result["section"]
        }

        if strict_match(
            metadata_like,
            expected
        ):

            strict_rank = result["rank"]
            break

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    for k in K_VALUES:

        if (
            strict_rank is not None
            and strict_rank <= k
        ):

            strict_hits[k] += 1

    if strict_rank is not None:

        rr_sum += 1.0 / strict_rank

    else:

        failures.append({
            "case_id": case["id"],
            "query": query,
            "domain": case["domain"],
            "expected": expected,
            "retrieved": retrieved
        })

    # -----------------------------------------------------
    # Save result
    # -----------------------------------------------------

    results_output.append({
        "case_id": case["id"],
        "query": query,
        "domain": case["domain"],
        "expected": expected,
        "strict_rank": strict_rank,
        "retrieved": retrieved
    })

    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    if index % 25 == 0 or index == len(cases):

        print(
            f"Processed {index}/{len(cases)} cases..."
        )


# =========================================================
# METRICS
# =========================================================

total_cases = len(cases)

recall = {
    k: strict_hits[k] / total_cases
    for k in K_VALUES
}

mrr = rr_sum / total_cases


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 70)
print("FILTERED RETRIEVAL RESULTS")
print("=" * 70)

print(
    f"\nTotal cases : {total_cases}"
)

for k in K_VALUES:

    print(
        f"Recall@{k} : "
        f"{recall[k] * 100:.2f}% "
        f"({strict_hits[k]}/{total_cases})"
    )

print(
    f"MRR        : {mrr:.4f}"
)

print(
    f"\nFailures   : {len(failures)}"
)


# =========================================================
# DOMAIN BREAKDOWN
# =========================================================

from collections import defaultdict

domain_total = defaultdict(int)
domain_success = {
    k: defaultdict(int)
    for k in K_VALUES
}

domain_mrr_sum = defaultdict(float)

for result in results_output:

    domain = result["domain"]

    domain_total[domain] += 1

    rank = result["strict_rank"]

    if rank is not None:

        domain_mrr_sum[domain] += 1.0 / rank

        for k in K_VALUES:

            if rank <= k:

                domain_success[k][domain] += 1


print("\n" + "-" * 70)
print("RESULTS BY DOMAIN")
print("-" * 70)

for domain in sorted(domain_total):

    total = domain_total[domain]

    print(f"\n{domain}")

    for k in K_VALUES:

        success = domain_success[k][domain]

        print(
            f"  Recall@{k}: "
            f"{success / total * 100:.2f}% "
            f"({success}/{total})"
        )

    print(
        f"  MRR: "
        f"{domain_mrr_sum[domain] / total:.4f}"
    )


# =========================================================
# FIRST 10 FAILURES
# =========================================================

if failures:

    print("\n" + "=" * 70)
    print("FIRST 10 FILTERED FAILURES")
    print("=" * 70)

    for failure in failures[:10]:

        print("\n" + "-" * 70)

        print(
            f"Case ID : {failure['case_id']}"
        )

        print(
            f"Query   : {failure['query']}"
        )

        print(
            f"Expected: {failure['expected']}"
        )

        print("Retrieved:")

        for result in failure["retrieved"]:

            print(
                f"  #{result['rank']} "
                f"| score={result['score']:.6f} "
                f"| {result['source_type']} "
                f"| {result['temple_id']} "
                f"| {result['section']} "
                f"| {result['chunk_id']}"
            )


# =========================================================
# SAVE RESULTS
# =========================================================

OUTPUT_FILE = (
    BASE_DIR /
    "phase4_filtered_evaluation_results.json"
)

output = {

    "configuration": {
        "model": MODEL_NAME,
        "collection": COLLECTION_NAME,
        "top_k": TOP_K,
        "retrieval_mode":
            "source_type_and_temple_id_filtered"
    },

    "metrics": {
        "recall": recall,
        "mrr": mrr
    },

    "domain_metrics": {
        domain: {
            "total": domain_total[domain],

            "recall": {
                str(k):
                    domain_success[k][domain]
                    / domain_total[domain]
                for k in K_VALUES
            },

            "mrr":
                domain_mrr_sum[domain]
                / domain_total[domain]
        }

        for domain in domain_total
    },

    "failures": failures,

    "results": results_output
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False
    )


# =========================================================
# DONE
# =========================================================

print("\n" + "=" * 70)
print("FILTERED EVALUATION COMPLETE")
print("=" * 70)

print("\nResults saved to:")
print(OUTPUT_FILE)