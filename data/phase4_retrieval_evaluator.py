import json
import sys
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# =========================================================
# CONFIGURATION
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
# E5 EMBEDDING WRAPPER
# =========================================================

class E5Embeddings(HuggingFaceEmbeddings):

    def embed_documents(self, texts):
        texts = [f"passage: {text}" for text in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")


# =========================================================
# LOAD EVALUATION DATASET
# =========================================================

print("=" * 70)
print("PHASE 4 — RETRIEVAL EVALUATION")
print("=" * 70)

print("\nLoading evaluation dataset...")

with open(EVALUATION_FILE, "r", encoding="utf-8") as f:
    evaluation_cases = json.load(f)

print(f"Evaluation cases : {len(evaluation_cases)}")


# =========================================================
# LOAD EMBEDDING MODEL
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

collection_count = vectorstore._collection.count()

print("Collection       :", COLLECTION_NAME)
print("ChromaDB records :", collection_count)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def entity_match(metadata, expected):
    """
    Checks whether the retrieved result belongs
    to the expected temple/entity.
    """

    return (
        metadata.get("temple_id")
        == expected.get("temple_id")
    )


def strict_match(metadata, expected):
    """
    Strict retrieval match.

    For temple:
        source_type + temple_id + section

    For other domains:
        source_type + temple_id
    """

    if (
        metadata.get("source_type")
        != expected.get("source_type")
    ):
        return False

    if (
        metadata.get("temple_id")
        != expected.get("temple_id")
    ):
        return False

    # Temple evaluation has section-level ground truth.
    if expected.get("source_type") == "temple":

        return (
            metadata.get("section")
            == expected.get("section")
        )

    return True


def reciprocal_rank(rank):

    if rank is None:
        return 0.0

    return 1.0 / rank


# =========================================================
# METRIC STORAGE
# =========================================================

total_cases = len(evaluation_cases)

entity_hits = {
    1: 0,
    3: 0,
    5: 0
}

strict_hits = {
    1: 0,
    3: 0,
    5: 0
}

temple_section_hits = {
    1: 0,
    3: 0,
    5: 0
}

entity_rr_sum = 0.0
strict_rr_sum = 0.0
section_rr_sum = 0.0

temple_cases = 0
non_temple_cases = 0

failures = []
all_results = []


# =========================================================
# RUN EVALUATION
# =========================================================

print("\n" + "=" * 70)
print("RUNNING RETRIEVAL EVALUATION")
print("=" * 70)

for index, case in enumerate(
    evaluation_cases,
    start=1
):

    query = case["query"]
    domain = case["domain"]

    # -----------------------------------------------------
    # Ground truth
    # -----------------------------------------------------

    expected = case["expected"]

    expected_source_type = expected["source_type"]
    expected_temple_id = expected["temple_id"]
    expected_section = expected.get("section")

    # -----------------------------------------------------
    # Semantic retrieval
    #
    # IMPORTANT:
    # No metadata filters here.
    #
    # This evaluates pure semantic retrieval.
    # -----------------------------------------------------

    results = vectorstore.similarity_search_with_score(
        query,
        k=TOP_K
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
    # Find first entity-correct result
    # -----------------------------------------------------

    entity_rank = None

    for result in retrieved:

        if (
            result["temple_id"]
            == expected_temple_id
        ):

            entity_rank = result["rank"]
            break

    # -----------------------------------------------------
    # Find first strict-correct result
    # -----------------------------------------------------

    strict_rank = None

    for result in retrieved:

        if (
            result["source_type"]
            == expected_source_type
            and
            result["temple_id"]
            == expected_temple_id
        ):

            # Temple queries require correct section.
            if expected_source_type == "temple":

                if (
                    result["section"]
                    == expected_section
                ):

                    strict_rank = result["rank"]
                    break

            # Other domains only require
            # correct domain + temple.
            else:

                strict_rank = result["rank"]
                break

    # -----------------------------------------------------
    # Case counts
    # -----------------------------------------------------

    if domain == "temple":

        temple_cases += 1

    else:

        non_temple_cases += 1

    # -----------------------------------------------------
    # Entity Recall@K
    # -----------------------------------------------------

    for k in K_VALUES:

        if (
            entity_rank is not None
            and entity_rank <= k
        ):

            entity_hits[k] += 1

    # -----------------------------------------------------
    # Strict Recall@K
    # -----------------------------------------------------

    for k in K_VALUES:

        if (
            strict_rank is not None
            and strict_rank <= k
        ):

            strict_hits[k] += 1

    # -----------------------------------------------------
    # Temple Section Recall
    # -----------------------------------------------------

    if domain == "temple":

        for k in K_VALUES:

            if (
                strict_rank is not None
                and strict_rank <= k
            ):

                temple_section_hits[k] += 1

    # -----------------------------------------------------
    # MRR
    # -----------------------------------------------------

    entity_rr_sum += reciprocal_rank(
        entity_rank
    )

    strict_rr_sum += reciprocal_rank(
        strict_rank
    )

    if domain == "temple":

        section_rr_sum += reciprocal_rank(
            strict_rank
        )

    # -----------------------------------------------------
    # Failure analysis
    # -----------------------------------------------------

    if strict_rank is None:

        failures.append({
            "case_id": case["id"],
            "query": query,
            "domain": domain,
            "expected": expected,
            "retrieved": retrieved
        })

    # -----------------------------------------------------
    # Store result
    # -----------------------------------------------------

    all_results.append({
        "case_id": case["id"],
        "query": query,
        "domain": domain,
        "expected": expected,
        "entity_rank": entity_rank,
        "strict_rank": strict_rank,
        "retrieved": retrieved
    })

    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    if (
        index % 25 == 0
        or index == total_cases
    ):

        print(
            f"Processed {index}/{total_cases} cases..."
        )


# =========================================================
# CALCULATE METRICS
# =========================================================

entity_recall = {
    k: entity_hits[k] / total_cases
    for k in K_VALUES
}

strict_recall = {
    k: strict_hits[k] / total_cases
    for k in K_VALUES
}

temple_section_recall = {
    k: (
        temple_section_hits[k] / temple_cases
        if temple_cases > 0
        else 0.0
    )
    for k in K_VALUES
}

entity_mrr = (
    entity_rr_sum / total_cases
)

strict_mrr = (
    strict_rr_sum / total_cases
)

temple_section_mrr = (
    section_rr_sum / temple_cases
    if temple_cases > 0
    else 0.0
)


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n" + "=" * 70)
print("RETRIEVAL EVALUATION RESULTS")
print("=" * 70)

print(f"\nTotal cases       : {total_cases}")
print(f"Temple cases      : {temple_cases}")
print(f"Other domain cases: {non_temple_cases}")


# ---------------------------------------------------------
# Entity Recall
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("ENTITY RECALL")
print("-" * 70)

for k in K_VALUES:

    print(
        f"Recall@{k} : "
        f"{entity_recall[k] * 100:.2f}% "
        f"({entity_hits[k]}/{total_cases})"
    )

print(
    f"Entity MRR : {entity_mrr:.4f}"
)


# ---------------------------------------------------------
# Strict Recall
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("STRICT DOMAIN + ENTITY / SECTION RECALL")
print("-" * 70)

for k in K_VALUES:

    print(
        f"Recall@{k} : "
        f"{strict_recall[k] * 100:.2f}% "
        f"({strict_hits[k]}/{total_cases})"
    )

print(
    f"Strict MRR : {strict_mrr:.4f}"
)


# ---------------------------------------------------------
# Temple Section Recall
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("TEMPLE SECTION RECALL")
print("-" * 70)

for k in K_VALUES:

    print(
        f"Recall@{k} : "
        f"{temple_section_recall[k] * 100:.2f}% "
        f"({temple_section_hits[k]}/{temple_cases})"
    )

print(
    f"Section MRR : {temple_section_mrr:.4f}"
)


# =========================================================
# FAILURE SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("FAILURE SUMMARY")
print("=" * 70)

print(
    f"\nStrict failures : {len(failures)}"
)

print(
    f"Strict successes: "
    f"{total_cases - len(failures)}"
)


# =========================================================
# FIRST 10 FAILURES
# =========================================================

if failures:

    print("\nFirst 10 failures:\n")

    for failure in failures[:10]:

        print("-" * 70)

        print(
            f"Case ID : {failure['case_id']}"
        )

        print(
            f"Query   : {failure['query']}"
        )

        print("\nExpected:")

        print(
            f"  Source type : "
            f"{failure['expected'].get('source_type')}"
        )

        print(
            f"  Temple ID   : "
            f"{failure['expected'].get('temple_id')}"
        )

        if failure["expected"].get("section"):

            print(
                f"  Section     : "
                f"{failure['expected'].get('section')}"
            )

        print("\nRetrieved:")

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
    BASE_DIR / "phase4_evaluation_results.json"
)

output = {

    "configuration": {
        "model": MODEL_NAME,
        "collection": COLLECTION_NAME,
        "top_k": TOP_K,
        "evaluation_cases": total_cases,
        "retrieval_mode": (
            "unfiltered_semantic_search"
        )
    },

    "metrics": {

        "entity_recall": entity_recall,

        "entity_mrr": entity_mrr,

        "strict_recall": strict_recall,

        "strict_mrr": strict_mrr,

        "temple_section_recall":
            temple_section_recall,

        "temple_section_mrr":
            temple_section_mrr
    },

    "counts": {

        "total_cases": total_cases,

        "temple_cases": temple_cases,

        "non_temple_cases":
            non_temple_cases,

        "strict_failures":
            len(failures)
    },

    "failures": failures,

    "results": all_results
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
print("EVALUATION COMPLETE")
print("=" * 70)

print("\nResults saved to:")
print(OUTPUT_FILE)