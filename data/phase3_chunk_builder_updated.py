import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from transformers import AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
INPUT_JSONL = BASE_DIR / "phase2_documents.jsonl"
OUTPUT_JSONL = BASE_DIR / "phase3_chunks.jsonl"

# E5-base supports up to 512 tokens. We keep a safety margin because
# word count != tokenizer token count.
MODEL_NAME = "intfloat/multilingual-e5-base"

# E5-base supports up to 512 input tokens. We keep a safety margin.
TARGET_TOKENS = 380
MAX_TOKENS = 450

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Order matters: more specific prefixes should be checked before broad ones.
TEMPLE_SECTION_RULES: List[Tuple[str, List[str]]] = [
    (
        "identity",
        [
            "temple_id",
            "name",
            "alternate_names",
            "multilingual.",
            "deity.",
            "temple_type",
            "religion",
            "category",
        ],
    ),
    (
        "location",
        [
            "location.",
            "latitude",
            "longitude",
        ],
    ),
    (
        "overview",
        [
            "overview",
            "summary",
        ],
    ),
    (
        "history",
        [
            "history",
            "sthala_puranam",
        ],
    ),
    (
        "significance",
        [
            "religious_significance",
            "spiritual_significance",
            "healing_beliefs",
            "miracles_and_devotee_experiences",
        ],
    ),
    (
        "architecture",
        [
            "architecture",
            "architectural_style",
            "temple_layout",
        ],
    ),
    (
        "worship",
        [
            "darshan_timings",
            "timing_conflict_note",
            "darshan_and_tickets",
            "sevas.",
            "sevas",
            "rituals",
            "special_poojas",
            "festivals.",
            "festivals",
        ],
    ),
    (
        "visitor",
        [
            "facilities",
            "dress_code",
            "visitor_information",
        ],
    ),
    (
        "accommodation",
        [
            "accommodation.",
            "accommodation",
        ],
    ),
    (
        "travel",
        [
            "travel.",
            "travel",
        ],
    ),
    (
        "nearby",
        [
            "nearby_places",
            "nearby_infrastructure.",
            "nearby_infrastructure",
        ],
    ),
    (
        "accessibility_contact",
        [
            "accessibility",
            "contact.",
            "contact",
        ],
    ),
    (
        "reviews",
        [
            "reviews.",
            "reviews",
        ],
    ),
    (
        "faq",
        [
            "faq.",
            "faq_",
            "query_variations.",
            "query_variations",
        ],
    ),
    (
        "quality",
        [
            "quality_check.",
            "quality_check",
            "last_verified",
        ],
    ),
]


def clean_text(value: str) -> str:
    value = value.replace("\r", "")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def get_section_for_line(line: str) -> str:
    """
    page_content is field-labelled, e.g.
    'darshan_timings.morning: 6:00 AM'
    or
    'history: ...'

    We classify using the field name before the first ':'.
    """
    if ":" in line:
        field = line.split(":", 1)[0].strip()
    else:
        field = line.strip()

    for section, prefixes in TEMPLE_SECTION_RULES:
        for prefix in prefixes:
            if field == prefix.rstrip(".") or field.startswith(prefix):
                return section

    return "other"


def parse_temple_sections(page_content: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}

    for raw_line in page_content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        section = get_section_for_line(line)
        sections.setdefault(section, []).append(line)

    return sections


def words(text: str) -> int:
    return len(text.split())


def tokens(text: str) -> int:
    # verbose=False prevents Transformers from printing a warning when
    # we are intentionally measuring a long document before splitting it.
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )
    )


def split_long_line(line: str, target_tokens: int) -> List[str]:
    """
    Last-resort splitter for one field that is itself larger than the
    token budget. We keep the field name in every continuation piece.
    """
    if tokens(line) <= target_tokens:
        return [line]

    if ":" in line:
        field_name, value = line.split(":", 1)
        field_name = field_name.strip()
        value = value.strip()
        prefix = f"{field_name}:"
    else:
        prefix = ""
        value = line.strip()

    pieces: List[str] = []
    current_words: List[str] = []

    for word in value.split():
        candidate_value = " ".join(current_words + [word])
        candidate = f"{prefix} {candidate_value}".strip()

        if current_words and tokens(candidate) > target_tokens:
            pieces.append(f"{prefix} {' '.join(current_words)}".strip())
            current_words = [word]
        else:
            current_words.append(word)

    if current_words:
        pieces.append(f"{prefix} {' '.join(current_words)}".strip())

    return pieces


def split_oversized_lines(lines: List[str], target_tokens: int) -> List[str]:
    """
    Split a semantic section at field/line boundaries first.

    If one field is itself too large, split that field into smaller
    continuation pieces so the final embedding input stays within
    the configured safety limit.
    """
    expanded_lines: List[str] = []

    for line in lines:
        expanded_lines.extend(split_long_line(line, target_tokens))

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for line in expanded_lines:
        line_tokens = tokens(line)

        if current and current_tokens + line_tokens > target_tokens:
            chunks.append("\n".join(current))
            current = []
            current_tokens = 0

        current.append(line)
        current_tokens += line_tokens

    if current:
        chunks.append("\n".join(current))

    return chunks


def build_temple_chunks(doc: Document) -> List[Document]:
    metadata = dict(doc.metadata)
    temple_id = metadata.get("temple_id", "unknown")
    temple_name = metadata.get("name", "Unknown Temple")

    sections = parse_temple_sections(doc.page_content)

    # Preserve a stable semantic order instead of alphabetical order.
    section_order = [
        "identity",
        "location",
        "overview",
        "history",
        "significance",
        "architecture",
        "worship",
        "visitor",
        "accommodation",
        "travel",
        "nearby",
        "accessibility_contact",
        "reviews",
        "faq",
        "quality",
        "other",
    ]

    chunks: List[Document] = []
    chunk_number = 1

    for section_name in section_order:
        lines = sections.get(section_name)
        if not lines:
            continue

        section_text = "\n".join(lines)

        # The semantic header is part of the final embedding input,
        # so reserve its token budget before deciding whether to split.
        header = (
            f"Temple: {temple_name}\n"
            f"Temple ID: {temple_id}\n"
            f"Section: {section_name}\n\n"
        )
        body_budget = min(
            TARGET_TOKENS,
            MAX_TOKENS - tokens(header),
        )

        if tokens(section_text) <= body_budget:
            pieces = [section_text]
        else:
            pieces = split_oversized_lines(lines, body_budget)

        for piece_index, piece in enumerate(pieces, start=1):
            chunk_metadata = dict(metadata)
            # Stable identifiers make later update/delete operations easier.
            # Every chunk belongs to one logical entity and one semantic section.
            chunk_id = f"{temple_id}__{section_name}__part_{piece_index}"

            chunk_metadata.update(
                {
                    "entity_id": temple_id,
                    "section": section_name,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_number,
                    "section_part": piece_index,
                    "section_parts": len(pieces),
                    "chunking_strategy": "semantic_section",
                    "chunk_word_count": words(piece),
                    "chunk_token_count": tokens(piece),
                }
            )

            # Add the compact semantic header to improve retrieval.
            chunks.append(
                Document(
                    page_content=header + clean_text(piece),
                    metadata=chunk_metadata,
                )
            )
            chunk_number += 1

    return chunks


def build_non_temple_documents(doc: Document) -> List[Document]:
    """
    Keep accommodation/restaurant/emergency documents whole when they fit.

    If an entity is larger than the embedding safety limit, split it at
    field/line boundaries. This is especially important for long restaurant
    records in the current dataset.
    """
    metadata = dict(doc.metadata)

    source_type = metadata.get("source_type", "entity")
    entity_id = (
        metadata.get("hotel_id")
        or metadata.get("restaurant_id")
        or metadata.get("facility_id")
        or "unknown"
    )

    if tokens(doc.page_content) <= TARGET_TOKENS:
        pieces = [doc.page_content]
    else:
        pieces = split_oversized_lines(
            doc.page_content.splitlines(),
            TARGET_TOKENS,
        )

    documents: List[Document] = []

    for piece_index, piece in enumerate(pieces, start=1):
        chunk_id = f"{source_type}__{entity_id}__part_{piece_index}"

        chunk_metadata = dict(metadata)
        chunk_metadata.update(
            {
                "entity_id": entity_id,
                "section": "entity",
                "chunk_id": chunk_id,
                "chunk_index": piece_index,
                "section_part": piece_index,
                "section_parts": len(pieces),
                "chunking_strategy": (
                    "entity_whole_document"
                    if len(pieces) == 1
                    else "entity_field_chunking"
                ),
                "chunk_word_count": words(piece),
                "chunk_token_count": tokens(piece),
            }
        )

        documents.append(
            Document(
                page_content=clean_text(piece),
                metadata=chunk_metadata,
            )
        )

    return documents


def load_documents() -> List[Document]:
    documents: List[Document] = []

    with INPUT_JSONL.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                documents.append(
                    Document(
                        page_content=obj["page_content"],
                        metadata=obj["metadata"],
                    )
                )
            except Exception as exc:
                raise ValueError(
                    f"Invalid document at JSONL line {line_number}: {exc}"
                ) from exc

    return documents


def save_documents(documents: List[Document]) -> None:
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(
                json.dumps(
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_JSONL}\n"
            "Place phase2_documents.jsonl in the data/ directory."
        )

    source_documents = load_documents()

    output_documents: List[Document] = []

    source_counts = {}
    output_counts = {}

    for doc in source_documents:
        source_type = doc.metadata.get("source_type", "unknown")
        source_counts[source_type] = source_counts.get(source_type, 0) + 1

        if source_type == "temple":
            chunks = build_temple_chunks(doc)
            output_documents.extend(chunks)
            output_counts[source_type] = output_counts.get(source_type, 0) + len(chunks)
        else:
            chunks = build_non_temple_documents(doc)
            output_documents.extend(chunks)
            output_counts[source_type] = (
                output_counts.get(source_type, 0) + len(chunks)
            )

    save_documents(output_documents)

    print("=" * 60)
    print("PHASE 3 - SEMANTIC CHUNKING COMPLETE")
    print("=" * 60)
    print(f"Input documents : {len(source_documents)}")
    print(f"Output documents: {len(output_documents)}")
    print()
    print("Source document counts:")
    for key, value in sorted(source_counts.items()):
        print(f"  {key:20s}: {value}")

    print()
    print("Output document/chunk counts:")
    for key, value in sorted(output_counts.items()):
        print(f"  {key:20s}: {value}")

    temple_chunks = [
        d for d in output_documents if d.metadata.get("source_type") == "temple"
    ]

    print()
    print("Temple chunk statistics:")
    print(f"  Temple chunks : {len(temple_chunks)}")

    if temple_chunks:
        chunk_word_lengths = [words(d.page_content) for d in temple_chunks]
        chunk_token_lengths = [tokens(d.page_content) for d in temple_chunks]
        print(f"  Min words    : {min(chunk_word_lengths)}")
        print(f"  Max words    : {max(chunk_word_lengths)}")
        print(f"  Avg words    : {sum(chunk_word_lengths) / len(chunk_word_lengths):.2f}")
        print(f"  Min tokens   : {min(chunk_token_lengths)}")
        print(f"  Max tokens   : {max(chunk_token_lengths)}")
        print(f"  Avg tokens   : {sum(chunk_token_lengths) / len(chunk_token_lengths):.2f}")

    oversized = [
        d for d in output_documents if tokens(d.page_content) > MAX_TOKENS
    ]

    hard_limit = [
        d for d in output_documents if tokens(d.page_content) > 512
    ]

    print()
    print(f"Chunks over {MAX_TOKENS} tokens: {len(oversized)}")

    if oversized:
        print("WARNING: review these chunks before embedding:")
        for doc in oversized[:20]:
            print(
                f"  {doc.metadata.get('chunk_id')} "
                f"({tokens(doc.page_content)} tokens)"
            )
    else:
        print("All output documents are within the 450-token safety limit.")

    print()
    print(f"Chunks over the model hard limit (512 tokens): {len(hard_limit)}")

    if hard_limit:
        print("ERROR: these chunks must be fixed before embedding:")
        for doc in hard_limit[:20]:
            print(
                f"  {doc.metadata.get('chunk_id')} "
                f"({tokens(doc.page_content)} tokens)"
            )

    print()
    print(f"Output written to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
