from src.retrieval.retriever import build_metadata_filter
from src.evidence.schemas import EvidenceChunk, GroundedAnswer
from src.conversation.router import route_message
from src.conversation.schemas import RouteType
from src.query_understanding.parser import understand_query
from src.retrieval.bm25 import BM25Index, tokenize
from src.retrieval.retriever import reciprocal_rank_fusion
from langchain_core.documents import Document


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def generate_json(self, prompt):
        return self.response


def test_empty_filter():
    assert build_metadata_filter() is None


def test_single_filter():
    assert build_metadata_filter(source_type="temple") == {"source_type": "temple"}


def test_combined_filter():
    assert build_metadata_filter(source_type="temple", temple_id="T0001", section="architecture") == {
        "$and": [
            {"source_type": "temple"},
            {"temple_id": "T0001"},
            {"section": "architecture"},
        ]
    }


def test_general_chat_route_does_not_require_retrieval():
    result = route_message(
        "Hello, how are you?",
        FakeLLM(
            {
                "route": "general_chat",
                "confidence": 0.99,
                "reason": "Casual greeting",
                "requires_retrieval": False,
            }
        ),
    )
    assert result.route is RouteType.GENERAL_CHAT
    assert result.requires_retrieval is False


def test_query_understanding_resolves_known_entity():
    result = understand_query(
        "Tell me about the temple history",
        FakeLLM(
            {
                "domain": "temple",
                "intent": "information",
                "entity_name": "Sri Sita Ramachandra Swamy Temple",
                "section": "history",
                "constraints": {},
                "confidence": 0.9,
            }
        ),
        entities=[
            {
                "source_type": "temple",
                "temple_id": "T0002",
                "name": "Sri Sita Ramachandra Swamy Temple",
            }
        ],
    )
    assert result.temple_id == "T0002"
    assert result.section == "history"


def test_query_understanding_normalizes_null_constraints():
    result = understand_query(
        "Tell me about the temple",
        FakeLLM(
            {
                "domain": "temple",
                "intent": "information",
                "entity_name": "Sri Sita Ramachandra Swamy Temple",
                "constraints": None,
                "confidence": 0.9,
            }
        ),
        entities=[
            {
                "source_type": "temple",
                "temple_id": "T0002",
                "name": "Sri Sita Ramachandra Swamy Temple",
            }
        ],
    )
    assert result.constraints == {}


def test_query_understanding_rejects_unknown_entity():
    try:
        understand_query(
            "Tell me about an unknown temple",
            FakeLLM(
                {
                    "domain": "temple",
                    "intent": "information",
                    "entity_name": "Unknown Temple",
                    "constraints": {},
                    "confidence": 0.8,
                }
            ),
            entities=[],
        )
    except ValueError as exc:
        assert "Unknown entity_name" in str(exc)
    else:
        raise AssertionError("Expected unknown entity to be rejected")


def test_grounded_answer_requires_available_citations():
    evidence = EvidenceChunk(
        chunk_id="T0002__history__part_1",
        content="Temple history evidence.",
        score=0.2,
        metadata={"section": "history"},
    )
    answer = GroundedAnswer(
        answer="The history is supported.",
        evidence=[evidence],
        claims=[
            {
                "claim": "The history is supported.",
                "supported_by": ["T0002__history__part_1"],
                "supported": True,
            }
        ],
    )
    assert answer.claims[0].supported is True


def test_grounded_answer_rejects_unknown_citation():
    try:
        GroundedAnswer(
            answer="Unsupported citation.",
            evidence=[],
            claims=[
                {
                    "claim": "Unsupported citation.",
                    "supported_by": ["missing-chunk"],
                    "supported": True,
                }
            ],
        )
    except ValueError as exc:
        assert "unavailable evidence" in str(exc)
    else:
        raise AssertionError("Expected unknown citation to be rejected")


def test_bm25_tokenization_and_exact_term_ranking():
    assert tokenize("Darshan Timings") == ["darshan", "timings"]
    index = BM25Index(
        [
            Document(page_content="darshan timings at 5 AM", metadata={"chunk_id": "a"}),
            Document(page_content="temple history and legends", metadata={"chunk_id": "b"}),
        ]
    )
    results = index.search("darshan timings", top_k=2)
    assert results[0].document.metadata["chunk_id"] == "a"


def test_reciprocal_rank_fusion_is_deterministic():
    assert reciprocal_rank_fusion(
        [["a", "b"], ["b", "a"]],
        top_k=2,
    ) == [("a", 0.03252247488101534), ("b", 0.03252247488101534)]
