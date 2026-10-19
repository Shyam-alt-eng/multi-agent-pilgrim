from src.retrieval.retriever import build_metadata_filter
from src.evidence.schemas import EvidenceChunk, GroundedAnswer
from src.conversation.router import route_message
from src.conversation.schemas import RouteType
from src.query_understanding.parser import understand_query
from src.retrieval.bm25 import BM25Index, tokenize
from src.retrieval.retriever import reciprocal_rank_fusion
from langchain_core.documents import Document
from tests.evaluate_llm_routed_retrieval import load_cache, save_cache, stratified_sample
from src.entity_registry.registry import EntityRegistry, EntityRegistryError
from src.llm.llm_client import _retry_after_seconds


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def generate_json(self, prompt):
        return self.response


def test_router_normalizes_string_payload():
    result = route_message(
        "How do I get there?",
        FakeLLM(
            {
                "route": "trip_planning",
                "confidence": 0.9,
                "reason": "Travel request",
                "requires_retrieval": True,
                "payload": "How do I get there?",
            }
        ),
    )
    assert result.payload == {"question": "How do I get there?"}


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


def test_query_understanding_rejects_non_object_constraints():
    try:
        understand_query(
            "Find vegetarian restaurants",
            FakeLLM(
                {
                    "domain": "restaurant",
                    "intent": "find",
                    "entity_name": None,
                    "constraints": ["vegetarian"],
                    "confidence": 0.8,
                }
            ),
        )
    except ValueError as exc:
        assert "constraints must be a JSON object or null" in str(exc)
    else:
        raise AssertionError("Expected non-object constraints to be rejected")


def test_query_understanding_accepts_generic_constraint_object():
    result = understand_query(
        "Find a place near the temple",
        FakeLLM(
            {
                "domain": "accommodation",
                "intent": "find",
                "entity_name": None,
                "constraints": {"near": "temple"},
                "confidence": 0.8,
            }
        ),
    )
    assert result.constraints == {"near": "temple"}


def test_query_understanding_rejects_noncanonical_temple_section():
    try:
        understand_query(
            "darshan timings at the temple",
            FakeLLM(
                {
                    "domain": "temple",
                    "intent": "information",
                    "entity_name": "Sri Lakshmi Narasimha Swamy Temple",
                    "section": "darshan timings and worship information",
                    "constraints": {},
                    "confidence": 0.8,
                }
            ),
            entities=[],
        )
    except ValueError as exc:
        assert "temple section must be one of" in str(exc)
    else:
        raise AssertionError("Expected noncanonical temple section to be rejected")


def test_query_understanding_resolves_cross_domain_reference_entity():
    result = understand_query(
        "Find accommodation near Sri Lakshmi Narasimha Swamy Temple",
        FakeLLM(
            {
                "domain": "accommodation",
                "intent": "find",
                "entity_name": None,
                "reference_entity_name": "Sri Lakshmi Narasimha Swamy Temple",
                "reference_entity_type": "temple",
                "constraints": {"near": True},
                "confidence": 0.9,
            }
        ),
    )
    assert result.domain.value == "accommodation"
    assert result.entity_id is None
    assert result.temple_id == "T0001"
    assert result.reference_entity_id == "T0001"
    assert result.reference_entity_type == "temple"


def test_openrouter_structured_client_builds_schema_format():
    from src.conversation.schemas import ConversationRoute
    from src.llm.llm_client import OpenRouterStructuredLLM

    client = OpenRouterStructuredLLM(response_model=ConversationRoute)
    schema = client.response_model.model_json_schema()
    assert schema["type"] == "object"
    assert "route" in schema["properties"]


def test_gemini_structured_client_builds_schema_format():
    from src.conversation.schemas import ConversationRoute
    from src.llm.llm_client import GeminiStructuredLLM

    client = GeminiStructuredLLM(response_model=ConversationRoute)
    schema = client.response_model.model_json_schema()
    assert schema["type"] == "object"
    assert "route" in schema["properties"]


def test_retry_after_seconds_accepts_seconds():
    assert _retry_after_seconds("2.5") == 2.5


def test_retry_after_seconds_rejects_invalid_value():
    assert _retry_after_seconds("not-a-delay") is None


def test_groq_structured_client_uses_explicit_timeout():
    from src.llm.llm_client import (
        DEFAULT_REQUEST_TIMEOUT_SECONDS,
        GroqStructuredLLM,
    )

    client = GroqStructuredLLM()
    assert client.timeout_seconds == DEFAULT_REQUEST_TIMEOUT_SECONDS


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


def test_query_understanding_prefers_exact_entity_name_match():
    result = understand_query(
        "Tell me about Sri Lakshmi Narasimha Swamy Temple",
        FakeLLM(
            {
                "domain": "temple",
                "intent": "information",
                "entity_name": "Sri Lakshmi Narasimha Swamy Temple",
                "constraints": {},
                "confidence": 0.9,
            }
        ),
        entities=[
            {
                "source_type": "temple",
                "temple_id": "T0017",
                "name": "Sri Lakshmi Narasimha Swamy Temple, Dharmapuri",
            },
            {
                "source_type": "temple",
                "temple_id": "T0001",
                "name": "Sri Lakshmi Narasimha Swamy Temple",
            },
        ],
    )
    assert result.temple_id == "T0001"


def test_registry_resolves_alias():
    result = EntityRegistry().resolve("Yadadri Temple", entity_type="temple")
    assert result.entity_id == "T0001"
    assert result.match_type == "exact_alias"


def test_registry_rejects_ambiguous_alias():
    try:
        EntityRegistry().resolve("Dakshina Kashi", entity_type="temple")
    except EntityRegistryError as exc:
        assert "Ambiguous entity_name" in str(exc)
    else:
        raise AssertionError("Expected ambiguous alias to be rejected")


def test_registry_keeps_similar_temple_names_distinct():
    registry = EntityRegistry()
    assert registry.resolve(
        "Sri Lakshmi Narasimha Swamy Temple", entity_type="temple"
    ).entity_id == "T0001"
    assert registry.resolve(
        "Sri Lakshmi Narasimha Swamy Temple, Dharmapuri",
        entity_type="temple",
    ).entity_id == "T0017"


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


def test_llm_router_sample_is_deterministic_and_stratified():
    cases = [
        {"id": "t1", "domain": "temple"},
        {"id": "a1", "domain": "accommodation"},
        {"id": "r1", "domain": "restaurant"},
        {"id": "e1", "domain": "emergency"},
    ]
    assert [case["id"] for case in stratified_sample(cases, limit=4)] == [
        "t1",
        "a1",
        "r1",
        "e1",
    ]


def test_llm_router_cache_round_trip(tmp_path):
    path = tmp_path / "cache.json"
    expected = {"cache-key": {"validated_query": None, "error": "test"}}
    save_cache(path, expected)
    assert load_cache(path) == expected
