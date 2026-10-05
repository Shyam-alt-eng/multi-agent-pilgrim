from src.retrieval.retriever import build_metadata_filter


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
