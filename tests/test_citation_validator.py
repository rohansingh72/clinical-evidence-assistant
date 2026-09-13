from app.services.citation_validator import (
    ABSTENTION_MESSAGE,
    validate_answer_citations,
)


def test_valid_citations_pass() -> None:
    result = validate_answer_citations(
        "The primary endpoint was treatment safety [1].",
        available_source_count=3,
    )

    assert result.is_valid is True
    assert result.cited_source_numbers == [1]
    assert result.warnings == []


def test_missing_citations_fail() -> None:
    result = validate_answer_citations(
        "The primary endpoint was treatment safety.",
        available_source_count=3,
    )

    assert result.is_valid is False
    assert result.cited_source_numbers == []
    assert "no source citations" in result.warnings[0]


def test_unavailable_citation_fails() -> None:
    result = validate_answer_citations(
        "The dose was 10 mg [5].",
        available_source_count=3,
    )

    assert result.is_valid is False
    assert result.cited_source_numbers == [5]
    assert "unavailable sources" in result.warnings[0]


def test_valid_abstention_passes_without_citation() -> None:
    result = validate_answer_citations(
        ABSTENTION_MESSAGE,
        available_source_count=3,
    )

    assert result.is_valid is True
    assert result.cited_source_numbers == []