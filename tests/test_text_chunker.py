import pytest

from app.services.pdf_extractor import ExtractedPage
from app.services.text_chunker import chunk_pages


def test_chunk_pages_preserves_page_number() -> None:
    pages = [
        ExtractedPage(
            page_number=4,
            text=" ".join(f"word{i}" for i in range(25)),
        )
    ]

    chunks = chunk_pages(
        pages,
        max_words=10,
        overlap_words=2,
    )

    assert len(chunks) == 3

    assert chunks[0].chunk_id == "page-4-chunk-1"
    assert chunks[0].page_number == 4
    assert chunks[1].page_number == 4
    assert chunks[2].page_number == 4


def test_chunks_overlap() -> None:
    pages = [
        ExtractedPage(
            page_number=1,
            text=" ".join(f"word{i}" for i in range(20)),
        )
    ]

    chunks = chunk_pages(
        pages,
        max_words=10,
        overlap_words=2,
    )

    first_chunk_words = chunks[0].text.split()
    second_chunk_words = chunks[1].text.split()

    assert first_chunk_words[-2:] == second_chunk_words[:2]


def test_chunking_rejects_invalid_overlap() -> None:
    pages = [
        ExtractedPage(
            page_number=1,
            text="Example clinical document.",
        )
    ]

    with pytest.raises(
        ValueError,
        match="overlap_words must be smaller",
    ):
        chunk_pages(
            pages,
            max_words=10,
            overlap_words=10,
        )