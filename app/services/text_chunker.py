from dataclasses import dataclass

from app.services.pdf_extractor import ExtractedPage


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    page_number: int
    chunk_index: int
    text: str


def chunk_pages(
    pages: list[ExtractedPage],
    max_words: int = 180,
    overlap_words: int = 30,
) -> list[DocumentChunk]:
    """
    Split extracted PDF pages into overlapping chunks.

    Each chunk remains within a single page so that citations
    can reliably reference the source page.
    """

    if max_words <= 0:
        raise ValueError("max_words must be greater than zero.")

    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative.")

    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words.")

    chunks: list[DocumentChunk] = []

    for page in pages:
        words = page.text.split()

        if not words:
            continue

        start = 0
        chunk_index = 1

        while start < len(words):
            end = min(start + max_words, len(words))
            chunk_text = " ".join(words[start:end])

            chunks.append(
                DocumentChunk(
                    chunk_id=f"page-{page.page_number}-chunk-{chunk_index}",
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )

            if end == len(words):
                break

            start = end - overlap_words
            chunk_index += 1

    return chunks