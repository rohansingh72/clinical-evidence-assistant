from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader


class PDFExtractionError(Exception):
    """Raised when text cannot be extracted from a PDF."""


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str


def extract_pages_from_pdf(pdf_bytes: bytes) -> list[ExtractedPage]:
    """Extract text from a PDF while preserving page numbers."""

    if not pdf_bytes:
        raise PDFExtractionError("The uploaded PDF is empty.")

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception as exc:
        raise PDFExtractionError(
            "The uploaded file could not be read as a PDF."
        ) from exc

    if reader.is_encrypted:
        raise PDFExtractionError(
            "Password-protected PDFs are not currently supported."
        )

    pages: list[ExtractedPage] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()

        pages.append(
            ExtractedPage(
                page_number=page_number,
                text=page_text,
            )
        )

    if not any(page.text for page in pages):
        raise PDFExtractionError(
            "No extractable text was found. The PDF may contain scanned images."
        )

    return pages


def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, int]:
    """Extract complete text while retaining the original interface."""

    pages = extract_pages_from_pdf(pdf_bytes)

    full_text = "\n\n".join(
        page.text for page in pages if page.text
    )

    return full_text, len(pages)