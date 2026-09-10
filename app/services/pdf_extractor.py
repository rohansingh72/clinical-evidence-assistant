from io import BytesIO

from pypdf import PdfReader


class PDFExtractionError(Exception):
    """Raised when text cannot be extracted from a PDF."""


def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, int]:
    """Extract text and page count from PDF bytes."""

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

    page_texts: list[str] = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_texts.append(page_text.strip())

    full_text = "\n\n".join(text for text in page_texts if text)

    if not full_text.strip():
        raise PDFExtractionError(
            "No extractable text was found. The PDF may contain scanned images."
        )

    return full_text, len(reader.pages)