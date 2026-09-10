from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.pdf_extractor import (
    PDFExtractionError,
    extract_text_from_pdf,
)


app = FastAPI(
    title="Clinical Evidence Assistant",
    description="A citation-grounded assistant for public clinical documents.",
    version="0.2.0",
)


class DocumentExtractionResponse(BaseModel):
    filename: str
    page_count: int
    character_count: int
    text_preview: str


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Clinical Evidence Assistant",
        "status": "running",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.post(
    "/documents/extract",
    response_model=DocumentExtractionResponse,
)
async def extract_document(
    file: UploadFile = File(...),
) -> DocumentExtractionResponse:
    filename = file.filename or "uploaded-document.pdf"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    pdf_bytes = await file.read()

    try:
        text, page_count = extract_text_from_pdf(pdf_bytes)
    except PDFExtractionError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return DocumentExtractionResponse(
        filename=filename,
        page_count=page_count,
        character_count=len(text),
        text_preview=text[:500],
    )