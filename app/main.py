from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.pdf_extractor import (
    PDFExtractionError,
    extract_pages_from_pdf,
)
from app.services.text_chunker import chunk_pages


app = FastAPI(
    title="Clinical Evidence Assistant",
    description="A citation-grounded assistant for public clinical documents.",
    version="0.3.0",
)


class ChunkResponse(BaseModel):
    chunk_id: str
    page_number: int
    chunk_index: int
    text: str


class DocumentExtractionResponse(BaseModel):
    filename: str
    page_count: int
    character_count: int
    chunk_count: int
    text_preview: str
    chunks: list[ChunkResponse]


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
        pages = extract_pages_from_pdf(pdf_bytes)
    except PDFExtractionError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    chunks = chunk_pages(pages)

    full_text = "\n\n".join(
        page.text for page in pages if page.text
    )

    return DocumentExtractionResponse(
        filename=filename,
        page_count=len(pages),
        character_count=len(full_text),
        chunk_count=len(chunks),
        text_preview=full_text[:500],
        chunks=[
            ChunkResponse(
                chunk_id=chunk.chunk_id,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
            )
            for chunk in chunks
        ],
    )