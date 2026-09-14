from uuid import uuid4
import os

from app.services.reranker_service import RerankerService

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.services.embedding_service import EmbeddingService
from app.services.pdf_extractor import (
    PDFExtractionError,
    extract_pages_from_pdf,
)
from app.services.text_chunker import chunk_pages
from app.services.vector_store import InMemoryVectorStore

from app.services.llm_service import (
    LLMServiceError,
    OllamaLLMService,
)

from app.services.citation_validator import (
    ABSTENTION_MESSAGE,
    validate_answer_citations,
)


app = FastAPI(
    title="Clinical Evidence Assistant",
    description="A citation-grounded assistant for public clinical documents.",
    version="0.4.0",
)

embedding_service = EmbeddingService()
vector_store = InMemoryVectorStore()

llm_service = OllamaLLMService(
    model="llama3.2:3b",
)

reranker_service = RerankerService()

RERANKER_THRESHOLD = float(
    os.getenv("RERANKER_THRESHOLD", "-3.5")
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


class DocumentIndexResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    total_indexed_chunks: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=2)
    top_k: int = Field(default=3, ge=1, le=10)


class SearchResultResponse(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    text: str
    similarity_score: float
    citation: str


class SearchResponse(BaseModel):
    query: str
    result_count: int
    results: list[SearchResultResponse]

class AnswerRequest(BaseModel):
    query: str = Field(min_length=2)
    top_k: int = Field(default=3, ge=1, le=10)


class CitationResponse(BaseModel):
    source_number: int
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    similarity_score: float
    reranker_score: float
    text: str


class AnswerResponse(BaseModel):
    query: str
    answer: str
    abstained: bool
    abstention_reason: str | None
    citation_validation_passed: bool
    citation_warnings: list[str]
    citations: list[CitationResponse]

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


@app.post(
    "/documents/index",
    response_model=DocumentIndexResponse,
)
async def index_document(
    file: UploadFile = File(...),
) -> DocumentIndexResponse:
    filename = file.filename or "uploaded-document.pdf"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    pdf_bytes = await file.read()
    document_id = uuid4().hex

    try:
        pages = extract_pages_from_pdf(pdf_bytes)
    except PDFExtractionError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    chunks = chunk_pages(
        pages,
        document_id=document_id,
        filename=filename,
    )

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="The PDF did not produce any searchable text chunks.",
        )

    embeddings = await run_in_threadpool(
        embedding_service.embed_texts,
        [chunk.text for chunk in chunks],
    )

    vector_store.add(chunks, embeddings)

    return DocumentIndexResponse(
        document_id=document_id,
        filename=filename,
        page_count=len(pages),
        chunk_count=len(chunks),
        total_indexed_chunks=vector_store.size,
    )


@app.post(
    "/search",
    response_model=SearchResponse,
)
async def search_documents(
    request: SearchRequest,
) -> SearchResponse:
    if vector_store.size == 0:
        raise HTTPException(
            status_code=409,
            detail="No documents have been indexed.",
        )

    query_embedding = await run_in_threadpool(
        embedding_service.embed_query,
        request.query,
    )

    search_results = vector_store.search(
        query_embedding,
        top_k=request.top_k,
    )

    results = [
        SearchResultResponse(
            chunk_id=result.chunk.chunk_id,
            document_id=result.chunk.document_id,
            filename=result.chunk.filename,
            page_number=result.chunk.page_number,
            text=result.chunk.text,
            similarity_score=round(result.score, 4),
            citation=(
                f"{result.chunk.filename}, "
                f"page {result.chunk.page_number}"
            ),
        )
        for result in search_results
    ]

    return SearchResponse(
        query=request.query,
        result_count=len(results),
        results=results,
    )

@app.post(
    "/answer",
    response_model=AnswerResponse,
)
async def answer_question(
    request: AnswerRequest,
) -> AnswerResponse:
    if vector_store.size == 0:
        raise HTTPException(
            status_code=409,
            detail="No documents have been indexed.",
        )

    query_embedding = await run_in_threadpool(
        embedding_service.embed_query,
        request.query,
    )

    candidate_count = max(
        request.top_k * 3,
        10,
    )

    candidate_results = vector_store.search(
        query_embedding,
        top_k=candidate_count,
    )

    reranker_scores = await run_in_threadpool(
        reranker_service.score_passages,
        request.query,
        [
            result.chunk.text
            for result in candidate_results
        ],
    )

    ranked_pairs = sorted(
        zip(
            candidate_results,
            reranker_scores,
            strict=True,
        ),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )

    selected_pairs = ranked_pairs[: request.top_k]

    citations = [
        CitationResponse(
            source_number=source_number,
            chunk_id=result.chunk.chunk_id,
            document_id=result.chunk.document_id,
            filename=result.chunk.filename,
            page_number=result.chunk.page_number,
            similarity_score=round(result.score, 4),
            reranker_score=round(
                float(reranker_score),
                4,
            ),
            text=result.chunk.text,
        )
        for source_number, (
            result,
            reranker_score,
        ) in enumerate(
            selected_pairs,
            start=1,
        )
    ]

    if not selected_pairs:
        answer = ABSTENTION_MESSAGE

        citation_validation = validate_answer_citations(
            answer=answer,
            available_source_count=0,
        )

        return AnswerResponse(
            query=request.query,
            answer=answer,
            abstained=True,
            abstention_reason=(
                "No candidate passages were retrieved."
            ),
            citation_validation_passed=(
                citation_validation.is_valid
            ),
            citation_warnings=(
                citation_validation.warnings
            ),
            citations=[],
        )

    top_reranker_score = float(
        selected_pairs[0][1]
    )

    if top_reranker_score < RERANKER_THRESHOLD:
        answer = ABSTENTION_MESSAGE

        citation_validation = validate_answer_citations(
            answer=answer,
            available_source_count=len(citations),
        )

        return AnswerResponse(
            query=request.query,
            answer=answer,
            abstained=True,
            abstention_reason=(
                f"Top reranker score "
                f"{top_reranker_score:.4f} was below "
                f"the provisional threshold "
                f"{RERANKER_THRESHOLD:.4f}."
            ),
            citation_validation_passed=(
                citation_validation.is_valid
            ),
            citation_warnings=(
                citation_validation.warnings
            ),
            citations=citations,
        )

    context_sections: list[str] = []

    for source_number, (
        result,
        reranker_score,
    ) in enumerate(
        selected_pairs,
        start=1,
    ):
        context_sections.append(
            "\n".join(
                [
                    (
                        f"[{source_number}] "
                        f"Source: {result.chunk.filename}, "
                        f"page {result.chunk.page_number}"
                    ),
                    result.chunk.text,
                ]
            )
        )

    context = "\n\n".join(context_sections)

    try:
        answer = await run_in_threadpool(
            llm_service.generate_answer,
            request.query,
            context,
        )
    except LLMServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    citation_validation = validate_answer_citations(
        answer=answer,
        available_source_count=len(citations),
    )

    return AnswerResponse(
        query=request.query,
        answer=answer,
        abstained=False,
        abstention_reason=None,
        citation_validation_passed=(
            citation_validation.is_valid
        ),
        citation_warnings=(
            citation_validation.warnings
        ),
        citations=citations,
    )