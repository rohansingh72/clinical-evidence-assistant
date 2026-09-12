from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app import main


client = TestClient(main.app)


def create_test_pdf() -> bytes:
    buffer = BytesIO()

    pdf = canvas.Canvas(buffer)
    pdf.drawString(
        72,
        750,
        "The primary endpoint was treatment safety and adverse events.",
    )
    pdf.save()

    return buffer.getvalue()


def test_index_and_search_document(monkeypatch) -> None:
    main.vector_store.clear()

    def fake_embed_texts(texts: list[str]) -> np.ndarray:
        return np.array(
            [[1.0, 0.0, 0.0] for _ in texts],
            dtype=np.float32,
        )

    def fake_embed_query(query: str) -> np.ndarray:
        return np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        )

    monkeypatch.setattr(
        main.embedding_service,
        "embed_texts",
        fake_embed_texts,
    )
    monkeypatch.setattr(
        main.embedding_service,
        "embed_query",
        fake_embed_query,
    )

    index_response = client.post(
        "/documents/index",
        files={
            "file": (
                "clinical-study.pdf",
                create_test_pdf(),
                "application/pdf",
            )
        },
    )

    assert index_response.status_code == 200
    assert index_response.json()["chunk_count"] == 1

    search_response = client.post(
        "/search",
        json={
            "query": "What was the primary safety endpoint?",
            "top_k": 1,
        },
    )

    assert search_response.status_code == 200

    result = search_response.json()["results"][0]

    assert result["filename"] == "clinical-study.pdf"
    assert result["page_number"] == 1
    assert "safety" in result["text"].lower()
    assert result["citation"] == "clinical-study.pdf, page 1"