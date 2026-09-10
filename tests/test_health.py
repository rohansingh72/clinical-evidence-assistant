from io import BytesIO

from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app.main import app


client = TestClient(app)


def create_test_pdf() -> bytes:
    buffer = BytesIO()

    pdf = canvas.Canvas(buffer)
    pdf.drawString(
        72,
        750,
        "This clinical study evaluates the safety of the investigational treatment.",
    )
    pdf.save()

    return buffer.getvalue()


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_extract_pdf_text() -> None:
    pdf_bytes = create_test_pdf()

    response = client.post(
        "/documents/extract",
        files={
            "file": (
                "clinical-study.pdf",
                pdf_bytes,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["filename"] == "clinical-study.pdf"
    assert result["page_count"] == 1
    assert result["character_count"] > 0
    assert "clinical study" in result["text_preview"].lower()


def test_reject_non_pdf_file() -> None:
    response = client.post(
        "/documents/extract",
        files={
            "file": (
                "notes.txt",
                b"This is not a PDF.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported."