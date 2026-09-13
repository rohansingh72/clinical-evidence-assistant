from pathlib import Path

from reportlab.pdfgen import canvas


EVALUATION_DIRECTORY = Path(__file__).resolve().parent
PDF_PATH = EVALUATION_DIRECTORY / "sample_clinical_study.pdf"


pages = [
    (
        "Study objective and design",
        [
            "The primary objective was to evaluate the efficacy of Drug A.",
            "This was a randomized, double-blind, placebo-controlled study.",
            "The treatment duration was twelve weeks.",
        ],
    ),
    (
        "Participants and treatment",
        [
            "The study enrolled 120 adults between 18 and 65 years of age.",
            "Participants received either Drug A 10 mg or placebo once daily.",
            "Treatment was administered orally for twelve weeks.",
        ],
    ),
    (
        "Safety assessment",
        [
            "The primary safety endpoint was the incidence of treatment-emergent adverse events.",
            "Adverse events were assessed at every study visit.",
            "Serious adverse events and treatment discontinuations were also recorded.",
        ],
    ),
]


def create_pdf() -> None:
    pdf = canvas.Canvas(str(PDF_PATH))

    for title, paragraphs in pages:
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(72, 750, title)

        pdf.setFont("Helvetica", 11)
        vertical_position = 710

        for paragraph in paragraphs:
            pdf.drawString(
                72,
                vertical_position,
                paragraph,
            )
            vertical_position -= 30

        pdf.showPage()

    pdf.save()

    print(f"Created: {PDF_PATH}")


if __name__ == "__main__":
    create_pdf()