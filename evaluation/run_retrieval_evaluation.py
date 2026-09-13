import json
import os
from pathlib import Path

import httpx


EVALUATION_DIRECTORY = Path(__file__).resolve().parent
PDF_PATH = EVALUATION_DIRECTORY / "sample_clinical_study.pdf"
QUESTIONS_PATH = EVALUATION_DIRECTORY / "questions.json"

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)

TOP_K = 3


def index_evaluation_document() -> None:
    with PDF_PATH.open("rb") as pdf_file:
        response = httpx.post(
            f"{API_BASE_URL}/documents/index",
            files={
                "file": (
                    PDF_PATH.name,
                    pdf_file,
                    "application/pdf",
                )
            },
            timeout=300,
        )

    response.raise_for_status()

    result = response.json()

    print(
        f"Indexed {result['filename']} "
        f"with {result['chunk_count']} chunks."
    )


def find_expected_rank(
    results: list[dict],
    expected_filename: str,
    expected_pages: list[int],
) -> int | None:
    for rank, result in enumerate(results, start=1):
        correct_filename = (
            result["filename"] == expected_filename
        )
        correct_page = (
            result["page_number"] in expected_pages
        )

        if correct_filename and correct_page:
            return rank

    return None


def run_evaluation() -> None:
    questions = json.loads(
        QUESTIONS_PATH.read_text(encoding="utf-8")
    )

    hits = 0
    reciprocal_rank_total = 0.0

    print()

    for item in questions:
        response = httpx.post(
            f"{API_BASE_URL}/search",
            json={
                "query": item["question"],
                "top_k": TOP_K,
            },
            timeout=300,
        )

        response.raise_for_status()

        results = response.json()["results"]

        expected_rank = find_expected_rank(
            results=results,
            expected_filename=item["expected_filename"],
            expected_pages=item["expected_pages"],
        )

        if expected_rank is not None:
            hits += 1
            reciprocal_rank_total += 1 / expected_rank
            status = "PASS"
            rank_text = str(expected_rank)
        else:
            status = "FAIL"
            rank_text = "not found"

        print(
            f"[{status}] {item['question']}\n"
            f"       Expected pages: {item['expected_pages']}\n"
            f"       Correct result rank: {rank_text}\n"
        )

    question_count = len(questions)
    hit_at_k = hits / question_count
    mean_reciprocal_rank = (
        reciprocal_rank_total / question_count
    )

    print("=" * 60)
    print(f"Questions: {question_count}")
    print(f"Hit@{TOP_K}: {hit_at_k:.2%}")
    print(f"MRR: {mean_reciprocal_rank:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            "Run evaluation/create_sample_data.py first."
        )

    index_evaluation_document()
    run_evaluation()