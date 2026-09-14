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
        if (
            result["filename"] == expected_filename
            and result["page_number"] in expected_pages
        ):
            return rank

    return None


def run_evaluation() -> None:
    questions = json.loads(
        QUESTIONS_PATH.read_text(encoding="utf-8")
    )

    answerable_count = 0
    hits = 0
    reciprocal_rank_total = 0.0

    answerable_top_scores: list[float] = []
    unanswerable_top_scores: list[float] = []

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

        top_score = (
            results[0]["similarity_score"]
            if results
            else 0.0
        )

        if item["answerable"]:
            answerable_count += 1
            answerable_top_scores.append(top_score)

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
                f"       Correct result rank: {rank_text}\n"
                f"       Top similarity score: {top_score:.4f}\n"
            )

        else:
            unanswerable_top_scores.append(top_score)

            top_match = (
                f"{results[0]['filename']}, "
                f"page {results[0]['page_number']}"
                if results
                else "none"
            )

            print(
                f"[UNANSWERABLE] {item['question']}\n"
                f"       Top returned source: {top_match}\n"
                f"       Top similarity score: {top_score:.4f}\n"
            )

    hit_at_k = hits / answerable_count
    mean_reciprocal_rank = (
        reciprocal_rank_total / answerable_count
    )

    minimum_answerable_score = min(answerable_top_scores)
    maximum_unanswerable_score = max(unanswerable_top_scores)

    print("=" * 60)
    print(f"Answerable questions: {answerable_count}")
    print(f"Hit@{TOP_K}: {hit_at_k:.2%}")
    print(f"MRR: {mean_reciprocal_rank:.3f}")
    print(
        "Answerable top-score range: "
        f"{min(answerable_top_scores):.4f}–"
        f"{max(answerable_top_scores):.4f}"
    )
    print(
        "Unanswerable top-score range: "
        f"{min(unanswerable_top_scores):.4f}–"
        f"{max(unanswerable_top_scores):.4f}"
    )

    if minimum_answerable_score > maximum_unanswerable_score:
        suggested_threshold = (
            minimum_answerable_score
            + maximum_unanswerable_score
        ) / 2

        print(
            "Suggested initial threshold: "
            f"{suggested_threshold:.4f}"
        )
    else:
        print(
            "No clean score threshold separates the two groups. "
            "A stronger abstention method is required."
        )

    print("=" * 60)


if __name__ == "__main__":
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            "Run evaluation/create_sample_data.py first."
        )

    index_evaluation_document()
    run_evaluation()