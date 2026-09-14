import json
import os
import sys
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.reranker_service import RerankerService


EVALUATION_DIRECTORY = Path(__file__).resolve().parent
PDF_PATH = EVALUATION_DIRECTORY / "sample_clinical_study.pdf"
QUESTIONS_PATH = EVALUATION_DIRECTORY / "questions.json"

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)

CANDIDATE_COUNT = 3

reranker = RerankerService()


def index_document() -> None:
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


def rerank_results(
    question: str,
    results: list[dict],
) -> list[dict]:
    scores = reranker.score_passages(
        question=question,
        passages=[
            result["text"]
            for result in results
        ],
    )

    scored_results = []

    for result, reranker_score in zip(
        results,
        scores,
        strict=True,
    ):
        scored_result = result.copy()
        scored_result["reranker_score"] = float(
            reranker_score
        )
        scored_results.append(scored_result)

    return sorted(
        scored_results,
        key=lambda result: result["reranker_score"],
        reverse=True,
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

    answerable_scores: list[float] = []
    unanswerable_scores: list[float] = []

    for item in questions:
        response = httpx.post(
            f"{API_BASE_URL}/search",
            json={
                "query": item["question"],
                "top_k": CANDIDATE_COUNT,
            },
            timeout=300,
        )

        response.raise_for_status()

        original_results = response.json()["results"]

        reranked_results = rerank_results(
            question=item["question"],
            results=original_results,
        )

        top_score = (
            reranked_results[0]["reranker_score"]
            if reranked_results
            else float("-inf")
        )

        if item["answerable"]:
            answerable_count += 1
            answerable_scores.append(top_score)

            expected_rank = find_expected_rank(
                results=reranked_results,
                expected_filename=item["expected_filename"],
                expected_pages=item["expected_pages"],
            )

            if expected_rank is not None:
                hits += 1
                reciprocal_rank_total += 1 / expected_rank
                status = "PASS"
            else:
                status = "FAIL"

            print(
                f"[{status}] {item['question']}\n"
                f"       Reranked correct-page rank: "
                f"{expected_rank}\n"
                f"       Top reranker score: "
                f"{top_score:.4f}\n"
            )

        else:
            unanswerable_scores.append(top_score)

            print(
                f"[UNANSWERABLE] {item['question']}\n"
                f"       Top reranker score: "
                f"{top_score:.4f}\n"
            )

    hit_at_k = hits / answerable_count
    mrr = reciprocal_rank_total / answerable_count

    minimum_answerable = min(answerable_scores)
    maximum_unanswerable = max(unanswerable_scores)

    print("=" * 60)
    print(f"Hit@{CANDIDATE_COUNT}: {hit_at_k:.2%}")
    print(f"Reranked MRR: {mrr:.3f}")
    print(
        "Answerable reranker-score range: "
        f"{min(answerable_scores):.4f}–"
        f"{max(answerable_scores):.4f}"
    )
    print(
        "Unanswerable reranker-score range: "
        f"{min(unanswerable_scores):.4f}–"
        f"{max(unanswerable_scores):.4f}"
    )

    if minimum_answerable > maximum_unanswerable:
        suggested_threshold = (
            minimum_answerable
            + maximum_unanswerable
        ) / 2

        print(
            "Suggested reranker threshold: "
            f"{suggested_threshold:.4f}"
        )
    else:
        print(
            "Reranker scores still overlap. "
            "Use an explicit answerability classifier."
        )

    print("=" * 60)


if __name__ == "__main__":
    index_document()
    run_evaluation()