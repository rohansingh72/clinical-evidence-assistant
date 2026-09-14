import numpy as np

from app.services.reranker_service import RerankerService


class FakeCrossEncoder:
    def predict(
        self,
        pairs: list[list[str]],
    ) -> np.ndarray:
        return np.array(
            [0.9, 0.2],
            dtype=np.float32,
        )


def test_reranker_scores_each_passage() -> None:
    service = RerankerService(
        model=FakeCrossEncoder()
    )

    scores = service.score_passages(
        question="What was the study objective?",
        passages=[
            "The objective was to evaluate efficacy.",
            "The weather was not recorded.",
        ],
    )

    assert scores.shape == (2,)
    assert scores[0] > scores[1]