from src.quality.scorer import QualityScorer


def test_good_document():

    scorer = QualityScorer()

    metadata = scorer.score(
        "Python is one of the most popular programming languages."
    )

    assert metadata["quality_score"] >= 90

    assert metadata["accepted"] is True


def test_empty_document():

    scorer = QualityScorer()

    metadata = scorer.score("")

    assert metadata["accepted"] is False