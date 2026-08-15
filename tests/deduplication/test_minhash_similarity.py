import pytest

from src.deduplication.minhash_similarity import MinHashSimilarity


def test_identical_signatures():

    signature = [1, 2, 3, 4, 5]

    similarity = MinHashSimilarity.estimate(
        signature,
        signature,
    )

    assert similarity == 1.0


def test_completely_different_signatures():

    signature_a = [1, 2, 3, 4, 5]

    signature_b = [6, 7, 8, 9, 10]

    similarity = MinHashSimilarity.estimate(
        signature_a,
        signature_b,
    )

    assert similarity == 0.0


def test_partial_match():

    signature_a = [1, 2, 3, 4, 5]

    signature_b = [1, 2, 8, 4, 9]

    similarity = MinHashSimilarity.estimate(
        signature_a,
        signature_b,
    )

    assert similarity == 0.6


def test_different_signature_lengths():

    with pytest.raises(ValueError):

        MinHashSimilarity.estimate(
            [1, 2, 3],
            [1, 2],
        )