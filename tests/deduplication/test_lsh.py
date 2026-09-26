import pytest

from src.deduplication.lsh import LSH


def make_signature(values: list[int]) -> list[int]:
    return values + values


# --------------------------------------------------
# Constructor validation
# --------------------------------------------------


def test_rejects_zero_bands():

    with pytest.raises(ValueError):

        LSH(num_bands=0, rows_per_band=5)


def test_rejects_zero_rows_per_band():

    with pytest.raises(ValueError):

        LSH(num_bands=20, rows_per_band=0)


# --------------------------------------------------
# Signature length validation
# --------------------------------------------------


def test_rejects_wrong_signature_length():

    lsh = LSH(num_bands=4, rows_per_band=2)

    with pytest.raises(ValueError):

        lsh.add("doc_1", [1, 2, 3])


def test_signature_length_property():

    lsh = LSH(num_bands=4, rows_per_band=2)

    assert lsh.signature_length == 8


# --------------------------------------------------
# Candidate pair generation
# --------------------------------------------------


def test_identical_signatures_become_candidates():

    lsh = LSH(num_bands=4, rows_per_band=2)

    signature = make_signature([1, 2, 3, 4])

    lsh.add("doc_1", signature)

    lsh.add("doc_2", signature)

    pairs = lsh.candidate_pairs()

    assert ("doc_1", "doc_2") in pairs


def test_different_signatures_are_not_candidates():

    lsh = LSH(num_bands=4, rows_per_band=2)

    signature_a = make_signature([1, 2, 3, 4])

    signature_b = make_signature([9, 9, 9, 9])

    lsh.add("doc_1", signature_a)

    lsh.add("doc_2", signature_b)

    assert lsh.candidate_pairs() == set()


def test_partial_band_match_generates_candidate():

    lsh = LSH(num_bands=4, rows_per_band=2)

    signature_a = [1, 2, 3, 4, 5, 6, 7, 8]

    # Only the third band matches.
    signature_b = [9, 9, 3, 4, 9, 9, 9, 9]

    lsh.add("doc_1", signature_a)

    lsh.add("doc_2", signature_b)

    pairs = lsh.candidate_pairs()

    assert ("doc_1", "doc_2") in pairs


def test_duplicate_documents_share_bands():

    lsh = LSH(num_bands=20, rows_per_band=5)

    signature_a = list(range(100))

    signature_b = list(range(100))

    lsh.add("doc_a", signature_a)

    lsh.add("doc_b", signature_b)

    lsh.add("doc_c", signature_a)

    pairs = lsh.candidate_pairs()

    assert ("doc_a", "doc_b") in pairs

    assert ("doc_a", "doc_c") in pairs

    assert ("doc_b", "doc_c") in pairs


def test_reindexing_same_document_does_not_duplicate_pairs():

    lsh = LSH(num_bands=4, rows_per_band=2)

    signature = make_signature([1, 2, 3, 4])

    lsh.add("doc_1", signature)

    lsh.add("doc_1", signature)

    lsh.add("doc_2", signature)

    pairs = lsh.candidate_pairs()

    assert pairs == {("doc_1", "doc_2")}


# --------------------------------------------------
# Band key behavior
# --------------------------------------------------


def test_band_key_is_deterministic():

    lsh = LSH(num_bands=2, rows_per_band=2)

    signature = [1, 2, 3, 4]

    key_1 = lsh._band_key(signature, 0)

    key_2 = lsh._band_key(signature, 0)

    assert key_1 == key_2


def test_band_key_matches_only_same_band_rows():

    lsh = LSH(num_bands=2, rows_per_band=2)

    signature = [1, 2, 3, 4]

    key_band_0 = lsh._band_key(signature, 0)

    key_band_1 = lsh._band_key(signature, 1)

    assert key_band_0 != key_band_1


# --------------------------------------------------
# Threshold property
# --------------------------------------------------


def test_threshold_for_default_configuration():

    lsh = LSH(num_bands=20, rows_per_band=5)

    # (1/20)^(1/5) ~ 0.5493
    assert 0.54 <= lsh.similarity_threshold <= 0.56


def test_threshold_is_one_for_single_band_single_row():

    lsh = LSH(num_bands=1, rows_per_band=1)

    assert lsh.similarity_threshold == 1.0
