import pytest

from src.deduplication.detector import DuplicateDetector


def make_document(document_id, text):

    return {
        "id": document_id,
        "text": text,
        "metadata": {},
    }


# --------------------------------------------------
# Exact duplicates (existing behavior preserved)
# --------------------------------------------------


def test_exact_duplicate_detection():

    documents = [
        make_document(1, "Python is awesome."),
        make_document(2, "Python is awesome."),
    ]

    detector = DuplicateDetector()

    unique_docs, duplicate_docs = detector.detect(documents)

    assert len(unique_docs) == 1

    assert len(duplicate_docs) == 1

    assert duplicate_docs[0]["reason"] == "duplicate"

    assert duplicate_docs[0]["duplicate_of"] == 1


# --------------------------------------------------
# Near-duplicate detection
# --------------------------------------------------


def test_near_duplicate_is_detected():

    documents = [
        make_document(
            1,
            "The quick brown fox jumps over the lazy dog "
            "near the river bank every single morning.",
        ),
        make_document(
            2,
            "The quick brown fox jumps over the lazy dog "
            "near the river bank every single evening.",
        ),
    ]

    detector = DuplicateDetector(
        near_duplicate_threshold=0.5,
    )

    unique_docs, duplicate_docs = detector.detect(documents)

    assert len(duplicate_docs) == 0

    assert len(detector.near_duplicates) == 1

    near_duplicate = detector.near_duplicates[0]

    assert near_duplicate["reason"] == "near_duplicate"

    assert near_duplicate["duplicate_of"] == 1

    assert near_duplicate["document"]["id"] == 2

    metadata = near_duplicate["document"]["metadata"]

    assert metadata["is_near_duplicate"] is True

    assert metadata["near_duplicate_of"] == 1

    assert metadata["is_duplicate"] is False


def test_unrelated_documents_are_not_near_duplicates():

    documents = [
        make_document(
            1,
            "The quick brown fox jumps over the lazy dog "
            "near the river bank every single morning.",
        ),
        make_document(
            2,
            "Docker containers orchestrate Kubernetes "
            "clusters across distributed cloud systems.",
        ),
    ]

    detector = DuplicateDetector()

    unique_docs, duplicate_docs = detector.detect(documents)

    assert len(unique_docs) == 2

    assert detector.near_duplicates == []


def test_unique_document_does_not_get_near_duplicate_metadata():

    documents = [
        make_document(
            1,
            "The quick brown fox jumps over the lazy dog "
            "near the river bank every single morning.",
        ),
        make_document(
            2,
            "Docker containers orchestrate Kubernetes "
            "clusters across distributed cloud systems.",
        ),
    ]

    detector = DuplicateDetector()

    unique_docs, _ = detector.detect(documents)

    for document in unique_docs:

        assert "is_near_duplicate" not in document["metadata"]


def test_three_way_near_duplicate_chain():

    documents = [
        make_document(
            1,
            "The quick brown fox jumps over the lazy dog "
            "near the river bank every single morning.",
        ),
        make_document(
            2,
            "The quick brown fox jumps over the lazy dog "
            "near the river bank every single evening.",
        ),
        make_document(
            3,
            "The quick brown fox leaps over the lazy dog "
            "near the river bank every single evening.",
        ),
    ]

    detector = DuplicateDetector(
        near_duplicate_threshold=0.5,
    )

    detector.detect(documents)

    # Document 2 matches document 1, document 3 matches
    # document 2, so both are flagged.
    assert len(detector.near_duplicates) == 2


def test_single_document_has_no_near_duplicates():

    documents = [
        make_document(
            1,
            "The quick brown fox jumps over the lazy dog.",
        ),
    ]

    detector = DuplicateDetector()

    unique_docs, duplicate_docs = detector.detect(documents)

    assert len(unique_docs) == 1

    assert detector.near_duplicates == []


def test_near_duplicate_threshold_can_be_configured():

    documents = [
        make_document(
            1,
            "The quick brown fox jumps over the lazy dog "
            "near the river bank every single morning.",
        ),
        make_document(
            2,
            "Docker containers orchestrate Kubernetes "
            "clusters across distributed cloud systems.",
        ),
    ]

    # Threshold above any achievable similarity keeps
    # everything unique.
    detector = DuplicateDetector(
        near_duplicate_threshold=1.1,
    )

    detector.detect(documents)

    assert detector.near_duplicates == []


# --------------------------------------------------
# Edge cases
# --------------------------------------------------


def test_empty_document_list():

    detector = DuplicateDetector()

    unique_docs, duplicate_docs = detector.detect([])

    assert unique_docs == []

    assert duplicate_docs == []

    assert detector.near_duplicates == []


def test_exact_duplicate_takes_precedence_over_near_duplicate():

    documents = [
        make_document(1, "Python is awesome."),
        make_document(2, "Python is awesome."),
        make_document(3, "Python is awesome."),
    ]

    detector = DuplicateDetector()

    unique_docs, duplicate_docs = detector.detect(documents)

    assert len(unique_docs) == 1

    assert len(duplicate_docs) == 2

    assert detector.near_duplicates == []


def test_detector_is_reusable_across_calls():

    documents = [
        make_document(
            1,
            "The quick brown fox jumps over the lazy dog "
            "near the river bank every single morning.",
        ),
        make_document(
            2,
            "The quick brown fox jumps over the lazy dog "
            "near the river bank every single evening.",
        ),
    ]

    detector = DuplicateDetector(
        near_duplicate_threshold=0.5,
    )

    detector.detect(documents)

    assert len(detector.near_duplicates) == 1

    # A fresh call must not accumulate stale results.
    detector.detect(
        [
            make_document(
                1,
                "The quick brown fox jumps over the lazy dog "
                "near the river bank every single morning.",
            ),
        ]
    )

    assert detector.near_duplicates == []
