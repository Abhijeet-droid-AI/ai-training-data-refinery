from src.deduplication.detector import DuplicateDetector


def test_duplicate_detection():

    documents = [
        {
            "id": 1,
            "text": "Python is awesome.",
            "metadata": {},
        },
        {
            "id": 2,
            "text": "Python is awesome.",
            "metadata": {},
        },
        {
            "id": 3,
            "text": "Artificial Intelligence is fascinating.",
            "metadata": {},
        },
    ]

    detector = DuplicateDetector()

    unique_docs, duplicate_docs = detector.detect(documents)

    # Two unique documents
    assert len(unique_docs) == 2

    # One duplicate
    assert len(duplicate_docs) == 1

    # --------------------------------------------------
    # Verify Unique Document Metadata
    # --------------------------------------------------

    first_document = unique_docs[0]

    assert first_document["metadata"]["is_duplicate"] is False

    assert "fingerprint" in first_document["metadata"]

    assert first_document["metadata"]["duplicate_group"].startswith(
        "sha256:"
    )

    # --------------------------------------------------
    # Verify Duplicate Metadata
    # --------------------------------------------------

    duplicate = duplicate_docs[0]

    assert duplicate["reason"] == "duplicate"

    assert duplicate["duplicate_of"] == 1

    assert duplicate["document"]["id"] == 2

    assert duplicate["document"]["metadata"]["is_duplicate"] is True

    assert "fingerprint" in duplicate["document"]["metadata"]