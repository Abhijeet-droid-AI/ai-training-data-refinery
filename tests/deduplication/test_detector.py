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

    assert len(unique_docs) == 2
    assert len(duplicate_docs) == 1

    assert duplicate_docs[0]["document"]["id"] == 2