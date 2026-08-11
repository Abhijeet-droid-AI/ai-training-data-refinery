from src.deduplication.report import DeduplicationReport


def test_deduplication_report():

    unique_docs = [
        {
            "id": 1,
            "text": "Python",
            "metadata": {},
        }
    ]

    duplicate_docs = [
        {
            "document": {
                "id": 2,
                "text": "Python",
                "metadata": {},
            },
            "reason": "duplicate",
        }
    ]

    report_writer = DeduplicationReport()

    report = report_writer.generate(
        unique_docs,
        duplicate_docs,
    )

    assert report["total_documents"] == 2
    assert report["unique_documents"] == 1
    assert report["duplicates_removed"] == 1