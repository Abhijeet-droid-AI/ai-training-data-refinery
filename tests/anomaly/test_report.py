import json

from src.anomaly.report import AnomalyReport


def test_anomaly_report_computes_counts():

    documents = [
        {"id": 1, "metadata": {}},
        {"id": 2, "metadata": {}},
        {"id": 3, "metadata": {}},
    ]

    flagged_documents = [
        {
            "document": {"id": 2, "metadata": {}},
            "reasons": ["too_short:5<20"],
        },
        {
            "document": {"id": 3, "metadata": {}},
            "reasons": ["gibberish", "symbols:0.40"],
        },
    ]

    report = AnomalyReport().generate(
        documents,
        flagged_documents,
    )

    assert report["total_documents"] == 3

    assert report["flagged_documents"] == 2

    assert report["flagged_rate"] == 0.667

    assert report["anomaly_reason_counts"] == {
        "too_short": 1,
        "gibberish": 1,
        "symbols": 1,
    }


def test_anomaly_report_empty_dataset():

    report = AnomalyReport().generate([], [])

    assert report["total_documents"] == 0

    assert report["flagged_documents"] == 0

    assert report["flagged_rate"] == 0


def test_anomaly_report_is_written_to_disk():

    documents = [{"id": 1, "metadata": {}}]

    AnomalyReport().generate(documents, [])

    with open(
        "docs/reports/anomaly_report.json",
        encoding="utf-8",
    ) as file:

        report = json.load(file)

    assert report["total_documents"] == 1
