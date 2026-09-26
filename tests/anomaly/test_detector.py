import pytest

from src.anomaly.detector import AnomalyDetector


def make_document(document_id, text):

    return {
        "id": document_id,
        "text": text,
        "metadata": {},
    }


# --------------------------------------------------
# Clean documents
# --------------------------------------------------


def test_clean_document_passes_through():

    documents = [
        make_document(
            1,
            "Python is a wonderful programming language "
            "for data science and machine learning.",
        ),
    ]

    detector = AnomalyDetector()

    accepted, flagged = detector.detect(documents)

    assert len(accepted) == 1

    assert flagged == []

    assert accepted[0]["metadata"]["anomaly"] is False


def test_missing_text_field_is_flagged():

    documents = [
        {
            "id": 1,
            "metadata": {},
        },
    ]

    detector = AnomalyDetector()

    accepted, flagged = detector.detect(documents)

    assert len(accepted) == 0

    assert len(flagged) == 1

    assert flagged[0]["reasons"] == ["too_short:0<20"]


def test_short_document_is_flagged():

    documents = [
        make_document(1, "short"),
    ]

    detector = AnomalyDetector()

    accepted, flagged = detector.detect(documents)

    assert len(accepted) == 0

    assert len(flagged) == 1

    assert flagged[0]["reasons"][0].startswith("too_short")


# --------------------------------------------------
# Repetition anomalies
# --------------------------------------------------


def test_repeated_word_document_is_flagged():

    documents = [
        make_document(
            1,
            "spam spam spam spam spam spam spam spam "
            "spam spam spam spam spam spam spam spam",
        ),
    ]

    detector = AnomalyDetector()

    accepted, flagged = detector.detect(documents)

    assert len(flagged) == 1

    assert any(
        reason.startswith("word_repetition")
        for reason in flagged[0]["reasons"]
    )


def test_gibberish_document_is_flagged():

    documents = [
        make_document(
            1,
            "xkcd vbnm qrtz plkw zzq bcdf gkt wqz "
            "reported unusual patterns near the coastline.",
        ),
    ]

    detector = AnomalyDetector()

    accepted, flagged = detector.detect(documents)

    assert len(flagged) == 1

    assert "gibberish" in flagged[0]["reasons"]


def test_symbol_heavy_document_is_flagged():

    documents = [
        make_document(
            1,
            "!@#$%^&*()!@#$%^&*()!@#$%^&*()!@#$%^&*()",
        ),
    ]

    detector = AnomalyDetector()

    accepted, flagged = detector.detect(documents)

    assert len(flagged) == 1

    assert any(
        reason.startswith("symbols")
        for reason in flagged[0]["reasons"]
    )


# --------------------------------------------------
# Flag-and-keep contract
# --------------------------------------------------


def test_flagged_documents_are_kept_not_removed():

    documents = [
        make_document(1, "short"),
        make_document(
            2,
            "Python is a wonderful programming language "
            "for data science and machine learning.",
        ),
    ]

    detector = AnomalyDetector()

    accepted, flagged = detector.detect(documents)

    # Every input document must still exist.
    assert len(accepted) + len(flagged) == 2

    flagged_entry = flagged[0]

    assert flagged_entry["document"]["id"] == 1

    assert flagged_entry["document"]["metadata"]["anomaly"] is True

    assert "anomaly_reasons" in flagged_entry["document"]["metadata"]


def test_multiple_reasons_accumulate():

    documents = [
        make_document(1, "!!!"),
    ]

    detector = AnomalyDetector()

    accepted, flagged = detector.detect(documents)

    reasons = flagged[0]["reasons"]

    assert "too_short:3<20" in reasons

    assert any(
        reason.startswith("symbols") for reason in reasons
    )


# --------------------------------------------------
# Threshold configuration
# --------------------------------------------------


def test_thresholds_are_configurable():

    documents = [
        make_document(
            1,
            "the cat sat the cat sat the cat sat the cat",
        ),
    ]

    # Lenient thresholds keep the document accepted.
    lenient = AnomalyDetector(
        word_repetition_threshold=0.99,
    )

    lenient_accepted, lenient_flagged = lenient.detect(documents)

    assert len(lenient_flagged) == 0

    # Strict thresholds flag the same document.
    strict = AnomalyDetector(
        word_repetition_threshold=0.05,
    )

    _, strict_flagged = strict.detect(documents)

    assert len(strict_flagged) == 1


def test_empty_document_list():

    detector = AnomalyDetector()

    accepted, flagged = detector.detect([])

    assert accepted == []

    assert flagged == []
