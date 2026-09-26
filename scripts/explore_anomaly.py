from src.anomaly.heuristics import AnomalyHeuristics
from src.anomaly.detector import AnomalyDetector


document_a = """
Python is an amazing programming language used
for data science and artificial intelligence.
"""


document_b = """
spam spam spam spam spam spam spam spam spam
spam spam spam spam spam spam spam spam spam
"""


document_c = """
xkcd vbnm qrtz plkw zzq vbt is the standard
reference for modern engineering work everywhere.
"""


documents = [
    {"id": "document_a", "text": document_a, "metadata": {}},
    {"id": "document_b", "text": document_b, "metadata": {}},
    {"id": "document_c", "text": document_c, "metadata": {}},
]


# --------------------------------------------------
# Individual Heuristics
# --------------------------------------------------

for document in documents:

    text = document["text"]

    print("=" * 60)
    print(document["id"].upper())
    print("=" * 60)

    print(
        "Character repetition : "
        f"{AnomalyHeuristics.character_repetition_ratio(text):.3f}"
    )

    print(
        "Word repetition      : "
        f"{AnomalyHeuristics.word_repetition_ratio(text):.3f}"
    )

    print(
        "Symbol ratio         : "
        f"{AnomalyHeuristics.symbol_ratio(text):.3f}"
    )

    print(
        "Gibberish            : "
        f"{AnomalyHeuristics.is_gibberish(text)}"
    )


# --------------------------------------------------
# Detector Output
# --------------------------------------------------

detector = AnomalyDetector()

accepted, flagged = detector.detect(documents)


print()
print("=" * 60)
print("ANOMALY DETECTION SUMMARY")
print("=" * 60)

print(f"Accepted: {len(accepted)} | Flagged: {len(flagged)}")

for flagged_entry in flagged:

    print(
        f"  {flagged_entry['document']['id']} -> "
        f"{flagged_entry['reasons']}"
    )

print("=" * 60)
