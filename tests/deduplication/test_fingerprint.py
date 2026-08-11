from src.deduplication.fingerprint import FingerprintGenerator


def test_same_text_produces_same_fingerprint():

    text = "Python is awesome."

    fingerprint_1 = FingerprintGenerator.generate(text)
    fingerprint_2 = FingerprintGenerator.generate(text)

    assert fingerprint_1 == fingerprint_2


def test_different_text_produces_different_fingerprint():

    fingerprint_1 = FingerprintGenerator.generate(
        "Python is awesome."
    )

    fingerprint_2 = FingerprintGenerator.generate(
        "Python is amazing."
    )

    assert fingerprint_1 != fingerprint_2