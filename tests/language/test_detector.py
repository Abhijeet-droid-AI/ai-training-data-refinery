from src.language.detector import LanguageDetector


def test_language_detection():
    detector = LanguageDetector()

    language = detector.detect_language(
        "Python is a programming language."
    )

    assert language == "en"