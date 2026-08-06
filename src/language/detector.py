from langdetect import DetectorFactory, detect

DetectorFactory.seed = 42


class LanguageDetector:
    """
    Detects the language of a document.
    """

    def detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except Exception:
            return "unknown"