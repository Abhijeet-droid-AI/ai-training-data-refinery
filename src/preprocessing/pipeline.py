from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.normalizer import TextNormalizer
from src.language.detector import LanguageDetector
from src.quality.scorer import QualityScorer

class PreprocessingPipeline:

    def __init__(self):
        self.cleaner = TextCleaner()
        self.normalizer = TextNormalizer()
        self.detector = LanguageDetector()
        self.quality = QualityScorer()

    
    def process(self, documents):
        processed_documents = []


        for doc in documents:

            cleaned_text, cleaner_metadata = self.cleaner.clean(
                doc["text"]
            )

            normalized_text, normalizer_metadata = (
                self.normalizer.normalize(cleaned_text)
            )

            language = self.detector.detect_language(normalized_text)

            quality_metadata = self.quality.score(normalized_text)

            processed_documents.append(
                {
                    **doc,
                    "text": normalized_text,
                    "metadata": {
                        **cleaner_metadata,
                        **normalizer_metadata,
                        "language": language,
                        **quality_metadata,
                    },
                }
            )

        return processed_documents