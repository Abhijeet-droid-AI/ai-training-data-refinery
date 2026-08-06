from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.normalizer import TextNormalizer


class PreprocessingPipeline:

    def __init__(self):

        self.cleaner = TextCleaner()

        self.normalizer = TextNormalizer()

    def process(self, documents):

        processed = []

        for doc in documents:

            processed.append(
                {
                    **doc,
                    "text": self.normalizer.normalize(
                        self.cleaner.clean(doc["text"])
                    ),
                }
            )

        return processed