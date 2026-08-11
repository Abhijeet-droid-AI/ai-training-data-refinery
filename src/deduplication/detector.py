from src.deduplication.fingerprint import FingerprintGenerator


class DuplicateDetector:
    """
    Removes exact duplicate documents.
    """

    def detect(self, documents):

        seen = set()

        unique = []

        duplicates = []

        for doc in documents:

            fingerprint = FingerprintGenerator.generate(
                doc["text"]
            )

            if fingerprint in seen:

                duplicates.append(
                    {
                        "document": doc,
                        "reason": "duplicate",
                        "fingerprint": fingerprint,
                    }
                )

            else:

                seen.add(fingerprint)

                doc["metadata"]["fingerprint"] = fingerprint

                unique.append(doc)

        return unique, duplicates