from src.deduplication.fingerprint import FingerprintGenerator


class DuplicateDetector:
    """
    Detects exact duplicate documents using SHA256 fingerprints.
    """

    def detect(self, documents):

        seen = {}

        unique_documents = []
        duplicate_documents = []

        for document in documents:

            document.setdefault("metadata", {})

            fingerprint = FingerprintGenerator.generate(
                document["text"]
            )

            duplicate_group = f"sha256:{fingerprint}"

            # --------------------------------------------------
            # Duplicate Document
            # --------------------------------------------------
            if fingerprint in seen:

                original_document = seen[fingerprint]

                document["metadata"]["fingerprint"] = fingerprint
                document["metadata"]["is_duplicate"] = True
                document["metadata"]["duplicate_group"] = (
                    duplicate_group
                )

                duplicate_documents.append(
                    {
                        "document": document,
                        "reason": "duplicate",
                        "fingerprint": fingerprint,
                        "duplicate_of": original_document["id"],
                    }
                )

            # --------------------------------------------------
            # Unique Document
            # --------------------------------------------------
            else:

                document["metadata"]["fingerprint"] = fingerprint
                document["metadata"]["is_duplicate"] = False
                document["metadata"]["duplicate_group"] = (
                    duplicate_group
                )

                seen[fingerprint] = document

                unique_documents.append(document)

        return unique_documents, duplicate_documents