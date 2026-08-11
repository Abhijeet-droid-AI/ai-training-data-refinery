import hashlib


class FingerprintGenerator:
    """
    Generates deterministic fingerprints for documents.
    """

    @staticmethod
    def generate(text: str) -> str:
        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()