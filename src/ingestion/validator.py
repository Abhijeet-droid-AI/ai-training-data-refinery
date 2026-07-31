class DataValidator:
    """Validates document structure."""

    REQUIRED_FIELDS = ["id", "title", "text"]

    def validate(self, documents):
        valid_docs = []

        for doc in documents:
            if all(field in doc for field in self.REQUIRED_FIELDS):
                valid_docs.append(doc)

        return valid_docs