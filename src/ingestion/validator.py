class DataValidator:
    """Validates document structure."""

    REQUIRED_FIELDS = ["id", "title", "text"]

    def validate(self, documents):
        valid_docs = []
        invalid_docs = []

        for doc in documents:
            missing_fields = [
                field for field in self.REQUIRED_FIELDS if field not in doc
            ]

            if not missing_fields:
                valid_docs.append(doc)
            else:
                invalid_docs.append({"document": doc, "missing_fields": missing_fields})

        return valid_docs, invalid_docs
