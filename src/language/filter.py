class LanguageFilter:
    """
    Filters documents by supported languages.
    """

    def __init__(self, supported_languages=None):
        self.supported_languages = supported_languages or ["en"]

    def filter(self, documents):
        accepted = []
        rejected = []

        for doc in documents:
            if doc["metadata"]["language"] in self.supported_languages:
                accepted.append(doc)
            else:
                rejected.append({
                    "documents": doc,
                    "reason": f"Unsupported language: {doc['metadata']['language']}" 
                })

        return accepted, rejected