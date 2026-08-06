import json
from collections import Counter

from src.utils.paths import DOCS_DIR


class LanguageReport:
    """
    Generates a language distribution report.
    """

    def generate(self, documents):

        languages = [
            doc["metadata"]["language"]
            for doc in documents
        ]

        distribution = dict(Counter(languages))

        output = {
            "total_documents": len(documents),
            "language_distribution": distribution,
        }

        report_path = DOCS_DIR / "reports" / "language_report.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)

        return output