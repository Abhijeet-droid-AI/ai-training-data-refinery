import json

from src.utils.paths import DOCS_DIR


class DeduplicationReport:

    def generate(self, unique_docs, duplicate_docs):

        report = {
            "total_documents":
                len(unique_docs) + len(duplicate_docs),

            "unique_documents":
                len(unique_docs),

            "duplicates_removed":
                len(duplicate_docs),
        }

        report_path = (
            DOCS_DIR
            / "reports"
            / "deduplication_report.json"
        )

        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        return report