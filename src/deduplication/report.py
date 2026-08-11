import json

from src.utils.paths import DOCS_DIR


class DeduplicationReport:
    """
    Generates a deduplication summary report.
    """

    def generate(self, unique_docs, duplicate_docs):

        report = {
            "total_documents": (
                len(unique_docs) + len(duplicate_docs)
            ),
            "unique_documents": len(unique_docs),
            "duplicates_removed": len(duplicate_docs),
            "duplicate_rate": round(
                len(duplicate_docs)
                / max(
                    len(unique_docs) + len(duplicate_docs),
                    1,
                ),
                3,
            ),
        }

        report_path = (
            DOCS_DIR
            / "reports"
            / "deduplication_report.json"
        )

        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            report_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
            )

        return report