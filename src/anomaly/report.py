import json

from src.utils.paths import DOCS_DIR


class AnomalyReport:
    """
    Generates an anomaly detection summary report.
    """

    def generate(self, documents, flagged_documents):

        reason_counts = {}

        for flagged in flagged_documents:

            for reason_code in (
                reason.split(":")[0]
                for reason in flagged["reasons"]
            ):

                reason_counts[reason_code] = (
                    reason_counts.get(reason_code, 0) + 1
                )

        report = {
            "total_documents": len(documents),
            "flagged_documents": len(flagged_documents),
            "flagged_rate": round(
                len(flagged_documents)
                / max(len(documents), 1),
                3,
            ),
            "anomaly_reason_counts": reason_counts,
        }

        report_path = (
            DOCS_DIR
            / "reports"
            / "anomaly_report.json"
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
