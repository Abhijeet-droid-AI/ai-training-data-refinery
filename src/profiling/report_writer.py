import json
from pathlib import Path

from src.utils.paths import DOCS_DIR


class ReportWriter:
    def save(self, report):

        report_dir = DOCS_DIR / "reports"
        report_dir.mkdir(exist_ok=True)

        output_file = report_dir / "dataset_profile.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)