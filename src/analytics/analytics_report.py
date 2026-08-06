import json

from src.analytics.duckdb_engine import DuckDBEngine
from src.utils.paths import DOCS_DIR, PARQUET_FILE


class AnalyticsReport:

    def generate(self):

        db = DuckDBEngine()

        query = f"""
        SELECT
            COUNT(*) AS total_documents,
            AVG(LENGTH(text)) AS average_text_length,
            MIN(LENGTH(text)) AS minimum_text_length,
            MAX(LENGTH(text)) AS maximum_text_length,
            COUNT(DISTINCT title) AS unique_titles,
            AVG(LENGTH(title)) AS average_title_length
        FROM read_parquet('{PARQUET_FILE.as_posix()}')
        """

        result = db.query(query)

        report = result.to_dict(orient="records")[0]

        report_path = DOCS_DIR / "reports" / "duckdb_report.json"

        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        db.close()

        return report

    #future dashboard preview.
    def dataset_preview(self):
        db = DuckDBEngine()

        query = f"""
        SELECT *
        FROM read_parquet('{PARQUET_FILE.as_posix()}')
        LIMIT 5
        """

        result = db.query(query)

        db.close()

        return result.to_dict(orient="records")