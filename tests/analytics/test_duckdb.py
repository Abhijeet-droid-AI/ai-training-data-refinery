from src.analytics.duckdb_engine import DuckDBEngine


def test_connection():

    db = DuckDBEngine()

    result = db.query("SELECT 1 AS number")

    assert result.iloc[0]["number"] == 1

    db.close()