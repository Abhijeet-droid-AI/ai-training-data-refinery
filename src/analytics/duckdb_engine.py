import duckdb


class DuckDBEngine:
    """
    Executes SQL queries against Parquet datasets.
    """

    def __init__(self):
        self.connection = duckdb.connect()

    def query(self, sql: str):
        return self.connection.execute(sql).fetchdf()

    def close(self):
        self.connection.close()