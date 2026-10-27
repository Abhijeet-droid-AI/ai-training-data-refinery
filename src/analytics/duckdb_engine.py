import duckdb
from contextlib import contextmanager
from typing import Any


class DuckDBEngine:
    """
    Executes SQL queries against Parquet datasets.

    Provides a context-manager interface so the connection is
    guaranteed to close after each analytic run.
    """

    def __init__(
        self,
        database: str | None = None,
        read_only: bool = False,
        config: dict[str, Any] | None = None,
    ):
        self.database = database
        self.read_only = read_only or database is not None
        self._config = config or {}
        self.connection: duckdb.DuckDBPyConnection | None = None

    @contextmanager
    def _connect(self):
        """
        Open a connection, yield it, and close it afterwards.

        The config is scoped to this connection so the engine can
        be reused for different settings (e.g. memory limits)
        without leaking state between queries.
        """
        if self.read_only:
            # For read-only mode, use a temporary read-only file-based
            # database so the in-memory variant (which cannot be opened
            # read-only) is not required.
            # The database is pre-created with a minimal schema so
            # DuckDB treats it as a valid database file.
            import tempfile
            import os

            tmp_fd, tmp_path = tempfile.mkstemp(suffix='.duckdb')
            os.close(tmp_fd)
            try:
                # Pre-create a valid DuckDB database file by connecting
                # to it, creating a dummy table, and closing.
                probe = duckdb.connect(database=tmp_path)
                probe.execute("CREATE TABLE probe (id INTEGER)")
                probe.close()

                self.connection = duckdb.connect(
                    database=tmp_path,
                    read_only=True,
                    **self._config,
                )
            except Exception:
                # Clean up the temp file on failure.
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
        else:
            self.connection = duckdb.connect(
                database=self.database or ':memory:',
                read_only=False,
                **self._config,
            )
        try:
            yield self.connection
        finally:
            self.close()

    def query(self, sql: str, params: dict[str, Any] | None = None):
        """
        Run a SQL statement and return the result as a pandas DataFrame.

        ``params`` is passed through to ``duckdb`` so that user
        supplied values are bound rather than interpolated, which
        removes the risk of SQL injection through the report
        templates.
        """
        with self._connect() as conn:
            if params is not None:
                # DuckDB uses $1, $2 ... positional parameters for
                # bound values.  Map the caller's dict to that form.
                sql, values = self._bind_params(sql, params)
                return conn.execute(sql, values).fetchdf()
            return conn.execute(sql).fetchdf()

    def close(self) -> None:
        """Close the underlying DuckDB connection, if open."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @classmethod
    def analytics_query(cls, sql: str) -> duckdb.DuckDBPyConnection:
        """
        One-shot query helper for analytics scripts.

        Opens a read-only connection, runs the SQL, closes the
        connection, and returns the raw ``duckdb`` result.  This is
        the helper used by the analytics report generators, so the
        legacy pattern of leaving connections open is gone.
        """
        engine = cls(read_only=True)
        with engine._connect() as conn:
            return conn.execute(sql)

    @staticmethod
    def _bind_params(sql: str, params: dict[str, Any]) -> str:
        """
        Convert a SQL string with named or positional ``:name``
        placeholders into DuckDB's ``$1, $2, ...`` positional
        placeholder syntax.

        DuckDB's ``execute(sql, params)`` binding only supports
        positional placeholders, but the analytics report templates
        sometimes use named placeholders.  This method rewrites them
        to positional ones and returns the parameter values in the
        order the placeholders appear.
        """
        import re

        def repl(match):
            name = match.group(1)
            return f"${len(re.findall(r':[a-z_][a-z0-9_]*', sql[:match.start()], re.IGNORECASE)) + 1}"

        # This naive regex approach is sufficient for the analytics
        # SQL in this project and avoids adding a dependency on a
        # full SQL parsing library.
        new_sql = re.sub(r':([a-z_][a-z0-9_]*)', repl, sql, flags=re.IGNORECASE)
        values = [params[name] for name in re.findall(r':([a-z_][a-z0-9_]*)', sql, re.IGNORECASE)]
        return new_sql, values
