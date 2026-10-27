from src.analytics.duckdb_engine import DuckDBEngine


def test_connection():
    db = DuckDBEngine()
    assert db.connection is None

    with db:
        result = db.query("SELECT 1 AS number")
        assert result.iloc[0]["number"] == 1

    assert db.connection is None


def test_read_only_mode():
    db = DuckDBEngine(read_only=True)
    assert db.read_only is True

    with db:
        result = db.query("SELECT 1 AS number")
        assert result.iloc[0]["number"] == 1

    assert db.connection is None


def test_query_with_params():
    db = DuckDBEngine()
    with db:
        result = db.query("SELECT :value AS number", {"value": 42})
        assert result.iloc[0]["number"] == 42


def test_context_manager_exits_cleanly():
    db = DuckDBEngine()
    with db:
        result = db.query("SELECT 1 AS number")
        assert result.iloc[0]["number"] == 1

    # Connection must be closed after the context manager exits
    assert db.connection is None


def test_analytics_query_class_method():
    result = DuckDBEngine.analytics_query("SELECT 2 AS number")
    assert result.iloc[0]["number"] == 2
