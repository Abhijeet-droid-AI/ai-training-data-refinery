from src.benchmark.polars_engine import PolarsEngine
from src.utils.paths import RAW_DATA_DIR


def test_polars_loader():

    df = PolarsEngine.load(
        RAW_DATA_DIR / "wiki_sample.json"
    )

    assert len(df) == 4