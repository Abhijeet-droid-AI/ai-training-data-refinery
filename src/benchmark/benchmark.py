import time

from pathlib import Path
from src.benchmark.pandas_engine import PandasEngine
from src.benchmark.polars_engine import PolarsEngine
from src.utils.paths import RAW_DATA_DIR

Dataset = RAW_DATA_DIR / "wiki_sample.json"

def benchmark(engine, name):
    start = time.perf_counter()
    df = engine.load(Dataset)
    end = time.perf_counter()
    elapsed = end - start
    print(f"{name}")
    print(f"Rows: {len(df)}")
    print(f"Execution Time: {elapsed:.6f} seconds")
    print("-" * 50)

benchmark(PandasEngine(), "Pandas")

benchmark(PolarsEngine(), "Polars")