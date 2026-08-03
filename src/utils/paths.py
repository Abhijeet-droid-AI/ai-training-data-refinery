from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA = DATA_DIR / "raw"

INTERIM_DATA = DATA_DIR / "interim"

PROCESSED_DATA = DATA_DIR / "processed"

PARQUET_DATA = DATA_DIR / "parquet"

CONFIG_DIR = PROJECT_ROOT / "configs"

DOCS_DIR = PROJECT_ROOT / "docs"