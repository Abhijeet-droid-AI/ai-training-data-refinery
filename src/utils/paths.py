from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "configs"

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

INTERIM_DATA_DIR = DATA_DIR / "interim"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

PARQUET_DATA_DIR = DATA_DIR / "parquet"

PARQUET_FILE = PARQUET_DATA_DIR / "training_dataset.parquet"

DOCS_DIR = PROJECT_ROOT / "docs"