from pathlib import Path
from src.ingestion.loader import DataLoader
from src.ingestion.validator import DataValidator
from src.utils.logger import get_logger
from src.utils.paths import RAW_DATA_DIR
from src.utils.config import load_config
from src.profiling.profiler import DatasetProfiler
from src.profiling.report_writer import ReportWriter
from src.storage.parquet_converter import ParquetConverter
from src.utils.paths import PARQUET_FILE

logger = get_logger(__name__)
# logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

def main():
    dataset_path = RAW_DATA_DIR / "wiki_sample.json"
    config = load_config()

    loader = DataLoader(
        RAW_DATA_DIR / config["data"]["raw_file"]
    )

    try:
        documents = loader.load()
    except FileNotFoundError:
        logger.error("Dataset not found.")
        return

    validator = DataValidator()

    valid_docs, invalid_docs = validator.validate(documents)

    logger.info("=" * 60)
    logger.info("Data Ingestion Summary")
    logger.info("=" * 60)
    logger.info(f"Total Documents   : {len(documents)}")
    logger.info(f"Valid Documents   : {len(valid_docs)}")
    logger.info(f"Invalid Documents : {len(invalid_docs)}")
    logger.info("=" * 60)

    profiler = DatasetProfiler(valid_docs)

    report = profiler.profile()

    report_writer = ReportWriter()

    report_writer.save(report)

    logger.info("Dataset profile generated successfully.")

    converter = ParquetConverter(valid_docs)

    converter.convert(PARQUET_FILE)

    logger.info("Parquet dataset created successfully.")

    if invalid_docs:
        logger.warning("Invalid document details:")

        for invalid in invalid_docs:
            logger.warning(
                f"Document ID: {invalid['document'].get('id', 'Unknown')} | "
                f"Missing Fields: {invalid['missing_fields']}"
            )

if __name__ == "__main__":
    main()