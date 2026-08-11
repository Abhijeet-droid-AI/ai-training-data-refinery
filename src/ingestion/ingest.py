from src.analytics.analytics_report import AnalyticsReport
from src.deduplication.detector import DuplicateDetector
from src.deduplication.report import DeduplicationReport
from src.ingestion.loader import DataLoader
from src.ingestion.validator import DataValidator
from src.language.filter import LanguageFilter
from src.language.report import LanguageReport
from src.preprocessing.pipeline import PreprocessingPipeline
from src.profiling.profiler import DatasetProfiler
from src.profiling.report_writer import ReportWriter
from src.storage.parquet_converter import ParquetConverter
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.paths import PARQUET_FILE, RAW_DATA_DIR


logger = get_logger(__name__)


def main() -> None:
    """
    Execute the complete AI Training Data Refinery pipeline.

    Pipeline:
        Load Dataset
            ↓
        Validate Documents
            ↓
        Preprocess Text
            ↓
        Detect Language
            ↓
        Filter Languages
            ↓
        Deduplicate
            ↓
        Profile Dataset
            ↓
        Convert to Parquet
            ↓
        Generate DuckDB Analytics
    """

    try:
        # --------------------------------------------------
        # Load Configuration
        # --------------------------------------------------
        config = load_config()

        dataset_path = RAW_DATA_DIR / config["data"]["raw_file"]

        logger.info("Loading dataset: %s", dataset_path)

        loader = DataLoader(dataset_path)

        documents = loader.load()

        # --------------------------------------------------
        # Validate Documents
        # --------------------------------------------------
        validator = DataValidator()

        valid_docs, invalid_docs = validator.validate(documents)

        logger.info("=" * 60)
        logger.info("Data Ingestion Summary")
        logger.info("=" * 60)
        logger.info("Total Documents   : %d", len(documents))
        logger.info("Valid Documents   : %d", len(valid_docs))
        logger.info("Invalid Documents : %d", len(invalid_docs))
        logger.info("=" * 60)

        # --------------------------------------------------
        # Preprocessing
        # --------------------------------------------------
        logger.info("Starting preprocessing pipeline...")

        preprocessing_pipeline = PreprocessingPipeline()

        processed_docs = preprocessing_pipeline.process(valid_docs)

        logger.info(
            "Successfully preprocessed %d documents.",
            len(processed_docs),
        )

        # --------------------------------------------------
        # Language Distribution Report
        # --------------------------------------------------
        logger.info("Generating language distribution report...")

        language_report_writer = LanguageReport()

        language_report = language_report_writer.generate(
            processed_docs
        )

        logger.info(
            "Language distribution: %s",
            language_report,
        )

        # --------------------------------------------------
        # Language Filtering
        # --------------------------------------------------
        logger.info("Filtering supported languages...")

        language_filter = LanguageFilter(["en"])

        filtered_docs, rejected_docs = language_filter.filter(
            processed_docs
        )

        logger.info(
            "Language Filter | Accepted: %d | Rejected: %d",
            len(filtered_docs),
            len(rejected_docs),
        )

        # --------------------------------------------------
        # Deduplication
        # --------------------------------------------------
        logger.info("Starting duplicate detection...")

        duplicate_detector = DuplicateDetector()

        unique_docs, duplicate_docs = duplicate_detector.detect(
            filtered_docs
        )

        logger.info(
            "Deduplication | Unique: %d | Duplicates: %d",
            len(unique_docs),
            len(duplicate_docs),
        )

        # --------------------------------------------------
        # Deduplication Report
        # --------------------------------------------------
        dedup_report_writer = DeduplicationReport()

        dedup_report = dedup_report_writer.generate(
            unique_docs,
            duplicate_docs,
        )

        logger.info(
            "Deduplication report generated: %s",
            dedup_report,
        )

        # --------------------------------------------------
        # Dataset Profiling
        # --------------------------------------------------
        logger.info("Generating dataset profile...")

        profiler = DatasetProfiler(unique_docs)

        profile_report = profiler.profile()

        report_writer = ReportWriter()

        report_writer.save(profile_report)

        logger.info("Dataset profile generated successfully.")

        # --------------------------------------------------
        # Convert to Parquet
        # --------------------------------------------------
        logger.info("Converting dataset to Parquet...")

        converter = ParquetConverter(unique_docs)

        records_written = converter.convert(PARQUET_FILE)

        logger.info(
            "Parquet conversion completed (%d records written).",
            records_written,
        )

        # --------------------------------------------------
        # DuckDB Analytics
        # --------------------------------------------------
        logger.info("Running DuckDB analytics...")

        analytics = AnalyticsReport()

        analytics_report = analytics.generate()

        logger.info("Analytics report generated successfully.")

        logger.info("Analytics Summary")

        for key, value in analytics_report.items():
            logger.info("%s : %s", key, value)

        # --------------------------------------------------
        # Invalid Documents
        # --------------------------------------------------
        if invalid_docs:
            logger.warning("=" * 60)
            logger.warning("Invalid Documents")
            logger.warning("=" * 60)

            for invalid in invalid_docs:
                logger.warning(
                    "Document ID: %s | Missing Fields: %s",
                    invalid["document"].get("id", "Unknown"),
                    invalid["missing_fields"],
                )

        # --------------------------------------------------
        # Pipeline Summary
        # --------------------------------------------------
        logger.info("=" * 60)
        logger.info("FINAL PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info("Input Documents       : %d", len(documents))
        logger.info("Valid Documents       : %d", len(valid_docs))
        logger.info("Language Accepted     : %d", len(filtered_docs))
        logger.info("Language Rejected     : %d", len(rejected_docs))
        logger.info("Unique Documents      : %d", len(unique_docs))
        logger.info("Duplicates Removed    : %d", len(duplicate_docs))
        logger.info("Parquet Records       : %d", records_written)
        logger.info("=" * 60)

        logger.info("Pipeline completed successfully.")

    except FileNotFoundError as exc:
        logger.error("Dataset file not found: %s", exc)

    except Exception:
        logger.exception("Pipeline execution failed.")
        raise


if __name__ == "__main__":
    main()