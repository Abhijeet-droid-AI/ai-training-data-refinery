import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class ParquetConverter:
    """
    Converts a list of documents into a Parquet file.
    """

    def __init__(self, documents: list[dict]):
        self.documents = documents

    def convert(self, output_path: Path, overwrite: bool = True) -> int:
        """
        Convert documents to Parquet.

        Args:
            output_path: Destination Parquet file.
            overwrite: Whether to overwrite the output file if it already exists.

        Returns:
            Number of records written.
        """

        if output_path.exists() and not overwrite:
            logger.info("Parquet file already exists. Skipping conversion.")
            return 0

        if not self.documents:
            logger.warning("No documents provided for Parquet conversion.")
            return 0

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame
        df = pd.DataFrame(self.documents)

        # Convert to Arrow Table
        table = pa.Table.from_pandas(df)

        # Write Parquet
        pq.write_table(table, output_path)

        record_count = len(df)

        logger.info(
            "Successfully wrote %d records to %s",
            record_count,
            output_path,
        )

        return record_count