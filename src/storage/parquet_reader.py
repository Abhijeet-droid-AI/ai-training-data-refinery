import pyarrow.parquet as pq


class ParquetReader:
    """Reads Parquet files."""

    def read(self, filepath):
        table = pq.read_table(filepath)

        return table.to_pandas()