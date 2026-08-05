import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class ParquetConverter:
    """Converts JSON documents into Parquet format."""

    def __init__(self, documents):
        self.documents = documents

    def convert(self, output_path):
        df = pd.DataFrame(self.documents)

        table = pa.Table.from_pandas(df)

        pq.write_table(table, output_path)

        return output_path