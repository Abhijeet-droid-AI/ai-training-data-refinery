from src.storage.parquet_converter import ParquetConverter
from src.storage.parquet_reader import ParquetReader
from src.utils.paths import PARQUET_DATA_DIR


def test_parquet_conversion(tmp_path):
    docs = [
        {
            "id": 1,
            "title": "AI",
            "text": "Artificial Intelligence"
        }
    ]

    output_file = tmp_path / "test.parquet"

    converter = ParquetConverter(docs)
    converter.convert(output_file)

    reader = ParquetReader()
    df = reader.read(output_file)

    assert len(df) == 1
    assert df.iloc[0]["title"] == "AI"