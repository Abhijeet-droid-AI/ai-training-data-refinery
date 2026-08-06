from src.ingestion.loader import DataLoader
from src.utils.paths import RAW_DATA_DIR


def test_loader():

    loader = DataLoader(
        RAW_DATA_DIR / "sample_documents.json"
    )

    docs = loader.load()

    assert len(docs) == 3