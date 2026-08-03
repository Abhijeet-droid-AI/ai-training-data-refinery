import logging
import yaml

from loader import DataLoader
from validator import DataValidator
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()

    loader = DataLoader(config["data"]["raw_path"])

    documents = loader.load()

    validator = DataValidator()

    valid_docs, invalid_docs = validator.validate(documents)

    logging.info(f"Invalid documents        : {len(invalid_docs)}")

    if invalid_docs:
        logging.warning("Invalid document details:")

        for invalid in invalid_docs:
            logging.warning(
                f"Document ID: {invalid['document'].get('id', 'Unknown')} "
                f"| Missing Fields: {invalid['missing_fields']}"
            )

if __name__ == "__main__":
    main()