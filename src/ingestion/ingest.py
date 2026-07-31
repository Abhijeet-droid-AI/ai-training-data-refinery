import logging
import yaml

from loader import DataLoader
from validator import DataValidator


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)


def load_config():
    with open("configs/config.yaml", "r") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()

    loader = DataLoader(config["data"]["raw_path"])

    documents = loader.load()

    validator = DataValidator()

    valid_docs = validator.validate(documents)

    logging.info(f"Loaded {len(valid_docs)} valid documents")


if __name__ == "__main__":
    main()