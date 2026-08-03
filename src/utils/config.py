import yaml
from src.utils.config import CONFIG_DIR


def load_config():
    config_path = CONFIG_DIR / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
