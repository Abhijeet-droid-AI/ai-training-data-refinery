import json


class DataLoader:
    """Loads raw documents from JSON files."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self):
        with open(self.filepath, "r", encoding="utf-8") as file:
            return json.load(file)