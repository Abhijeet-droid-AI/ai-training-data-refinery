import pandas as pd

class PandasEngine:
    @staticmethod
    def load(filepath):
        return pd.read_json(filepath)