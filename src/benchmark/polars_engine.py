import polars as pl

class PolarsEngine:
    @staticmethod
    def load(filepath):
        return pl.read_json(filepath)