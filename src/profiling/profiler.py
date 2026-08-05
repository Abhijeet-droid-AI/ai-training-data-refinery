import pandas as pd

class DatasetProfiler:
    def __init__(self, documents):
        self.df = pd.DataFrame(documents)

    def profile(self):
        self.df['text_length'] = self.df['text'].str.len()
        self.df['word_count'] = self.df['text'].str.split().str.len()

        report = {
            "total_documents": len(self.df),
            "average_text_length": round(self.df["text_length"].mean(), 2),
            "minimum_text_length": int(self.df["text_length"].min()),
            "maximum_text_length": int(self.df["text_length"].max()),
            "average_word_count": round(self.df["word_count"].mean(), 2),
            "missing_titles": int(self.df["title"].isna().sum()),
            "empty_documents": int((self.df["text"] == "").sum()),
        }

        return report