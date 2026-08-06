import re


class QualityHeuristics:

    @staticmethod
    def text_length(text):
        return len(text)

    @staticmethod
    def punctuation_ratio(text):

        punctuation = re.findall(r"[^\w\s]", text)

        return len(punctuation) / max(len(text), 1)

    @staticmethod
    def digit_ratio(text):

        digits = re.findall(r"\d", text)

        return len(digits) / max(len(text), 1)

    @staticmethod
    def unique_word_ratio(text):

        words = text.lower().split()

        if not words:
            return 0

        return len(set(words)) / len(words)