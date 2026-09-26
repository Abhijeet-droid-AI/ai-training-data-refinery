import re
from collections import Counter


class AnomalyHeuristics:
    """
    Rule-based anomaly checks for text documents.
    """

    @staticmethod
    def character_repetition_ratio(text):

        if not text:
            return 0.0

        characters = list(text)

        return max(
            Counter(characters).values()
        ) / len(characters)

    @staticmethod
    def word_repetition_ratio(text):

        words = text.lower().split()

        if not words:
            return 0.0

        return max(
            Counter(words).values()
        ) / len(words)

    @staticmethod
    def symbol_ratio(text):

        if not text:
            return 0.0

        symbols = re.findall(r"[^\w\s]", text)

        return len(symbols) / len(text)

    @staticmethod
    def is_gibberish(text):

        words = re.findall(r"[a-zA-Z]+", text)

        if len(words) < 3:
            return False

        gibberish_words = [
            word
            for word in words
            if AnomalyHeuristics._vowel_ratio(word) == 0
            or len(set(word)) <= 2
        ]

        return (
            len(gibberish_words) / len(words) > 0.5
        )

    @staticmethod
    def _vowel_ratio(word):

        vowels = re.findall(r"[aeiou]", word.lower())

        return len(vowels) / max(len(word), 1)
