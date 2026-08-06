import ftfy
import unicodedata
import regex


class TextNormalizer:
    """
    Normalizes Unicode text.
    """

    def normalize(self, text: str) -> str:

        text = ftfy.fix_text(text)

        text = unicodedata.normalize("NFKC", text)

        text = regex.sub(r"\p{C}+", "", text)

        return text