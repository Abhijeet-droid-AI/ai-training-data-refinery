import ftfy
import unicodedata
import regex


class TextNormalizer:
    """
    Normalizes Unicode text.
    """

    def normalize(self, text: str) -> tuple[str, dict]:

        metadata = {
            "unicode_normalized": False,
            "control_characters_removed": False,
        }

        original = text

        text = ftfy.fix_text(text)

        text = unicodedata.normalize("NFKC", text)

        if text != original:
            metadata["unicode_normalized"] = True

        cleaned = regex.sub(r"\p{C}+", "", text)

        if cleaned != text:
            metadata["control_characters_removed"] = True

        return cleaned, metadata