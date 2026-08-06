import re
from bs4 import BeautifulSoup


class TextCleaner:
    """
    Removes HTML and unnecessary whitespace.
    """

    def clean(self, text: str) -> tuple[str, dict]:
        metadata = {
            "html_removed": False,
            "whitespace_normalized": False,
        }

        original_text = text

        # Remove HTML
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")

        if text != original_text:
            metadata["html_removed"] = True

        # Normalize whitespace
        cleaned_text = re.sub(r"\s+", " ", text).strip()

        if cleaned_text != text:
            metadata["whitespace_normalized"] = True

        return cleaned_text, metadata