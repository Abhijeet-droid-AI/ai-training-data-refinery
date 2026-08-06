import re

from bs4 import BeautifulSoup


class TextCleaner:
    """
    Removes HTML and unnecessary whitespace.
    """

    def clean(self, text: str) -> str:

        soup = BeautifulSoup(text, "html.parser")

        text = soup.get_text(separator=" ")

        text = re.sub(r"\s+", " ", text)

        return text.strip()