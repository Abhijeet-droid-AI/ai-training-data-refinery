from src.preprocessing.cleaner import TextCleaner


def test_html_removal():
    cleaner = TextCleaner()

    text = "<div>Hello <b>World</b></div>"

    cleaned_text, metadata = cleaner.clean(text)

    assert cleaned_text == "Hello World"

    assert metadata["html_removed"] is True