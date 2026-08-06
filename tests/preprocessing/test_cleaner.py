from src.preprocessing.cleaner import TextCleaner


def test_html_removal():

    cleaner = TextCleaner()

    text = "<div>Hello <b>World</b></div>"

    assert cleaner.clean(text) == "Hello World"