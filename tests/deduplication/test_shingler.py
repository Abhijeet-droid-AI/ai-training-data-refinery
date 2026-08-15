from src.deduplication.shingler import Shingler


def test_generate_three_word_shingles():

    shingler = Shingler(size=3)

    text = "Python is an amazing language"

    shingles = shingler.generate(text)

    assert shingles == {
        "python is an",
        "is an amazing",
        "an amazing language",
    }


def test_short_text_returns_empty_set():

    shingler = Shingler(size=3)

    shingles = shingler.generate("Python is")

    assert shingles == set()