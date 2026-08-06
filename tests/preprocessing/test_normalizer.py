from src.preprocessing.normalizer import TextNormalizer


def test_unicode_normalization():
    normalizer = TextNormalizer()

    text = "CafÃ©"

    normalized_text, metadata = normalizer.normalize(text)

    assert normalized_text == "Café"

    assert metadata["unicode_normalized"] is True