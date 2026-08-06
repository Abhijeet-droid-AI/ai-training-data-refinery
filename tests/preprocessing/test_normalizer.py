from src.preprocessing.normalizer import TextNormalizer


def test_unicode_normalization():

    normalizer = TextNormalizer()

    text = "CafÃ©"

    assert "Café" == normalizer.normalize(text)