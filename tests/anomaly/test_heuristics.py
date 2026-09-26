import pytest

from src.anomaly.heuristics import AnomalyHeuristics


# --------------------------------------------------
# Character repetition
# --------------------------------------------------


def test_character_repetition_identical_characters():

    ratio = AnomalyHeuristics.character_repetition_ratio(
        "aaaaaaaaaa"
    )

    assert ratio == 1.0


def test_character_repetition_varied_text():

    ratio = AnomalyHeuristics.character_repetition_ratio(
        "abcdefghij"
    )

    assert ratio == 0.1


def test_character_repetition_empty_text():

    ratio = AnomalyHeuristics.character_repetition_ratio("")

    assert ratio == 0.0


# --------------------------------------------------
# Word repetition
# --------------------------------------------------


def test_word_repetition_identical_words():

    ratio = AnomalyHeuristics.word_repetition_ratio(
        "hello hello hello hello"
    )

    assert ratio == 1.0


def test_word_repetition_varied_words():

    ratio = AnomalyHeuristics.word_repetition_ratio(
        "the quick brown fox"
    )

    assert ratio == 0.25


def test_word_repetition_empty_text():

    ratio = AnomalyHeuristics.word_repetition_ratio("")

    assert ratio == 0.0


# --------------------------------------------------
# Symbol ratio
# --------------------------------------------------


def test_symbol_ratio_all_symbols():

    ratio = AnomalyHeuristics.symbol_ratio("!!!???")

    assert ratio == 1.0


def test_symbol_ratio_clean_text():

    ratio = AnomalyHeuristics.symbol_ratio(
        "Python is great"
    )

    assert ratio == 0.0


def test_symbol_ratio_empty_text():

    ratio = AnomalyHeuristics.symbol_ratio("")

    assert ratio == 0.0


# --------------------------------------------------
# Gibberish detection
# --------------------------------------------------


def test_gibberish_words_detected():

    assert (
        AnomalyHeuristics.is_gibberish(
            "xkcd vbnm qrtz zzzz plkw"
        )
        is True
    )


def test_normal_words_not_gibberish():

    assert (
        AnomalyHeuristics.is_gibberish(
            "Python is a great programming language"
        )
        is False
    )


def test_too_few_words_not_gibberish():

    assert AnomalyHeuristics.is_gibberish("xkcd vbnm") is False
