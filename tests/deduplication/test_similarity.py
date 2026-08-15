from src.deduplication.similarity import JaccardSimilarity


def test_identical_sets():

    a = {"python", "java", "sql"}

    b = {"python", "java", "sql"}

    similarity = JaccardSimilarity.calculate(a, b)

    assert similarity == 1.0


def test_completely_different_sets():

    a = {"python", "java"}

    b = {"docker", "kubernetes"}

    similarity = JaccardSimilarity.calculate(a, b)

    assert similarity == 0.0


def test_partial_overlap():

    a = {"python", "java", "sql"}

    b = {"python", "java", "docker"}

    similarity = JaccardSimilarity.calculate(a, b)

    assert similarity == 0.5

# Edge cases

def test_both_empty_sets():

    similarity = JaccardSimilarity.calculate(
        set(),
        set(),
    )

    assert similarity == 1.0


def test_one_empty_set():

    similarity = JaccardSimilarity.calculate(
        {"python"},
        set(),
    )

    assert similarity == 0.0