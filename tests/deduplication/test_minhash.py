from src.deduplication.minhash import MinHash


def test_signature_length():

    minhash = MinHash(num_hashes=10)

    shingles = {
        "python is",
        "is an",
        "an amazing",
    }

    signature = minhash.signature(shingles)

    assert len(signature) == 10


def test_signature_is_deterministic():

    minhash = MinHash(num_hashes=10)

    shingles = {
        "python is",
        "is an",
        "an amazing",
    }

    signature_1 = minhash.signature(shingles)

    signature_2 = minhash.signature(shingles)

    assert signature_1 == signature_2

def test_different_sets_produce_signatures():

    minhash = MinHash(num_hashes=20)

    shingles_a = {
        "python is",
        "is an",
        "an amazing",
    }

    shingles_b = {
        "docker is",
        "is a",
        "a container",
    }

    signature_a = minhash.signature(shingles_a)

    signature_b = minhash.signature(shingles_b)

    assert signature_a != signature_b

def test_similarity_identical_sets():

    minhash = MinHash(num_hashes=100)

    shingles = {
        "python is",
        "is an",
        "an amazing",
    }

    similarity = minhash.similarity(
        shingles,
        shingles,
    )

    assert similarity == 1.0

def test_similarity_different_sets():

    minhash = MinHash(num_hashes=100)

    shingles_a = {
        "python is",
        "is an",
        "an amazing",
    }

    shingles_b = {
        "docker is",
        "is a",
        "a container",
    }

    similarity = minhash.similarity(
        shingles_a,
        shingles_b,
    )

    assert similarity < 0.5

def test_similarity_partial_overlap():

    minhash = MinHash(num_hashes=500)

    shingles_a = {
        "python is",
        "is an",
        "an amazing",
        "amazing language",
    }

    shingles_b = {
        "python is",
        "is an",
        "an amazing",
        "amazing programming",
    }

    similarity = minhash.similarity(
        shingles_a,
        shingles_b,
    )

    # Exact Jaccard = 3 / 5 = 0.6.
    # MinHash should produce an estimate reasonably
    # close to that value.
    assert 0.45 <= similarity <= 0.75

def test_similarity_empty_sets():

    minhash = MinHash(num_hashes=100)

    similarity = minhash.similarity(
        set(),
        set(),
    )

    assert similarity == 1.0