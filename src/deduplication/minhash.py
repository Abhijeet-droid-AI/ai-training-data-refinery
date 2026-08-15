import hashlib

from src.deduplication.minhash_similarity import (
    MinHashSimilarity,
)


class MinHash:
    """
    Generates MinHash signatures for sets of shingles.
    """

    def __init__(self, num_hashes: int = 100):

        if num_hashes < 1:
            raise ValueError(
                "num_hashes must be at least 1."
            )

        self.num_hashes = num_hashes

    def _hash(
        self,
        shingle: str,
        seed: int,
    ) -> int:
        """
        Generate a deterministic hash for a shingle and seed.
        """

        value = f"{seed}:{shingle}".encode("utf-8")

        digest = hashlib.sha256(value).hexdigest()

        return int(digest, 16)

    def signature(
        self,
        shingles: set[str],
    ) -> list[int]:
        """
        Generate the MinHash signature for a set of shingles.
        """

        if not shingles:
            return [0] * self.num_hashes

        signature = []

        for seed in range(self.num_hashes):

            minimum_hash = min(
                self._hash(shingle, seed)
                for shingle in shingles
            )

            signature.append(minimum_hash)

        return signature

    def similarity(
        self,
        shingles_a: set[str],
        shingles_b: set[str],
    ) -> float:
        """
        Estimate Jaccard similarity between two shingle sets
        using MinHash signatures.
        """

        signature_a = self.signature(shingles_a)

        signature_b = self.signature(shingles_b)

        return MinHashSimilarity.estimate(
            signature_a,
            signature_b,
        )