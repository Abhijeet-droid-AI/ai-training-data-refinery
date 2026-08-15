class JaccardSimilarity:
    """
    Calculates Jaccard similarity between two sets.
    """

    @staticmethod
    def calculate(
        shingles_a: set[str],
        shingles_b: set[str],
    ) -> float:

        if not shingles_a and not shingles_b:
            return 1.0

        if not shingles_a or not shingles_b:
            return 0.0

        intersection = shingles_a.intersection(shingles_b)

        union = shingles_a.union(shingles_b)

        return len(intersection) / len(union)

    @staticmethod
    def is_similar(
        shingles_a: set[str],
        shingles_b: set[str],
        threshold: float = 0.8,
    ) -> bool:

        similarity = JaccardSimilarity.calculate(
            shingles_a,
            shingles_b,
        )

        return similarity >= threshold
    
    