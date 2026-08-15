class MinHashSimilarity:
    """
    Estimates Jaccard similarity using MinHash signatures.
    """

    @staticmethod
    def estimate(
        signature_a: list[int],
        signature_b: list[int],
    ) -> float:

        if len(signature_a) != len(signature_b):
            raise ValueError(
                "MinHash signatures must have equal length."
            )

        if not signature_a:
            return 1.0

        matches = sum(
            a == b
            for a, b in zip(
                signature_a,
                signature_b,
            )
        )

        return matches / len(signature_a)