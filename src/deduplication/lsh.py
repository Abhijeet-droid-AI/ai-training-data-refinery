import hashlib

from itertools import combinations


class LSH:
    """
    Bands MinHash signatures to generate candidate
    near-duplicate pairs without O(n^2) comparison.
    """

    def __init__(
        self,
        num_bands: int = 20,
        rows_per_band: int = 5,
    ):

        if num_bands < 1:
            raise ValueError(
                "num_bands must be at least 1."
            )

        if rows_per_band < 1:
            raise ValueError(
                "rows_per_band must be at least 1."
            )

        self.num_bands = num_bands
        self.rows_per_band = rows_per_band

        self.signature_length = num_bands * rows_per_band

        self._buckets: dict[int, dict[str, set[str]]] = {
            band: {}
            for band in range(num_bands)
        }

    def _band_key(
        self,
        signature: list[int],
        band: int,
    ) -> str:
        """
        Generate a deterministic bucket key for one band.
        """

        start = band * self.rows_per_band

        end = start + self.rows_per_band

        rows = signature[start:end]

        value = ",".join(
            str(hash_value)
            for hash_value in rows
        ).encode("utf-8")

        return hashlib.sha256(value).hexdigest()

    def add(
        self,
        document_id: str,
        signature: list[int],
    ) -> None:
        """
        Index a document signature into the band buckets.
        """

        if len(signature) != self.signature_length:
            raise ValueError(
                "Signature length must equal "
                "num_bands * rows_per_band."
            )

        for band in range(self.num_bands):

            band_key = self._band_key(
                signature,
                band,
            )

            bucket = self._buckets[band]

            bucket.setdefault(band_key, set()).add(
                document_id
            )

    def reset(self) -> None:
        """
        Clear all indexed documents from the buckets.
        """

        self._buckets = {
            band: {}
            for band in range(self.num_bands)
        }

    def candidate_pairs(self) -> set[tuple[str, str]]:
        """
        Return all candidate pairs that share at least
        one band bucket.
        """

        pairs: set[tuple[str, str]] = set()

        for bucket in self._buckets.values():

            for bucket_ids in bucket.values():

                if len(bucket_ids) < 2:
                    continue

                for pair in combinations(
                    sorted(bucket_ids),
                    2,
                ):

                    pairs.add(pair)

        return pairs

    @property
    def similarity_threshold(self) -> float:
        """
        Approximate Jaccard similarity at which a pair of
        documents has a 50% chance of becoming a candidate.
        """

        return (1.0 / self.num_bands) ** (
            1.0 / self.rows_per_band
        )
