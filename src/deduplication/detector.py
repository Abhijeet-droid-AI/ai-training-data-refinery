from src.deduplication.fingerprint import FingerprintGenerator
from src.deduplication.lsh import LSH
from src.deduplication.minhash import MinHash
from src.deduplication.shingler import Shingler
from src.deduplication.similarity import JaccardSimilarity


class DuplicateDetector:
    """
    Detects exact duplicate documents using SHA256
    fingerprints and near-duplicate documents using
    MinHash signatures with LSH banding.
    """

    def __init__(
        self,
        shingle_size: int = 3,
        num_hashes: int = 100,
        num_bands: int = 20,
        rows_per_band: int = 5,
        near_duplicate_threshold: float = 0.8,
    ):

        self.shingler = Shingler(size=shingle_size)

        self.minhash = MinHash(num_hashes=num_hashes)

        self.lsh = LSH(
            num_bands=num_bands,
            rows_per_band=rows_per_band,
        )

        self.near_duplicate_threshold = (
            near_duplicate_threshold
        )

        self.near_duplicates = []

        self._documents_by_id = {}

    def detect(self, documents):

        seen = {}

        unique_documents = []
        duplicate_documents = []

        for document in documents:

            document.setdefault("metadata", {})

            fingerprint = FingerprintGenerator.generate(
                document["text"]
            )

            duplicate_group = f"sha256:{fingerprint}"

            # --------------------------------------------------
            # Duplicate Document
            # --------------------------------------------------
            if fingerprint in seen:

                original_document = seen[fingerprint]

                document["metadata"]["fingerprint"] = fingerprint
                document["metadata"]["is_duplicate"] = True
                document["metadata"]["duplicate_group"] = (
                    duplicate_group
                )

                duplicate_documents.append(
                    {
                        "document": document,
                        "reason": "duplicate",
                        "fingerprint": fingerprint,
                        "duplicate_of": original_document["id"],
                    }
                )

            # --------------------------------------------------
            # Unique Document
            # --------------------------------------------------
            else:

                document["metadata"]["fingerprint"] = fingerprint
                document["metadata"]["is_duplicate"] = False
                document["metadata"]["duplicate_group"] = (
                    duplicate_group
                )

                seen[fingerprint] = document

                unique_documents.append(document)

        self._detect_near_duplicates(unique_documents)

        return unique_documents, duplicate_documents

    def _detect_near_duplicates(self, unique_documents):
        """
        Index unique document signatures into LSH bands,
        collect candidates, and verify them with exact
        Jaccard similarity.
        """

        self.near_duplicates = []

        self._documents_by_id = {}

        self.lsh.reset()

        if len(unique_documents) < 2:
            return

        shingle_cache = {}

        for document in unique_documents:

            self._documents_by_id[document["id"]] = document

            shingle_cache[document["id"]] = self.shingler.generate(
                document["text"]
            )

        for document in unique_documents:

            signature = self.minhash.signature(
                shingle_cache[document["id"]]
            )

            candidate_originals = self._candidates_for_document(
                document,
                signature,
            )

            # --------------------------------------------------
            # Jaccard Verification
            # --------------------------------------------------
            for original in candidate_originals:

                similarity = JaccardSimilarity.calculate(
                    shingle_cache[document["id"]],
                    shingle_cache[original["id"]],
                )

                if similarity < self.near_duplicate_threshold:
                    continue

                if document["metadata"].get(
                    "is_near_duplicate"
                ):
                    continue

                self._mark_near_duplicate(
                    document,
                    original,
                    similarity,
                )

    def _candidates_for_document(
        self,
        document,
        signature,
    ):
        """
        Return already-indexed documents sharing a band,
        then index the current document.
        """

        candidate_originals = []

        for band in range(self.lsh.num_bands):

            band_key = self.lsh._band_key(
                signature,
                band,
            )

            bucket = self.lsh._buckets[band]

            # --------------------------------------------------
            # Existing Bucket Members
            # --------------------------------------------------
            for member_id in bucket.get(band_key, set()):

                candidate_originals.append(
                    self._documents_by_id[member_id]
                )

            # --------------------------------------------------
            # Index This Document
            # --------------------------------------------------
            bucket.setdefault(band_key, set()).add(
                document["id"]
            )

        return candidate_originals

    def _mark_near_duplicate(
        self,
        document,
        original,
        similarity,
    ):
        """
        Record near-duplicate lineage on the document.
        """

        document["metadata"]["is_near_duplicate"] = True
        document["metadata"]["near_duplicate_of"] = (
            original["id"]
        )
        document["metadata"]["near_duplicate_similarity"] = round(
            similarity,
            4,
        )

        self.near_duplicates.append(
            {
                "document": document,
                "reason": "near_duplicate",
                "duplicate_of": original["id"],
                "similarity": round(similarity, 4),
            }
        )