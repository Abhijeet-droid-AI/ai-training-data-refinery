"""Rule-based quality filter.

Filters documents by threshold rules on the metadata produced by
the quality scorer. Rejects the documents that cannot contribute
to training data, while keeping a lineage record of every
decision in the returned rejected list.
"""

from __future__ import annotations

from src.quality.scorer import QualityScorer


class QualityFilter:
    """Applies scorer-derived thresholds to accept or reject documents."""

    def __init__(
        self,
        min_quality_score: int = 70,
        max_punctuation_ratio: float = 0.30,
        max_digit_ratio: float = 0.50,
        min_unique_word_ratio: float = 0.40,
        min_text_length: int = 20,
    ):
        self.min_quality_score = min_quality_score
        self.max_punctuation_ratio = max_punctuation_ratio
        self.max_digit_ratio = max_digit_ratio
        self.min_unique_word_ratio = min_unique_word_ratio
        self.min_text_length = min_text_length

    def filter(self, documents):
        """
        Split documents into accepted and rejected.

        Returns:
            (accepted, rejected) where each rejected entry is a
            dict containing the document and a list of rejection
            reasons.
        """
        accepted = []
        rejected = []

        for document in documents:
            text = document.get("text") or ""
            metadata = document.get("metadata", {}) or {}

            reasons = []

            if len(text) < self.min_text_length:
                reasons.append(
                    f"too_short:{len(text)}<{self.min_text_length}"
                )

            score = metadata.get("quality_score")
            if score is not None and score < self.min_quality_score:
                reasons.append(
                    f"quality_score:{score}<{self.min_quality_score}"
                )

            punctuation_ratio = metadata.get("punctuation_ratio")
            if (
                punctuation_ratio is not None
                and punctuation_ratio > self.max_punctuation_ratio
            ):
                reasons.append(
                    f"punctuation_ratio:{punctuation_ratio:.3f}"
                    f">{self.max_punctuation_ratio:.3f}"
                )

            digit_ratio = metadata.get("digit_ratio")
            if (
                digit_ratio is not None
                and digit_ratio > self.max_digit_ratio
            ):
                reasons.append(
                    f"digit_ratio:{digit_ratio:.3f}>{self.max_digit_ratio:.3f}"
                )

            unique_word_ratio = metadata.get("unique_word_ratio")
            if (
                unique_word_ratio is not None
                and unique_word_ratio < self.min_unique_word_ratio
            ):
                reasons.append(
                    f"unique_word_ratio:{unique_word_ratio:.3f}"
                    f"<{self.min_unique_word_ratio:.3f}"
                )

            if reasons:
                document.setdefault("metadata", {})
                document["metadata"]["rejection_reasons"] = reasons
                rejected.append(
                    {
                        "document": document,
                        "reasons": reasons,
                    }
                )
            else:
                accepted.append(document)

        return accepted, rejected
