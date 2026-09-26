from src.anomaly.heuristics import AnomalyHeuristics


class AnomalyDetector:
    """
    Flags anomalous documents using rule-based
    heuristics without removing them from the dataset.
    """

    def __init__(
        self,
        character_repetition_threshold: float = 0.30,
        word_repetition_threshold: float = 0.30,
        symbol_ratio_threshold: float = 0.30,
        min_length: int = 20,
    ):

        self.character_repetition_threshold = (
            character_repetition_threshold
        )

        self.word_repetition_threshold = (
            word_repetition_threshold
        )

        self.symbol_ratio_threshold = symbol_ratio_threshold

        self.min_length = min_length

    def detect(self, documents):

        accepted_documents = []
        flagged_documents = []

        for document in documents:

            document.setdefault("metadata", {})

            # --------------------------------------------------
            # Preprocessing Artifacts
            # --------------------------------------------------
            text = document.get("text") or ""

            document["metadata"]["text_length"] = len(text)

            reasons = []

            # --------------------------------------------------
            # Empty or Too Short
            # --------------------------------------------------
            if len(text) < self.min_length:

                reasons.append(
                    f"too_short:{len(text)}<{self.min_length}"
                )

            # --------------------------------------------------
            # Repetition and Symbol Heuristics
            # --------------------------------------------------
            character_repetition = (
                AnomalyHeuristics.character_repetition_ratio(text)
            )

            word_repetition = (
                AnomalyHeuristics.word_repetition_ratio(text)
            )

            symbol_ratio = AnomalyHeuristics.symbol_ratio(text)

            if (
                character_repetition
                > self.character_repetition_threshold
            ):
                reasons.append(
                    f"character_repetition:{character_repetition:.2f}"
                )

            if word_repetition > self.word_repetition_threshold:
                reasons.append(
                    f"word_repetition:{word_repetition:.2f}"
                )

            if symbol_ratio > self.symbol_ratio_threshold:
                reasons.append(f"symbols:{symbol_ratio:.2f}")

            # --------------------------------------------------
            # Gibberish Detection
            # >50% vowel-less or tiny-alphabet words
            # --------------------------------------------------
            if AnomalyHeuristics.is_gibberish(text):
                reasons.append("gibberish")

            # --------------------------------------------------
            # Flagged Document (kept, not removed)
            # --------------------------------------------------
            if reasons:

                document["metadata"]["anomaly"] = True
                document["metadata"]["anomaly_reasons"] = reasons

                flagged_documents.append(
                    {
                        "document": document,
                        "reasons": reasons,
                    }
                )

            # --------------------------------------------------
            # Clean Document
            # --------------------------------------------------
            else:

                document["metadata"]["anomaly"] = False

                accepted_documents.append(document)

        return accepted_documents, flagged_documents
