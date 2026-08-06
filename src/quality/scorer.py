from src.quality.heuristics import QualityHeuristics


class QualityScorer:

    def score(self, text):

        score = 100

        metadata = {}

        length = QualityHeuristics.text_length(text)

        metadata["length"] = length

        if length < 30:
            score -= 40

        if length > 10000:
            score -= 10

        punctuation = QualityHeuristics.punctuation_ratio(text)

        metadata["punctuation_ratio"] = round(punctuation, 3)

        if punctuation > 0.30:
            score -= 20

        digits = QualityHeuristics.digit_ratio(text)

        metadata["digit_ratio"] = round(digits, 3)

        if digits > 0.50:
            score -= 25

        uniqueness = QualityHeuristics.unique_word_ratio(text)

        metadata["unique_word_ratio"] = round(uniqueness, 3)

        if uniqueness < 0.40:
            score -= 15

        score = max(score, 0)

        metadata["quality_score"] = score

        metadata["quality_grade"] = self.grade(score)

        metadata["accepted"] = score >= 70

        return metadata

    @staticmethod
    def grade(score):

        if score >= 90:
            return "A"

        if score >= 80:
            return "B"

        if score >= 70:
            return "C"

        if score >= 60:
            return "D"

        return "F"