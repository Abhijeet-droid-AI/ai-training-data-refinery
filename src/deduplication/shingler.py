class Shingler:
    """
    Generates word-based shingles from text.
    """

    def __init__(self, size: int = 3):
        if size < 1:
            raise ValueError("Shingle size must be at least 1.")

        self.size = size

    def generate(self, text: str) -> set[str]:
        words = text.lower().split()

        if len(words) < self.size:
            return set()

        return {
            " ".join(words[index:index + self.size])
            for index in range(len(words) - self.size + 1)
        }