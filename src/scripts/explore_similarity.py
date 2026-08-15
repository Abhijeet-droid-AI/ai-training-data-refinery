from src.deduplication.shingler import Shingler
from src.deduplication.similarity import JaccardSimilarity


document_a = """
Python is an amazing programming language used
for data science and artificial intelligence.
"""


document_b = """
Python is a powerful programming language used
for data science and artificial intelligence.
"""


shingler = Shingler(size=4)

shingles_a = shingler.generate(document_a)

shingles_b = shingler.generate(document_b)

similarity = JaccardSimilarity.calculate(
    shingles_a,
    shingles_b,
)


print("Document A shingles:")
for shingle in shingles_a:
    print(" ", shingle)

print("\nDocument B shingles:")
for shingle in shingles_b:
    print(" ", shingle)

print("\nJaccard Similarity:", similarity)