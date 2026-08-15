from src.deduplication.shingler import Shingler
from src.deduplication.similarity import JaccardSimilarity
from src.deduplication.minhash import MinHash


document_a = """
Python is an amazing programming language used
for data science and artificial intelligence.
"""

document_b = """
Python is a powerful programming language used
for data science and artificial intelligence.
"""


# --------------------------------------------------
# Generate shingles
# --------------------------------------------------

shingler = Shingler(size=3)

shingles_a = shingler.generate(document_a)
shingles_b = shingler.generate(document_b)


# --------------------------------------------------
# Exact Jaccard
# --------------------------------------------------

jaccard = JaccardSimilarity.calculate(
    shingles_a,
    shingles_b,
)


# --------------------------------------------------
# MinHash
# --------------------------------------------------

minhash = MinHash(num_hashes=100)

signature_a = minhash.signature(shingles_a)
signature_b = minhash.signature(shingles_b)


# --------------------------------------------------
# Estimated Jaccard
# --------------------------------------------------

estimated_jaccard = minhash.similarity(
    shingles_a,
    shingles_b,
)


print("=" * 60)
print("NEAR-DUPLICATE ANALYSIS")
print("=" * 60)

print(f"Shingles A       : {len(shingles_a)}")
print(f"Shingles B       : {len(shingles_b)}")

print(f"Exact Jaccard    : {jaccard:.4f}")

print(
    f"MinHash Estimate : {estimated_jaccard:.4f}"
)

print("=" * 60)