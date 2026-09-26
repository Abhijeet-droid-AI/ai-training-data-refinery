from src.deduplication.shingler import Shingler
from src.deduplication.minhash import MinHash
from src.deduplication.lsh import LSH


document_a = """
Python is an amazing programming language used
for data science and artificial intelligence.
"""


document_b = """
Python is an amazing programming language used
for data science and artificial intelligence.
"""


document_c = """
The docker container orchestrates Kubernetes
clusters across multiple cloud environments.
"""


# --------------------------------------------------
# Generate shingles
# --------------------------------------------------

shingler = Shingler(size=3)

shingles_a = shingler.generate(document_a)
shingles_b = shingler.generate(document_b)
shingles_c = shingler.generate(document_c)


# --------------------------------------------------
# Generate MinHash signatures
# --------------------------------------------------

minhash = MinHash(num_hashes=100)

signature_a = minhash.signature(shingles_a)
signature_b = minhash.signature(shingles_b)
signature_c = minhash.signature(shingles_c)


# --------------------------------------------------
# Index signatures with LSH banding
# --------------------------------------------------

lsh = LSH(num_bands=20, rows_per_band=5)

lsh.add("document_a", signature_a)
lsh.add("document_b", signature_b)
lsh.add("document_c", signature_c)


print("=" * 60)
print("LSH NEAR-DUPLICATE CANDIDATE ANALYSIS")
print("=" * 60)

print(f"Signature length  : {lsh.signature_length}")
print(
    f"Similarity thresh : {lsh.similarity_threshold:.4f}"
)

print("\nCandidate pairs:")
for pair in sorted(lsh.candidate_pairs()):
    print("  ", pair)

print("=" * 60)
