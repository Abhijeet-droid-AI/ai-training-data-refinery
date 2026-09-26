# AI Training Data Refinery

> Production-grade data engineering pipeline for transforming raw web-scale text into high-quality datasets for LLM training.

---

## Features

- Data ingestion
- Text preprocessing
- Language detection
- Quality scoring
- Near-duplicate detection
- Anomaly detection
- Dashboard
- Mini LLM training

---

## Tech Stack

- Python
- Polars
- PyArrow
- DuckDB
- Hugging Face
- Docker
- Kubernetes
- Streamlit

---

## Architecture

(Project diagram coming soon)

---

## Repository Structure

(Add folder tree)

---

## Roadmap

- [x] Project Initialization
- [x] Data Ingestion
- [x] Dataset Profiling
- [x] Parquet Storage
- [x] Benchmarking (Pandas vs Polars)
- [x] Analytics Layer (DuckDB)
- [x] Data Cleaning
- [x] Quality Scoring
- [x] Deduplication (Exact + Near-Duplicate)
- [x] Anomaly Detection
- [ ] Dashboard
- [ ] LLM Experiment

---

## Current Progress

### ✅ Phase 1 — Foundation

- Repository initialized
- Development environment configured
- Project architecture and documentation structure

### ✅ Phase 2 — Core Pipeline

- Data ingestion with validation and logging
- Dataset profiling and report generation
- Parquet storage layer (PyArrow)
- Benchmarking framework (Pandas vs Polars)
- DuckDB analytics layer with JSON reports
- Text preprocessing (cleaning and normalization)
- Language detection and filtering
- Rule-based quality scoring (grades A–F)

### ✅ Phase 3 — Deduplication

- Exact duplicate detection with SHA256 fingerprints
- Document lineage and duplicate tracking
- MinHash signatures with Jaccard similarity estimation
- LSH banding for candidate pair generation
- Near-duplicate detection with Jaccard verification

### ✅ Phase 4 — Anomaly Detection

- Rule-based anomaly heuristics
- Flag-and-keep detection (no silent data loss)
- Per-document anomaly reasons in metadata
- Anomaly summary report with reason counts

### 🚧 Phase 5 — Upcoming

- Dashboard
- Mini LLM training experiment

## Pipeline

Current pipeline

Raw JSON

↓

Loader

↓

Validator

↓

Logging

↓

## Ready for preprocessing

## Current Architecture

Raw JSON

↓

Loader

↓

Validator

↓

Logger

↓

Future Preprocessing

## Development Tools

- Black
- Ruff
- Pytest

## Storage Layer

Current storage pipeline:

Raw JSON
│
▼
Validation
│
▼
Profiling
│
▼
Parquet Conversion
│
▼
Training Dataset

## Benchmarking

Current benchmark modules:

- Pandas
- Polars

Goal:

Measure execution time and memory usage for different processing engines.

## Analytics Layer

Current analytics stack:

Parquet

↓

DuckDB

↓

SQL Analytics

↓

JSON Reports

## NLP Preprocessing

Current preprocessing stages:

- HTML Removal
- Unicode Normalization
- Whitespace Cleanup
- Control Character Removal

## Language Processing

Current features:

- Language Detection
- Language Filtering
- Language Distribution Report

## Document Quality Engine

Current heuristics:

- Document Length
- Punctuation Ratio
- Digit Ratio
- Vocabulary Diversity
- Quality Grade (A–F)

## Deduplication

Current capabilities:

- SHA256 fingerprint generation
- Exact duplicate detection
- Near-duplicate detection (MinHash + LSH)
- Jaccard verification of LSH candidate pairs
- Deduplication report generation

### Data Lineage

Each document receives a deterministic SHA256 fingerprint during
deduplication.

Unique documents contain:

- `fingerprint`
- `is_duplicate`
- `duplicate_group`

Rejected duplicate documents retain:

- `fingerprint`
- `duplicate_of`
- `reason`

Near-duplicate documents additionally contain:

- `is_near_duplicate`
- `near_duplicate_of`
- `near_duplicate_similarity`

This allows duplicate decisions to be traced back to the
original document.

## Near-Duplicate Detection

### Shingling

Documents are converted into word-based shingles to represent
local text patterns.

### Jaccard Similarity

Jaccard similarity measures the overlap between two sets of shingles:

J(A,B) = |A ∩ B| / |A ∪ B|

### MinHash

MinHash produces compact signatures that approximately preserve
Jaccard similarity.

Instead of comparing potentially thousands of shingles directly,
documents can be represented using a fixed-size MinHash signature.

The current implementation supports configurable numbers of
hash functions.

### Locality Sensitive Hashing (LSH)

MinHash signatures alone require comparing every document against
every other document, which grows as O(n²).

LSH banding solves this by splitting each signature into bands.
Documents that share an identical band are grouped into the same
bucket and become candidate near-duplicate pairs.

Current configuration:

- 20 bands × 5 rows per band = 100 hash functions
- Band keys are deterministic SHA256 digests
- Similar documents share at least one band bucket

The approximate similarity threshold for a 50% chance of becoming
a candidate is:

T = (1 / bands)^(1 / rows_per_band)

With 20 bands and 5 rows per band this is roughly 0.55, so documents
with an estimated Jaccard similarity of about 0.55 or higher are
likely to be surfaced as candidates.

## Anomaly Detection

The anomaly detector flags suspicious documents using
rule-based heuristics instead of removing them.

Flagged documents stay in the dataset and carry metadata:

- `anomaly`
- `anomaly_reasons`

This keeps every filtering decision auditable and
reversible.

### Current heuristics

- Minimum document length
- Character repetition ratio
- Word repetition ratio
- Symbol ratio
- Gibberish detection (vowel-less or tiny-alphabet words)

Each document records the list of triggered reasons,
for example:

`["too_short:5<20", "word_repetition:0.85"]`

## License

MIT
