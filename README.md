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
- [ ] Data Ingestion
- [ ] Data Cleaning
- [ ] Quality Scoring
- [ ] Deduplication
- [ ] Dashboard
- [ ] LLM Experiment

---

## Current Progress

### ✅ Phase 1

- Repository initialized
- Development environment configured

### 🚧 Phase 2

- Basic data ingestion pipeline
- JSON loader
- Document validator
- Configuration management

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
- Deduplication report generation

Upcoming:

- MinHash
- Locality Sensitive Hashing (LSH)
- Near-duplicate detection

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

### Current Limitation

MinHash signatures alone do not solve the O(n²) comparison problem.

Locality Sensitive Hashing (LSH) will be introduced to efficiently
generate candidate near-duplicate pairs.

## License

MIT
