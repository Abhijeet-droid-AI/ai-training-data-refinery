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

## License

MIT
