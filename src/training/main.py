"""Training experiment entry point.

Runs the mini character-level GPT on the refined Parquet corpus
and writes checkpoints plus a training report.

Usage:
    python -m src.training.main
"""

from src.training.train import main

if __name__ == "__main__":
    main()
