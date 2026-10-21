import time

import numpy as np
import polars as pl

from src.training.dataset import TokenStream
from src.training.model import MiniGPT
from src.training.optimizer import Adam
from src.training.report import TrainingReport
from src.training.sampler import Sampler
from src.training.tokenizer import CharTokenizer
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.paths import (
    MODEL_CHECKPOINT_FILE,
    PARQUET_FILE,
    TOKENIZER_FILE,
)


logger = get_logger(__name__)


TRAINING_DEFAULTS = {
    "seq_len": 16,
    "batch_size": 4,
    "steps": 300,
    "d_model": 32,
    "n_layers": 2,
    "n_heads": 2,
    "learning_rate": 1e-2,
    "clip_norm": 1.0,
    "seed": 42,
    "sample_prompts": 3,
    "sample_tokens": 60,
}


def load_corpus(path):

    frame = pl.read_parquet(path)

    texts = frame["text"].cast(pl.Utf8).fill_null("")
    texts = texts.filter(texts != "")

    documents = texts.to_list()

    if not documents:

        raise ValueError(
            "No usable text documents found in %s" % path
        )

    return documents


def ascii_safe(text, limit=80):

    encoded = text.encode("ascii", "replace").decode()

    return encoded[:limit]


def main() -> None:
    """
    Train a mini character-level GPT on the refined
    Parquet dataset to validate the pipeline output
    end-to-end.
    """

    try:
        # --------------------------------------------------
        # Configuration
        # --------------------------------------------------
        config = load_config()

        training_config = {
            **TRAINING_DEFAULTS,
            **config.get("training", {}),
        }

        # --------------------------------------------------
        # Corpus from Parquet
        # --------------------------------------------------
        logger.info("Loading training corpus: %s", PARQUET_FILE)

        documents = load_corpus(PARQUET_FILE)

        text = "\n\n".join(documents)

        logger.info(
            "Corpus | Documents: %d | Characters: %d",
            len(documents),
            len(text),
        )

        # --------------------------------------------------
        # Tokenization
        # --------------------------------------------------
        tokenizer = CharTokenizer.build(text)

        token_ids = tokenizer.encode(text)

        logger.info(
            "Vocabulary size: %d | Tokens: %d",
            tokenizer.vocab_size,
            len(token_ids),
        )

        data = TokenStream(token_ids)

        # --------------------------------------------------
        # Model and Optimizer
        # --------------------------------------------------
        seq_len = training_config["seq_len"]

        model = MiniGPT(
            vocab_size=tokenizer.vocab_size,
            seq_len=seq_len,
            d_model=training_config["d_model"],
            n_layers=training_config["n_layers"],
            n_heads=training_config["n_heads"],
            seed=training_config["seed"],
        )

        logger.info(
            "MiniGPT | Parameters: %d | Layers: %d | d_model: %d",
            model.param_count,
            model.n_layers,
            model.d_model,
        )

        rng = np.random.default_rng(training_config["seed"])

        optimizer = Adam(
            model.params,
            learning_rate=training_config["learning_rate"],
        )

        trainer = Trainer(
            model,
            data,
            optimizer,
            rng,
            clip_norm=training_config["clip_norm"],
        )

        # --------------------------------------------------
        # Training
        # --------------------------------------------------
        logger.info("Starting training...")

        start = time.perf_counter()

        losses = trainer.train(
            steps=training_config["steps"],
            batch_size=training_config["batch_size"],
            seq_len=seq_len,
        )

        wall_time = time.perf_counter() - start

        logger.info(
            "Training finished | Final loss: %.4f | %.1fs",
            losses[-1],
            wall_time,
        )

        # --------------------------------------------------
        # Text Generation Samples
        # --------------------------------------------------
        sampler = Sampler(model, tokenizer)

        sample_rng = np.random.default_rng(
            training_config["seed"]
        )

        samples = []

        prompt_source = documents[0]

        for index in range(training_config["sample_prompts"]):

            prompt = prompt_source[: 3 + index]

            generated = sampler.generate(
                prompt,
                max_new_tokens=training_config["sample_tokens"],
                temperature=0.8,
                rng=sample_rng,
            )

            samples.append(
                {
                    "prompt": prompt,
                    "generated": generated,
                }
            )

            logger.info(
                "Sample %d | %s...",
                index + 1,
                ascii_safe(generated),
            )

        # --------------------------------------------------
        # Checkpoints
        # --------------------------------------------------
        model.save(MODEL_CHECKPOINT_FILE)

        tokenizer.save(TOKENIZER_FILE)

        logger.info(
            "Checkpoints saved to %s",
            MODEL_CHECKPOINT_FILE.parent,
        )

        # --------------------------------------------------
        # Report
        # --------------------------------------------------
        corpus_stats = {
            "source": str(PARQUET_FILE),
            "documents": len(documents),
            "characters": len(text),
            "tokens": len(token_ids),
            "vocab_size": tokenizer.vocab_size,
        }

        model_config = {
            "d_model": model.d_model,
            "n_layers": model.n_layers,
            "n_heads": model.n_heads,
            "d_ff": model.d_ff,
            "seq_len": model.seq_len,
            "param_count": model.param_count,
        }

        run_config = {
            key: value
            for key, value in training_config.items()
        }

        report = TrainingReport().generate(
            corpus_stats,
            model_config,
            run_config,
            losses,
            samples,
            wall_time_s=round(wall_time, 3),
        )

        logger.info(
            "Training report generated | Final loss: %.4f",
            report["final_loss"],
        )

        logger.info(
            "Experiment completed successfully."
        )

    except Exception:

        logger.exception("Training experiment failed.")

        raise


if __name__ == "__main__":

    main()
