import numpy as np
import polars as pl
import pytest

from src.training.model import MiniGPT
from src.training.dataset import TokenStream
from src.training.tokenizer import CharTokenizer
from src.training.trainer import Trainer
from src.training.optimizer import Adam


@pytest.fixture()
def corpus_docs():
    """Load the training corpus directly from JSON (multi-doc)."""
    path = "data/raw/training_corpus.json"
    frame = pl.read_json(path)
    texts = frame["text"].cast(pl.Utf8)
    texts = texts.filter(texts != "")
    return texts.to_list()


@pytest.fixture()
def training_config():
    return {
        "seq_len": 8,
        "batch_size": 2,
        "steps": 60,
        "d_model": 16,
        "n_layers": 1,
        "n_heads": 2,
        "learning_rate": 1e-2,
        "clip_norm": 1.0,
        "seed": 42,
        "sample_prompts": 1,
        "sample_tokens": 20,
    }


def _build_components(corpus, config):
    text = "\n\n".join(corpus)
    tokenizer = CharTokenizer.build(text)
    token_ids = tokenizer.encode(text)

    rng = np.random.default_rng(config["seed"])
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        seq_len=config["seq_len"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        seed=config["seed"],
    )
    optimizer = Adam(
        model.params,
        learning_rate=config["learning_rate"],
    )
    data = TokenStream(token_ids)
    trainer = Trainer(
        model,
        data,
        optimizer,
        rng,
        clip_norm=config["clip_norm"],
    )
    return text, tokenizer, data, trainer


def test_training_loss_decreases(corpus_docs, training_config):
    """A short run on the corpus must show the loss going down."""
    _, _, _, trainer = _build_components(corpus_docs, training_config)
    losses = trainer.train(
        steps=training_config["steps"],
        batch_size=training_config["batch_size"],
        seq_len=training_config["seq_len"],
    )
    assert len(losses) == training_config["steps"]
    assert losses[0] > losses[-1], "Loss did not decrease"
    assert losses[-1] < losses[0] * 0.9


def test_checkpoints_are_writable(corpus_docs, training_config, tmp_path):
    """The experiment writes a model checkpoint and tokenizer file."""
    from src.training.model import MiniGPT
    from src.training.tokenizer import CharTokenizer
    from src.training.optimizer import Adam
    from src.training.dataset import TokenStream
    from src.training.trainer import Trainer

    text, tokenizer, data, trainer = _build_components(corpus_docs, training_config)

    trainer.train(
        steps=training_config["steps"],
        batch_size=training_config["batch_size"],
        seq_len=training_config["seq_len"],
    )

    model = trainer.model
    model.save(tmp_path / "model.npz")
    tokenizer.save(tmp_path / "tokenizer.json")

    assert (tmp_path / "model.npz").exists()
    assert (tmp_path / "tokenizer.json").exists()
