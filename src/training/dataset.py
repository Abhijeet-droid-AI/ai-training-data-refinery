import numpy as np


class TokenStream:
    """
    A flat stream of token ids supporting wraparound
    window sampling for tiny corpora.

    Random contiguous windows are drawn with index
    wraparound, so even corpora shorter than the
    context length produce valid training batches.
    """

    def __init__(self, token_ids):

        self.token_ids = np.asarray(
            token_ids,
            dtype=np.int64,
        )

        if self.token_ids.size == 0:

            raise ValueError(
                "TokenStream requires at least one token."
            )

    def __len__(self):

        return self.token_ids.size

    def get_batch(self, rng, batch_size, seq_len):

        starts = rng.integers(
            0,
            len(self),
            size=batch_size,
        )

        offsets = np.arange(seq_len + 1)

        windows = (
            starts[:, None]
            + offsets[None, :]
        ) % len(self)

        batch = self.token_ids[windows]

        contexts = batch[:, :-1]

        targets = batch[:, 1:]

        return contexts, targets
