import numpy as np


class Sampler:
    """
    Autoregressive text generation from a trained model.
    """

    def __init__(self, model, tokenizer):

        self.model = model

        self.tokenizer = tokenizer

    def generate(
        self,
        prompt,
        max_new_tokens=50,
        temperature=1.0,
        rng=None,
    ):

        rng = rng or np.random.default_rng()

        tokens = self.tokenizer.encode(prompt)

        if not tokens:

            tokens = [self.tokenizer.unk_id]

        for _ in range(max_new_tokens):

            context = tokens[-self.model.seq_len :]

            context_array = np.array(
                [context],
                dtype=np.int64,
            )

            logits, _ = self.model.forward(context_array)

            next_logits = logits[0, -1] / temperature

            probabilities = self._softmax(next_logits)

            next_token = int(
                rng.choice(
                    len(probabilities),
                    p=probabilities,
                )
            )

            tokens.append(next_token)

        return self.tokenizer.decode(tokens)

    @staticmethod
    def _softmax(values):

        shifted = values - values.max()

        exp = np.exp(shifted)

        return exp / exp.sum()
