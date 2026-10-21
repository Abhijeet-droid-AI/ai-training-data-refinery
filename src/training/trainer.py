from src.training.optimizer import clip_gradients
from src.utils.logger import get_logger


logger = get_logger(__name__)


class Trainer:
    """
    Runs the mini-GPT training loop over a TokenStream.

    Tracks the full loss history so reports and tests can
    verify that the model actually learns.
    """

    def __init__(
        self,
        model,
        data,
        optimizer,
        rng,
        clip_norm=1.0,
        log_every=50,
    ):

        self.model = model

        self.data = data

        self.optimizer = optimizer

        self.rng = rng

        self.clip_norm = clip_norm

        self.log_every = log_every

    def train(self, steps, batch_size, seq_len):

        losses = []

        for step in range(1, steps + 1):

            loss = self._train_step(
                batch_size,
                seq_len,
            )

            losses.append(loss)

            if step % self.log_every == 0 or step == 1:

                recent = sum(losses[-self.log_every :]) / len(
                    losses[-self.log_every :]
                )

                logger.info(
                    "Step %d | Loss %.4f | Avg %.4f",
                    step,
                    loss,
                    recent,
                )

        return losses

    def _train_step(self, batch_size, seq_len):

        contexts, targets = self.data.get_batch(
            self.rng,
            batch_size,
            seq_len,
        )

        logits, cache = self.model.forward(contexts)

        loss, dlogits = self.model.compute_loss(
            logits,
            targets,
        )

        grads = self.model.backward(cache, dlogits)

        clip_gradients(grads, self.clip_norm)

        self.optimizer.step(self.model.params, grads)

        return float(loss)
