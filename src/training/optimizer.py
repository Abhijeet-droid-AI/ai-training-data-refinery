import numpy as np


def clip_gradients(grads, max_norm):
    """
    Clip gradients globally to a maximum L2 norm.

    Returns the pre-clip norm so the trainer can log it.
    """

    total_norm = np.sqrt(
        sum(
            float((grad**2).sum())
            for grad in grads.values()
        )
    )

    if total_norm > max_norm:

        scale = max_norm / (total_norm + 1e-12)

        for name in grads:

            grads[name] *= scale

    return total_norm


class Adam:
    """
    Standard Adam optimizer over a dict of parameters.
    """

    def __init__(
        self,
        params,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):

        self.learning_rate = learning_rate

        self.beta1 = beta1

        self.beta2 = beta2

        self.epsilon = epsilon

        self.step_count = 0

        self.m = {
            name: np.zeros_like(value)
            for name, value in params.items()
        }

        self.v = {
            name: np.zeros_like(value)
            for name, value in params.items()
        }

    def step(self, params, grads):

        self.step_count += 1

        bias1 = 1.0 - self.beta1**self.step_count

        bias2 = 1.0 - self.beta2**self.step_count

        for name, gradient in grads.items():

            self.m[name] = (
                self.beta1 * self.m[name]
                + (1.0 - self.beta1) * gradient
            )

            self.v[name] = (
                self.beta2 * self.v[name]
                + (1.0 - self.beta2) * gradient**2
            )

            m_hat = self.m[name] / bias1

            v_hat = self.v[name] / bias2

            params[name] -= self.learning_rate * (
                m_hat
                / (np.sqrt(v_hat) + self.epsilon)
            )
