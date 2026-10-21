import numpy as np


EPSILON = 1e-12


class MiniGPT:
    """
    A tiny decoder-only transformer implemented in pure NumPy
    with manual backpropagation.

    Architecture per layer:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    Followed by a final LayerNorm and a linear head that maps
    hidden states to next-token logits.
    """

    def __init__(
        self,
        vocab_size,
        seq_len,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=None,
        seed=0,
    ):

        if d_model % n_heads != 0:

            raise ValueError(
                "d_model must be divisible by n_heads."
            )

        self.vocab_size = vocab_size

        self.seq_len = seq_len

        self.d_model = d_model

        self.n_layers = n_layers

        self.n_heads = n_heads

        self.d_ff = d_ff or 4 * d_model

        self.head_dim = d_model // n_heads

        self.scale = 1.0 / np.sqrt(self.head_dim)

        rng = np.random.default_rng(seed)

        self.params = self._initialize_parameters(rng)

        self.grads = {
            name: np.zeros_like(value)
            for name, value in self.params.items()
        }

    # --------------------------------------------------
    # Parameter Initialization
    # --------------------------------------------------

    def _initialize_parameters(self, rng):

        params = {}

        std = 0.02

        params["token_embedding"] = rng.normal(
            0.0,
            std,
            (self.vocab_size, self.d_model),
        )

        params["position_embedding"] = rng.normal(
            0.0,
            std,
            (self.seq_len, self.d_model),
        )

        for layer in range(self.n_layers):

            prefix = f"layer{layer}_"

            params.update(
                self._initialize_layer(
                    rng,
                    prefix,
                    std,
                )
            )

        params["final_gain"] = np.ones(self.d_model)

        params["final_bias"] = np.zeros(self.d_model)

        params["w_head"] = rng.normal(
            0.0,
            std,
            (self.d_model, self.vocab_size),
        )

        params["b_head"] = np.zeros(self.vocab_size)

        return params

    def _initialize_layer(self, rng, prefix, std):

        d = self.d_model

        return {
            f"{prefix}ln1_gain": np.ones(d),
            f"{prefix}ln1_bias": np.zeros(d),
            f"{prefix}wqkv": rng.normal(0.0, std, (d, 3 * d)),
            f"{prefix}bqkv": np.zeros(3 * d),
            f"{prefix}wo": rng.normal(0.0, std, (d, d)),
            f"{prefix}bo": np.zeros(d),
            f"{prefix}ln2_gain": np.ones(d),
            f"{prefix}ln2_bias": np.zeros(d),
            f"{prefix}w1": rng.normal(0.0, std, (d, self.d_ff)),
            f"{prefix}b1": np.zeros(self.d_ff),
            f"{prefix}w2": rng.normal(0.0, std, (self.d_ff, d)),
            f"{prefix}b2": np.zeros(d),
        }

    @property
    def param_count(self):

        return sum(
            value.size
            for value in self.params.values()
        )

    # --------------------------------------------------
    # Primitive Operations
    # --------------------------------------------------

    @staticmethod
    def _linear_forward(x, w, b):

        return x @ w + b, (x, w)

    @staticmethod
    def _linear_backward(dout, cache):

        x, w = cache

        dout_2d = dout.reshape(-1, dout.shape[-1])

        x_2d = x.reshape(-1, x.shape[-1])

        dw = x_2d.T @ dout_2d

        db = dout_2d.sum(axis=0)

        dx = (dout_2d @ w.T).reshape(x.shape)

        return dx, dw, db

    @staticmethod
    def _layernorm_forward(x, gain, bias):

        mean = x.mean(axis=-1, keepdims=True)

        centered = x - mean

        variance = (centered**2).mean(
            axis=-1,
            keepdims=True,
        )

        rstd = 1.0 / np.sqrt(variance + EPSILON)

        xhat = centered * rstd

        out = xhat * gain + bias

        return out, (xhat, rstd, gain)

    @staticmethod
    def _layernorm_backward(dout, cache):

        xhat, rstd, gain = cache

        d_gain = (dout * xhat).sum(axis=(0, 1))

        d_bias = dout.sum(axis=(0, 1))

        dxhat = dout * gain

        n = xhat.shape[-1]

        dx = (
            rstd
            / n
        ) * (
            n * dxhat
            - dxhat.sum(axis=-1, keepdims=True)
            - xhat
            * (dxhat * xhat).sum(
                axis=-1,
                keepdims=True,
            )
        )

        return d_gain, d_bias, dx

    def _split_heads(self, qkv):

        batch, seq, _ = qkv.shape

        stacked = qkv.reshape(
            batch,
            seq,
            3,
            self.n_heads,
            self.head_dim,
        )

        q = stacked[:, :, 0].transpose(0, 2, 1, 3)

        k = stacked[:, :, 1].transpose(0, 2, 1, 3)

        v = stacked[:, :, 2].transpose(0, 2, 1, 3)

        return q, k, v

    def _merge_heads(self, q, k, v):

        parts = []

        for head in (q, k, v):

            merged = head.transpose(0, 2, 1, 3)

            parts.append(
                merged.reshape(
                    merged.shape[0],
                    merged.shape[1],
                    -1,
                )
            )

        return np.concatenate(parts, axis=-1)

    @staticmethod
    def _softmax(scores):

        shifted = scores - scores.max(
            axis=-1,
            keepdims=True,
        )

        exp = np.exp(shifted)

        return exp / exp.sum(
            axis=-1,
            keepdims=True,
        )

    def _causal_mask(self, seq_len):

        return np.triu(
            np.ones((seq_len, seq_len), dtype=bool),
            k=1,
        )

    # --------------------------------------------------
    # Forward Pass
    # --------------------------------------------------

    def forward(self, context):

        context = np.asarray(context, dtype=np.int64)

        batch, seq = context.shape

        if seq > self.seq_len:

            raise ValueError(
                f"Context length {seq} exceeds "
                f"model sequence length {self.seq_len}."
            )

        params = self.params

        cache = {
            "context": context,
            "seq": seq,
            "layers": [dict() for _ in range(self.n_layers)],
        }

        x = (
            params["token_embedding"][context]
            + params["position_embedding"][:seq][None, :, :]
        )

        mask = self._causal_mask(seq)

        for layer in range(self.n_layers):

            x, layer_cache = self._forward_layer(
                x,
                layer,
                mask,
            )

            cache["layers"][layer] = layer_cache

        x, cache["final_ln"] = (
            self._layernorm_forward(
                x,
                params["final_gain"],
                params["final_bias"],
            )
        )

        logits, cache["head"] = self._linear_forward(
            x,
            params["w_head"],
            params["b_head"],
        )

        return logits, cache

    def _forward_layer(self, x, layer, mask):

        params = self.params

        prefix = f"layer{layer}_"

        layer_cache = {}

        # --------------------------------------------------
        # Attention Block
        # --------------------------------------------------
        normed, layer_cache["ln1"] = (
            self._layernorm_forward(
                x,
                params[f"{prefix}ln1_gain"],
                params[f"{prefix}ln1_bias"],
            )
        )

        qkv, layer_cache["qkv"] = self._linear_forward(
            normed,
            params[f"{prefix}wqkv"],
            params[f"{prefix}bqkv"],
        )

        q, k, v = self._split_heads(qkv)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        scores = np.where(
            mask[None, None, :, :],
            -np.inf,
            scores,
        )

        probs = self._softmax(scores)

        attended = probs @ v

        merged = attended.transpose(0, 2, 1, 3).reshape(
            x.shape[0],
            x.shape[1],
            self.d_model,
        )

        output, layer_cache["attn_out"] = (
            self._linear_forward(
                merged,
                params[f"{prefix}wo"],
                params[f"{prefix}bo"],
            )
        )

        layer_cache["attention"] = (q, k, v, probs)

        x = x + output

        # --------------------------------------------------
        # MLP Block
        # --------------------------------------------------
        normed, layer_cache["ln2"] = (
            self._layernorm_forward(
                x,
                params[f"{prefix}ln2_gain"],
                params[f"{prefix}ln2_bias"],
            )
        )

        hidden, layer_cache["w1"] = self._linear_forward(
            normed,
            params[f"{prefix}w1"],
            params[f"{prefix}b1"],
        )

        activated = np.maximum(hidden, 0.0)

        layer_cache["relu_mask"] = hidden > 0

        output, layer_cache["w2"] = self._linear_forward(
            activated,
            params[f"{prefix}w2"],
            params[f"{prefix}b2"],
        )

        x = x + output

        return x, layer_cache

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    def compute_loss(self, logits, targets):

        batch, seq, vocab = logits.shape

        flat_logits = logits.reshape(batch * seq, vocab)

        flat_targets = np.asarray(
            targets,
            dtype=np.int64,
        ).reshape(batch * seq)

        shifted = flat_logits - flat_logits.max(
            axis=1,
            keepdims=True,
        )

        exp = np.exp(shifted)

        probs = exp / exp.sum(axis=1, keepdims=True)

        indices = np.arange(batch * seq)

        target_probs = probs[
            indices,
            flat_targets,
        ]

        loss = -np.log(
            target_probs + EPSILON
        ).mean()

        dlogits = probs.copy()

        dlogits[indices, flat_targets] -= 1.0

        dlogits /= batch * seq

        dlogits = dlogits.reshape(
            batch,
            seq,
            vocab,
        )

        return loss, dlogits

    # --------------------------------------------------
    # Backward Pass
    # --------------------------------------------------

    def backward(self, cache, dlogits):

        for name in self.grads:

            self.grads[name].fill(0.0)

        params = self.params

        context = cache["context"]

        seq = cache["seq"]

        dx, self.grads["w_head"], self.grads["b_head"] = (
            self._linear_backward(
                dlogits,
                cache["head"],
            )
        )

        d_gain, d_bias, dx = self._layernorm_backward(
            dx,
            cache["final_ln"],
        )

        self.grads["final_gain"] += d_gain

        self.grads["final_bias"] += d_bias

        for layer in reversed(range(self.n_layers)):

            dx = self._backward_layer(
                dx,
                cache["layers"][layer],
                layer,
            )

        # --------------------------------------------------
        # Embedding Gradients
        # --------------------------------------------------
        d_token = np.zeros_like(
            params["token_embedding"]
        )

        np.add.at(
            d_token,
            context.reshape(-1),
            dx.reshape(-1, self.d_model),
        )

        self.grads["token_embedding"] += d_token

        d_position = np.zeros_like(
            params["position_embedding"]
        )

        # position_embedding is (seq_len, d_model). The accumulated
        # gradient from the forward pass is (batch, seq, d_model), so
        # we sum over the batch dimension and write the first `seq`
        # rows.
        d_position[:seq] = dx.sum(axis=0)

        self.grads["position_embedding"] += d_position

        return self.grads

    def _backward_layer(self, dx, layer_cache, layer):

        params = self.params

        prefix = f"layer{layer}_"

        # --------------------------------------------------
        # MLP Block
        # --------------------------------------------------
        d_hidden, d_w2, d_b2 = self._linear_backward(
            dx,
            layer_cache["w2"],
        )

        self.grads[f"{prefix}w2"] += d_w2

        self.grads[f"{prefix}b2"] += d_b2

        d_hidden = (
            d_hidden * layer_cache["relu_mask"]
        )

        d_normed, d_w1, d_b1 = self._linear_backward(
            d_hidden,
            layer_cache["w1"],
        )

        self.grads[f"{prefix}w1"] += d_w1

        self.grads[f"{prefix}b1"] += d_b1

        d_gain, d_bias, d_x = self._layernorm_backward(
            d_normed,
            layer_cache["ln2"],
        )

        self.grads[f"{prefix}ln2_gain"] += d_gain

        self.grads[f"{prefix}ln2_bias"] += d_bias

        dx = dx + d_x

        # --------------------------------------------------
        # Attention Block
        # --------------------------------------------------
        d_merged, d_wo, d_bo = self._linear_backward(
            dx,
            layer_cache["attn_out"],
        )

        self.grads[f"{prefix}wo"] += d_wo

        self.grads[f"{prefix}bo"] += d_bo

        q, k, v, probs = layer_cache["attention"]

        heads = d_merged.reshape(
            d_merged.shape[0],
            d_merged.shape[1],
            self.n_heads,
            self.head_dim,
        )

        d_attended = heads.transpose(0, 2, 1, 3)

        d_v = probs.transpose(0, 1, 3, 2) @ d_attended

        d_probs = d_attended @ v.transpose(0, 1, 3, 2)

        d_scores = probs * (
            d_probs
            - (d_probs * probs).sum(
                axis=-1,
                keepdims=True,
            )
        )

        d_scores *= self.scale

        d_q = d_scores @ k

        d_k = d_scores.transpose(0, 1, 3, 2) @ q

        d_qkv = self._merge_heads(d_q, d_k, d_v)

        d_normed, d_wqkv, d_bqkv = (
            self._linear_backward(
                d_qkv,
                layer_cache["qkv"],
            )
        )

        self.grads[f"{prefix}wqkv"] += d_wqkv

        self.grads[f"{prefix}bqkv"] += d_bqkv

        d_gain, d_bias, d_x = self._layernorm_backward(
            d_normed,
            layer_cache["ln1"],
        )

        self.grads[f"{prefix}ln1_gain"] += d_gain

        self.grads[f"{prefix}ln1_bias"] += d_bias

        dx = dx + d_x

        return dx

    # --------------------------------------------------
    # Checkpoints
    # --------------------------------------------------

    def save(self, path):

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = dict(self.params)

        payload["__n_heads__"] = np.array(
            self.n_heads
        )

        np.savez(path, **payload)

    @classmethod
    def load(cls, path):

        with np.load(path) as payload:

            n_heads = int(payload["__n_heads__"])

            params = {
                key: value
                for key, value in payload.items()
                if key != "__n_heads__"
            }

        vocab_size = params["token_embedding"].shape[0]

        seq_len = params["position_embedding"].shape[0]

        d_model = params["token_embedding"].shape[1]

        n_layers = len(
            {
                key.split("_")[0]
                for key in params
                if key.startswith("layer")
            }
        )

        model = cls(
            vocab_size=vocab_size,
            seq_len=seq_len,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
        )

        for key, value in params.items():

            model.params[key] = value

        return model
