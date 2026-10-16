"""The JEDI-linear network (official 3-feature models, HGQ2 fixed-point inference), layer by layer.

Network(n).run(x) gives, for particles x (J, n, 3):
  z       (J, 16)  the last hidden layer before its ReLU ("the neurons" the formulas describe)
  h       (J, 16)  the last hidden layer (ReLU of z), before the last layer's input rounding
  logits  (J, 5)   the class scores (g, q, W, Z, t)
Network.last = (K (16, 5), b (5,), int bits (16,), frac bits (16,)): the network's own last layer. It rounds each neuron to
a multiple of 2^-f and wraps it modulo 2^i (unsigned fixed point), then K·h + b.  logits_from_h() applies it.
untrained=True: every kernel, bias and batch-norm scale is redrawn from a Gaussian with that weight's trained mean and
spread (the control; no trained value is reused), the trained number formats are kept."""
import os
os.environ.setdefault('KERAS_BACKEND', 'jax')
from pathlib import Path
import numpy as np
from .config import MODELS
from .data import inputs

LAYERS = dict(l1='q_einsum_dense_batchnorm', qs='q_sum', l2='q_einsum_dense_batchnorm_1', dd='q_einsum_dense_batchnorm_2', add='q_add',
              l3='q_einsum_dense_batchnorm_3', qs1='q_sum_1', d4='q_einsum_dense_batchnorm_4', d5='q_einsum_dense_batchnorm_5',
              d6='q_einsum_dense_batchnorm_6', d7='q_einsum_dense_batchnorm_7')


def model_path(n):
    return sorted((Path(MODELS) / '3-feature' / f'jet_classifier_large_{n}' / 'models').glob('*.keras'))[0]


class Network:
    def __init__(self, n, untrained=False, seed=0):
        import keras, hgq, jax, jax.numpy as jnp  # noqa: F401  (hgq registers the quantized layers)
        self.n = n; self.model = keras.models.load_model(model_path(n), compile=False)
        if untrained:
            rng = np.random.default_rng(seed)
            for w in self.model.weights:
                if getattr(w, 'path', w.name).endswith(('/kernel', '/bias', '/gamma')):
                    v = np.asarray(w.numpy()); w.assign(rng.normal(v.mean(), v.std(), v.shape).astype(v.dtype))
        L = {k: self.model.get_layer(v) for k, v in LAYERS.items()}; KB = {}
        for k in ('l1', 'l2', 'dd', 'l3', 'd4', 'd5', 'd6', 'd7'):
            K, b = L[k].get_fused_qkernel_and_qbias(False, L[k].moving_mean, L[k].moving_variance); KB[k] = (jnp.asarray(K), jnp.asarray(b))

        def dense(k, x, act=True):
            z = jnp.einsum(L[k].equation, L[k].iq(x, training=False), KB[k][0]) + KB[k][1]
            return L[k].activation(z) if act and L[k].activation is not None else z

        def pool(k, a):
            return jnp.sum(L[k].iq(a, training=False), axis=L[k].axes, keepdims=L[k].keepdims) * L[k].scale

        def forward(x):
            a1 = dense('l1', x); d = dense('dd', pool('qs', a1)); s = dense('l2', a1)
            qs_, qd_ = L['add'].iq([s, d], training=False)
            h = dense('d5', dense('d4', pool('qs1', dense('l3', qs_ + qd_))))
            z = dense('d6', h, act=False).reshape(len(x), -1); hh = dense('d6', h)
            return z, hh.reshape(len(x), -1), dense('d7', hh).reshape(len(x), -1)

        self._fwd = jax.jit(forward); self._d7 = jax.jit(lambda h: dense('d7', h))
        q = L['d7'].iq.quantizer; k_, i_, f_ = (np.asarray(getattr(q, a)).ravel() for a in ('k', 'i', 'f'))
        assert not k_.any(), 'the last layer rounds its inputs as unsigned numbers'
        K7, b7 = (np.asarray(x) for x in KB['d7'])
        self.last = (K7.reshape(-1, 5).astype(np.float64), b7.ravel().astype(np.float64), i_.astype(int), f_.astype(int))

    def run(self, x, batch=10000):
        """x: particles (J, n, 3) -> dict(z, h, logits)"""
        out = {'z': [], 'h': [], 'logits': []}
        for i in range(0, len(x), batch):
            for k, v in zip(out, self._fwd(inputs(x[i:i + batch], self.n))): out[k].append(np.asarray(v, np.float64))
        return {k: np.concatenate(v) for k, v in out.items()}

    def logits_from_h(self, H):
        """the network's own last layer applied to neuron values H (J, 16), in float64 (same as the exported files)"""
        return logits_from_h(H, *self.last)


def round_wrap(H, i_bits, f_bits):
    """each neuron rounded to a multiple of 2^-f, then wrapped modulo 2^i (the last layer's unsigned fixed-point input)"""
    s = 2.0 ** np.asarray(f_bits, np.float64)
    return (np.floor(np.asarray(H, np.float64) * s + 0.5) / s) % 2.0 ** np.asarray(i_bits, np.float64)


def logits_from_h(H, K, b, i_bits, f_bits):
    return round_wrap(H, i_bits, f_bits) @ K + b
