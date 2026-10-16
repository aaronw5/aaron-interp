"""JEDI-linear's exact forward pass with every intermediate exposed and a per-unit override (for ablations).

Layers (all ReLU except z; fixed point as in the HGQ2 model; same arithmetic as jetdistill.network.Network):
  a   (J, n, 64)  l1: one cut per particle, ReLU(w·x_i + b) on (pT, Δη, Δφ)        "particle statements"
  P   (J, 64)     Σ_i a_i · 2^-round(log2 n)  (= / 64 at n = 64; pool_scale[0])       "first-round objects"
  g   (J, 64)     dd: ReLU(Wd P + bd)                                                "jet statements"
  s   (J, n, 64)  l2: ReLU(W2 a_i + b2)
  u   (J, n, 64)  l3: ReLU(W3 (s_i + g) + b3)                                        "second-round statements"
  S   (J, 64)     Σ_i u_i / 16                                                       "second-round objects"
  h4  (J, 64), h5 (J, 32), z (J, 16) pre-ReLU neurons, logits (J, 5)
override: {name: (mask (units,), value (units,))} replaces the masked units of that layer by value (after the layer).
family: 3-feature / 3-feature-perminv (inputs pT, Δη, Δφ; scaled by norm_stats.json, as jetdistill.data.inputs) or
        16-feature / 16-feature-perminv (all 16 jetConstituentList columns FEAT16; scaled by norm_stats_16f.json, the
        training-archive mean / std over the first n slots, padding included, as the official dataloader).
untrained=True: weights redrawn by jetdistill.network.redraw_untrained (the Network(untrained=True) control).
"""
import os
os.environ.setdefault('KERAS_BACKEND', 'jax')
import numpy as np

STAGES = ('a', 'P', 'g', 's', 'u', 'S', 'h4', 'h5', 'z')
WIDTH = dict(a=64, P=64, g=64, s=64, u=64, S=64, h4=64, h5=32, z=16)
FEAT16 = ['px', 'py', 'pz', 'e', 'erel', 'pt', 'ptrel', 'eta', 'etarel', 'etarot', 'phi', 'phirel', 'phirot', 'deltaR',
          'costheta', 'costhetarel']          # jetConstituentList columns 0..15 (the 16-feature input, in this order)
FEAT3 = ['pt', 'etarel', 'phirel']          # columns 5, 8, 11 (the 3-feature input)


class Jedi:
    def __init__(self, n=64, family='3-feature', untrained=False, seed=0):
        import json
        import jax, jax.numpy as jnp
        from jetdistill import network as N
        from jetdistill.data import inputs
        from jetdistill.config import DATA
        import keras, hgq  # noqa: F401
        self.n = n; self.family = family; self.untrained = untrained
        self.feat = FEAT16 if family.startswith('16-feature') else FEAT3
        if family.startswith('16-feature'):
            st = json.loads((DATA / 'norm_stats_16f.json').read_text())[str(n)]
            sh, sc = np.float32(st['shift']), np.float32(st['scale'])
            self.inputs = lambda x, n_: (np.asarray(x[:, :n_]).astype(np.float32) - sh) / sc
        else:
            st = json.loads((DATA / 'norm_stats.json').read_text())[str(n)]
            self.inputs = inputs
        self.shift, self.scale = np.asarray(st['shift'], np.float64), np.asarray(st['scale'], np.float64)
        self.path = N.model_path(n, family)
        self.model = keras.models.load_model(self.path, compile=False)
        if untrained:
            N.redraw_untrained(self.model, seed)
        L = {k: self.model.get_layer(v) for k, v in N.LAYERS.items()}; self.L = L; KB = {}
        self.pool_scale = (float(np.asarray(L['qs'].scale)), float(np.asarray(L['qs1'].scale)))
        for k in ('l1', 'l2', 'dd', 'l3', 'd4', 'd5', 'd6', 'd7'):
            K, b = L[k].get_fused_qkernel_and_qbias(False, L[k].moving_mean, L[k].moving_variance)
            KB[k] = (jnp.asarray(K), jnp.asarray(b))
        self.KB = {k: (np.asarray(v[0], np.float64), np.asarray(v[1], np.float64)) for k, v in KB.items()}

        def dense(k, x, act=True):
            z = jnp.einsum(L[k].equation, L[k].iq(x, training=False), KB[k][0]) + KB[k][1]
            return L[k].activation(z) if act and L[k].activation is not None else z

        def pool(k, a):
            return jnp.sum(L[k].iq(a, training=False), axis=L[k].axes, keepdims=L[k].keepdims) * L[k].scale

        def ov(name, t, O):
            if name not in O: return t
            m, v = O[name]
            return t * (1 - m) + m * v

        def forward(x, O):
            a = ov('a', dense('l1', x), O)
            P = ov('P', pool('qs', a), O)
            g = ov('g', dense('dd', P), O)
            s = ov('s', dense('l2', a), O)
            qs_, qd_ = L['add'].iq([s, g], training=False)
            u = ov('u', dense('l3', qs_ + qd_), O)
            S = ov('S', pool('qs1', u), O)
            h4 = ov('h4', dense('d4', S), O); h5 = ov('h5', dense('d5', h4), O)
            z = ov('z', dense('d6', h5, act=False), O)
            lo = dense('d7', jax.nn.relu(z))
            J = x.shape[0]
            return dict(a=a, P=P.reshape(J, -1), g=g.reshape(J, -1), s=s, u=u, S=S.reshape(J, -1), h4=h4.reshape(J, -1),
                        h5=h5.reshape(J, -1), z=z.reshape(J, -1), logits=lo.reshape(J, -1))
        self._f = jax.jit(forward)

    def run(self, x, override=None, keep=('logits',), batch=5000):
        """x raw particles (J, n, F) in physical units (F = 3 or 16, see family) -> dict of the stages in keep (float64)"""
        O = {k: (np.asarray(m, np.float32), np.asarray(v, np.float32)) for k, (m, v) in (override or {}).items()}
        out = {k: [] for k in keep}
        for i in range(0, len(x), batch):
            r = self._f(self.inputs(x[i:i + batch], self.n), O)
            for k in keep: out[k].append(np.asarray(r[k], np.float64))
        return {k: np.concatenate(v) for k, v in out.items()}

    def run_jetwise(self, x, override, keep=('logits',), batch=5000):
        """as run(), but each override value is per jet: {name: (mask (units,), values (J, units) or (J, n, units))},
        e.g. freeze g at another input's g.  Values are reshaped to the layer's internal shape (P, g: (J, 1, units))."""
        out = {k: [] for k in keep}
        for i in range(0, len(x), batch):
            xb = x[i:i + batch]; O = {}
            for k, (m, v) in override.items():
                v = np.asarray(v[i:i + batch], np.float32)
                if k in ('P', 'g'): v = v.reshape(len(xb), 1, -1)
                O[k] = (np.asarray(m, np.float32), v)
            r = self._f(self.inputs(xb, self.n), O)
            for k in keep: out[k].append(np.asarray(r[k], np.float64))
        return {k: np.concatenate(v) for k, v in out.items()}


def sample(per_class=5000, seed=0, path=None):
    """test jets: raw particles (J, 64, 3), labels, observables dict"""
    from jetdistill.config import RESULTS
    from pathlib import Path
    d = np.load(path or Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')
    y = np.asarray(d['y']); rng = np.random.default_rng(seed)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in range(5)]))
    Q = {k[2:]: np.asarray(d[k][idx]) for k in d.files if k.startswith('Q_')}
    return np.asarray(d['x'][idx], np.float32), y[idx], Q
