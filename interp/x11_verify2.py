"""X11 verifier, second pass: the X11 native-vs-rotation naming test on a third, fresh jet sample, and an untrained
architecture control that the first verifier could not run (the quantized untrained JEDI-linear has 1 live S unit).

Parts (python -m interp.x11_verify2 --part <p>; writes $AI_RESULTS/jedi/x11_verify2_<p>.json):
  trained    JEDI-linear S (the real quantized network, interp.jedi_core.Jedi(64)): native vs 50 raw (rng 9000+i) and 10
             z-scored (rng 9100+i) rotations; Deep Sets dsets-d64 seed 0 S: native vs 10 raw rotations (rng 9200+i).
  untrained  architecture controls in float arithmetic, no quantizers:
             - JEDI-linear architecture (l1 -> pool -> dd, l2, l3(s + g) -> pool, same einsum equations, pool scales and input
               scaling as the trained network), every fused kernel and bias redrawn from a Gaussian with that layer's trained
               mean and spread (the project's untrained rule, applied to the fused weights), seeds 0 and 1;
             - Deep Sets dsets-d64 with phi_0 / phi_2 weights and biases redrawn the same way (seed 0; padded slots masked as in
               the trained Deep Sets).
             S layer of each: native vs 20 raw (rng 9300+i) and 5 z-scored (rng 9400+i) rotations (Deep Sets: raw only).
Jets: fit = sample(3000, 401) (permutation rng(0): 2/3 fit-train, 1/3 fit-dev); test = sample(3000, 409) minus every jet also in
the fit sample. Library, naming rule (<= 8 MARS terms on the 97 hand-built rank-window observables, held-out R^2 >= 0.9) and
live-unit rule as interp.x11_sae_atlas (imported unchanged); comparison code from interp.x11_verify.compare.
Bootstrap: B = 300 test-jet resamples (rng 11). p = (1 + #{rot >= native}) / (n + 1).
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import interp.x11_sae_atlas as X
import interp.x11_verify as V

OUT = Path(os.environ.get('AI_RESULTS', Path(__file__).resolve().parents[1] / 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
B = 300


def log(*a): print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def jets():
    from interp.jedi_core import sample
    x, y, _ = sample(3000, 401); xt, yt, _ = sample(3000, 409)
    i0, i1 = X.indices(3000, 401), X.indices(3000, 409); keep = ~np.isin(i1, i0); xt, yt = xt[keep], yt[keep]
    J = len(x); perm = np.random.default_rng(0).permutation(J); nf = 2 * J // 3
    log(f'jets: fit {J} (train {nf}, dev {J - nf}), test {len(xt)} (removed {int((~keep).sum())} overlapping)')
    return x, y, xt, yt, perm[:nf], perm[nf:], dict(fit=J, fit_train=nf, fit_dev=J - nf, test=len(xt), overlap_removed=int((~keep).sum()),
                                                    fit_seed=401, test_seed=409, per_class=3000)


def setup():
    x, y, xt, yt, itr, idv, Jinfo = jets()
    H = X.library(x); Ht = X.library(xt)
    keys = [k for k in H if np.isfinite(H[k]).all() and np.isfinite(Ht[k]).all() and np.ptp(H[k][itr]) > 0]
    fitter = X.Fitter({k: H[k][itr] for k in keys}, {k: H[k][idv] for k in keys}, keys, X.N_SEL)
    W = np.stack([np.bincount(r_, minlength=len(xt)) for r_ in np.random.default_rng(11).integers(0, len(xt), (B, len(xt)))]).astype(np.float64)
    return x, y, xt, yt, itr, idv, Jinfo, Ht, fitter, W, len(keys)


class FloatJediUntrained:
    """JEDI-linear's architecture in float64, no quantizers; fused kernels / biases of l1, dd, l2, l3 redrawn N(mean, sd)"""

    def __init__(self, net, seed):
        rng = np.random.default_rng(seed); self.L = net.L; self.inputs = net.inputs; self.KB = {}
        for k in ('l1', 'dd', 'l2', 'l3'):
            K, b = net.KB[k]; self.KB[k] = (rng.normal(K.mean(), K.std(), K.shape), rng.normal(b.mean(), b.std(), b.shape))
        self.info = {k: dict(K_shape=list(net.KB[k][0].shape), eq=self.L[k].equation, K_mean=float(net.KB[k][0].mean()),
                             K_sd=float(net.KB[k][0].std()), b_mean=float(net.KB[k][1].mean()), b_sd=float(net.KB[k][1].std()))
                     for k in self.KB}

    def dense(self, k, x):
        K, b = self.KB[k]; return np.maximum(np.einsum(self.L[k].equation, x, K) + b, 0)

    def pool(self, k, a):
        L = self.L[k]; ax = tuple(int(i) for i in np.atleast_1d(L.axes))
        return np.sum(a, axis=ax, keepdims=L.keepdims) * float(np.asarray(L.scale))

    def S(self, x, batch=2000):
        out = []
        for i in range(0, len(x), batch):
            xn = np.asarray(self.inputs(x[i:i + batch], 64), np.float64)
            a = self.dense('l1', xn); g = self.dense('dd', self.pool('qs', a)); s = self.dense('l2', a)
            u = self.dense('l3', s + g); out.append(self.pool('qs1', u).reshape(len(xn), -1))
        return np.concatenate(out)


class DSUntrained(X.FloatDS):
    def __init__(self, path, seed):
        super().__init__(path); rng = np.random.default_rng(seed)
        for k in ('phi_0_weight', 'phi_0_bias', 'phi_2_weight', 'phi_2_bias'):
            v = self.z[k]; self.z[k] = rng.normal(v.mean(), v.std(), v.shape)


def dsets_path():
    return Path(os.environ.get('AI_SMALL_MODELS', '/j-jepa-vol/aaron-interp/models/small')) / 'dsets-d64-n64' / 'seed0' / 'weights.npz'


def part_trained():
    from interp.jedi_core import Jedi
    x, y, xt, yt, itr, idv, Jinfo, Ht, fitter, W, nk = setup()
    R = dict(part='trained', jets=Jinfo, library_n=nk, B=B)
    net = Jedi(64); A = net.run(x, keep=('S', 'logits')); At = net.run(xt, keep=('S', 'logits'))
    R['jedi_accuracy'] = dict(fit=float((A['logits'].argmax(1) == y).mean()), test=float((At['logits'].argmax(1) == yt).mean()))
    log(f'JEDI-linear accuracy {R["jedi_accuracy"]}')
    R['jedi_S'], *_ = V.compare('trained jedi S', fitter, (A['S'][itr], A['S'][idv], At['S']), Ht, W, 50, 10, 9000, 9100)
    (OUT / 'x11_verify2_trained.json').write_text(json.dumps(X.clean(R), ensure_ascii=False))
    run, _, acc_ref = X.get_model('dsets'); D = run(x); Dt = run(xt)
    R['dsets_accuracy'] = dict(fit=float((D['logits'].argmax(1) == y).mean()), reference=acc_ref); log(f'Deep Sets accuracy {R["dsets_accuracy"]}')
    R['dsets_S'], *_ = V.compare('trained dsets S', fitter, (D['S'][itr], D['S'][idv], Dt['S']), Ht, W, 10, 0, 9200, 0)
    R['seconds'] = time.time() - T0
    (OUT / 'x11_verify2_trained.json').write_text(json.dumps(X.clean(R), ensure_ascii=False)); log('wrote x11_verify2_trained.json')


def part_untrained():
    from interp.jedi_core import Jedi
    x, y, xt, yt, itr, idv, Jinfo, Ht, fitter, W, nk = setup()
    R = dict(part='untrained', jets=Jinfo, library_n=nk, B=B)
    net = Jedi(64)
    for seed in (0, 1):
        fj = FloatJediUntrained(net, seed); h = fj.S(x); ht = fj.S(xt)
        R[f'jedi_float_untrained_s{seed}_weights'] = fj.info
        log(f'untrained float JEDI seed {seed}: S shape {h.shape}, S sd range [{h.std(0).min():.3g}, {h.std(0).max():.3g}]')
        R[f'jedi_float_untrained_s{seed}'], *_ = V.compare(f'untrained float jedi s{seed} S', fitter, (h[itr], h[idv], ht), Ht, W, 20, 5, 9300, 9400)
        (OUT / 'x11_verify2_untrained.json').write_text(json.dumps(X.clean(R), ensure_ascii=False))
    ds = DSUntrained(dsets_path(), 0); h = ds.run(x)['S']; ht = ds.run(xt)['S']
    R['dsets_untrained_s0'], *_ = V.compare('untrained dsets s0 S', fitter, (h[itr], h[idv], ht), Ht, W, 20, 0, 9300, 0)
    R['seconds'] = time.time() - T0
    (OUT / 'x11_verify2_untrained.json').write_text(json.dumps(X.clean(R), ensure_ascii=False)); log('wrote x11_verify2_untrained.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', choices=['trained', 'untrained'], required=True); a = ap.parse_args()
    log('args', vars(a))
    part_trained() if a.part == 'trained' else part_untrained()
