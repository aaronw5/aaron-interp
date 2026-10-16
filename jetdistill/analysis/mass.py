"""Jet mass in the network and in the formulas.

edges(n):       where the network's own neurons change with jet mass (the network never receives the mass). For each
                neuron, on the whole test file, two shapes are fitted by least squares with thresholds scanned on a 0.5-GeV
                grid:  step  h = a + b·[m > t];  window  h = a + b·[t1 < m < t2].  Uncertainty: 100 bootstrap resamplings.
thresholds(f):  every threshold on a mass observable in a formula, the nearest of m_W, m_Z, m_H, m_t, and the spacing
                of step 1's candidate grid there (a threshold at a particle mass is only notable if it is closer than
                that spacing).
Usage: python -m jetdistill.analysis.mass <N> edges | <N> thresholds <setup> <tag> ..."""
import json, sys
import numpy as np
from .. import pipeline as P
from ..config import MASSES, RESULTS
from ..observables import mass_ids

EDGES = np.arange(20, 300.01, 0.5)


def _fit(bi, h, nb):
    """exact least squares of the step and window shapes from per-bin sums (thresholds on bin edges)"""
    n = np.bincount(bi, minlength=nb).astype(float); s = np.bincount(bi, h, minlength=nb); s2 = (h * h).sum()
    cn, cs = np.concatenate([[0], np.cumsum(n)]), np.concatenate([[0], np.cumsum(s)]); N, S = cn[-1], cs[-1]; sst = s2 - S * S / N

    def sse(nin, sin):
        nout, sout = N - nin, S - sin
        with np.errstate(divide='ignore', invalid='ignore'):
            return s2 - np.where(nin > 0, sin ** 2 / nin, 0) - np.where(nout > 0, sout ** 2 / nout, 0)
    ok = lambda nin: (nin > 0.01 * N) & (N - nin > 0.01 * N)
    nin, sin = N - cn, S - cs; e = np.where(ok(nin), sse(nin, sin), np.inf); k = int(np.argmin(e))
    step = dict(t=float(EDGES[k]), r2=float(1 - e[k] / sst), up=bool(sin[k] / nin[k] > (S - sin[k]) / (N - nin[k])))
    I, J = np.triu_indices(nb + 1, 1); nin, sin = cn[J] - cn[I], cs[J] - cs[I]; e = np.where(ok(nin), sse(nin, sin), np.inf); k = int(np.argmin(e))
    win = dict(t1=float(EDGES[I[k]]), t2=float(EDGES[J[k]]), r2=float(1 - e[k] / sst), up=bool(sin[k] / nin[k] > (S - sin[k]) / (N - nin[k])))
    return step, win


def edges(n, boot=100):
    T = P.jets(n, 'full_test'); m = T['Q']['mass']; H = T['H']; nb = len(EDGES) - 1
    keep = (m >= EDGES[0]) & (m < EDGES[-1]); bi = np.clip(np.digitize(m, EDGES) - 1, 0, nb - 1)[keep]; rng = np.random.default_rng(0); out = []
    for j in range(H.shape[1]):
        h = H[keep, j]
        if h.std() == 0: continue
        step, win = _fit(bi, h, nb); bs = [_fit(bi[i], h[i], nb) for i in (rng.integers(0, len(h), len(h)) for _ in range(boot))]
        sd = lambda f: float(np.std([f(x) for x in bs]))
        step['sd'] = sd(lambda x: x[0]['t']); win['sd1'] = sd(lambda x: x[1]['t1']); win['sd2'] = sd(lambda x: x[1]['t2'])
        out.append(dict(neuron=j, step=step, window=win, frac_on=float((H[:, j] > 0).mean())))
    res = dict(masses=MASSES, bin=0.5, n_jets=int(keep.sum()), neurons=out)
    d = RESULTS / 'analysis' / f'n{n}'; d.mkdir(parents=True, exist_ok=True); (d / 'mass_edges.json').write_text(json.dumps(res, indent=1)); return res


def thresholds(setup, n, tag):
    fit = P.jets(n, 'fit'); f = P.load_formula(setup, n, tag); mids = mass_ids(n); out = []
    for nr in f:
        for t in nr['terms']:
            for s in ('', '2'):
                q, k, th = t.get('q' + s), t.get('kind' + s), t.get('t' + s)
                if not q or k == 'lin' or q not in mids: continue
                x = fit['Q'][q]; g = np.unique(np.quantile(x, np.linspace(.05, .95, 19))); name, mv = min(MASSES.items(), key=lambda kv: abs(kv[1] - th))
                out.append(dict(neuron=nr['neuron'], q=q, kind=k, t=round(float(th), 3), nearest=name, diff=round(float(th - mv), 2),
                                grid_spacing=round(float(np.diff(g)[np.clip(np.searchsorted(g, th) - 1, 0, len(g) - 2)]), 2), passes=round(float(((x > th) if k == 'gt' else (x < th)).mean()), 3)))
    return out


if __name__ == '__main__':
    n, what, *rest = sys.argv[1:]; n = int(n)
    if what == 'edges':
        for r in edges(n)['neurons']: print(r['neuron'], 'window', r['window']['t1'], '-', r['window']['t2'], 'R²', round(r['window']['r2'], 3))
    else:
        setup, *tags = rest
        for tag in tags:
            for r in sorted(thresholds(setup, n, tag), key=lambda r: abs(r['diff'])): print(tag, r)
