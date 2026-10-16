"""Step 1: each of the network's 16 neurons as a sum of terms of observables (MARS-style forward selection, Friedman 1991).

Target: the neuron's value before its ReLU, z, clipped below at −0.2·(spread of its positive values): below 0 the ReLU
erases every difference, so the fit only follows z a little way into the negative range.
Candidate terms of an observable Q: Q itself, max(0, Q − t) and max(0, t − Q) for every candidate threshold t (the 5 %,
10 %, …, 95 % quantiles of Q on the fitting jets; for mass observables also m_W, m_Z, m_H, m_t and ×½, ×1.5, ×2 when the
setup offers them), and products of an already chosen threshold term with a threshold term of one of the 10 observables
most correlated with the current residual.
Each step adds the candidate that lowers the squared error most (on 12,000 of the fitting jets), refits all coefficients
by least squares on all 40,000 fitting jets, and scores R² of max(0, formula) against the neuron on 25,000 validation jets;
the kept length is the best validation R² (stop after 5 steps without improvement, or 100 terms). Only the network's
neuron values are used, never the classes.
Output: results/<setup>/n<N>/step1.json (terms in the order chosen; later steps take the first K of each neuron)."""
import json, re, sys, time
import numpy as np
from . import config, data
from .config import SETUPS, MASSES, MASS_MULT, RESULTS
from .network import Network
from .observables import library, compute, mass_ids


def log(*a):
    print(*a, flush=True)

N_KNOTS, MAX_TERMS, N_SELECT, STALL = 19, (100 if not config.SMOKE else 10), 12000, 5


def splits(n, untrained=False, sizes=None):
    """{split: (particles, network outputs, observables)} for the fitting, validation and test jets of step 1"""
    sizes = sizes or dict(fit=config.N_STEP1_FIT, dev=config.N_DEV, test=config.N_TEST_SPLIT)
    net = Network(n, untrained=untrained); out = {}
    for s, N in sizes.items():
        x = data.particles(s, n, stop=N); out[s] = (x, net.run(x), compute(x, n))
    return out, net


def candidate_observables(setup, n, Q, lowlevel=False, log=log):
    """the observables step 1 may use in this setup (finite and not constant on every split)"""
    lib = library(n); drop = mass_ids(n) if setup.no_mass else set()
    if setup.strict: drop |= strict_equivalents(Q['fit'], drop, log)   # also every observable that is (m / ΣpT)² in disguise
    keep = [k for k in lib if k not in drop and (not lowlevel or re.fullmatch(r'(pt|eta|phi)_\d+', k))]
    return [k for k in keep if all(np.isfinite(Q[s][k]).all() for s in Q) and np.ptp(Q['fit'][k]) > 0]


def strict_equivalents(Qfit, drop=(), log=None):
    """observables (not already dropped) with |correlation| > 0.98 with (m / ΣpT)² on the training jets"""
    ref = Qfit['mass_over_sum_pt_sq']; out = set()
    for k, v in Qfit.items():
        if k in drop or not (np.ptp(v) > 0 and np.isfinite(v).all()): continue
        r = np.corrcoef(v, ref)[0, 1]
        if abs(r) > 0.98:
            out.add(k)
            if log: log(f'strict: excluded {k} (correlation {r:.4f})')
    return out


def thresholds(v, mass_like, offer_masses):
    ks = set(np.unique(np.round(np.quantile(v, np.linspace(.05, .95, N_KNOTS)), 12)).tolist())
    if mass_like and offer_masses:
        lo, hi = np.quantile(v, [.02, .98])
        ks |= {float(c * m) for m in MASSES.values() for c in MASS_MULT if lo < c * m < hi}
    return sorted(ks)


def _one(v, kind, t):
    return v if kind == 'lin' else np.maximum(0, v - t) if kind == 'gt' else np.maximum(0, t - v)


def term_value(tm, Q):
    v = _one(Q[tm['q']], tm['kind'], tm.get('t'))
    return v * _one(Q[tm['q2']], tm['kind2'], tm.get('t2')) if tm.get('q2') else v


def fit_neuron(z, Q, keys, knots, sel, log=None):
    """forward selection for one neuron; z = {split: pre-activation}. Returns (terms, coefficients incl. intercept, path)"""
    zf, zd = z['fit'], z['dev']; NF = len(zf)
    zpos = zf[zf > 0]; floor = -0.2 * (zpos.std() if len(zpos) > 50 else zf.std()); target = np.maximum(zf, floor)
    cache = {}
    for k in keys:
        v = Q['fit'][k][sel]
        cache[k] = (np.stack([v] + [np.maximum(0, v - t) for t in knots[k]] + [np.maximum(0, t - v) for t in knots[k]], 1).astype(np.float32),
                    [('lin', None)] + [('gt', t) for t in knots[k]] + [('lt', t) for t in knots[k]])
    chosen, Bsel, path = [], [np.ones(len(sel))], []; best = (-np.inf, 0); stall = 0
    ys = target[sel]; hd = np.maximum(zd, 0)
    def gain(C, Qb, r):
        Cp = C - Qb @ (Qb.T @ C); nn = (Cp ** 2).sum(0); g = np.where(nn > 1e-9, (Cp.T @ r) ** 2 / np.maximum(nn, 1e-12), 0); i = int(g.argmax()); return float(g[i]), i
    for _ in range(MAX_TERMS):
        Qb, _ = np.linalg.qr(np.stack(Bsel, 1)); r = ys - Qb @ (Qb.T @ ys); Qb32, r32 = Qb.astype(np.float32), r.astype(np.float32)
        cand, rel = [], []
        for k in keys:
            C, D = cache[k]; g, i = gain(C, Qb32, r32); cand.append((g, dict(q=k, kind=D[i][0], t=D[i][1])))
            v = Q['fit'][k][sel]; rel.append((abs(np.corrcoef(v, r)[0, 1]) if np.ptp(v) > 0 else 0, k))
        top = [k for _, k in sorted(rel, reverse=True)[:10]]
        for tm in chosen:                                   # products: a chosen threshold term × a threshold term of a related observable
            if tm.get('q2') is not None or tm['kind'] == 'lin': continue
            pv = term_value(tm, {tm['q']: Q['fit'][tm['q']][sel]}).astype(np.float32)
            for k in top:
                if k == tm['q']: continue
                C, D = cache[k]; g, i = gain(C[:, 1:] * pv[:, None], Qb32, r32)
                cand.append((g, dict(q=tm['q'], kind=tm['kind'], t=tm['t'], q2=k, kind2=D[i + 1][0], t2=D[i + 1][1])))
        g, tm = max(cand, key=lambda c: c[0])
        if g <= 0: break
        chosen.append(tm); Bsel.append(term_value(tm, {k: Q['fit'][k][sel] for k in (tm['q'], tm.get('q2')) if k}))
        coef = _lstsq(chosen, Q['fit'], target, NF)
        Bd = np.stack([np.ones(len(zd))] + [term_value(t, Q['dev']) for t in chosen], 1)
        r2 = 1 - ((hd - np.maximum(Bd @ coef, 0)) ** 2).mean() / max(hd.var(), 1e-12); path.append(round(float(r2), 4))
        if r2 > best[0] + 1e-4: best, stall = (r2, len(chosen)), 0
        else: stall += 1
        if stall >= STALL or r2 > .9995: break
    chosen = chosen[:best[1]]
    return chosen, _lstsq(chosen, Q['fit'], target, NF), path


def _lstsq(terms, Qf, target, NF):
    B = np.stack([np.ones(NF)] + [term_value(t, Qf) for t in terms], 1); return np.linalg.lstsq(B, target, rcond=None)[0]


def run(setup_name, n, untrained=False, lowlevel=False, out=None, log=log):
    t0 = time.time(); setup = SETUPS[setup_name]
    S, net = splits(n, untrained)
    Q = {s: S[s][2] for s in S}; Z = {s: S[s][1]['z'] for s in S}
    keys = candidate_observables(setup, n, Q, lowlevel, log); mids = mass_ids(n)
    knots = {k: thresholds(Q['fit'][k], k in mids, setup.masses_as_thresholds) for k in keys}
    sel = np.random.default_rng(0).choice(len(Z['fit']), min(N_SELECT, len(Z['fit'])), replace=False)
    log(f'{len(keys)} observables, {sum(1 + 2 * len(v) for v in knots.values())} candidate terms, {time.time() - t0:.0f} s')
    neurons = []
    for j in range(Z['fit'].shape[1]):
        zj = {s: Z[s][:, j] for s in Z}
        if np.ptp(zj['fit']) < 1e-9:
            neurons.append(dict(neuron=j, intercept=float(zj['fit'].mean()), terms=[], constant=True)); continue
        terms, coef, path = fit_neuron(zj, Q, keys, knots, sel)
        Bt = np.stack([np.ones(len(zj['test']))] + [term_value(t, Q['test']) for t in terms], 1); ht = np.maximum(zj['test'], 0); zt = Bt @ coef
        r2h = lambda z: float(1 - ((ht - np.maximum(z, 0)) ** 2).mean() / max(ht.var(), 1e-12)); r2 = r2h(zt)
        imp = [round(r2 - r2h(zt - c * Bt[:, i + 1]), 5) for i, c in enumerate(coef[1:])]         # R² lost when the term alone is left out
        neurons.append(dict(neuron=j, intercept=float(coef[0]), terms=[dict(t, coef=float(c), importance=m) for t, c, m in zip(terms, coef[1:], imp)], dev_r2_path=path, test_r2=r2))
        log(f'neuron {j}: {len(terms)} terms, R² on test jets {r2:.4f}, {time.time() - t0:.0f} s')
    res = dict(setup=setup_name, n=n, untrained=untrained, lowlevel=lowlevel, observables=keys, n_thresholds={k: len(v) for k, v in knots.items()},
               jets=dict(fit=len(Z['fit']), dev=len(Z['dev']), test=len(Z['test'])), neurons=neurons)
    out = out or RESULTS / setup_name / f'n{n}' / 'step1.json'; out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=1))
    return res


def load(setup_name, n):
    """the step-1 neurons; a setup with step1_from reuses that setup's step 1"""
    src = SETUPS[setup_name].step1_from or setup_name
    return json.loads((RESULTS / src / f'n{n}' / 'step1.json').read_text())['neurons']


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]))
