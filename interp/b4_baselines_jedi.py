"""B4 (Paper B claims B2, B3, B4; round-2 review weaknesses 5 and 6): baselines for the nested if-statement program of E11.
Fidelity vs description length (DL) on identical splits, and edit agreement under the E13 / E7 / E20 edit set.

Splits (as E11, interp/e11_mars_all.Data): fit on sample(per_class=6000, seed=0) after a fixed permutation (rng 0), first 2/3
= fit (20,000 jets), last 1/3 = dev (10,000; MARS stopping, model selection); test = sample(per_class=2000, seed=11) minus every
jet of the seed-0 sample; confirmation = sample(per_class=2000, seed=91) minus every jet of the seed-0 and seed-11 samples.
Target: the network's call (argmax of the logits) and its 5 logits. Inputs of every physics-based surrogate: the E11 physics set
(the Q observable library of the test file + E11's hand-built rank-window sums H:..., same key list as E11).

Methods (DL = number of fitted terms / nodes / non-zero weights; bits = DL * (log2 #candidate inputs + 16)):
  mars{K}   nested MARS program of E11 (b), max_terms per unit K in {2, 4, 8, 16, 30} (K = 30 = the stored E11 fit), every
            stage evaluated on the previous stage's chain values, end to end from particles to z; logits = the network's own
            last layer d7 applied to the chain z (interp.e11_verify.Last). DL = total hinge terms over the 144 live units.
  zphys     E11 (a): the 16 z neurons fitted directly on physics (stored e11_fit_a_z), then d7.  DL = its terms.
  direct    MARS (jetdistill.mars.fit_neuron, same candidates and knots) from physics directly to each of the 5 logits
            (target shifted by a constant per class so that it is positive; the shift is subtracted again), forward path of up
            to 100 terms per logit with the stall rule off; prefix lengths k in {2, 4, 8, 16, 32, 64, 100} per logit, each
            refitted by least squares; DL = 5 k (or fewer when the dev optimum is shorter).
  tree      sklearn DecisionTreeClassifier on the network's call, depth 2-12; DL = internal nodes.
  treereg   DecisionTreeRegressor on the 5 logits (same depths; logit R2; calls = argmax).
  gbdt      HistGradientBoostingClassifier on the call, max_leaf_nodes in {4, 8, 16, 32} x max_iter in {4, ..., 128},
            learning rate 0.2, no early stopping; DL = total leaves over all trees (5 trees per iteration).
  gbdtreg   HistGradientBoostingRegressor per logit at the same settings (logit R2; calls = argmax); DL = total leaves.
  l1logit   L1 logistic regression (saga, multinomial) on standardised physics, C path; DL = non-zero weights.
  lasso     Lasso per logit on standardised physics (alpha path); DL = non-zero weights over the 5 logits.
  efp       energy-flow polynomials up to degree 3 (all 8 connected multigraphs with <= 3 edges, beta in {1, 2}, plus their
            logs; numpy, from (pT, Δη, Δφ); energyflow is not in the job env), L1 logistic on these 32 features, C path.
  PySR: not in the job env (reported as missing).
Edits (on the test jets; physics inputs recomputed from the edited particles with jetdistill.observables.compute + handbuilt):
  trans  all real particles Δη += 0.06 rad (E6v/E7)          control trans_c: each real particle moved 0.06 rad, random direction
  rot    rotation by 45° about the axis (E7)                 control rot_c:   each particle moved 2 r sin(22.5°), random direction
  A13    swap (Δη, Δφ) of pT ranks 13 and 14 (E20), jets with >= 15 particles
                                                             control A13_c:   ranks 13 and 14 each moved by their distance,
                                                                              random directions
  a32x   one particle 9-13 GeV: 10 -> 12 GeV, rank kept (E13/E7; a32 edge 11 GeV)
                                                             control a32c:    one particle 12-16 GeV: 13 -> 15 GeV (same 2 GeV)
Edit agreement per jet: [surrogate flips] == [network flips] and, if the network flips, the surrogate's new call equals the
network's new call.  Chance reference: a surrogate that never flips agrees on 1 - (network flip rate).
Modes:  mars K | direct | trees | eval      (python -m interp.b4_baselines_jedi <mode>; AI_RESULTS/jedi receives the files)
  mars/direct/trees write b4_pred_<tag>.npz (per-jet calls per set; scores on test) + b4_pred_<tag>.json (model list)
  eval writes b4.json and b4_fidelity_dl.png
"""
import argparse, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp import e11_mars_all as E
from interp.jedi_core import sample

OUT = E.OUT
T0 = time.time()
STG, UNITS, RELU, PREV = E.STG, E.UNITS, E.RELU, E.PREV
EDITS = [('trans', 'test_rc', 'trans', None), ('rot', 'test_rc', 'rot', None), ('A13', 'A13_b', 'A13', None),
         ('a32x', 'a32x_b', 'a32x_a', None),
         ('trans_c', 'test_rc', 'trans_c', 'trans'), ('rot_c', 'test_rc', 'rot_c', 'rot'), ('A13_c', 'A13_b', 'A13_c', 'A13'),
         ('a32c', 'a32c_b', 'a32c_a', 'a32x')]                  # (edit, before set, after set, edit it controls)
MAIN_EDITS = ['trans', 'A13', 'a32x', 'rot']
EDIT_LABEL = dict(trans='0.06 rad translation', rot='45° rotation', A13='angle swap of pT ranks 13|14', a32x='a32 crossing 10→12 GeV')


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


# ------------------------------------------------------------------ edits
def edit_shift(xx, de):
    e = xx.copy(); rl = e[..., 0] > 0; e[..., 1] += np.float32(de) * rl; return e


def rotate(xx, deg):
    th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th); e = xx.copy()
    e[..., 1] = c * xx[..., 1] - s * xx[..., 2]; e[..., 2] = s * xx[..., 1] + c * xx[..., 2]; return e.astype(np.float32)


def randdisp(xx, mag, seed):
    """move every particle by mag (J, 64) in a uniformly random direction (independent per particle)"""
    rng = np.random.default_rng(seed); th = rng.uniform(0, 2 * np.pi, mag.shape); e = xx.copy()
    e[..., 1] += (mag * np.cos(th)).astype(np.float32); e[..., 2] += (mag * np.sin(th)).astype(np.float32); return e


def ramp(xx, lo, hi, win):
    """E13: one particle per jet with pT in win set to lo (before) and hi (after), only where the pT rank order is kept"""
    pt = xx[..., 0]; nreal = (pt > 0).sum(1); J, slots = [], []
    for j in range(len(xx)):
        n = nreal[j]; cand = np.flatnonzero((pt[j, :n] >= win[0]) & (pt[j, :n] <= win[1]))
        ok = [i for i in cand if (i == 0 or pt[j, i - 1] >= hi) and (pt[j, i + 1] <= lo if i + 1 < 64 else True)]
        if ok:
            i = min(ok, key=lambda i: abs(pt[j, i] - (lo + hi) / 2)); J.append(j); slots.append(i)
    J, slots = np.array(J, int), np.array(slots, int)
    xb, xa = xx[J].copy(), xx[J].copy(); xb[np.arange(len(J)), slots, 0] = lo; xa[np.arange(len(J)), slots, 0] = hi
    return J, xb, xa


def efps(x, chunk=500):
    """energy-flow polynomials, z_i = pT_i / Σ pT, θ_ij = ΔR_ij^β, all connected multigraphs with 1-3 edges, β in {1, 2}"""
    x = np.asarray(x, np.float64); out = []
    for i0 in range(0, len(x), chunk):
        xb = x[i0:i0 + chunk]; pt = xb[..., 0]; z = pt / np.maximum(pt.sum(1, keepdims=True), 1e-12)
        d = np.sqrt((xb[:, :, None, 1] - xb[:, None, :, 1]) ** 2 + (xb[:, :, None, 2] - xb[:, None, :, 2]) ** 2)
        F = []
        for beta in (1, 2):
            T = d ** beta; zt = np.einsum('ij,ijk->ik', z, T); zt2 = np.einsum('ij,ijk->ik', z, T ** 2); zt3 = np.einsum('ij,ijk->ik', z, T ** 3)
            B = T * z[:, None, :]; B2 = B @ B
            F += [(z * zt).sum(1),                                   # 1 edge
                  (z * zt2).sum(1), (z * zt ** 2).sum(1),             # 2 edges: double edge; path of 3
                  (z * zt3).sum(1), (z * zt2 * zt).sum(1),            # 3 edges: triple edge; double + single
                  np.einsum('ijk,ikj->i', B2, B),                     # triangle
                  np.einsum('ij,ijk,ik->i', z * zt, T, z * zt),       # path of 4
                  (z * zt ** 3).sum(1)]                               # star
        F = np.stack(F, 1); out.append(np.concatenate([F, np.log(F + 1e-8)], 1))
    return np.concatenate(out).astype(np.float32)


EFP_NAMES = [f'{"log " if l else ""}EFP[{g}, β={b}]' for l in (0, 1) for b in (1, 2)
             for g in ('1 edge', 'double edge', 'path3', 'triple edge', 'double+single', 'triangle', 'path4', 'star')]


# ------------------------------------------------------------------ data
class Prep:
    """E11's fit/dev/test split + confirmation + edit sets; physics inputs as float32 matrices (key order D.keys)"""

    def __init__(self):
        a = SimpleNamespace(per_class=6000, test_per_class=2000, max_terms=30, n_select=6000, prefix='e11', halfplanes=False)
        E.ARGS = a; self.a = a
        D = E.Data(a); self.D = D; self.net = D.net; self.keys = D.keys; K = self.keys
        mat = lambda ph: np.stack([np.asarray(ph[k], np.float32) for k in K], 1)
        self.x, self.X, self.y = {}, {}, {}
        for s in ('fit', 'dev', 'test'):
            self.x[s], self.X[s], self.y[s] = D.x[s], mat(D.phys[s]), D.y[s]
        i0, i1, i2 = E.indices(6000, 0), E.indices(2000, 11), E.indices(2000, 91); keep = ~np.isin(i2, np.concatenate([i0, i1]))
        xc, yc, Qc = sample(2000, 91); xc, yc = xc[keep], yc[keep]; Qc = {k: np.asarray(v[keep], np.float64) for k, v in Qc.items()}
        self.x['conf'], self.y['conf'] = xc, yc; self.X['conf'] = mat({**Qc, **E.handbuilt(xc, False)}); del Qc
        log(f'confirmation sample(2000, seed 91): {len(i2)} jets, {int((~keep).sum())} also in seed 0 / 11 removed -> {len(yc)}')

    def build_sets(self):
        """edit sets (built after any MARS fit, to keep the fit's candidate cache and these matrices apart in memory)"""
        from jetdistill.observables import compute
        D, K = self.D, self.keys; mat = lambda ph: np.stack([np.asarray(ph[k], np.float32) for k in K], 1)
        xt, yt = D.x['test'], D.y['test']; real = xt[..., 0] > 0; nreal = real.sum(1); r = np.hypot(xt[..., 1], xt[..., 2])
        sets = dict(test_rc=xt, trans=edit_shift(xt, 0.06), trans_c=randdisp(xt, 0.06 * real, 101),
                    rot=rotate(xt, 45.0), rot_c=randdisp(xt, 2 * r * np.sin(np.deg2rad(22.5)) * real, 102))
        self.J = dict(test_rc=np.arange(len(yt)), trans=np.arange(len(yt)), trans_c=np.arange(len(yt)), rot=np.arange(len(yt)), rot_c=np.arange(len(yt)))
        J13 = np.flatnonzero(nreal >= 15); xb = xt[J13]; xa = xb.copy(); xa[:, [13, 14], 1:] = xb[:, [14, 13], 1:]
        dist = np.hypot(xb[:, 13, 1] - xb[:, 14, 1], xb[:, 13, 2] - xb[:, 14, 2]); mag = np.zeros(xb.shape[:2]); mag[:, 13] = mag[:, 14] = dist
        sets.update(A13_b=xb, A13=xa, A13_c=randdisp(xb, mag, 103)); self.J.update(A13_b=J13, A13=J13, A13_c=J13)
        for nm, lo, hi, win in (('a32x', 10., 12., (9, 13)), ('a32c', 13., 15., (12, 16))):
            Jr, b, af = ramp(xt, lo, hi, win); sets[f'{nm}_b'], sets[f'{nm}_a'] = b, af; self.J[f'{nm}_b'] = self.J[f'{nm}_a'] = Jr
        self.edit_info = {}
        for nm, xx in sets.items():
            t = time.time(); q = compute(np.asarray(xx, np.float64), 64)
            miss = [k for k in K if not k.startswith('H:') and k not in q]
            assert not miss, f'recomputed observables missing {miss[:5]}'
            ph = {k: np.asarray(v, np.float32).astype(np.float64) for k, v in q.items()}; ph.update(E.handbuilt(xx, False))
            self.x[nm], self.X[nm], self.y[nm] = xx, mat(ph), yt[self.J[nm]]; del q, ph
            log(f'edit set {nm}: {len(xx)} jets, physics recomputed in {time.time() - t:.0f} s')
        dq = np.abs(self.X['test_rc'] - self.X['test']).max(0) / np.maximum(np.abs(self.X['test']).max(0), 1e-12)
        self.edit_info['recompute_vs_stored_max_rel'] = float(dq.max()); self.edit_info['n_keys_rel_gt_1e-4'] = int((dq > 1e-4).sum())
        log(f'unedited test: recomputed vs stored physics, max relative difference {dq.max():.2e} ({int((dq > 1e-4).sum())} keys > 1e-4)')
        self.sets = list(self.x)
        self.logits = {}
        for s in self.sets:
            self.logits[s] = D.V[s]['logits'] if s in ('fit', 'dev', 'test') else self.net.run(self.x[s])['logits']
        self.call = {s: v.argmax(1) for s, v in self.logits.items()}
        for s in ('test', 'conf'):
            log(f'network accuracy on {s}: {(self.call[s] == self.y[s]).mean() * 100:.2f} % (N {len(self.y[s])})')

    def phys64(self, s):
        if s in ('fit', 'dev', 'test'): return self.D.phys[s]
        return {k: self.X[s][:, i].astype(np.float64) for i, k in enumerate(self.keys)}


class Store:
    """collects per-model predictions: calls on every set (int8), scores (J, 5) on test and conf"""

    def __init__(self, tag):
        self.tag, self.arr, self.meta = tag, {}, []

    def add(self, P, name, method, dl, n_inputs, calls, scores=None, kind='call', extra=None):
        for s, c in calls.items(): self.arr[f'{name}|{s}|call'] = np.asarray(c, np.int8)
        for s, v in (scores or {}).items(): self.arr[f'{name}|{s}|score'] = np.asarray(v, np.float32)
        bits = float(dl * (np.log2(max(n_inputs, 2)) + 16))
        sc = float((calls['test'] == P.call['test']).mean()); sd = float((calls['dev'] == P.call['dev']).mean())
        self.meta.append(dict(name=name, method=method, dl=int(dl), bits=bits, n_inputs=int(n_inputs), kind=kind, same_call_test=sc, same_call_dev=sd, **(extra or {})))
        log(f'{self.tag}: {name} ({method}) DL {dl} bits {bits:.0f}: same call dev {sd * 100:.2f} % test {sc * 100:.2f} %')

    def save(self):
        np.savez_compressed(OUT / f'b4_pred_{self.tag}.npz', **self.arr)
        (OUT / f'b4_pred_{self.tag}.json').write_text(json.dumps(dict(tag=self.tag, models=self.meta), indent=1, ensure_ascii=False))
        log(f'wrote {OUT}/b4_pred_{self.tag}.npz/.json ({len(self.meta)} models)')


def calls_from_logits(lo): return np.asarray(lo).argmax(1)


def logsoftmax(lo):
    lo = np.asarray(lo, np.float64); m = lo.max(1, keepdims=True); return lo - m - np.log(np.exp(lo - m).sum(1, keepdims=True))


# ------------------------------------------------------------------ modes
def mode_mars(P, K):
    from interp.e11_verify import Last, chain, count_terms
    prefix = 'e11' if K == 30 else f'b4m{K}'
    if not (OUT / f'{prefix}_fit_b_z.json').exists():
        E.ARGS.prefix, E.ARGS.max_terms = prefix, K; log(f'fitting the nested program, max_terms {K}, prefix {prefix}'); E.mode_fit_b(P.D)
    P.build_sets()
    rd = lambda n: json.loads((OUT / f'{prefix}_{n}.json').read_text())
    fb = {'P': rd('fit_a_P'), **{s: rd(f'fit_b_{s}') for s in STG[1:]}}
    for s in STG: assert [u['unit'] for u in fb[s]['units']] == UNITS[s], s
    ct = count_terms(fb); last = Last(P.net); st = Store(f'mars{K}')
    nin = int(round(np.mean([len(fb[s]['keys']) for s in STG])))
    calls, scores = {}, {}
    for s in P.sets:
        V = chain(fb, P.phys64(s)); lo = last.jax_logits(V['z']); calls[s] = calls_from_logits(lo)
        if s in ('test', 'conf'): scores[s] = lo
    st.add(P, f'mars{K}', 'nested MARS program', ct['terms'], nin, calls, scores, kind='logit', extra=dict(max_terms=K, counts=ct))
    if K == 30:
        fa = json.loads((OUT / 'e11_fit_a_z.json').read_text()); nt = sum(u['n_terms'] for u in fa['units'])
        calls, scores = {}, {}
        for s in P.sets:
            lo = last.jax_logits(E.evaluate(fa['units'], P.phys64(s), False)); calls[s] = calls_from_logits(lo)
            if s in ('test', 'conf'): scores[s] = lo
        st.add(P, 'zphys30', 'direct MARS physics → z → d7 (E11 a)', nt, len(P.keys), calls, scores, kind='logit', extra=dict(max_terms=30))
    st.save()


def mode_direct(P, kmax=100):
    from jetdistill import mars
    D = P.D; keys = P.keys; Q = {'fit': D.phys['fit'], 'dev': D.phys['dev']}
    knots = E.knots_of(D.phys['fit'], keys); NF = D.n['fit']
    sel = np.random.default_rng(1).choice(NF, min(6000, NF), replace=False)
    mars.MAX_TERMS, mars.STALL = kmax, 10 ** 6
    fits = []
    for k in range(5):
        t = time.time(); zf, zd = D.V['fit']['logits'][:, k], D.V['dev']['logits'][:, k]
        c = 1.0 - min(zf.min(), zd.min()) + 0.5 * zf.std()
        terms, coef, path = mars.fit_neuron(dict(fit=zf + c, dev=zd + c), Q, keys, knots, sel)
        fits.append(dict(cls=k, shift=float(c), terms=terms, path=path))
        log(f'direct logit {k} ({E.CL[k]}): {len(terms)} terms kept of {len(path)} tried (best dev R² {max(path):.4f}); {time.time() - t:.0f} s')
    (OUT / 'b4_direct_terms.json').write_text(json.dumps(fits, indent=1, ensure_ascii=False, default=float))
    P.build_sets()
    st = Store('direct')
    for kb in (2, 4, 8, 16, 32, 64, 100):
        if kb > kmax: continue
        parts = []
        for f in fits:
            tr = f['terms'][:kb]; tgt = D.V['fit']['logits'][:, f['cls']] + f['shift']
            co = mars._lstsq(tr, D.phys['fit'], tgt, NF); parts.append((tr, co, f['shift']))
        dl = sum(len(tr) for tr, _, _ in parts)
        if any(m['dl'] == dl for m in st.meta): continue
        calls, scores = {}, {}
        for s in P.sets:
            ph = P.phys64(s); J = len(P.y[s])
            lo = np.stack([np.stack([np.ones(J)] + [mars.term_value(t, ph) for t in tr], 1) @ co - sh for tr, co, sh in parts], 1)
            calls[s] = calls_from_logits(lo)
            if s in ('test', 'conf'): scores[s] = lo
        st.add(P, f'direct{kb}', 'direct MARS physics → logits', dl, len(keys), calls, scores, kind='logit', extra=dict(max_terms_per_logit=kb))
    st.save()


def mode_trees(P):
    import warnings
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, lasso_path
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning); warnings.filterwarnings('ignore', category=FutureWarning)
    P.build_sets(); st = Store('trees'); F = len(P.keys); Xf, cf, Lf = P.X['fit'], P.call['fit'], P.logits['fit']
    have = dict(pysr=False, energyflow=False)
    for m in have:
        try: __import__(m); have[m] = True
        except Exception: pass  # noqa: BLE001
    log(f'optional packages in the job env: {have}')

    def run_cls(name, method, dl, nin, predict, proba, extra=None):
        calls = {s: predict(s) for s in P.sets}; scores = {s: np.log(np.clip(proba(s), 1e-6, 1)) for s in ('test', 'conf')}
        st.add(P, name, method, dl, nin, calls, scores, kind='call', extra=extra)

    def run_reg(name, method, dl, nin, predict_logits, extra=None):
        L = {s: predict_logits(s) for s in P.sets}
        st.add(P, name, method, dl, nin, {s: v.argmax(1) for s, v in L.items()}, {s: L[s] for s in ('test', 'conf')}, kind='logit', extra=extra)

    t = time.time()
    for d in range(2, 13):
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xf, cf); n_int = int(m.tree_.node_count - m.tree_.n_leaves)
        run_cls(f'tree_d{d}', 'decision tree (call)', n_int, F, lambda s: m.predict(P.X[s]), lambda s: m.predict_proba(P.X[s]), dict(depth=d))
        r = DecisionTreeRegressor(max_depth=d, random_state=0).fit(Xf, Lf); n_int = int(r.tree_.node_count - r.tree_.n_leaves)
        run_reg(f'treereg_d{d}', 'decision tree (logits)', n_int, F, lambda s: r.predict(P.X[s]), dict(depth=d))
    log(f'trees done in {time.time() - t:.0f} s'); t = time.time()

    def leaves_cls(m):
        try: return int(sum(int(p.nodes['is_leaf'].sum()) for it in m._predictors for p in it))
        except Exception: return int(m.n_iter_ * 5 * m.max_leaf_nodes)  # noqa: BLE001
    for L_ in (4, 8, 16, 32):
        for T_ in (4, 8, 16, 32, 64, 128):
            m = HistGradientBoostingClassifier(max_leaf_nodes=L_, max_iter=T_, learning_rate=0.2, early_stopping=False, random_state=0).fit(Xf, cf)
            run_cls(f'gbdt_L{L_}_T{T_}', 'gradient-boosted trees (call)', leaves_cls(m), F, lambda s: m.predict(P.X[s]), lambda s: m.predict_proba(P.X[s]),
                    dict(max_leaf_nodes=L_, max_iter=T_))
            rs = [HistGradientBoostingRegressor(max_leaf_nodes=L_, max_iter=T_, learning_rate=0.2, early_stopping=False, random_state=0).fit(Xf, Lf[:, k]) for k in range(5)]
            run_reg(f'gbdtreg_L{L_}_T{T_}', 'gradient-boosted trees (logits)', sum(leaves_cls_reg(r) for r in rs), F,
                    lambda s: np.stack([r.predict(P.X[s]) for r in rs], 1), dict(max_leaf_nodes=L_, max_iter=T_))
        log(f'gbdt L {L_} done ({time.time() - t:.0f} s)')
    t = time.time()
    sc = StandardScaler().fit(Xf)
    class _Z(dict):
        def __missing__(self, s): return np.nan_to_num(sc.transform(P.X[s])).astype(np.float32)
    Z = _Z(); Z['fit'] = np.nan_to_num(sc.transform(Xf)).astype(np.float64)
    m = LogisticRegression(penalty='l1', solver='saga', C=1e-3, max_iter=200, tol=1e-3, warm_start=True, random_state=0)
    for C in (1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.3, 1.0):
        m.set_params(C=C); m.fit(Z['fit'], cf); nz = int((np.abs(m.coef_) > 0).sum())
        run_cls(f'l1logit_C{C:g}', 'L1 logistic (call)', nz, F, lambda s: m.predict(Z[s]), lambda s: m.predict_proba(Z[s]), dict(C=C))
    log(f'l1 logistic done ({time.time() - t:.0f} s)'); t = time.time()
    alphas = np.logspace(0, -3, 10); mu = Lf.mean(0); paths = []
    for k in range(5):
        _, coefs, _ = lasso_path(Z['fit'], Lf[:, k] - mu[k], alphas=alphas, max_iter=2000, tol=1e-3); paths.append(coefs)   # (F, n_alpha)
    for i, al in enumerate(alphas):
        W = np.stack([p[:, i] for p in paths], 1); nz = int((np.abs(W) > 0).sum())
        run_reg(f'lasso_a{al:.3g}', 'Lasso (logits)', nz, F, lambda s: Z[s] @ W + mu, dict(alpha=float(al)))
    log(f'lasso done ({time.time() - t:.0f} s)'); t = time.time()
    G = {s: efps(P.x[s]) for s in P.sets}; log(f'EFPs computed for {len(G)} sets ({time.time() - t:.0f} s)')
    sg = StandardScaler().fit(G['fit']); G = {s: sg.transform(v) for s, v in G.items()}
    m = LogisticRegression(penalty='l1', solver='saga', C=1e-3, max_iter=500, tol=1e-4, warm_start=True, random_state=0)
    for C in (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 10.0):
        m.set_params(C=C); m.fit(G['fit'], cf); nz = int((np.abs(m.coef_) > 0).sum())
        run_cls(f'efp_C{C:g}', 'EFP (deg ≤ 3) + L1 logistic (call)', nz, len(EFP_NAMES), lambda s: m.predict(G[s]), lambda s: m.predict_proba(G[s]), dict(C=C))
    log(f'EFP logistic done ({time.time() - t:.0f} s)')
    st.meta.append(dict(name='_env', method='env', packages=have)); st.save()


def leaves_cls_reg(r):
    try: return int(sum(int(p.nodes['is_leaf'].sum()) for it in r._predictors for p in it))
    except Exception: return int(r.n_iter_ * r.max_leaf_nodes)  # noqa: BLE001


# ------------------------------------------------------------------ evaluation
def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n); prev = 1.0
    for rank, i in reversed(list(enumerate(o, 1))):
        prev = min(prev, p[i] * n / rank); q[i] = prev
    return q


def mcnemar(a, b):
    from scipy.stats import binomtest
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = 1.0 if n10 + n01 == 0 else float(binomtest(n10, n10 + n01, 0.5).pvalue)
    return dict(a_only=n10, b_only=n01, p=p)


def mode_eval(P, B=1000):
    P.build_sets(); rng = np.random.default_rng(7)
    models, arr = [], {}
    for tag in ['mars2', 'mars4', 'mars8', 'mars16', 'mars30', 'direct', 'trees']:
        p = OUT / f'b4_pred_{tag}.json'
        if not p.exists(): log(f'missing {p}'); continue
        meta = json.loads(p.read_text())['models']; z = np.load(OUT / f'b4_pred_{tag}.npz')
        env = [m for m in meta if m['name'] == '_env']
        if env: arr['_env'] = env[0]['packages']
        for m in meta:
            if m['name'] == '_env': continue
            models.append(m); arr[m['name']] = {k.split('|', 1)[1]: z[k] for k in z.files if k.split('|')[0] == m['name']}
    log(f'{len(models)} models loaded')
    Jt = len(P.y['test']); _bl = {}                 # one bootstrap index matrix per set length (sets of equal length share it)
    for s in ['test', 'conf'] + [e[2] for e in EDITS]:
        n = len(P.y[s]); _bl.setdefault(n, rng.integers(0, n, (B, n)).astype(np.int32))
    idx = {s: _bl[len(P.y[s])] for s in ['test', 'conf'] + [e[2] for e in EDITS]}

    def ci(ind, s):
        ind = np.asarray(ind, np.float32); b = ind[idx[s]].mean(1); return [float(ind.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

    def r2(lo, ln, s):
        res = ((lo - ln) ** 2).sum(1); sq = (ln ** 2).sum(1); s1 = ln
        def one(ix):
            n = len(ix); sst = (sq[ix].sum() - ((s1[ix].sum(0)) ** 2).sum() / n); return 1 - res[ix].sum() / sst
        v = one(np.arange(len(lo))); bs = [one(i) for i in idx[s][:200]]
        return [float(v), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    pairs = rng.integers(0, Jt, (200000, 2)); pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    nls = logsoftmax(P.logits['test'])
    def ado(sc):
        ls = logsoftmax(sc); out = []
        for k in range(5):
            dn = nls[pairs[:, 0], k] - nls[pairs[:, 1], k]; ds = ls[pairs[:, 0], k] - ls[pairs[:, 1], k]; ok = (dn != 0) & (ds != 0)
            out.append(float((np.sign(dn[ok]) == np.sign(ds[ok])).mean()) if ok.any() else 0.5)
        return out

    net = dict(acc_test=ci(P.call['test'] == P.y['test'], 'test'), acc_conf=ci(P.call['conf'] == P.y['conf'], 'conf'), N_test=Jt, N_conf=len(P.y['conf']),
               acc_dev=float((P.call['dev'] == P.y['dev']).mean()), edit_N={}, edit_flip={})
    AG = {}                                       # model -> edit -> per-jet agreement
    for name, b, a, _ in EDITS:
        nf = P.call[a] != P.call[b]; net['edit_N'][name] = int(len(nf)); net['edit_flip'][name] = ci(nf, a)
        net.setdefault('edit_acc', {})[name] = [float((P.call[b] == P.y[b]).mean()), float((P.call[a] == P.y[a]).mean())]
    log('network: acc test ' + str(np.round(net['acc_test'], 4)) + ' conf ' + str(np.round(net['acc_conf'], 4)) +
        '; flip rates ' + ', '.join(f'{k} {v[0] * 100:.2f} % (N {net["edit_N"][k]})' for k, v in net['edit_flip'].items()))
    W_ = P.y['test_rc'] == 2; net['trans_W_to_Z'] = float((P.call['trans'][W_] == 3).mean()); net['trans_W_to_Z_before'] = float((P.call['test_rc'][W_] == 3).mean())
    rows = []
    for m in models:
        A = arr[m['name']]; c = A['test|call']; row = dict(m)
        row['same_call_test'] = ci(c == P.call['test'], 'test'); row['acc_test'] = ci(c == P.y['test'], 'test')
        row['same_call_conf'] = ci(A['conf|call'] == P.call['conf'], 'conf'); row['acc_conf'] = ci(A['conf|call'] == P.y['conf'], 'conf')
        row['same_call_dev'] = float((A['dev|call'] == P.call['dev']).mean())
        if m['kind'] == 'logit': row['logit_r2_test'] = r2(A['test|score'].astype(np.float64), P.logits['test'], 'test')
        row['ado_test'] = ado(A['test|score']); row['ado_test_mean'] = float(np.mean(row['ado_test']))
        row['edits'] = {}; AG[m['name']] = {}
        for name, b, a, _ in EDITS:
            nb, na = P.call[b], P.call[a]; sb, sa = A[f'{b}|call'], A[f'{a}|call']; nf, sf = na != nb, sa != sb
            ag = (nf == sf) & (~nf | (sa == na)); AG[m['name']][name] = ag
            row['edits'][name] = dict(agree=ci(ag, a), flip_rate=float(sf.mean()), recall=float((sf & (sa == na))[nf].mean()) if nf.any() else None,
                                      precision=float((nf & (sa == na))[sf].mean()) if sf.any() else None, chance=float(1 - nf.mean()))
            if name == 'trans':
                row['edits'][name]['W_to_Z'] = float((sa[P.y[a] == 2] == 3).mean()); row['edits'][name]['W_to_Z_before'] = float((sb[P.y[a] == 2] == 3).mean())
        rows.append(row)
        log(f"{m['name']:22s} DL {m['dl']:5d} same call {row['same_call_test'][0] * 100:6.2f} [{row['same_call_test'][1] * 100:.2f}, {row['same_call_test'][2] * 100:.2f}] "
            f"conf {row['same_call_conf'][0] * 100:6.2f} acc {row['acc_test'][0] * 100:.2f}" + (f" R² {row['logit_r2_test'][0]:.3f}" if 'logit_r2_test' in row else '') +
            ' | agree ' + ' '.join(f"{k} {row['edits'][k]['agree'][0] * 100:.2f}" for k, *_ in EDITS))
    # paired comparisons at matched DL: nested mars K vs the best baseline with DL <= DL(mars K)
    base = [r for r in rows if not r['name'].startswith('mars')]; nested = sorted([r for r in rows if r['name'].startswith('mars')], key=lambda r: r['dl'])
    comp, pv, cells = [], [], []
    for r in nested:
        elig = [b for b in base if b['dl'] <= r['dl']]
        if not elig: continue
        best = max(elig, key=lambda b: b['same_call_dev'])                  # selected on the dev split (not the test jets)
        sc_n, sc_b = arr[r['name']]['test|call'] == P.call['test'], arr[best['name']]['test|call'] == P.call['test']
        d = (sc_n.astype(float) - sc_b)[idx['test']].mean(1)
        e = dict(nested=r['name'], dl=r['dl'], best_baseline=best['name'], best_dl=best['dl'], best_method=best['method'],
                 same_call_diff=[float((sc_n.astype(float) - sc_b).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                 same_call_mcnemar=mcnemar(sc_n, sc_b), edits={})
        for name in MAIN_EDITS + [c[0] for c in EDITS if c[3]]:
            a_, b_ = AG[r['name']][name], AG[best['name']][name]; ix = idx[[c[2] for c in EDITS if c[0] == name][0]]
            dd = (a_.astype(float) - b_)[ix].mean(1)
            orc = max(elig, key=lambda b: b['edits'][name]['agree'][0])        # oracle: best baseline on this edit (selected on test)
            e['edits'][name] = dict(diff=[float((a_.astype(float) - b_).mean()), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                                    mcnemar=mcnemar(a_, b_), nested_agree=float(a_.mean()), baseline_agree=float(b_.mean()),
                                    oracle_baseline=orc['name'], oracle_agree=orc['edits'][name]['agree'][0],
                                    oracle_mcnemar=mcnemar(a_, AG[orc['name']][name]))
            if name in MAIN_EDITS: pv.append(e['edits'][name]['mcnemar']['p']); cells.append((len(comp), name))
        comp.append(e)
    q = bh(pv) if pv else []
    for (i, name), qq in zip(cells, q): comp[i]['edits'][name]['mcnemar']['q_bh'] = float(qq)
    for e in comp:
        log(f"{e['nested']} (DL {e['dl']}) vs {e['best_baseline']} (DL {e['best_dl']}): same call diff {e['same_call_diff'][0] * 100:+.2f} "
            f"[{e['same_call_diff'][1] * 100:+.2f}, {e['same_call_diff'][2] * 100:+.2f}] | " +
            ' '.join(f"{k} {v['diff'][0] * 100:+.2f} p {v['mcnemar']['p']:.2g}" + (f" q {v['mcnemar']['q_bh']:.2g}" if 'q_bh' in v['mcnemar'] else '') for k, v in e['edits'].items()))
    # lowest DL at which each baseline method reaches the nested program's same call (K = 30)
    reach = {}
    ref = {r['name']: r['same_call_test'][0] for r in nested}
    for meth in sorted({b['method'] for b in base}):
        rr = sorted([b for b in base if b['method'] == meth], key=lambda b: b['dl'])
        reach[meth] = {k: next((dict(name=b['name'], dl=b['dl']) for b in rr if b['same_call_test'][0] >= v), None) for k, v in ref.items()}
        reach[meth]['max_same_call'] = max(b['same_call_test'][0] for b in rr); reach[meth]['max_dl'] = max(b['dl'] for b in rr)
    res = dict(procedure=__doc__, network=net, edit_info=P.edit_info, models=rows, comparisons=comp, n_bh_cells=len(pv), reach_nested_same_call=reach,
               packages=arr.get('_env'), splits=dict(fit=P.D.n['fit'], dev=P.D.n['dev'], test=Jt, conf=len(P.y['conf']), overlap_removed_test=P.D.overlap),
               bootstrap=B)
    (OUT / 'b4.json').write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float)); log(f'wrote {OUT}/b4.json')
    plot(res)


def plot(res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = res['models']
    fam = [('nested MARS program', 'nested MARS program (E11)', '#1c5cab', 'o', '-'),
           ('direct MARS physics → logits', 'direct MARS → logits', '#e07b00', 's', '-'),
           ('decision tree (call)', 'decision tree', '#2a9d3c', '^', '-'),
           ('gradient-boosted trees (call)', 'GBDT on call (leaves 16)', '#c0392b', 'D', '-'),
           ('gradient-boosted trees (logits)', 'GBDT on logits (leaves 32)', '#e377c2', 'd', '--'),
           ('L1 logistic (call)', 'L1 logistic', '#7b4fa0', 'v', '-'),
           ('Lasso (logits)', 'Lasso on logits', '#8c6d31', 'P', '--'),
           ('EFP (deg ≤ 3) + L1 logistic (call)', 'EFP deg ≤ 3 + L1 logistic', '#555555', 'X', ':')]
    def series(meth):
        rr = [r for r in rows if r['method'] == meth]
        if meth == 'gradient-boosted trees (call)': rr = [r for r in rr if r.get('max_leaf_nodes') == 16]
        if meth == 'gradient-boosted trees (logits)': rr = [r for r in rr if r.get('max_leaf_nodes') == 32]
        return sorted(rr, key=lambda r: r['dl'])
    panels = [('same call with the network, test jets (%)', lambda r: r['same_call_test'], None)] + \
             [(f'edit agreement: {EDIT_LABEL[e]} (%)', (lambda e: lambda r: r['edits'][e]['agree'])(e), e) for e in MAIN_EDITS] + \
             [('edit agreement, matched random-direction controls (mean of 4) (%)', lambda r: [np.mean([r['edits'][c]['agree'][0] for c in ('trans_c', 'rot_c', 'A13_c', 'a32c')])] * 3, 'ctrl')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6)); axes = axes.ravel()
    for ax, (title, f, e) in zip(axes, panels):
        for meth, lab, col, mk, ls in fam:
            rr = series(meth)
            if not rr: continue
            x = [r['dl'] for r in rr]; v = np.array([f(r) for r in rr]) * 100
            ax.plot(x, v[:, 0], ls=ls, marker=mk, color=col, ms=4.5, lw=1.4, label=lab)
            if e is None or e != 'ctrl': ax.fill_between(x, v[:, 1], v[:, 2], color=col, alpha=0.15, lw=0)
        if e in MAIN_EDITS:
            ch = (1 - res['network']['edit_flip'][e][0]) * 100
            ax.axhline(ch, color='#999999', lw=1, ls='--'); ax.text(0.01, ch, f'never-flip surrogate {ch:.1f} %', fontsize=8, color='#666666', va='bottom', transform=ax.get_yaxis_transform())
            ax.set_title(title + f'\nnetwork changes {res["network"]["edit_flip"][e][0] * 100:.1f} % of calls, N {res["network"]["edit_N"][e]:,}', fontsize=9.5, loc='left')
        else:
            ax.set_title(title, fontsize=9.5, loc='left')
        ax.set_xscale('log'); ax.set_xlabel('description length (fitted terms / tree nodes / non-zero weights)', fontsize=8.5); ax.grid(alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('JEDI-linear surrogates vs description length: same call on held-out jets, and agreement with the network\'s call changes under input edits '
                 f'(test N {res["splits"]["test"]:,}; bands 95 % bootstrap)', fontsize=10.5, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.96)); p = OUT / 'b4_fidelity_dl.png'; fig.savefig(p, dpi=140); log(f'wrote {p}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['mars', 'direct', 'trees', 'eval', 'plot']); ap.add_argument('k', nargs='?', type=int, default=30)
    a = ap.parse_args(); log('args', vars(a))
    if a.mode == 'plot':
        plot(json.loads((OUT / 'b4.json').read_text()))
    else:
        P = Prep()
        if a.mode == 'mars': mode_mars(P, a.k)
        elif a.mode == 'direct': mode_direct(P, a.k if a.k != 30 else 100)
        elif a.mode == 'trees': mode_trees(P)
        else: mode_eval(P)
    log('done')
