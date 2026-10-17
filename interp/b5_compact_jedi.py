"""B5 (thesis v3 / claims B2-B4; round-3 review weakness 3): the smallest direct hinge program on a restricted observable pool
that reproduces the network's call changes under the 0.06 rad shift, the WTA re-axis and the axis jump.

Splits (identical to B4 / E11, interp.b4_baselines_jedi): fit / dev = sample(per_class 6000, seed 0) after the rng-0
permutation (first 2/3 fit, last 1/3 dev); test = sample(2000, seed 11) minus the seed-0 jets; confirmation =
sample(2000, seed 91) minus the seed-0 and seed-11 jets.  Target: the network's 5 logits (calls = argmax).
Observable pools (keys of the E11 physics set = Q library + E11 hand-built rank-window sums H:..., plus extras):
  pT-only   (in both pools) ΣpT, pT_i, z_i, multiplicities, n_for, entropy, ΣpT / N over pT-rank windows (S0-, S19-like)
  P_axis    pT-only + every observable measured about the stored axis (radial moments, girth, Δη_i, Δφ_i, ΔR_i, (Δη, Δφ)
            tensor, rings, Ψ, H: radial window sums ΣpT·r², Σr², Σr, Σ(r−0.2)₊, Σr[r<0.165] = S29-, S1-, S30-like, |M1|,
            √(M0·M2)) + explicit X:|M1|², X:M0·M2, X:|M1|²/M0 (M0 = ΣpT, M2 = ΣpT r², M1 = ΣpT (Δη, Δφ), all about the
            stored axis).  The invariant mass family, pair distances / masses, N-subjettiness, ECFs, subjets, soft drop
            are removed.
  P_inv     pT-only + m, m/ΣpT, (m/ΣpT)² + the B4 energy-flow polynomials (8 connected multigraphs <= 3 edges, β 1 and 2,
            and their logs; numpy, interp.b4_baselines_jedi.efps).  Every angular input is a function of pairwise ΔR_ij
            or of the four-vector sum: invariant under a rigid translation / rotation in (Δη, Δφ) and under a re-axis.
  P_all     P_axis ∪ P_inv.
Fit: forward-stepwise MARS with the candidate rules of jetdistill.mars.fit_neuron (Q, max(0, Q − t), max(0, t − Q) at the
19 quantile knots 5-95 %; products of a chosen hinge with a hinge of one of the 10 observables most correlated with the
residual; gain on the same 6,000-jet subsample, rng 1), target the raw logit, no clipping; ONE term budget T shared by the
5 logits: at every step each logit proposes its best next term (least-squares refit on the fit jets) and the term goes to
the logit with the largest reduction of its dev mean squared error.  Programs T in {5, 10, 20, 30, 50} are prefixes of
the 50-step path, each refitted by least squares.  DL bits = T (log2 #pool inputs + 16) as B4.
References: mars30 (stored E11 nested program, 3,824 terms), mars16 (B4 refit, 2,219 terms), GBDT on the call
(HistGradientBoostingClassifier, B4 settings, E11 physics set) at DL (total leaves) 10-80.
Edits on test jets (physics recomputed from the edited particles: jetdistill.observables.compute + handbuilt + extras):
  trans     every real particle Δη += 0.06 rad                       control trans_c  each particle 0.06 rad, random direction
  rot       45° rotation about the axis
  wta       re-axis on the WTA axis (C/A, WTA recombination; E17 code) control wta_c    axis moved by the same distance in a
                                                                                        uniformly random direction
  hardest   re-axis on the hardest particle                         control hardest_c same distance, random direction
  jump      X4 selection on the test jets (all classes): exclusive kT two subjets, z_A in (0.2, 0.8), the hardest particle
            of the other subjet raised to 1.005 pT_hardest (raise <= 5 % of its pT), slots re-sorted by pT (stable)
                                                                    control jump_null a random other particle of that subjet
                                                                                        raised by the same ΔpT (re-sorted)
  A13       swap of (Δη, Δφ) of pT ranks 13 and 14 (jets with >= 15 particles)   control A13_c (as B4)
Metrics per program and edit: % calls changed, W→Z after the edit (true-W jets called Z), Z→W, accuracy change, per-jet
edit agreement (B4: same flip decision, and same new call when both flip), each with a 95 % jet bootstrap CI (B 1000);
paired difference surrogate − network of the per-jet call-change indicator (paired bootstrap CI) and McNemar on the paired
call-change indicators; Holm over all programs x edits.  "reproduces" = the surrogate's call-change CI overlaps the
network's CI and the paired-difference CI contains 0.
Modes:  fit  -> AI_RESULTS/jedi/<prefix>_programs.json        eval -> <prefix>.json, <prefix>_calls.npz, <prefix>.png
        all  -> both (smoke)        python -m interp.b5_compact_jedi <mode> [--per-class 6000 --test-per-class 2000
                                    --conf-per-class 2000 --prefix b5 --tmax 50]
"""
import argparse, json, re, sys, time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp import e11_mars_all as E
from interp.jedi_core import sample

OUT = E.OUT
T0 = time.time()
CL = ['g', 'q', 'W', 'Z', 't']; W_, Z_ = 2, 3
TS = (5, 10, 20, 30, 50)
POOLS = ('axis', 'inv', 'all')
POOL_LABEL = dict(axis='P_axis (axis-relative + pT)', inv='P_inv (m, EFPs + pT)', all='P_all (both)')
EDITS = [('trans', 'test_rc', 'trans', None), ('rot', 'test_rc', 'rot', None), ('wta', 'test_rc', 'wta', None),
         ('hardest', 'test_rc', 'hardest', None), ('jump', 'jump_b', 'jump', None), ('A13', 'A13_b', 'A13', None),
         ('trans_c', 'test_rc', 'trans_c', 'trans'), ('wta_c', 'test_rc', 'wta_c', 'wta'), ('hardest_c', 'test_rc', 'hardest_c', 'hardest'),
         ('jump_null', 'jump_b', 'jump_null', 'jump'), ('A13_c', 'A13_b', 'A13_c', 'A13')]
TARGET_EDITS = ('trans', 'wta', 'jump')
EPS, FMAX, ZCUT = 0.005, 0.05, 0.2           # X4 axis-jump rule
B = 1000


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


# ------------------------------------------------------------------ pools
PT_ONLY = re.compile(r'sum_pt|log_sum_pt|pt_dispersion|z_1st|z_2nd|z_3rd|z_top5|n_particles|n_pt_above_\d+|n_for_\d+pct|pt_entropy|'
                     r'pt_\d+|z_\d+|pt1_over_pt0|pt_balance01|pt2_over_pt0|soft\d+_(pt|z)|sum_pt_top\d+|z_top\d+_slots|n_real_top\d+|'
                     r'H:N\[.*\]|H:ΣpT\[.*\]|H:N\(pT>1\)\[.*\]|H:N\(pT<2\)\[.*\]')
AXIS = re.compile(r'max_dr|girth2?|mean_(eta|phi)2?|LHA|centroid_offset|planar_flow|eccentricity|width|lam[12]|orientation_deg|'
                  r'n_dr_.*|z_dr_.*|psi_.*|(eta|phi|dr|abseta|absphi|zdr)_\d+|soft\d+_(abseta|absphi|dr)|girth2_top\d+|'
                  r'H:ΣpT·r²\[.*\]|H:Σr²\[.*\]|H:Σr\[.*\]|H:ΣpT·r\[.*\]|H:Σ\(r−0\.2\)₊\[.*\]|H:Σr\[r<0\.165\]\[.*\]|'
                  r'H:N\(pT<2,r>0\.1\)\[.*\]|H:\|M1\||H:√\(M0·M2\)|X:.*')
INV_M = ('mass', 'mass_over_sum_pt', 'mass_over_sum_pt_sq')


def kind_of(k):
    if k.startswith('E:') or k in INV_M: return 'inv'
    if AXIS.fullmatch(k): return 'axis'
    if PT_ONLY.fullmatch(k): return 'pt'
    return None


def pools_of(keys):
    kd = {k: kind_of(k) for k in keys}
    P = dict(axis=[k for k in keys if kd[k] in ('pt', 'axis')], inv=[k for k in keys if kd[k] in ('pt', 'inv')],
             all=[k for k in keys if kd[k] in ('pt', 'axis', 'inv')])
    excl = [k for k in keys if kd[k] is None]
    return P, excl, kd


def extras(x):
    """explicit axis moments (stored axis) and the B4 EFPs for particles x (J, 64, 3)"""
    from interp.b4_baselines_jedi import efps, EFP_NAMES
    xx = np.asarray(x, np.float64); pt, eta, phi = xx[..., 0], xx[..., 1], xx[..., 2]
    M0 = pt.sum(1); M2 = (pt * (eta ** 2 + phi ** 2)).sum(1); M1sq = (pt * eta).sum(1) ** 2 + (pt * phi).sum(1) ** 2
    out = {'X:|M1|²': M1sq, 'X:M0·M2': M0 * M2, 'X:|M1|²/M0': M1sq / np.maximum(M0, 1e-9)}
    F = efps(x).astype(np.float64)
    out.update({f'E:{n}': F[:, i] for i, n in enumerate(EFP_NAMES)})
    return out


class Cols(Mapping):
    """float32 matrix with named columns, read as float64 columns (a dict-like physics set)"""

    def __init__(self, X, keys):
        self.X, self.keys_ = X, list(keys); self.ix = {k: i for i, k in enumerate(self.keys_)}

    def __getitem__(self, k): return self.X[:, self.ix[k]].astype(np.float64)
    def __iter__(self): return iter(self.keys_)
    def __len__(self): return len(self.keys_)


def to_cols(ph, keys):
    return Cols(np.stack([np.asarray(ph[k], np.float32) for k in keys], 1), keys)


# ------------------------------------------------------------------ data
class Prep:
    def __init__(self, a):
        ea = SimpleNamespace(per_class=a.per_class, test_per_class=a.test_per_class, max_terms=30, n_select=6000, prefix='e11', halfplanes=False)
        E.ARGS = ea
        D = E.Data(ea); self.D = D; self.net = D.net
        ex = {s: extras(D.x[s]) for s in ('fit', 'dev', 'test')}
        self.keys = [k for k in list(D.keys) + list(ex['fit']) if (k in D.keys) or (np.isfinite(ex['fit'][k]).all() and np.isfinite(ex['test'][k]).all() and np.ptp(ex['fit'][k]) > 0)]
        self.phys = {s: {**D.phys[s], **{k: v for k, v in ex[s].items() if k in self.keys}} for s in ('fit', 'dev', 'test')}
        self.x = {s: D.x[s] for s in ('fit', 'dev', 'test')}; self.y = {s: D.y[s] for s in ('fit', 'dev', 'test')}
        self.logits = {s: D.V[s]['logits'] for s in ('fit', 'dev', 'test')}
        i0, i1, i2 = E.indices(a.per_class, 0), E.indices(a.test_per_class, 11), E.indices(a.conf_per_class, 91)
        keep = ~np.isin(i2, np.concatenate([i0, i1]))
        xc, yc, Qc = sample(a.conf_per_class, 91); xc, yc = xc[keep], yc[keep]
        ph = {**{k: np.asarray(v[keep], np.float64) for k, v in Qc.items()}, **E.handbuilt(xc, False), **extras(xc)}; del Qc
        self.x['conf'], self.y['conf'] = xc, yc; self.phys['conf'] = to_cols(ph, self.keys); del ph
        self.logits['conf'] = self.net.run(xc)['logits']
        self.call = {s: v.argmax(1) for s, v in self.logits.items()}
        self.pools, self.excl, self.kind = pools_of(self.keys)
        log(f'splits fit {len(self.y["fit"])} dev {len(self.y["dev"])} test {len(self.y["test"])} conf {len(yc)} ({int((~keep).sum())} conf jets in seed 0/11 removed); '
            f'inputs {len(self.keys)}; pools ' + ', '.join(f'{p} {len(v)}' for p, v in self.pools.items()) + f'; excluded from every pool {len(self.excl)}')
        for s in ('test', 'conf'):
            log(f'network accuracy on {s}: {(self.call[s] == self.y[s]).mean() * 100:.2f} % (N {len(self.y[s])})')


# ------------------------------------------------------------------ shared-budget forward stepwise MARS
def fit_shared(pool, keys, Qf, Qd, Lf, Ld, tmax):
    from jetdistill import mars
    NF = len(Lf); sel = np.random.default_rng(1).choice(NF, min(6000, NF), replace=False)
    knots = {k: mars.thresholds(Qf[k], False, False) for k in keys}
    cache = {}
    for k in keys:
        v = Qf[k][sel]
        cache[k] = (np.stack([v] + [np.maximum(0, v - t) for t in knots[k]] + [np.maximum(0, t - v) for t in knots[k]], 1).astype(np.float32),
                    [('lin', None)] + [('gt', t) for t in knots[k]] + [('lt', t) for t in knots[k]])
    Vs = np.stack([Qf[k][sel] for k in keys], 1); Vs = Vs - Vs.mean(0); Vn = np.sqrt((Vs ** 2).sum(0)); Vn[Vn < 1e-12] = np.inf
    log(f'[{pool}] {len(keys)} inputs, {sum(1 + 2 * len(v) for v in knots.values())} single candidate terms; subsample {len(sel)}')

    def gain(C, Qb, r):
        Cp = C - Qb @ (Qb.T @ C); nn = (Cp ** 2).sum(0); g = np.where(nn > 1e-9, (Cp.T @ r) ** 2 / np.maximum(nn, 1e-12), 0); i = int(g.argmax()); return float(g[i]), i

    def refit(terms, k):
        Bf = np.stack([np.ones(NF)] + [mars.term_value(t, Qf) for t in terms], 1); co = np.linalg.lstsq(Bf, Lf[:, k], rcond=None)[0]
        Bd = np.stack([np.ones(len(Ld))] + [mars.term_value(t, Qd) for t in terms], 1); return co, float(((Ld[:, k] - Bd @ co) ** 2).mean())

    chosen = {k: [] for k in range(5)}; Bsel = {k: [np.ones(len(sel))] for k in range(5)}
    mse = {k: float(((Ld[:, k] - Lf[:, k].mean()) ** 2).mean()) for k in range(5)}

    def propose(k):
        ys = Lf[sel, k]; Qb, _ = np.linalg.qr(np.stack(Bsel[k], 1)); r = ys - Qb @ (Qb.T @ ys)
        Qb32, r32 = Qb.astype(np.float32), r.astype(np.float32); cand = []
        for q in keys:
            C, Dd = cache[q]; g, i = gain(C, Qb32, r32); cand.append((g, dict(q=q, kind=Dd[i][0], t=Dd[i][1])))
        rc = np.abs(Vs.T @ (r - r.mean())) / (Vn * max(np.sqrt(((r - r.mean()) ** 2).sum()), 1e-12))
        top = [keys[i] for i in np.argsort(-rc)[:10]]
        for tm in chosen[k]:
            if tm.get('q2') is not None or tm['kind'] == 'lin': continue
            pv = mars.term_value(tm, {tm['q']: Qf[tm['q']][sel]}).astype(np.float32)
            for q in top:
                if q == tm['q']: continue
                C, Dd = cache[q]; g, i = gain(C[:, 1:] * pv[:, None], Qb32, r32)
                cand.append((g, dict(q=tm['q'], kind=tm['kind'], t=tm['t'], q2=q, kind2=Dd[i + 1][0], t2=Dd[i + 1][1])))
        g, tm = max(cand, key=lambda c: c[0])
        co, m = refit(chosen[k] + [tm], k)
        return dict(term=tm, mse=m, gain_sel=g)

    prop = {k: propose(k) for k in range(5)}; order = []
    for step in range(tmax):
        t = time.time(); k = max(range(5), key=lambda k: mse[k] - prop[k]['mse']); p = prop[k]
        tm = p['term']; chosen[k].append(tm); Bsel[k].append(mars.term_value(tm, {q: Qf[q][sel] for q in (tm['q'], tm.get('q2')) if q}))
        order.append(dict(step=step + 1, cls=k, term=tm, dev_mse_before=mse[k], dev_mse_after=p['mse'], dev_mse_gain=mse[k] - p['mse'],
                          total_dev_mse=sum(mse.values()) - mse[k] + p['mse']))
        mse[k] = p['mse']; prop[k] = propose(k)
        log(f'[{pool}] step {step + 1}: logit {CL[k]} += {term_str(tm)}; dev MSE gain {order[-1]["dev_mse_gain"]:.4f}, total dev MSE {order[-1]["total_dev_mse"]:.4f} ({time.time() - t:.1f} s)')
    return order


def program(order, T, Qf, Lf):
    from jetdistill import mars
    NF = len(Lf); parts = []
    for k in range(5):
        terms = [o['term'] for o in order[:T] if o['cls'] == k]
        Bf = np.stack([np.ones(NF)] + [mars.term_value(t, Qf) for t in terms], 1); co = np.linalg.lstsq(Bf, Lf[:, k], rcond=None)[0]
        parts.append(dict(cls=k, intercept=float(co[0]), terms=[dict(t, coef=float(c)) for t, c in zip(terms, co[1:])]))
    return parts


def prog_logits(parts, Q):
    from jetdistill import mars
    J = len(next(iter(Q.values())))
    return np.stack([np.full(J, p['intercept']) + sum((t['coef'] * mars.term_value(t, Q) for t in p['terms']), np.zeros(J)) for p in parts], 1)


# ------------------------------------------------------------------ readable form
def key_tex(k):
    named = {'sum_pt': r'M_0', 'H:ΣpT·r²[all]': r'M_2', 'H:√(M0·M2)': r'\sqrt{M_0 M_2}', 'X:|M1|²': r'|\vec M_1|^2', 'H:|M1|': r'|\vec M_1|',
             'X:M0·M2': r'M_0 M_2', 'X:|M1|²/M0': r'|\vec M_1|^2/M_0', 'mass': 'm', 'mass_over_sum_pt': r'm/M_0', 'mass_over_sum_pt_sq': r'(m/M_0)^2',
             'n_particles': r'N', 'girth': r'\textstyle\sum_i z_i \Delta R_i', 'girth2': r'\textstyle\sum_i z_i \Delta R_i^2'}
    if k in named: return named[k]
    m = re.fullmatch(r'H:(.*)\[(.*)\]', k)
    if m:
        body, w = m.groups(); w = {'all': 'all', 'top14': r'0\text{-}13', 'r1-63': r'1\text{-}63', 'r13-63': r'13\text{-}63', 'top39': r'0\text{-}38',
                                   'top26': r'0\text{-}25', 'top3': r'0\text{-}2', 'top11': r'0\text{-}10', 'r8-63': r'8\text{-}63'}.get(w, w)
        b = (body.replace('Σ(r−0.2)₊', r'\Sigma (r-0.2)_+').replace('Σr[r<0.165]', r'\Sigma r\,[r<0.165]').replace('ΣpT·r²', r'\Sigma p_T r^2')
             .replace('Σr²', r'\Sigma r^2').replace('ΣpT·r', r'\Sigma p_T r').replace('ΣpT', r'\Sigma p_T').replace('Σr', r'\Sigma r')
             .replace('N(pT<2,r>0.1)', r'N(p_T<2,\,r>0.1)').replace('N(pT>1)', r'N(p_T>1)').replace('N(pT<2)', r'N(p_T<2)'))
        return b + r'_{[' + w + r']}'
    if k.startswith('E:'): return r'\mathrm{' + k[2:].replace(' ', r'\ ').replace('β', r'}\beta\mathrm{').replace('_', r'\_') + '}'
    return r'\mathrm{' + k.replace('_', r'\_') + '}'


def _one_tex(q, kind, t):
    s = key_tex(q)
    if kind == 'lin': return s
    return rf'({s}-{t:.4g})_+' if kind == 'gt' else rf'({t:.4g}-{s})_+'


def term_tex(tm):
    a = _one_tex(tm['q'], tm['kind'], tm.get('t'))
    return a + (r'\,' + _one_tex(tm['q2'], tm['kind2'], tm.get('t2')) if tm.get('q2') else '')


def term_str(tm):
    one = lambda q, k, t: q if k == 'lin' else (f'max(0, {q} − {t:.4g})' if k == 'gt' else f'max(0, {t:.4g} − {q})')
    return one(tm['q'], tm['kind'], tm.get('t')) + (' × ' + one(tm['q2'], tm['kind2'], tm.get('t2')) if tm.get('q2') else '')


def prog_tex(parts):
    rows = []
    for p in parts:
        s = rf'\ell_{{{CL[p["cls"]]}}} &= {p["intercept"]:.4g}'
        for t in p['terms']:
            c = t['coef']; s += (' - ' if c < 0 else ' + ') + f'{abs(c):.4g}\\,' + term_tex(t)
        rows.append(s)
    return r'\begin{aligned}' + r' \\ '.join(rows) + r'\end{aligned}'


# ------------------------------------------------------------------ modes
def mode_fit(P, a):
    res = dict(procedure=__doc__, pools={p: v for p, v in P.pools.items()}, excluded=P.excl, prefix=a.prefix, tmax=a.tmax, programs={}, paths={})
    for pool in POOLS:
        t = time.time(); keys = P.pools[pool]
        order = fit_shared(pool, keys, P.phys['fit'], P.phys['dev'], P.logits['fit'], P.logits['dev'], a.tmax)
        res['paths'][pool] = order
        for T in TS:
            if T > a.tmax: continue
            parts = program(order, T, P.phys['fit'], P.logits['fit'])
            sc = {s: float((prog_logits(parts, P.phys[s]).argmax(1) == P.call[s]).mean()) for s in ('dev', 'test')}
            res['programs'][f'{pool}{T}'] = dict(pool=pool, T=T, n_inputs=len(keys), bits=float(T * (np.log2(len(keys)) + 16)), parts=parts,
                                                 same_call_dev=sc['dev'], same_call_test=sc['test'], tex=prog_tex(parts))
            log(f'[{pool}] T {T}: same call dev {sc["dev"] * 100:.2f} % test {sc["test"] * 100:.2f} %; terms per logit ' +
                str([len(p['terms']) for p in parts]))
        log(f'[{pool}] done in {time.time() - t:.0f} s')
        (OUT / f'{a.prefix}_programs.json').write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    log(f'wrote {OUT / (a.prefix + "_programs.json")}')


# ---- edits
def cluster2(x, alg='kt', batch=1000):          # verbatim from interp/x4_axis_models.py (N = 64)
    N = 64

    def _kin(Pq):
        pt = np.hypot(Pq[..., 0], Pq[..., 1])
        yy = 0.5 * np.log(np.maximum(Pq[..., 3] + Pq[..., 2], 1e-300) / np.maximum(Pq[..., 3] - Pq[..., 2], 1e-300))
        return pt, yy, np.arctan2(Pq[..., 1], Pq[..., 0])

    def _dist(pt_a, y_a, p_a, pt_b, y_b, p_b):
        dp = np.remainder(p_a - p_b + np.pi, 2 * np.pi) - np.pi; d = (y_a - y_b) ** 2 + dp ** 2
        return d * np.minimum(pt_a, pt_b) ** 2 if alg == 'kt' else d
    J = len(x); lab = np.full((J, N), -1, np.int8); ptAB = np.zeros((J, 2))
    for s in range(0, J, batch):
        xb = x[s:s + batch].astype(np.float64); Bn = len(xb)
        pt, eta, phi = xb[..., 0], xb[..., 1], xb[..., 2]; alive = pt > 0
        Pq = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1)
        rep = np.tile(np.arange(N), (Bn, 1))
        pc, yc, phc = _kin(Pq); pc = np.where(alive, pc, 0.0); yc = np.where(alive, yc, 0.0)
        Dm = _dist(pc[:, :, None], yc[:, :, None], phc[:, :, None], pc[:, None, :], yc[:, None, :], phc[:, None, :])
        Dm[~(alive[:, :, None] & alive[:, None, :])] = np.inf; Dm[:, np.arange(N), np.arange(N)] = np.inf
        while True:
            act = np.flatnonzero(alive.sum(1) > 2)
            if len(act) == 0: break
            f = Dm[act].reshape(len(act), -1).argmin(1); i, j = np.divmod(f, N)
            Pq[act, i] += Pq[act, j]; alive[act, j] = False
            r_ = rep[act]; rep[act] = np.where(r_ == j[:, None], i[:, None], r_)
            Dm[act, j, :] = np.inf; Dm[act, :, j] = np.inf
            pi_, yi, phi_i = _kin(Pq[act, i]); pa, ya, pha = _kin(Pq[act])
            d = _dist(pi_[:, None], yi[:, None], phi_i[:, None], pa, ya, pha)
            d[~alive[act]] = np.inf; d[np.arange(len(act)), i] = np.inf
            Dm[act, i, :] = d; Dm[act, :, i] = d
        real = xb[..., 0] > 0; A = rep[:, 0]
        other = np.where(alive & (np.arange(N)[None] != A[:, None]), np.arange(N)[None], N).min(1)
        lb = np.where(rep == A[:, None], 0, 1).astype(np.int8); lb[~real] = -1; lab[s:s + Bn] = lb
        pA = _kin(Pq[np.arange(Bn), A])[0]
        pB = np.where(other < N, _kin(Pq[np.arange(Bn), np.minimum(other, N - 1)])[0], 0.0)
        ptAB[s:s + Bn] = np.stack([pA, pB], 1)
    return lab, ptAB


def pt_sort(x):
    o = np.argsort(-x[..., 0], axis=1, kind='stable'); return np.take_along_axis(x, o[..., None], 1)


def jump_sets(xt):
    pt = xt[..., 0].astype(np.float64); lab, ptk = cluster2(xt, 'kt'); zA = ptk[:, 0] / np.maximum(ptk.sum(1), 1e-12)
    nreal = (pt > 0).sum(1); two = (zA > ZCUT) & (zA < 1 - ZCUT) & (nreal >= 2)
    ptB = np.where(lab == 1, pt, -1.0); j = ptB.argmax(1); ptj = ptB[np.arange(len(xt)), j]; has_b = ptj > 0
    new = (1 + EPS) * pt[:, 0]; frac = np.where(has_b, new / np.maximum(ptj, 1e-12) - 1, np.inf)
    cand = two & has_b & (frac <= FMAX); dpt = new - ptj
    rng = np.random.default_rng(72); kk = np.full(len(xt), -1)
    for n in np.flatnonzero(cand):
        ok = np.flatnonzero((lab[n] == 1) & (np.arange(64) != j[n]) & (pt[n] > 0) & (pt[n] + dpt[n] < pt[n, 0] * (1 - 1e-3)))
        if len(ok): kk[n] = rng.choice(ok)
    sel = np.flatnonzero(cand & (kk >= 0)); xs = xt[sel]; ar = np.arange(len(sel))
    xj = xs.copy(); xn = xs.copy()
    xj[ar, j[sel], 0] = new[sel].astype(np.float32); xn[ar, kk[sel], 0] = (xs[ar, kk[sel], 0] + dpt[sel]).astype(np.float32)
    info = dict(n_two_prong=int(two.sum()), n_cand=int(cand.sum()), n_selected=int(len(sel)), zA_quantiles=np.percentile(zA[sel], [0, 10, 50, 90, 100]).tolist() if len(sel) else None,
                dpt_GeV_quantiles=np.percentile(dpt[sel], [0, 10, 50, 90, 100]).tolist() if len(sel) else None,
                hardest_after_jump_is_j=float(np.mean(xj[ar, j[sel], 0] >= xj[..., 0].max(1))) if len(sel) else None)
    return sel, pt_sort(xs), pt_sort(xj), pt_sort(xn), info


def build_sets(P):
    from jetdistill.observables import compute
    from interp.b4_baselines_jedi import edit_shift, rotate, randdisp
    from interp.e17_axis_jedi import axis_set, reaxis, wta_selftest
    xt = P.x['test']; real = xt[..., 0] > 0; nreal = real.sum(1); Jt = len(xt)
    st = wta_selftest(); assert st['ok'], st
    t = time.time(); A, ainfo = axis_set(xt); log(f'axes (WTA, hardest) computed in {time.time() - t:.0f} s: {ainfo}')
    rng = np.random.default_rng(104); sets, J = {}, {}
    sets.update(test_rc=xt, trans=edit_shift(xt, 0.06), trans_c=randdisp(xt, 0.06 * real, 101), rot=rotate(xt, 45.0))
    for nm in ('wta', 'hardest'):
        ae, ap = A[nm]; d = np.hypot(ae, ap); th = rng.uniform(0, 2 * np.pi, Jt)
        sets[nm] = reaxis(xt, ae, ap); sets[nm + '_c'] = reaxis(xt, d * np.cos(th), d * np.sin(th))
    for nm in ('test_rc', 'trans', 'trans_c', 'rot', 'wta', 'wta_c', 'hardest', 'hardest_c'): J[nm] = np.arange(Jt)
    t = time.time(); sel, xb, xj, xn, jinfo = jump_sets(xt); log(f'axis-jump selection in {time.time() - t:.0f} s: {jinfo}')
    sets.update(jump_b=xb, jump=xj, jump_null=xn); J.update(jump_b=sel, jump=sel, jump_null=sel)
    J13 = np.flatnonzero(nreal >= 15); x13 = xt[J13]; xa = x13.copy(); xa[:, [13, 14], 1:] = x13[:, [14, 13], 1:]
    dist = np.hypot(x13[:, 13, 1] - x13[:, 14, 1], x13[:, 13, 2] - x13[:, 14, 2]); mag = np.zeros(x13.shape[:2]); mag[:, 13] = mag[:, 14] = dist
    sets.update(A13_b=x13, A13=xa, A13_c=randdisp(x13, mag, 103)); J.update(A13_b=J13, A13=J13, A13_c=J13)
    info = dict(axes=ainfo, jump=jinfo, wta_selftest=st,
                axis_disp_median=dict(wta=float(np.median(np.hypot(*A['wta']))), hardest=float(np.median(np.hypot(*A['hardest'])))))
    for nm, xx in sets.items():
        t = time.time(); q = compute(np.asarray(xx, np.float64), 64)
        ph = {k: np.asarray(v, np.float32).astype(np.float64) for k, v in q.items()}; ph.update(E.handbuilt(xx, False)); ph.update(extras(xx))
        miss = [k for k in P.keys if k not in ph]; assert not miss, f'{nm}: recomputed observables missing {miss[:5]}'
        P.phys[nm] = to_cols(ph, P.keys); P.x[nm] = xx; P.y[nm] = P.y['test'][J[nm]]; del q, ph
        P.logits[nm] = P.net.run(xx)['logits']; P.call[nm] = P.logits[nm].argmax(1)
        log(f'edit set {nm}: {len(xx)} jets; physics recomputed, network run ({time.time() - t:.0f} s)')
    Xs = np.stack([P.phys['test'][k] for k in P.keys], 1); Xr = P.phys['test_rc'].X.astype(np.float64)
    dq = np.abs(Xr - Xs).max(0) / np.maximum(np.abs(Xs).max(0), 1e-12)
    info['recompute_vs_stored_max_rel'] = float(dq.max()); info['n_keys_rel_gt_1e-4'] = int((dq > 1e-4).sum())
    info['same_call_test_rc_vs_test'] = float((P.call['test_rc'] == P.call['test']).mean())
    log(f'unedited test: recomputed vs stored physics max relative difference {dq.max():.2e}; network calls identical on {info["same_call_test_rc_vs_test"] * 100:.2f} %')
    del Xs, Xr
    return info


def mcnemar(a, b):
    from scipy.stats import binomtest
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return dict(a_only=n10, b_only=n01, p=1.0 if n10 + n01 == 0 else float(binomtest(n10, n10 + n01, 0.5).pvalue))


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (n - r) * p[i])); q[i] = run
    return q


def mode_eval(P, a):
    from interp.e11_verify import Last, chain
    from sklearn.ensemble import HistGradientBoostingClassifier
    progs = json.loads((OUT / f'{a.prefix}_programs.json').read_text())
    einfo = build_sets(P); sets = ['test', 'conf'] + sorted({s for e in EDITS for s in e[1:3]})
    calls, models = {}, []
    for name, pr in progs['programs'].items():
        calls[name] = {s: prog_logits(pr['parts'], P.phys[s]).argmax(1) for s in sets}
        models.append(dict(name=name, method=f'direct shared-budget MARS, {POOL_LABEL[pr["pool"]]}', pool=pr['pool'], dl=pr['T'], bits=pr['bits'], n_inputs=pr['n_inputs']))
        log(f'evaluated {name} on {len(sets)} sets')
    last = Last(P.net); rd = lambda pre, n: json.loads((OUT / f'{pre}_{n}.json').read_text())
    for nm, pre in (('mars30', 'e11'), ('mars16', 'b4m16')):
        t = time.time(); fb = {'P': rd(pre, 'fit_a_P'), **{s: rd(pre, f'fit_b_{s}') for s in E.STG[1:]}}
        nt = sum(len(u['terms']) for s in E.STG for u in fb[s]['units'])
        calls[nm] = {s: last.jax_logits(chain(fb, P.phys[s])['z']).argmax(1) for s in sets}
        models.append(dict(name=nm, method='nested MARS program (E11 / B4)', pool='nested', dl=nt, bits=None, n_inputs=None))
        log(f'evaluated {nm} ({nt} terms) on {len(sets)} sets ({time.time() - t:.0f} s)')
    Kg = [k for k in P.D.keys]; Xm = lambda s: np.stack([P.phys[s][k] for k in Kg], 1).astype(np.float32)
    Xf = Xm('fit')
    for L_, T_ in ((2, 1), (2, 2), (2, 3), (4, 2), (2, 5), (4, 4)):
        t = time.time(); m = HistGradientBoostingClassifier(max_leaf_nodes=L_, max_iter=T_, learning_rate=0.2, early_stopping=False, random_state=0).fit(Xf, P.call['fit'])
        try: nl = int(sum(int(p.nodes['is_leaf'].sum()) for it in m._predictors for p in it))
        except Exception: nl = int(m.n_iter_ * 5 * L_)  # noqa: BLE001
        nm = f'gbdt_L{L_}_T{T_}'; calls[nm] = {s: m.predict(Xm(s)) for s in sets}
        models.append(dict(name=nm, method='GBDT on the call (B4 settings, E11 physics set)', pool='gbdt', dl=nl, bits=float(nl * (np.log2(len(Kg)) + 16)), n_inputs=len(Kg)))
        log(f'GBDT {nm}: {nl} leaves ({time.time() - t:.0f} s)')
    del Xf
    rng = np.random.default_rng(7); bidx = {}
    for s in ['test', 'conf'] + [e[2] for e in EDITS]:
        n = len(P.y[s]); bidx.setdefault(n, rng.integers(0, n, (B, n)).astype(np.int32))
    BI = lambda s: bidx[len(P.y[s])]

    def ci(v, s):
        v = np.asarray(v, np.float32); b = v[BI(s)].mean(1); return [float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

    def edit_stats(cb, ca, yb, s):
        f = ca != cb; Wm, Zm = yb == W_, yb == Z_
        return dict(flip=ci(f, s), WtoZ=[float((ca[Wm] == Z_).mean()), float((cb[Wm] == Z_).mean())] if Wm.any() else None,
                    ZtoW=[float((ca[Zm] == W_).mean()), float((cb[Zm] == W_).mean())] if Zm.any() else None,
                    dacc=ci((ca == yb).astype(float) - (cb == yb), s), N=int(len(f)), N_W=int(Wm.sum()))

    net = dict(acc_test=ci(P.call['test'] == P.y['test'], 'test'), acc_conf=ci(P.call['conf'] == P.y['conf'], 'conf'),
               N_test=len(P.y['test']), N_conf=len(P.y['conf']), edits={})
    for name, b, af, _ in EDITS:
        net['edits'][name] = edit_stats(P.call[b], P.call[af], P.y[af], af)
        log(f'network {name}: calls changed {net["edits"][name]["flip"][0] * 100:.2f} % [{net["edits"][name]["flip"][1] * 100:.2f}, {net["edits"][name]["flip"][2] * 100:.2f}] '
            f'(N {net["edits"][name]["N"]}); W→Z {net["edits"][name]["WtoZ"]}; Δacc {net["edits"][name]["dacc"][0] * 100:+.2f}')
    log(f'network accuracy test {net["acc_test"]}, conf {net["acc_conf"]}')
    rows, pv, cells = [], [], []
    for m in models:
        C = calls[m['name']]; r = dict(m)
        r['same_call_test'] = ci(C['test'] == P.call['test'], 'test'); r['same_call_conf'] = ci(C['conf'] == P.call['conf'], 'conf')
        r['acc_test'] = ci(C['test'] == P.y['test'], 'test'); r['edits'] = {}
        for name, b, af, _ in EDITS:
            nb, na, sb, sa = P.call[b], P.call[af], C[b], C[af]; nf, sf = na != nb, sa != sb
            e = edit_stats(sb, sa, P.y[af], af)
            d = (sf.astype(np.float32) - nf)[BI(af)].mean(1); e['flip_diff'] = [float(sf.mean() - nf.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            e['mcnemar'] = mcnemar(sf, nf); ag = (nf == sf) & (~nf | (sa == na)); e['agree'] = ci(ag, af); e['agree_never_flip'] = float(1 - nf.mean())
            Wm = P.y[af] == W_
            if Wm.any():
                wz_s, wz_n = (sa[Wm] == Z_).astype(np.float32), (na[Wm] == Z_).astype(np.float32); bw = np.random.default_rng(8).integers(0, Wm.sum(), (B, int(Wm.sum())))
                dd = (wz_s - wz_n)[bw].mean(1); e['WtoZ_diff'] = [float(wz_s.mean() - wz_n.mean()), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
            ne = net['edits'][name]['flip']
            e['overlap'] = bool(e['flip'][1] <= ne[2] and ne[1] <= e['flip'][2]); e['paired_ci_has_0'] = bool(e['flip_diff'][1] <= 0 <= e['flip_diff'][2])
            e['reproduces'] = e['overlap'] and e['paired_ci_has_0']
            r['edits'][name] = e; pv.append(e['mcnemar']['p']); cells.append((len(rows), name))
        rows.append(r)
        log(f"{m['name']:14s} DL {m['dl']:5d} same call test {r['same_call_test'][0] * 100:6.2f} conf {r['same_call_conf'][0] * 100:6.2f} | " +
            ' '.join(f"{k} {v['flip'][0] * 100:.1f}{'*' if v['reproduces'] else ''}" for k, v in r['edits'].items()))
    q = holm(pv)
    for (i, name), qq in zip(cells, q): rows[i]['edits'][name]['mcnemar']['p_holm'] = float(qq)
    smallest = {}
    for pool in POOLS:
        rr = sorted([r for r in rows if r['pool'] == pool], key=lambda r: r['dl'])
        hit = next((r for r in rr if all(r['edits'][e]['reproduces'] for e in TARGET_EDITS)), None)
        smallest[pool] = None if hit is None else dict(name=hit['name'], T=hit['dl'], same_call_test=hit['same_call_test'], same_call_conf=hit['same_call_conf'],
                                                      bits=hit['bits'], tex=progs['programs'][hit['name']]['tex'],
                                                      terms=[dict(cls=CL[p['cls']], intercept=p['intercept'], terms=[dict(text=term_str(t), tex=term_tex(t), coef=t['coef']) for t in p['terms']])
                                                             for p in progs['programs'][hit['name']]['parts']])
        log(f'smallest T reproducing {TARGET_EDITS} in pool {pool}: {smallest[pool]["T"] if smallest[pool] else None}')
    # paired same-call difference: each pool's programs vs mars30
    ref = calls['mars30']['test'] == P.call['test']; paired = {}
    for r in rows:
        if r['name'] == 'mars30': continue
        sc = calls[r['name']]['test'] == P.call['test']; d = (sc.astype(np.float32) - ref)[BI('test')].mean(1)
        paired[r['name']] = dict(diff=[float(sc.mean() - ref.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], mcnemar=mcnemar(sc, ref))
    res = dict(procedure=__doc__, prefix=a.prefix, network=net, edit_info=einfo, models=rows, smallest_T=smallest, same_call_vs_mars30=paired,
               n_holm=len(pv), bootstrap=B, pools={p: len(v) for p, v in P.pools.items()}, pool_keys=P.pools, excluded=P.excl,
               splits=dict(fit=len(P.y['fit']), dev=len(P.y['dev']), test=len(P.y['test']), conf=len(P.y['conf'])))
    (OUT / f'{a.prefix}.json').write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float)); log(f'wrote {OUT / (a.prefix + ".json")}')
    np.savez_compressed(OUT / f'{a.prefix}_calls.npz', **{f'{m}|{s}': np.asarray(v, np.int8) for m, d in calls.items() for s, v in d.items()},
                        **{f'network|{s}': np.asarray(P.call[s], np.int8) for s in sets})
    plot(res)


def plot(res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = res['models']; net = res['network']['edits']
    col = dict(axis='#1c5cab', inv='#c0392b', all='#2a9d3c', gbdt='#7b4fa0')
    lab = dict(axis='direct MARS on P_axis (axis-relative + pT)', inv='direct MARS on P_inv (m, EFPs + pT)', all='direct MARS on P_axis ∪ P_inv',
               gbdt='GBDT on the call (all physics)')
    panels = [('same call with the network, test jets (%)', None), ('0.06 rad shift in Δη: calls changed (%)', 'trans'),
              ('re-axis on the WTA axis: calls changed (%)', 'wta'), ('re-axis on the hardest particle: calls changed (%)', 'hardest'),
              ('axis jump (X4 rule): calls changed (%)', 'jump'), ('control: each particle 0.06 rad, random direction: calls changed (%)', 'trans_c')]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.4)); axs = axs.ravel()
    for ax, (title, e) in zip(axs, panels):
        for pool in ('axis', 'inv', 'all', 'gbdt'):
            rr = sorted([r for r in rows if r['pool'] == pool], key=lambda r: r['dl'])
            if not rr: continue
            x = [r['dl'] for r in rr]; v = np.array([r['same_call_test'] if e is None else r['edits'][e]['flip'] for r in rr]) * 100
            ax.plot(x, v[:, 0], marker='o', ms=4.5, lw=1.6, color=col[pool], label=lab[pool], ls='-' if pool != 'gbdt' else ':')
            ax.fill_between(x, v[:, 1], v[:, 2], color=col[pool], alpha=0.15, lw=0)
        for nm, ls in (('mars30', '--'), ('mars16', ':')):
            r = next((r for r in rows if r['name'] == nm), None)
            if r is None: continue
            v = (r['same_call_test'] if e is None else r['edits'][e]['flip'])[0] * 100
            ax.axhline(v, color='#444444', lw=1.1, ls=ls); ax.text(0.99, v, f'{nm} ({r["dl"]:,} terms)', ha='right', va='bottom', fontsize=7.5, color='#444444', transform=ax.get_yaxis_transform())
        if e is not None:
            n = net[e]['flip']; ax.axhspan(n[1] * 100, n[2] * 100, color='#f2b134', alpha=0.45, lw=0)
            ax.axhline(n[0] * 100, color='#b07800', lw=1.6); ax.text(0.01, n[0] * 100, f'network {n[0] * 100:.1f} % (N {net[e]["N"]:,})', fontsize=8, color='#7a5300', va='bottom', transform=ax.get_yaxis_transform())
        ax.set_title(title, fontsize=10, loc='left'); ax.set_xscale('log'); ax.set_xticks([5, 10, 20, 30, 50, 80]); ax.set_xticklabels(['5', '10', '20', '30', '50', '80'])
        ax.set_xlabel('program size (hinge terms summed over the 5 logits; GBDT: leaves)', fontsize=8.5); ax.grid(alpha=0.3, lw=0.5); ax.tick_params(labelsize=8)
    axs[0].legend(fontsize=7.5, loc='lower right')
    fig.suptitle(f'JEDI-linear: short direct programs on axis-relative vs invariant observables (test N {res["splits"]["test"]:,}; bands 95 % jet bootstrap; '
                 'orange band = network)', fontsize=10.5, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.96)); p = OUT / f'{res["prefix"]}.png'; fig.savefig(p, dpi=130); log(f'wrote {p}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['fit', 'eval', 'all', 'plot'])
    ap.add_argument('--per-class', type=int, default=6000); ap.add_argument('--test-per-class', type=int, default=2000)
    ap.add_argument('--conf-per-class', type=int, default=2000); ap.add_argument('--prefix', default='b5'); ap.add_argument('--tmax', type=int, default=50)
    a = ap.parse_args(); log('args', vars(a))
    if a.mode == 'plot':
        plot(json.loads((OUT / f'{a.prefix}.json').read_text()))
    else:
        P = Prep(a)
        if a.mode in ('fit', 'all'): mode_fit(P, a)
        if a.mode in ('eval', 'all'): mode_eval(P, a)
    log('done')
