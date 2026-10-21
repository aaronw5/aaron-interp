"""E26w (claims A14 / A3; round 12): is total pT (S0) a specific class feature of JEDI-linear (3-feature n64) under the exact
Lorentz boost along the jet axis at fixed jet mass, or would any high-variance direction of the pooled second-round layer S do?

Sample: interp.jedi_core.sample(per_class=2000, seed=1203) = 10,000 test jets (evaluation). Principal directions, covariances and
the boost-difference second moments used to build the matched nulls are fit on a separate sample(per_class=2000, seed=1204) (unedited
jets plus their boosted copies), drawn separately from the evaluation sample (398 of 10,000 jets shared, 4 %). DAS (distributed alignment search)
M0 / M2 subspaces at S refit with X10's data, pairs and seeds exactly as E26 / E26v (sample(4000, 0), 8000 pairs, seed 10 + k).
Edit (imported from interp.h12_boost_panel, as E26): B = exact Lorentz boost along the jet momentum with |P_jet| -> lam |P_jet| and
the jet invariant mass unchanged (particle angles to the axis shrink ~ 1 / lam), lam in {0.9, 1.1}.
Metric: share of B's call changes removed by a hold = 1 - #(calls differing from the unboosted call with the hold) / #(calls
differing without it); a hold resets the named direction(s) of a layer to the unboosted value while the network sees the boosted jet.
Arms
 1. S-level holds at matched dimension k = 1, 2, 4: S0, S29, S0+S29, DAS M0 / M2 / M0+M2 (k1: 1 / 1 / 2 dims; k2: 4 dims), top-k
    principal directions of S (PC1, PC1-2, PC1-4). Nulls (all on the live S coordinates only): (a) isotropic random k-dim subspaces;
    (b) variance-matched: Q = orth(Sigma^{p/2} Z), Z Gaussian on the live coordinates, p bisected per draw so tr(Q^T Sigma Q) equals
    the variance the named hold captures (S0 at k = 1); (c) boost-shift-matched: the same with C = E[D D^T], D = S(B) - S(0), so a
    null direction moves as much under the boost as the named direction. Rank p = (1 + #null >= obs) / (1 + #null); Holm over holds.
    Paired comparisons (S0 vs PC1, S0+S29 vs PC1-2, DAS M0M2 k2 vs PC1-4): McNemar on remaining call changes, bootstrap CI of the
    difference.
 2. Every live h4 and h5 unit held alone vs isotropic random 1-dim live-h4 / live-h5 directions (Holm over units); variance-matched
    1-dim nulls for the named E26 units (h4 53, h4 39, h4 3, h5 24, h5 25, h5 1) and the top-3 units per lam.
 3. Path-specific patch a15 -> g49 -> S0 (a15 = first-layer cut linear in particle pT; g49 = context unit computed from the pooled
    first-round objects; S0 = sum pT of the 14 hardest particles). S0 set to (i) its unboosted value, (ii) S0(boosted jet, g49 at
    its unboosted value) (context route reverted), (iii) S0(unboosted jet, g49 at its boosted value) (direct route reverted), other
    S units boosted; g49 held (all routes) and g49 held with S0 kept at its boosted value; a15 held (all routes), via g only, via
    g49 only, not via g, via S0 only.
 4. Data check: training archive (train_ptetaphi_f16.npy, 620k jets, 150 particles) and the n64 test set (260k): class-conditional
    sum pT (all particles, and the 14 hardest) in 5 GeV bins of jet mass; stratified AUC (probability that a jet of class A has larger
    sum pT than a jet of class B in the same mass bin) for all 10 class pairs, 200 Poisson bootstraps, and a within-bin label-permutation
    null (200); calibration observables: particle multiplicity (known q/g discriminant) and the jet mass itself.
Statistics: 1000 Poisson bootstrap weights over jets (shared by all holds), McNemar (exact binomial) on paired call changes.
Run: ./job.sh e26w-run 1 2 'python -m interp.e26w_boost_specificity'   (--smoke N: N jets per class, small null counts)
Writes $AI_RESULTS/jedi/e26w_boost_specificity[_smoke].json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0_ = time.time()
LAMS = (0.9, 1.1); CL = ['g', 'q', 'W', 'Z', 't']


def log(*a):
    print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


from interp.e26_boost_route import Net, Ctx, pair_inputs, qci, prand  # noqa: E402
from interp.h12_boost_panel import boost, hls_prep, sortT, mcnemar, jet_obs  # noqa: E402
from interp import x10_das_jedi as X  # noqa: E402


# ------------------------------------------------------------------------------------------------ matched random subspaces
class Tilt:
    """random k-dim subspaces on the live coordinates, tilted toward the high-variance directions of a PSD matrix C:
    Q = orth(C^{p/2} Z), Z ~ N(0, 1) (live x k); p is bisected per draw so that tr(Q^T C Q) = target."""

    def __init__(self, C, live, d):
        self.live = np.asarray(live); self.d = d
        e, V = np.linalg.eigh(C[np.ix_(self.live, self.live)]); self.e = np.maximum(e, e.max() * 1e-9); self.V = V
        self.C = C

    def make(self, Z, p):
        M = self.V @ ((self.e ** (p / 2))[:, None] * (self.V.T @ Z)); Q = np.linalg.qr(M)[0]
        F = np.zeros((self.d, Z.shape[1])); F[self.live] = Q; return F

    def f(self, Q): return float(np.trace(Q.T @ self.C @ Q))

    def draw(self, k, target, rng, tol=0.01, tries=20):
        for _ in range(tries):
            Z = rng.normal(size=(len(self.live), k)); lo, hi = -8.0, 80.0
            flo, fhi = self.f(self.make(Z, lo)), self.f(self.make(Z, hi))
            if not (flo <= target <= fhi): continue
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if self.f(self.make(Z, mid)) < target: lo = mid
                else: hi = mid
            Q = self.make(Z, 0.5 * (lo + hi))
            if abs(self.f(Q) / target - 1) <= tol: return Q, 0.5 * (lo + hi)
        return None, None


def iso(live, d, k, rng):
    F = np.zeros((d, k)); F[live] = X.rand_Q(len(live), k, rng); return F


def capt(C, Q): return float(np.trace(Q.T @ C @ Q))


# ------------------------------------------------------------------------------------------------ stratified AUC (arm 4)
class StratAUC:
    """P(v_A > v_B) + 0.5 P(tie) over pairs (A, B) in the same bin; sort once, then any per-jet weights / label assignment"""

    def __init__(self, v, b):
        o = np.lexsort((v, b)); self.o = o; vs, bs = v[o], b[o]
        new = np.ones(len(o), bool); new[1:] = (vs[1:] != vs[:-1]) | (bs[1:] != bs[:-1]); self.g = np.cumsum(new) - 1
        self.ng = int(self.g[-1]) + 1; self.bs = bs; self.gbin = bs[new]; self.nb = int(b.max()) + 1

    def __call__(self, wA, wB):
        wA, wB = wA[self.o], wB[self.o]
        gB = np.bincount(self.g, wB, self.ng); exB = np.cumsum(gB) - gB
        binB = np.bincount(self.bs, wB, self.nb); binA = np.bincount(self.bs, wA, self.nb); bstart = np.cumsum(binB) - binB
        below = exB - bstart[self.gbin]; sc = below[self.g] + 0.5 * gB[self.g]
        num = (wA * sc).sum(); den = (binA * binB).sum(); return float(num / den) if den > 0 else np.nan


def jet_feats(x):
    """x (J, L, 3) pT, deta, dphi -> mass, sum pT, sum pT of the 14 hardest, multiplicity"""
    x = np.asarray(x, np.float64); pt = x[..., 0]
    top = -np.sort(-pt, 1)[:, :14]
    return dict(m=X.mass(x), sumpt=pt.sum(1), sumpt14=top.sum(1), n=(pt > 0).sum(1).astype(np.float64))


def data_check(sm, NBd, NPd):
    import h5py
    from jetdistill.config import DATA, RESULTS
    out = {}
    srcs = []
    Xt = np.load(Path(DATA) / 'train_ptetaphi_f16.npy', mmap_mode='r')
    with h5py.File(Path(DATA) / 'train_meta.h5', 'r') as f: yt = np.asarray(f['label'][:])
    srcs.append(('train', Xt, yt))
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); srcs.append(('test_n64', d['x'], np.asarray(d['y'])))
    for nm, Xa, ya in srcs:
        t = time.time(); J = len(ya) if not sm else min(len(ya), 20000); F = {k: np.empty(J) for k in ('m', 'sumpt', 'sumpt14', 'n')}
        for i in range(0, J, 20000):
            f = jet_feats(np.asarray(Xa[i:i + 20000][: J - i], np.float32))
            for k in F: F[k][i:i + len(f['m'])] = f[k]
        y = ya[:J].astype(int); edges = np.arange(0, 305, 5.0); b = np.clip(np.digitize(F['m'], edges) - 1, 0, len(edges) - 1)
        R = out[nm] = dict(N=int(J), labels=np.bincount(y, minlength=5).tolist(), bin_GeV=5, m_range_GeV=[0, 300, 'last bin = m >= 300'])
        R['class_means'] = {CL[c]: dict(m=float(F['m'][y == c].mean()), sumpt=float(F['sumpt'][y == c].mean()), sumpt14=float(F['sumpt14'][y == c].mean()),
                                         n=float(F['n'][y == c].mean())) for c in range(5)}
        rng = np.random.default_rng(1203)
        R['auc'] = {}
        for ca in range(5):
            for cb in range(ca + 1, 5):
                sel = (y == ca) | (y == cb); lab = (y[sel] == ca).astype(np.float64); bb = b[sel]
                # bins with both classes present (at least 20 jets each)
                cA = np.bincount(bb, lab, len(edges)); cB = np.bincount(bb, 1 - lab, len(edges)); okb = (cA >= 20) & (cB >= 20)
                keep = okb[bb]; lab, bb = lab[keep], bb[keep]; P = {}
                for feat in ('sumpt', 'sumpt14', 'n', 'm'):
                    v = F[feat][sel][keep]; A = StratAUC(v, bb); one = np.ones(len(v))
                    a0 = A(lab, 1 - lab)
                    W = rng.poisson(1.0, (NBd, len(v))).astype(np.float64); bt = np.array([A(w * lab, w * (1 - lab)) for w in W])
                    # within-bin label permutation null
                    P[feat] = dict(auc=a0, ci=qci(bt))
                    if feat in ('sumpt', 'sumpt14'):   # within-bin label permutation null (two-sided in |AUC - 0.5|)
                        st = np.argsort(bb, kind='stable'); pn = []
                        for _ in range(NPd):
                            lp = np.empty_like(lab); lp[st] = lab[np.argsort(bb + rng.random(len(bb)), kind='stable')]; pn.append(A(lp, 1 - lp))
                        pn = np.asarray(pn); P[feat].update(perm_q=np.quantile(pn, [.025, .975]).tolist(),
                                                             p_perm=float((1 + (np.abs(pn - 0.5) >= abs(a0 - 0.5)).sum()) / (1 + len(pn))))
                    del one
                # per-bin AUC of sum pT (for the figure) in the bins that hold both classes
                pb = {}
                for bi in np.flatnonzero(okb):
                    s = bb == bi
                    if s.sum() < 200: continue
                    A1 = StratAUC(F['sumpt'][sel][keep][s], np.zeros(s.sum(), int)); pb[f'{edges[bi]:.0f}'] = dict(auc=A1(lab[s], 1 - lab[s]), nA=int(lab[s].sum()), nB=int((1 - lab[s]).sum()))
                R['auc'][f'{CL[ca]}_vs_{CL[cb]}'] = dict(n_jets=int(keep.sum()), n_bins=int(okb.sum()), **P, per_bin_sumpt=pb)
        log(f'data {nm}: N {J}; ' + '; '.join(f'{k} sumpt {v["sumpt"]["auc"]:.3f} {[round(c, 3) for c in v["sumpt"]["ci"]]} n {v["n"]["auc"]:.3f} m {v["m"]["auc"]:.3f}'
                                              for k, v in R['auc'].items()) + f' ({time.time() - t:.0f} s)')
    return out


# ------------------------------------------------------------------------------------------------ main
def main(a):
    from interp.jedi_core import sample
    from interp.h1_irc_panel import hls_table
    sm = a.smoke; pc = sm or 2000; seed = a.seed
    NB = 200 if sm else 1000; NN = 40 if sm else 1000; NH = 100 if sm else 4000; NHM = 40 if sm else 500; NBd = 20 if sm else 200; NPd = 20 if sm else 200
    OUT.mkdir(parents=True, exist_ok=True); JS = OUT / f'e26w_boost_specificity{"_smoke" if sm else ""}.json'
    R = dict(sample=f'interp.jedi_core.sample(per_class={pc}, seed={seed})', fit_sample=f'interp.jedi_core.sample(per_class={pc}, seed={seed + 1})',
             lams=list(LAMS), n_boot=NB, n_null_S=NN, n_null_h=NH, n_null_h_matched=NHM)

    def save(): JS.write_text(json.dumps(X.clean(R), indent=1))

    if not a.skip_data:
        R['data'] = data_check(sm, NBd, NPd); save()

    x, y, _ = sample(pc, seed); J = len(y); T0 = hls_table(x); x0 = hls_prep(sortT(T0))
    net = Net(); R0 = net.fwd(x0); c0 = R0['logits'].argmax(1)
    R['N'] = J; R['acc'] = float((c0 == y).mean())
    R['tail_check'] = net.jn.check(x0[:1000], {k: v[:1000] for k, v in R0.items()})
    log(f'N {J} acc {R["acc"]:.4f} (expected 0.806-0.813) tail {R["tail_check"]}')
    Wb = np.random.default_rng(seed + 11).poisson(1.0, (NB, J)).astype(np.float32)

    # fit sample: covariance of S / h4 / h5 and boost-difference second moments
    xf, yf, _ = sample(pc, seed + 1); Tf = hls_table(xf); xf0 = hls_prep(sortT(Tf)); Rf0 = net.fwd(xf0, keep=('S', 'h4', 'h5'))
    R['fit_overlap_jets'] = int(len(np.intersect1d(X.indices(pc, seed), X.indices(pc, seed + 1)))) if not sm else None
    COV = {st: np.cov(Rf0[st], rowvar=False) for st in ('S', 'h4', 'h5')}; CD = {}
    for lam in LAMS:
        Uf, _ = boost(Tf, lam); RfB = net.fwd(pair_inputs(Tf, Uf)[0], keep=('S',)); D = RfB['S'] - Rf0['S']; CD[lam] = D.T @ D / len(D)
    _, _, Vt = np.linalg.svd(Rf0['S'] - Rf0['S'].mean(0), full_matrices=False)
    del xf, Tf, xf0

    # DAS refit at S (as E26 / E26v)
    t = time.time(); pcF = 300 if sm else 4000; xF, yF, _ = sample(pcF, 0); mF = X.moments(xF); NF = len(yF)
    npair = 800 if sm else 8000; steps = 100 if sm else 800
    r2 = np.random.default_rng(2); fb = r2.integers(0, NF, npair); fs = r2.integers(0, NF, npair)
    while (fb == fs).any(): fs[fb == fs] = r2.integers(0, NF, (fb == fs).sum())
    RF = net.jn.reps(xF, keep=('S', 'logits')); baseF = RF['logits'].argmax(1)[fb]
    labF = {V: net.jn.calls(X.do_edit(xF[fb], V, (mF[0][fb], mF[1][fb]), (mF[0][fs], mF[1][fs]))) for V in ('M0', 'M2')}
    Q = {}; R['das'] = {}
    for k in (1, 2):
        for V in ('M0', 'M2'):
            q, info = X.das_fit(net.jn.tail('S'), RF['S'][fb], RF['S'][fs], labF[V], baseF, k, seed=10 + k, steps=steps, rg=net.jn.rng_['S'])
            Q[(V, k)] = q; R['das'][f'S_{V}_k{k}'] = info
        Q[('M0M2', k)] = np.linalg.qr(np.concatenate([Q[('M0', k)], Q[('M2', k)]], 1))[0]
    del xF, RF
    log(f'DAS ({time.time() - t:.0f} s): {R["das"]}'); save()

    # live coordinates (unedited or boosted at either lam)
    RB = {}; XB = {}
    for lam in LAMS:
        U, _ = boost(T0, lam); xB, x0B, o = pair_inputs(T0, U); XB[lam] = (U, xB, x0B); RB[lam] = net.fwd(xB)
    live = {st: np.flatnonzero(np.maximum(R0[st].std(0), np.maximum(*[RB[l][st].std(0) for l in LAMS])) > 1e-9) for st in ('S', 'h4', 'h5')}
    R['live'] = {k: v.tolist() for k, v in live.items()}
    eye = np.eye(64)
    NAMED = {'S0': (eye[:, [0]], 1), 'S29': (eye[:, [29]], 1), 'S0+S29': (eye[:, [0, 29]], 2),
             'DAS_M0_k1': (Q[('M0', 1)], 1), 'DAS_M2_k1': (Q[('M2', 1)], 1), 'DAS_M0M2_k1': (Q[('M0M2', 1)], 2), 'DAS_M0M2_k2': (Q[('M0M2', 2)], 4),
             'PC1': (Vt[:1].T, 1), 'PC1-2': (Vt[:2].T, 2), 'PC1-4': (Vt[:4].T, 4)}
    MATCHED = ('S0', 'S29', 'S0+S29', 'DAS_M0_k1', 'DAS_M2_k1', 'DAS_M0M2_k1', 'DAS_M0M2_k2')
    ev = np.linalg.eigvalsh(COV['S'])[::-1]
    R['S_geometry'] = dict(var_S0=float(COV['S'][0, 0]), var_S29=float(COV['S'][29, 29]), pc_var_top6=ev[:6].tolist(),
                           var_S0_rank_among_units=int((np.diag(COV['S']) > COV['S'][0, 0]).sum()) + 1,
                           cos2_S0_PC=[float(Vt[i, 0] ** 2) for i in range(6)], cos2_S29_PC=[float(Vt[i, 29] ** 2) for i in range(6)],
                           captured_var={nm: capt(COV['S'], q) for nm, (q, _) in NAMED.items()},
                           dead_energy_PC=[float((Vt[i, np.setdiff1d(np.arange(64), live['S'])] ** 2).sum()) for i in range(4)])
    log(f'S geometry: {R["S_geometry"]}')

    ex_jet = None; R['lam'] = {}
    for lam in LAMS:
        t = time.time(); E = R['lam'][str(lam)] = {}; U, xB, x0B = XB[lam]; Rb = RB[lam]
        ctx = Ctx(R0['logits'], Rb['logits'], y, Wb, np.ones(J, bool)); chB = ctx.chB
        O0, O1 = jet_obs(T0), jet_obs(U)
        E['B'] = dict(changed=float(chB.mean()), n_changed=int(chB.sum()), ci=qci((Wb @ chB.astype(np.float32)) / Wb.sum(1)),
                      dm_rel_absmax=float(np.max(np.abs(O1['m'] / np.maximum(O0['m'], 1e-9) - 1))), dsumpt_rel_mean=float(np.mean(O1['sumpt'] / O0['sumpt'] - 1)),
                      dS0_rel_mean=float(np.mean(Rb['S'][:, 0] / np.maximum(R0['S'][:, 0], 1e-9) - 1)))
        lo_, hi_ = net.jn.rng_['S']; lo2, hi2 = np.minimum(np.minimum(lo_, Rb['S']), R0['S']), np.maximum(np.maximum(hi_, Rb['S']), R0['S'])
        Ssub = lambda Qm: net.tail('S', np.clip(Rb['S'] + ((R0['S'] - Rb['S']) @ Qm) @ Qm.T, lo2, hi2))[0]
        # ---- arm 1: named holds
        H = E['holds'] = {}; CH = {}
        for nm, (q, k) in NAMED.items():
            LH = Ssub(q); H[nm] = ctx.stats(LH); H[nm]['dim'] = k; H[nm]['captured_var'] = capt(COV['S'], q); H[nm]['boost_shift_ms'] = capt(CD[lam], q)
            CH[nm] = LH.argmax(1) != c0
        H['S_all'] = ctx.stats(net.tail('S', R0['S'])[0])
        # paired comparisons between holds
        E['paired'] = {}
        for A_, B_ in (('S0', 'PC1'), ('S0+S29', 'PC1-2'), ('DAS_M0M2_k1', 'PC1-2'), ('DAS_M0M2_k2', 'PC1-4'), ('S0', 'DAS_M0_k1'), ('S0+S29', 'DAS_M0M2_k1')):
            ra = 1 - (Wb @ CH[A_].astype(np.float32)) / np.maximum(ctx.bB, 1); rb = 1 - (Wb @ CH[B_].astype(np.float32)) / np.maximum(ctx.bB, 1)
            E['paired'][f'{A_}_vs_{B_}'] = dict(diff=H[A_]['removed'] - H[B_]['removed'], ci=qci(ra - rb), mcnemar_remaining=mcnemar(CH[A_], CH[B_]))
        log(f'lam {lam}: B {E["B"]}; holds ' + '; '.join(f'{k} {v["removed"]:.3f}' for k, v in H.items()) + f'; paired {E["paired"]}')
        # ---- arm 1: nulls
        rq = np.random.default_rng(seed + 100 + int(lam * 10)); NU = E['nulls_S'] = {}
        for k in (1, 2, 4):
            v = [ctx.rem(Ssub(iso(live['S'], 64, k, rq))) for _ in range(NN)]
            NU[f'iso_k{k}'] = dict(n=len(v), q=np.quantile(v, [.025, .5, .95, .975, 1]).tolist(), mean=float(np.mean(v)), vals=v)
        TV = Tilt(COV['S'], live['S'], 64); TD = Tilt(CD[lam], live['S'], 64)
        for nm in MATCHED:
            q, k = NAMED[nm]
            for kind, Tl, C_ in (('var', TV, COV['S']), ('shift', TD, CD[lam])):
                if kind == 'shift' and nm not in ('S0', 'S0+S29', 'DAS_M0M2_k1', 'DAS_M0_k1'): continue
                tg = capt(C_, q); v, ps, c2, fail = [], [], [], 0
                while len(v) < NN and fail < 5 * NN:
                    Qn, p = Tl.draw(k, tg, rq)
                    if Qn is None: fail += 1; continue
                    v.append(ctx.rem(Ssub(Qn))); ps.append(p); c2.append(float(np.linalg.norm(Vt[:k] @ Qn) ** 2 / k))
                key = f'{kind}_{nm}'
                NU[key] = dict(target=tg, n=len(v), failed=fail, q=np.quantile(v, [.025, .5, .95, .975, 1]).tolist() if v else None,
                               mean=float(np.mean(v)) if v else None, tilt_p_q=np.quantile(ps, [.05, .5, .95]).tolist() if ps else None,
                               overlap_topk_PC_mean=float(np.mean(c2)) if c2 else None, vals=v)
                if v: H[nm][f'p_{kind}'] = prand(v, H[nm]['removed'])
            H[nm]['p_iso'] = prand(NU[f'iso_k{k}']['vals'], H[nm]['removed'])
        for nm in ('PC1', 'PC1-4'):
            H[nm]['p_iso'] = prand(NU[f'iso_k{NAMED[nm][1]}']['vals'], H[nm]['removed'])
        log(f'lam {lam}: S nulls ' + '; '.join(f'{k} q {np.round(v["q"], 3).tolist() if v["q"] else None} fail {v.get("failed")} ov {v.get("overlap_topk_PC_mean")}' for k, v in NU.items())
            + '; p ' + str({k: {kk: v[kk] for kk in v if kk.startswith('p_')} for k, v in H.items()}) + f' ({time.time() - t:.0f} s)')
        save()

        # ---- arm 2: single h4 / h5 holds vs live nulls
        E['h_units'] = {}
        for st in ('h4', 'h5'):
            lo4, hi4 = net.jn.rng_[st]; lb, hb = np.minimum(np.minimum(lo4, Rb[st]), R0[st]), np.maximum(np.maximum(hi4, Rb[st]), R0[st])
            d = Rb[st].shape[1]; hsub = lambda Qm: net.tail(st, np.clip(Rb[st] + ((R0[st] - Rb[st]) @ Qm) @ Qm.T, lb, hb))[0]
            obs = {}
            for u in live[st]:
                h = Rb[st].copy(); h[:, u] = R0[st][:, u]; obs[int(u)] = ctx.rem(net.tail(st, h)[0])
            rh = np.random.default_rng(seed + 500 + int(lam * 10) + (st == 'h5')); vn = np.array([ctx.rem(hsub(iso(live[st], d, 1, rh))) for _ in range(NH)])
            ps = {u: prand(vn, r) for u, r in obs.items()}; hp = X.holm(list(ps.values()))
            top = sorted(obs, key=lambda u: -obs[u])[:3]; named = [u for u in ((53, 39, 3) if st == 'h4' else (24, 25, 1)) if u in obs]
            Tl = Tilt(COV[st], live[st], d); mt = {}
            for u in sorted(set(top) | set(named)):
                e1 = np.zeros((d, 1)); e1[u] = 1; tg = capt(COV[st], e1); v = []; fail = 0
                while len(v) < NHM and fail < 5 * NHM:
                    Qn, _ = Tl.draw(1, tg, rh)
                    if Qn is None: fail += 1; continue
                    v.append(ctx.rem(hsub(Qn)))
                mt[str(u)] = dict(removed=obs[u], n=len(v), failed=fail, q=np.quantile(v, [.025, .5, .95, .975, 1]).tolist() if v else None,
                                  p_var=prand(v, obs[u]) if v else None)
            E['h_units'][st] = dict(n_live=len(obs), removed={str(u): r for u, r in obs.items()}, p_iso={str(u): p for u, p in ps.items()},
                                    holm_iso={str(u): float(h_) for u, h_ in zip(ps, hp)}, n_holm_005=int((hp < 0.05).sum()),
                                    passing=[int(u) for u, h_ in zip(ps, hp) if h_ < 0.05], top3=[int(u) for u in top],
                                    null_iso=dict(n=len(vn), q=np.quantile(vn, [.025, .5, .95, .975, .999, 1]).tolist()), var_matched=mt)
            if mt:
                hv = X.holm([v['p_var'] for v in mt.values()])
                for (u, v), h_ in zip(mt.items(), hv): v['holm_var'] = float(h_)
            log(f'lam {lam} {st}: top {[(u, round(obs[u], 3)) for u in top]}; iso null q {np.round(E["h_units"][st]["null_iso"]["q"], 3).tolist()}; Holm pass {E["h_units"][st]["passing"]}; matched {mt}')
        save()

        # ---- arm 3: path-specific patch a15 -> g49 -> S0
        P_ = E['path'] = {}; m49 = np.zeros(64); m49[49] = 1; m15 = np.zeros(64); m15[15] = 1
        Rg0 = net.fwd(xB, {'g': (m49, R0['g'])}, keep=('logits', 'S', 'g'))          # boosted jet, g49 unboosted
        Rx0gB = net.fwd(x0, {'g': (m49, Rb['g'])}, keep=('logits', 'S'))             # unboosted jet, g49 boosted
        S0_0, S0_B, S0_dir, S0_ctx = R0['S'][:, 0], Rb['S'][:, 0], Rg0['S'][:, 0], Rx0gB['S'][:, 0]

        def setS0(v): h = Rb['S'].copy(); h[:, 0] = v; return net.tail('S', h)[0]
        P_['S0_full'] = ctx.stats(setS0(S0_0))
        P_['S0_context_route_reverted'] = ctx.stats(setS0(S0_dir))
        P_['S0_direct_route_reverted'] = ctx.stats(setS0(S0_ctx))
        P_['g49_all_routes'] = ctx.stats(Rg0['logits'])
        h = Rg0['S'].copy(); h[:, 0] = S0_B; P_['g49_with_S0_kept_boosted'] = ctx.stats(net.tail('S', h)[0])
        P_['g49_patch_on_unboosted_changed'] = float((Rx0gB['logits'].argmax(1) != c0).mean())
        a0B = net.a_of(x0B)
        Ra = net.fwd(xB, {'a': (m15, a0B)}, keep=('logits', 'S', 'g')); P_['a15_all_routes'] = ctx.stats(Ra['logits'])
        P_['a15_via_g_only'] = ctx.stats(net.fwd(xB, {'g': (np.ones(64), Ra['g'])}, keep=('logits',))['logits'])
        P_['a15_via_g49_only'] = ctx.stats(net.fwd(xB, {'g': (m49, Ra['g'])}, keep=('logits',))['logits'])
        P_['a15_not_via_g'] = ctx.stats(net.fwd(xB, {'a': (m15, a0B), 'g': (np.ones(64), Rb['g'])}, keep=('logits',))['logits'])
        P_['a15_via_S0_only'] = ctx.stats(setS0(Ra['S'][:, 0]))
        del a0B
        dS = S0_B - S0_0
        P_['dS0'] = dict(total_mean=float(dS.mean()), S0_0_mean=float(S0_0.mean()), context_part_mean=float((S0_B - S0_dir).mean()),
                         direct_part_mean=float((S0_B - S0_ctx).mean()), interaction_mean=float((dS - (S0_B - S0_dir) - (S0_B - S0_ctx)).mean()),
                         context_share=float((S0_B - S0_dir).mean() / dS.mean()), direct_share=float((S0_B - S0_ctx).mean() / dS.mean()),
                         dg49_mean=float((Rb['g'][:, 49] - R0['g'][:, 49]).mean()), g49_0_mean=float(R0['g'][:, 49].mean()),
                         dS0_under_a15_hold_mean=float((Ra['S'][:, 0] - S0_0).mean()), dg49_under_a15_hold_mean=float((Ra['g'][:, 49] - R0['g'][:, 49]).mean()))
        ps = {k: v['mcnemar']['p'] for k, v in P_.items() if isinstance(v, dict) and 'mcnemar' in v}; hp = X.holm(list(ps.values()))
        for k_, h_ in zip(ps, hp): P_[k_]['holm_mcnemar'] = float(h_)
        log(f'lam {lam} path: ' + '; '.join(f'{k} {v["removed"]:.3f} {np.round(v["removed_ci"], 3).tolist()}' for k, v in P_.items() if isinstance(v, dict) and 'removed' in v) + f'; dS0 {P_["dS0"]}')

        # example jet for the figure (lam 0.9): a W jet whose call changes under B and returns under the S0 hold
        if ex_jet is None:
            cS0 = Ssub(eye[:, [0]]).argmax(1); cand = np.flatnonzero((y == 2) & chB & (cS0 == c0) & (c0 == 2))
            if len(cand):
                i = int(cand[0]); n_ = int((T0['pt'][i] > 0).sum())
                ex_jet = dict(index=i, lam=lam, true='W', call0=CL[c0[i]], callB=CL[Rb['logits'][i].argmax()], callS0hold=CL[cS0[i]],
                              m0=float(O0['m'][i]), mB=float(O1['m'][i]), sumpt0=float(O0['sumpt'][i]), sumptB=float(O1['sumpt'][i]),
                              S0_0=float(R0['S'][i, 0]), S0_B=float(Rb['S'][i, 0]), n=n_,
                              p0=x0[i, :n_].tolist(), pB=xB[i, :int((xB[i, :, 0] > 0).sum())].tolist())
                R['example_jet'] = ex_jet
        save()

    # Holm over the S-level holds (both lam) for each null family
    R['holm_S'] = {}
    for kind in ('iso', 'var', 'shift'):
        ks = [(l, nm) for l in LAMS for nm, v in R['lam'][str(l)]['holds'].items() if f'p_{kind}' in v]
        hp = X.holm([R['lam'][str(l)]['holds'][nm][f'p_{kind}'] for l, nm in ks])
        R['holm_S'][kind] = {f'{l}_{nm}': float(h_) for (l, nm), h_ in zip(ks, hp)}
        for (l, nm), h_ in zip(ks, hp): R['lam'][str(l)]['holds'][nm][f'holm_{kind}'] = float(h_)
    log(f'Holm S: {R["holm_S"]}')
    save(); log(f'wrote {JS}; E26w DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=1203); ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--skip-data', action='store_true')
    main(ap.parse_args())
