"""E34 (new claim A17, exploratory; A3, A15): which jet property does JEDI-linear (3-feature n64, official, fixed point) use at the
gluon / light-quark boundary: particle count N, girth, pT^D or jet mass, and through which named objects?

Candidates (computed on the 64-slot input the network sees; r = sqrt(deta^2 + dphi^2) to the jet axis):
  N      = number of real particles (pT > 0)
  girth  = sum pT r / sum pT  (= M1 / M0)
  pTD    = sqrt(sum pT^2) / sum pT
  mass   = invariant mass of the massless constituents (pT, eta = deta, phi = dphi; exact four-vectors)
Margin = logit_g - logit_q (5-class logits; = log p_g / p_q).
Samples: main = sample(5000, 3401) restricted to g and q jets (10,000); rep = sample(5000, 3402) restricted to g and q.
Edits (each moves one candidate by about +5 % and keeps the other three within 1 %; jets where the construction is infeasible or a
held candidate moves by > 1 % are excluded, the fraction is reported):
  Ecount   the k = max(1, round(0.05 N)) softest particles are each split into two halves of equal pT at the same (deta, dphi)
           (exact collinear split; N + k <= 64 required). M0, M1, M2, girth and mass are exactly unchanged; pTD falls slightly.
  Egirth   radial power map r -> c r^gamma (gamma < 1) of every particle, azimuth kept, with c chosen per jet so that M2 = sum pT r^2
           is unchanged; gamma solved per jet (bisection) for girth x 1.05. pT values, hence N, M0, pTD, unchanged.
  EptD     pT transfer delta from the second-hardest to the hardest particle at fixed positions, delta solved for pTD x 1.05 (M0, N
           fixed); the change of M1 = delta (r1 - r2) is undone by a radial move of one other particle (the one with the largest
           pT r among ranks >= 2), so girth is unchanged. EptDm: the reverse transfer (pTD x 0.95), as a sign check.
  Emass    radial power map r -> c r^gamma (gamma > 1) with c fixing M1 (girth), gamma solved for mass x 1.05.
  Edil     angular dilation (all deta, dphi x s) solved for mass x 1.05 (reference: girth also rises 5 %; not in the Holm family).
Nulls (same particles moved, no candidate changed):
  Egirth, Emass, Edil: each moved particle is rotated about the jet axis by an angle +-|dr_i| / r_i (random sign), i.e. the same
           displacement length along the circle instead of along the radius (N, pT, girth, M2 unchanged).
  Ecount:  the k softest particles are rotated about the axis by uniform random azimuths (N unchanged).
  EptD(m): the compensating particle is rotated by its arc-matched angle (the pT transfer itself has no candidate-neutral analogue
           with positions fixed; a pT swap is the only transfer that keeps sum pT^2).
  untrained JEDI-linear (Jedi(untrained=True, seed 0)): the same edits and nulls.
References: MLPs (sklearn, 2 x 32 ReLU) trained on g vs q jets disjoint from both samples (10,000 + 10,000 from full_test.npz not
  drawn by sample(5000, 3401) or sample(5000, 3402)), on each candidate alone (log N, girth, pTD, log mass) and on all four; scored
  on the same edits (MLP log-odds g:q). Ratio = (JEDI margin change, edit - null) / (MLP margin change, edit - null).
Route holds (edit run with an element kept at its unedited value; share removed = 1 - sum dmargin_held / sum dmargin_edit):
  S19 (soft multiplicity), S30 (core r sum), S1, S0, S19+S30, S19+S30+S1+S0, g (all 64 context units), a32 (soft pT cut, per slot).
  Nulls: every other single live S (18), all 171 live-S pairs, 200 random live-S 4-sets.
Statistics: 1000 multinomial bootstrap resamples over jets; paired sign-flip permutation test (edit - null, 10,000 permutations);
  McNemar on paired call changes (edit vs null); Holm over the four primary edits (Ecount, Egirth, EptD, Emass).
Run: ./job.sh e34-run 'python -m interp.e34_quark_gluon_jedi'   (--smoke K: K jets per class)
Writes $AI_RESULTS/jedi/e34_quark_gluon[_smoke].json.
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
from interp.e28w_split_route_jedi import Net, Boot, binom_p, mcnemar, holm, qci, nreal, resort, B_FWD, LIVE_S  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0_ = time.time()
CL = ['g', 'q', 'W', 'Z', 't']
CANDS = ('N', 'girth', 'pTD', 'mass')
PRIMARY = ('Ecount', 'Egirth', 'EptD', 'Emass')
EDITS = ('Ecount', 'Egirth', 'EptD', 'EptDm', 'Emass', 'Edil')
TARGET = dict(Ecount='N', Egirth='girth', EptD='pTD', EptDm='pTD', Emass='mass', Edil='mass')
HOLDS = {'S19': [19], 'S30': [30], 'S1': [1], 'S0': [0], 'S19+S30': [19, 30], 'S19+S30+S1+S0': [19, 30, 1, 0]}


def log(*a): print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


# ------------------------------------------------------------------ jet quantities
def coords(X):
    X = np.asarray(X, np.float64); return X[:, :, 0], X[:, :, 1], X[:, :, 2]


def mass_of(pt, de, dp):
    px, py = (pt * np.cos(dp)).sum(1), (pt * np.sin(dp)).sum(1)
    pz, E = (pt * np.sinh(de)).sum(1), (pt * np.cosh(de)).sum(1)
    return np.sqrt(np.maximum(E ** 2 - px ** 2 - py ** 2 - pz ** 2, 0))


def cands(X):
    pt, de, dp = coords(X); r = np.hypot(de, dp); M0 = pt.sum(1)
    return dict(N=(pt > 0).sum(1).astype(np.float64), girth=(pt * r).sum(1) / M0, pTD=np.sqrt((pt ** 2).sum(1)) / M0,
                mass=mass_of(pt, de, dp), M0=M0, M2=(pt * r * r).sum(1))


def place(X, rnew, rows):
    """set radii of every particle to rnew (J, 64), azimuth kept, for jets in rows"""
    pt, de, dp = coords(X); r = np.hypot(de, dp); f = np.where(r > 0, rnew / np.where(r > 0, r, 1), 1.0)
    Y = X.copy(); Y[rows, :, 1] = (de * f)[rows].astype(np.float32); Y[rows, :, 2] = (dp * f)[rows].astype(np.float32)
    return Y


def bisect(fn, lo, hi, target, it=60):
    """vectorised bisection of fn(gamma) = target on [lo, hi] per jet; ok where the bracket holds"""
    flo, fhi = fn(lo) - target, fn(hi) - target; ok = np.isfinite(flo) & np.isfinite(fhi) & (flo * fhi < 0)
    a, b, fa = lo.copy(), hi.copy(), flo.copy()
    for _ in range(it):
        m = 0.5 * (a + b); fm = fn(m) - target; same = np.sign(fm) == np.sign(fa)
        a = np.where(same, m, a); fa = np.where(same, fm, fa); b = np.where(same, b, m)
    return 0.5 * (a + b), ok


# ------------------------------------------------------------------ edits: return Y, ok, disp (J, 64 radial displacement) and extra
def e_count(X, rng):
    J = len(X); N = nreal(X); k = np.maximum(1, np.rint(0.05 * N)).astype(int); ok = (N >= 2) & (N + k <= 64)
    Y = X.copy(); Yn = X.copy()
    for t in range(int(k.max())):
        rows = np.flatnonzero(ok & (k > t)); src = N[rows] - k[rows] + t; dst = N[rows] + t
        p = X[rows, src, 0].astype(np.float64); h1 = (0.5 * p).astype(np.float32)
        Y[rows, src, 0] = h1; Y[rows, dst, 0] = (p - h1.astype(np.float64)).astype(np.float32)
        Y[rows, dst, 1] = X[rows, src, 1]; Y[rows, dst, 2] = X[rows, src, 2]
        de, dp = X[rows, src, 1].astype(np.float64), X[rows, src, 2].astype(np.float64); r = np.hypot(de, dp)
        th = np.arctan2(de, dp) + rng.uniform(0, 2 * np.pi, len(rows))
        Yn[rows, src, 2] = (r * np.cos(th)).astype(np.float32); Yn[rows, src, 1] = (r * np.sin(th)).astype(np.float32)
    Y, _ = resort(Y)
    return Y, ok, Yn, dict(k=k)


def arc_null(X, rnew, rows, rng):
    pt, de, dp = coords(X); r = np.hypot(de, dp); d = np.abs(rnew - r)
    th = np.where(r > 0, d / np.where(r > 0, r, 1), 0) * rng.choice([-1.0, 1.0], size=r.shape)
    th = np.clip(th, -np.pi, np.pi); ph = np.arctan2(de, dp) + th
    Y = X.copy(); Y[rows, :, 2] = (r * np.cos(ph))[rows].astype(np.float32); Y[rows, :, 1] = (r * np.sin(ph))[rows].astype(np.float32)
    return Y


def e_power(X, rng, mode):
    pt, de, dp = coords(X); r = np.hypot(de, dp); M0 = pt.sum(1); M1 = (pt * r).sum(1); M2 = (pt * r * r).sum(1)
    m0 = mass_of(pt, de, dp); J = len(X); f = np.where(r > 0, 1 / np.where(r > 0, r, 1), 0)
    ph_c, ph_s = de * f, dp * f       # unit vector components (sin, cos of azimuth) per particle

    def radii(gm):
        rg = np.where(r > 0, r ** gm[:, None], 0.0)
        if mode == 'girth': c = np.sqrt(M2 / np.maximum((pt * rg * rg).sum(1), 1e-300))
        else: c = M1 / np.maximum((pt * rg).sum(1), 1e-300)
        return c[:, None] * rg
    if mode == 'girth':
        fn = lambda gm: (pt * radii(gm)).sum(1) / M0 / np.maximum(M1 / M0, 1e-300)
        gm, ok = bisect(fn, np.full(J, 0.05), np.ones(J), 1.05)
    elif mode == 'mass':
        def fn(gm):
            rn = radii(gm); return mass_of(pt, rn * ph_c, rn * ph_s) / np.maximum(m0, 1e-300)
        gm, ok = bisect(fn, np.ones(J), np.full(J, 4.0), 1.05)
    else:   # dilation
        fn = lambda s: mass_of(pt, de * s[:, None], dp * s[:, None]) / np.maximum(m0, 1e-300)
        gm, ok = bisect(fn, np.ones(J), np.full(J, 1.5), 1.05)
    ok &= (m0 > 0) & (nreal(X) >= 2)
    rnew = (r * gm[:, None]) if mode == 'dil' else radii(gm)
    rows = np.flatnonzero(ok); Y = place(X, rnew, rows); Yn = arc_null(X, rnew, rows, rng)
    return Y, ok, Yn, dict(gamma=gm)


def e_ptd(X, rng, f):
    pt, de, dp = coords(X); r = np.hypot(de, dp); N = nreal(X); J = len(X)
    p1, p2 = pt[:, 0], pt[:, 1]; S2 = (pt ** 2).sum(1); dT = (f * f - 1) * S2
    disc = (p1 - p2) ** 2 + 2 * dT; delta = 0.5 * (-(p1 - p2) + np.sqrt(np.maximum(disc, 0)))
    ok = (N >= 3) & (disc >= 0) & (p2 - delta > 0.05 * p2) & (p1 + delta > 0)
    dM1 = delta * (r[:, 0] - r[:, 1])
    w = pt * r; w[:, :2] = -1; w[pt <= 0] = -1; jc = w.argmax(1); pj = pt[np.arange(J), jc]; rj = r[np.arange(J), jc]
    rjn = rj - dM1 / np.maximum(pj, 1e-300); ok &= (rj > 0) & (rjn > 0) & (rjn < 0.8)
    rows = np.flatnonzero(ok); Y = X.copy()
    Y[rows, 0, 0] = (p1 + delta)[rows].astype(np.float32); Y[rows, 1, 0] = (p2 - delta)[rows].astype(np.float32)
    rnew = r.copy(); rnew[rows, jc[rows]] = rjn[rows]
    Y = place(Y, rnew, rows); Yn = arc_null(X, rnew, rows, rng)
    Y, _ = resort(Y)
    return Y, ok, Yn, dict(delta=delta, comp_slot=jc)


def make_edit(name, X, rng):
    if name == 'Ecount': return e_count(X, rng)
    if name == 'Egirth': return e_power(X, rng, 'girth')
    if name == 'Emass': return e_power(X, rng, 'mass')
    if name == 'Edil': return e_power(X, rng, 'dil')
    if name == 'EptD': return e_ptd(X, rng, 1.05)
    if name == 'EptDm': return e_ptd(X, rng, 0.95)


# ------------------------------------------------------------------ statistics helpers
def perm_p(d, n_perm=10000, seed=0):
    """paired sign-flip permutation test of mean(d) = 0 (two-sided)"""
    rng = np.random.default_rng(seed); d = np.asarray(d, np.float64); obs = abs(d.mean()); cnt = 0; done = 0
    while done < n_perm:
        b = min(500, n_perm - done); s = rng.choice([-1.0, 1.0], size=(b, len(d)))
        cnt += int((np.abs(s @ d / len(d)) >= obs).sum()); done += b
    return float((1 + cnt) / (1 + n_perm))


def bratio(bt, a, b, sel):
    """bootstrap distribution of sum(a) / sum(b) over jets (no clamp on the denominator; Boot.ratio clamps at 1 for counts)"""
    W = bt.W[:, sel].astype(np.float64); den = W @ np.asarray(b, np.float64)[sel]
    return (W @ np.asarray(a, np.float64)[sel]) / np.where(np.abs(den) > 1e-12, den, np.nan)


def flows(c0, ce, sel):
    F = np.zeros((5, 5), int); np.add.at(F, (c0[sel], ce[sel]), 1)
    return dict(g_to_q=int(F[0, 1]), q_to_g=int(F[1, 0]), p_sign=binom_p(int(F[0, 1]), int(F[0, 1] + F[1, 0])),
                to_other=int(F[:2, 2:].sum()), from_other=int(F[2:, :2].sum()), flow=F.tolist())


def mlp_feats(C):
    return np.stack([np.log(C['N']), C['girth'], C['pTD'], np.log(np.maximum(C['mass'], 1e-3))], 1)


def train_mlps(Xtr, ytr):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    F = mlp_feats(cands(Xtr)); t = (ytr == 0).astype(int); models = {}
    for nm, cols in (('MLP_N', [0]), ('MLP_girth', [1]), ('MLP_ptD', [2]), ('MLP_m', [3]), ('MLP_all', [0, 1, 2, 3])):
        sc = StandardScaler().fit(F[:, cols])
        m = MLPClassifier(hidden_layer_sizes=(32, 32), alpha=1e-4, max_iter=400, early_stopping=True, random_state=0)
        m.fit(sc.transform(F[:, cols]), t); models[nm] = (cols, sc, m)
        log(f'  {nm} trained: {m.n_iter_} iterations, train acc {m.score(sc.transform(F[:, cols]), t):.4f}')
    return models


def mlp_margin(models, X):
    F = mlp_feats(cands(X)); out = {}
    for nm, (cols, sc, m) in models.items():
        p = np.clip(m.predict_proba(sc.transform(F[:, cols])), 1e-7, 1 - 1e-7); out[nm] = np.log(p[:, 1]) - np.log(p[:, 0])
    return out


# ------------------------------------------------------------------ one sample
def analyse(net, unt, models, X0, ys, tag, rng, B, scan, smoke):
    J = len(X0); bt = Boot(J, B, int(rng.integers(1 << 30)))
    R0 = net.fwd(X0, keep=('logits', 'S', 'g', 'a'), a_units=[32]); L0 = R0['logits']; c0 = L0.argmax(1); mg0 = L0[:, 0] - L0[:, 1]
    lt = net.tail(R0['S']); tail_ok = float((lt.argmax(1) == c0).mean()); tail_dev = float(np.abs(lt - L0).max())
    C0 = cands(X0); M0m = mlp_margin(models, X0)
    res = dict(n=J, class_counts=np.bincount(ys, minlength=5).tolist(), acc_5class_on_gq=float((c0 == ys).mean()),
               gq_acc_margin_sign=float(((mg0 > 0) == (ys == 0)).mean()),
               calls_on_gq=np.bincount(c0, minlength=5).tolist(), tail_check=dict(calls_equal=tail_ok, max_abs_logit_diff=tail_dev),
               cand_means={c: {CL[k]: float(C0[c][ys == k].mean()) for k in (0, 1)} for c in CANDS},
               cand_sd={c: float(C0[c].std()) for c in CANDS},
               mlp_gq_acc={nm: float(((v > 0) == (ys == 0)).mean()) for nm, v in M0m.items()})
    from sklearn.metrics import roc_auc_score
    res['auc_g_vs_q'] = dict(JEDI=float(roc_auc_score(ys == 0, mg0)), **{nm: float(roc_auc_score(ys == 0, v)) for nm, v in M0m.items()},
                             **{f'raw_{c}': float(roc_auc_score(ys == 0, C0[c])) for c in CANDS})
    log(f'[{tag}] {J} g/q jets; 5-class acc on g/q {res["acc_5class_on_gq"]:.4f}; g-q margin sign acc {res["gq_acc_margin_sign"]:.4f}; '
        f'tail check {tail_ok:.5f} {tail_dev:.1e}; AUC {res["auc_g_vs_q"]}')
    U0 = unt.fwd(X0, keep=('logits',))['logits'] if unt is not None else None
    edits = {}; pj = dict(y=ys, c0=c0)
    for name in EDITS:
        Y, ok0, Yn, extra = make_edit(name, X0, rng)
        C1, Cn = cands(Y), cands(Yn); tgt = TARGET[name]
        rel = {c: (C1[c] - C0[c]) / np.maximum(np.abs(C0[c]), 1e-12) for c in CANDS}
        reln = {c: (Cn[c] - C0[c]) / np.maximum(np.abs(C0[c]), 1e-12) for c in CANDS}
        iso = np.ones(J, bool)
        for c in CANDS:
            if c != tgt and not (name == 'Edil' and c == 'girth'): iso &= np.abs(rel[c]) <= 0.01
        ok = ok0 & iso; sel = np.flatnonzero(ok)
        R1 = net.fwd(Y, keep=('logits', 'S')); Rn = net.fwd(Yn, keep=('logits',))
        dm = R1['logits'][:, 0] - R1['logits'][:, 1] - mg0; dn = Rn['logits'][:, 0] - Rn['logits'][:, 1] - mg0
        ce, cn = R1['logits'].argmax(1), Rn['logits'].argmax(1); chE, chN = (ce != c0) & ok, (cn != c0) & ok
        d = (dm - dn)[sel]; dt = 100 * rel[tgt]
        r = dict(target=tgt, n=len(sel), applicable_frac=float(ok0.mean()), iso_frac_of_applicable=float(iso[ok0].mean()) if ok0.any() else None,
                 n_by_class={CL[k]: int((ys[sel] == k).sum()) for k in (0, 1)},
                 cand_rel_change={c: dict(mean=float(rel[c][sel].mean()), absmax=float(np.abs(rel[c][sel]).max())) for c in CANDS},
                 null_cand_rel_change={c: dict(mean=float(reln[c][sel].mean()), absmax=float(np.abs(reln[c][sel]).max())) for c in CANDS},
                 dmargin_edit=dict(mean=float(dm[sel].mean()), ci=qci(bt.mean(dm, sel))),
                 dmargin_null=dict(mean=float(dn[sel].mean()), ci=qci(bt.mean(dn, sel))),
                 edit_minus_null=dict(mean=float(d.mean()), ci=qci(bt.mean(dm - dn, sel)), perm_p=perm_p(d, 2000 if smoke else 10000)),
                 by_class={CL[k]: dict(n=int((ys[sel] == k).sum()), edit=float(dm[sel][ys[sel] == k].mean()), null=float(dn[sel][ys[sel] == k].mean()))
                           for k in (0, 1)},
                 slope_per_pct=dict(mean=float(dm[sel].sum() / dt[sel].sum()), ci=qci(bratio(bt, dm, dt, sel)),
                                    null_corrected=float(d.sum() / dt[sel].sum()), null_corrected_ci=qci(bratio(bt, dm - dn, dt, sel))),
                 slope_per_sd=dict(mean=float(d.sum() / (rel[tgt][sel] * C0[tgt][sel]).sum() * res['cand_sd'][tgt]),
                                   ci=qci(res['cand_sd'][tgt] * bratio(bt, dm - dn, rel[tgt] * C0[tgt], sel))),
                 calls_changed_edit=dict(rate=float(chE[sel].mean()), ci=qci(bt.mean(chE, sel))),
                 calls_changed_null=dict(rate=float(chN[sel].mean()), ci=qci(bt.mean(chN, sel))),
                 mcnemar_edit_vs_null=mcnemar(chE[sel], chN[sel]),
                 flows_edit=flows(c0, ce, sel), flows_null=flows(c0, cn, sel),
                 extra={k: float(np.median(v[sel])) for k, v in extra.items()})
        # MLP references
        Me, Mn = mlp_margin(models, Y), mlp_margin(models, Yn); ref = {}
        for nm in models:
            de_, dn_ = Me[nm] - M0m[nm], Mn[nm] - M0m[nm]
            ref[nm] = dict(edit=float(de_[sel].mean()), null=float(dn_[sel].mean()), edit_minus_null=float((de_ - dn_)[sel].mean()),
                           ci=qci(bt.mean(de_ - dn_, sel)),
                           jedi_over_mlp=float(d.sum() / (de_ - dn_)[sel].sum()) if abs((de_ - dn_)[sel].sum()) > 1e-9 else None,
                           jedi_over_mlp_ci=qci(bratio(bt, dm - dn, de_ - dn_, sel)))
        r['mlp'] = ref
        # untrained
        if unt is not None:
            Ue, Un = unt.fwd(Y, keep=('logits',))['logits'], unt.fwd(Yn, keep=('logits',))['logits']
            due = (Ue[:, 0] - Ue[:, 1]) - (U0[:, 0] - U0[:, 1]); dun = (Un[:, 0] - Un[:, 1]) - (U0[:, 0] - U0[:, 1])
            r['untrained'] = dict(edit=float(due[sel].mean()), null=float(dun[sel].mean()), edit_minus_null=float((due - dun)[sel].mean()),
                                  ci=qci(bt.mean(due - dun, sel)), margin_sd=float((U0[:, 0] - U0[:, 1]).std()),
                                  trained_margin_sd=float(mg0.std()),
                                  edit_minus_null_over_sd=float((due - dun)[sel].mean() / (U0[:, 0] - U0[:, 1]).std()),
                                  ci_over_sd=[v / float((U0[:, 0] - U0[:, 1]).std()) for v in qci(bt.mean(due - dun, sel))],
                                  trained_edit_minus_null_over_sd=float(d.mean() / mg0.std()),
                                  trained_ci_over_sd=[v / float(mg0.std()) for v in qci(bt.mean(dm - dn, sel))])
        # route holds
        sdm = dm[sel].sum(); H = {}

        def share_of(dh):
            return dict(removed=float(1 - dh[sel].sum() / sdm) if abs(sdm) > 1e-9 else None,
                        ci=qci(1 - bt.mean(dh, sel) / np.where(np.abs(bt.mean(dm, sel)) > 1e-12, bt.mean(dm, sel), np.nan)))

        def hold_S(units):
            m = np.zeros(64, np.float32); m[units] = 1; Lh = net.tail(R1['S'], m, R0['S']); return Lh[:, 0] - Lh[:, 1] - mg0
        for hn, units in HOLDS.items():
            H[hn] = share_of(hold_S(units))
        gv = R0['g']
        Lg = net.fwd(Y, lambda i, n: {'g': (np.ones(64, np.float32), np.concatenate(
            [gv[i:i + n], np.zeros((B_FWD - n, 64))]).astype(np.float32).reshape(B_FWD, 1, 64))}, keep=('logits',))['logits']
        H['g'] = share_of(Lg[:, 0] - Lg[:, 1] - mg0)
        a0 = R0['a'][:, :, 0]; m32 = np.zeros(64, np.float32); m32[32] = 1

        def f32(i, n):
            V = np.zeros((B_FWD, 64, 64), np.float32); V[:n, :, 32] = a0[i:i + n]; return {'a': (m32, V)}
        La = net.fwd(Y, f32, keep=('logits',))['logits']; H['a32'] = share_of(La[:, 0] - La[:, 1] - mg0)
        if scan and abs(sdm) > 1e-9:
            sh = lambda units: float(1 - hold_S(units)[sel].sum() / sdm)
            singles = {f'S{u}': sh([u]) for u in LIVE_S}
            pairs = [sh([a_, b_]) for i_, a_ in enumerate(LIVE_S) for b_ in LIVE_S[i_ + 1:]]
            quads = [sh(list(rng.choice(LIVE_S, 4, replace=False))) for _ in range(25 if smoke else 200)]
            nl = dict(singles=singles, pair_q95=float(np.quantile(pairs, .95)), pair_mean=float(np.mean(pairs)), n_pairs=len(pairs),
                      quad_q95=float(np.quantile(quads, .95)), quad_mean=float(np.mean(quads)), n_quads=len(quads))
            for hn, units in HOLDS.items():
                v = H[hn]['removed']
                if len(units) == 1: pool = [singles[f'S{u}'] for u in LIVE_S]
                elif len(units) == 2: pool = pairs
                else: pool = quads
                H[hn]['rank_p'] = float((1 + sum(x >= v for x in pool)) / (1 + len(pool))) if len(units) != 1 else \
                    float(sum(x >= v for x in pool) / len(pool))
            H['null'] = nl
        r['holds'] = H
        edits[name] = r
        pj[f'dm_{name}'] = np.where(ok, dm, np.nan); pj[f'dn_{name}'] = np.where(ok, dn, np.nan)
        pj[f'ce_{name}'] = np.where(ok, ce, -1); pj[f'cn_{name}'] = np.where(ok, cn, -1)
        for nm_ in models: pj[f'mlp_{nm_}_{name}'] = np.where(ok, (Me[nm_] - M0m[nm_]) - (Mn[nm_] - M0m[nm_]), np.nan)
        mr = r['mlp']
        log(f'[{tag}] {name}: n {len(sel)} (applicable {ok0.mean():.3f}, iso {r["iso_frac_of_applicable"]}); '
            f'target {tgt} {r["cand_rel_change"][tgt]["mean"]:+.4f}; others ' +
            ', '.join(f'{c} {r["cand_rel_change"][c]["mean"]:+.4f}/{r["cand_rel_change"][c]["absmax"]:.4f}' for c in CANDS if c != tgt) +
            f'; dmargin edit {r["dmargin_edit"]["mean"]:+.4f} null {r["dmargin_null"]["mean"]:+.4f} diff {r["edit_minus_null"]["mean"]:+.4f} '
            f'{r["edit_minus_null"]["ci"]} p {r["edit_minus_null"]["perm_p"]:.2g}; calls {r["calls_changed_edit"]["rate"]:.4f} vs '
            f'{r["calls_changed_null"]["rate"]:.4f} (McN p {r["mcnemar_edit_vs_null"]["p"]:.2g}); g->q {r["flows_edit"]["g_to_q"]} q->g '
            f'{r["flows_edit"]["q_to_g"]}; MLP ' + ', '.join(f'{k} {v["edit_minus_null"]:+.4f}' for k, v in mr.items()) +
            (f'; untrained diff {r["untrained"]["edit_minus_null"]:+.4f}' if 'untrained' in r else '') +
            '; holds ' + ', '.join(f'{k} {v["removed"]:.3f}' if v.get('removed') is not None else f'{k} n/a'
                                   for k, v in H.items() if k != 'null'))
    ps = holm([edits[e]['edit_minus_null']['perm_p'] for e in PRIMARY])
    for e, p in zip(PRIMARY, ps): edits[e]['edit_minus_null']['p_holm'] = p
    psm = holm([edits[e]['mcnemar_edit_vs_null']['p'] for e in PRIMARY])
    for e, p in zip(PRIMARY, psm): edits[e]['mcnemar_edit_vs_null']['p_holm'] = p
    res['edits'] = edits
    return res, pj


def sample_idx(y, per_class, seed):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in range(5)]))


def main(a):
    from interp.jedi_core import sample
    from jetdistill.config import RESULTS
    sm = a.smoke; B = 200 if sm else 1000; tag = 'e34_quark_gluon' + ('_smoke' if sm else '')
    OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(3401)
    net = Net(); log('network loaded', net.n.path)
    unt = Net(untrained=True, seed=0); log('untrained network loaded (seed 0)')
    # ---- full main sample: accuracy check
    x, y, _ = sample(per_class=5000, seed=3401); x = np.asarray(x, np.float32)
    acc_all = None
    if not sm:
        Lall = net.fwd(x, keep=('logits',))['logits']; acc_all = float((Lall.argmax(1) == y).mean())
        log(f'sample(5000, 3401): {len(x)} jets, 5-class accuracy {acc_all:.4f}')
    # ---- MLP training jets: g/q jets of full_test not drawn by either sample
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); yall = np.asarray(d['y'])
    used = np.union1d(sample_idx(yall, 5000, 3401), sample_idx(yall, 5000, 3402))
    pool = [np.setdiff1d(np.flatnonzero(yall == c), used) for c in (0, 1)]
    ntr = 1000 if sm else 10000; rtr = np.random.default_rng(3403)
    itr = np.sort(np.concatenate([rtr.choice(p, ntr, replace=False) for p in pool]))
    Xtr = np.asarray(d['x'][itr], np.float32); ytr = yall[itr]
    log(f'MLP training jets: {len(itr)} (g {int((ytr == 0).sum())}, q {int((ytr == 1).sum())}); overlap with samples {len(np.intersect1d(itr, used))}')
    models = train_mlps(Xtr, ytr)
    res = dict(seeds=dict(main=3401, rep=3402, mlp_train=3403), n_boot=B, acc_full_sample_5class=acc_all, n_mlp_train=int(len(itr)),
               edits_primary=list(PRIMARY), holds=HOLDS, live_S=LIVE_S)
    for nm, seed in (('main', 3401), ('rep', 3402)):
        if nm == 'rep':
            x, y, _ = sample(per_class=5000, seed=3402); x = np.asarray(x, np.float32)
        sel = np.flatnonzero(y <= 1)
        if sm: sel = np.sort(np.concatenate([np.flatnonzero(y == c)[:sm] for c in (0, 1)]))
        res[nm], pj = analyse(net, unt if nm == 'main' else None, models, x[sel], y[sel], nm, rng, B, nm == 'main', sm)
        if nm == 'main':
            # example gluon jet: N <= 40, applicable for every edit
            okall = np.all([np.isfinite(pj[f'dm_{e}']) for e in PRIMARY], 0) & (y[sel] == 0) & (nreal(x[sel]) <= 40) & (nreal(x[sel]) >= 20)
            js = np.flatnonzero(okall & (pj['c0'] == 0))
            if not len(js): js = np.flatnonzero(okall)
            if len(js):
                j = int(js[0]); xj = x[sel][j:j + 1]; ex = dict(index_in_gq=j, N=int(nreal(xj)[0]), jets={})
                rr = np.random.default_rng(5)
                for e in ('orig',) + PRIMARY:
                    xe = xj if e == 'orig' else make_edit(e, xj, rr)[0]
                    L = net.fwd(xe, keep=('logits',))['logits'][0]; n_ = int(nreal(xe)[0]); C = cands(xe)
                    ex['jets'][e] = dict(pt=xe[0, :n_, 0].tolist(), deta=xe[0, :n_, 1].tolist(), dphi=xe[0, :n_, 2].tolist(),
                                         margin=float(L[0] - L[1]), call=CL[int(L.argmax())], **{c: float(C[c][0]) for c in CANDS})
                res['example'] = ex; log('example jet', j, {e: (v['call'], round(v['margin'], 3)) for e, v in ex['jets'].items()})
        np.savez_compressed(OUT / f'{tag}_perjet_{nm}.npz', **pj)
        (OUT / f'{tag}.json').write_text(json.dumps(res, indent=1)); log(f'{nm} written')
    res['wall_s'] = time.time() - T0_
    (OUT / f'{tag}.json').write_text(json.dumps(res, indent=1))
    log(f'wrote {OUT / tag}.json; E34 DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', type=int, default=0); main(ap.parse_args())
