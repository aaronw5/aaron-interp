"""E34 verification (skeptical re-run + added controls).  ./job.sh e34-verify 'python -m interp.e34_verify'

1. Re-run of E34's analyse() unchanged (same edits, nulls, MLP references, holds, scans) on a fresh sample: the 10,000 g/q jets
   of sample(5000, 3421); MLP references retrained on 20,000 g/q jets of full_test disjoint from that sample (seed 3423).
2. Added edits (same analyse() machinery, so the same nulls / MLP / untrained / hold read-outs apply):
   Egirthm  girth x 0.95 at fixed M2 (power map with gamma > 1): sign/linearity check of Egirth.
   Emassm   mass x 0.95 at fixed M1 (power map with gamma < 1): sign/linearity check of Emass.
   Emrot    mass x 1.05 by rotating ONE particle (the largest pT*r among ranks >= 1) about the jet axis at fixed r and pT.
            N, pT, every r (so girth, M2, pTD and every radial sum) are exactly unchanged; only the azimuthal arrangement,
            hence the invariant mass, changes.  Null = no change (the edit itself is the candidate-neutral control for every
            radial quantity).  Prediction if the network reads invariant mass: JEDI / MLP_m ratio close to Emass's (1.02).
            Prediction if it reads only radial (fixed-axis) sums: response near 0.
   Emrotm   the same rotation for mass x 0.95.  Only Emrotm is run: with the axis at the pT centroid, m^2 ~ M0 M2 - |sum pT r_vec|^2
            is already maximal at fixed radii, so no azimuthal rearrangement raises the mass (smoke e34-vsmoke: 0 feasible jets).
   Ecin     count +5 % by collinear splits of the k softest particles with r < 0.2 (instead of the k softest overall).
   Ecout    the same with r >= 0.2.  A pure count reading predicts the same response per added particle; the S1 / S30
            reading (sums over r > 0.2 / empty-slot terms) predicts Ecout >> Ecin.
3. A second untrained network (seed 1) on the primary edits, Edil, Emrot, Ecin, Ecout.
Writes $AI_RESULTS/jedi/e34_verify.json only.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import interp.e34_quark_gluon_jedi as E
from interp.e28w_split_route_jedi import Net, nreal, resort, qci

RES = Path(os.environ.get('AI_RESULTS', Path(__file__).resolve().parents[1] / 'results')); OUT = RES / 'jedi'
T0_ = time.time()
SEED, SEED_MLP = 3421, 3423


def log(*a): print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


def e_power_t(X, rng, mode, target):
    """E.e_power with an arbitrary target (girth or mass ratio); brackets chosen by direction"""
    pt, de, dp = E.coords(X); r = np.hypot(de, dp); M1 = (pt * r).sum(1); M2 = (pt * r * r).sum(1); M0 = pt.sum(1)
    m0 = E.mass_of(pt, de, dp); J = len(X); f = np.where(r > 0, 1 / np.where(r > 0, r, 1), 0); ph_c, ph_s = de * f, dp * f

    def radii(gm):
        rg = np.where(r > 0, r ** gm[:, None], 0.0)
        c = np.sqrt(M2 / np.maximum((pt * rg * rg).sum(1), 1e-300)) if mode == 'girth' else M1 / np.maximum((pt * rg).sum(1), 1e-300)
        return c[:, None] * rg
    if mode == 'girth':
        fn = lambda gm: (pt * radii(gm)).sum(1) / np.maximum(M1, 1e-300)
        lo, hi = (np.full(J, 0.05), np.ones(J)) if target > 1 else (np.ones(J), np.full(J, 4.0))
    else:
        fn = lambda gm: E.mass_of(pt, radii(gm) * ph_c, radii(gm) * ph_s) / np.maximum(m0, 1e-300)
        lo, hi = (np.ones(J), np.full(J, 4.0)) if target > 1 else (np.full(J, 0.05), np.ones(J))
    gm, ok = E.bisect(fn, lo, hi, target); ok &= (m0 > 0) & (nreal(X) >= 2) & (M0 > 0)
    rnew = radii(gm); rows = np.flatnonzero(ok)
    return E.place(X, rnew, rows), ok, E.arc_null(X, rnew, rows, rng), dict(gamma=gm)


def e_mrot(X, rng, target):
    """rotate one particle (largest pT*r among ranks >= 1) about the jet axis, |theta| minimal, for mass x target"""
    pt, de, dp = E.coords(X); r = np.hypot(de, dp); J = len(X); N = nreal(X); m0 = E.mass_of(pt, de, dp)
    w = pt * r; w[:, 0] = -1; w[pt <= 0] = -1; j = w.argmax(1); a = np.arange(J)
    pj, rj = pt[a, j], r[a, j]; ph0 = np.arctan2(de[a, j], dp[a, j])
    Px, Py = (pt * np.cos(dp)).sum(1), (pt * np.sin(dp)).sum(1); Pz, PE = (pt * np.sinh(de)).sum(1), (pt * np.cosh(de)).sum(1)
    Rx, Ry = Px - pj * np.cos(dp[a, j]), Py - pj * np.sin(dp[a, j])
    Rz, RE = Pz - pj * np.sinh(de[a, j]), PE - pj * np.cosh(de[a, j])

    def ratio(th):   # th (J,) or (J, G)
        th = np.asarray(th); sh = (J,) + (1,) * (th.ndim - 1)
        ph = ph0.reshape(sh) + th; e_, p_ = rj.reshape(sh) * np.sin(ph), rj.reshape(sh) * np.cos(ph); q = pj.reshape(sh)
        x_, y_ = Rx.reshape(sh) + q * np.cos(p_), Ry.reshape(sh) + q * np.sin(p_)
        z_, E_ = Rz.reshape(sh) + q * np.sinh(e_), RE.reshape(sh) + q * np.cosh(e_)
        return np.sqrt(np.maximum(E_ ** 2 - x_ ** 2 - y_ ** 2 - z_ ** 2, 0)) / np.maximum(m0, 1e-300).reshape(sh)
    G = np.linspace(0, np.pi, 181)[1:]; best = np.full(J, np.inf); lo_b = np.zeros(J); hi_b = np.zeros(J)
    for s in (1.0, -1.0):
        R = ratio(s * G[None, :] * np.ones((J, 1)))
        hit = (R >= target) if target > 1 else (R <= target)
        has = hit.any(1); i = np.where(has, hit.argmax(1), 0)
        thi = G[i]; tlo = np.where(i > 0, G[np.maximum(i - 1, 0)], 0.0)
        upd = has & (thi < np.abs(best)); best = np.where(upd, s * thi, best)
        lo_b = np.where(upd, s * tlo, lo_b); hi_b = np.where(upd, s * thi, hi_b)
    ok = np.isfinite(best) & (N >= 2) & (m0 > 0) & (rj > 0)
    lo, hi = lo_b.copy(), hi_b.copy()
    for _ in range(60):   # f(lo) on the near side of target, f(hi) past it
        m = 0.5 * (lo + hi); fm = ratio(m) - target; past = (fm >= 0) if target > 1 else (fm <= 0)
        hi = np.where(past, m, hi); lo = np.where(past, lo, m)
    th = 0.5 * (lo + hi); ph = ph0 + th
    Y = X.copy(); rows = np.flatnonzero(ok)
    Y[rows, j[rows], 1] = (rj * np.sin(ph))[rows].astype(np.float32); Y[rows, j[rows], 2] = (rj * np.cos(ph))[rows].astype(np.float32)
    return Y, ok, X.copy(), dict(theta=np.abs(th), rot_rank=j.astype(float), rot_r=rj)


def e_count_region(X, rng, region):
    """E.e_count, but the k split particles are the k softest with r < 0.2 ('in') or r >= 0.2 ('out')"""
    pt, de, dp = E.coords(X); r = np.hypot(de, dp); J = len(X); N = nreal(X); k = np.maximum(1, np.rint(0.05 * N)).astype(int)
    elig = (pt > 0) & ((r < 0.2) if region == 'in' else (r >= 0.2))
    sc = np.where(elig, np.arange(64)[None, :], -1); order = np.argsort(-sc, axis=1, kind='stable')   # softest eligible first
    ok = (N >= 2) & (N + k <= 64) & (elig.sum(1) >= k)
    Y = X.copy(); Yn = X.copy()
    for t in range(int(k.max())):
        rows = np.flatnonzero(ok & (k > t)); src = order[rows, t]; dst = N[rows] + t
        p = X[rows, src, 0].astype(np.float64); h1 = (0.5 * p).astype(np.float32)
        Y[rows, src, 0] = h1; Y[rows, dst, 0] = (p - h1.astype(np.float64)).astype(np.float32)
        Y[rows, dst, 1] = X[rows, src, 1]; Y[rows, dst, 2] = X[rows, src, 2]
        d1, d2 = X[rows, src, 1].astype(np.float64), X[rows, src, 2].astype(np.float64); rr = np.hypot(d1, d2)
        th = np.arctan2(d1, d2) + rng.uniform(0, 2 * np.pi, len(rows))
        Yn[rows, src, 2] = (rr * np.cos(th)).astype(np.float32); Yn[rows, src, 1] = (rr * np.sin(th)).astype(np.float32)
    Y, _ = resort(Y)
    return Y, ok, Yn, dict(k=k)


ORIG_MAKE = E.make_edit
CACHE_FOR = ('Ecount', 'Egirth', 'EptD', 'Emass', 'Edil', 'Emrotm', 'Ecin', 'Ecout'); CACHE = {}


def make_edit(name, X, rng):
    if name == 'Egirthm': out = e_power_t(X, rng, 'girth', 0.95)
    elif name == 'Emassm': out = e_power_t(X, rng, 'mass', 0.95)
    elif name == 'Emrot': out = e_mrot(X, rng, 1.05)
    elif name == 'Emrotm': out = e_mrot(X, rng, 0.95)
    elif name == 'Ecin': out = e_count_region(X, rng, 'in')
    elif name == 'Ecout': out = e_count_region(X, rng, 'out')
    else: out = ORIG_MAKE(name, X, rng)
    if name in CACHE_FOR: CACHE[name] = (out[0], out[1], out[2])
    return out


def untrained_check(unt, X0, names):
    U0 = unt.fwd(X0, keep=('logits',))['logits']; m0 = U0[:, 0] - U0[:, 1]; sd = float(m0.std()); C0 = E.cands(X0); out = {}
    rng = np.random.default_rng(11)
    for nm in names:
        Y, ok0, Yn = CACHE[nm]; tgt = E.TARGET[nm]; C1 = E.cands(Y); iso = np.ones(len(X0), bool)
        for c in E.CANDS:
            if c != tgt and not (nm == 'Edil' and c == 'girth'):
                iso &= np.abs((C1[c] - C0[c]) / np.maximum(np.abs(C0[c]), 1e-12)) <= 0.01
        sel = np.flatnonzero(ok0 & iso)
        Ue, Un = unt.fwd(Y, keep=('logits',))['logits'], unt.fwd(Yn, keep=('logits',))['logits']
        d = ((Ue[:, 0] - Ue[:, 1]) - (Un[:, 0] - Un[:, 1]))[sel]
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)])
        out[nm] = dict(n=len(sel), edit_minus_null=float(d.mean()), ci=qci(bs), margin_sd=sd, over_sd=float(d.mean() / sd),
                       ci_over_sd=[v / sd for v in qci(bs)], perm_p=E.perm_p(d, 10000))
        log(f'untrained seed 1 {nm}: n {len(sel)} diff/sd {d.mean() / sd:+.4f} {[round(v / sd, 4) for v in qci(bs)]}')
    return out


def main(a):
    from interp.jedi_core import sample
    from jetdistill.config import RESULTS
    E.make_edit = make_edit
    E.EDITS = E.EDITS + ('Egirthm', 'Emassm', 'Emrotm', 'Ecin', 'Ecout')   # Emrot (mass up at fixed radii) is infeasible: see docstring
    E.TARGET.update(Egirthm='girth', Emassm='mass', Emrot='mass', Emrotm='mass', Ecin='N', Ecout='N')
    sm = a.smoke; B = 200 if sm else 1000; OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED)
    net = Net(); unt = Net(untrained=True, seed=0); log('networks loaded')
    x, y, _ = sample(per_class=5000, seed=SEED); x = np.asarray(x, np.float32)
    acc = float((net.fwd(x, keep=('logits',))['logits'].argmax(1) == y).mean()); log(f'sample(5000, {SEED}) 5-class accuracy {acc:.4f}')
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); yall = np.asarray(d['y'])
    used = E.sample_idx(yall, 5000, SEED); pool = [np.setdiff1d(np.flatnonzero(yall == c), used) for c in (0, 1)]
    ntr = 1000 if sm else 10000; rtr = np.random.default_rng(SEED_MLP)
    itr = np.sort(np.concatenate([rtr.choice(p, ntr, replace=False) for p in pool]))
    models = E.train_mlps(np.asarray(d['x'][itr], np.float32), yall[itr])
    sel = np.flatnonzero(y <= 1)
    if sm: sel = np.sort(np.concatenate([np.flatnonzero(y == c)[:sm] for c in (0, 1)]))
    res = dict(seed=SEED, mlp_seed=SEED_MLP, n_mlp_train=int(len(itr)), mlp_overlap=int(len(np.intersect1d(itr, used))),
               acc_full_sample_5class=acc, n_boot=B)
    res['main'], _ = E.analyse(net, unt, models, x[sel], y[sel], 'verify', rng, B, True, sm)
    tag = 'e34_verify' + ('_smoke' if sm else '')
    (OUT / f'{tag}.json').write_text(json.dumps(res, indent=1)); log('main written')
    del unt; unt1 = Net(untrained=True, seed=1); log('untrained seed 1 loaded')
    res['untrained_seed1'] = untrained_check(unt1, x[sel], CACHE_FOR)
    res['wall_s'] = time.time() - T0_
    (OUT / f'{tag}.json').write_text(json.dumps(res, indent=1)); log(f'wrote {OUT / tag}.json; E34 VERIFY DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', type=int, default=0); main(ap.parse_args())
