"""E38 (claims A18, A3, A5; exploratory: a quantity beyond mass): the leading-particle (hardness) channel in JEDI-linear.

Network: official JEDI-linear 3-feature n64 (inputs pT, deta, dphi about the stored axis), exact fixed-point forward pass
(interp.jedi_core.Jedi.run / run_jetwise for unit holds).

Edit family T(k->j, f): a fraction f in {0.05, 0.10, 0.20} of the pT of particle rank k is moved to particle rank j, with
k, j in {0, ..., 5, 'rest'} ('rest' = ranks 6-63 pooled: as a source every rest particle gives the fraction f of its own pT, as
a destination the moved pT is shared in proportion to the rest particles' pT). Particle angles are then changed by the smallest
amount (Gauss-Newton minimum-norm step in angle space only, built on the E36 four-momentum Jacobian interp.e36_counterfactual_
flip_jedi.jac / fourmom, pT changes ramped in 4 steps) so that the summed four-momentum (massless particles, axis at (0, 0)) equals
its unedited value: jet pT, invariant mass and E-scheme axis are restored to float64 precision; the particle count is unchanged.
Per edit the residual angular change (pT-weighted mean and max |d angle|, mrad), the M2 = sum pT r^2 change and the pT^D change
are recorded. Particles are re-sorted by pT before the network sees them (slot = pT rank).
Controls: (i) soft-sector null: a random zero-sum pT reshuffle among ranks 6-63 with the same L2 size (GeV) as the T edit
(a random half of the rank 6-63 particles give the same fraction a <= 0.9 of their pT to 2 random other rank 6-63 particles;
a matched by bisection; achieved size ratio recorded), same projection; (ii) matched rank
distance: transfer of the same fraction f between two particles at the same rank distance |k - j| (same direction) placed at
random inside ranks 14-40; (iii) rotation null: rigid rotation of the particle pattern about the jet axis by a random angle, no
pT change, same projection; (iv) 10 untrained redraws (Jedi(untrained=True, seed 0-9)) on a 2,000-jet subset at f = 0.2;
(v) decomposition at f = 0.2: the pT transfer alone (no angle projection, so the four-momentum is not restored) and the
projection's angle moves alone (no pT transfer).
Measurements: rank x rank maps of the change in the W-Z logit margin (logit_W - logit_Z) and the q-g margin (logit_q - logit_g)
per true class, 95 % jet-bootstrap CI; every live S unit's change in sd units; call-change fractions with McNemar vs control (i).
Naming: S20, S51, S43 fitted (ridge on hinge bases) by leading-particle formulas (pT_0, pT_0 + pT_1, pT_0 - pT_1, top-3 sum,
pT^D, z_lead) vs radial formulas (sum_{ranks>=1}(r-0.2)+, sum pT r^2, sum r^2, sum r core, sum pT r top14, sum r top14, soft count);
(A) fitted on unedited jets, (B) fitted on edit changes of a fit half; slope of observed vs predicted change on the held-out half.
Causal share: on E36's stored minimal flip edits ($AI_RESULTS/jedi/e36_edits.npz, which store the unedited jets x0 and edit v)
hold S20 + S51 + S43 at their unedited values; fraction of flips undone and of target-margin change removed, vs up to 500 random
live-S triples matched on displacement norm and vs the radial triple S1 + S29 + S30 (rank p, Holm over the two sets). Same hold
on the 4 T cells with the largest W-Z response.
References: 5-class MLPs on (log m, log pT, N), + pT^D, + z_lead, trained on true labels of a disjoint pool; their margin change
on the same edited jets; ratio network / reference.
Samples: main sample(1000, 3801) (5,000 jets); replicate sample(1000, 3802) minus overlaps (f = 0.2 map); pool sample(1000, 3804)
minus both (references). Writes $AI_RESULTS/jedi/e38.json (+ e38_maps.npz, e38_fig_jet.npz).
Run: python -m interp.e38_hardness_channel_jedi [--smoke 40]
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('KERAS_BACKEND', 'jax')
from interp import e36_counterfactual_flip_jedi as E  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0 = time.time()
G_, Q_, W_, Z_, T_ = range(5); CL = ['g', 'q', 'W', 'Z', 't']
RANKS = [0, 1, 2, 3, 4, 5, 'rest']; FR = (0.05, 0.10, 0.20)
CELLS = [(k, j) for k in RANKS for j in RANKS if k != j]
LIVE_S = E.LIVE_S; NAMED = [20, 51, 43]; RADIAL = [1, 29, 30]
NMIN = 8


def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def cname(k, j, f): return f'{k}->{j}@{f:.2f}'


# ------------------------------------------------------------------------------------------------ bootstrap helpers
class Boot:
    """jet-bootstrap via a (B, n) count matrix: means of many columns at once"""
    def __init__(self, n, B=1000, seed=0):
        rng = np.random.default_rng(seed); idx = rng.integers(0, n, (B, n))
        self.C = np.zeros((B, n), np.float32)
        for b in range(B): self.C[b] = np.bincount(idx[b], minlength=n)
        self.n = n; self.B = B

    def mean(self, D, mask=None):
        """D (n,) or (n, c) -> dict(mean, ci) per column; mask (n,) restricts to a subset (weights renormalised)"""
        D = np.asarray(D, np.float64); one = D.ndim == 1; D = D[:, None] if one else D
        w = np.ones(self.n) if mask is None else mask.astype(float)
        fin = np.isfinite(D); Dz = np.where(fin, D, 0.0); Wm = fin * w[:, None]
        num = self.C @ (Dz * w[:, None]); den = self.C @ Wm
        with np.errstate(all='ignore'):
            bs = num / den; mu = (Dz * w[:, None]).sum(0) / Wm.sum(0)
        lo, hi = np.nanquantile(bs, .025, 0), np.nanquantile(bs, .975, 0)
        p = np.minimum(1.0, 2 * np.minimum((bs <= 0).mean(0), (bs >= 0).mean(0)) + 1 / self.B)
        p = np.where(Wm.sum(0) > 0, p, 1.0)
        r = dict(mean=mu, lo=lo, hi=hi, p=p, n=Wm.sum(0))
        return {k: (v[0] if one else v) for k, v in r.items()}

    def ratio(self, a, b, mask=None):
        """ratio of means mean(a)/mean(b) on mask with CI"""
        w = np.ones(self.n) if mask is None else mask.astype(float)
        fin = np.isfinite(a) & np.isfinite(b); w = w * fin; a = np.where(fin, a, 0); b = np.where(fin, b, 0)
        with np.errstate(all='ignore'):
            bs = (self.C @ (a * w)) / (self.C @ (b * w)); v = (a * w).sum() / (b * w).sum()
        return dict(value=float(v), ci=[float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))])


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); out = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out


def mcnemar(a, b):
    from scipy.stats import binomtest
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
    return dict(first_only=n10, second_only=n01, p=p)


# ------------------------------------------------------------------------------------------------ edits
def project_angles(x0, v, P0, ramp=2, iters=10):
    """v (J, n, 3): v[..., 0] = target log pT ratios (fixed), v[..., 1:] = starting angle moves (in E.SC units).  The pT part is
    ramped in `ramp` steps; after each, Gauss-Newton minimum-norm steps in the angle coordinates only (pseudo-inverse of the
    4 x 4 normal matrix, backtracking: step halved up to 6 times until the residual decreases) restore sum p = P0.
    -> v, good (finite)"""
    real = x0[..., 0] > 0; J = len(x0); fr = np.zeros(x0.shape); fr[..., 1:] = 1; fr = (fr * real[..., None]).reshape(J, 1, -1)
    s = np.maximum(np.hypot(P0[:, 0], P0[:, 1]), 1e-9)[:, None]; ut = np.where(real, v[..., 0], 0.0)
    v = np.where(real[..., None], v, 0.0).copy(); v[..., 0] = 0; good = np.ones(J, bool)
    res = lambda ii, vv: np.sqrt((((E.fourmom(x0[ii], vv) - P0[ii]) / s[ii]) ** 2).sum(1))
    for k in range(1, ramp + 1):
        v[..., 0] = ut * k / ramp
        act = np.arange(J)
        for _ in range(iters):
            with np.errstate(all='ignore'):
                xa, va, sa = x0[act], v[act], s[act]
                r = (E.fourmom(xa, va) - P0[act]) / sa; r0 = np.sqrt((r ** 2).sum(1))
                keep = ~(r0 < 1e-13); act, xa, va, sa, r, r0 = act[keep], xa[keep], va[keep], sa[keep], r[keep], r0[keep]
                if not len(act): break
                A = E.jac(xa, va) / sa[..., None] * fr[act]
                AA = A @ np.transpose(A, (0, 2, 1))
                fin = np.isfinite(AA).all((1, 2)) & np.isfinite(r).all(1); good[act[~fin]] = False
                AA[~fin] = np.eye(4); r[~fin] = 0; A[~fin] = 0
                d = -np.einsum('jkn,jk->jn', A, np.einsum('jkl,jl->jk', np.linalg.pinv(AA, rcond=1e-12), r)).reshape(va.shape)
                d[~np.isfinite(d).all((1, 2))] = 0
                step = 1.0; pend = np.arange(len(act))
                for _h in range(7):
                    trial = va[pend] + step * d[pend]; rt = res(act[pend], trial); better = np.isfinite(rt) & (rt < r0[pend])
                    v[act[pend[better]]] = trial[better]; pend = pend[~better]; step /= 2
                    if not len(pend): break
    good &= np.isfinite(v).all((1, 2)); v[~good] = 0
    return np.where(real[..., None], v, 0.0), good


def transfer_u(x0, k, j, f):
    """log pT ratios (J, n) for T(k->j, f) and the moved pT (J,) in GeV"""
    pt = x0[..., 0]; real = pt > 0; J, n = pt.shape; newpt = pt.copy()
    rest = real.copy(); rest[:, :6] = False
    if k == 'rest':
        A = f * (pt * rest).sum(1); newpt = np.where(rest, pt * (1 - f), newpt)
    else:
        A = f * pt[:, k]; newpt[:, k] = pt[:, k] * (1 - f)
    if j == 'rest':
        tot = (pt * rest).sum(1); newpt = np.where(rest, newpt + A[:, None] * pt / np.maximum(tot, 1e-12)[:, None], newpt)
    else:
        newpt[:, j] = newpt[:, j] + A
    with np.errstate(all='ignore'):
        u = np.where(real, np.log(np.maximum(newpt, 1e-30) / np.maximum(pt, 1e-30)), 0.0)
    return u, newpt - pt


def soft_null_u(x0, dpt_ref, rng):
    """random zero-sum pT reshuffle among ranks 6-63 with L2 (GeV) = ||dpt_ref|| where reachable: a random half of the real
    rank 6-63 particles (givers) each lose the same fraction a <= 0.9 of their pT; the total goes to 2 random other rank 6-63
    particles (receivers, Dirichlet(1, 1) split); a is set by bisection so that the L2 norm of the pT change matches the T edit
    (capped at a = 0.9).  -> u, achieved L2 / target L2"""
    pt = x0[..., 0]; real = pt > 0; soft = real.copy(); soft[:, :6] = False; J, n = pt.shape
    key = np.where(soft, rng.random(pt.shape), np.inf); order = np.argsort(key, 1); ns = soft.sum(1)
    rk = np.empty_like(order); np.put_along_axis(rk, order, np.arange(n)[None].repeat(J, 0), 1)
    recv = soft & (rk < 2); give = soft & (rk >= 2) & (rk < 2 + (ns[:, None] - 2) // 2)
    w = np.where(recv, rng.gamma(1.0, size=pt.shape), 0.0); w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    G1 = (pt * give).sum(1)                                     # pT moved per unit fraction a
    def l2(a): return np.sqrt(((a[:, None] * pt * give) ** 2).sum(1) + (((a * G1)[:, None] * w) ** 2).sum(1))
    L = np.sqrt((dpt_ref ** 2).sum(1)); lo, hi = np.zeros(J), np.full(J, 0.9)
    for _ in range(40):
        mid = (lo + hi) / 2; big = l2(mid) > L; hi = np.where(big, mid, hi); lo = np.where(big, lo, mid)
    a = np.where(l2(np.full(J, 0.9)) <= L, 0.9, lo); a = np.where(ns >= 4, a, 0.0)
    d = -a[:, None] * pt * give + (a * G1)[:, None] * w
    with np.errstate(all='ignore'):
        u = np.where(soft, np.log(np.maximum(pt + d, 1e-30) / np.maximum(pt, 1e-30)), 0.0)
    return u, np.sqrt((d ** 2).sum(1)) / np.maximum(L, 1e-12)


def matched_u(x0, k, j, f, rng):
    """transfer of fraction f at the same rank distance and direction inside ranks 14-40 -> u, defined (J,)"""
    pt = x0[..., 0]; nreal = (pt > 0).sum(1); d = abs(j - k); J = len(x0)
    hi = np.minimum(40, nreal - 1) - d                                  # largest allowed lower rank
    ok = hi >= 14; a = 14 + np.floor(rng.random(J) * np.maximum(hi - 14 + 1, 1)).astype(int); a = np.where(ok, a, 14)
    src, dst = (a, a + d) if k < j else (a + d, a)
    u = np.zeros(pt.shape); r = np.arange(J); A = f * pt[r, src]
    with np.errstate(all='ignore'):
        u[r, src] = np.log1p(-f); u[r, dst] = np.log1p(A / np.maximum(pt[r, dst], 1e-30))
    u[~ok] = 0
    return u, ok


def rotation_v(x0, rng):
    """rigid rotation of (deta, dphi) about the axis by a random angle -> v angle part (E.SC units)"""
    th = rng.uniform(0, 2 * np.pi, len(x0)); c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    de, dp = x0[..., 1], x0[..., 2]; v = np.zeros(x0.shape)
    v[..., 1] = (c * de - s * dp - de) / E.SC; v[..., 2] = (s * de + c * dp - dp) / E.SC
    return np.where((x0[..., 0] > 0)[..., None], v, 0.0)


def make_edit(x0, u, P0, vang=None):
    v = np.zeros(x0.shape); v[..., 0] = u
    if vang is not None: v[..., 1:] = vang[..., 1:]
    v, good = project_angles(x0, v, P0)
    xe, _ = E.apply_edit(x0, v); dm, dax, dpt = E.check_constraints(x0, xe)
    comp = good & (dm < 1e-3) & (dax < 1e-3) & (dpt < 1e-3)
    return xe, v, comp, dict(dm=dm, dax=dax, dpt=dpt)


# ------------------------------------------------------------------------------------------------ features
def lead_feats(x, jpt):
    pt = x[..., 0]; M0 = pt.sum(1)
    return {'pT_0': pt[:, 0], 'pT_0+pT_1': pt[:, 0] + pt[:, 1], 'pT_0-pT_1': pt[:, 0] - pt[:, 1], 'pT_0+pT_1+pT_2': pt[:, :3].sum(1),
            'pT^D': np.sqrt((pt ** 2).sum(1)) / np.maximum(M0, 1e-12), 'z_lead': pt[:, 0] / np.maximum(jpt, 1e-12)}


def radial_feats(x):
    pt, de, dp = x[..., 0], x[..., 1], x[..., 2]; real = pt > 0; r = np.where(real, np.hypot(de, dp), 0.0)
    rk = np.arange(x.shape[1])[None]; t14 = real & (rk < 14)
    return {'sum_{rank>=1}(r-0.2)+': (np.maximum(r - 0.2, 0) * real * (rk >= 1)).sum(1), 'M2 = sum pT r^2': (pt * r ** 2).sum(1),
            'sum r^2': (r ** 2 * real).sum(1), 'sum r (r<0.165)': (r * (real & (r < 0.165))).sum(1),
            'sum pT r (top14)': (pt * r * t14).sum(1), 'sum r (top14)': (r * t14).sum(1), 'N(pT<2)': (real & (pt < 2)).sum(1).astype(float)}


def physics(x, jpt):
    pt = x[..., 0]; real = pt > 0; r = np.hypot(x[..., 1], x[..., 2]); M0 = pt.sum(1)
    return dict(ptD=np.sqrt((pt ** 2).sum(1)) / np.maximum(M0, 1e-12), M2=(pt * r ** 2 * real).sum(1), zlead=pt[:, 0] / jpt)


class Basis:
    """hinge basis per feature: x, (x - q25)+, (x - q50)+, (x - q75)+, (q25 - x)+, standardised on reference jets"""
    def __init__(self, F):
        self.keys = list(F); self.q = {k: np.quantile(F[k], [.25, .5, .75]) for k in self.keys}
        X = self.raw(F); self.mu, self.sd = X.mean(0), X.std(0) + 1e-12

    def raw(self, F):
        cols = []
        for k in self.keys:
            x = np.asarray(F[k], np.float64); q = self.q[k]
            cols += [x, np.maximum(x - q[0], 0), np.maximum(x - q[1], 0), np.maximum(x - q[2], 0), np.maximum(q[0] - x, 0)]
        return np.stack(cols, 1)

    def __call__(self, F): return (self.raw(F) - self.mu) / self.sd


# ------------------------------------------------------------------------------------------------ one sample: the map
def run_map(name, net, x, y, rng, fr, null=True, matched=True, rot=True, keep_feats=False):
    x0 = np.asarray(x, np.float64); J = len(x0); P0 = E.p4(x0).sum(1); jpt = np.hypot(P0[:, 0], P0[:, 1])
    nreal = (x0[..., 0] > 0).sum(1); use = nreal >= NMIN
    b = net.run(x, keep=('logits', 'S')); L0, S0 = b['logits'], b['S']; c0 = L0.argmax(1)
    out = dict(N=J, acc_exact=float((c0 == y).mean()), n_used=int(use.sum()), nmin=NMIN)
    log(f'{name}: N {J}; exact accuracy {out["acc_exact"]:.4f}; jets with >= {NMIN} particles {use.sum()}')
    sdS = S0.std(0) + 1e-9; ph0 = physics(x0, jpt); sdp = {k: v.std() for k, v in ph0.items()}
    wz0, qg0 = L0[:, W_] - L0[:, Z_], L0[:, Q_] - L0[:, G_]
    D = dict(cells=[], wz=[], qg=[], chg=[], dS=[], comp=[], wz_null=[], qg_null=[], chg_null=[], dS_null=[], comp_null=[], size_null=[],
             wz_m=[], qg_m=[], chg_m=[], ok_m=[], ang_mean=[], ang_max=[], dM2=[], dptD=[], dzlead=[], lead=[], rad=[])
    for f in fr:
        for k, j in CELLS:
            u, dpt = transfer_u(x0, k, j, f); xe, v, comp, cc = make_edit(x0, u, P0); comp &= use
            r = net.run(xe.astype(np.float32), keep=('logits', 'S')); L = r['logits']
            D['cells'].append((k, j, f)); D['wz'].append(np.where(comp, L[:, W_] - L[:, Z_] - wz0, np.nan)); D['qg'].append(np.where(comp, L[:, Q_] - L[:, G_] - qg0, np.nan))
            D['chg'].append(np.where(comp, (L.argmax(1) != c0).astype(float), np.nan)); D['comp'].append(comp)
            D['dS'].append(((r['S'][:, LIVE_S] - S0[:, LIVE_S]) / sdS[LIVE_S]).astype(np.float32))
            ang = E.SC * np.hypot(v[..., 1], v[..., 2]); w = x0[..., 0] * (x0[..., 0] > 0)
            D['ang_mean'].append((ang * w).sum(1) / w.sum(1) * 1e3); D['ang_max'].append(ang.max(1) * 1e3)
            ph = physics(xe, jpt); D['dM2'].append((ph['M2'] - ph0['M2']) / sdp['M2']); D['dptD'].append((ph['ptD'] - ph0['ptD']) / sdp['ptD'])
            D['dzlead'].append((ph['zlead'] - ph0['zlead']) / sdp['zlead'])
            if keep_feats:
                D['lead'].append(lead_feats(xe, jpt))
                D['rad'].append(radial_feats(xe))
            if f == max(FR) and keep_feats:                                     # decomposition: pT part only / angle part only
                for lab, vv in (('pt_only', np.concatenate([u[..., None], np.zeros(x0.shape[:2] + (2,))], -1)),
                                ('angle_only', np.concatenate([np.zeros(x0.shape[:2] + (1,)), v[..., 1:]], -1))):
                    xd, _ = E.apply_edit(x0, vv); Ld = net.run(xd.astype(np.float32))['logits']
                    D.setdefault(f'wz_{lab}', {})[len(D['cells']) - 1] = np.where(comp, Ld[:, W_] - Ld[:, Z_] - wz0, np.nan)
                    D.setdefault(f'qg_{lab}', {})[len(D['cells']) - 1] = np.where(comp, Ld[:, Q_] - Ld[:, G_] - qg0, np.nan)
                    D.setdefault(f'chg_{lab}', {})[len(D['cells']) - 1] = np.where(comp, (Ld.argmax(1) != c0).astype(float), np.nan)
            if null:
                un, achieved = soft_null_u(x0, dpt, rng); xn, _, cn, _ = make_edit(x0, un, P0); cn &= use
                rn = net.run(xn.astype(np.float32), keep=('logits', 'S')); Ln = rn['logits']
                D['wz_null'].append(np.where(cn, Ln[:, W_] - Ln[:, Z_] - wz0, np.nan)); D['qg_null'].append(np.where(cn, Ln[:, Q_] - Ln[:, G_] - qg0, np.nan))
                D['chg_null'].append(np.where(cn, (Ln.argmax(1) != c0).astype(float), np.nan)); D['comp_null'].append(cn); D['size_null'].append(achieved)
                D['dS_null'].append(((rn['S'][:, LIVE_S] - S0[:, LIVE_S]) / sdS[LIVE_S]).astype(np.float32))
            if matched and k != 'rest' and j != 'rest':
                um, okm = matched_u(x0, k, j, f, rng); xm, _, cm, _ = make_edit(x0, um, P0); cm &= use & okm
                Lm = net.run(xm.astype(np.float32))['logits']
                D['wz_m'].append(np.where(cm, Lm[:, W_] - Lm[:, Z_] - wz0, np.nan)); D['qg_m'].append(np.where(cm, Lm[:, Q_] - Lm[:, G_] - qg0, np.nan))
                D['chg_m'].append(np.where(cm, (Lm.argmax(1) != c0).astype(float), np.nan)); D['ok_m'].append(cm)
            elif matched:
                for kk in ('wz_m', 'qg_m', 'chg_m'): D[kk].append(np.full(J, np.nan))
                D['ok_m'].append(np.zeros(J, bool))
        log(f'{name}: f {f:.2f}: {len(CELLS)} cells done; compliance T {np.mean([c.mean() for c in D["comp"][-len(CELLS):]]):.4f}'
            + (f', null {np.mean([c.mean() for c in D["comp_null"][-len(CELLS):]]):.4f}' if null else ''))
    rotd = None
    if rot:
        rotd = dict(wz=[], qg=[], chg=[])
        for rr in range(3):
            xr, _, cr, _ = make_edit(x0, np.zeros(x0.shape[:2]), P0, vang=rotation_v(x0, rng)); cr &= use
            Lr = net.run(xr.astype(np.float32))['logits']
            rotd['wz'].append(np.where(cr, Lr[:, W_] - Lr[:, Z_] - wz0, np.nan)); rotd['qg'].append(np.where(cr, Lr[:, Q_] - Lr[:, G_] - qg0, np.nan))
            rotd['chg'].append(np.where(cr, (Lr.argmax(1) != c0).astype(float), np.nan))
        log(f'{name}: rotation null (3 draws) compliance {np.mean([np.isfinite(a).mean() for a in rotd["wz"]]):.4f}')
    return out, D, dict(x0=x0, P0=P0, jpt=jpt, L0=L0, S0=S0, c0=c0, sdS=sdS, use=use, wz0=wz0, qg0=qg0, rot=rotd)


def summarise_map(D, base, y, boot, label):
    """per-cell table: margins per class, changes vs null, McNemar, S units"""
    cells = D['cells']; nC = len(cells); res = dict(cells=[cname(*c) for c in cells])
    WZ = np.stack(D['wz'], 1); QG = np.stack(D['qg'], 1); CH = np.stack(D['chg'], 1)
    have_null = len(D['wz_null']) == nC
    by_class = {}
    for c in range(5):
        m = y == c; a, b_ = boot.mean(WZ, m), boot.mean(QG, m)
        by_class[CL[c]] = dict(wz=dict(mean=a['mean'], lo=a['lo'], hi=a['hi']), qg=dict(mean=b_['mean'], lo=b_['lo'], hi=b_['hi']))
    res['by_class'] = by_class
    mWZ, mQG = (y == W_) | (y == Z_), (y == Q_) | (y == G_)
    res['wz_on_WZ'] = boot.mean(WZ, mWZ); res['qg_on_qg'] = boot.mean(QG, mQG); res['call_change'] = boot.mean(CH)
    res['compliance'] = np.array([c.mean() for c in D['comp']])
    res['ang_mean_mrad_median'] = np.array([np.nanmedian(a) for a in D['ang_mean']]); res['ang_max_mrad_median'] = np.array([np.nanmedian(a) for a in D['ang_max']])
    res['dM2_sd_mean'] = np.array([np.nanmean(a) for a in D['dM2']]); res['dptD_sd_mean'] = np.array([np.nanmean(a) for a in D['dptD']])
    res['dzlead_sd_mean'] = np.array([np.nanmean(a) for a in D['dzlead']])
    dS = np.stack(D['dS'], 1)                                                           # (J, cells, units)
    res['S_mean_sd'] = {f'S{u}': np.nanmean(dS[:, :, i], 0) for i, u in enumerate(LIVE_S)}
    if have_null:
        WZn, QGn, CHn = np.stack(D['wz_null'], 1), np.stack(D['qg_null'], 1), np.stack(D['chg_null'], 1)
        res['null'] = dict(wz_on_WZ=boot.mean(WZn, mWZ), qg_on_qg=boot.mean(QGn, mQG), call_change=boot.mean(CHn),
                           size_ratio_median=np.array([np.median(s) for s in D['size_null']]))
        dw, dq = boot.mean(WZ - WZn, mWZ), boot.mean(QG - QGn, mQG)
        res['T_minus_null'] = dict(wz_on_WZ=dw, qg_on_qg=dq, wz_p_holm=holm(dw['p']), qg_p_holm=holm(dq['p']), n_tests_each=nC)
        mc = []
        for i in range(nC):
            ok = np.isfinite(CH[:, i]) & np.isfinite(CHn[:, i]); mc.append(mcnemar(CH[ok, i] > 0, CHn[ok, i] > 0))
        ph = holm([m['p'] for m in mc])
        for m, p in zip(mc, ph): m['p_holm'] = float(p)
        res['mcnemar_T_vs_null'] = mc
        dSn = np.stack(D['dS_null'], 1)
        res['S_null_mean_sd'] = {f'S{u}': np.nanmean(dSn[:, :, i], 0) for i, u in enumerate(LIVE_S)}
        Sp = {}
        for i, u in enumerate(LIVE_S):
            bb = boot.mean(dS[:, :, i] - dSn[:, :, i]); Sp[f'S{u}'] = bb['p']
        allp = holm(np.concatenate([Sp[k] for k in Sp])); off = 0; res['S_T_minus_null_p_holm'] = {}
        for k in Sp: res['S_T_minus_null_p_holm'][k] = allp[off:off + nC]; off += nC
        res['S_holm_n_tests'] = int(len(allp))
    if len(D['wz_m']) == nC:
        WZm, QGm, CHm = np.stack(D['wz_m'], 1), np.stack(D['qg_m'], 1), np.stack(D['chg_m'], 1)
        res['matched_rank_distance'] = dict(wz_on_WZ=boot.mean(WZm, mWZ), qg_on_qg=boot.mean(QGm, mQG), call_change=boot.mean(CHm),
                                            coverage=np.array([o.mean() for o in D['ok_m']]))
        res['T_minus_matched'] = dict(wz_on_WZ=boot.mean(np.where(np.isfinite(WZm), WZ, np.nan) - WZm, mWZ),
                                      qg_on_qg=boot.mean(np.where(np.isfinite(QGm), QG, np.nan) - QGm, mQG))
    if 'wz_pt_only' in D:
        ids = sorted(D['wz_pt_only']); res['decomposition_cells'] = [cname(*cells[i]) for i in ids]; res['decomposition'] = {}
        for lab in ('pt_only', 'angle_only'):
            res['decomposition'][lab] = dict(wz_on_WZ=boot.mean(np.stack([D[f'wz_{lab}'][i] for i in ids], 1), mWZ),
                                             qg_on_qg=boot.mean(np.stack([D[f'qg_{lab}'][i] for i in ids], 1), mQG),
                                             call_change=boot.mean(np.stack([D[f'chg_{lab}'][i] for i in ids], 1)))
    if base.get('rot'):
        rr = base['rot']; RW, RQ, RC = np.stack(rr['wz'], 1).mean(1), np.stack(rr['qg'], 1).mean(1), np.stack(rr['chg'], 1).mean(1)
        res['rotation_null'] = dict(wz_on_WZ=boot.mean(RW, mWZ), qg_on_qg=boot.mean(RQ, mQG), call_change=boot.mean(RC),
                                    abs_wz_on_WZ=boot.mean(np.abs(RW), mWZ), abs_qg_on_qg=boot.mean(np.abs(RQ), mQG))
    log(f'{label}: map summarised')
    return res


# ------------------------------------------------------------------------------------------------ naming
def naming(D, base, x0, boot_seed=3811):
    from sklearn.linear_model import RidgeCV
    J = len(x0); use = base['use']; rng = np.random.default_rng(boot_seed); half = rng.permutation(J) < J // 2   # fit half
    lead0, rad0 = lead_feats(x0, base['jpt']), radial_feats(x0)
    S0 = base['S0']; sdS = base['sdS']; nC = len(D['cells'])
    fams = {'leading-particle': (lead0, 'lead'), 'radial': (rad0, 'rad'), 'leading + radial': ({**lead0, **rad0}, 'both')}
    def feats_cell(c, fam):
        if fam == 'lead': return D['lead'][c]
        if fam == 'rad': return D['rad'][c]
        return {**D['lead'][c], **D['rad'][c]}
    out = {}
    alphas = np.logspace(-3, 3, 13)
    for u in NAMED + RADIAL:
        iu = LIVE_S.index(u); ou = {}
        for fname, (F0, key) in fams.items():
            Bs = Basis({k: v[half & use] for k, v in F0.items()}); X0 = Bs(F0)
            # (A) fitted on unedited jets of the fit half
            mA = RidgeCV(alphas=alphas).fit(X0[half & use], S0[half & use, u]); r2A = float(mA.score(X0[~half & use], S0[~half & use, u]))
            # per cell predicted / observed changes (sd units)
            dX = []; obs = []
            for c in range(nC):
                ok = D['comp'][c] & use
                Xe = Bs({k: np.asarray(v, np.float64) for k, v in feats_cell(c, key).items()})
                dX.append(np.where(ok[:, None], Xe - X0, np.nan).astype(np.float32)); obs.append(np.where(ok, D['dS'][c][:, iu], np.nan))
            dX = np.stack(dX, 1); obs = np.stack(obs, 1)                       # (J, cells, p), (J, cells)
            predA = (dX @ mA.coef_) / sdS[u]
            # (B) fitted on edit changes of the fit half
            fitrows = half[:, None] & np.isfinite(obs) & np.isfinite(dX).all(2)
            mB = RidgeCV(alphas=alphas, fit_intercept=False).fit(dX[fitrows], obs[fitrows]); predB = dX @ mB.coef_
            res = {'r2_unedited_heldout_A': r2A}
            for lab, pred in (('A (fit on unedited jets)', predA), ('B (fit on edit changes, fit half)', predB)):
                te = (~half)[:, None] & np.isfinite(obs) & np.isfinite(pred)
                res[lab] = slope_boot(pred, obs, te)
            ou[fname] = res
            log(f'naming S{u} {fname}: A slope {res["A (fit on unedited jets)"]["slope"]:.3f} {res["A (fit on unedited jets)"]["ci"]} '
                f'R2 {res["A (fit on unedited jets)"]["r2"]:.3f}; B slope {res["B (fit on edit changes, fit half)"]["slope"]:.3f} '
                f'{res["B (fit on edit changes, fit half)"]["ci"]} R2 {res["B (fit on edit changes, fit half)"]["r2"]:.3f}; unedited R2 {r2A:.3f}')
        out[f'S{u}'] = ou
    return out


def slope_boot(pred, obs, te, B=1000, seed=3812):
    """OLS slope of observed on predicted change over held-out (jet, cell) rows; CI by resampling jets"""
    p = np.where(te, pred, 0.0); o = np.where(te, obs, 0.0); n = te.sum(1).astype(float)
    agg = np.stack([n, p.sum(1), o.sum(1), (p * p).sum(1), (p * o).sum(1), (o * o).sum(1)], 1)
    jets = np.flatnonzero(n > 0); agg = agg[jets]; rng = np.random.default_rng(seed)
    def sl(a):
        N, sp, so, spp, spo, soo = a.sum(0); vp = spp / N - (sp / N) ** 2; cv = spo / N - sp * so / N ** 2; vo = soo / N - (so / N) ** 2
        return cv / vp, cv ** 2 / (vp * vo), 1 - ((soo - 2 * spo + spp) / N - ((so - sp) / N) ** 2) / vo
    s, r2, r2pred = sl(agg); bs = [sl(agg[rng.integers(0, len(agg), len(agg))])[0] for _ in range(B)]
    ci = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
    return dict(slope=float(s), ci=ci, r2=float(r2), r2_identity=float(r2pred), n_jets=int(len(jets)), n_rows=int(agg[:, 0].sum()),
                within_0p8_1p2=bool(ci[0] >= 0.8 and ci[1] <= 1.2))


# ------------------------------------------------------------------------------------------------ holds
def hold_eval(net, xe32, S_hold, units, tgt, cur):
    m = np.zeros(64, np.float32); m[list(units)] = 1
    L = net.run_jetwise(xe32, {'S': (m, S_hold)}, keep=('logits',))['logits']
    return L, L[np.arange(len(L)), tgt] - L[np.arange(len(L)), cur]


def matched_triples(dnorm_fn, named, nmax, rng, excl):
    """all live-S triples (excluding the named set) ranked by |log(norm / named norm)|; up to nmax closest"""
    tr = [t for t in itertools.combinations(LIVE_S, 3) if set(t) != set(named)]
    dn = np.array([dnorm_fn(t) for t in tr]); d0 = dnorm_fn(tuple(named))
    o = np.argsort(np.abs(np.log(np.maximum(dn, 1e-9) / d0)))[:nmax]
    return [tr[i] for i in o], dn[o], d0


def e36_hold(net, rng, nrand):
    path = OUT / 'e36_edits.npz'
    k = np.load(path); x0 = k['x0'].astype(np.float64); v = k['v'].astype(np.float64)
    ok, prim, tgt, c_e = k['ok'], k['prim'], k['tgt'], k['c_e']
    b = net.run(x0.astype(np.float32), keep=('logits', 'S')); L0, S0 = b['logits'], b['S']; cur = L0.argmax(1)
    xe, _ = E.apply_edit(x0, v); xe32 = xe.astype(np.float32); e = net.run(xe32, keep=('logits', 'S')); LE, SE = e['logits'], e['S']
    reproduced = float((LE.argmax(1)[ok] == c_e[ok]).mean())
    fl = ok & prim & (LE.argmax(1) == tgt); idx = np.flatnonzero(fl)
    log(f'E36 edits: {len(x0)} jets stored (x0 present); exact calls reproduced on ok edits {reproduced:.4f}; flips (primary) {len(idx)}')
    xe32 = xe32[idx]; tg, cu = tgt[idx], cur[idx]; S0f, SEf = S0[idx], SE[idx]
    m0 = L0[idx, tg] - L0[idx, cu]; mE = LE[idx, tg] - LE[idx, cu]; sdS = S0.std(0) + 1e-9
    dS = (SEf - S0f) / sdS
    mall = np.zeros(64, np.float32); mall[LIVE_S] = 1
    Lid = net.run_jetwise(xe32, {'S': (mall, SEf)}, keep=('logits',))['logits']
    identity_maxdiff = float(np.abs(Lid - LE[idx]).max()); log(f'E36 hold identity check (all live S held at their edited values): max |dlogit| {identity_maxdiff:.2e}')
    grp ={'W->Z': cu == W_, 'Z->W': cu == Z_, 'q->g': cu == Q_, 'g->q': cu == G_}
    boot = Boot(len(idx), 1000, 3813)
    def evalset(units):
        L, mh = hold_eval(net, xe32, S0f, units, tg, cu)
        und = (L.argmax(1) != tg).astype(float); rem = boot.ratio((mE - mh), (mE - m0))
        return und, mh, rem
    dnorm = lambda t: float(np.mean(np.sqrt((dS[:, list(t)] ** 2).sum(1))))
    res = dict(n_flips=int(len(idx)), calls_reproduced=reproduced, hold_identity_max_dlogit=identity_maxdiff, margin_definition='logit(target) - logit(unedited call)',
               d_margin_edit=boot.mean(mE - m0), named_dnorm_sd=dnorm(NAMED), radial_dnorm_sd=dnorm(RADIAL),
               dS_abs_sd={f'S{u}': float(np.abs(dS[:, u]).mean()) for u in LIVE_S})
    sets = {}
    for lab, units in (('S20+S51+S43 (leading-particle)', NAMED), ('S1+S29+S30 (radial)', RADIAL)):
        und, mh, rem = evalset(units)
        sets[lab] = dict(units=units, flips_undone=boot.mean(und), margin_removed=rem,
                         by_group={g: dict(n=int(m.sum()), flips_undone=float(und[m].mean()) if m.any() else None) for g, m in grp.items()},
                         _und=und.mean(), _rem=rem['value'])
        log(f'E36 hold {lab}: flips undone {und.mean():.3f}, margin removed {rem["value"]:.3f} {rem["ci"]}')
    trip, dn, d0 = matched_triples(dnorm, NAMED, nrand, rng, None)
    ru, rr = [], []
    for t in trip:
        und, _, rem = evalset(t); ru.append(und.mean()); rr.append(rem['value'])
    ru, rr = np.array(ru), np.array(rr)
    res['random_triples'] = dict(n=len(trip), matched_on='mean over flipped jets of sqrt(sum_{u in triple} (dS_u/sd_u)^2)',
                                 norm_named=d0, norm_random_q05_q50_q95=np.quantile(dn, [.05, .5, .95]).tolist(),
                                 flips_undone_q50_q95_max=[float(np.median(ru)), float(np.quantile(ru, .95)), float(ru.max())],
                                 margin_removed_q50_q95_max=[float(np.median(rr)), float(np.quantile(rr, .95)), float(rr.max())],
                                 top5_by_flips_undone=[dict(units=list(trip[i]), flips_undone=float(ru[i]), margin_removed=float(rr[i])) for i in np.argsort(-ru)[:5]])
    ps = []
    for lab in sets:
        s = sets[lab]; pu = (1 + (ru >= s['_und']).sum()) / (1 + len(ru)); pm = (1 + (rr >= s['_rem']).sum()) / (1 + len(rr))
        s['rank_p_flips_undone'] = float(pu); s['rank_p_margin_removed'] = float(pm); ps += [pu, pm]
    ph = holm(ps)
    for i, lab in enumerate(sets):
        sets[lab]['rank_p_holm'] = dict(flips_undone=float(ph[2 * i]), margin_removed=float(ph[2 * i + 1]))
        sets[lab].pop('_und'); sets[lab].pop('_rem')
    res['sets'] = sets
    log(f'E36 hold: random matched triples ({len(trip)}): flips undone median {np.median(ru):.3f} max {ru.max():.3f}; margin removed median {np.median(rr):.3f} max {rr.max():.3f}')
    return res


def cell_holds(net, base, D, y, nrand, rng, top=4):
    """hold triples on the T cells (f = 0.2) with the largest mean W-Z response on W/Z jets"""
    x0, P0, S0, sdS, use = base['x0'], base['P0'], base['S0'], base['sdS'], base['use']
    mWZ = ((y == W_) | (y == Z_)); ids = [i for i, c in enumerate(D['cells']) if c[2] == max(FR)]
    eff = np.array([np.nanmean(D['wz'][i][mWZ]) for i in ids]); pick = [ids[i] for i in np.argsort(-np.abs(eff))[:top]]
    out = {}
    for i in pick:
        k, j, f = D['cells'][i]; u, _ = transfer_u(x0, k, j, f); xe, _, comp, _ = make_edit(x0, u, P0); sel = np.flatnonzero(comp & use & mWZ)
        xe32 = xe[sel].astype(np.float32); r = net.run(xe32, keep=('logits', 'S')); LE, SE = r['logits'], r['S']
        d_wz = (LE[:, W_] - LE[:, Z_]) - base['wz0'][sel]; dS = (SE - S0[sel]) / sdS; boot = Boot(len(sel), 500, 3814)
        def removed(units):
            m = np.zeros(64, np.float32); m[list(units)] = 1
            L = net.run_jetwise(xe32, {'S': (m, S0[sel])}, keep=('logits',))['logits']; dh = (L[:, W_] - L[:, Z_]) - base['wz0'][sel]
            return boot.ratio(d_wz - dh, d_wz)
        dn = lambda t: float(np.mean(np.sqrt((dS[:, list(t)] ** 2).sum(1))))
        trip, dnr, d0 = matched_triples(dn, NAMED, nrand, rng, None)
        rr = np.array([removed(t)['value'] for t in trip])
        rn, rad = removed(NAMED), removed(RADIAL)
        out[cname(k, j, f)] = dict(n=int(len(sel)), d_wz=boot.mean(d_wz), named=rn, radial=rad, named_dnorm=d0,
                                   random_triples=dict(n=len(trip), removed_q50_q95_max=[float(np.median(rr)), float(np.quantile(rr, .95)), float(rr.max())],
                                                       norm_q05_q95=np.quantile(dnr, [.05, .95]).tolist()),
                                   rank_p_named=float((1 + (rr >= rn['value']).sum()) / (1 + len(rr))),
                                   rank_p_radial=float((1 + (rr >= rad['value']).sum()) / (1 + len(rr))))
        log(f'cell hold {cname(k, j, f)}: dWZ {np.mean(d_wz):.3f}; named removes {rn["value"]:.3f} {rn["ci"]}, radial {rad["value"]:.3f}; random median {np.median(rr):.3f} max {rr.max():.3f}')
    return out


# ------------------------------------------------------------------------------------------------ references
def references(xp, yp, D, base, y, boot):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    def jf(x):
        x = np.asarray(x, np.float64); P = E.p4(x).sum(1); jpt = np.hypot(P[:, 0], P[:, 1])
        m = np.sqrt(np.maximum(P[:, 3] ** 2 - (P[:, :3] ** 2).sum(1), 0)); pt = x[..., 0]; N = (pt > 0).sum(1)
        return dict(logm=np.log(np.maximum(m, 1e-3)), logpt=np.log(jpt), N=N.astype(float), ptD=np.sqrt((pt ** 2).sum(1)) / pt.sum(1), zlead=pt[:, 0] / jpt)
    sets = {'MLP(m, pT, N)': ['logm', 'logpt', 'N'], 'MLP(m, pT, N, pT^D)': ['logm', 'logpt', 'N', 'ptD'], 'MLP(m, pT, N, z_lead)': ['logm', 'logpt', 'N', 'zlead']}
    Fp = jf(xp); out = {}; F0 = jf(base['x0']); mWZ, mQG = (y == W_) | (y == Z_), (y == Q_) | (y == G_)
    for lab, ks in sets.items():
        X = np.stack([Fp[k] for k in ks], 1); sc = StandardScaler().fit(X)
        mlp = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=500, early_stopping=True, random_state=3815).fit(sc.transform(X), yp)
        def marg(F):
            lp = mlp.predict_log_proba(sc.transform(np.stack([F[k] for k in ks], 1))); return lp[:, W_] - lp[:, Z_], lp[:, Q_] - lp[:, G_]
        w0, q0 = marg(F0); acc = float((mlp.predict(sc.transform(np.stack([F0[k] for k in ks], 1))) == y).mean())
        rows = {}
        for c, (k, j, f) in enumerate(D['cells']):
            if f != max(FR) and not (k in (0, 1, 2) and j in (0, 1, 2)): continue
            lead = D['lead'][c]; Fe = dict(logm=F0['logm'], logpt=F0['logpt'], N=F0['N'], ptD=np.asarray(lead['pT^D'], np.float64),
                                           zlead=np.asarray(lead['z_lead'], np.float64))
            w1, q1 = marg(Fe); ok = D['comp'][c]
            dwr, dqr = np.where(ok, w1 - w0, np.nan), np.where(ok, q1 - q0, np.nan)
            rows[cname(k, j, f)] = dict(ref_wz_on_WZ=float(np.nanmean(dwr[mWZ])), ref_qg_on_qg=float(np.nanmean(dqr[mQG])),
                                        ratio_net_over_ref_wz=boot.ratio(D['wz'][c], dwr, mWZ), ratio_net_over_ref_qg=boot.ratio(D['qg'][c], dqr, mQG))
        out[lab] = dict(features=ks, acc_on_main=acc, val_score=float(mlp.best_validation_score_), cells=rows)
        log(f'reference {lab}: accuracy on main {acc:.4f}; e.g. 0->1@0.20 ratio W-Z {rows.get("0->1@0.20", {}).get("ratio_net_over_ref_wz")}')
    return out


# ------------------------------------------------------------------------------------------------ untrained
def untrained(x, y, nred, n_sub, rng):
    from interp.jedi_core import Jedi
    sub = np.sort(rng.choice(len(y), min(n_sub, len(y)), replace=False)); x0 = np.asarray(x[sub], np.float64); P0 = E.p4(x0).sum(1)
    use = (x0[..., 0] > 0).sum(1) >= NMIN
    cells = [(k, j, max(FR)) for k, j in CELLS]; XE = []
    for k, j, f in cells:
        u, _ = transfer_u(x0, k, j, f); xe, _, comp, _ = make_edit(x0, u, P0); XE.append((xe.astype(np.float32), comp & use))
    out = []
    for s in range(nred):
        nu = Jedi(64, untrained=True, seed=s); L0 = nu.run(x[sub])['logits']; wz0, qg0 = L0[:, W_] - L0[:, Z_], L0[:, Q_] - L0[:, G_]
        sw, sq = wz0.std() + 1e-12, qg0.std() + 1e-12; mw, mq = [], []
        for xe, ok in XE:
            L = nu.run(xe)['logits']; mw.append(np.nanmean(np.where(ok, (L[:, W_] - L[:, Z_] - wz0) / sw, np.nan))); mq.append(np.nanmean(np.where(ok, (L[:, Q_] - L[:, G_] - qg0) / sq, np.nan)))
        mw, mq = np.array(mw), np.array(mq)
        out.append(dict(seed=s, wz_sd=float(sw), qg_sd=float(sq), max_abs_wz_cell_sd=float(np.nanmax(np.abs(mw))), max_abs_qg_cell_sd=float(np.nanmax(np.abs(mq))),
                        mean_abs_wz_cell_sd=float(np.nanmean(np.abs(mw))), mean_abs_qg_cell_sd=float(np.nanmean(np.abs(mq))), calls=np.bincount(L0.argmax(1), minlength=5).tolist(),
                        wz_cells_sd=mw.tolist(), qg_cells_sd=mq.tolist()))
        log(f'untrained {s}: max |cell| W-Z {out[-1]["max_abs_wz_cell_sd"]:.4f} sd, q-g {out[-1]["max_abs_qg_cell_sd"]:.4f} sd; calls {out[-1]["calls"]}')
        del nu
    return dict(jets=int(len(sub)), cells=[cname(*c) for c in cells], redraws=out)


# ------------------------------------------------------------------------------------------------ main
def main(a):
    from interp.jedi_core import Jedi, sample
    from interp.e11_mars_all import indices
    from interp.x10_das_jedi import clean
    OUT.mkdir(parents=True, exist_ok=True); sfx = '_smoke' if a.smoke else ''; JS = OUT / f'e38{sfx}.json'
    pc = a.smoke or 1000
    R = dict(node='E38', samples=dict(main=f'sample({pc}, 3801)', replicate=f'sample({pc}, 3802) minus main', pool=f'sample({pc}, 3804) minus both'),
             edit='T(k->j, f): fraction f of rank-k pT moved to rank j; angles min-norm projected to restore the jet four-momentum',
             ranks=[str(r) for r in RANKS], fractions=list(FR), cells=[cname(k, j, f) for f in FR for k, j in CELLS], live_S=LIVE_S,
             margins=dict(wz='logit_W - logit_Z', qg='logit_q - logit_g'), nmin_particles=NMIN)
    save = lambda: JS.write_text(json.dumps(clean(R), indent=1))
    x, y, _ = sample(pc, 3801); i1 = indices(pc, 3801)
    x2, y2, _ = sample(pc, 3802); i2 = indices(pc, 3802); k2 = ~np.isin(i2, i1); x2, y2 = np.asarray(x2[k2]), y2[k2]
    xp, yp, _ = sample(pc, 3804); ip = indices(pc, 3804); kp = ~np.isin(ip, np.concatenate([i1, i2])); xp, yp = np.asarray(xp[kp]), yp[kp]
    log(f'jets: main {len(y)}, replicate {len(y2)}, pool {len(yp)}')
    net = Jedi(64); rng = np.random.default_rng(3801)
    _run, _rjw = net.run, net.run_jetwise                       # smaller forward batches (memory: the 5,000-jet batch OOM-killed run2 at 2 GiB)
    net.run = lambda x_, override=None, keep=('logits',), batch=1000: _run(x_, override, keep, batch)
    net.run_jetwise = lambda x_, override, keep=('logits',), batch=1000: _rjw(x_, override, keep, batch)
    # ---- main map
    out, D, base = run_map('main', net, x, y, rng, FR, keep_feats=True); R['main'] = out
    boot = Boot(len(y), a.B, 3816)
    R['main']['map'] = summarise_map(D, base, y, boot, 'main'); save()
    # example jet for the figure: a W jet, cell 0->1 at f 0.2
    try:
        iw = int(np.flatnonzero((y == W_) & base['use'])[0]); x0 = base['x0'][iw:iw + 1]; u, _ = transfer_u(x0, 0, 1, 0.2)
        xe, v, comp, _ = make_edit(x0, u, E.p4(x0).sum(1))
        np.savez_compressed(OUT / f'e38_fig_jet{sfx}.npz', x0=x0[0].astype(np.float32), xe=xe[0].astype(np.float32), v=v[0].astype(np.float32))
    except Exception as ex:  # noqa: BLE001
        log(f'figure jet failed: {ex}')
    # ---- naming
    R['naming'] = naming(D, base, base['x0']); save()
    # ---- references
    R['references'] = references(xp, yp, D, base, y, boot); save()
    # ---- cell holds
    R['cell_holds'] = cell_holds(net, base, D, y, a.ntrip_cell, rng); save()
    # maps for the figure
    M = R['main']['map']; np.savez_compressed(OUT / f'e38_maps{sfx}.npz', cells=np.array(M['cells']), wz=M['wz_on_WZ']['mean'], qg=M['qg_on_qg']['mean'],
                                                wz_null=M['null']['wz_on_WZ']['mean'], qg_null=M['null']['qg_on_qg']['mean'],
                                                wz_lo=M['wz_on_WZ']['lo'], wz_hi=M['wz_on_WZ']['hi'], qg_lo=M['qg_on_qg']['lo'], qg_hi=M['qg_on_qg']['hi'],
                                                **{f'S{u}': M['S_mean_sd'][f'S{u}'] for u in LIVE_S})
    del D
    # ---- E36 holds
    if (OUT / 'e36_edits.npz').exists():
        R['e36_hold'] = e36_hold(net, rng, a.ntrip)
    else:
        R['e36_hold'] = dict(error='e36_edits.npz not found'); log('E36 edits not found')
    save()
    # ---- replicate (f = 0.2)
    o2, D2, b2 = run_map('replicate', net, x2, y2, rng, (max(FR),), matched=False, rot=False); R['replicate'] = o2
    R['replicate']['map'] = summarise_map(D2, b2, y2, Boot(len(y2), a.B, 3817), 'replicate')
    m1 = np.array([c.endswith(f'@{max(FR):.2f}') for c in R['main']['map']['cells']])
    for key in ('wz_on_WZ', 'qg_on_qg'):
        r = np.corrcoef(R['main']['map'][key]['mean'][m1], R['replicate']['map'][key]['mean'])[0, 1]
        R['replicate'][f'corr_with_main_{key}'] = float(r); log(f'replicate vs main {key} cell correlation {r:.3f}')
    del D2; save()
    # ---- untrained
    R['untrained'] = untrained(x, y, a.untrained, a.unt_n, rng); save()
    log(f'wrote {JS}'); log('done')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=int, default=0); ap.add_argument('--B', type=int, default=1000)
    ap.add_argument('--ntrip', type=int, default=500); ap.add_argument('--ntrip-cell', type=int, default=30)
    ap.add_argument('--untrained', type=int, default=10); ap.add_argument('--unt-n', type=int, default=2000)
    a = ap.parse_args()
    if a.smoke: a.B, a.ntrip, a.ntrip_cell, a.untrained, a.unt_n = 100, 10, 5, 1, 100
    main(a)
