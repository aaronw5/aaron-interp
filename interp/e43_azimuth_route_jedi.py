"""E43 (stage 3; claims A4m, A4w, A1, A3; new A21az; round 21): which quantity of the hardest particles' azimuthal arrangement
about the stored jet axis JEDI-linear 3-feature n64 reads, and through which units (half-plane cuts -> s/u -> S31/S58).

PLAN (fixed before the first run)
Sample: interp.jedi_core.sample(per_class=1500, seed=2143) (W, Z, top, q, g; 7,500 jets); fit / test split by a random
  permutation (seed+1), not first-k.  All edits touch only the 14 hardest real particles and keep each particle's pT and its
  distance r to the stored axis, so M0 = sum pT, M2 = sum pT r^2 and m_A = sqrt(M0 M2) are fixed exactly.
Edits (per jet; kept where the solver converges):
  turn10   E41 / E27w arm B: hardest 14 turned in azimuth toward phi(M1) until the true mass m falls 10 %.
  ctrl10   E41v / E27w control: same |dtheta_i| with random signs, minimum-norm azimuth correction restoring the vector
           M1 = sum pT (deta, dphi), common rescale holding D = sqrt(sum_i |dx_i|^2) equal to turn10's; m held (to ~1e-5).
  dil10    all angles x a (per jet) so that m equals turn10's m: a real 10 % mass change (reference).
  Q2p/Q2m  the pT r^2-weighted quadrupole C2 = sum pT r^2 e^{2 i theta} scaled to (1 +/- 0.20) C2 at fixed orientation by
           azimuth moves of the hardest 14 (homotopy + minimum-norm Gauss-Newton), with vector M1 and the exact m held
           (M0, M2 fixed by construction).  Q2 = |C2| / M2.
  Q2cp/Q2cm matched controls: same particles, same |dtheta_i| with random signs, projected to hold vector M1, m and complex
           C2, rescaled to the edit's displacement D (alternating projection, 40 iterations).
  (top harmonic candidate from step 2, if it is not Q2: the same edit / control pair for it.)
  rot45 / rotA  whole-jet rotation by 45 deg and by +/- the per-jet median |dtheta| of the Q2p edit (rotation leaves every
           (pT, r) and every |C_n| fixed; it changes fixed-frame components).
Step 1 (checks): exact forward replica (this module's Fwd, reproduces jedi_core.Jedi.run to 0), accuracy on the sample
  (expect 0.806-0.813), l1 structure (units with angular weight vs E14's nine half-planes), edit checks.
Step 2 (naming): candidate observables per jet about the stored axis: Q1..Q4 (pT r^2-weighted harmonic magnitudes / M2),
  Q2 fixed-frame components, count-weighted q1..q4 = |sum r e^{i n theta}| / sum r and the E24x fixed-frame count dipole
  (sum deta, sum dphi) and count quadrupole (sum r cos 2theta, sum r sin 2theta), occupancy and pT in each of the nine
  half-planes (a_j > 0 in the network's own first layer) and all nine together, pT^2-weighted azimuthal circular variance.
  Targets: change of the W-Z logit margin (logit_W - logit_Z) and of each live S unit under ctrl10 / turn10 / both.
  Held-out R^2 (fit jets -> test jets) of a linear fit on the candidate's change; MARS hinge fit (<= 6 terms; hinges on the
  candidate's before / after levels) for the margin, S31 and S58; shuffled-pairing null (candidate changes permuted across
  jets, 200 draws for margin / S31 / S58, 50 otherwise); Holm over candidates per target; 95 % bootstrap CI over test jets.
Step 3 (causal, Q2): margin change, calls changed, W->Z / Z->W flow, McNemar edit vs matched control; R_Q = mean margin change
  under the edit / (b x mean edit dQ2), b = slope of the margin difference on the Q2 difference across real nearest-neighbour
  jet pairs (matched on log m, log pT, N, log M2/M0, pT^D; regression also on those differences); C2ST (HistGBDT, 5-fold AUC)
  natural vs edited on axis-free, axis-relative and axis-relative + harmonic features.
Step 4 (route): rotations: per half-plane P unit change and occupancy change.  Holds (unit held at its value on the unedited
  jet; per particle for a, s, u): a of the nine half-planes (structural complete block), P nine half-planes (context
  branch), s polygon units (E14: s43, s60, s53, s32, s52), the u units they feed, g (all 64), S31, S58, S31+S58, and each
  live S unit and each half-plane P unit singly.  Removal share = 1 - sum_j |dmargin_held| / sum_j |dmargin_free|.  Null:
  200 random holds of the same size among live units of the same layer (excluding the named ones; holds of units the edit
  does not move are evaluated without a forward pass); rank p, Holm over named holds.
Step 5 (controls): 5 untrained redraws, 20 full first-layer permutations (all 192 fused quantised l1 kernel entries permuted
  across units and input columns, biases and all downstream trained layers kept), 3 E41 row-permuted copies (distinct
  permutations p0, p3, p5).  Per net: margin response to each edit / sd of its natural margin, calls changed, Q2-specific
  response (Q2 edits minus their controls), S31+S58 hold removal.  H0: trained statistic inside the permuted-l1 distribution.
Outputs ($AI_RESULTS/jedi/): e43_main.json, e43_example.npz (figure), log lines per step.
  python -m interp.e43_azimuth_route_jedi [--seed 2143] [--pc 1500] [--nrand 200] [--nhold 600] [--smoke]
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0 = time.time(); W_, Z_ = 2, 3
HP = [29, 6, 40, 57, 3, 43, 30, 36, 46]            # E14: the nine l1 half-plane statements
SPOLY = [43, 60, 53, 32, 52]                       # E14: s units that sum half-planes
UPOLY = [1, 29, 56, 60, 52, 62, 43, 13, 30, 51, 63, 17, 20, 50, 40]   # E14: u units fed by those s units
CTX = [31, 58]
warnings.filterwarnings('ignore')


def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def qci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return np.quantile(v, [.025, .975]).tolist() if len(v) else [None, None]


# ================================================================================================ forward replica
class Fwd:
    """jedi_core's forward pass, re-jitted with (i) optional replacement of the fused l1 kernel and (ii) per-jet override values
    reshaped to each layer's internal shape.  O = {stage: (mask (units,), values (B, units) or (B, n, units))}."""

    def __init__(self, net, l1_kernel=None):
        import jax, jax.numpy as jnp
        self.net = net; L = net.L; self.n = net.n
        KB = {k: (jnp.asarray(np.asarray(v[0], np.float32)), jnp.asarray(np.asarray(v[1], np.float32))) for k, v in net.KB.items()}
        if l1_kernel is not None: KB['l1'] = (jnp.asarray(np.asarray(l1_kernel, np.float32)), KB['l1'][1])
        self.K1 = np.asarray(KB['l1'][0]); self.b1 = np.asarray(KB['l1'][1])

        def dense(k, x, act=True):
            z = jnp.einsum(L[k].equation, L[k].iq(x, training=False), KB[k][0]) + KB[k][1]
            return L[k].activation(z) if act and L[k].activation is not None else z

        def pool(k, a):
            return jnp.sum(L[k].iq(a, training=False), axis=L[k].axes, keepdims=L[k].keepdims) * L[k].scale

        def fwd(x, O, keep):
            def ov(name, t):
                if name not in O: return t
                m, v = O[name]
                if v.ndim >= 2: v = v.reshape(t.shape)
                return t * (1 - m) + m * v
            a = ov('a', dense('l1', x)); P = ov('P', pool('qs', a)); g = ov('g', dense('dd', P)); s = ov('s', dense('l2', a))
            qs_, qd_ = L['add'].iq([s, g], training=False)
            u = ov('u', dense('l3', qs_ + qd_)); S = ov('S', pool('qs1', u))
            h4 = dense('d4', S); h5 = dense('d5', h4); z = dense('d6', h5, act=False); lo = dense('d7', jax.nn.relu(z))
            J = x.shape[0]
            D = dict(a=a, P=P.reshape(J, -1), g=g.reshape(J, -1), s=s, u=u, S=S.reshape(J, -1), logits=lo.reshape(J, -1))
            return {k: D[k] for k in keep}
        self._f = jax.jit(fwd, static_argnames=('keep',))

    def _batches(self, x, B):
        for i in range(0, len(x), B):
            xb = np.asarray(x[i:i + B], np.float32); nb = len(xb)
            if nb < B: xb = np.concatenate([xb, np.zeros((B - nb,) + xb.shape[1:], np.float32)])
            yield i, nb, self.net.inputs(xb, self.n)

    def run(self, x, keep=('logits',), B=500):
        keep = tuple(keep); out = {k: [] for k in keep}
        for i, nb, xi in self._batches(x, B):
            r = self._f(xi, {}, keep)
            for k in keep: out[k].append(np.asarray(r[k][:nb], np.float32))
        return {k: np.concatenate(v) for k, v in out.items()}

    def hold_run(self, xe, x0, holds, keep=('logits',), B=500):
        """holds: list of {stage: mask (64,)}; each stage held at its value on x0 (same jets) while xe runs.
        -> list (one per hold) of dict of keep arrays"""
        keep = tuple(keep); need = tuple(sorted({s for h in holds for s in h}))
        out = [{k: [] for k in keep} for _ in holds]
        for (i, nb, xi0), (_, _, xie) in zip(self._batches(x0, B), self._batches(xe, B)):
            base = self._f(xi0, {}, need) if need else {}
            for h, H in enumerate(holds):
                O = {s: (np.asarray(m, np.float32), base[s]) for s, m in H.items()}
                r = self._f(xie, O, keep)
                for k in keep: out[h][k].append(np.asarray(r[k][:nb], np.float32))
        return [{k: np.concatenate(v) for k, v in o.items()} for o in out]


def margin(lo): return np.asarray(lo[:, W_] - lo[:, Z_], np.float64)


# ================================================================================================ geometry and edits
def polar(x):
    x = np.asarray(x, np.float64); pt = x[..., 0]; real = pt > 0
    return pt, real, np.hypot(x[..., 1], x[..., 2]) * real, np.arctan2(x[..., 2], x[..., 1])


def hard14(pt, real):
    rank = np.argsort(np.argsort(-np.where(real, pt, -1), 1), 1); return real & (rank < 14)


def harm(w, th, n): return (w * np.cos(n * th)).sum(-1), (w * np.sin(n * th)).sum(-1)


def m2parts(pt, r, th):
    eta, phi = r * np.cos(th), r * np.sin(th)
    E = (pt * np.cosh(eta)).sum(-1); px = (pt * np.cos(phi)).sum(-1); py = (pt * np.sin(phi)).sum(-1); pz = (pt * np.sinh(eta)).sum(-1)
    return eta, phi, E, px, py, pz, E ** 2 - px ** 2 - py ** 2 - pz ** 2


def cons(pt, r, th, w, n, sub, M10, Ct, m20, M0, Wn):
    """residuals (J, 5) [M1x, M1y, Re C, Im C, m^2] (scaled) and Jacobian (J, 5, P) wrt theta of the movable particles"""
    eta, phi, E, px, py, pz, m2 = m2parts(pt, r, th); de, dp = -phi * sub, eta * sub
    M1x, M1y = (pt * eta).sum(-1), (pt * phi).sum(-1); Cr, Ci = harm(w, th, n)
    dm2 = 2 * E[:, None] * pt * np.sinh(eta) * de + 2 * px[:, None] * pt * np.sin(phi) * dp - 2 * py[:, None] * pt * np.cos(phi) * dp \
        - 2 * pz[:, None] * pt * np.cosh(eta) * de
    res = np.stack([(M1x - M10[:, 0]) / M0, (M1y - M10[:, 1]) / M0, (Cr - Ct[:, 0]) / Wn, (Ci - Ct[:, 1]) / Wn, (m2 - m20) / (2 * m20)], 1)
    Jm = np.stack([pt * de / M0[:, None], pt * dp / M0[:, None], -n * w * np.sin(n * th) * sub / Wn[:, None],
                   n * w * np.cos(n * th) * sub / Wn[:, None], dm2 / (2 * m20[:, None])], 1)
    return res, Jm


def gn_step(res, Jm, cap=0.3):
    G = np.einsum('jap,jbp->jab', Jm, Jm); G = G + (1e-14 + 1e-10 * np.trace(G, axis1=1, axis2=2))[:, None, None] * np.eye(G.shape[1])
    d = -np.einsum('jap,ja->jp', Jm, np.linalg.solve(G, res[..., None])[..., 0])
    mx = np.abs(d).max(1); return d * np.minimum(1.0, cap / np.maximum(mx, 1e-30))[:, None]


def harm_weight(kind, pt, r, real):
    return pt * r ** 2 * real if kind == 'pt' else r * real


def ok_cons(res, C, Ct, C0):
    return np.isfinite(res).all(1) & (np.abs(res[:, :2]).max(1) <= 1e-6) & (np.abs(res[:, 4]) <= 1e-5) & \
        (np.hypot(*(C - Ct).T) <= 1e-3 * np.maximum(np.hypot(*C0.T), 1e-30))


def solve_harm(pt, r, th, real, sub, kind, n, s, K=12, it=6):
    """scale the complex harmonic C_n (weight kind) to (1 + s) C_n at fixed orientation, holding vector M1 and m"""
    w = harm_weight(kind, pt, r, real); Wn = w.sum(1); M0 = pt.sum(1); M10 = np.stack([(pt * r * np.cos(th)).sum(1), (pt * r * np.sin(th)).sum(1)], 1)
    C0 = np.stack(harm(w, th, n), 1); m20 = m2parts(pt, r, th)[-1]; t2 = th.copy()
    for k in range(1, K + 1):
        Ct = C0 * (1 + s * k / K)
        for _ in range(it):
            res, Jm = cons(pt, r, t2, w, n, sub, M10, Ct, m20, M0, Wn); t2 = t2 + gn_step(res, Jm)
    Ct = C0 * (1 + s)
    for _ in range(10):
        res, Jm = cons(pt, r, t2, w, n, sub, M10, Ct, m20, M0, Wn); t2 = t2 + gn_step(res, Jm)
    res, _ = cons(pt, r, t2, w, n, sub, M10, Ct, m20, M0, Wn); C = np.stack(harm(w, t2, n), 1)
    q0 = np.hypot(*C0.T) / Wn
    ok = ok_cons(res, C, Ct, C0) & (q0 >= 0.01) & (m20 > 0)
    return t2, ok


def solve_harm_ctrl(pt, r, th, real, sub, kind, n, dth, DB, rng, iters=40):
    """same particles, same |dtheta_i| with random signs, projected to hold vector M1, m and complex C_n; displacement = DB"""
    from interp.e27w_mass_vs_m2_jedi import disp
    w = harm_weight(kind, pt, r, real); Wn = w.sum(1); M0 = pt.sum(1); M10 = np.stack([(pt * r * np.cos(th)).sum(1), (pt * r * np.sin(th)).sum(1)], 1)
    C0 = np.stack(harm(w, th, n), 1); m20 = m2parts(pt, r, th)[-1]
    dl = np.abs(dth) * rng.choice((-1.0, 1.0), dth.shape) * sub
    for _ in range(iters):
        for _ in range(2):
            res, Jm = cons(pt, r, th + dl, w, n, sub, M10, C0, m20, M0, Wn); dl = dl + gn_step(res, Jm)
        D = disp(r, th, th + dl); dl = dl * np.where(D > 0, DB / np.maximum(D, 1e-15), 1.0)[:, None]
    for _ in range(10):
        res, Jm = cons(pt, r, th + dl, w, n, sub, M10, C0, m20, M0, Wn); dl = dl + gn_step(res, Jm)
    res, _ = cons(pt, r, th + dl, w, n, sub, M10, C0, m20, M0, Wn); C = np.stack(harm(w, th + dl, n), 1)
    ok = ok_cons(res, C, C0, C0) & (np.abs(disp(r, th, th + dl) / np.maximum(DB, 1e-15) - 1) <= 0.02) & (DB > 0)
    return th + dl, ok


def harm_edits(x, fams, rng, E, info):
    from interp.e27w_mass_vs_m2_jedi import to_x, wrap, disp
    pt, real, r, th = polar(x); sub = hard14(pt, real)
    for nm, kind, n in fams:
        for sg, s in (('p', 0.2), ('m', -0.2)):
            t2, ok = solve_harm(pt, r, th, real, sub, kind, n, s); E[f'{nm}{sg}'] = (to_x(x, real, r, t2), ok)
            dth = wrap(t2 - th) * sub; Dq = disp(r, th, t2)
            tc, okc = solve_harm_ctrl(pt, r, th, real, sub, kind, n, dth, Dq, rng); E[f'{nm}c{sg}'] = (to_x(x, real, r, tc), ok & okc)
            info[f'{nm}{sg}'] = dict(kept=int(ok.sum()), ctrl_kept=int((ok & okc).sum()), median_D=float(np.median(Dq[ok])) if ok.any() else None,
                                     median_abs_dtheta_hard=float(np.median(np.abs(dth[ok][sub[ok]]))) if ok.any() else None)
            if nm == 'Q2' and sg == 'p':
                med = np.array([np.median(np.abs(dth[j][sub[j]])) if sub[j].any() else 0.0 for j in range(len(x))])
                sgn = rng.choice((-1.0, 1.0), len(x)); E['rotA'] = (to_x(x, real, r, th + (sgn * med)[:, None]), ok)
    return E, info


def make_edits(x, seed):
    """-> dict name -> (x_edit (J, 64, 3) float32, ok (J,)), info"""
    from interp.e27w_mass_vs_m2_jedi import solve_B, solve_ctrl, M1v, to_x, wrap, disp
    from interp.x10_das_jedi import mass
    pt, real, r, th = polar(x); sub = hard14(pt, real); rng = np.random.default_rng(seed); E = {}; info = {}
    M10 = M1v(pt, r, th); phu = np.arctan2(M10[:, 1], M10[:, 0]); d = np.where(sub, wrap(phu[:, None] - th), 0.0)
    tB, okB = solve_B(pt, r, th, d, 0.10); thB = th + tB[:, None] * d; xB = to_x(x, real, r, thB); m0 = mass(x)
    dl = np.log(np.maximum(mass(xB), 1e-9) / np.maximum(m0, 1e-9)); okB = okB & np.isfinite(dl) & (np.abs(dl - np.log(0.9)) <= 0.1 * abs(np.log(0.9)))
    E['turn10'] = (xB, okB)
    DB = disp(r, th, thB); thC, okC = solve_ctrl(pt, r, th, sub, tB[:, None] * d, DB, rng); E['ctrl10'] = (to_x(x, real, r, thC), okB & okC)
    aD = mass(xB) / np.maximum(m0, 1e-9)
    for _ in range(6):
        xD = to_x(x, real, r * aD[:, None], th); aD = aD * mass(xB) / np.maximum(mass(xD), 1e-9)
    xD = to_x(x, real, r * aD[:, None], th); okD = okB & (np.abs(np.log(np.maximum(mass(xD), 1e-9) / np.maximum(mass(xB), 1e-9))) <= 1e-3)
    E['dil10'] = (xD, okD)
    harm_edits(x, [('Q2', 'pt', 2)], rng, E, info)
    E['rot45'] = (to_x(x, real, r, th + np.pi / 4), np.ones(len(x), bool))
    info['turn10'] = dict(kept=int(okB.sum())); info['ctrl10'] = dict(kept=int(E['ctrl10'][1].sum())); info['dil10'] = dict(kept=int(okD.sum()))
    return E, info, sub


# ================================================================================================ observables
def jet_obs(x):
    from interp.x10_das_jedi import mass
    pt, real, r, th = polar(x); M0 = pt.sum(1); M2 = (pt * r ** 2).sum(1)
    M1 = np.stack([(pt * r * np.cos(th)).sum(1), (pt * r * np.sin(th)).sum(1)], 1); m = mass(x)
    return dict(M0=M0, M2=M2, M1x=M1[:, 0], M1y=M1[:, 1], m=m, mA=np.sqrt(M0 * M2), N=real.sum(1),
                ptD=np.sqrt((pt ** 2).sum(1)) / M0)


def cand_groups(x, occ, ptin):
    """candidate arrangement observables (per jet); occ / ptin (J, 9): count and pT of real particles with a_j > 0"""
    pt, real, r, th = polar(x); M2 = (pt * r ** 2).sum(1); Sr = np.maximum(r.sum(1), 1e-12); w = pt * r ** 2; G = {}
    for n in (1, 2, 3, 4):
        c, s = harm(w, th, n); G[f'Q{n}'] = (np.hypot(c, s) / M2)[:, None]
        c2, s2 = harm(r * real, th, n); G[f'q{n}'] = (np.hypot(c2, s2) / Sr)[:, None]
    G['Q2vec'] = np.stack(harm(w, th, 2), 1) / M2[:, None]
    G['q2vec_E24x'] = np.stack(harm(r * real, th, 2), 1)                       # sum r cos 2a, sum r sin 2a (count quadrupole)
    de, dp = r * np.cos(th), r * np.sin(th); G['q1vec_E24x'] = np.stack([(de * real).sum(1), (dp * real).sum(1)], 1)   # sum deta, sum dphi
    for k, u in enumerate(HP):
        G[f'occ_a{u}'] = occ[:, k:k + 1]; G[f'pt_a{u}'] = ptin[:, k:k + 1]
    G['occ_all9'] = occ; G['pt_all9'] = ptin
    p2 = pt ** 2 * real; c, s = harm(p2, th, 1); G['spread_ptD'] = (1 - np.hypot(c, s) / np.maximum(p2.sum(1), 1e-30))[:, None]
    return {k: np.asarray(v, np.float64) for k, v in G.items()}


def occupancy(a, x):
    pt = np.asarray(x[..., 0], np.float64); real = (pt > 0)[..., None]; on = (a[..., HP] > 0) & real
    return on.sum(1).astype(np.float64), (on * pt[..., None]).sum(1)


# ================================================================================================ fits
def lin_fit(Xf, yf, Xt):
    A = np.column_stack([Xf, np.ones(len(Xf))]); w = np.linalg.lstsq(A, yf, rcond=None)[0]
    return np.column_stack([Xt, np.ones(len(Xt))]) @ w


def r2w(y, p, Wb=None):
    pt = float(1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30))
    if Wb is None: return pt
    sw = Wb.sum(1); my = (Wb @ y) / sw; sse = Wb @ ((y - p) ** 2); sst = Wb @ (y ** 2) - sw * my ** 2
    return pt, qci(1 - sse / np.maximum(sst, 1e-30))


def mars_cols(D0, D1):
    """candidate columns for a hinge fit: change, and hinge-of-level changes at quantiles of the before level"""
    cols = {}
    for k in range(D0.shape[1]):
        cols[f'd{k}'] = D1[:, k] - D0[:, k]
        for q in (.1, .25, .5, .75, .9):
            t = np.quantile(D0[:, k], q)
            cols[f'up{k}_{q}'] = np.maximum(0, D1[:, k] - t) - np.maximum(0, D0[:, k] - t)
            cols[f'dn{k}_{q}'] = np.maximum(0, t - D1[:, k]) - np.maximum(0, t - D0[:, k])
    return cols


def mars_r2(cf, yf, ct, yt, maxt=6):
    names = list(cf); chosen = []; Af = [np.ones(len(yf))]; At = [np.ones(len(yt))]; best_r2 = None
    for _ in range(maxt):
        best = None
        for nm in names:
            if nm in chosen: continue
            A = np.column_stack(Af + [cf[nm]]); w = np.linalg.lstsq(A, yf, rcond=None)[0]; rss = ((A @ w - yf) ** 2).sum()
            if best is None or rss < best[0]: best = (rss, nm)
        if best is None: break
        chosen.append(best[1]); Af.append(cf[best[1]]); At.append(ct[best[1]])
    A = np.column_stack(Af); w = np.linalg.lstsq(A, yf, rcond=None)[0]; p = np.column_stack(At) @ w
    best_r2 = r2w(yt, p); return best_r2, chosen, p


def boot_mean(v, Wb): return (Wb @ v) / Wb.sum(1)


# ================================================================================================ control nets
def l1_permuted(net, seed):
    K = np.asarray(net.KB['l1'][0], np.float64); rng = np.random.default_rng(50_000 + seed)
    return K.reshape(-1)[rng.permutation(K.size)].reshape(K.shape)


def angular_units(K):
    """l1 units with a non-zero weight on deta or dphi (rows 1, 2 of the (3, 64) fused kernel)"""
    K = np.asarray(K); K = K if K.shape[0] == 3 else K.T
    return np.flatnonzero((np.abs(K[1]) > 0) | (np.abs(K[2]) > 0)).tolist(), np.flatnonzero(np.abs(K).sum(0) > 0).tolist()


# ================================================================================================ main
def main(a):
    from interp.jedi_core import Jedi, sample
    from interp.x10_das_jedi import clean, holm
    from interp.h12_boost_panel import mcnemar
    import interp.e41_true_mass_layers_jedi as E41
    OUT.mkdir(parents=True, exist_ok=True); tag = '_smoke' if a.smoke else ('' if a.seed == 2143 else f'_s{a.seed}')
    JS = OUT / f'e43_main{tag}.json'; R = dict(node='E43', plan='module docstring (fixed before the first run)', seed=a.seed, per_class=a.pc)
    save = lambda: JS.write_text(json.dumps(clean(R), indent=1))  # noqa: E731
    NB = 200 if a.smoke else 1000; NR = a.nrand; NSH = 20 if a.smoke else 200

    # ---------------------------------------------------------------- step 1: sample, network, checks
    x, y, _ = sample(a.pc, a.seed); N = len(x); perm = np.random.default_rng(a.seed + 1).permutation(N)
    isfit = np.zeros(N, bool); isfit[perm[:N // 2]] = True; ist = ~isfit
    R['sample'] = dict(call=f'interp.jedi_core.sample(per_class={a.pc}, seed={a.seed})', N=N, N_fit=int(isfit.sum()), N_test=int(ist.sum()),
                       split=f'random permutation seed {a.seed + 1}, first half fit')
    net = Jedi(64); F = Fwd(net); t = time.time()
    ref = net.run(x[:1000], keep=('logits', 'S'), batch=500); mine = F.run(x[:1000], keep=('logits', 'S'))
    R['replica_check'] = dict(max_abs_dlogit=float(np.abs(ref['logits'] - mine['logits']).max()), max_abs_dS=float(np.abs(ref['S'] - mine['S']).max()))
    base = F.run(x, keep=('logits', 'S', 'P', 'g')); dt = time.time() - t
    acc = float((base['logits'].argmax(1) == y).mean()); R['accuracy'] = dict(all=acc, fit=float((base['logits'][isfit].argmax(1) == y[isfit]).mean()),
                                                                              test=float((base['logits'][ist].argmax(1) == y[ist]).mean()), expected='0.806-0.813')
    ang, live1 = angular_units(F.K1); R['l1_structure'] = dict(angular_units=ang, live_units=live1, HP_E14=HP, match=sorted(ang) == sorted(HP),
                                                               kernel_shape=list(F.K1.shape))
    log(f'replica {R["replica_check"]}; accuracy {acc:.4f}; forward of {N + 1000} jets {dt:.0f} s; l1 angular units {ang} (E14 HP match {R["l1_structure"]["match"]}); live {live1}')
    save()
    # live units per layer (natural fit jets, real particles)
    t = time.time(); xs = x[isfit][:1000]; Rs = F.run(xs, keep=('a', 'P', 'g', 's', 'u', 'S')); realm = xs[..., 0] > 0; live = {}
    for st in ('a', 's', 'u'): v = Rs[st][realm]; live[st] = np.flatnonzero(v.std(0) > 1e-7).tolist()
    for st in ('P', 'g', 'S'): live[st] = np.flatnonzero(Rs[st].std(0) > 1e-7).tolist()
    del Rs; R['live_units'] = live; log(f'live units: ' + ', '.join(f'{k} {len(v)}' for k, v in live.items()) + f' ({time.time() - t:.0f} s)')
    sdS = base['S'].std(0); sdP = base['P'].std(0); m0all = margin(base['logits'])

    # ---------------------------------------------------------------- edits
    t = time.time(); Ed, einfo, sub = make_edits(x, a.seed + 7); R['edits'] = dict(info=einfo); log(f'edits ({time.time() - t:.0f} s): {json.dumps(einfo)}')
    o0 = jet_obs(x); chk = {}
    for k, (xe, ok) in Ed.items():
        if not ok.any(): chk[k] = dict(n=0); continue
        oe = jet_obs(xe[ok]); b = {q: v[ok] for q, v in o0.items()}
        pt, real, r, th = polar(x[ok]); _, _, r2, th2 = polar(xe[ok])
        from interp.e27w_mass_vs_m2_jedi import disp
        c0, c1 = cand_groups(x[ok], np.zeros((ok.sum(), 9)), np.zeros((ok.sum(), 9))), cand_groups(xe[ok], np.zeros((ok.sum(), 9)), np.zeros((ok.sum(), 9)))
        chk[k] = dict(n=int(ok.sum()), median_dlogm=float(np.median(np.log(oe['m'] / b['m']))), max_abs_dlogm=float(np.abs(np.log(oe['m'] / b['m'])).max()),
                      max_abs_dlogmA=float(np.abs(np.log(oe['mA'] / b['mA'])).max()), max_abs_dM1_over_M0=float((np.hypot(oe['M1x'] - b['M1x'], oe['M1y'] - b['M1y']) / b['M0']).max()),
                      median_M1_over_M0_after=float(np.median(np.hypot(oe['M1x'], oe['M1y']) / oe['M0'])), max_abs_dr=float(np.abs(r2 - r).max()),
                      median_D=float(np.median(disp(r, th, th2))),
                      **{f'median_rel_d{q}': float(np.median(c1[q][:, 0] / np.maximum(c0[q][:, 0], 1e-12) - 1)) for q in ('Q1', 'Q2', 'Q3', 'Q4', 'q2')},
                      **{f'median_abs_d{q}': float(np.median(np.abs(c1[q][:, 0] - c0[q][:, 0]))) for q in ('Q1', 'Q2', 'Q3', 'Q4', 'q2')})
    R['edits']['checks'] = chk; log('edit checks:', json.dumps(clean({k: (v['n'], v.get('median_dlogm'), v.get('max_abs_dlogmA'), v.get('median_rel_dQ2'), v.get('median_D')) for k, v in chk.items()})))
    save()

    # ---------------------------------------------------------------- free runs of every edit (trained)
    t = time.time(); FR = {}
    def run_with_occ(xx):
        out = {k: [] for k in ('logits', 'S', 'P', 'g')}; oc, pp = [], []
        for i in range(0, len(xx), 2000):
            r_ = F.run(xx[i:i + 2000], keep=('logits', 'S', 'P', 'g', 'a'))
            for k in out: out[k].append(r_[k])
            o_, p_ = occupancy(r_['a'], xx[i:i + 2000]); oc.append(o_); pp.append(p_)
        o = {k: np.concatenate(v) for k, v in out.items()}; o['occ'] = np.concatenate(oc); o['ptin'] = np.concatenate(pp); return o
    FR['nat'] = run_with_occ(x)
    for k, (xe, ok) in Ed.items(): FR[k] = run_with_occ(xe)
    log(f'free runs of {len(Ed)} edits ({time.time() - t:.0f} s)')
    c0 = FR['nat']['logits'].argmax(1); mg = {k: margin(v['logits']) for k, v in FR.items()}; cl = {k: v['logits'].argmax(1) for k, v in FR.items()}
    G0 = cand_groups(x, FR['nat']['occ'], FR['nat']['ptin'])

    # ---------------------------------------------------------------- step 2: naming on ctrl10 / turn10
    t = time.time(); liveS = live['S']; NM = {}
    resp = {}
    for k in ('ctrl10', 'turn10', 'Q2p', 'Q2m', 'Q2cp', 'Q2cm', 'rot45', 'rotA', 'dil10'):
        ok = Ed[k][1]; dS = FR[k]['S'][ok] - FR['nat']['S'][ok]
        resp[k] = dict(n=int(ok.sum()), mean_abs_dS_over_sd={int(u): float(np.abs(dS[:, u]).mean() / max(sdS[u], 1e-12)) for u in liveS},
                       mean_abs_dmargin=float(np.abs(mg[k][ok] - mg['nat'][ok]).mean()), mean_dmargin=float((mg[k][ok] - mg['nat'][ok]).mean()),
                       mean_abs_dP_HP_over_sd={int(u): float(np.abs(FR[k]['P'][ok][:, u] - FR['nat']['P'][ok][:, u]).mean() / max(sdP[u], 1e-12)) for u in HP},
                       mean_abs_docc={int(u): float(np.abs(FR[k]['occ'][ok][:, j] - FR['nat']['occ'][ok][:, j]).mean()) for j, u in enumerate(HP)},
                       frac_jets_occ_changed={int(u): float((FR[k]['occ'][ok][:, j] != FR['nat']['occ'][ok][:, j]).mean()) for j, u in enumerate(HP)},
                       mean_abs_dg_L1=float(np.abs(FR[k]['g'][ok] - FR['nat']['g'][ok]).sum(1).mean()))
        top = sorted(resp[k]['mean_abs_dS_over_sd'].items(), key=lambda kv: -kv[1])[:6]
        log(f'  response {k}: |dmargin| {resp[k]["mean_abs_dmargin"]:.3f}; top S |dS|/sd {[(u, round(v, 3)) for u, v in top]}')
    R['responses'] = resp; save()
    targets = ['margin'] + [f'S{u}' for u in liveS]
    def tgt(k, nm, ok):
        if nm == 'margin': return mg[k][ok] - mg['nat'][ok]
        u = int(nm[1:]); return (FR[k]['S'][ok][:, u] - FR['nat']['S'][ok][:, u]) / max(sdS[u], 1e-12)
    for eset in (('ctrl10',), ('turn10',), ('ctrl10', 'turn10')):
        ename = '+'.join(eset); rows = []
        for k in eset:
            ok = Ed[k][1]; Gk = cand_groups(Ed[k][0], FR[k]['occ'], FR[k]['ptin'])
            rows.append((k, ok, Gk))
        fitm = np.concatenate([isfit[ok] for _, ok, _ in rows]); D0 = {}; D1 = {}
        for g in G0: D0[g] = np.concatenate([G0[g][ok] for _, ok, _ in rows]); D1[g] = np.concatenate([Gk[g][ok] for _, ok, Gk in rows])
        T = {nm: np.concatenate([tgt(k, nm, ok) for k, ok, _ in rows]) for nm in targets}
        nte = int((~fitm).sum()); Wb = np.random.default_rng(4301).poisson(1.0, (NB, nte)).astype(np.float64); rng = np.random.default_rng(4302)
        out = {}
        for nm in targets:
            yf, yt = T[nm][fitm], T[nm][~fitm]; nsh = NSH if nm in ('margin', 'S31', 'S58') else min(50, NSH); o = {}
            if yt.std() < 1e-9: out[nm] = dict(constant=True); continue
            allX = np.column_stack([D1[g] - D0[g] for g in G0]); p = lin_fit(allX[fitm], yf, allX[~fitm]); o['_all_linear'] = dict(r2=r2w(yt, p))
            ps = []
            for g in G0:
                X = D1[g] - D0[g]; p = lin_fit(X[fitm], yf, X[~fitm]); r2p, ci = r2w(yt, p, Wb); null = []
                for _ in range(nsh):
                    Xs = X.copy(); Xs[fitm] = X[fitm][rng.permutation(fitm.sum())]; Xs[~fitm] = X[~fitm][rng.permutation(nte)]
                    null.append(r2w(yt, lin_fit(Xs[fitm], yf, Xs[~fitm])))
                null = np.array(null); pv = float((1 + (null >= r2p).sum()) / (1 + len(null))); ps.append(pv)
                o[g] = dict(r2=r2p, ci=ci, null_q95=float(np.quantile(null, .95)), p=pv, n_shuffle=len(null))
            hp = holm(ps)
            for g, h in zip(G0, hp): o[g]['p_holm'] = float(h)
            if nm in ('margin', 'S31', 'S58'):
                rank = sorted(G0, key=lambda g: -o[g]['r2'])[:4]
                for g in rank:
                    cf = mars_cols(D0[g][fitm], D1[g][fitm])
                    # hinge thresholds from the fit rows: rebuild test columns with the same thresholds
                    ctc = {}; D0t, D1t = D0[g][~fitm], D1[g][~fitm]
                    for kk in range(D0t.shape[1]):
                        ctc[f'd{kk}'] = D1t[:, kk] - D0t[:, kk]
                        for q in (.1, .25, .5, .75, .9):
                            th_ = np.quantile(D0[g][fitm][:, kk], q)
                            ctc[f'up{kk}_{q}'] = np.maximum(0, D1t[:, kk] - th_) - np.maximum(0, D0t[:, kk] - th_)
                            ctc[f'dn{kk}_{q}'] = np.maximum(0, th_ - D1t[:, kk]) - np.maximum(0, th_ - D0t[:, kk])
                    r2m, terms, pm = mars_r2(cf, yf, ctc, yt); null = []
                    for _ in range(min(20, NSH)):
                        pi = rng.permutation(fitm.sum()); pj = rng.permutation(nte)
                        null.append(mars_r2({c: v[pi] for c, v in cf.items()}, yf, {c: v[pj] for c, v in ctc.items()}, yt)[0])
                    o[g]['mars'] = dict(r2=r2m, ci=r2w(yt, pm, Wb)[1], terms=terms, null_max=float(np.max(null)), p=float((1 + (np.array(null) >= r2m).sum()) / (1 + len(null))))
            o['_best'] = max(G0, key=lambda g: o[g]['r2']); out[nm] = o
        NM[ename] = out
        b = out['margin']; log(f'  naming {ename} ({time.time() - t:.0f} s): margin best {b["_best"]} R2 {b[b["_best"]]["r2"]:.3f}; Q2 {b["Q2"]["r2"]:.3f}, q2vec {b["q2vec_E24x"]["r2"]:.3f}, '
                               f'occ_all9 {b["occ_all9"]["r2"]:.3f}, all {b["_all_linear"]["r2"]:.3f}; S58 best {out["S58"].get("_best")} '
                               f'{out["S58"][out["S58"]["_best"]]["r2"] if "_best" in out["S58"] else None}; S31 best {out["S31"].get("_best")}')
        R['naming'] = NM; save()

    # pick the top harmonic candidate (margin, ctrl10+turn10) for an extra causal edit if it is not Q2
    harmc = {'Q1': ('pt', 1), 'Q2': ('pt', 2), 'Q3': ('pt', 3), 'Q4': ('pt', 4), 'q1': ('r', 1), 'q2': ('r', 2), 'q3': ('r', 3), 'q4': ('r', 4),
             'Q2vec': ('pt', 2), 'q2vec_E24x': ('r', 2)}
    nmc = NM['ctrl10+turn10']['margin']; tops = sorted([g for g in harmc], key=lambda g: -nmc[g]['r2']); topc = tops[0]
    extra = None
    if harmc[topc] != ('pt', 2):
        extra = (f'X{topc}', harmc[topc][0], harmc[topc][1])
        t = time.time(); E2, info2 = harm_edits(x, [extra], np.random.default_rng(a.seed + 8), {}, {})
        for k in E2: Ed[k] = E2[k]; FR[k] = run_with_occ(E2[k][0])
        R['edits']['info_extra'] = info2
        mg = {k: margin(v['logits']) for k, v in FR.items()}; cl = {k: v['logits'].argmax(1) for k, v in FR.items()}
        log(f'extra harmonic edit for top candidate {topc} -> {extra} ({time.time() - t:.0f} s): {json.dumps(R["edits"]["info_extra"])}')
    R['top_harmonic_candidate'] = dict(name=topc, ranking=tops, extra_edit=extra); save()

    # ---------------------------------------------------------------- step 3: causal Q2 edit
    t = time.time(); C3 = {}
    # pair calibration on natural jets: nearest neighbour in covariates, slope of dmargin on dQ (+ covariate differences)
    from sklearn.neighbors import NearestNeighbors
    ob = jet_obs(x); cov = np.stack([np.log(np.maximum(ob['m'], 1e-3)), np.log(ob['M0']), ob['N'], np.log(np.maximum(ob['M2'] / ob['M0'], 1e-12)), ob['ptD']], 1)
    cz = (cov - cov[isfit].mean(0)) / cov[isfit].std(0); fi = np.flatnonzero(isfit)
    nn = NearestNeighbors(n_neighbors=2).fit(cz[fi]); _, nb = nn.kneighbors(cz); partner = np.where(fi[nb[:, 0]] == np.arange(N), fi[nb[:, 1]], fi[nb[:, 0]])
    def pair_slope(qname, rows):
        dq = G0[qname][:, 0] - G0[qname][partner, 0]; dm = m0all - m0all[partner]; dc = cz - cz[partner]
        rows = np.asarray(rows); nrow = int(rows.sum()) if rows.dtype == bool else len(rows)
        A = np.column_stack([dq, dc]); WbP = np.random.default_rng(4303).poisson(1.0, (NB, nrow)).astype(np.float64)
        b = np.linalg.lstsq(A[rows], dm[rows], rcond=None)[0][0]
        bb = np.array([np.linalg.lstsq(A[rows] * np.sqrt(w)[:, None], dm[rows] * np.sqrt(w), rcond=None)[0][0] for w in WbP[:200]])
        return float(b), qci(bb), bb
    fams = [('Q2', 'Q2')] + ([(extra[0], topc if topc in ('Q1', 'Q2', 'Q3', 'Q4', 'q1', 'q2', 'q3', 'q4') else ('Q2' if extra[1] == 'pt' else 'q2'))] if extra else [])
    for fam, qn in fams:
        b, bci, _ = pair_slope(qn, ist)   # real-jet pairs among test jets (partners from fit jets)
        o = dict(pair_slope=dict(observable=qn, b=b, ci=bci, n_pairs=int(ist.sum())))
        for sg in ('p', 'm'):
            ke, kc = f'{fam}{sg}', f'{fam}c{sg}'; ok = Ed[kc][1] & ist; n = int(ok.sum())
            if n < 20: o[sg] = dict(n=n); continue
            Wb = np.random.default_rng(4304).poisson(1.0, (NB, n)).astype(np.float64)
            Ge = cand_groups(Ed[ke][0][ok], FR[ke]['occ'][ok], FR[ke]['ptin'][ok]); Gc = cand_groups(Ed[kc][0][ok], FR[kc]['occ'][ok], FR[kc]['ptin'][ok])
            dQ = Ge[qn][:, 0] - G0[qn][ok, 0]; dQc = Gc[qn][:, 0] - G0[qn][ok, 0]
            dme = mg[ke][ok] - m0all[ok]; dmc = mg[kc][ok] - m0all[ok]; pred = b * dQ
            ce, cc = cl[ke][ok] != c0[ok], cl[kc][ok] != c0[ok]; bw, bz = c0[ok] == W_, c0[ok] == Z_
            o[sg] = dict(n=n, mean_dQ=float(dQ.mean()), mean_dQ_ctrl=float(dQc.mean()), mean_dmargin=float(dme.mean()), mean_dmargin_ci=qci(boot_mean(dme, Wb)),
                         mean_dmargin_ctrl=float(dmc.mean()), mean_dmargin_ctrl_ci=qci(boot_mean(dmc, Wb)),
                         mean_abs_dmargin=float(np.abs(dme).mean()), mean_abs_dmargin_ci=qci(boot_mean(np.abs(dme), Wb)),
                         mean_abs_dmargin_ctrl=float(np.abs(dmc).mean()), mean_abs_dmargin_ctrl_ci=qci(boot_mean(np.abs(dmc), Wb)),
                         abs_diff_ci=qci(boot_mean(np.abs(dme) - np.abs(dmc), Wb)), abs_diff_p=float((boot_mean(np.abs(dme) - np.abs(dmc), Wb) <= 0).mean()),
                         predicted_dmargin=float(pred.mean()), R_Q=float(dme.mean() / pred.mean()) if pred.mean() != 0 else None,
                         R_Q_ci=qci(boot_mean(dme, Wb) / boot_mean(pred, Wb)), R_Q_ctrl_subtracted=float((dme - dmc).mean() / pred.mean()) if pred.mean() != 0 else None,
                         R_Q_ctrl_subtracted_ci=qci(boot_mean(dme - dmc, Wb) / boot_mean(pred, Wb)),
                         calls_changed=float(ce.mean()), calls_changed_ci=qci(boot_mean(ce.astype(float), Wb)), calls_changed_ctrl=float(cc.mean()),
                         calls_changed_ctrl_ci=qci(boot_mean(cc.astype(float), Wb)),
                         WZ=float((cl[ke][ok][bw] == Z_).mean()) if bw.any() else None, ZW=float((cl[ke][ok][bz] == W_).mean()) if bz.any() else None,
                         WZ_ctrl=float((cl[kc][ok][bw] == Z_).mean()) if bw.any() else None, ZW_ctrl=float((cl[kc][ok][bz] == W_).mean()) if bz.any() else None,
                         n_W=int(bw.sum()), n_Z=int(bz.sum()), mcnemar_edit_vs_ctrl=mcnemar(ce, cc),
                         dS31_over_sd=float(((FR[ke]['S'][ok, 31] - FR['nat']['S'][ok, 31]) / max(sdS[31], 1e-12)).mean()),
                         dS58_over_sd=float(((FR[ke]['S'][ok, 58] - FR['nat']['S'][ok, 58]) / max(sdS[58], 1e-12)).mean()),
                         dS58_over_sd_ctrl=float(((FR[kc]['S'][ok, 58] - FR['nat']['S'][ok, 58]) / max(sdS[58], 1e-12)).mean()),
                         dS31_over_sd_ctrl=float(((FR[kc]['S'][ok, 31] - FR['nat']['S'][ok, 31]) / max(sdS[31], 1e-12)).mean()))
            log(f'  {ke} (n {n}): dQ {dQ.mean():+.4f} (ctrl {dQc.mean():+.5f}); dmargin {dme.mean():+.3f} (ctrl {dmc.mean():+.3f}); |dmargin| {np.abs(dme).mean():.3f} vs ctrl {np.abs(dmc).mean():.3f}; '
                f'pred {pred.mean():+.4f} R_Q {o[sg]["R_Q"]}; calls {ce.mean():.4f} vs {cc.mean():.4f} McNemar {o[sg]["mcnemar_edit_vs_ctrl"]}; dS58/sd {o[sg]["dS58_over_sd"]:+.3f} ctrl {o[sg]["dS58_over_sd_ctrl"]:+.3f}')
        # antisymmetric (+ minus -) response on jets with both signs kept
        okb = Ed[f'{fam}cp'][1] & Ed[f'{fam}cm'][1] & ist
        if okb.sum() > 20:
            Wb = np.random.default_rng(4305).poisson(1.0, (NB, int(okb.sum()))).astype(np.float64)
            Gp = cand_groups(Ed[f'{fam}p'][0][okb], FR[f'{fam}p']['occ'][okb], FR[f'{fam}p']['ptin'][okb]); Gm = cand_groups(Ed[f'{fam}m'][0][okb], FR[f'{fam}m']['occ'][okb], FR[f'{fam}m']['ptin'][okb])
            dQa = (Gp[qn][:, 0] - Gm[qn][:, 0]) / 2; dma = (mg[f'{fam}p'][okb] - mg[f'{fam}m'][okb]) / 2; dmca = (mg[f'{fam}cp'][okb] - mg[f'{fam}cm'][okb]) / 2
            o['antisym'] = dict(n=int(okb.sum()), mean_dQ=float(dQa.mean()), mean_dmargin=float(dma.mean()), ci=qci(boot_mean(dma, Wb)), ctrl=float(dmca.mean()), ctrl_ci=qci(boot_mean(dmca, Wb)),
                                predicted=float(b * dQa.mean()), R_Q=float(dma.mean() / (b * dQa.mean())) if b * dQa.mean() != 0 else None,
                                R_Q_ci=qci(boot_mean(dma, Wb) / boot_mean(b * dQa, Wb)), p_edit_le_ctrl=float((boot_mean(np.abs(dma), Wb) - boot_mean(np.abs(dmca), Wb) <= 0).mean()),
                                mean_abs=float(np.abs(dma).mean()), mean_abs_ctrl=float(np.abs(dmca).mean()))
            log(f'  {fam} antisymmetric: dQ {dQa.mean():+.4f} dmargin {dma.mean():+.4f} {o["antisym"]["ci"]} ctrl {dmca.mean():+.4f}; R_Q {o["antisym"]["R_Q"]}')
        C3[fam] = o
    # ctrl10 / turn10 calls (context, as E41v)
    okc = Ed['ctrl10'][1] & ist
    C3['turn10_vs_ctrl10'] = dict(n=int(okc.sum()), turn=float((cl['turn10'][okc] != c0[okc]).mean()), ctrl=float((cl['ctrl10'][okc] != c0[okc]).mean()),
                                  dil=float((cl['dil10'][okc & Ed['dil10'][1]] != c0[okc & Ed['dil10'][1]]).mean()),
                                  mcnemar=mcnemar(cl['turn10'][okc] != c0[okc], cl['ctrl10'][okc] != c0[okc]))
    R['causal'] = C3; save(); log(f'step 3 causal ({time.time() - t:.0f} s); turn vs ctrl calls {C3["turn10_vs_ctrl10"]}')
    # C2ST
    t = time.time(); FAf, FXf = E41.c2st_feats(x); FQ = np.column_stack([FXf] + [G0[q] for q in ('Q1', 'Q2', 'Q3', 'Q4')])
    half = np.random.default_rng(4306).permutation(N); h0, h1 = half[:N // 2], half[N // 2:]; CC = {}
    for k in [k for k in Ed if k not in ('rot45', 'rotA')] + ['rot45']:
        ok1 = h1[Ed[k][1][h1]]
        if len(ok1) < 100: continue
        xe = Ed[k][0][ok1]; Fa, Fx = E41.c2st_feats(xe); Ge = cand_groups(xe, np.zeros((len(xe), 9)), np.zeros((len(xe), 9)))
        Fq = np.column_stack([Fx] + [Ge[q] for q in ('Q1', 'Q2', 'Q3', 'Q4')])
        CC[k] = dict(axis_free=E41.c2st_auc(FAf[h0], Fa, 4307), axis_relative=E41.c2st_auc(FXf[h0], Fx, 4308), axis_relative_plus_harmonics=E41.c2st_auc(FQ[h0], Fq, 4309))
    CC['null_nat_vs_nat'] = dict(axis_free=E41.c2st_auc(FAf[h0], FAf[h1], 4310), axis_relative=E41.c2st_auc(FXf[h0], FXf[h1], 4311),
                                 axis_relative_plus_harmonics=E41.c2st_auc(FQ[h0], FQ[h1], 4312))
    R['c2st'] = CC; save(); log(f'C2ST ({time.time() - t:.0f} s):', json.dumps(clean({k: (v['axis_free']['auc'], v['axis_relative']['auc'], v['axis_relative_plus_harmonics']['auc']) for k, v in CC.items()})))

    # ---------------------------------------------------------------- step 4: route (holds)
    t = time.time(); m64 = lambda us: np.isin(np.arange(64), us).astype(np.float32)  # noqa: E731
    named = {'a_HP9': {'a': m64(HP)}, 'P_HP9': {'P': m64(HP)}, 's_poly5': {'s': m64(SPOLY)}, 'u_poly15': {'u': m64(UPOLY)}, 'g_all': {'g': np.ones(64, np.float32)},
             'S31': {'S': m64([31])}, 'S58': {'S': m64([58])}, 'S31+S58': {'S': m64(CTX)}, 'P_HP9+s_poly5': {'P': m64(HP), 's': m64(SPOLY)},
             'g_all+s_poly5': {'g': np.ones(64, np.float32), 's': m64(SPOLY)}}
    singles = {f'S{u}': {'S': m64([u])} for u in liveS if u not in CTX}
    singles.update({f'P{u}': {'P': m64([u])} for u in HP})
    hold_edits = ['ctrl10', 'turn10', 'Q2p', 'Q2m'] + ([f'{extra[0]}p', f'{extra[0]}m'] if extra else []) + ['rotA', 'rot45']
    RT = {}; idx_all = {}
    for k in hold_edits:
        ok = np.flatnonzero(Ed[k][1] & ist)
        if len(ok) < 20: continue
        idx_all[k] = ok; xe, xb = Ed[k][0][ok], x[ok]; d_free = mg[k][ok] - m0all[ok]; den = np.abs(d_free)
        Wb = np.random.default_rng(4313).poisson(1.0, (NB, len(ok))).astype(np.float64)
        HL = dict(named);
        if k in ('ctrl10', 'turn10', 'Q2p', 'Q2m'): HL.update(singles)
        outs = F.hold_run(xe, xb, list(HL.values()), keep=('logits',)); o = dict(n=len(ok), mean_abs_dmargin_free=float(den.mean()))
        for nm, rr in zip(HL, outs):
            dh = margin(rr['logits']) - m0all[ok]; rem = 1 - np.abs(dh).sum() / max(den.sum(), 1e-30)
            o[nm] = dict(removal=float(rem), ci=qci(1 - (Wb @ np.abs(dh)) / np.maximum(Wb @ den, 1e-30)), mean_dmargin_held=float(dh.mean()),
                         calls_changed_held=float((rr['logits'].argmax(1) != c0[ok]).mean()))
        o['calls_changed_free'] = float((cl[k][ok] != c0[ok]).mean()); RT[k] = o
        log(f'  holds {k} (n {len(ok)}, {time.time() - t:.0f} s): ' + ', '.join(f'{nm} {o[nm]["removal"]:.3f}' for nm in named))
    R['holds'] = RT; save()
    # random size-matched holds on a subset
    rngH = np.random.default_rng(4314); RN = {}
    pools = {'P': [u for u in live['P'] if u not in HP], 's': [u for u in live['s'] if u not in SPOLY], 'u': [u for u in live['u'] if u not in UPOLY],
             'S': [u for u in live['S'] if u not in CTX]}
    sizes = {'P_HP9': ('P', 9), 's_poly5': ('s', 5), 'u_poly15': ('u', 15), 'S31': ('S', 1), 'S58': ('S', 1), 'S31+S58': ('S', 2)}
    for k in [e for e in ('ctrl10', 'Q2p', 'Q2m') if e in idx_all]:
        t1 = time.time(); ok = idx_all[k]; sel = ok[rngH.permutation(len(ok))[:a.nhold]]; xe, xb = Ed[k][0][sel], x[sel]
        d_free = mg[k][sel] - m0all[sel]; den = np.abs(d_free).sum()
        nmd = F.hold_run(xe, xb, [named[h] for h in sizes], keep=('logits',)); obs = {h: 1 - np.abs(margin(r_['logits']) - m0all[sel]).sum() / max(den, 1e-30) for h, r_ in zip(sizes, nmd)}
        # unit change under the edit (to skip holds of units the edit does not move)
        Rb = F.run(xb, keep=('P', 's', 'u', 'S')); Re = F.run(xe, keep=('P', 's', 'u', 'S'))
        moved = {st: np.abs(Re[st] - Rb[st]).reshape(-1, 64).max(0) > 0 for st in ('P', 's', 'u', 'S')}; del Rb, Re
        o = dict(n=len(sel), named=obs, moved_units={st: np.flatnonzero(v).tolist() for st, v in moved.items()}, null={})
        done = {}
        for h, (st, kk) in sizes.items():
            pool = pools[st]
            if (st, kk) in done:
                vals = done[(st, kk)]
                o['null'][h] = dict(layer=st, size=kk, pool=len(pool), n_draws=NR, shared_draws=True, median=float(np.median(vals)), q95=float(np.quantile(vals, .95)),
                                    max=float(vals.max()), observed=obs[h], p_rank=float((1 + (vals >= obs[h]).sum()) / (1 + NR))); continue
            if len(pool) < kk: o['null'][h] = dict(note=f'pool of {len(pool)} live units < {kk}'); continue
            draws = [np.array(rngH.choice(pool, kk, replace=False)) for _ in range(NR)]
            vals = np.zeros(NR); todo = [i for i, d_ in enumerate(draws) if moved[st][d_].any()]
            if todo:
                outs = F.hold_run(xe, xb, [{st: m64(draws[i])} for i in todo], keep=('logits',))
                for i, rr in zip(todo, outs): vals[i] = 1 - np.abs(margin(rr['logits']) - m0all[sel]).sum() / max(den, 1e-30)
            o['null'][h] = dict(layer=st, size=kk, pool=len(pool), n_draws=NR, n_draws_moved=len(todo), median=float(np.median(vals)), q95=float(np.quantile(vals, .95)),
                                max=float(vals.max()), observed=obs[h], p_rank=float((1 + (vals >= obs[h]).sum()) / (1 + NR)))
            done[(st, kk)] = vals
        hp = holm([o['null'][h].get('p_rank', 1.0) for h in sizes])
        for h, v in zip(sizes, hp):
            if 'p_rank' in o['null'][h]: o['null'][h]['p_holm'] = float(v)
        RN[k] = o; R['holds_random'] = RN; save()
        log(f'  random holds {k} (n {len(sel)}, {time.time() - t1:.0f} s): ' + ', '.join(f'{h} obs {o["null"][h].get("observed", float("nan")):.3f} null q95 {o["null"][h].get("q95", float("nan")):.3f} p {o["null"][h].get("p_rank")}' for h in sizes))

    # ---------------------------------------------------------------- example jet for the figure
    okx = np.flatnonzero((y == W_) & Ed['Q2cp'][1] & Ed['ctrl10'][1] & ist)
    if len(okx):
        nr = (x[okx, :, 0] > 0).sum(1); ex = int(okx[np.argsort(nr)[len(okx) // 2]])
        np.savez_compressed(OUT / f'e43_example{tag}.npz', x=x[ex], x_Q2p=Ed['Q2p'][0][ex], x_Q2cp=Ed['Q2cp'][0][ex], x_Q2m=Ed['Q2m'][0][ex], x_ctrl10=Ed['ctrl10'][0][ex],
                            moved=sub[ex], K1=F.K1, b1=F.b1, shift=net.shift, scale=net.scale, label=y[ex], Q2=G0['Q2'][ex, 0],
                            Q2p=cand_groups(Ed['Q2p'][0][ex:ex + 1], np.zeros((1, 9)), np.zeros((1, 9)))['Q2'][0, 0],
                            m=ob['m'][ex], margin=m0all[ex], margin_Q2p=mg['Q2p'][ex], margin_Q2cp=mg['Q2cp'][ex])
        R['example'] = dict(index=ex, label=int(y[ex]))
    save()

    # ---------------------------------------------------------------- step 5: control networks
    t = time.time(); CN = {}
    sel = np.flatnonzero(ist & Ed['ctrl10'][1] & Ed['Q2cp'][1] & Ed['Q2cm'][1] & Ed['dil10'][1])[: (300 if a.smoke else 1500)]
    vers = ['ctrl10', 'turn10', 'dil10', 'Q2p', 'Q2m', 'Q2cp', 'Q2cm', 'rot45']
    WbC = np.random.default_rng(4315).poisson(1.0, (NB, len(sel))).astype(np.float64)
    def net_stats(Fn, name, hold=True):
        r0 = Fn.run(x[sel], keep=('logits', 'P')); mm0 = margin(r0['logits']); sd = float(mm0.std()); cc0 = r0['logits'].argmax(1)
        o = dict(acc_subset=float((cc0 == y[sel]).mean()), margin_sd=sd); ang_, _ = angular_units(Fn.K1); o['angular_units'] = ang_
        dd = {}
        for k in vers:
            rk = Fn.run(Ed[k][0][sel], keep=('logits', 'P')); dm = margin(rk['logits']) - mm0; dd[k] = dm
            o[k] = dict(mean_abs_dmargin_over_sd=float(np.abs(dm).mean() / max(sd, 1e-12)), calls_changed=float((rk['logits'].argmax(1) != cc0).mean()),
                        mean_abs_dP_angular=float(np.abs(rk['P'][:, ang_] - r0['P'][:, ang_]).mean()) if ang_ else 0.0)
        q2 = (np.abs(dd['Q2p']) + np.abs(dd['Q2m']) - np.abs(dd['Q2cp']) - np.abs(dd['Q2cm'])) / 2 / max(sd, 1e-12)
        o['Q2_specific_over_sd'] = float(q2.mean()); o['Q2_specific_over_sd_ci'] = qci(boot_mean(q2, WbC))
        o['ctrl10_over_dil10'] = float(np.abs(dd['ctrl10']).mean() / max(np.abs(dd['dil10']).mean(), 1e-12))
        o['Q2_antisym_over_sd'] = float(((dd['Q2p'] - dd['Q2m']) / 2).mean() / max(sd, 1e-12))
        if hold and sd > 0:
            for k in ('ctrl10', 'Q2p'):
                den = np.abs(dd[k]).sum(); outs = Fn.hold_run(Ed[k][0][sel], x[sel], [named['S31+S58'], named['g_all']], keep=('logits',))
                o[f'{k}_removal_S31+S58'] = float(1 - np.abs(margin(outs[0]['logits']) - mm0).sum() / max(den, 1e-30))
                o[f'{k}_removal_g'] = float(1 - np.abs(margin(outs[1]['logits']) - mm0).sum() / max(den, 1e-30))
        return o
    CN['trained'] = net_stats(F, 'trained'); log(f'  control trained: {json.dumps(clean({k: v for k, v in CN["trained"].items() if not isinstance(v, dict)}))}')
    nets = [('u', i) for i in range(2 if a.smoke else 5)] + [('l1p', i) for i in range(3 if a.smoke else 20)] + ([] if a.smoke else [('rowp', i) for i in (0, 3, 5)])
    for kind, i in nets:
        t1 = time.time()
        try:
            if kind == 'u': n_ = Jedi(64, untrained=True, seed=i); Fn = Fwd(n_); nm = f'untrained_{i}'
            elif kind == 'l1p': Fn = Fwd(net, l1_kernel=l1_permuted(net, i)); nm = f'l1perm_{i}'
            else: n_ = E41.permuted_jedi(i); Fn = Fwd(n_); nm = f'rowperm_{i}'
            CN[nm] = net_stats(Fn, nm, hold=(kind != 'u')); CN[nm]['kind'] = kind
        except Exception as e:  # noqa: BLE001
            CN[f'{kind}_{i}'] = dict(error=repr(e)); log(f'  control {kind} {i} failed: {e!r}'); continue
        c_ = CN[nm]; log(f'  control {nm} ({time.time() - t1:.0f} s): acc {c_["acc_subset"]:.3f} sd {c_["margin_sd"]:.3f} ctrl10 {c_["ctrl10"]["mean_abs_dmargin_over_sd"]:.4f} '
                         f'Q2spec {c_["Q2_specific_over_sd"]:.4f} ctrl/dil {c_["ctrl10_over_dil10"]:.3f} S31+58 rem {c_.get("ctrl10_removal_S31+S58")}')
        R['control_nets'] = CN; save()
    # tests: trained vs each control family
    T = {}
    for fam, pre in (('untrained', 'untrained_'), ('l1perm', 'l1perm_'), ('rowperm', 'rowperm_')):
        ks = [k for k in CN if k.startswith(pre) and 'error' not in CN[k]]
        if not ks: continue
        T[fam] = {}
        for stat in ('Q2_specific_over_sd', 'ctrl10_over_dil10'):
            v = np.array([CN[k][stat] for k in ks]); tv = CN['trained'][stat]
            T[fam][stat] = dict(trained=tv, values=v.tolist(), max=float(v.max()), median=float(np.median(v)), p_rank=float((1 + (v >= tv).sum()) / (1 + len(v))))
        v = np.array([CN[k]['ctrl10']['mean_abs_dmargin_over_sd'] for k in ks]); tv = CN['trained']['ctrl10']['mean_abs_dmargin_over_sd']
        T[fam]['ctrl10_abs_over_sd'] = dict(trained=tv, values=v.tolist(), max=float(v.max()), p_rank=float((1 + (v >= tv).sum()) / (1 + len(v))))
    R['control_tests'] = T; R['seconds'] = time.time() - T0; save(); log(f'control nets ({time.time() - t:.0f} s); tests {json.dumps(clean(T))[:1500]}')
    log(f'done, wrote {JS}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=2143); ap.add_argument('--pc', type=int, default=1500)
    ap.add_argument('--nrand', type=int, default=200); ap.add_argument('--nhold', type=int, default=600); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke: a.pc = min(a.pc, 150); a.nrand = min(a.nrand, 10); a.nhold = min(a.nhold, 150)
    main(a)
