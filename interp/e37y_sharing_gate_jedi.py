"""E37y (claim A4s v3, D18; stage 3): does a mass-window z neuron gate the class dependence of the h4 prong-sharing
coordinate's effect on the W/Z call at equal jet mass, does the width part of the Ez edit act separately from prong mass, and
which other h4 directions carry the Z-jet response to a more lopsided prong split?

Network: official JEDI-linear 3-feature n64 (fixed point).  Exact tail from h4 written here (d5 -> d6 -> z override -> ReLU ->
  d7, the same fused fixed-point kernels and input quantizers as x10 JNet); checked against JNet.exact and the stored run.
Samples: test T = W+Z jets of interp.jedi_core.sample(3000, 5757) (6,000 jets); fit F = W+Z jets of sample(3000, 5758) minus
  rows in T.  5-class accuracy on all 15,000 stored T jets.  Inputs of every jet (edited or not) rebuilt about its own E-scheme
  axis, pT-sorted (E37 model_xn).  Jet mass m = invariant mass of the massless constituents of the unedited jet.
The coordinate u = h4 . v; v = 1-dim DAS direction (distributed alignment search: a unit vector in the live h4 units such that
  copying the component of h4 along v from a D(dc)-edited jet into its unedited version reproduces the edited jet's W-Z logit
  margin; MSE, 400 Adam steps; E37 code) refitted on F on D(dc) self + cross pairs; oriented so corr(u, z_h) > 0 on F
  (z_h = harder-prong vector-pT fraction of the two exclusive-kT prongs).  M2 = E37's stored 2-dim DAS mass subspace at h4.
Arm 1 (steering): h4' = clip(h4 + c sd_u v), c in CS (sd_u = s.d. of u on T), exact tail; per-jet slope = least squares through
  0 of the W-Z margin change (margin = logit_W - logit_Z) on c sd_u (log-odds per unit u).  Mean slope per class in 5-GeV bins;
  matched contrast Delta = sum_b w_b (slope_W,b - slope_Z,b) / sum_b w_b over MB (80-100 GeV), w_b = nW nZ / (nW + nZ)
  (W and Z jets reweighted to a common mass distribution at 5-GeV resolution).  95 % CI: 2,000 jet bootstrap resamples.
  H0: Delta = 0.  p: permutation of the W/Z labels within 1-GeV mass bins (2,000) and two-sided bootstrap.  Checks: 1-GeV
  matched Delta; linear model slope ~ 1 + isW + (m-90)/10 + isW (m-90)/10 on 80-100 GeV jets (isW coefficient = W-Z at 90 GeV).
  Null: 200 random unit directions in the live h4 units steered by the same multiples of their own s.d. (Delta in log-odds per
  s.d. step; rank p of |Delta|).  Decomposition: margin = sum_k c_k relu(z_k) + const (c_k = d7 weight W minus Z, input
  quantizer ignored); per-unit share of Delta (descriptive; quantizer residual reported).
Arm 2 (gate): the same steering with z units held at their unsteered values (per-unit override of z, before the ReLU): named holds
  z0 (W-side mass window, near 75 GeV), z7, z14 (94-98 GeV windows), singly, in pairs and jointly (7 sets).  Removed share =
  1 - Delta(held) / Delta(free) (bootstrap CI; bootstrap p of share <= 0.5).  Nulls: 200 random d-dim holds in the live z span
  (d = set size; z' = z + (z_0 - z) P, P random rank-d projector), rank p one-sided; Holm over the 7 named sets; also every
  axis-aligned hold of other units of the same size (d = 1, 2, 3; exhaustive) as a second reference.
Arm 3 (width at fixed prong mass): Ws(alpha): inside each prong with >= 3 constituents, constituent pT multiplied by
  exp(alpha (r_i - <r>) / r_rms) (r_i = distance to the prong axis; <r>, r_rms pT-weighted mean / rms over the prong), so alpha > 0
  moves pT from the prong core to its periphery; then, iterated 4 times: prong pT rescaled to its original vector pT (z exact),
  prong translated back onto its original axis (Delta R12 exact), prong offsets about the axis scaled to restore the prong mass
  (bisection).  Prong four-vectors, hence the jet four-momentum, unchanged.  Width W_p = sum_i pT_i r_i / sum_i pT_i (r_i about
  each particle's own prong axis).  Controls: Eshuf(alpha) = the same pT factors randomly permuted within each prong, then the
  same re-solving (same pT-spectrum change, no systematic width change); Erot = rigid rotation about the jet axis with the same
  pT-weighted displacement.  Comparison edit Ewid(alpha) = each prong's offsets scaled by one common factor q (bisection) so W_p
  moves as in Ws, prong masses and jet mass co-moving (the E37xv width-only edit); EmW = dilation of all angles matched to
  Ewid's jet mass.  Per alpha (not pooled over signs: W_p can rise for both signs of alpha at fixed prong mass):
  R_w = mean(du_edit - du_control) / (beta_w mean(dW_p,edit - dW_p,control)); beta_w = natural partial slope of u on W_p across
  unedited F jets, OLS with 1, log m, (log m)^2, log pT, z_h, z_h^2, Delta R12, m_hi, m_lo (so at fixed z, Delta R12 and prong
  masses); kNN (log m, log pT) slope also given.  Joint read-out: the same OLS with W_p and the leading-particle pT fraction
  (lead) together; ratio = observed du contrast / (beta_Wp dW_p + beta_lead dlead).  Bootstrap (500) over
  T and F jets jointly.  Acceptance: kT-2 reclustering moves < 1 % of jet pT; prong centres within 0.75 of the axis; bisections.
Arm 4 (redundant Z carrier): DAS k = 1, 2, 3 at h4 inside the complement of span(v, M2) on D(+0.3) self pairs of F Z jets; test on
  T Z jets: frac = mean patched margin change / mean full change, IIA (interchange-intervention accuracy) = share of call-changing
  pairs whose patched W/Z call equals the edited jet's call; nulls 1,000 random k-dim subspaces of the same complement (rank p,
  Holm over 3); holds of v + M2 + Q_k vs v + M2 and random (2 + k + 1)-dim subspaces (200); cosines with E37's M0 subspace.
Figure data: an example W jet whose Ez(+0.15) edit (E37x: harder-prong fraction +0.15 at fixed Delta R12 and prong masses) moves the
  realised z_h by >= 0.12; steering slopes per 5-GeV bin 65-105 GeV (interp/e37x_plot.py redraws results/jedi/e37x.png).
Run: ./job.sh e37y-run 1 5 'python -m interp.e37y_sharing_gate_jedi'   (--smoke K: K jets per class, fewer steps / nulls)
Writes $AI_RESULTS/jedi/e37y[_smoke].json.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import e37_sharing_das_jedi as E  # noqa: E402
from interp import e37x_sharing_physics_jedi as X  # noqa: E402
from interp import h50_z_vs_dr as H50  # noqa: E402
from interp.h45_prong_sharing import rap, wrap, axis, prong_offsets_ok, label_sep  # noqa: E402
from interp.e35_prong_sharing_route_jedi import kt2_labels  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0 = time.time()
TSEED, FSEED = 5757, 5758
CS = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
MB = ((80, 85), (85, 90), (90, 95), (95, 100))
FIGB = tuple((lo, lo + 5) for lo in range(65, 105, 5))
NAMED = {'z0': [0], 'z7': [7], 'z14': [14], 'z0+z7': [0, 7], 'z0+z14': [0, 14], 'z7+z14': [7, 14], 'z0+z7+z14': [0, 7, 14]}
AL = (-0.5, 0.5)
MIGMAX = 0.01
LQ = ['zh', 'dR', 'm_hi', 'm_lo', 'm', 'pT', 'Wp', 'lead', 'pTD']

jsonable, qci, margin = E.jsonable, E.qci, E.margin
bmean, pboot, bweights, mcn = X.bmean, X.pboot, X.bweights, X.mcn


def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


# ------------------------------------------------------------------ exact tail from h4 with a z override
def make_tail(net):
    import jax, jax.numpy as jnp
    L = net.net.L; KB = {}
    for k in ('d5', 'd6', 'd7'):
        K, b = L[k].get_fused_qkernel_and_qbias(False, L[k].moving_mean, L[k].moving_variance); KB[k] = (jnp.asarray(K), jnp.asarray(b))

    def dense(k, v, act=True):
        z = jnp.einsum(L[k].equation, L[k].iq(v, training=False), KB[k][0]) + KB[k][1]
        return L[k].activation(z) if act and L[k].activation is not None else z

    def f(h4, Pz, zv):
        h5 = dense('d5', h4); z = dense('d6', h5, act=False); z = z + (zv - z) @ Pz
        lo = dense('d7', jax.nn.relu(z)); return lo.reshape(h4.shape[0], -1), z.reshape(h4.shape[0], -1)
    jf = jax.jit(f)
    K7 = np.asarray(KB['d7'][0], np.float64); K7 = K7.reshape(-1, K7.shape[-1])

    def run(h4, Pz=None, zv=None):
        h4 = np.asarray(h4, np.float32); J = len(h4)
        Pz = np.zeros((16, 16), np.float32) if Pz is None else np.asarray(Pz, np.float32)
        zv = np.zeros((J, 16), np.float32) if zv is None else np.asarray(zv, np.float32)
        lo, z = jf(h4, Pz, zv); return np.asarray(lo, np.float64), np.asarray(z, np.float64)
    return run, K7[:, 2] - K7[:, 3]


# ------------------------------------------------------------------ jet geometry
def mass_of(P4, nn):
    real = np.arange(P4.shape[1])[None, :] < nn[:, None]; S = (P4.astype(np.float64) * real[..., None]).sum(1)
    return np.sqrt(np.maximum(S[:, 3] ** 2 - (S[:, :3] ** 2).sum(1), 0))


def width_of(pt, y, phi, m, lab):
    tot = 0.0
    for k in (0, 1):
        s = lab == k
        if not s.any(): continue
        ya, pa, _ = H50.paxis(pt, y, phi, m, s); tot += float((pt[s] * np.hypot(y[s] - ya, wrap(phi[s] - pa))).sum())
    return tot / max(float(pt.sum()), 1e-12)


def lite(P, lab):
    """z_h, Delta R12, m_hi, m_lo, m, pT, W_p, leading-particle pT fraction, pT^D of one jet (massless constituents, labels 0/1)"""
    pt, y, phi, _ = H50.kin(P); m = np.zeros_like(pt); V = [P[lab == k].sum(0) for k in (0, 1)]
    w = np.array([math.hypot(v[0], v[1]) for v in V]); pm = [H50.mass4(v) for v in V]; T = P.sum(0)
    return np.array([w.max() / max(w.sum(), 1e-12), label_sep(P, lab), max(pm), min(pm), H50.mass4(T), math.hypot(T[0], T[1]), width_of(pt, y, phi, m, lab),
                     pt.max() / pt.sum(), math.sqrt((pt ** 2).sum()) / pt.sum()])


def shape_edit(pt, y, phi, m, lab, alpha, perm_rng=None, iters=4):
    """Ws (perm_rng None) or Eshuf (factors permuted within each prong); returns (pt2, y2, p2) or None"""
    a = [H50.paxis(pt, y, phi, m, lab == k) for k in (0, 1)]; w = [math.hypot(v[2][0], v[2][1]) for v in a]
    M0 = [H50.pmass(pt, y, phi, m, lab == k) for k in (0, 1)]
    pt2 = pt.copy(); y2, p2 = y.copy(), phi.copy(); nch = 0
    for k in (0, 1):
        s = lab == k
        if s.sum() < 3: continue
        r = np.hypot(y[s] - a[k][0], wrap(phi[s] - a[k][1])); wt = pt[s]
        rb = (wt * r).sum() / wt.sum(); rr = math.sqrt((wt * r * r).sum() / wt.sum())
        f = np.exp(alpha * (r - rb) / max(rr, 1e-9))
        if perm_rng is not None: f = perm_rng.permutation(f)
        pt2[s] = pt[s] * f; nch += 1
    if nch == 0: return None
    for _ in range(iters):
        for k in (0, 1):
            s = lab == k; V = H50.paxis(pt2, y2, p2, m, s)[2]; pt2[s] *= w[k] / max(math.hypot(V[0], V[1]), 1e-12)
            y2, p2 = H50.scale_prong(pt2, y2, p2, m, s, 1.0, target_axis=(a[k][0], a[k][1]))
        r_ = H50.restore_pm(pt2, y2, p2, m, lab, M0)
        if r_ is None: return None
        y2, p2 = r_
    return pt2, y2, p2


def scale_both(pt, y, phi, m, lab, q):
    yy, pp = y, phi
    for k in (0, 1):
        if (lab == k).sum() >= 2: yy, pp = H50.scale_prong(pt, yy, pp, m, lab == k, q)
    return yy, pp


def width_variants(P, lab, seed, j):
    """Ws, Eshuf, Ewid, EmW, Erot for alpha in AL; dict key -> P_new (accepted only), info"""
    pt, y, phi, _ = H50.kin(P); m = np.zeros_like(pt); T = P.sum(0); ya, pa = float(rap(T)), float(np.arctan2(T[1], T[0]))
    W0 = width_of(pt, y, phi, m, lab); m0 = H50.mass4(T); rng = np.random.default_rng([seed, int(j)])
    r_ = np.hypot(y - ya, wrap(phi - pa)); mr = float((pt * r_).sum() / pt.sum())
    def dil(f): return ya + f * (y - ya), pa + f * wrap(phi - pa)
    def rot(th):
        dy, dp = y - ya, wrap(phi - pa); c, s = math.cos(th), math.sin(th); return ya + c * dy - s * dp, pa + s * dy + c * dp
    out, info = {}, {}
    for al in AL:
        r1 = shape_edit(pt, y, phi, m, lab, al)
        r2 = shape_edit(pt, y, phi, m, lab, al, perm_rng=rng)
        if r1 is None or r2 is None: info[al] = 'shape_fail'; continue
        QS = H50.build(*r1, m); QH = H50.build(*r2, m)
        ok = True
        for Q, ptq in ((QS, r1[0]), (QH, r2[0])):
            ok &= (H50.migration(lab, kt2_labels(Q), ptq) < MIGMAX) and prong_offsets_ok(Q, lab)
        if not ok: info[al] = 'migration_or_offset'; continue
        Wt = width_of(*r1, m, lab)
        q = H50.bisect(lambda q_: width_of(pt, *scale_both(pt, y, phi, m, lab, q_), m, lab), 0.3, 3.0, Wt, n=45)
        if q is None: info[al] = 'wid_bisect'; continue
        QW = H50.build(pt, *scale_both(pt, y, phi, m, lab, q), m); mW = H50.mass4(QW.sum(0))
        k = H50.bisect(lambda f: H50.jm(pt, *dil(f), m), 0.3, 3.0, mW, n=45)
        if k is None: info[al] = 'dil_bisect'; continue
        QM = H50.build(pt, *dil(k), m)
        disp = float((pt * np.hypot(r1[1] - y, wrap(r1[2] - phi))).sum() / pt.sum())
        th = 2 * math.asin(min(1.0, disp / max(2 * mr, 1e-9))) * float(rng.choice([-1.0, 1.0]))
        QT = H50.build(pt, *rot(th), m)
        info[al] = 'ok'; info[('q', al)] = q; info[('k', al)] = k; info[('disp', al)] = disp
        out[('Ws', al)] = QS; out[('Eshuf', al)] = QH; out[('Ewid', al)] = QW; out[('EmW', al)] = QM; out[('Erot', al)] = QT
    return out, info, W0, m0


# ------------------------------------------------------------------ statistics
def cstat(sl, y, mT, W=None, bins=MB, nmin=5):
    """matched W-Z contrast of a per-jet quantity; returns dict, boot array (or None)"""
    rows = {}; num = den = mw = mz = 0.0; bnum = bmw = bmz = 0.0
    for lo, hi in bins:
        inb = (mT >= lo) & (mT < hi) & np.isfinite(sl); sW = np.flatnonzero(inb & (y == 2)); sZ = np.flatnonzero(inb & (y == 3))
        r = dict(nW=int(len(sW)), nZ=int(len(sZ)))
        if len(sW) >= nmin:
            r['W'] = float(sl[sW].mean())
        if len(sZ) >= nmin:
            r['Z'] = float(sl[sZ].mean())
        if len(sW) >= nmin and len(sZ) >= nmin:
            w = len(sW) * len(sZ) / (len(sW) + len(sZ)); d = r['W'] - r['Z']; r.update(diff=d, weight=w)
            num += w * d; den += w; mw += w * r['W']; mz += w * r['Z']
            if W is not None:
                bW, bZ = bmean(W, sl, sW), bmean(W, sl, sZ); r.update(W_ci=qci(bW), Z_ci=qci(bZ), diff_ci=qci(bW - bZ))
                bnum = bnum + w * (bW - bZ); bmw = bmw + w * bW; bmz = bmz + w * bZ
        rows[f'{lo}-{hi}'] = r
    if den == 0: return dict(bins=rows, delta=None), None
    o = dict(delta=num / den, W_matched=mw / den, Z_matched=mz / den, bins=rows)
    if W is not None:
        bd = bnum / den; o.update(ci=qci(bd), p_boot=pboot(bd), W_matched_ci=qci(bmw / den), Z_matched_ci=qci(bmz / den)); return o, bd
    return o, None


def delta_fast(sl, y, mT, bins=MB, nmin=5):
    num = den = 0.0
    for lo, hi in bins:
        inb = (mT >= lo) & (mT < hi); a = sl[inb & (y == 2)]; b = sl[inb & (y == 3)]
        if len(a) >= nmin and len(b) >= nmin:
            w = len(a) * len(b) / (len(a) + len(b)); num += w * (a.mean() - b.mean()); den += w
    return num / den if den else np.nan


def perm_p(sl, y, mT, nperm, seed):
    obs = delta_fast(sl, y, mT); rng = np.random.default_rng(seed)
    sel = np.flatnonzero((mT >= MB[0][0]) & (mT < MB[-1][1]) & ((y == 2) | (y == 3)))
    fb = np.floor(mT[sel]).astype(int); groups = [sel[fb == b] for b in np.unique(fb)]; cnt = 0
    for _ in range(nperm):
        yp = y.copy()
        for g in groups: yp[g] = y[rng.permutation(g)]
        cnt += abs(delta_fast(sl, yp, mT)) >= abs(obs)
    return float((1 + cnt) / (1 + nperm))


def fine_delta(sl, y, mT, W):
    bins = tuple((lo, lo + 1) for lo in range(80, 100)); o, bd = cstat(sl, y, mT, W, bins=bins, nmin=3)
    return dict(delta=o['delta'], ci=o.get('ci'), p_boot=o.get('p_boot'), n_bins_used=int(sum('diff' in r for r in o['bins'].values())))


def lin_class(sl, y, mT, W, nb=500):
    s = np.flatnonzero((mT >= 80) & (mT < 100) & ((y == 2) | (y == 3)) & np.isfinite(sl))
    isw = (y[s] == 2).astype(float); mc = (mT[s] - 90) / 10; Xm = np.stack([np.ones(len(s)), isw, mc, isw * mc], 1)
    b = np.linalg.lstsq(Xm, sl[s], rcond=None)[0]; bb = []
    for w in W[:nb].astype(np.float64):
        ww = np.sqrt(w[s]); bb.append(np.linalg.lstsq(Xm * ww[:, None], sl[s] * ww, rcond=None)[0][1])
    return dict(n=int(len(s)), isW_at_90=float(b[1]), ci=qci(bb), p_boot=pboot(bb), mass_slope_per_10GeV=float(b[2]), isW_x_mass=float(b[3]))


def wols(Xm, t, w=None):
    if w is None: return np.linalg.lstsq(Xm, t, rcond=None)[0]
    sw = np.sqrt(w); return np.linalg.lstsq(Xm * sw[:, None], t * sw, rcond=None)[0]


def rand_proj(rng, live, d, k):
    G = np.zeros((d, k)); G[live] = rng.standard_normal((len(live), k)); Q = np.linalg.qr(G)[0]; return Q @ Q.T


# ------------------------------------------------------------------ DAS in a complement (E37 das_fit_margin with a projector)
def das_fit_proj(net, hb, hs, tgt, grp, k, live, Pc, seed, steps, lr=0.01):
    import jax, jax.numpy as jnp
    rng = np.random.default_rng(seed); ug = np.unique(grp); vg = rng.choice(ug, max(1, len(ug) // 5), replace=False)
    va = np.isin(grp, vg); tr = ~va
    ftail = net.tail('h4'); lo_r, hi_r = (jnp.asarray(np.nan_to_num(v, posinf=3e38, neginf=-3e38).astype(np.float32)) for v in net.rng_['h4'])
    cl = lambda b_, h_: jnp.clip(h_, jnp.minimum(lo_r, b_), jnp.maximum(hi_r, b_))
    lm = jnp.asarray(np.isin(np.arange(hb.shape[1]), live).astype(np.float32))[:, None]; PcJ = jnp.asarray(Pc.astype(np.float32))
    Hb, Hs, Tg = jnp.asarray(hb[tr], jnp.float32), jnp.asarray(hs[tr], jnp.float32), jnp.asarray(tgt[tr], jnp.float32)
    Vb, Vs, Vt = jnp.asarray(hb[va], jnp.float32), jnp.asarray(hs[va], jnp.float32), np.asarray(tgt[va])

    def Qof(W): return jnp.linalg.qr(PcJ @ (W * lm))[0]

    def mg(Q, B_, S_):
        lo = ftail(cl(B_, B_ + ((S_ - B_) @ Q) @ Q.T)); return lo[:, 2] - lo[:, 3]
    m0tr = np.asarray(mg(jnp.zeros((hb.shape[1], 1)), Hb, Hb)); den = float(((m0tr - tgt[tr]) ** 2).mean()) + 1e-9
    m0va = np.asarray(mg(jnp.zeros((hb.shape[1], 1)), Vb, Vb)); denv = float(((m0va - Vt) ** 2).sum()) + 1e-9

    def loss(W, B_, S_, T_): return ((mg(Qof(W), B_, S_) - T_) ** 2).mean() / den
    vg0 = jax.jit(jax.value_and_grad(loss)); vm0 = jax.jit(lambda W, B_, S_: mg(Qof(W), B_, S_))
    W = jnp.asarray(rng.normal(size=(hb.shape[1], k)).astype(np.float32)); m = jnp.zeros_like(W); v = jnp.zeros_like(W)
    best = (-np.inf, None, 0); L0 = None
    for t in range(1, steps + 1):
        L, g = vg0(W, Hb, Hs, Tg); L0 = float(L) if L0 is None else L0
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        W = W - lr * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        if t % 25 == 0 or t == steps:
            sc = 1 - float(((np.asarray(vm0(W, Vb, Vs)) - Vt) ** 2).sum()) / denv
            if sc > best[0]: best = (sc, np.asarray(Qof(W), np.float64), t)
    del vg0, vm0; jax.clear_caches()
    return best[1], dict(val_score=best[0], best_step=best[2], loss_first=L0, loss_last=float(L))


# ------------------------------------------------------------------ main
def main(a):
    from interp.jedi_core import sample
    from interp.x10_das_jedi import JNet, proj
    import jax
    sm = a.smoke; per = sm or 3000; tag = 'e37y' + ('_smoke' if sm else ''); OUT.mkdir(parents=True, exist_ok=True)
    steps = 60 if sm else 400; nrd = 20 if sm else 200; B = 200 if sm else 2000; nperm = 200 if sm else 2000
    nrand_das = 50 if sm else 1000; nrh = 20 if sm else 200
    res = dict(node='E37y', smoke=sm, per_class=per, seeds=dict(T=TSEED, F=FSEED), cs=list(CS), match_bins=[list(b) for b in MB], B=B,
               n_random_dirs=nrd, n_perm=nperm, n_random_das=nrand_das, n_random_holds=nrh, named_holds=NAMED, alphas=list(AL))
    save = lambda: (OUT / f'{tag}.json').write_text(json.dumps(jsonable(res), indent=1))
    xT, yT, _ = sample(per, TSEED); xF, yF, _ = sample(per, FSEED); rT = E.rows_of(per, TSEED); rF = E.rows_of(per, FSEED)
    xT = np.asarray(xT, np.float32); xF = np.asarray(xF, np.float32)
    log(f'samples: T {len(yT)} jets (seed {TSEED}), F {len(yF)} (seed {FSEED}); F rows also in T: {int(np.isin(rF, rT).sum())}')
    net = JNet(); net.off = np.zeros(5); net.shp = 'a'
    LT = net.net.run(xT)['logits']; res['acc_T_5class_stored'] = float((LT.argmax(1) == yT).mean())
    res['acc_F_5class_stored'] = float((net.net.run(xF)['logits'].argmax(1) == yF).mean())
    log(f'accuracy 5-class, stored inputs: T {res["acc_T_5class_stored"]:.4f}, F {res["acc_F_5class_stored"]:.4f}')
    # ---- F (E37 prep: D, cn, dil, cross) and T (D(+0.3) only)
    log('F edits (E37 prep)'); DF = E.prep(xF, yF, rF, FSEED, exclude_rows=rT, sm=sm)
    wz = (yT == 2) | (yT == 3); x = xT[wz]; y = yT[wz]; J = len(y)
    log('T prongs and D(+0.3)'); P4, nn, LAB, c0, PE, accD, dP = E.make_edits(x, TSEED, dcs=(0.3,), kinds=('D',))
    XT = {'id': E.model_xn(P4, nn, axis(P4, nn))[0]}; PD = PE[(0.3, 'D')]; XT['D'] = E.model_xn(PD, nn, axis(PD, nn))[0]; del PE
    mT = mass_of(P4, nn); res['N'] = dict(T_WZ=J, F_WZ=DF['J'], T_W=int((y == 2).sum()), T_Z=int((y == 3).sum()))
    res['mass_counts_T'] = {f'{lo}-{hi}': dict(W=int(((y == 2) & (mT >= lo) & (mT < hi)).sum()), Z=int(((y == 3) & (mT >= lo) & (mT < hi)).sum())) for lo, hi in FIGB}
    log('T mass counts ' + json.dumps(res['mass_counts_T']))
    RF = E.reps_all(net, DF); RT = {k: net.reps(v, keep=('h4', 'logits')) for k, v in XT.items()}
    h0 = RT['id']['h4']; m0 = margin(RT['id']['logits']); live = [int(u) for u in np.flatnonzero(h0.std(0) > 1e-9)]; lm = np.isin(np.arange(64), live)
    res['n_live_h4'] = len(live); res['acc_T_WZ_rebuilt_5class'] = float((RT['id']['logits'].argmax(1) == y).mean())
    tail, cWZ = make_tail(net)
    lo_t, z0 = tail(h0); lo_x, _ = net.exact('h4', h0)
    res['tail_check'] = dict(vs_JNet_exact=float(np.abs(lo_t - lo_x).max()), vs_run=float(np.abs(lo_t - RT['id']['logits']).max()),
                             full_hold_identity=float(np.abs(tail(h0 + 0.5, np.eye(16), z0)[0] - lo_t).max()))
    zl = np.maximum(z0, 0); livez = [int(u) for u in np.flatnonzero(zl.std(0) > 1e-9)]; res['live_z'] = livez
    res['z_units'] = {f'z{k}': dict(active_frac_W=float((z0[y == 2, k] > 0).mean()), active_frac_Z=float((z0[y == 3, k] > 0).mean()),
                                    mean_relu_W=float(zl[y == 2, k].mean()), mean_relu_Z=float(zl[y == 3, k].mean()), c_W_minus_Z=float(cWZ[k])) for k in range(16)}
    log(f'tail check {res["tail_check"]}; live z {livez}; live h4 {len(live)}')
    # ---- v: E37 DAS k = 1 refit on F
    PF = E.build_pairs(DF, RF, 'h4'); fit = [PF['self'], PF['cross']]
    hb = np.concatenate([p.hb for p in fit]); hs = np.concatenate([p.hs for p in fit]); tg = np.concatenate([p.mt for p in fit]); grp = np.concatenate([p.jet for p in fit])
    Q1, info = E.das_fit_margin(net, 'h4', hb, hs, tg, grp, 1, live, seed=TSEED, steps=steps); jax.clear_caches()
    del hb, hs, tg, grp, PF, fit
    v = Q1[:, 0].copy()
    QS = dict(np.load(OUT / 'e37_das_Q.npz')); MS = dict(np.load(OUT / 'e37_das_M.npz'))
    M2 = MS['M2_h4'].copy(); M2[~lm] = 0; M2 = np.linalg.qr(M2)[0]; M0s = MS['M0_h4'].copy(); M0s[~lm] = 0; M0s = np.linalg.qr(M0s)[0]
    v37 = QS['h4_k1'][:, 0].copy(); v37[~lm] = 0; v37 /= np.linalg.norm(v37)
    OF = np.full((DF['J'], len(LQ)), np.nan)
    for j in range(DF['J']):
        if DF['nn'][j] >= 3: OF[j] = lite(DF['P4'][j, :DF['nn'][j]], DF['LAB'][j, :DF['nn'][j]])
    okF = np.isfinite(OF).all(1) & (OF[:, 4] > 1)
    if np.corrcoef(RF['id']['h4'][okF] @ v, OF[okF, 0])[0, 1] < 0: v = -v
    u0 = h0 @ v; sdu = float(u0.std())
    res['direction'] = dict(das_train=info, cos_vs_E37_stored=float(abs(v @ v37)), max_cos_vs_M2=float(np.linalg.norm(M2.T @ v)),
                            corr_u_zh_F=float(np.corrcoef(RF['id']['h4'][okF] @ v, OF[okF, 0])[0, 1]), sd_u_T=sdu,
                            top_units=[int(i) for i in np.argsort(-np.abs(v))[:6]], top_weights=np.round(v[np.argsort(-np.abs(v))[:6]], 3).tolist())
    PT_ = E.build_pairs(DF, RF, 'h4')
    res['direction']['F_interchange_self'] = E.metrics(E.patch(net, 'h4', PT_['self'].hb, PT_['self'].hs, Q1), PT_['self']); del PT_
    log('direction ' + json.dumps(jsonable(res['direction'])))
    save()
    W = bweights(J, B, 11); WF = bweights(DF['J'], B, 12); rg = net.rng_['h4']

    # ---- Arm 1: steering per class and mass, matched contrast
    def steer(vv, sd, Pz=None, keepz=False):
        dms, zs, clip = [], [], 0.0
        for c in CS:
            ht = h0 + c * sd * vv[None, :]; hp = proj(rg, h0, ht)
            if abs(c) == 2: clip = max(clip, float((np.abs(hp - ht) > 1e-9).any(1).mean()))
            lo, z = tail(hp, Pz, z0 if Pz is not None else None); dms.append(margin(lo) - m0)
            if keepz: zs.append(z)
        dms = np.array(dms); cc = np.array(CS) * sd; sl = (cc[:, None] * dms).sum(0) / (cc ** 2).sum()
        return sl, dms, zs, clip
    t = time.time(); sl, dms, zs, clip = steer(v, sdu, keepz=True)
    A1 = dict(clip_frac_c2=clip, slope_units='W-Z margin change (log-odds) per unit u (u = h4 . v; sd_u on T given)')
    o, bd_free = cstat(sl, y, mT, W); A1['matched'] = o
    A1['perm_p'] = perm_p(sl, y, mT, nperm, 5); A1['fine_1GeV'] = fine_delta(sl, y, mT, W); A1['linear_model'] = lin_class(sl, y, mT, W)
    fig = {}
    for c_, cn in ((2, 'W'), (3, 'Z')):
        for lo, hi in FIGB + ((0, 1000),):
            s = np.flatnonzero((y == c_) & (mT >= lo) & (mT < hi))
            if len(s) >= 10: fig[f'{lo}-{hi}|{cn}'] = dict(n=int(len(s)), slope=float(sl[s].mean()), ci=qci(bmean(W, sl, s)),
                                                           dm_by_c={f'{c:+}': float(dms[i][s].mean()) for i, c in enumerate(CS)})
    A1['per_bin_5GeV'] = fig
    # per-c matched margin change (linearity)
    A1['per_c_matched'] = {f'{c:+}': {k: cstat(dms[i], y, mT)[0].get(k) for k in ('W_matched', 'Z_matched', 'delta')} for i, c in enumerate(CS)}
    # decomposition over z units
    zl0 = np.maximum(z0, 0); cc = np.array(CS) * sdu; dec = {}; tot = 0.0
    for kz in range(16):
        slk = sum(cc[i] * cWZ[kz] * (np.maximum(zs[i][:, kz], 0) - zl0[:, kz]) for i in range(len(CS))) / (cc ** 2).sum()
        ok_, _ = cstat(slk, y, mT, W); dec[f'z{kz}'] = dict(delta=ok_['delta'], ci=ok_.get('ci'), W_matched=ok_.get('W_matched'), Z_matched=ok_.get('Z_matched'))
        tot += ok_['delta'] or 0.0
    A1['z_decomposition'] = dict(per_unit=dec, sum_of_units=tot, total=o['delta'], quantizer_residual=(o['delta'] or 0) - tot)
    log(f'arm 1 steering [{time.time() - t:.0f}s]: matched Delta {o["delta"]} {o.get("ci")} (W {o["W_matched"]:.3f}, Z {o["Z_matched"]:.3f}); perm p {A1["perm_p"]}; '
        f'fine {A1["fine_1GeV"]}; linear {A1["linear_model"]}; bins ' + json.dumps(jsonable({k: (r.get("W"), r.get("Z"), r["nW"], r["nZ"]) for k, r in o["bins"].items()})))
    log('  z decomposition: ' + ', '.join(f'{k} {d["delta"]:+.3f}' for k, d in dec.items() if d['delta'] is not None and abs(d['delta']) > 0.01) + f'; residual {A1["z_decomposition"]["quantizer_residual"]:+.3f}')
    # random directions (per s.d. step)
    rr = np.random.default_rng(77); rd = []; rdW = []; rdZ = []
    for i in range(nrd):
        g = E.rand_live(rr, live, 64, 1)[:, 0]; sdg = float((h0 @ g).std()); slg, _, _, _ = steer(g, sdg); og, _ = cstat(slg * sdg, y, mT)
        rd.append(og['delta']); rdW.append(og['W_matched']); rdZ.append(og['Z_matched'])
    rd = np.array(rd, float); dv = o['delta'] * sdu
    A1['random_dirs'] = dict(n=nrd, units='log-odds per s.d. step of the coordinate', v_delta_per_sd=dv, v_W_per_sd=o['W_matched'] * sdu, v_Z_per_sd=o['Z_matched'] * sdu,
                             delta_q=np.quantile(rd, [.025, .5, .975]).tolist(), abs_delta_median=float(np.median(np.abs(rd))),
                             rank_p_abs=float((1 + (np.abs(rd) >= abs(dv)).sum()) / (1 + nrd)),
                             frac_opposite_sign_W_Z=float(np.mean(np.sign(rdW) != np.sign(rdZ))))
    log(f'  random directions: {A1["random_dirs"]}')
    res['arm1_steering'] = A1; save()

    # ---- Arm 2: z holds
    t = time.time(); A2 = dict(free_delta=o['delta'], free_ci=o.get('ci')); named = {}; ps = []
    rr = np.random.default_rng(78)
    nullR = {}
    for d in (1, 2, 3):
        vals = []
        for i in range(nrh):
            sl_h, _, _, _ = steer(v, sdu, Pz=rand_proj(rr, livez, 16, d)); vals.append(1 - delta_fast(sl_h, y, mT) / o['delta'])
        nullR[d] = np.array(vals)
    others = [k for k in range(16) if k not in (0, 7, 14)]; nullA = {}
    from itertools import combinations
    for d in (1, 2, 3):
        vals = []
        for comb in combinations(others, d):
            Pz = np.zeros((16, 16)); Pz[list(comb), list(comb)] = 1; sl_h, _, _, _ = steer(v, sdu, Pz=Pz); vals.append(1 - delta_fast(sl_h, y, mT) / o['delta'])
        nullA[d] = np.array(vals)
    for nm, units in NAMED.items():
        Pz = np.zeros((16, 16)); Pz[units, units] = 1; sl_h, _, _, _ = steer(v, sdu, Pz=Pz); oh, bd = cstat(sl_h, y, mT, W)
        bs = 1 - bd / bd_free; sh = 1 - oh['delta'] / o['delta']; d = len(units)
        r = dict(units=units, delta_held=oh['delta'], delta_held_ci=oh.get('ci'), W_matched=oh['W_matched'], Z_matched=oh['Z_matched'],
                 share_removed=float(sh), share_ci=qci(bs), p_boot_share_le_half=float(np.mean(bs <= 0.5)),
                 rank_p_vs_random_dirs=float((1 + (nullR[d] >= sh).sum()) / (1 + len(nullR[d]))),
                 rank_p_vs_other_units=float((1 + (nullA[d] >= sh).sum()) / (1 + len(nullA[d]))), bins={k: (b.get('W'), b.get('Z')) for k, b in oh['bins'].items()})
        named[nm] = r; ps.append(r['rank_p_vs_random_dirs'])
    hp = E.holm(ps)
    for (nm, r), p_ in zip(named.items(), hp): r['holm_p'] = float(p_)
    A2['named'] = named
    A2['null_random_dirs'] = {d: dict(n=len(v_), q=np.quantile(v_, [.025, .5, .95, .975]).tolist()) for d, v_ in nullR.items()}
    A2['null_other_units'] = {d: dict(n=len(v_), q=np.quantile(v_, [.025, .5, .95, .975]).tolist(), max=float(v_.max())) for d, v_ in nullA.items()}
    # every single unit (for the table)
    single = {}
    for k in range(16):
        Pz = np.zeros((16, 16)); Pz[k, k] = 1; sl_h, _, _, _ = steer(v, sdu, Pz=Pz); single[f'z{k}'] = float(1 - delta_fast(sl_h, y, mT) / o['delta'])
    A2['single_unit_share'] = single
    res['arm2_gate'] = A2
    log(f'arm 2 holds [{time.time() - t:.0f}s]: ' + '; '.join(f'{nm} share {r["share_removed"]:+.3f} {r["share_ci"]} p_rand {r["rank_p_vs_random_dirs"]:.4f} Holm {r["holm_p"]:.4f} p_units {r["rank_p_vs_other_units"]:.3f}'
                                                     for nm, r in named.items()))
    log('  single-unit shares ' + json.dumps(jsonable(single)) + '; nulls ' + json.dumps(jsonable({'dirs': A2['null_random_dirs'], 'units': A2['null_other_units']})))
    save()

    # ---- example jet for the e37x figure (Ez +0.15 visibly changes the prong split)
    t = time.time(); cand = np.flatnonzero((y == 2) & (mT >= 75) & (mT < 85) & (nn >= 15) & (nn <= 40) & (RT['id']['logits'].argmax(1) == 2))
    best = None
    for j in cand[:400]:
        n = int(nn[j]); P = P4[j, :n].astype(np.float64); lab = LAB[j, :n]; pt, yy, phi, _ = H50.kin(P); mm = np.zeros_like(pt)
        T_ = P.sum(0); ya, pa = float(rap(T_)), float(np.arctan2(T_[1], T_[0])); M0_ = [H50.pmass(pt, yy, phi, mm, lab == k) for k in (0, 1)]
        r, inf = H50.arm_c(pt, yy, phi, mm, lab, -0.15, M0_)
        if r is None: continue
        pt2, y2, p2 = r; V = H50.build(pt2, y2, p2, mm).sum(0); y2 = y2 + (ya - float(rap(V))); p2 = p2 + float(wrap(pa - np.arctan2(V[1], V[0])))
        Q = H50.build(pt2, y2, p2, mm)
        if H50.migration(lab, kt2_labels(Q), pt2) >= MIGMAX or not prong_offsets_ok(Q, lab): continue
        o0, o1 = lite(P, lab), lite(Q, lab); dz = o1[0] - o0[0]
        if dz < 0.12: continue
        score = abs(math.log(o1[4] / o0[4]))
        if best is None or score < best[0]: best = (score, j, Q, o0, o1)
    if best is not None:
        _, j, Q, o0, o1 = best; n = int(nn[j]); lab = LAB[j, :n]
        xe = E.model_xn(np.r_[P4[j:j + 1, :n], Q[None]].astype(np.float64), np.array([n, n]), axis(np.r_[P4[j:j + 1, :n], Q[None]].astype(np.float64), np.array([n, n])))[0]
        xe = np.concatenate([xe, np.zeros((2, 64 - xe.shape[1], 3), np.float32)], 1) if xe.shape[1] < 64 else xe
        Re = net.reps(xe, keep=('h4', 'logits')); ex = dict(index=int(j), N=n, prong=lab.tolist())
        for i_, (nm, P) in enumerate((('unedited', P4[j, :n].astype(np.float64)), ('Ez+0.15', Q))):
            pt = np.hypot(P[:, 0], P[:, 1]); T_ = P.sum(0); eT = np.arcsinh(T_[2] / np.hypot(T_[0], T_[1])); fT = np.arctan2(T_[1], T_[0])
            oo = o0 if i_ == 0 else o1
            ex[nm] = dict(pt=pt.tolist(), deta=(np.arcsinh(P[:, 2] / pt) - eT).tolist(), dphi=wrap(np.arctan2(P[:, 1], P[:, 0]) - fT).tolist(),
                          obs=dict(zip(LQ, oo.tolist())), prong_pt=[float(np.hypot(*P[lab == k].sum(0)[:2])) for k in (0, 1)],
                          margin=float(margin(Re['logits'][i_:i_ + 1])[0]), u=float(Re['h4'][i_] @ v), call=['q', 'g', 'W', 'Z', 't'][int(Re['logits'][i_].argmax())])
        res['example'] = ex; log(f'example jet {j} ({len(cand)} candidates) [{time.time() - t:.0f}s]: z_h {o0[0]:.3f} -> {o1[0]:.3f}, m {o0[4]:.1f} -> {o1[4]:.1f}, '
                                 f'margin {ex["unedited"]["margin"]:+.2f} -> {ex["Ez+0.15"]["margin"]:+.2f}')
    save()

    # ---- Arm 3: width at fixed prong mass
    t = time.time(); keys = [(k, al) for al in AL for k in ('Ws', 'Eshuf', 'Ewid', 'EmW', 'Erot')]
    PV = {k: P4.copy() for k in keys}; ACC = {k: np.zeros(J, bool) for k in keys}; OB = {k: np.full((J, len(LQ)), np.nan) for k in keys}
    OT = np.full((J, len(LQ)), np.nan); reasons = {al: {} for al in AL}; QPAR = {al: [] for al in AL}
    for j in range(J):
        n = int(nn[j])
        if n < 3: continue
        P = P4[j, :n].astype(np.float64); lab = LAB[j, :n]; OT[j] = lite(P, lab)
        outv, inf, _, _ = width_variants(P, lab, TSEED, j)
        for al in AL:
            rs = inf.get(al, 'none'); reasons[al][rs] = reasons[al].get(rs, 0) + 1
            if rs == 'ok': QPAR[al].append(inf[('q', al)])
        for k, Q in outv.items():
            PV[k][j, :n] = Q; ACC[k][j] = True; OB[k][j] = lite(Q, lab)
        if j % 1000 == 0: log(f'  width variants: jet {j}/{J} [{time.time() - t:.0f}s]')
    inv = {}
    for al in AL:
        s = np.flatnonzero(ACC[('Ws', al)])
        if len(s):
            d = OB[('Ws', al)][s] - OT[s]
            inv[f'{al:+}'] = dict(n=int(len(s)), max_abs_dzh=float(np.abs(d[:, 0]).max()), max_abs_ddR12=float(np.abs(d[:, 1]).max()),
                                  max_rel_dmprong=float((np.abs(d[:, 2:4]) / np.maximum(OT[s, 2:4], 1e-3)).max()), max_rel_dm=float((np.abs(d[:, 4]) / OT[s, 4]).max()),
                                  max_rel_dpT=float((np.abs(d[:, 5]) / OT[s, 5]).max()), mean_dWp=float(d[:, 6].mean()), mean_rel_dWp=float((d[:, 6] / OT[s, 6]).mean()),
                                  Eshuf_mean_dWp=float((OB[('Eshuf', al)][s, 6] - OT[s, 6]).mean()), Ewid_q_median=float(np.median(QPAR[al])),
                                  Ewid_mean_rel_dm=float(((OB[('Ewid', al)][s, 4] - OT[s, 4]) / OT[s, 4]).mean()),
                                  Ewid_mean_rel_dmprong_hi=float(((OB[('Ewid', al)][s, 2] - OT[s, 2]) / np.maximum(OT[s, 2], 1e-3)).mean()))
    log(f'  width variants built [{time.time() - t:.0f}s]; reasons {json.dumps({str(k): v_ for k, v_ in reasons.items()})}; invariants {json.dumps(jsonable(inv))}')
    XW = {}
    for k in keys:
        Pk = PV.pop(k); XW[k] = E.model_xn(Pk, nn, axis(Pk, nn))[0]; del Pk
    RW = {k: net.reps(xv, keep=('h4', 'logits')) for k, xv in XW.items()}; del XW
    DU = {k: RW[k]['h4'] @ v - u0 for k in keys}; DM = {k: margin(RW[k]['logits']) - m0 for k in keys}
    c0c = RT['id']['logits'].argmax(1)
    A3 = dict(reasons={str(k): v_ for k, v_ in reasons.items()}, invariants=inv, accepted={f'{k[0]}{k[1]:+}': float(ACC[k].mean()) for k in keys})
    tab = {}
    for k in keys:
        for cn, cm in (('W', y == 2), ('Z', y == 3), ('WZ', np.ones(J, bool))):
            sel = ACC[k] & cm; s = np.flatnonzero(sel)
            tab[f'{k[0]}{k[1]:+}|{cn}'] = dict(n=int(len(s)), du=X.mstat(W, DU[k], sel), dmargin=X.mstat(W, DM[k], sel),
                                               dWp=float((OB[k][s, 6] - OT[s, 6]).mean()) if len(s) else None,
                                               dlogm=float(np.log(OB[k][s, 4] / OT[s, 4]).mean()) if len(s) else None,
                                               dlead=float((OB[k][s, 7] - OT[s, 7]).mean()) if len(s) else None, dpTD=float((OB[k][s, 8] - OT[s, 8]).mean()) if len(s) else None,
                                               call_change=float((RW[k]['logits'][s].argmax(1) != c0c[s]).mean()) if len(s) else None)
    A3['table'] = tab
    # natural slope beta_w on F
    Fi = np.flatnonzero(okF); uF = RF['id']['h4'][Fi] @ v; O = OF[Fi]; lmF = np.log(O[:, 4])
    XF = np.stack([np.ones(len(Fi)), lmF, lmF ** 2, np.log(O[:, 5]), O[:, 0], O[:, 0] ** 2, O[:, 1], O[:, 2], O[:, 3], O[:, 6]], 1)
    bw = float(wols(XF, uF)[-1]); bwb = np.array([wols(XF, uF, w[Fi].astype(np.float64))[-1] for w in WF[:500]])
    (ru, rw), _, _ = X.knn_resid(np.c_[lmF, np.log(O[:, 5])], [uF, O[:, 6]]); bk = float((ru * rw).sum() / (rw ** 2).sum())
    A3['beta_w'] = dict(ols_full=bw, ci=qci(bwb), knn_mpt=bk, units='u per unit W_p (rad)', partial_corr_knn=float(np.corrcoef(ru, rw)[0, 1]),
                        sd_Wp_F=float(O[:, 6].std()))
    nb = len(bwb); Wb = W[:nb]

    # joint natural model: u ~ controls + W_p + lead (leading-particle pT fraction), same controls
    XF2 = np.c_[XF, O[:, 7]]; bj = wols(XF2, uF); bjb = np.array([wols(XF2, uF, w[Fi].astype(np.float64))[-2:] for w in WF[:500]])
    A3['beta_joint'] = dict(Wp=float(bj[-2]), Wp_ci=qci(bjb[:, 0]), lead=float(bj[-1]), lead_ci=qci(bjb[:, 1]), sd_lead_F=float(O[:, 7].std()))

    def Rcal(ed, ct, al):
        """per alpha: R_w = mean(du_ed - du_ct) / (beta_w mean(dWp_ed - dWp_ct)); joint ratio = observed / (beta_Wp dWp + beta_lead dlead)"""
        s = np.flatnonzero(ACC[(ed, al)] & ACC[(ct, al)])
        if len(s) < 10: return None
        ddu = DU[(ed, al)] - DU[(ct, al)]; dw = OB[(ed, al)][:, 6] - OB[(ct, al)][:, 6]; dl = OB[(ed, al)][:, 7] - OB[(ct, al)][:, 7]
        bu, bwm, blm = bmean(Wb, ddu, s), bmean(Wb, np.nan_to_num(dw), s), bmean(Wb, np.nan_to_num(dl), s)
        pred = bj[-2] * dw[s].mean() + bj[-1] * dl[s].mean(); predb = bjb[:, 0] * bwm + bjb[:, 1] * blm
        return dict(n=int(len(s)), du=float(ddu[s].mean()), du_ci=qci(bu), dWp=float(dw[s].mean()), dlead=float(dl[s].mean()),
                    R_w=float(ddu[s].mean() / (bw * dw[s].mean())), R_w_ci=qci(bu / (bwb * bwm)),
                    predicted_du_joint=float(pred), ratio_joint=float(ddu[s].mean() / pred), ratio_joint_ci=qci(bu / predb), p_boot_du_eq0=pboot(bu))
    A3['R'] = {f'{ed}_vs_{ct}|{al:+}': Rcal(ed, ct, al) for al in AL for ed, ct in (('Ws', 'Erot'), ('Ws', 'Eshuf'), ('Ewid', 'EmW'), ('Eshuf', 'Erot'))}
    # paired du contrasts and McNemar on call changes vs the shuffle control
    pc = {}
    for al in AL:
        for cn, cm in (('W', y == 2), ('Z', y == 3)):
            sel = ACC[('Ws', al)] & cm
            pc[f'{al:+}|{cn}'] = dict(du_Ws_minus_Eshuf=X.mstat(W, DU[('Ws', al)] - DU[('Eshuf', al)], sel),
                                      dmargin_Ws_minus_Eshuf=X.mstat(W, DM[('Ws', al)] - DM[('Eshuf', al)], sel),
                                      du_Ewid_minus_EmW=X.mstat(W, DU[('Ewid', al)] - DU[('EmW', al)], sel),
                                      mcnemar_calls_Ws_vs_Eshuf=mcn((RW[('Ws', al)]['logits'].argmax(1) != c0c)[sel], (RW[('Eshuf', al)]['logits'].argmax(1) != c0c)[sel]))
    A3['paired'] = pc
    res['arm3_width'] = A3
    log(f'arm 3 width [{time.time() - t:.0f}s]: beta_w {A3["beta_w"]}; joint {A3["beta_joint"]}; R ' + json.dumps(jsonable(A3['R'])))
    for k_, o_ in tab.items(): log(f'  {k_}: n {o_["n"]} du {o_["du"].get("mean")} {o_["du"].get("ci")} dmargin {o_["dmargin"].get("mean")} dWp {o_["dWp"]} dlogm {o_["dlogm"]}')
    log('  paired ' + json.dumps(jsonable(pc))[:3000])
    del RW; save()

    # ---- Arm 4: redundant Z carrier (DAS in the complement of v + M2)
    t = time.time(); Bv = np.linalg.qr(np.c_[v, M2])[0]; Pc = np.eye(64) - Bv @ Bv.T
    sF = np.flatnonzero(DF['acc'][0.3] & (DF['y'] == 3))
    hbF, hsF = RF['id']['h4'][sF], RF[(0.3, 'D')]['h4'][sF]; tgF = margin(RF[(0.3, 'D')]['logits'][sF])
    sT = np.flatnonzero(accD[0.3] & (y == 3))
    PTz = E.PairSet(h0[sT], RT['D']['h4'][sT], RT['D']['logits'][sT], RT['id']['logits'][sT], np.full(len(sT), 3), sT)
    mfree = margin(RT['D']['logits'][sT]) - m0[sT]
    A4 = dict(n_fit_pairs=int(len(sF)), n_test_pairs=int(len(sT)), free_dmargin_Z=float(mfree.mean()), copy_v=None)
    lp = E.patch(net, 'h4', PTz.hb, PTz.hs, v[:, None]); A4['copy_v'] = E.metrics(lp, PTz)
    lp = E.patch(net, 'h4', PTz.hb, PTz.hs, Bv); A4['copy_v_M2'] = E.metrics(lp, PTz)
    WT = W[:, :]; das = {}; pk = []
    rr = np.random.default_rng(79)

    def hold_share(Q):
        h = proj(rg, PTz.hs, PTz.hs + ((PTz.hb - PTz.hs) @ Q) @ Q.T); mh = margin(net.exact('h4', h)[0]) - m0[sT]
        return float(1 - mh.mean() / mfree.mean()), mh
    A4['hold_v_M2'] = hold_share(Bv)[0]
    for k in (1, 2, 3):
        Qk, inf = das_fit_proj(net, hbF, hsF, tgF, sF, k, live, Pc, seed=TSEED + k, steps=steps)
        mt = E.metrics(E.patch(net, 'h4', PTz.hb, PTz.hs, Qk), PTz); mt.update(E.metrics(E.patch(net, 'h4', PTz.hb, PTz.hs, Qk), PTz, WT))
        mt['fit'] = inf; mt['max_abs_cos_with_v_M2'] = float(np.abs(Bv.T @ Qk).max()); mt['cos_with_M0'] = np.linalg.svd(M0s.T @ Qk, compute_uv=False).tolist()
        rf, ri = [], []
        for _ in range(nrand_das):
            G = np.zeros((64, k)); G[live] = rr.standard_normal((len(live), k)); G = np.linalg.qr(Pc @ G)[0]
            mr = E.metrics(E.patch(net, 'h4', PTz.hb, PTz.hs, G), PTz); rf.append(mr['frac']); ri.append(mr['iia_wz'] if mr['iia_wz'] is not None else np.nan)
        rf, ri = np.array(rf, float), np.array(ri, float)
        mt['random'] = dict(frac_q=np.nanquantile(rf, [.5, .95]).tolist(), iia_q=np.nanquantile(ri, [.5, .95]).tolist())
        mt['rank_p_iia'] = float((1 + np.nansum(ri >= (mt['iia_wz'] or 0))) / (1 + nrand_das)); mt['rank_p_frac'] = float((1 + np.nansum(rf >= (mt['frac'] or 0))) / (1 + nrand_das))
        Qall = np.linalg.qr(np.c_[Bv, Qk])[0]; sh, _ = hold_share(Qall); hn = []
        for _ in range(nrh):
            G = np.zeros((64, 3 + k)); G[live] = rr.standard_normal((len(live), 3 + k)); hn.append(hold_share(np.linalg.qr(G)[0])[0])
        mt['hold_v_M2_Q'] = dict(share=sh, random_q=np.quantile(hn, [.5, .95]).tolist(), rank_p=float((1 + (np.array(hn) >= sh).sum()) / (1 + nrh)))
        das[f'k{k}'] = mt; pk.append(mt['rank_p_iia'])
        log(f'  arm 4 k={k}: frac {mt["frac"]} {mt.get("frac_ci")} IIA {mt["iia_wz"]} {mt.get("iia_wz_ci")} (n flips {mt["n_flip"]}); random {mt["random"]}; p {mt["rank_p_iia"]}; '
            f'hold v+M2+Q {mt["hold_v_M2_Q"]}; cos M0 {mt["cos_with_M0"]}')
        save()
    for (kk, mt), p_ in zip(das.items(), E.holm(pk)): mt['holm_p_iia'] = float(p_)
    A4['das'] = das; res['arm4_z_carrier'] = A4
    log(f'arm 4 [{time.time() - t:.0f}s]: copy v {A4["copy_v"]["frac"]}, copy v+M2 {A4["copy_v_M2"]["frac"]}, hold v+M2 {A4["hold_v_M2"]:.3f}')
    res['wall_s'] = time.time() - T0; save(); log(f'wrote {OUT / tag}.json; E37y DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', type=int, default=0)
    main(ap.parse_args())
