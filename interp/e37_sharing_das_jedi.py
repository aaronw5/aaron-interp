"""E37 (claims A4s, A4, A10 interchange weakness, A1x-type recurrence; stage 3): where does JEDI-linear's response to two-prong
momentum sharing at fixed jet four-momentum live?  Learned interchange subspace (DAS) for the D(dc) edit, interchange accuracy,
specificity against the mass route, physics read-out, steering, and recurrence across trained JEDI-linear networks.

Edit D(dc) = E35's edit_jet (identical to H45 edit_one): the jet (massless constituents) is split into two exclusive-kT prongs;
  all constituents are boosted to the jet rest frame, rotated so that |cos theta*| (angle between the prong axis and the flight
  direction; 0 = equal momentum sharing) changes from c0 to c1 = clip(c0 + dc, 0, 0.95), boosted back.  Jet four-momentum, prong
  masses and N are exactly fixed.  dc in {-0.3, -0.15, +0.15, +0.3}.  Controls: cn (rotation by the same angle about the prong
  axis: prong four-momenta, z and |cos theta*| fixed); dilation dil(f) (all angles x f, f = 0.95, 1.05: a real mass change).
  Inputs rebuilt about each jet's own E-scheme axis, pT-sorted (H45 model_x), for edited and unedited jets alike.
Samples: fit F = W+Z jets of sample(2000, 3701); test T = W+Z jets of sample(2000, 3702) minus jets also in F.
Arm 1 (--arm das): JNet (x10) exact fixed-point tail from S / h4 / h5 with x10's straight-through training tail and range clip.
  Pairs: self (base = unedited jet, source = its own D(dc) version, target = network on D(dc)); cross (base b, source = another
  jet s, target = network on b edited to |cos theta*| = c0(s), i.e. dc = c0(s) - c0(b)).  DAS: orthonormal k-dim Q restricted
  to the live units; h' = h_b + Q Q^T (h_s - h_b), clipped; loss = MSE of the patched W-Z logit margin against the target's
  margin; Adam, model selection on 20 % of the fit jets.  k in {1, 2, 3, 4, 6}, stages S (64), h4 (64), h5 (32).
  Metrics on T: IIA_WZ = fraction of pairs with sign(margin_patched) == sign(margin_target) among pairs where
  sign(margin_target) != sign(margin_base); frac = sum_cells s_c mean(m_patch - m_base) / sum_cells s_c mean(m_tgt - m_base)
  (cells = (dc, class), s_c = sign of the target mean); slope = regression through the origin of (m_patch - m_base) on
  (m_tgt - m_base).  95 % CI: 1000 jet bootstrap resamples.  Nulls: 1000 random k-dim subspaces of the live span; DAS with
  deranged targets (shuffled source labels); DAS on 3 untrained redraws (--arm untr).  H0: DAS IIA <= random (rank p), Holm 15.
  Specificity: the same Q on cn pairs and on dilation pairs (frac transferred); principal angles between Q and refitted X10-style
  M0 / M2 DAS subspaces (k = 2, cross-entropy on calls after do(M0) / do(M2) input edits) and the dilation direction, vs random.
Arm 2: scalar coordinate u = h . v, v = Q w / |Q w| (w = least squares of the margin change on the Q-coordinate change, fit
  pairs).  Delta u under D regressed on Delta observable (z_min, D2 beta 1, tau21 WTA-approx, prong masses, pT^D, leading
  fraction); calibrated against the natural partial slope of u on the observable across unedited jets (controls log m, log pT):
  ratio = mean Delta u / mean (natural slope x Delta observable).  Steering at h4: h4 + delta v, delta = c x mean |Delta u_D(+0.3)|,
  c in {+-0.5, +-1, +-2}; Delta margin per unit u vs D's Delta margin per unit u; 200 random live directions; dilation direction.
Arm 3 (--arm recur): D(+-0.3) vs cn on the 10 trained E33 networks (official n32/n64/n128, perminv, 16-feature, HGQ r150 s0-2,
  float s0-1) and 10 untrained redraws (Jedi(untrained=True, seed 0-9)), test sample T (W+Z): per-network W and Z mean of
  Delta margin(D) - Delta margin(cn) with CI, sign-flip p, McNemar on per-jet W/Z call changes; pattern = W < 0 and Z > 0 (CIs);
  Fisher exact (= permutation over networks) trained vs untrained.  n128: edit applied to the 128-slot jets; 16-feature: the
  16 columns recomputed from the edited (pT, eta, phi) (E33 to16; rotated-frame columns 9, 12, 15 kept at unedited values).
Run: ./job.sh e37-das 1 2 'python -m interp.e37_sharing_das_jedi --arm das'   (also --arm untr, --arm recur; --smoke K)
Writes $AI_RESULTS/jedi/e37_<arm>[_smoke].json, e37_Q.npz.
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
from interp.h45_prong_sharing import p4, axis, sort_pt, wrap, rap, prong_offsets_ok, CMAX  # noqa: E402
from interp.e35_prong_sharing_route_jedi import kt2_labels, edit_jet, jsonable, DCS  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0 = time.time()
STAGES = ('S', 'h4', 'h5'); KS = (1, 2, 3, 4, 6); WID = dict(S=64, h4=64, h5=32)
FSEED, TSEED = 3701, 3702


def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def qci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))] if len(v) else [None, None]


def binom_p(k, n):
    if n == 0: return 1.0
    k = min(k, n - k)
    lp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    m = max(lp); return float(min(1.0, 2 * math.exp(m) * sum(math.exp(v - m) for v in lp)))


def mcnemar(a, b):
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum()); return dict(D_only=n10, cn_only=n01, p=binom_p(n10, n10 + n01))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); out = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out.tolist()


def margin(L): return L[:, 2] - L[:, 3]


def fm(v, f='+.3f'):
    return 'None' if v is None else format(v, f)


def rows_of(per, seed):
    from jetdistill.config import RESULTS
    y = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y']); rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per, replace=False) for c in range(5)]))


def model_xn(P4, nn, ax, n=64):
    """(J, n, 3) (pT, deta, dphi) about ax, pT-sorted; also the sort permutation"""
    P, o = sort_pt(P4.astype(np.float64), nn); real = np.arange(P.shape[1])[None, :] < nn[:, None]
    pt = np.hypot(P[..., 0], P[..., 1]); ptc = np.where(real, pt, 1.0)
    eta = np.arcsinh(np.where(real, P[..., 2], 0) / ptc) - ax[0][:, None]; phi = wrap(np.arctan2(P[..., 1], P[..., 0]) - ax[1][:, None])
    return (np.stack([pt, eta, phi], -1) * real[..., None])[:, :n].astype(np.float32), o[:, :n]


def to_p4(x):
    x = np.asarray(x, np.float64); real = x[..., 0] > 0
    return p4(x[..., 0], x[..., 1], x[..., 2], 0.0) * real[..., None], real.sum(1).astype(np.int64)


# ------------------------------------------------------------------ edits
def make_edits(x, seed, dcs=DCS, kinds=('D', 'cn')):
    """x (J, L, 3) stored -> P4, nn, LAB, c0, PE {(dc, kind): P4}, acc {dc: bool}"""
    P4, nn = to_p4(x); J, L = P4.shape[:2]; LAB = -np.ones((J, L), np.int64); t = time.time()
    for j in np.flatnonzero(nn >= 3): LAB[j, :nn[j]] = kt2_labels(P4[j, :nn[j]])
    log(f'  kT-2 prongs for {J} jets ({L} slots) in {time.time() - t:.0f}s')
    c0 = np.full(J, np.nan); acc = {dc: np.zeros(J, bool) for dc in dcs}; PE = {(dc, k): P4.copy() for dc in dcs for k in kinds}
    dP = 0.0; t = time.time()
    for j in np.flatnonzero(nn >= 3):
        n = nn[j]; P = P4[j, :n]; lab = LAB[j, :n]
        for i, dc in enumerate(dcs):
            r, inf = edit_jet(P, lab, dc, np.random.default_rng([seed, int(j), i]))
            if r is None: continue
            c0[j] = inf['c0']; acc[dc][j] = prong_offsets_ok(r['D'], lab)
            dP = max(dP, float(np.abs(r['D'].sum(0) - P.sum(0)).max() / max(P.sum(0)[3], 1e-9)))
            for k in kinds: PE[(dc, k)][j, :n] = r[k]
    log(f'  edits in {time.time() - t:.0f}s; max relative jet four-momentum change {dP:.1e}; accepted ' + ', '.join(f'{dc:+.2f}: {acc[dc].mean():.3f}' for dc in dcs))
    return P4, nn, LAB, c0, PE, acc, dP


def cross_edits(P4, nn, LAB, c0, seed, nsrc=1):
    """for each base b, nsrc random sources s != b: b edited to |cos theta*| = c0(s); returns base, src, P4 edited, ok"""
    rng = np.random.default_rng(seed); J = len(P4); B, S_, PEs, OK = [], [], [], []
    good = np.flatnonzero(np.isfinite(c0))
    for r_ in range(nsrc):
        for b in good:
            s = int(rng.choice(good))
            while s == b: s = int(rng.choice(good))
            n = nn[b]; P = P4[b, :n]; lab = LAB[b, :n]; Pe = P4[b].copy(); ok = False
            if c0[s] <= CMAX:
                r, inf = edit_jet(P, lab, float(c0[s] - c0[b]), np.random.default_rng([seed, int(b), r_]))
                if r is not None:
                    ok = prong_offsets_ok(r['D'], lab); Pe[:n] = r['D']
            B.append(b); S_.append(s); PEs.append(Pe); OK.append(ok)
    return np.array(B), np.array(S_), np.stack(PEs), np.array(OK)


def jet_obs(P, lab):
    """physics observables of one jet (P (n, 4) massless, lab 0/1 kT prongs): z_min, D2, tau21, prong masses, pT^D, lead frac, m, pT"""
    pt = np.hypot(P[:, 0], P[:, 1]); y = rap(P); ph = np.arctan2(P[:, 1], P[:, 0]); n = len(P)
    S = [P[lab == k].sum(0) for k in (0, 1)]; spt = np.array([np.hypot(v[0], v[1]) for v in S])
    msub = np.array([math.sqrt(max(v[3] ** 2 - (v[:3] ** 2).sum(), 0.0)) for v in S])
    D = np.hypot(y[:, None] - y[None, :], wrap(ph[:, None] - ph[None, :])); z = pt / pt.sum()
    e2 = 0.5 * z @ D @ z; sz = np.sqrt(z); A = sz[:, None] * D * sz[None, :]; e3 = np.trace(A @ A @ A) / 6.0
    i1 = int(np.argmax(pt)); t1 = (pt * D[:, i1]).sum()
    ax2 = [int(np.flatnonzero(lab == k)[np.argmax(pt[lab == k])]) for k in (0, 1)]; t2 = (pt * np.minimum(D[:, ax2[0]], D[:, ax2[1]])).sum()
    T = P.sum(0); m = math.sqrt(max(T[3] ** 2 - (T[:3] ** 2).sum(), 0.0))
    return np.array([spt.min() / spt.sum(), e3 / max(e2 ** 3, 1e-30), t2 / max(t1, 1e-12), msub.max(), msub.min(),
                     math.sqrt((pt ** 2).sum()) / pt.sum(), pt.max() / pt.sum(), m, math.hypot(T[0], T[1])])


OBS = ['z_min', 'D2', 'tau21', 'm_prong_hi', 'm_prong_lo', 'pTD', 'lead_frac', 'm', 'pT']


def obs_all(P4, nn, LAB, sel):
    out = np.full((len(P4), len(OBS)), np.nan)
    for j in sel:
        if nn[j] >= 3: out[j] = jet_obs(P4[j, :nn[j]], LAB[j, :nn[j]])
    return out


# ------------------------------------------------------------------ data for arms 1 / 2
def load_split(sm):
    from interp.jedi_core import sample
    per = sm or 2000
    xF, yF, _ = sample(per, FSEED); xT, yT, _ = sample(per, TSEED)
    rF, rT = rows_of(per, FSEED), rows_of(per, TSEED)
    other = {s: rows_of(2000, s) for s in (3501, 3502, 3601)}
    info = dict(F_all=len(yF), T_all=len(yT), T_in_F=int(np.isin(rT, rF).sum()),
                overlap_with_E35_E35v={str(s): dict(F=int(np.isin(rF, r).sum()), T=int(np.isin(rT, r).sum())) for s, r in other.items()})
    return (np.asarray(xF, np.float32), yF, rF), (np.asarray(xT, np.float32), yT, rT), info


def prep(x, y, rows, seed, exclude_rows=None, sm=0):
    wz = (y == 2) | (y == 3)
    if exclude_rows is not None: wz &= ~np.isin(rows, exclude_rows)
    x = x[wz]; y = y[wz]; P4, nn, LAB, c0, PE, acc, dP = make_edits(x, seed)
    ax0 = axis(P4, nn); X = {'id': model_xn(P4, nn, ax0)[0]}
    for k, Pe in PE.items(): X[k] = model_xn(Pe, nn, axis(Pe, nn))[0]
    for f in (0.95, 1.05):
        xd = X['id'].copy(); xd[..., 1:] *= np.float32(f); X[('dil', f)] = xd
    B, Sx, Pc, okc = cross_edits(P4, nn, LAB, c0, seed + 11, nsrc=2 if seed == FSEED else 1)
    nb = nn[B]; X['cross'] = model_xn(Pc, nb, axis(Pc, nb))[0]
    return dict(x=x, y=y, P4=P4, nn=nn, LAB=LAB, c0=c0, PE=PE, acc=acc, dP=dP, X=X, cross=(B, Sx, okc), J=len(y))


# ------------------------------------------------------------------ DAS
def das_fit_margin(net, st, hb, hs, tgt, grp, k, live, seed, steps, lr=0.01):
    """orthonormal k-dim Q in the live span; MSE of the patched margin vs target; model selection on 20 % of the groups"""
    import jax, jax.numpy as jnp
    rng = np.random.default_rng(seed); ug = np.unique(grp); vg = rng.choice(ug, max(1, len(ug) // 5), replace=False)
    va = np.isin(grp, vg); tr = ~va
    ftail = net.tail(st); lo_r, hi_r = (jnp.asarray(np.nan_to_num(v, posinf=3e38, neginf=-3e38).astype(np.float32)) for v in net.rng_[st])
    cl = lambda b_, h_: jnp.clip(h_, jnp.minimum(lo_r, b_), jnp.maximum(hi_r, b_))
    lm = jnp.asarray(np.isin(np.arange(hb.shape[1]), live).astype(np.float32))[:, None]
    Hb, Hs, Tg = jnp.asarray(hb[tr], jnp.float32), jnp.asarray(hs[tr], jnp.float32), jnp.asarray(tgt[tr], jnp.float32)
    Vb, Vs, Vt = jnp.asarray(hb[va], jnp.float32), jnp.asarray(hs[va], jnp.float32), np.asarray(tgt[va])

    def Qof(W): return jnp.linalg.qr(W * lm)[0]

    def mg(Q, B_, S_):
        lo = ftail(cl(B_, B_ + ((S_ - B_) @ Q) @ Q.T)); return lo[:, 2] - lo[:, 3]
    m0tr = np.asarray(mg(jnp.zeros((hb.shape[1], 1)), Hb, Hb)); den = float(((m0tr - tgt[tr]) ** 2).mean()) + 1e-9
    m0va = np.asarray(mg(jnp.zeros((hb.shape[1], 1)), Vb, Vb)); denv = float(((m0va - Vt) ** 2).sum()) + 1e-9

    # debug 1 (OOM): data passed as arguments (not jit constants); caches cleared after each fit
    def loss(W, B_, S_, T_): return ((mg(Qof(W), B_, S_) - T_) ** 2).mean() / den
    vg0 = jax.jit(jax.value_and_grad(loss)); vm0 = jax.jit(lambda W, B_, S_: mg(Qof(W), B_, S_))
    vgf = lambda W: vg0(W, Hb, Hs, Tg); vm = lambda W: vm0(W, Vb, Vs)
    W = jnp.asarray(rng.normal(size=(hb.shape[1], k)).astype(np.float32)); m = jnp.zeros_like(W); v = jnp.zeros_like(W)
    best = (-np.inf, None, 0); L0 = None
    for t in range(1, steps + 1):
        L, g = vgf(W); L0 = float(L) if L0 is None else L0
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        W = W - lr * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        if t % 25 == 0 or t == steps:
            sc = 1 - float(((np.asarray(vm(W)) - Vt) ** 2).sum()) / denv
            if sc > best[0]: best = (sc, np.asarray(Qof(W), np.float64), t)
    del vg0, vm0; jax.clear_caches()
    return best[1], dict(val_score=best[0], best_step=best[2], loss_first=L0, loss_last=float(L))


def patch(net, st, hb, hs, Q):
    from interp.x10_das_jedi import proj
    h = proj(net.rng_[st], hb, hb + ((hs - hb) @ Q) @ Q.T); return net.exact(st, h)[0]


class PairSet:
    """pairs (base row, source state, target logits, cell id, jet id) at one stage"""

    def __init__(self, hb, hs, lt, lb, cell, jet):
        self.hb, self.hs, self.lt, self.lb, self.cell, self.jet = hb, hs, lt, lb, cell, jet
        self.mt, self.mb = margin(lt), margin(lb)


def metrics(lp, P, W=None):
    """lp patched logits; P PairSet; W per-jet bootstrap weights (B, J) or None -> dict (point) or arrays (boot)"""
    mp = margin(lp); dt = P.mt - P.mb; dp = mp - P.mb
    flip = np.sign(P.mt) != np.sign(P.mb); ok = np.sign(mp) == np.sign(P.mt)
    cb, ct, cp = P.lb.argmax(1), P.lt.argmax(1), lp.argmax(1); f5 = ct != cb
    if W is None:
        cells = np.unique(P.cell); sg = {c: np.sign(dt[P.cell == c].mean()) for c in cells}
        num = sum(sg[c] * dp[P.cell == c].mean() for c in cells); den = sum(sg[c] * dt[P.cell == c].mean() for c in cells)
        return dict(iia_wz=float(ok[flip].mean()) if flip.sum() >= 10 else None, n_flip=int(flip.sum()),
                    iia5=float((cp == ct)[f5].mean()) if f5.sum() >= 10 else None, n_flip5=int(f5.sum()),
                    frac=float(num / den) if den != 0 else None, slope=float((dp * dt).sum() / max((dt * dt).sum(), 1e-12)),
                    target_change=float(den), mean_abs_target=float(np.abs(dt).mean()), mean_abs_patch=float(np.abs(dp).mean()))
    w = W[:, P.jet]; cells = np.unique(P.cell); num = 0; den = 0
    for c in cells:
        s = P.cell == c; sg = np.sign(dt[s].mean()); ws = w[:, s]; n_ = np.maximum(ws.sum(1), 1)
        num = num + sg * (ws @ dp[s]) / n_; den = den + sg * (ws @ dt[s]) / n_
    iia = (w[:, flip] @ ok[flip].astype(float)) / np.maximum(w[:, flip].sum(1), 1)
    return dict(iia_wz_ci=qci(iia), frac_ci=qci(num / den), slope_ci=qci((w @ (dp * dt)) / np.maximum(w @ (dt * dt), 1e-12)))


def rand_live(rng, live, d, k):
    G = np.zeros((d, k)); G[live] = rng.standard_normal((len(live), k)); return np.linalg.qr(G)[0]


def build_pairs(D, R, st, cross_ok=True):
    """PairSet dict for one sample and stage: self (all dc), self per dc, cross, cn, dil"""
    out = {}; J = D['J']; y = D['y']; base = R['id']
    hb, hs, lt, lb, cell, jet = [], [], [], [], [], []
    for i, dc in enumerate(DCS):
        s = np.flatnonzero(D['acc'][dc]); hb.append(base[st][s]); hs.append(R[(dc, 'D')][st][s]); lt.append(R[(dc, 'D')]['logits'][s])
        lb.append(base['logits'][s]); cell.append(10 * i + y[s]); jet.append(s)
    out['self'] = PairSet(*(np.concatenate(v) for v in (hb, hs, lt, lb, cell, jet)))
    for i, dc in enumerate(DCS):
        s = np.flatnonzero(D['acc'][dc]); out[f'self{dc:+.2f}'] = PairSet(base[st][s], R[(dc, 'D')][st][s], R[(dc, 'D')]['logits'][s], base['logits'][s], y[s], s)
    hb, hs, lt, lb, cell, jet = [], [], [], [], [], []
    for i, dc in enumerate(DCS):
        s = np.flatnonzero(D['acc'][dc]); hb.append(base[st][s]); hs.append(R[(dc, 'cn')][st][s]); lt.append(R[(dc, 'cn')]['logits'][s])
        lb.append(base['logits'][s]); cell.append(10 * i + y[s]); jet.append(s)
    out['cn'] = PairSet(*(np.concatenate(v) for v in (hb, hs, lt, lb, cell, jet)))
    hb, hs, lt, lb, cell, jet = [], [], [], [], [], []
    for i, f in enumerate((0.95, 1.05)):
        s = np.arange(J); hb.append(base[st]); hs.append(R[('dil', f)][st]); lt.append(R[('dil', f)]['logits']); lb.append(base['logits']); cell.append(10 * i + y); jet.append(s)
    out['dil'] = PairSet(*(np.concatenate(v) for v in (hb, hs, lt, lb, cell, jet)))
    B, Sx, ok = D['cross']; s = np.flatnonzero(ok)
    out['cross'] = PairSet(base[st][B[s]], base[st][Sx[s]], R['cross']['logits'][s], base['logits'][B[s]],
                           np.sign(R['cross']['logits'][s][:, 2] - R['cross']['logits'][s][:, 3] - margin(base['logits'][B[s]])).astype(int) + 10 * y[B[s]], B[s])
    return out


def reps_all(net, D):
    keep = ('S', 'h4', 'h5', 'logits'); R = {}
    for k, v in D['X'].items(): R[k] = net.reps(v, keep=keep)
    if net.off is not None and np.any(net.off):
        for k in R: R[k]['logits'] = R[k]['logits'] - net.off
    return R


def das_arm(net, DF, DT, RF, RT, steps, nrand, tag, sm, shuffled=True, rand_eval=True):
    """DAS + nulls + specificity for one network; returns results and Q dict"""
    res = {}; Qs = {}
    bt = np.random.default_rng(7); W = np.stack([np.bincount(bt.integers(0, DT['J'], DT['J']), minlength=DT['J']) for _ in range(200 if sm else 1000)]).astype(np.float32)
    for st in STAGES:
        d = WID[st]; sd = RF['id'][st].std(0); live = [int(u) for u in np.flatnonzero(sd > 1e-9)]
        PF = build_pairs(DF, RF, st); PT = build_pairs(DT, RT, st)
        fit = [PF['self'], PF['cross']]; hb = np.concatenate([p.hb for p in fit]); hs = np.concatenate([p.hs for p in fit])
        tg = np.concatenate([p.mt for p in fit]); grp = np.concatenate([p.jet for p in fit])
        rs = dict(d=d, n_live=len(live), n_fit_pairs=dict(self=len(PF['self'].mt), cross=len(PF['cross'].mt)),
                  n_test_pairs={k: len(v.mt) for k, v in PT.items()}, k={})
        # no-patch and full-replacement references on the test pairs
        rs['full_replacement_check'] = {nm: float(np.abs(margin(net.exact(st, PT[nm].hs)[0]) - PT[nm].mt).max()) for nm in ('self', 'cn', 'dil')}
        evalsets = ('self', 'cross', 'cn', 'dil') + tuple(f'self{dc:+.2f}' for dc in DCS)
        for k in KS:
            if k > len(live): continue
            t = time.time(); Q, info = das_fit_margin(net, st, hb, hs, tg, grp, k, live, seed=100 + k, steps=steps)
            r = dict(train=info, test={}); Qs[(st, k)] = Q
            for nm in evalsets:
                lp = patch(net, st, PT[nm].hb, PT[nm].hs, Q); r['test'][nm] = metrics(lp, PT[nm])
                if nm in ('self', 'cross', 'dil', 'cn'): r['test'][nm].update(metrics(lp, PT[nm], W))
            if shuffled:
                pr = np.random.default_rng(200 + k); pi = pr.permutation(len(tg))
                while (pi == np.arange(len(tg))).any(): pi = pr.permutation(len(tg))
                Qp, ip = das_fit_margin(net, st, hb, hs, tg[pi], grp, k, live, seed=300 + k, steps=steps)
                r['shuffled'] = dict(train=ip, **{nm: metrics(patch(net, st, PT[nm].hb, PT[nm].hs, Qp), PT[nm]) for nm in ('self', 'cross', 'dil')})
            if rand_eval:
                rr = np.random.default_rng(400 + k); NUL = {nm: dict(iia=[], frac=[]) for nm in ('self', 'cross', 'dil')}; ang = []
                cat = [PT[nm] for nm in ('self', 'cross', 'dil')]; HB = np.concatenate([p.hb for p in cat]); HS = np.concatenate([p.hs for p in cat])
                cuts = np.cumsum([0] + [len(p.mt) for p in cat])
                for _ in range(nrand):
                    Qr = rand_live(rr, live, d, k); lp = patch(net, st, HB, HS, Qr)
                    for i, nm in enumerate(('self', 'cross', 'dil')):
                        mm = metrics(lp[cuts[i]:cuts[i + 1]], PT[nm]); NUL[nm]['iia'].append(np.nan if mm['iia_wz'] is None else mm['iia_wz']); NUL[nm]['frac'].append(mm['frac'])
                r['random'] = {}
                for nm in NUL:
                    a = np.asarray(NUL[nm]['iia'], float); f = np.asarray(NUL[nm]['frac'], float); o = r['test'][nm]
                    r['random'][nm] = dict(n=int(np.isfinite(a).sum()), iia_mean=float(np.nanmean(a)), iia_q=np.nanquantile(a, [.025, .5, .95, .975]).tolist(),
                                           frac_mean=float(np.nanmean(f)), frac_q=np.nanquantile(f, [.025, .5, .95, .975]).tolist(),
                                           p_iia=float((1 + (a[np.isfinite(a)] >= o['iia_wz']).sum()) / (1 + np.isfinite(a).sum())) if o['iia_wz'] is not None else None,
                                           p_frac=float((1 + (np.abs(f) >= abs(o['frac'])).sum()) / (1 + len(f))) if o['frac'] is not None else None)
            rs['k'][str(k)] = r
            o = r['test']
            log(f'{tag} {st} k={k}: val {fm(info["val_score"])} (step {info["best_step"]}); self IIA {fm(o["self"]["iia_wz"])} {o["self"].get("iia_wz_ci")} '
                f'(n {o["self"]["n_flip"]}) frac {fm(o["self"]["frac"])} {o["self"].get("frac_ci")}; cross IIA {fm(o["cross"]["iia_wz"])} frac {fm(o["cross"]["frac"])}; '
                f'cn frac {fm(o["cn"]["frac"])}; dil frac {fm(o["dil"]["frac"])}; '
                + (f'shuffled self IIA {fm(r["shuffled"]["self"]["iia_wz"])} frac {fm(r["shuffled"]["self"]["frac"])}; ' if shuffled else '')
                + (f'random self IIA mean {fm(r["random"]["self"]["iia_mean"])} q95 {fm(r["random"]["self"]["iia_q"][2])} p {fm(r["random"]["self"]["p_iia"], ".4f")}; '
                   f'random frac q95 {fm(r["random"]["self"]["frac_q"][2])}; random dil frac mean {fm(r["random"]["dil"]["frac_mean"])}' if rand_eval else '')
                + f' [{time.time() - t:.0f}s]')
        res[st] = rs
    return res, Qs


# ------------------------------------------------------------------ M0 / M2 subspaces (X10-style refit) and principal angles
def mass_subspaces(net, DF, RF, steps, sm):
    from interp.x10_das_jedi import das_fit, do_edit, moments, eval_sub, scores
    x = DF['X']['id']; J = len(x); mm = moments(x); rng = np.random.default_rng(5)
    n = 1200 if sm else 6000; b = rng.integers(0, J, n); s = rng.integers(0, J, n); s[s == b] = (s[s == b] + 1) % J
    basec = net.calls(x)[b]; out = {}; Ms = {}
    for V in ('M0', 'M2'):
        lab = net.calls(do_edit(x[b], V, (mm[0][b], mm[1][b]), (mm[0][s], mm[1][s])))
        for st in STAGES:
            hb, hs = RF['id'][st][b], RF['id'][st][s]
            Q, info = das_fit(net.tail(st), hb, hs, lab, basec, 2, seed=50, steps=steps, rg=net.rng_[st]); Ms[(V, st)] = Q
            import jax; jax.clear_caches()
            c = eval_sub(net, st, hb[-n // 5:], hs[-n // 5:], Q, None, None); sc = scores(c, lab[-n // 5:], basec[-n // 5:])
            out[f'{V}|{st}'] = dict(train=info, heldout_eff_iia=sc['eff'], n_eff=sc['n_eff'], cf_change_frac=float((lab != basec).mean()))
            log(f'  refit {V} at {st}: held-out effective IIA {sc["eff"]} (n {sc["n_eff"]}); val {info["val_balanced_agreement"]:.3f}')
    # dilation direction (mean raw displacement under all angles x 1.05)
    for st in STAGES:
        d_ = (RF[('dil', 1.05)][st] - RF['id'][st]).mean(0); Ms[('dil', st)] = (d_ / np.linalg.norm(d_))[:, None]
    return out, Ms


def angles(Qs, Ms, RF, nrand):
    out = {}
    for (st, k), Q in Qs.items():
        sd = RF['id'][st].std(0); live = [int(u) for u in np.flatnonzero(sd > 1e-9)]; rr = np.random.default_rng(600 + k)
        rq = [rand_live(rr, live, WID[st], k) for _ in range(nrand)]
        for V in ('M0', 'M2', 'dil'):
            M = Ms[(V, st)].copy(); M[sd <= 1e-9] = 0; M = np.linalg.qr(M)[0]; c = np.linalg.svd(Q.T @ M, compute_uv=False); cr = np.array([np.linalg.svd(R_.T @ M, compute_uv=False).max() for R_ in rq])
            out[f'{st}|{k}|{V}'] = dict(max_cos=float(c.max()), angles_deg=np.degrees(np.arccos(np.clip(c, -1, 1))).tolist(),
                                       random_max_cos_q=np.quantile(cr, [.05, .5, .95]).tolist(), p_closer_than_random=float((1 + (cr >= c.max()).sum()) / (1 + len(cr))),
                                       p_farther_than_random=float((1 + (cr <= c.max()).sum()) / (1 + len(cr))))
    return out


# ------------------------------------------------------------------ arm 2: read-outs and steering
def readout(net, DT, RT, DF, RF, Q, st, sm):
    """scalar coordinate u along v = Q w; regress Delta u on Delta observables; natural-slope calibration"""
    PF = build_pairs(DF, RF, st)['self']; dQ = (PF.hs - PF.hb) @ Q; w = np.linalg.lstsq(dQ, PF.mt - PF.mb, rcond=None)[0]
    v = Q @ w; v = v / np.linalg.norm(v)
    J = DT['J']; y = DT['y']; u0 = RT['id'][st] @ v
    sel = np.arange(J); O0 = obs_all(DT['P4'], DT['nn'], DT['LAB'], sel)
    rows = []
    for dc in DCS:
        s = np.flatnonzero(DT['acc'][dc]); Oe = obs_all(DT['PE'][(dc, 'D')], DT['nn'], DT['LAB'], s)
        du = RT[(dc, 'D')][st][s] @ v - u0[s]; dm = margin(RT[(dc, 'D')]['logits'][s]) - margin(RT['id']['logits'][s])
        rows.append((dc, s, Oe[s] - O0[s], du, dm))
    res = dict(stage=st, k=Q.shape[1], v_top_units=[int(u) for u in np.argsort(-np.abs(v))[:6]], v_top_weights=np.round(v[np.argsort(-np.abs(v))[:6]], 3).tolist(), obs={})
    okn = np.isfinite(O0).all(1); Xn_base = np.c_[np.ones(okn.sum()), np.log(O0[okn, OBS.index('m')] + 1e-3), np.log(O0[okn, OBS.index('pT')])]
    DO = np.concatenate([r[2] for r in rows]); DU = np.concatenate([r[3] for r in rows]); JS = np.concatenate([r[1] for r in rows]); DCv = np.concatenate([np.full(len(r[1]), r[0]) for r in rows])
    DM = np.concatenate([r[4] for r in rows]); fin = np.isfinite(DO).all(1)
    rb = np.random.default_rng(9); BW = [np.bincount(rb.integers(0, J, J), minlength=J).astype(float) for _ in range(200 if sm else 1000)]
    res['delta_u_mean_per_dc'] = {f'{dc:+.2f}': dict(W=float(DU[(DCv == dc) & (y[JS] == 2)].mean()), Z=float(DU[(DCv == dc) & (y[JS] == 3)].mean())) for dc in DCS}
    res['u_sd_unedited'] = float(u0.std()); res['corr_du_dmargin'] = float(np.corrcoef(DU, DM)[0, 1])
    for i, nm in enumerate(OBS):
        if nm in ('m', 'pT'): continue
        do = DO[fin, i]; du = DU[fin]; js = JS[fin]
        moved = float(np.abs(do).mean()); r = dict(mean_abs_delta_obs=moved, mean_delta_obs_per_dc={f'{dc:+.2f}': float(DO[fin & (DCv == dc), i].mean()) for dc in DCS})
        Xn = np.c_[Xn_base, O0[okn, i]]; bn = np.linalg.lstsq(Xn, u0[okn], rcond=None)[0]; r['natural_slope'] = float(bn[-1])
        r['natural_partial_r'] = float(np.corrcoef(Xn[:, -1] - Xn[:, :-1] @ np.linalg.lstsq(Xn[:, :-1], Xn[:, -1], rcond=None)[0], u0[okn] - Xn[:, :-1] @ np.linalg.lstsq(Xn[:, :-1], u0[okn], rcond=None)[0])[0, 1])
        if moved < 1e-4 * (np.nanmean(np.abs(O0[:, i])) + 1e-12):
            r['note'] = 'not moved by D (fixed by construction)'; res['obs'][nm] = r; continue
        A = np.c_[np.ones(len(do)), do]; be = np.linalg.lstsq(A, du, rcond=None)[0]; pred = du - A @ be
        r['edit_slope'] = float(be[1]); r['edit_r2'] = float(1 - pred.var() / du.var())
        r['calibrated_slope'] = float(be[1] / bn[-1]) if bn[-1] != 0 else None
        p3 = DCv[fin] == 0.3; du3, do3, js3 = du[p3], do[p3], js[p3]   # matched-change ratio on the dc +0.3 cell (pooled dc means cancel)
        r['ratio_mean_dc03'] = float(du3.mean() / (bn[-1] * do3.mean())) if bn[-1] * do3.mean() != 0 else None
        cs, rm = [], []
        oknj = np.flatnonzero(okn)
        for wj in BW:
            ww = wj[js]; ws = wj[oknj]
            A_ = A * np.sqrt(ww)[:, None]; b_ = np.linalg.lstsq(A_, du * np.sqrt(ww), rcond=None)[0]
            Xw = Xn * np.sqrt(ws)[:, None]; bnw = np.linalg.lstsq(Xw, u0[okn] * np.sqrt(ws), rcond=None)[0]
            cs.append(b_[1] / bnw[-1]); w3 = wj[js3]; rm.append((w3 @ du3) / (bnw[-1] * (w3 @ do3)))
        r['calibrated_slope_ci'] = qci(cs); r['ratio_mean_dc03_ci'] = qci(rm)
        res['obs'][nm] = r
    A = np.c_[np.ones(fin.sum()), DO[fin][:, [OBS.index(o) for o in ('z_min', 'D2', 'tau21', 'pTD', 'lead_frac')]]]
    be = np.linalg.lstsq(A, DU[fin], rcond=None)[0]; res['multi_r2'] = float(1 - (DU[fin] - A @ be).var() / DU[fin].var())
    return res, v


def steering(net, DT, RT, v, st, nrand, dmass):
    """add delta v at stage st on unedited jets; Delta margin per unit vs D's per unit"""
    from interp.x10_das_jedi import proj
    y = DT['y']; h0 = RT['id'][st]; m0 = margin(RT['id']['logits']); u0 = h0 @ v
    s3 = np.flatnonzero(DT['acc'][0.3]); scale = float(np.abs(RT[(0.3, 'D')][st][s3] @ v - u0[s3]).mean())
    cs = (-2, -1, -0.5, 0.5, 1, 2)

    def slope_dir(vv):
        out = {}
        dl, dmg, cl = [], [], []
        for c in cs:
            h = proj(net.rng_[st], h0, h0 + c * scale * vv[None, :]); lo = net.exact(st, h)[0]
            dl.append(np.full(len(h0), c * scale)); dmg.append(margin(lo) - m0); cl.append(y)
        dl, dmg, cl = map(np.concatenate, (dl, dmg, cl))
        for c_, nm in ((2, 'W'), (3, 'Z')):
            s = cl == c_; out[nm] = float((dl[s] * dmg[s]).sum() / (dl[s] ** 2).sum())
        return out, (dl, dmg, cl)
    st_v, (dl, dmg, cl) = slope_dir(v)
    res = dict(stage=st, scale_mean_abs_du_D03=scale, steer_slope=st_v, steer_dmargin_by_c={f'{c:+.1f}': dict(W=float(dmg[(dl == c * scale) & (cl == 2)].mean()), Z=float(dmg[(dl == c * scale) & (cl == 3)].mean())) for c in cs})
    D_sl = {}
    for c_, nm in ((2, 'W'), (3, 'Z')):
        du, dm = [], []
        for dc in DCS:
            s = np.flatnonzero(DT['acc'][dc] & (y == c_)); du.append(RT[(dc, 'D')][st][s] @ v - u0[s]); dm.append(margin(RT[(dc, 'D')]['logits'][s]) - m0[s])
        du, dm = np.concatenate(du), np.concatenate(dm); D_sl[nm] = float((du * dm).sum() / (du ** 2).sum())
    res['D_slope_per_unit_u'] = D_sl; res['ratio_steer_over_D'] = {nm: (st_v[nm] / D_sl[nm] if D_sl[nm] != 0 else None) for nm in ('W', 'Z')}
    # bootstrap over jets for the ratio
    rng = np.random.default_rng(11); J = DT['J']; rat = {'W': [], 'Z': []}
    jet_idx = np.tile(np.arange(J), len(cs))
    du_all, dm_all, js_all = [], [], []
    for dc in DCS:
        s = np.flatnonzero(DT['acc'][dc]); du_all.append(RT[(dc, 'D')][st][s] @ v - u0[s]); dm_all.append(margin(RT[(dc, 'D')]['logits'][s]) - m0[s]); js_all.append(s)
    du_all, dm_all, js_all = map(np.concatenate, (du_all, dm_all, js_all))
    for _ in range(300):
        w = np.bincount(rng.integers(0, J, J), minlength=J).astype(float)
        for c_, nm in ((2, 'W'), (3, 'Z')):
            s = cl == c_; ww = w[jet_idx][s]; sl = (ww * dl[s] * dmg[s]).sum() / (ww * dl[s] ** 2).sum()
            t_ = y[js_all] == c_; wd = w[js_all][t_]; sd_ = (wd * du_all[t_] * dm_all[t_]).sum() / (wd * du_all[t_] ** 2).sum(); rat[nm].append(sl / sd_)
    res['ratio_ci'] = {nm: qci(rat[nm]) for nm in rat}
    sd = h0.std(0); live = [int(u) for u in np.flatnonzero(sd > 1e-9)]; rr = np.random.default_rng(12); rs = {'W': [], 'Z': []}
    for _ in range(nrand):
        g = rand_live(rr, live, h0.shape[1], 1)[:, 0]; o, _ = slope_dir(g)
        for nm in rs: rs[nm].append(o[nm])
    res['random_dirs'] = {nm: dict(n=nrand, abs_q50=float(np.median(np.abs(rs[nm]))), abs_q95=float(np.quantile(np.abs(rs[nm]), .95)),
                                   p_abs=float((1 + (np.abs(rs[nm]) >= abs(st_v[nm])).sum()) / (1 + nrand))) for nm in rs}
    o, _ = slope_dir(dmass); res['dilation_direction_slope'] = o; res['cos_v_dilation_dir'] = float(abs(v @ dmass))
    return res


# ------------------------------------------------------------------ arms
def arm_das(a, untr=False):
    from interp.x10_das_jedi import JNet
    sm = a.smoke; tag = ('e37_untr' if untr else 'e37_das') + ('_smoke' if sm else ''); OUT.mkdir(parents=True, exist_ok=True)
    steps = 60 if sm else 400; nrand = 20 if sm else 1000
    (xF, yF, rF), (xT, yT, rT), info = load_split(sm)
    res = dict(node='E37', arm='untr' if untr else 'das', smoke=sm, dcs=list(DCS), ks=list(KS), samples=info, steps=steps, nrand=nrand)
    log('samples', info)
    log('fit sample edits'); DF = prep(xF, yF, rF, FSEED, sm=sm)
    log('test sample edits'); DT = prep(xT, yT, rT, TSEED, exclude_rows=rF, sm=sm)
    res['N'] = dict(F_WZ=DF['J'], T_WZ=DT['J'], accepted_F={str(dc): int(DF['acc'][dc].sum()) for dc in DCS}, accepted_T={str(dc): int(DT['acc'][dc].sum()) for dc in DCS},
                    cross_F=int(DF['cross'][2].sum()), cross_T=int(DT['cross'][2].sum()), max_rel_p4_change=max(DF['dP'], DT['dP']),
                    c0_mean_T=float(np.nanmean(DT['c0'])))
    log('N', res['N'])
    import gc, jax
    nets = ['trained'] if not untr else [f'untrained{s}' for s in range(3)]
    for name in nets:
        net = JNet() if name == 'trained' else JNet(untrained=True, seed=int(name[-1]))
        t = time.time(); net.off = np.zeros(5)
        if untr:
            net.off = net.net.run(DF['X']['id'])['logits'].mean(0)
        R0 = net.reps(xT, keep=('S', 'h4', 'h5', 'logits'))
        out = dict(acc_T_all_stored=float(((R0['logits'] - net.off).argmax(1) == yT).mean()))
        out['acc_F_all_stored'] = float(((net.net.run(xF)['logits'] - net.off).argmax(1) == yF).mean())
        R0c = {k: v[:2000] for k, v in R0.items()}; R0c['logits'] = R0c['logits'] - net.off
        out['tail_check'] = net.check(xT[:2000], R0c)
        RF = reps_all(net, DF); RT = reps_all(net, DT)
        out['acc_T_WZ_rebuilt'] = float((RT['id']['logits'].argmax(1) == DT['y']).mean())
        wzm = margin(RT['id']['logits']); out['margin_sd_T'] = float(wzm.std())
        for dc in DCS:
            s = DT['acc'][dc]
            for kd in ('D', 'cn'):
                dm = margin(RT[(dc, kd)]['logits']) - wzm
                out[f'dmargin_{kd}_{dc:+.2f}'] = dict(W=float(dm[s & (DT['y'] == 2)].mean()), Z=float(dm[s & (DT['y'] == 3)].mean()))
        for f in (0.95, 1.05):
            dm = margin(RT[('dil', f)]['logits']) - wzm; out[f'dmargin_dil_{f}'] = dict(W=float(dm[DT['y'] == 2].mean()), Z=float(dm[DT['y'] == 3].mean()))
        log(f'{name}: acc all-class stored T {out["acc_T_all_stored"]:.4f} F {out["acc_F_all_stored"]:.4f}; W+Z rebuilt {out["acc_T_WZ_rebuilt"]:.4f}; '
            f'D+0.3 {out["dmargin_D_+0.30"]}, cn+0.3 {out["dmargin_cn_+0.30"]}, dil1.05 {out["dmargin_dil_1.05"]} [{time.time() - t:.0f}s]')
        r, Qs = das_arm(net, DF, DT, RF, RT, steps, nrand if not untr else (10 if sm else 100), name, sm, shuffled=not untr, rand_eval=True)
        out['das'] = r
        if not untr:
            ph = []; keys = []
            for st in STAGES:
                for k in KS:
                    rr_ = r[st]['k'].get(str(k))
                    if rr_ and rr_['random']['self']['p_iia'] is not None: ph.append(rr_['random']['self']['p_iia']); keys.append(f'{st}|{k}')
            out['holm_iia_self'] = dict(zip(keys, holm(ph)))
            np.savez_compressed(OUT / f'{tag}_Q.npz', **{f'{st}_k{k}': Q for (st, k), Q in Qs.items()})
            res[name] = out; (OUT / f'{tag}.json').write_text(json.dumps(jsonable(res), indent=1)); log('DAS written')
            # specificity: principal angles
            mo, Ms = mass_subspaces(net, DF, RF, 100 if sm else 800, sm); out['mass_refit'] = mo
            np.savez_compressed(OUT / f'{tag}_M.npz', **{f'{V}_{st}': M for (V, st), M in Ms.items()})
            out['angles'] = angles(Qs, Ms, RF, nrand); log('angles: ' + json.dumps(jsonable({k: (v['max_cos'], v['random_max_cos_q'][1], v['p_closer_than_random']) for k, v in out['angles'].items() if k.split('|')[1] in ('1', '3')})))
            res[name] = out; (OUT / f'{tag}.json').write_text(json.dumps(jsonable(res), indent=1))
            # arm 2: primary Q = h4 with the k <= 3 of best validation score (fit sample)
            kb = max((1, 2, 3), key=lambda k: r['h4']['k'][str(k)]['train']['val_score']); out['primary'] = dict(stage='h4', k=kb, rule='h4, k <= 3 with the best fit-sample validation score')
            out['readout'] = {}
            for st, k in (('h4', kb), ('h4', 1), ('h5', 1), ('h5', 3), ('S', 3)):
                key = f'{st}|{k}'
                if key in out['readout']: continue
                ro, v = readout(net, DT, RT, DF, RF, Qs[(st, k)], st, sm); out['readout'][key] = ro
                log(f'readout {key}: corr(du, dmargin) {ro["corr_du_dmargin"]:+.3f}; multi R2 {ro["multi_r2"]:.3f}; ' + '; '.join(
                    f'{nm}: edit R2 {o.get("edit_r2")} cal slope {o.get("calibrated_slope")} {o.get("calibrated_slope_ci")} ratio(+0.3) {o.get("ratio_mean_dc03")}' for nm, o in ro['obs'].items()))
                if st == 'h4' and k == kb:
                    out['steering'] = steering(net, DT, RT, v, 'h4', 20 if sm else 200, Ms[('dil', 'h4')][:, 0])
                    log('steering: ' + json.dumps(jsonable(out['steering'])))
                res[name] = out; (OUT / f'{tag}.json').write_text(json.dumps(jsonable(res), indent=1))
            # example jet for the figure (test W jet, D+0.3 flips W -> Z)
            y = DT['y']; c0 = RT['id']['logits'].argmax(1); c3 = RT[(0.3, 'D')]['logits'].argmax(1)
            pick = np.flatnonzero((y == 2) & DT['acc'][0.3] & (DT['nn'] >= 15) & (DT['nn'] <= 40) & (c0 == 2) & (c3 == 3))
            if len(pick):
                j = int(pick[0]); n = int(DT['nn'][j]); ex = dict(index=j, N=n, c0=float(DT['c0'][j]), prong=DT['LAB'][j, :n].tolist())
                for nm, P in (('unedited', DT['P4'][j, :n]), ('D+0.3', DT['PE'][(0.3, 'D')][j, :n])):
                    pt = np.hypot(P[:, 0], P[:, 1]); T = P.sum(0); eT = np.arcsinh(T[2] / np.hypot(T[0], T[1])); fT = np.arctan2(T[1], T[0])
                    ex[nm] = dict(pt=pt.tolist(), deta=(np.arcsinh(P[:, 2] / pt) - eT).tolist(), dphi=wrap(np.arctan2(P[:, 1], P[:, 0]) - fT).tolist(),
                                  obs=dict(zip(OBS, jet_obs(P, DT['LAB'][j, :n]).tolist())))
                ex['margin'] = dict(unedited=float(margin(RT['id']['logits'][j:j + 1])[0]), D=float(margin(RT[(0.3, 'D')]['logits'][j:j + 1])[0]))
                out['example'] = ex
        res[name] = out; (OUT / f'{tag}.json').write_text(json.dumps(jsonable(res), indent=1))
        del net, RF, RT, R0; gc.collect(); jax.clear_caches()
    res['wall_s'] = time.time() - T0; (OUT / f'{tag}.json').write_text(json.dumps(jsonable(res), indent=1)); log(f'wrote {OUT / tag}.json; E37 {res["arm"]} DONE')


def arm_recur(a):
    from interp import e33_program_recurrence_jedi as E
    from interp.jedi_core import Jedi
    sm = a.smoke; tag = 'e37_recur' + ('_smoke' if sm else ''); OUT.mkdir(parents=True, exist_ok=True); per = sm or 2000
    dcs = (-0.3, 0.3); res = dict(node='E37', arm='recur', smoke=sm, dcs=list(dcs), sample=dict(seed=TSEED, per_class=per), nets={})
    cache = {}

    def inputs(n, f16):
        key = 128 if n == 128 else ('f16' if f16 else 64)
        if key in cache: return cache[key]
        xin, xk, x3, y, chk = E.load(per, TSEED, 128 if n == 128 else 64, f16)
        wz = (y == 2) | (y == 3); xk = np.asarray(xk[wz]); y = y[wz]; L = xk.shape[1]
        P4, nn, LAB, c0, PE, acc, dP = make_edits(xk, TSEED, dcs=dcs)
        X = {}; ax0 = axis(P4, nn); X['id'], o0 = model_xn(P4, nn, ax0, L); perm = {'id': o0}
        for k, Pe in PE.items(): X[k], perm[k] = model_xn(Pe, nn, axis(Pe, nn), L)
        if f16:
            X16 = np.asarray(xin[wz]); x3w = np.asarray(x3[wz]); Xo = {}
            for k in X:
                o = perm[k]; Xp = np.take_along_axis(X16, o[..., None], 1); x0p = np.take_along_axis(x3w, o[..., None], 1)
                Xo[k] = E.to16(Xp, x0p, X[k])
            X = Xo
        cache[key] = dict(X=X, y=y, acc=acc, dP=dP, N=len(y), chk=chk, xall=xin, yall_acc=None); log(f'inputs n {n} f16 {f16}: N {len(y)}; max rel p4 change {dP:.1e}')
        return cache[key]
    nb = 300 if sm else 1000; rng = np.random.default_rng(3709)

    def analyse(name, run, n, f16):
        d = inputs(n, f16); y = d['y']; X = d['X']; L0 = run(X['id']); m0 = margin(L0); sd = float(m0.std())
        r = dict(N=d['N'], margin_sd=sd, acc_WZ_rebuilt=float((L0.argmax(1) == y).mean()), call_dist=np.bincount(L0.argmax(1), minlength=5).tolist(), per_dc={})
        for dc in dcs:
            Ld, Lc = run(X[(dc, 'D')]), run(X[(dc, 'cn')]); dD = margin(Ld) - m0; dC = margin(Lc) - m0; acc = d['acc'][dc]; e = {}
            chD = np.sign(margin(Ld)) != np.sign(m0); chC = np.sign(margin(Lc)) != np.sign(m0)
            for c, nm in ((2, 'W'), (3, 'Z')):
                s = np.flatnonzero(acc & (y == c)); df = dD[s] - dC[s]; bs = [df[rng.integers(0, len(s), len(s))].mean() for _ in range(nb)]
                sf = rng.choice([-1.0, 1.0], (2000, len(df))); p = float((1 + (np.abs((sf * df).mean(1)) >= abs(df.mean())).sum()) / 2001)
                e[nm] = dict(n=len(s), dD=float(dD[s].mean()), dC=float(dC[s].mean()), diff=float(df.mean()), diff_ci=qci(bs), diff_sd_units=float(df.mean() / sd) if sd > 1e-9 else None,
                             diff_sd_ci=[v / sd for v in qci(bs)] if sd > 1e-9 else None, p_signflip=p, mcnemar_wz_change=mcnemar(chD[s], chC[s]))
            e['pattern'] = bool(e['W']['diff_ci'][1] < 0 and e['Z']['diff_ci'][0] > 0)
            e['contrast_sd'] = float((e['Z']['diff'] - e['W']['diff']) / sd) if sd > 1e-9 else 0.0
            r['per_dc'][f'{dc:+.2f}'] = e
        e = r['per_dc']['+0.30']
        log(f'{name}: acc W+Z {r["acc_WZ_rebuilt"]:.4f}; sd {sd:.3f}; dc+0.3 W D-cn {e["W"]["diff"]:+.3f} {np.round(e["W"]["diff_ci"], 3).tolist()} Z {e["Z"]["diff"]:+.3f} '
            f'{np.round(e["Z"]["diff_ci"], 3).tolist()} pattern {e["pattern"]}; McNemar W {e["W"]["mcnemar_wz_change"]} Z {e["Z"]["mcnemar_wz_change"]}; '
            f'dc-0.3 W {r["per_dc"]["-0.30"]["W"]["diff"]:+.3f} Z {r["per_dc"]["-0.30"]["Z"]["diff"]:+.3f}')
        return r
    for name in E.TRAINED:
        t = time.time(); M = E.Model(name); kind, fam, n = E.MODELS[name]; f16 = M.f16
        run = lambda x, M=M: M.run(x)['logits']
        r = analyse(name, run, n, f16); r['family'] = fam; r['n'] = n
        # all-class accuracy on stored inputs of the same sample
        xin, _, _, yall, _ = E.load(per, TSEED, 128 if n == 128 else 64, f16); r['acc_all_stored'] = float((run(np.asarray(xin)).argmax(1) == yall).mean())
        log(f'  {name} all-class accuracy (stored) {r["acc_all_stored"]:.4f} [{time.time() - t:.0f}s]')
        res['nets'][name] = r; (OUT / f'{tag}.json').write_text(json.dumps(jsonable(res), indent=1)); del M
    for s in range(3 if sm else 10):
        un = Jedi(64, '3-feature', untrained=True, seed=s); run = lambda x, un=un: un.run(x)['logits']
        res['nets'][f'untrained{s}'] = analyse(f'untrained{s}', run, 64, False); del un
        (OUT / f'{tag}.json').write_text(json.dumps(jsonable(res), indent=1))
    # network-level tests
    TR = [k for k in res['nets'] if not k.startswith('untrained')]; UN = [k for k in res['nets'] if k.startswith('untrained')]
    summ = {}
    for dc in ('+0.30', '-0.30'):
        pt = [res['nets'][k]['per_dc'][dc]['pattern'] for k in TR]; pu = [res['nets'][k]['per_dc'][dc]['pattern'] for k in UN]
        a_, b_ = sum(pt), sum(pu); n1, n2 = len(pt), len(pu); tot = a_ + b_
        from math import comb
        pf = sum(comb(n1, i) * comb(n2, tot - i) for i in range(a_, min(n1, tot) + 1)) / comb(n1 + n2, tot)
        ct = np.array([res['nets'][k]['per_dc'][dc]['contrast_sd'] for k in TR]); cu = np.array([res['nets'][k]['per_dc'][dc]['contrast_sd'] for k in UN])
        allc = np.r_[ct, cu]; obs = ct.mean() - cu.mean(); pr = np.random.default_rng(5); cnt = 0
        for _ in range(10000):
            p_ = pr.permutation(len(allc)); cnt += (allc[p_[:len(ct)]].mean() - allc[p_[len(ct):]].mean()) >= obs
        mc = [res['nets'][k]['per_dc'][dc][c]['mcnemar_wz_change']['p'] for k in TR for c in ('W', 'Z')]
        summ[dc] = dict(pattern_trained=f'{a_}/{n1}', pattern_untrained=f'{b_}/{n2}', fisher_one_sided_p=float(pf),
                        contrast_sd_trained=ct.tolist(), contrast_sd_untrained=cu.tolist(), contrast_perm_p=float((1 + cnt) / 10001),
                        sign_trained_W_neg=int(sum(res['nets'][k]['per_dc'][dc]['W']['diff'] < 0 for k in TR)),
                        sign_trained_Z_pos=int(sum(res['nets'][k]['per_dc'][dc]['Z']['diff'] > 0 for k in TR)),
                        mcnemar_holm=dict(zip([f'{k}|{c}' for k in TR for c in ('W', 'Z')], holm(mc))))
        log(f'recurrence {dc}: pattern trained {a_}/{n1}, untrained {b_}/{n2}, Fisher p {pf:.2e}; contrast perm p {summ[dc]["contrast_perm_p"]:.4f}')
    res['summary'] = summ; res['wall_s'] = time.time() - T0
    (OUT / f'{tag}.json').write_text(json.dumps(jsonable(res), indent=1)); log(f'wrote {OUT / tag}.json; E37 recur DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='das', choices=['das', 'untr', 'recur']); ap.add_argument('--smoke', type=int, default=0)
    a = ap.parse_args()
    if a.arm == 'recur': arm_recur(a)
    else: arm_das(a, untr=a.arm == 'untr')
