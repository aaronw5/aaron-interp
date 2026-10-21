"""E35 (claims A4, A3, D18 handed from loop D; stage 3): through which program elements does JEDI-linear (3-feature n64,
official, fixed point) respond to two-prong momentum sharing at fixed jet four-momentum?

Edit D(dc) (H45, interp/h45_prong_sharing.py edit_one): the jet (massless constituents, pT, eta = deta, phi = dphi about the stored
  axis) is split into two exclusive-kT prongs (p = 1, E-scheme, rapidity-azimuth distance; every constituent belongs to one prong).
  All constituents are boosted to the jet rest frame (beta = p_jet / E_jet); n+ = rest-frame direction of the forward prong,
  c0 = |cos theta*| = n+ . beta_hat; target c1 = clip(c0 + dc, 0, 0.95); every constituent is rotated by
  alpha = arccos c0 - arccos c1 about k = n+ x beta_hat and boosted back. One Lorentz transformation for all constituents: jet
  four-momentum (pT, eta, phi, m), every pairwise invariant mass (hence each prong mass) and N are exactly unchanged; the prong
  energy sharing z and the lab angles change. dc in {-0.3, -0.15, +0.15, +0.3}; dc > 0 = more asymmetric prongs.
  Network inputs are rebuilt about each jet's own E-scheme axis, pT-sorted (H45 model_x), for edited and unedited jets alike.
Controls (same jets, same rest-frame angle |alpha|):
  cn    rotation by +-|alpha| (random sign) about the prong axis n+ in the rest frame: both prong four-momenta (hence z,
        |cos theta*|, Delta R) exactly fixed; constituents turn about their prong axis.
  cphi  H45's control: rotation by +-|alpha| about beta_hat in the rest frame (z and |cos theta*| fixed; prongs move around the axis).
  R     rigid rotation of the whole jet about its axis in (deta, dphi) by theta (random sign), 2 sin(theta/2) sum pT r / sum pT =
        D's pT-weighted mean constituent displacement: every r, pT, M0, M2, N fixed.
  Acceptance (H45): D and cphi keep both prong centres within Delta R 0.75 of the jet axis; jets with >= 3 particles.
Readouts per edit: margin = logit_W - logit_Z (d7 logits; = log p_W / p_Z), Delta margin = edit - unedited per jet; W->Z (Z->W) =
  % of W (Z) jets whose call goes from W to Z (Z to W); changes of M0 = sum pT, |M1| = |sum pT (deta, dphi)|, M2 = sum pT r^2 (inputs
  about the jet's own axis), invariant mass m, and every live S unit in units of its SD (unedited W+Z jets).
Route holds (edited jet, one element kept at its unedited value; exact fixed-point tail S -> h4 -> h5 -> z -> logits):
  singles S29 (pT-weighted squared radius + soft count, the mass-like object), S1 (sum over ranks >= 1 of (r - 0.2)+), S0 (pT sum of
  the 14 hardest), S19 (soft multiplicity), S30 (sum r over the core), S31 / S58 (g-thresholded dipole / quadrupole counts);
  named7 = all seven; radial = {S1, S30, S19}; h4dir / h5dir = the mass direction at h4 / h5, held by projection
  (h' = h - ((h - h_0) . d) d in SD-standardised coordinates); mass = S29 + h4dir; mass5 = S29 + h5dir; g = all 64 context units
  (full forward with g overridden). The X10 DAS rotations were not stored, so the mass direction d is the mean standardised
  displacement of h4 (h5) under a real mass change (all angles x 1.05, dil_M2) on the same W+Z jets.
  Share removed = 1 - P_held / P_free, P = sum over dc of s_dc x mean Delta margin (s_dc = sign of the free mean; W+Z jets accepted at
  that dc), also per dc and class. 95 % CI: 1000 jet bootstrap resamples.
  Nulls (point estimates of the same share): 'sets' = size-matched live-S unit sets (19 singles; all 969 triples; 1000 random 7-sets);
  'dirs' = 1000 random k-dim subspaces of the standardised live-S space (k = 1, 3, 7) or of live h4 / h5 (k = 1); mass set null =
  random single live S + random h4 direction. p = (1 + #null >= named) / (1 + #null); Holm over the 12 holds with a null.
Program fidelity: the same edits on (i) the float program with HGQ's zero-bit masks (E3m 'mask', no rounding) and (ii) the E11
  hinge-formula chain (b): P <- physics, g <- P^, S <- g^ + physics, h4 <- S^, h5 <- h4^, z <- h5^, the network's last layer d7
  kept (physics recomputed from the edited inputs with jetdistill.observables.compute + handbuilt); variant (iii) 'chainS' = the
  network's own P and g, formulas from S on. Per program: call-change agreement with the network per jet, share of network flips
  reproduced (any change / same new call), McNemar (changed by network vs by program, paired over jets), corr of Delta margin.
Untrained: Jedi(untrained=True, seeds 0-9): free D and cn response in units of each network's margin SD (unedited W+Z jets).
Calibration: ridge probe of log m on standardised live h4 (fitted on a random half of all unedited jets, scored on the rest);
  Delta log m^ under D per dc; matched dilation dil_M2 with angle factor k_dc = exp(Delta log m^_D / gain), gain = the probe's
  response per unit log dilation (from +-5 %); R = mean Delta margin(D) / mean Delta margin(dil matched) (W+Z jets).
Samples: main sample(2000, 3501) (10,000 jets; W, Z primary; q, g, t secondary), rep sample(2000, 3502) (no untrained / programs).
Run: ./job.sh e35-run 1 2 'python -m interp.e35_prong_sharing_route_jedi'   (--smoke K: K jets per class, fewer nulls)
     ./job.sh e35-singles 1 2 'python -m interp.e35_prong_sharing_route_jedi --singles'  (per-cell shares of all 19 live S and 171 pairs)
Writes $AI_RESULTS/jedi/e35_prong_sharing_route[_smoke].json and ..._perjet.npz.
"""
import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp.h45_prong_sharing import boost, rodrigues, ptetaphim, p4, rap, wrap, edit_one, prong_offsets_ok, model_x, axis, CMAX  # noqa: E402
from interp.e28w_split_route_jedi import binom_p, mcnemar, holm, qci, LIVE_S  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0_ = time.time()
CL = ['g', 'q', 'W', 'Z', 't']
DCS = (-0.3, -0.15, 0.15, 0.3)
VAR = ('D', 'cn', 'cphi', 'R')
NAMED = [29, 1, 0, 19, 30, 31, 58]
DESC = {29: 'pT-weighted squared radius + soft count (mass-like)', 1: 'sum over ranks >= 1 of (r - 0.2)+', 0: 'pT sum of the 14 hardest',
        19: 'soft multiplicity', 30: 'sum of r over the core', 31: 'g-thresholded dipole count', 58: 'g-thresholded quadrupole count'}


def log(*a): print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return jsonable(o.tolist())
    return o


# ------------------------------------------------------------------ prongs (numpy port of h41_colour_flow._ktn_labels, nobj = 2)
def kt2_labels(P):
    n = len(P); v = P.astype(np.float64).copy()
    def kin(w):
        pt = np.hypot(w[:, 0], w[:, 1]); e = np.maximum(w[:, 3], np.abs(w[:, 2]) + 1e-9)
        return pt, np.arctan2(w[:, 1], w[:, 0]), 0.5 * np.log((e + w[:, 2]) / (e - w[:, 2]))
    pt, ph, yy = kin(v); alive = np.ones(n, bool); lab = np.arange(n)
    dphi = np.abs(ph[:, None] - ph[None, :]); dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    D = np.minimum(pt[:, None] ** 2, pt[None, :] ** 2) * ((yy[:, None] - yy[None, :]) ** 2 + dphi ** 2)
    D[np.arange(n), np.arange(n)] = np.inf
    for _ in range(n - 2):
        f = int(np.argmin(D)); a, b = divmod(f, n); a, b = min(a, b), max(a, b)
        v[a] += v[b]; alive[b] = False; lab[lab == b] = a; D[b, :] = np.inf; D[:, b] = np.inf
        p_, h_, y_ = kin(v[a:a + 1]); pt[a], ph[a], yy[a] = p_[0], h_[0], y_[0]
        dp = np.abs(ph[a] - ph); dp = np.where(dp > np.pi, 2 * np.pi - dp, dp)
        row = np.minimum(pt[a] ** 2, pt ** 2) * ((yy[a] - yy) ** 2 + dp ** 2); row[~alive] = np.inf; row[a] = np.inf
        D[a, :] = row; D[:, a] = row
    out = -np.ones(n, np.int64)
    for k, a in enumerate(np.flatnonzero(alive)): out[lab == a] = k
    return out


def edit_jet(P, lab, dc, rng):
    """D, cphi (as h45.edit_one) plus cn (rotation about the prong axis n+ by the same |alpha|); info"""
    s0 = lab == 0
    if s0.sum() < 1 or (~s0).sum() < 1: return None, dict(reason='empty_prong')
    Va = P[s0].sum(0); T = P.sum(0); be = T[:3] / T[3]; bn = np.linalg.norm(be)
    if bn < 1e-6: return None, dict(reason='at_rest')
    nh = be / bn; na = boost(Va, be)[:3]; na = na / np.linalg.norm(na); cs = float(na @ nh); npl = na if cs >= 0 else -na
    c0 = abs(cs); c1 = float(np.clip(c0 + dc, 0.0, CMAX)); al = float(np.arccos(min(c0, 1.0)) - np.arccos(c1))
    k = np.cross(npl, nh)
    if np.linalg.norm(k) < 1e-9: return None, dict(reason='collinear')
    Pr = boost(P, be)

    def apply(R):
        Q = Pr.copy(); Q[:, :3] = Q[:, :3] @ R.T; return boost(Q, -be)
    sg = rng.choice([-1.0, 1.0], 2)
    out = {'D': apply(rodrigues(k, al)), 'cphi': apply(rodrigues(nh, sg[0] * abs(al))), 'cn': apply(rodrigues(npl, sg[1] * abs(al)))}
    pt0, e0, p0, _ = ptetaphim(P); _, e1, p1, _ = ptetaphim(out['D'])
    DS = float((pt0 * np.hypot(e1 - e0, wrap(p1 - p0))).sum() / pt0.sum())
    return out, dict(reason='ok', c0=c0, c1=c1, alpha=al, disp=DS)


def make_edits(x, rng_seed, smoke):
    """x (J, 64, 3) stored inputs -> dict of model inputs: id, (dc, kind); acceptance; info; prong labels (P4 order)"""
    J = len(x); real = x[..., 0] > 0; nn = real.sum(1).astype(np.int64)
    P4 = (p4(x[..., 0].astype(np.float64), x[..., 1].astype(np.float64), x[..., 2].astype(np.float64), 0.0) * real[..., None])
    ax0 = axis(P4, nn); X = {'id': model_x(P4, nn, ax0)}
    LAB = -np.ones((J, 64), np.int64); t = time.time()
    for j in range(J):
        if nn[j] >= 3: LAB[j, :nn[j]] = kt2_labels(P4[j, :nn[j]])
    log(f'  kT-2 prongs for {J} jets in {time.time() - t:.0f}s')
    acc = {dc: np.zeros(J, bool) for dc in DCS}; accphi = {dc: np.zeros(J, bool) for dc in DCS}
    info = {dc: dict(c0=np.full(J, np.nan), c1=np.full(J, np.nan), alpha=np.full(J, np.nan), disp=np.full(J, np.nan), theta=np.full(J, np.nan),
                     dPjet=np.zeros(J), reasons={}) for dc in DCS}
    PE = {(dc, k): P4.copy() for dc in DCS for k in ('D', 'cn', 'cphi')}
    chk = []
    t = time.time()
    for j in np.flatnonzero(nn >= 3):
        n = nn[j]; P = P4[j, :n]; lab = LAB[j, :n]
        for i, dc in enumerate(DCS):
            rng = np.random.default_rng([rng_seed, int(j), i]); r, inf = edit_jet(P, lab, dc, rng); I = info[dc]
            I['reasons'][inf['reason']] = I['reasons'].get(inf['reason'], 0) + 1
            if r is None: continue
            for kk in ('c0', 'c1', 'alpha', 'disp'): I[kk][j] = inf[kk]
            I['dPjet'][j] = float(np.abs(r['D'].sum(0) - P.sum(0)).max() / max(P.sum(0)[3], 1e-9))
            acc[dc][j] = prong_offsets_ok(r['D'], lab); accphi[dc][j] = prong_offsets_ok(r['cphi'], lab)
            for kk in ('D', 'cn', 'cphi'): PE[(dc, kk)][j, :n] = r[kk]
            if len(chk) < 20 and i == 3:   # identity with H45's own edit_one for D
                r2, _ = edit_one(P, lab, dc, np.random.default_rng(0)); chk.append(float(np.abs(r2['D'] - r['D']).max()))
    log(f'  edits in {time.time() - t:.0f}s; D vs h45.edit_one max |diff| {max(chk) if chk else None}; ' +
        json.dumps({dc: info[dc]['reasons'] for dc in DCS}))
    for (dc, kk), Pe in PE.items(): X[(dc, kk)] = model_x(Pe, nn, axis(Pe, nn))
    # R: rigid rotation of the unedited input about its axis, displacement matched to D
    rr = np.random.default_rng(rng_seed + 7)
    pt = X['id'][..., 0].astype(np.float64); r_ = np.hypot(X['id'][..., 1], X['id'][..., 2]); mr = (pt * r_).sum(1) / np.maximum(pt.sum(1), 1e-9)
    for dc in DCS:
        s = np.clip(np.nan_to_num(info[dc]['disp']) / np.maximum(2 * mr, 1e-9), 0, 1); th = 2 * np.arcsin(s) * rr.choice([-1.0, 1.0], J)
        info[dc]['theta'] = th; c, sn = np.cos(th)[:, None], np.sin(th)[:, None]
        e = X['id'][..., 1] * c - X['id'][..., 2] * sn; f = X['id'][..., 1] * sn + X['id'][..., 2] * c
        Pr = p4(pt, e, f, 0.0) * (pt > 0)[..., None]; X[(dc, 'R')] = model_x(Pr, nn, axis(Pr, nn))
    minv = {k: ptetaphim((p4(v[..., 0].astype(np.float64), v[..., 1].astype(np.float64), v[..., 2].astype(np.float64), 0.0) * (v[..., :1] > 0)).sum(1))[3] for k, v in X.items()}
    return X, acc, accphi, info, LAB, nn, minv


def moments(X):
    x = X.astype(np.float64); pt = x[..., 0]; M0 = pt.sum(1); M1 = np.hypot((pt * x[..., 1]).sum(1), (pt * x[..., 2]).sum(1))
    M2 = (pt * (x[..., 1] ** 2 + x[..., 2] ** 2)).sum(1); return dict(M0=M0, M1=M1, M2=M2)


# ------------------------------------------------------------------ network helpers
class Tail:
    """exact fixed-point tail with projection holds: S' = S - (S - S0) MS; h4' = h4 - (h4 - h40) M4; h5' = h5 - (h5 - h50) M5"""

    def __init__(self, net):
        import jax, jax.numpy as jnp
        L = net.L; self.jnp = jnp; self.kd = bool(getattr(L['qs1'], 'keepdims', False))
        KB = {k: (jnp.asarray(net.KB[k][0], jnp.float32), jnp.asarray(net.KB[k][1], jnp.float32)) for k in ('d4', 'd5', 'd6', 'd7')}

        def dense(k, t, act=True):
            z = jnp.einsum(L[k].equation, L[k].iq(t, training=False), KB[k][0]) + KB[k][1]
            return L[k].activation(z) if act and L[k].activation is not None else z

        def f(S, S0, MS, h40, M4, h50, M5, lo, hi):
            S = jnp.clip(S - (S - S0) @ MS, lo[0], hi[0])
            h4 = dense('d4', S); h4 = jnp.clip(h4 - (h4 - h40) @ M4, lo[1], hi[1])
            h5 = dense('d5', h4); h5 = jnp.clip(h5 - (h5 - h50) @ M5, lo[2, :32], hi[2, :32])
            z = dense('d6', h5, act=False); lo_ = dense('d7', jax.nn.relu(z))
            return lo_.reshape(S.shape[0], -1)
        self._f = jax.jit(f)

    def sh(self, a): a = np.asarray(a, np.float32); return a[:, None, :] if self.kd else a

    def __call__(self, S, S0, MS=None, h40=None, M4=None, h50=None, M5=None, lo=None, hi=None):
        J = len(S); z64 = np.zeros((64, 64), np.float32); z32 = np.zeros((32, 32), np.float32)
        h40 = np.zeros((J, 64)) if h40 is None else h40; h50 = np.zeros((J, 32)) if h50 is None else h50
        lo = np.full((3, 64), -1e9, np.float32) if lo is None else lo; hi = np.full((3, 64), 1e9, np.float32) if hi is None else hi
        r = self._f(self.sh(S), self.sh(S0), z64 if MS is None else np.asarray(MS, np.float32), self.sh(h40), z64 if M4 is None else np.asarray(M4, np.float32),
                    self.sh(h50), z32 if M5 is None else np.asarray(M5, np.float32), np.asarray(lo, np.float32), np.asarray(hi, np.float32))
        return np.asarray(r, np.float64)


def proj(Q, sd):
    """row-vector projection matrix for a hold of the subspace Q (d x k, orthonormal in SD-standardised coordinates)"""
    sd = np.where(sd > 0, sd, 1.0); return (np.diag(1 / sd) @ Q @ Q.T @ np.diag(sd)).astype(np.float32)


def unitQ(units, d=64):
    Q = np.zeros((d, len(units))); Q[units, np.arange(len(units))] = 1; return Q


def randQ(rng, live, k, d=64):
    G = np.zeros((d, k)); G[live] = rng.standard_normal((len(live), k)); return np.linalg.qr(G)[0]


def mask_program(net):
    """E3m: float program with HGQ zero-bit entries as masks (no rounding, no wrap)"""
    import jax, jax.numpy as jnp
    L = net.L; KB = {k: (jnp.asarray(v[0], jnp.float32), jnp.asarray(v[1], jnp.float32)) for k, v in net.KB.items()}
    MASK = {nm: jnp.asarray(np.asarray(L[nm].iq.quantizer.b) > 0, jnp.float32) for nm in ('l1', 'qs', 'l2', 'dd', 'l3', 'qs1', 'd4', 'd5', 'd6', 'd7')}
    q = lambda nm, t: t * MASK[nm]
    def dense(k, t, act=True):
        z = jnp.einsum(L[k].equation, q(k, t), KB[k][0]) + KB[k][1]; return jax.nn.relu(z) if act and L[k].activation is not None else z
    pool = lambda k, a: jnp.sum(q(k, a), axis=L[k].axes, keepdims=L[k].keepdims) * L[k].scale
    def f(xn):
        a = dense('l1', xn); g = dense('dd', pool('qs', a)); s = dense('l2', a); u = dense('l3', s + g)
        h = dense('d5', dense('d4', pool('qs1', u))); return dense('d7', jax.nn.relu(dense('d6', h, act=False))).reshape(len(xn), -1)
    fj = jax.jit(f)
    return lambda X: np.concatenate([np.asarray(fj(net.inputs(X[i:i + 5000], 64)), np.float64) for i in range(0, len(X), 5000)])


# ------------------------------------------------------------------ statistics
class BootW:
    def __init__(self, J, B, seed):
        rng = np.random.default_rng(seed); self.W = np.stack([np.bincount(rng.integers(0, J, J), minlength=J) for _ in range(B)]).astype(np.float64)

    def mean(self, v, sel):
        W = self.W[:, sel]; return (W @ v[sel]) / np.maximum(W.sum(1), 1)


def signflip_p(d, rng, n=5000):
    d = d[np.isfinite(d)]
    if len(d) < 2: return None
    m = abs(d.mean()); s = rng.choice([-1.0, 1.0], (n, len(d))); return float((1 + (np.abs((s * d).mean(1)) >= m).sum()) / (n + 1))


def margin(L): return L[:, 2] - L[:, 3]


# ------------------------------------------------------------------ one sample
def analyse(net, tail, seed, per, smoke, full, rng):
    from interp.jedi_core import sample
    x, y, _ = sample(per_class=per, seed=seed); x = np.asarray(x, np.float32); J = len(x)
    out = dict(seed=seed, N=J, per_class=per)
    Lst = net.run(x)['logits']; out['acc_stored_inputs'] = float((Lst.argmax(1) == y).mean())
    log(f'sample {seed}: {J} jets; network accuracy on stored inputs {out["acc_stored_inputs"]:.4f}')
    X, acc, accphi, info, LAB, nn, minv = make_edits(x, seed, smoke)
    keep = ('logits', 'S', 'h4', 'h5', 'g', 'P')
    R0 = net.run(X['id'], keep=keep); L0 = R0['logits']; c0 = L0.argmax(1)
    out['acc_rebuilt_inputs'] = float((c0 == y).mean()); out['same_call_stored_vs_rebuilt'] = float((c0 == Lst.argmax(1)).mean())
    log(f'  accuracy on inputs rebuilt about the own axis {out["acc_rebuilt_inputs"]:.4f}; same call as stored {out["same_call_stored_vs_rebuilt"]:.4f}')
    chk = np.abs(tail(R0['S'], R0['S']) - L0).max(); out['tail_check_max_abs'] = float(chk); log(f'  tail check max |logit diff| {chk:.2e}')
    m0 = margin(L0); RE = {}
    for k, v in X.items():
        if k == 'id': continue
        RE[k] = net.run(v, keep=keep if k[1] == 'D' else ('logits', 'S'))
    wz = np.flatnonzero((y == 2) | (y == 3)); Jw = len(wz); ywz = y[wz]
    sdS = R0['S'][wz].std(0); sd4 = R0['h4'][wz].std(0); sd5 = R0['h5'][wz].std(0); sdm = m0[wz].std()
    live4 = [int(u) for u in np.flatnonzero(sd4 > 0)]; live5 = [int(u) for u in np.flatnonzero(sd5 > 0)]
    out['live'] = dict(S=LIVE_S, S_live_here=[int(u) for u in np.flatnonzero(sdS > 0)], h4=len(live4), h5=len(live5))
    out['margin_sd_WZ'] = float(sdm)
    # ---- invariants and per-edit descriptions
    Mi = moments(X['id']); out['edits'] = {}
    bt = BootW(J, 300 if smoke else 1000, seed + 1)
    for dc in DCS:
        a = acc[dc]; I = info[dc]
        e = dict(accepted={CL[c]: float(a[y == c].mean()) for c in range(5)}, reasons=I['reasons'],
                 c0_mean_WZ=float(np.nanmean(I['c0'][wz][a[wz]])), dcos_realised_mean_WZ=float(np.nanmean((I['c1'] - I['c0'])[wz][a[wz]])),
                 alpha_deg_median=float(np.degrees(np.nanmedian(np.abs(I['alpha'][wz][a[wz]])))),
                 disp_mean_WZ=float(np.nanmean(I['disp'][wz][a[wz]])), jet_p4_max_rel_change=float(I['dPjet'][a].max()) if a.any() else None)
        for kind in VAR:
            k = (dc, kind); sel = a & (accphi[dc] if kind == 'cphi' else True)
            Me = moments(X[k]); Le = RE[k]['logits']; ce = Le.argmax(1); dmg = margin(Le) - m0
            d = dict(n_WZ=int(sel[wz].sum()))
            for nm in ('M0', 'M2', 'M1'):
                rel = (Me[nm] - Mi[nm]) if nm == 'M1' else (Me[nm] / np.maximum(Mi[nm], 1e-12) - 1)
                d[f'd{nm}' + ('' if nm == 'M1' else '_rel') + '_mean'] = float(rel[wz][sel[wz]].mean()); d[f'd{nm}_absmax'] = float(np.abs(rel[wz][sel[wz]]).max())
            dm = minv[k] - minv['id']; d['dm_inv_GeV_mean'] = float(dm[wz][sel[wz]].mean()); d['dm_inv_GeV_absmax'] = float(np.abs(dm[wz][sel[wz]]).max())
            dS = (RE[k]['S'] - R0['S']) / np.where(sdS > 0, sdS, 1)
            d['dS_sd'] = {f'S{u}': dict(W=float(dS[(y == 2) & sel, u].mean()), Z=float(dS[(y == 3) & sel, u].mean()),
                                        W_ci=qci(bt.mean(dS[:, u], np.flatnonzero((y == 2) & sel)))) for u in LIVE_S}
            for c in range(5):
                s = np.flatnonzero((y == c) & sel)
                if not len(s): continue
                d[CL[c]] = dict(n=len(s), dmargin=float(dmg[s].mean()), dmargin_ci=qci(bt.mean(dmg, s)), changed=float((ce[s] != c0[s]).mean()),
                                W_to_Z=float(((c0[s] == 2) & (ce[s] == 3)).mean()), Z_to_W=float(((c0[s] == 3) & (ce[s] == 2)).mean()),
                                to_W=float(((c0[s] != 2) & (ce[s] == 2)).mean()), to_Z=float(((c0[s] != 3) & (ce[s] == 3)).mean()))
            e[kind] = d
        # D vs controls, paired (W+Z jets accepted for D, and cphi for that comparison)
        dD = margin(RE[(dc, 'D')]['logits']) - m0; chD = RE[(dc, 'D')]['logits'].argmax(1) != c0
        for kind in ('cn', 'cphi', 'R'):
            sel = a & (accphi[dc] if kind == 'cphi' else True); dK = margin(RE[(dc, kind)]['logits']) - m0; chK = RE[(dc, kind)]['logits'].argmax(1) != c0
            for c in (2, 3):
                s = np.flatnonzero((y == c) & sel); diff = dD - dK
                e[f'D_minus_{kind}_{CL[c]}'] = dict(mean=float(diff[s].mean()), ci=qci(bt.mean(diff, s)), p_signflip=signflip_p(diff[s], rng, 2000 if smoke else 5000),
                                                    mcnemar_changed=mcnemar(chD[s], chK[s]))
        out['edits'][str(dc)] = e
        log(f'  dc {dc:+.2f}: accepted W {e["accepted"]["W"]:.3f} Z {e["accepted"]["Z"]:.3f}; dcos {e["dcos_realised_mean_WZ"]:+.3f}; p4 max rel change {e["jet_p4_max_rel_change"]:.1e}; '
            + '; '.join(f'{kd}: W dmg {e[kd]["W"]["dmargin"]:+.3f} W->Z {100 * e[kd]["W"]["W_to_Z"]:.1f}% Z dmg {e[kd]["Z"]["dmargin"]:+.3f} Z->W {100 * e[kd]["Z"]["Z_to_W"]:.1f}% '
                     f'dM2 {e[kd]["dM2_rel_mean"]:+.4f} dm {e[kd]["dm_inv_GeV_mean"]:+.3f}' for kd in VAR))
        log(f'    D-cn W {e["D_minus_cn_W"]["mean"]:+.3f} {e["D_minus_cn_W"]["ci"]} p {e["D_minus_cn_W"]["p_signflip"]}; D-R W {e["D_minus_R_W"]["mean"]:+.3f}; '
            f'S1 W {e["D"]["dS_sd"]["S1"]["W"]:+.3f} S29 W {e["D"]["dS_sd"]["S29"]["W"]:+.3f} S30 {e["D"]["dS_sd"]["S30"]["W"]:+.3f} S19 {e["D"]["dS_sd"]["S19"]["W"]:+.3f} sd')

    # ---- route holds on W+Z jets
    A = {dc: acc[dc][wz] for dc in DCS}; btw = BootW(Jw, 300 if smoke else 1000, seed + 2)
    S0w = R0['S'][wz]; h40w = R0['h4'][wz]; h50w = R0['h5'][wz]; m0w = m0[wz]
    SE = np.concatenate([RE[(dc, 'D')]['S'][wz] for dc in DCS]); S0c = np.concatenate([S0w] * 4)
    h40c = np.concatenate([h40w] * 4); h50c = np.concatenate([h50w] * 4); m0c = np.concatenate([m0w] * 4)
    allS = np.concatenate([S0w, SE]); lo = np.zeros((3, 64), np.float32); hi = np.zeros((3, 64), np.float32)
    allh4 = np.concatenate([h40w] + [RE[(dc, 'D')]['h4'][wz] for dc in DCS]); allh5 = np.concatenate([h50w] + [RE[(dc, 'D')]['h5'][wz] for dc in DCS])
    lo[0], hi[0] = allS.min(0), allS.max(0); lo[1], hi[1] = allh4.min(0), allh4.max(0); lo[2, :32], hi[2, :32] = allh5.min(0), allh5.max(0)
    free = margin(tail(SE, S0c)) - m0c
    dfree_chk = float(np.abs(free - np.concatenate([margin(RE[(dc, 'D')]['logits'])[wz] - m0w for dc in DCS])).max()); log(f'  hold tail free check {dfree_chk:.2e}')
    split = lambda v: {dc: v[i * Jw:(i + 1) * Jw] for i, dc in enumerate(DCS)}
    F = split(free); CELLS = [(dc, c) for dc in DCS for c in (2, 3)]; CS = {(dc, c): np.flatnonzero(A[dc] & (ywz == c)) for dc, c in CELLS}
    sgn = {k: float(np.sign(F[k[0]][CS[k]].mean())) for k in CELLS}   # W and Z jets move in opposite directions: sign per (dc, class)

    def pooled(H):   # point estimate of the pooled response: sum over (dc, class) cells of sign x mean
        return sum(sgn[k] * H[k[0]][CS[k]].mean() for k in CELLS)
    Pfree = pooled(F)

    def share_pt(held): return 1 - pooled(split(held)) / Pfree

    def boot_pool(H): return sum(sgn[k] * btw.mean(H[k[0]], CS[k]) for k in CELLS)

    def boot_share(H): return 1 - boot_pool(H) / boot_pool(F)
    # mass directions at h4 / h5: mean standardised displacement under dil_M2 (angles x 1.05) on these W+Z jets
    xd = X['id'][wz].copy(); xd[..., 1:] *= np.float32(1.05); Rd = net.run(xd, keep=('h4', 'h5', 'logits', 'S'))
    s4 = np.where(sd4 > 0, sd4, 1); s5 = np.where(sd5 > 0, sd5, 1)
    d4 = ((Rd['h4'] - h40w) / s4).mean(0); d4[sd4 == 0] = 0; d4 /= np.linalg.norm(d4)
    d5 = ((Rd['h5'] - h50w) / s5).mean(0); d5[sd5 == 0] = 0; d5 /= np.linalg.norm(d5)
    M4dir = proj(d4[:, None], sd4); M5dir = proj(d5[:, None], sd5)
    HOLDS = {f'S{u}': dict(MS=proj(unitQ([u]), sdS)) for u in NAMED}
    HOLDS['named7'] = dict(MS=proj(unitQ(NAMED), sdS)); HOLDS['radial'] = dict(MS=proj(unitQ([1, 30, 19]), sdS))
    HOLDS['h4dir'] = dict(M4=M4dir); HOLDS['h5dir'] = dict(M5=M5dir)
    HOLDS['mass'] = dict(MS=proj(unitQ([29]), sdS), M4=M4dir); HOLDS['mass5'] = dict(MS=proj(unitQ([29]), sdS), M5=M5dir)
    def tlL(kw):   # range clip (X10 debug 2: the next quantizer wraps out-of-range values) only at the layers that are projected
        lo_, hi_ = np.full((3, 64), -1e9, np.float32), np.full((3, 64), 1e9, np.float32)
        for i, key in enumerate(('MS', 'M4', 'M5')):
            if kw.get(key) is not None and not (key == 'MS' and np.allclose(np.diag(np.diag(kw[key])), kw[key])): lo_[i], hi_[i] = lo[i], hi[i]
        return tail(SE, S0c, kw.get('MS'), h40c, kw.get('M4'), h50c, kw.get('M5'), lo_, hi_)
    tl = lambda kw: margin(tlL(kw)) - m0c
    c0c = np.concatenate([c0[wz]] * 4); accc = np.concatenate([A[dc] for dc in DCS]); chfree = (np.concatenate([RE[(dc, 'D')]['logits'][wz].argmax(1) for dc in DCS]) != c0c) & accc
    held, heldch = {}, {}
    for nm, kw in HOLDS.items():
        Lh = tlL(kw); held[nm] = margin(Lh) - m0c; heldch[nm] = (Lh.argmax(1) != c0c) & accc
    # g hold: full forward with g at its unedited value
    gh = []
    for dc in DCS:
        Lg = net.run_jetwise(X[(dc, 'D')][wz], {'g': (np.ones(64, np.float32), R0['g'][wz])})['logits']; gh.append(Lg)
    gh = np.concatenate(gh); held['g'] = margin(gh) - m0c; heldch['g'] = (gh.argmax(1) != c0c) & accc
    # nulls
    nnull = 100 if smoke else 1000; nr = np.random.default_rng(seed + 3); NULL = {}
    t = time.time()
    sets1 = [[u] for u in LIVE_S]; sets3 = [list(c) for c in itertools.combinations(LIVE_S, 3)]
    sets7 = [sorted(nr.choice(LIVE_S, 7, replace=False).tolist()) for _ in range(nnull)]
    if smoke: sets3 = [sets3[i] for i in nr.choice(len(sets3), nnull, replace=False)]
    NULL['sets1'] = [share_pt(tl(dict(MS=proj(unitQ(s), sdS)))) for s in sets1]
    NULL['sets3'] = [share_pt(tl(dict(MS=proj(unitQ(s), sdS)))) for s in sets3]
    NULL['sets7'] = [share_pt(tl(dict(MS=proj(unitQ(s), sdS)))) for s in sets7]
    for k in (1, 3, 7): NULL[f'dirs{k}'] = [share_pt(tl(dict(MS=proj(randQ(nr, LIVE_S, k), sdS)))) for _ in range(nnull)]
    NULL['h4dirs'] = [share_pt(tl(dict(M4=proj(randQ(nr, live4, 1), sd4)))) for _ in range(nnull)]
    NULL['h5dirs'] = [share_pt(tl(dict(M5=proj(randQ(nr, live5, 1, 32), sd5)))) for _ in range(nnull)]
    NULL['mass_null'] = [share_pt(tl(dict(MS=proj(unitQ([int(nr.choice(LIVE_S))]), sdS), M4=proj(randQ(nr, live4, 1), sd4)))) for _ in range(nnull)]
    log(f'  nulls in {time.time() - t:.0f}s: ' + ', '.join(f'{k} n {len(v)} q50 {np.median(v):+.3f} q95 {np.quantile(v, .95):+.3f}' for k, v in NULL.items()))
    NMAP = {**{f'S{u}': ('sets1', 'dirs1') for u in NAMED}, 'named7': ('sets7', 'dirs7'), 'radial': ('sets3', 'dirs3'),
            'h4dir': (None, 'h4dirs'), 'h5dir': (None, 'h5dirs'), 'mass': ('mass_null', None), 'mass5': (None, None), 'g': (None, None)}
    hr = {}
    for nm, v in held.items():
        H = split(v); sh = share_pt(v); ci = qci(boot_share(H))
        r = dict(share=float(sh), ci=ci, calls_changed_free=int(chfree.sum()), calls_changed_held=int(heldch[nm].sum()),
                 calls_removed=float(1 - heldch[nm].sum() / max(chfree.sum(), 1)), mcnemar_free_vs_held=mcnemar(chfree, heldch[nm]), per_dc={str(dc): dict((CL[c], float(1 - H[dc][A[dc] & (ywz == c)].mean() / F[dc][A[dc] & (ywz == c)].mean())) for c in (2, 3)) for dc in DCS},
                 held_dmargin={str(dc): dict((CL[c], float(H[dc][A[dc] & (ywz == c)].mean())) for c in (2, 3)) for dc in DCS})
        for lab_, key in zip(('p_sets', 'p_dirs'), NMAP[nm]):
            if key: nv = np.asarray(NULL[key]); r[lab_] = float((1 + (nv >= sh).sum()) / (1 + len(nv))); r[lab_ + '_null_q95'] = float(np.quantile(nv, .95)); r[lab_ + '_n'] = len(nv)
        r['p_primary'] = r.get('p_sets') if NMAP[nm][0] and len(NULL[NMAP[nm][0]]) >= 900 else r.get('p_dirs')
        if nm.startswith('S') and nm[1:].isdigit(): r['computes'] = DESC[int(nm[1:])]
        # calls with the hold (W->Z among W, Z->W among Z) at dc +0.3
        hr[nm] = r
    fam = [nm for nm in hr if hr[nm].get('p_primary') is not None]; ph = holm([hr[nm]['p_primary'] for nm in fam])
    for nm, p in zip(fam, ph): hr[nm]['p_holm'] = float(p)
    out['holds'] = dict(pooled_free=float(Pfree), sign_per_cell={f'{k[0]}|{CL[k[1]]}': v for k, v in sgn.items()}, free_ci=qci(boot_pool(F)),
                        results=hr, null_summary={k: dict(n=len(v), q05=float(np.quantile(v, .05)), q50=float(np.median(v)), q95=float(np.quantile(v, .95)), max=float(np.max(v))) for k, v in NULL.items()},
                        mass_dir=dict(h4_units_top=[int(u) for u in np.argsort(-np.abs(d4))[:6]], h5_units_top=[int(u) for u in np.argsort(-np.abs(d5))[:6]],
                                      dil_dmargin_W=float((margin(Rd['logits']) - m0w)[ywz == 2].mean()), dil_dmargin_Z=float((margin(Rd['logits']) - m0w)[ywz == 3].mean())),
                        clip_frac=float(np.mean((SE < lo[0]) | (SE > hi[0]))))
    log('  holds: ' + '; '.join(f'{nm} {r["share"]:+.3f} {np.round(r["ci"], 3).tolist()} p {r.get("p_primary")} holm {r.get("p_holm")}' for nm, r in hr.items()))

    # ---- calibration: probe of log m on h4, matched dilation
    rp = np.random.default_rng(seed + 4).permutation(J); fit, tst = rp[:J // 2], rp[J // 2:]
    lm = np.log(np.maximum(minv['id'], 1e-3)); H4z = (R0['h4'] - R0['h4'][fit].mean(0)) / np.where(R0['h4'][fit].std(0) > 0, R0['h4'][fit].std(0), 1)
    Xf = np.c_[np.ones(len(fit)), H4z[fit]]; lam = 1e-3 * len(fit) * np.diag(np.r_[0.0, np.ones(64)])
    w = np.linalg.solve(Xf.T @ Xf + lam + 1e-9 * np.eye(65), Xf.T @ lm[fit])
    pr = lambda h4: np.c_[np.ones(len(h4)), (h4 - R0['h4'][fit].mean(0)) / np.where(R0['h4'][fit].std(0) > 0, R0['h4'][fit].std(0), 1)] @ w
    r2 = 1 - ((pr(R0['h4'][tst]) - lm[tst]) ** 2).mean() / lm[tst].var()
    tw = np.intersect1d(tst, wz); gains = []
    for f_ in (1.05, 0.95):
        xx = X['id'][tw].copy(); xx[..., 1:] *= np.float32(f_); gains.append((pr(net.run(xx, keep=('h4',))['h4']) - pr(R0['h4'][tw])).mean() / math.log(f_))
    gain = float(np.mean(gains)); cal = dict(probe_r2_test=float(r2), probe_gain_per_log_dilation=gain, gains=gains, per_dc={})
    for dc, c in [(dc, c) for dc in DCS for c in (2, 3)]:
        s = tw[acc[dc][tw] & (y[tw] == c)]; dlm = pr(RE[(dc, 'D')]['h4'][s]) - pr(R0['h4'][s]); k_ = math.exp(dlm.mean() / gain)
        xx = X['id'][s].copy(); xx[..., 1:] *= np.float32(k_); Ld = net.run(xx)['logits']
        dD = margin(RE[(dc, 'D')]['logits'][s]) - m0[s]; dd = margin(Ld) - m0[s]
        bR = []; rb = np.random.default_rng(seed + 5)
        for _ in range(300 if smoke else 1000):
            i = rb.integers(0, len(s), len(s)); bR.append(dD[i].mean() / dd[i].mean() if abs(dd[i].mean()) > 0 else np.nan)
        cal['per_dc'][f'{dc}|{CL[c]}'] = dict(n=len(s), dlogm_probe_D=float(dlm.mean()), dlogm_true_D=float((np.log(np.maximum(minv[(dc, 'D')][s], 1e-3)) - lm[s]).mean()),
                                      k_matched=k_, dmargin_D=float(dD.mean()), dmargin_dil=float(dd.mean()), R=float(dD.mean() / dd.mean()) if dd.mean() != 0 else None,
                                      R_ci=qci(np.asarray(bR)[np.isfinite(bR)]), dil_changed=float((Ld.argmax(1) != c0[s]).mean()))
    out['calibration'] = cal; log('  calibration: ' + json.dumps(jsonable(cal)))

    # ---- program fidelity
    if full:
        prog = mask_program(net); P0 = prog(X['id']); out['mask_program_acc'] = float((P0.argmax(1) == y).mean())
        from interp import e11_verify as V11
        from jetdistill.observables import compute
        fa, fb = V11.formulas('e11'); last = V11.Last(net)
        def chain_logits(Xin, Rn):
            Q = {k: np.asarray(v, np.float32).astype(np.float64) for k, v in compute(Xin.astype(np.float64), 64).items()}
            ph_ = V11.physics(Xin, Q, False); full_ = last.jax_logits(V11.chain(fb, ph_)['z'])
            fromS = last.jax_logits(V11.chain(fb, ph_, start='g', true={'P': Rn['P'], 'g': Rn['g']})['z']); return full_, fromS
        t = time.time(); C0full, C0S = chain_logits(X['id'], R0); log(f'  chain on unedited in {time.time() - t:.0f}s')
        out['chain_acc'] = dict(full=float((C0full.argmax(1) == y).mean()), fromS=float((C0S.argmax(1) == y).mean()),
                                same_call_full=float((C0full.argmax(1) == c0).mean()), same_call_fromS=float((C0S.argmax(1) == c0).mean()))
        log(f'  programs on unedited: mask acc {out["mask_program_acc"]:.4f}; chain {out["chain_acc"]}')
        out['programs'] = {}
        for dc in DCS:
            k = (dc, 'D'); Ln = RE[k]['logits']; cn_ = Ln.argmax(1); dn = margin(Ln) - m0
            cf, cs_ = chain_logits(X[k], RE[k]); PL = dict(mask=(P0, prog(X[k])), chain=(C0full, cf), chainS=(C0S, cs_)); res = {}
            for nm, (Pa, Pb) in PL.items():
                ca, cb = Pa.argmax(1), Pb.argmax(1); dp = margin(Pb) - margin(Pa); r = {}
                for c in (2, 3):
                    s = np.flatnonzero((y == c) & acc[dc]); chN = cn_[s] != c0[s]; chP = cb[s] != ca[s]
                    r[CL[c]] = dict(n=len(s), net_changed=float(chN.mean()), prog_changed=float(chP.mean()), agree_changed=float((chN == chP).mean()),
                                    net_flips_reproduced=float(chP[chN].mean()) if chN.any() else None,
                                    net_flips_same_new_call=float((cb[s][chN] == cn_[s][chN]).mean()) if chN.any() else None,
                                    prog_flips_in_net=float(chN[chP].mean()) if chP.any() else None,
                                    mcnemar=mcnemar(chN, chP), dmargin_net=float(dn[s].mean()), dmargin_prog=float(dp[s].mean()),
                                    corr_dmargin=float(np.corrcoef(dn[s], dp[s])[0, 1]) if dp[s].std() > 0 else None,
                                    W_to_Z_prog=float(((ca[s] == 2) & (cb[s] == 3)).mean()), Z_to_W_prog=float(((ca[s] == 3) & (cb[s] == 2)).mean()))
                res[nm] = r
            out['programs'][str(dc)] = res
            log(f'  programs dc {dc:+.2f}: ' + '; '.join(f'{nm} W flips repro {r["W"]["net_flips_reproduced"]} dmg net {r["W"]["dmargin_net"]:+.3f} prog {r["W"]["dmargin_prog"]:+.3f} corr {r["W"]["corr_dmargin"]}' for nm, r in res.items()))

    # ---- example W jet for the figure
    pick = np.flatnonzero((y == 2) & acc[0.3] & (nn >= 15) & (nn <= 40) & (c0 == 2) & (RE[(0.3, 'D')]['logits'].argmax(1) == 3))
    if not len(pick): pick = np.flatnonzero((y == 2) & acc[0.3] & (nn >= 15) & (nn <= 40))
    if len(pick):
        j = int(pick[0]); n = int(nn[j]); P = (p4(x[j, :n, 0].astype(np.float64), x[j, :n, 1], x[j, :n, 2], 0.0)); lab = LAB[j, :n]
        r_, inf = edit_jet(P, lab, 0.3, np.random.default_rng([seed, j, 3]))
        axj = axis(P[None], np.array([n])); ex = dict(index=j, N=n, c0=inf['c0'], c1=inf['c1'], alpha_deg=float(np.degrees(inf['alpha'])), particles={})
        for nm, Q in (('unedited', P), ('D+0.3', r_['D']), ('cn+0.3', r_['cn'])):
            pt_, e_, f_, _ = ptetaphim(Q); ex['particles'][nm] = dict(pt=pt_.tolist(), deta=(e_ - axj[0][0]).tolist(), dphi=wrap(f_ - axj[1][0]).tolist(), prong=lab.tolist())
            T = Q.sum(0); ptT, eT, fT, mT = ptetaphim(T); sub = [Q[lab == k].sum(0) for k in (0, 1)]
            ex[nm] = dict(jet_pt=float(ptT), jet_m=float(mT), prong_pt=[float(np.hypot(v[0], v[1])) for v in sub], prong_m=[float(ptetaphim(v)[3]) for v in sub])
        sm_ = lambda l: (np.exp(l - l.max()) / np.exp(l - l.max()).sum())
        for nm, Lx in (('unedited', L0[j]), ('D+0.3', RE[(0.3, 'D')]['logits'][j]), ('cn+0.3', RE[(0.3, 'cn')]['logits'][j])):
            ex[nm]['call'] = CL[int(Lx.argmax())]; ex[nm]['p'] = sm_(Lx).tolist(); ex[nm]['margin_WZ'] = float(Lx[2] - Lx[3])
        out['example'] = ex; log('  example', {k: ex[k] for k in ('index', 'N', 'c0', 'c1')}, {nm: (ex[nm]['call'], round(ex[nm]['margin_WZ'], 3), round(ex[nm]['jet_m'], 2), np.round(ex[nm]['prong_pt'], 1).tolist()) for nm in ('unedited', 'D+0.3', 'cn+0.3')})
    perjet = dict(y=y, c0=c0, m0=m0, nn=nn, **{f'acc_{dc}': acc[dc] for dc in DCS},
                  **{f'dmargin_{dc}_{k}': margin(RE[(dc, k)]['logits']) - m0 for dc in DCS for k in VAR}, **{f'call_{dc}_{k}': RE[(dc, k)]['logits'].argmax(1) for dc in DCS for k in VAR})
    return out, X, perjet


def untrained_band(X, y, acc, seeds):
    from interp.jedi_core import Jedi
    out = {}
    wz = np.flatnonzero((y == 2) | (y == 3))
    for s in seeds:
        un = Jedi(64, '3-feature', untrained=True, seed=s); L0 = un.run(X['id'])['logits']; m0 = margin(L0); sd = m0[wz].std()
        r = dict(margin_sd=float(sd), constant_margin=bool(sd < 1e-6), call_dist=np.bincount(L0.argmax(1), minlength=5).tolist())
        for dc in DCS:
            for kind in ('D', 'cn'):
                d = margin(un.run(X[(dc, kind)])['logits']) - m0; sel = acc[dc]
                r[f'{dc}|{kind}|raw'] = {CL[c]: float(d[(y == c) & sel].mean()) for c in (2, 3)}
                r[f'{dc}|{kind}'] = {CL[c]: (float(d[(y == c) & sel].mean() / sd) if sd >= 1e-6 else None) for c in (2, 3)}
        out[str(s)] = r; log(f'  untrained seed {s}: ' + json.dumps(jsonable({k: v for k, v in r.items() if '0.3|' in k or k == 'margin_sd'})))
        del un
    return out


def singles(a):
    """supplement (--singles): per (dc, class) cell, the share of D's W-Z margin change removed by holding each of the 19 live S
    units and each of the 171 live-S pairs (main sample 3501); singles with 1000-resample jet-bootstrap CIs"""
    from interp.jedi_core import Jedi, sample
    sm = a.smoke; tag = 'e35_singles' + ('_smoke' if sm else ''); net = Jedi(64, '3-feature'); tail = Tail(net); log('network loaded')
    x, y, _ = sample(per_class=sm or 2000, seed=3501); x = np.asarray(x, np.float32)
    X, acc, _, _, _, _, _ = make_edits(x, 3501, sm)
    wz = np.flatnonzero((y == 2) | (y == 3)); ywz = y[wz]; Jw = len(wz)
    R0 = net.run(X['id'][wz], keep=('logits', 'S')); m0 = margin(R0['logits']); sdS = R0['S'].std(0)
    SE = {dc: net.run(X[(dc, 'D')][wz], keep=('S', 'logits')) for dc in DCS}
    free = {dc: margin(SE[dc]['logits']) - m0 for dc in DCS}
    cells = {f'{dc}|{CL[c]}': (dc, np.flatnonzero(acc[dc][wz] & (ywz == c))) for dc in DCS for c in (2, 3)}
    bt = BootW(Jw, 200 if sm else 1000, 3511); out = dict(N_WZ=Jw, free={k: float(free[dc][s].mean()) for k, (dc, s) in cells.items()}, singles={}, pairs={})
    def run_hold(units):
        MS = proj(unitQ(units), sdS); return {dc: margin(tail(SE[dc]['S'], R0['S'], MS)) - m0 for dc in DCS}
    for u in LIVE_S:
        H = run_hold([u]); r = {}
        for k, (dc, s) in cells.items():
            r[k] = dict(share=float(1 - H[dc][s].mean() / free[dc][s].mean()), ci=qci(1 - bt.mean(H[dc], s) / bt.mean(free[dc], s)), held=float(H[dc][s].mean()))
        out['singles'][f'S{u}'] = r
        log(f'  S{u}: ' + ', '.join(f'{k} {v["share"]:+.2f}' for k, v in r.items()))
    for p_ in itertools.combinations(LIVE_S, 2):
        H = run_hold(list(p_)); out['pairs'][f'S{p_[0]}+S{p_[1]}'] = {k: float(1 - H[dc][s].mean() / free[dc][s].mean()) for k, (dc, s) in cells.items()}
    for k in cells:
        top = sorted(out['pairs'], key=lambda q: -out['pairs'][q][k])[:5]; log(f'  top pairs {k}: ' + ', '.join(f'{q} {out["pairs"][q][k]:+.2f}' for q in top))
    (OUT / f'{tag}.json').write_text(json.dumps(jsonable(out), indent=1)); log(f'wrote {OUT / tag}.json; E35 singles DONE')


def main(a):
    from interp.jedi_core import Jedi
    if a.singles: return singles(a)
    sm = a.smoke; tag = 'e35_prong_sharing_route' + ('_smoke' if sm else ''); OUT.mkdir(parents=True, exist_ok=True)
    net = Jedi(64, '3-feature'); tail = Tail(net); log('network loaded', net.path)
    rng = np.random.default_rng(3503)
    res = dict(node='E35', smoke=sm, dcs=list(DCS), named=NAMED, named_desc={f'S{k}': v for k, v in DESC.items()}, live_S=LIVE_S)
    per = sm or 2000
    res['main'], X, pj = analyse(net, tail, 3501, per, sm, True, rng)
    np.savez_compressed(OUT / f'{tag}_perjet.npz', **pj); (OUT / f'{tag}.json').write_text(json.dumps(jsonable(res), indent=1)); log('main written')
    # untrained band (main sample)
    acc = {dc: pj[f'acc_{dc}'] for dc in DCS}
    tr = {}
    sd = res['main']['margin_sd_WZ']
    for dc in DCS:
        for kind in ('D', 'cn'):
            d = pj[f'dmargin_{dc}_{kind}'] / sd; tr[f'{dc}|{kind}'] = {CL[c]: float(d[(pj['y'] == c) & acc[dc]].mean()) for c in (2, 3)}
    res['untrained'] = dict(trained_sd_units=tr, redraws=untrained_band(X, pj['y'], acc, range(3 if sm else 10)))
    U = res['untrained']['redraws']
    ok = [s for s in U if not U[s]['constant_margin']]
    res['untrained']['n_redraws_nonconstant'] = len(ok)
    res['untrained']['band'] = {k: {c: [float(min(U[s][k][c] for s in ok)), float(max(U[s][k][c] for s in ok))] if ok else None for c in ('W', 'Z')} for k in tr}
    log('untrained band vs trained (SD units): ' + json.dumps(jsonable({k: (tr[k], res['untrained']['band'][k]) for k in tr})))
    (OUT / f'{tag}.json').write_text(json.dumps(jsonable(res), indent=1))
    del X
    res['rep'], _, pj2 = analyse(net, tail, 3502, per, sm, False, rng)
    np.savez_compressed(OUT / f'{tag}_perjet_rep.npz', **pj2)
    res['wall_s'] = time.time() - T0_
    (OUT / f'{tag}.json').write_text(json.dumps(jsonable(res), indent=1)); log(f'wrote {OUT / tag}.json; E35 DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', type=int, default=0); ap.add_argument('--singles', action='store_true'); main(ap.parse_args())
