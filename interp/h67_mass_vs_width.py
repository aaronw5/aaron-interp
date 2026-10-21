"""H67 (loop D; claims D25, D9; links A4, D4m): does each tagger read the jet's invariant mass m, or its pT-weighted angular width
w = sum_i pT_i dR_i / sum_i pT_i together with the jet pT, or both?  H64v could not separate these: a boost along the jet axis moves w
and pT together at fixed m, and an MLP(w, pT) without mass reproduced ParT_full's boost ratio.

Geometry (fixed per jet, from the unedited jet)
  E-scheme axis (ee, ep) = (pseudorapidity, azimuth) of the summed constituent three-momentum. Coordinates x_i = (eta_i - ee,
  wrap(phi_i - ep)). Prongs: exclusive-kT subjets (p = 1, E-scheme recombination; interp.h41_colour_flow.ktn_batch), 3 for jets of
  the top class, 2 for every other jet. Prong k: pT P_k, pT-weighted centroid c_k, particle radius rho_i = |x_i - c_k|; core = the
  particles with rho_i <= the pT-weighted median rho of the prong, edge = the rest. a_C,k / a_E,k = core / edge share of P_k.
Edit map apply(f, u) (every step keeps the particle count and the particle order; constituent masses scale with their pT so each
  particle keeps its direction):
  1 prong sharing f: every particle of prong k has its pT x (1 + f (P/K - P_k) / P_k) (K = number of prongs): sum pT and every prong's
    centroid unchanged; f = 1 gives all prongs equal pT, f < 0 makes the sharing more unequal (the hardest prong gains). For two prongs
    this changes z(1 - z) (z = softer prong's pT share) at fixed prong positions.
    [The plan's lever, moving pT between the core and the edge of each prong (core = particles inside the prong's pT-weighted median
    radius), was measured first (lever_diag in h67_edits_<ds>.json): it changes log m and log w by nearly the same amount, as the
    opening angle u does, so the two-target system is singular for most jets. In the small-angle limit m ~ pT Delta sqrt(z(1 - z))
    and w ~ 2 z(1 - z) Delta for two prongs at separation Delta, so the prong sharing moves log w twice as much as log m, and with u
    it separates the two. The core/edge lever is kept only as the diagnostic and in the crand control.]
  2 each prong is translated so that its pT-weighted centroid is back at c_k (a no-op for lever 1; needed after the crand shuffle).
  3 opening angle u: every prong is translated rigidly by (u - 1) c_k (prong centroids scaled by u about the jet axis).
  4 exact axis restore: all particles translated in (eta, phi) until the summed three-momentum points at (ee, ep) again (4 passes,
    residual < 1e-7 rad), then every four-vector x k so the jet pT is exactly the unedited jet pT.
  m, pT, eta, phi of the jet are then exact functions of (f, u); (f, u) are solved per jet by a damped 2-D Newton iteration with a
  finite-difference Jacobian for two targets (log m, log w).
Edits (all jets of the evaluated sample; a jet enters a comparison only if every edit in it converged, |residual| < 1e-5 in log):
  Ew+-10, Ew+-20   w x (1 +- 0.10 / 0.20) at fixed m (and exact jet pT, eta, phi)
  Em+-10           m x (1 +- 0.10) at fixed w (and exact jet pT, eta, phi)
  crot             each prong rotated about its centroid by a random angle U(0, 2 pi), then (f, u) solved for Delta m = Delta w = 0
                   (N, m, w, jet pT fixed; changes only the orientation of each prong's particles)
  crand_free       random pT redistribution inside each prong (factor 1 + s eps_i, eps_i ~ N(0, 1) centred so P_k is unchanged; s set so
                   the total moved pT equals that of Ew-10 on the same jet), prong recentred, then u alone solved for Delta m = 0 (w free)
  crand_match      the same random redistribution, then (f, u) solved for Delta m = 0 and Delta w = -10 % (same targets as Ew-10)
  TN               all particles translated about the stored axis by the pT-weighted displacement of Ew-10 on that jet, random
                   direction (interp.h12_boost_panel.translate; inputs move relative to the stored axis, the jet itself is unchanged)
  For every edit: Delta log m, Delta log w, Delta log jet pT, Delta sum pT, Delta m_A (m_A = sqrt(M0 M2) about the stored axis, the
  JEDI-linear mass variable, A4), Delta D2 (beta 1), Delta tau21 (beta 1, kT WTA axes), moved-pT share, displacement.
Response: own-class one-vs-rest log-odds l_k; Delta s = l_k(edited) - l_k(unedited) per jet of true class k.
  rho = sum |Delta s(Ew)| / sum |Delta s(Em)| over the jets of class k at matched fractional size: pair 'uneq' = (Ew-10, Em+10), both reached
  by more unequal prong sharing and wider prongs; pair 'eq' = (Ew+10, Em-10), both reached by more equal sharing and narrower prongs
  (z(1 - z) <= 1/4 limits how many jets reach them); a jet enters a pair only if both edits and crot converged on it;
  phi = sum|Delta s(Ew)| / (sum|Delta s(Ew)| + sum|Delta s(Em)|) in [0, 1] (0: responds only to the mass edit; 1: only to the width
  edit); 95 % bootstrap over jets (1,000 resamples).
References (trained in --part summary on disjoint reference jets of the same dataset): sklearn MLP (32, 32) on [log m, log pT],
  [log w, log pT], [log m, log w, log pT], [log m_A, log pT]; HistGradientBoosting on 32 energy-flow polynomials (interp.b4_baselines_jedi.efps).
Nulls: 6 weight-permuted copies of JEDI-linear (hls4ml) and of ParT (JetClass: 3 ParT_full + 3 ParT_kin; Top: 6 ParT).
Samples: hls4ml jedi_core.sample(2000, 6701) (same draw) + 3,000 disjoint rows per class (rng 6711) as reference; JetClass 5,000 random
  rows per class of Wqq, Zqq, Tbqq, QCD (first test file, rng 6702), split at random (rng 6712) into 2,000 evaluated + 3,000 reference;
  Top Landscape test 5,000 + 5,000 random rows (rng 6703) + 5,000 + 5,000 disjoint reference rows (rng 6713).
Run
  AI_PY=conda ./job.sh d-h67-ed-hls 'python -m interp.h67_mass_vs_width --part edits --ds hls'        (likewise jetclass, top)
  ./job.sh d-h67-ev-jedi 'python -m interp.h67_mass_vs_width --part eval --ds hls_jedi'
  AI_PY=conda ./job.sh d-h67-ev-small 'python -m interp.h67_mass_vs_width --part eval --ds hls_small'
  ./part_job.sh d-h67-ev-gpu 'python -m interp.h67_mass_vs_width --part eval --ds jetclass && python -m interp.h67_mass_vs_width --part eval --ds top'
  AI_PY=conda ./job.sh d-h67-sum 'python -m interp.h67_mass_vs_width --part summary'
  -> $AI_RESULTS/physics/h67_work/*.npz|pkl, h67_edits_<ds>.json, h67_summary.json, h67_jet.png, h67_phi.png
"""
import argparse
import copy
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault('NUMBA_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h12_boost_panel as H  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
OUT = RES / 'physics'
WORK = OUT / 'h67_work'
V = Path('/j-jepa-vol/aaron-interp')
T00 = time.time()
SEEDS = dict(hls=6701, jetclass=6702, top=6703)
NB = 1000
EDITS = ['Ew+10', 'Ew-10', 'Ew+20', 'Ew-20', 'Em+10', 'Em-10', 'crot', 'crand_free', 'crand_match', 'TN']
TARGETS = {'Ew+10': (0.0, math.log(1.1)), 'Ew-10': (0.0, math.log(0.9)), 'Ew+20': (0.0, math.log(1.2)), 'Ew-20': (0.0, math.log(0.8)),
           'Em+10': (math.log(1.1), 0.0), 'Em-10': (math.log(0.9), 0.0), 'crot': (0.0, 0.0), 'crand_match': (0.0, math.log(0.9))}
TOL = 1e-5
# matched-size pairs of a width edit and a mass edit, both 10 %: 'uneq' = the pair reached by making the prong pT sharing more unequal
# and opening the prongs (Ew-10 with Em+10); 'eq' = the pair reached by making the sharing more equal and closing the prongs (Ew+10, Em-10)
PAIRS = {'uneq': ('Ew-10', 'Em+10'), 'eq': ('Ew+10', 'Em-10')}
JC_CL = ['Wqq', 'Zqq', 'Tbqq', 'QCD']


def log(*a):
    print(f'[{time.time() - T00:7.0f}s]', *a, flush=True)


# ============================================================================================================== kinematics
def jet3(pt, eta, phi, m):
    real = pt > 0
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    E = np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)
    S = [np.where(real, a, 0).sum(1) for a in (px, py, pz, E)]
    jpt = np.hypot(S[0], S[1]); jeta = np.arcsinh(S[2] / np.maximum(jpt, 1e-300)); jphi = np.arctan2(S[1], S[0])
    mj = np.sqrt(np.maximum(S[3] ** 2 - S[0] ** 2 - S[1] ** 2 - S[2] ** 2, 0))
    return jpt, jeta, jphi, mj


def width(pt, eta, phi, ee, ep):
    real = pt > 0; de = eta - ee[:, None]; dp = H.wrap(phi - ep[:, None])
    r = np.where(real, np.hypot(de, dp), 0)
    return (pt * r).sum(1) / np.maximum(pt.sum(1), 1e-300)


def mA_stored(T):
    de, dp = H.rel(T); M0 = T['pt'].sum(1); M2 = (T['pt'] * (de ** 2 + dp ** 2)).sum(1); return np.sqrt(M0 * M2)


def F4(T):
    return np.ascontiguousarray(np.stack(H.four(T), -1).astype(np.float64))


def obs(T, ee, ep):
    jpt, jeta, jphi, m = jet3(T['pt'], T['eta'], T['phi'], T['m'])
    return dict(m=m, w=width(T['pt'], T['eta'], T['phi'], ee, ep), jpt=jpt, jeta=jeta, jphi=jphi, sumpt=T['pt'].sum(1), mA=mA_stored(T),
                N=(T['pt'] > 0).sum(1).astype(float))


def shapes(T):
    from interp.h45_prong_sharing import tau21_batch, d2_batch
    F = F4(T); nn = (T['pt'] > 0).sum(1).astype(np.int64)
    # tables keep the unedited particle order (real particles first, see geometry()); padded slots are zero
    return dict(D2=d2_batch(F, nn), tau21=tau21_batch(F, nn))


# ============================================================================================================== geometry
def geometry(T, ntop):
    """T: pT-sorted table (real particles first). ntop: (J,) bool, True = 3 prongs. Returns geometry dict G"""
    from interp.h41_colour_flow import ktn_batch
    pt = T['pt']; real = pt > 0; J, L = pt.shape; nn = real.sum(1).astype(np.int64)
    assert np.all(real == (np.arange(L)[None, :] < nn[:, None])), 'real particles must come first'
    jpt, ee, ep, m0 = jet3(T['pt'], T['eta'], T['phi'], T['m'])
    F = F4(T); lab = -np.ones((J, L), np.int64)
    for K, sel in ((3, ntop), (2, ~ntop)):
        ii = np.flatnonzero(sel)
        if len(ii): lab[ii] = ktn_batch(np.ascontiguousarray(F[ii]), nn[ii], K)
    lab = np.where(real & (lab < 0), 0, lab); lab = np.where(real, lab, -1); K = 3
    x = np.stack([np.where(real, T['eta'] - ee[:, None], 0), np.where(real, H.wrap(T['phi'] - ep[:, None]), 0)], -1)
    G = dict(lab=lab, real=real, x0=x, ee=ee, ep=ep, jpt0=jpt, m0=m0, ntop=ntop, K=K)
    G['c'] = prong_centroid(pt, x, lab, K)
    rho = np.hypot(*(x - gather(G['c'], lab)).transpose(2, 0, 1)); rho = np.where(real, rho, np.inf)
    core = np.zeros((J, L), bool)
    for k in range(K):
        mk = lab == k; rk = np.where(mk, rho, np.inf); o = np.argsort(rk, 1, kind='stable')
        ws = np.take_along_axis(np.where(mk, pt, 0), o, 1); cs = np.cumsum(ws, 1); P = cs[:, -1:]
        first = np.argmax(cs >= 0.5 * P - 1e-12, 1); rmed = np.take_along_axis(np.take_along_axis(rk, o, 1), first[:, None], 1)
        core |= mk & (rk <= rmed)
    G['core'] = core; G['rho'] = np.where(real, rho, 0)
    return G


def gather(A, lab):
    """A: (J, K, ...) per-prong values -> per-particle values (J, L, ...) (padding gets prong 0)"""
    idx = np.clip(lab, 0, None)
    if A.ndim == 2: return np.take_along_axis(A, idx, 1)
    return np.take_along_axis(A, idx[..., None].repeat(A.shape[2], 2), 1)


def prong_sum(v, lab, K):
    return np.stack([np.where(lab == k, v, 0).sum(1) for k in range(K)], 1)


def prong_centroid(pt, x, lab, K):
    P = prong_sum(pt, lab, K)
    c = np.stack([prong_sum(pt * x[..., d], lab, K) for d in (0, 1)], -1)
    return c / np.maximum(P, 1e-300)[..., None]


def lever_factors(G, pt0, f, lever):
    """per-particle pT factors. lever 'z' (used for every edit): prong k's particles x (1 + f (P / K - P_k) / P_k), K = number of
    prongs with pT > 0: f = 1 gives every prong the same pT, f < 0 makes the sharing more unequal; sum pT and every prong's centroid
    unchanged. lever 'core' (diagnostic only): core particles x (1 - f a_E,k), edge particles x (1 + f a_C,k)."""
    real, lab, K = G['real'], G['lab'], G['K']; P = prong_sum(pt0, lab, K)
    if lever == 'z':
        Kj = (P > 0).sum(1); Pt = P.sum(1); tgt = (Pt / np.maximum(Kj, 1))[:, None]
        fac = np.where(P > 0, 1 + f[:, None] * (tgt - P) / np.maximum(P, 1e-300), 1.0)
        return np.where(real, gather(fac, lab), 0)
    PC = prong_sum(np.where(G['core'], pt0, 0), lab, K)
    aC = np.where(P > 0, PC / np.maximum(P, 1e-300), 1.0); aE = 1 - aC
    return np.where(G['core'], 1 - f[:, None] * gather(aE, lab), 1 + f[:, None] * gather(aC, lab))


def apply(T0, G, f, u, base_pt=None, base_x=None, lever='z'):
    """the edit map (module docstring). f, u: (J,). base_pt / base_x: modified starting pT / coordinates (controls)."""
    real, lab, K = G['real'], G['lab'], G['K']
    pt0 = T0['pt'] if base_pt is None else base_pt; x = (G['x0'] if base_x is None else base_x).copy()
    mfac0 = np.where(real, pt0 / np.maximum(T0['pt'], 1e-300), 0)
    g = lever_factors(G, pt0, f, lever)
    pt = np.where(real, pt0 * g, 0); mm = np.where(real, T0['m'] * mfac0 * g, 0)
    cnew = prong_centroid(pt, x, lab, K); x = x - gather(cnew - G['c'], lab)
    x = x + (u[:, None, None] - 1) * gather(G['c'], lab)
    eta = G['ee'][:, None] + x[..., 0]; phi = G['ep'][:, None] + x[..., 1]
    for _ in range(4):
        _, je, jp, _ = jet3(pt, eta, phi, mm)
        eta = eta + (G['ee'] - je)[:, None]; phi = phi + H.wrap(G['ep'] - jp)[:, None]
    jpt, _, _, _ = jet3(pt, eta, phi, mm); kk = G['jpt0'] / np.maximum(jpt, 1e-300)
    pt = pt * kk[:, None]; mm = mm * kk[:, None]
    U = dict(pt=np.where(real, pt, 0), eta=np.where(real, eta, 0), phi=np.where(real, H.wrap(phi), 0), m=np.where(real, mm, 0),
             ax_eta=T0['ax_eta'], ax_phi=T0['ax_phi'], ext=T0['ext'])
    return U


def lmw(U, G):
    _, _, _, m = jet3(U['pt'], U['eta'], U['phi'], U['m']); w = width(U['pt'], U['eta'], U['phi'], G['ee'], G['ep'])
    return np.log(np.maximum(m, 1e-12)), np.log(np.maximum(w, 1e-12))


def f_bounds(G, base_pt, T0):
    """z lever: every prong keeps >= 5 % of its pT; f <= 1 (equal sharing)"""
    pt0 = T0['pt'] if base_pt is None else base_pt; lab, K = G['lab'], G['K']
    P = prong_sum(pt0, lab, K); Kj = (P > 0).sum(1); tgt = (P.sum(1) / np.maximum(Kj, 1))[:, None]; d = tgt - P
    lo = np.where((P > 0) & (d > 0), -0.95 * P / np.maximum(d, 1e-300), -np.inf).max(1)
    hi = np.minimum(np.where((P > 0) & (d < 0), 0.95 * P / np.maximum(-d, 1e-300), np.inf).min(1), 1.0)
    return np.maximum(np.where(np.isfinite(lo), lo, -5.0), -5.0), hi


def solve(T0, G, am, aw, base_pt=None, base_x=None, u_only=False, iters=60):
    """per-jet (f, u) such that log m - log m0 = am and log w - log w0 = aw (u_only: f = 0, only the m target)"""
    J = len(G['ee']); lm0, lw0 = np.log(G['m0']), np.log(G['w0'])
    f = np.zeros(J); lu = np.zeros(J); fl, fh = f_bounds(G, base_pt, T0); h = 1e-4
    res = lambda f_, lu_: (lambda a: np.stack([a[0] - lm0 - am, a[1] - lw0 - aw], 1))(lmw(apply(T0, G, f_, np.exp(lu_), base_pt, base_x), G))
    for it in range(iters):
        r = res(f, lu)
        if u_only:
            r1 = res(f, lu + h); d = (r1[:, 0] - r[:, 0]) / h
            step = -r[:, 0] / np.where(np.abs(d) > 1e-8, d, 1e-8); lu = lu + np.clip(step, -0.1, 0.1)
        else:
            rf = res(f + h, lu); ru = res(f, lu + h)
            Jm = np.stack([(rf - r) / h, (ru - r) / h], 2)            # (J, 2 residuals, 2 params)
            det = Jm[:, 0, 0] * Jm[:, 1, 1] - Jm[:, 0, 1] * Jm[:, 1, 0]; det = np.where(np.abs(det) > 1e-10, det, 1e-10)
            df = -(Jm[:, 1, 1] * r[:, 0] - Jm[:, 0, 1] * r[:, 1]) / det
            du = -(-Jm[:, 1, 0] * r[:, 0] + Jm[:, 0, 0] * r[:, 1]) / det
            sc = np.minimum(1, np.minimum(0.1 / np.maximum(np.abs(df), 1e-12), 0.1 / np.maximum(np.abs(du), 1e-12)))
            f = np.clip(f + sc * df, fl, fh); lu = np.clip(lu + sc * du, math.log(0.5), math.log(2.0))
    r = res(f, lu); conv = np.abs(r[:, 0]) < TOL if u_only else np.max(np.abs(r), 1) < TOL
    U = apply(T0, G, f, np.exp(lu), base_pt, base_x)
    return U, dict(f=f, u=np.exp(lu), conv=conv, resid=np.max(np.abs(r), 1) if not u_only else np.abs(r[:, 0]))


def displacement(G, U, pt0):
    """pT-weighted (unedited pT) mean distance each particle moved in the (eta, phi) plane"""
    x1 = np.stack([U['eta'] - G['ee'][:, None], H.wrap(U['phi'] - G['ep'][:, None])], -1)
    d = np.where(G['real'], np.hypot(*(x1 - G['x0']).transpose(2, 0, 1)), 0)
    return (d * pt0).sum(1) / np.maximum(pt0.sum(1), 1e-300)


def moved_share(T0, U):
    return np.abs(U['pt'] - T0['pt']).sum(1) / np.maximum(T0['pt'].sum(1), 1e-300) / 2


# ============================================================================================================== edits step
def make_edits(T0, y, ntop, seed):
    """returns {edit: (table, info)} and G"""
    G = geometry(T0, ntop); G['w0'] = width(T0['pt'], T0['eta'], T0['phi'], G['ee'], G['ep'])
    U = apply(T0, G, np.zeros(len(y)), np.ones(len(y)))
    rt = max(float(np.abs(U[k] - T0[k]).max()) for k in ('pt', 'eta', 'm')); rp = float(np.abs(H.wrap(U['phi'] - T0['phi'])).max())
    log(f'round trip apply(f=0, u=1): max |diff| pt/eta/m {rt:.2e}, phi {rp:.2e}; prongs: 3 for {int(ntop.sum())} jets, 2 for {int((~ntop).sum())}')
    out = {}; rng = np.random.default_rng(seed + 1000)
    for e in ('Ew+10', 'Ew-10', 'Ew+20', 'Ew-20', 'Em+10', 'Em-10'):
        t = time.time(); U, inf = solve(T0, G, *TARGETS[e]); out[e] = (U, inf)
        log(f'{e}: converged {inf["conv"].mean():.3f}; f median {np.median(inf["f"][inf["conv"]]) if inf["conv"].any() else None}; '
            f'u median {np.median(inf["u"][inf["conv"]]) if inf["conv"].any() else None} ({time.time() - t:.0f} s)')
    # crot: rotate each prong about its centroid by a random angle, then restore m and w exactly
    th = rng.uniform(0, 2 * np.pi, (len(y), G['K'])); tp = gather(th, G['lab']); c = gather(G['c'], G['lab']); d = G['x0'] - c
    xr = np.where(G['real'][..., None], c + np.stack([np.cos(tp) * d[..., 0] - np.sin(tp) * d[..., 1], np.sin(tp) * d[..., 0] + np.cos(tp) * d[..., 1]], -1), 0)
    U, inf = solve(T0, G, 0.0, 0.0, base_x=xr); out['crot'] = (U, inf); log(f'crot: converged {inf["conv"].mean():.3f}')
    # crand: random pT redistribution inside each prong with the moved pT of Ew-10
    ref_U, ref_inf = out['Ew-10']; tgt = moved_share(T0, ref_U); tgt = np.where(ref_inf['conv'], tgt, np.median(tgt[ref_inf['conv']]))
    eps = rng.normal(size=T0['pt'].shape) * G['real']; lab, K = G['lab'], G['K']; pt = T0['pt']
    eps = eps - gather(prong_sum(pt * eps, lab, K) / np.maximum(prong_sum(pt, lab, K), 1e-300), lab); eps = np.where(G['real'], eps, 0)
    s = 2 * tgt * pt.sum(1) / np.maximum((pt * np.abs(eps)).sum(1), 1e-300)
    bp = pt * np.maximum(1 + s[:, None] * eps, 0.05); bp = np.where(G['real'], bp, 0)
    bp = bp * gather(prong_sum(pt, lab, K) / np.maximum(prong_sum(bp, lab, K), 1e-300), lab)
    U, inf = solve(T0, G, 0.0, 0.0, base_pt=bp, u_only=True); out['crand_free'] = (U, inf)
    log(f'crand_free: converged {inf["conv"].mean():.3f}, moved share median {np.median(moved_share(T0, U)):.4f} vs Ew-10 {np.median(tgt):.4f}')
    U, inf = solve(T0, G, *TARGETS['crand_match'], base_pt=bp); out['crand_match'] = (U, inf); log(f'crand_match: converged {inf["conv"].mean():.3f}')
    D = displacement(G, ref_U, T0['pt']); D = np.where(ref_inf['conv'], D, np.median(D[ref_inf['conv']]))
    U, _ = H.translate(T0, D, rng); out['TN'] = (U, dict(conv=np.ones(len(y), bool), f=np.zeros(len(y)), u=np.ones(len(y)), resid=np.zeros(len(y))))
    return out, G


def lever_diag(T0, G, y, classes):
    """median (Delta log m, Delta log w) per class for each lever at fixed u = 1: core/edge f = +-0.3, sharing f = -0.3 / +0.3, and the
    opening angle u = 1.05 at f = 0 (shows why the core/edge lever cannot separate m from w)"""
    J = len(y); z, o = np.zeros(J), np.ones(J); a0 = lmw(apply(T0, G, z, o), G); out = {}
    for nm, (f, u, lv) in {'core+0.3': (z + 0.3, o, 'core'), 'core-0.3': (z - 0.3, o, 'core'), 'z+0.3': (z + 0.3, o, 'z'), 'z-0.3': (z - 0.3, o, 'z'),
                           'u1.05': (z, o * 1.05, 'z')}.items():
        a = lmw(apply(T0, G, f, u, lever=lv), G); dm, dw = a[0] - a0[0], a[1] - a0[1]
        out[nm] = {}
        for k in sorted(set(y.tolist())):
            sel = y == k; ok = sel & (np.abs(dm) > 1e-9)
            out[nm][classes[k]] = dict(dlogm=float(np.median(dm[sel])), dlogw=float(np.median(dw[sel])),
                                       ratio_w_over_m=float(np.median(dw[ok] / dm[ok])) if ok.any() else None)
        log(f'lever {nm}: ' + ' '.join(f'{c}: dlogm {v["dlogm"]:+.4f} dlogw {v["dlogw"]:+.4f}' for c, v in out[nm].items()))
    return out


def edit_stats(T0, G, out, y, classes):
    O0 = obs(T0, G['ee'], G['ep']); S0 = shapes(T0); res = dict(unedited={}, edits={})
    for k, v in {**O0, **S0}.items(): res['unedited'][k] = dict(median=float(np.median(v)), q5_95=np.percentile(v, [5, 95]).tolist())
    per = dict(m0=O0['m'], w0=O0['w'], jpt0=O0['jpt'], mA0=O0['mA'], D20=S0['D2'], tau210=S0['tau21'])
    for e, (U, inf) in out.items():
        t = time.time(); S1 = shapes(U); c = inf['conv']
        if e == 'TN': _, ue, up, _ = jet3(U['pt'], U['eta'], U['phi'], U['m']); O1 = obs(U, ue, up)    # TN moves the jet: w about its own axis
        else: O1 = obs(U, G['ee'], G['ep'])
        rel = lambda a, b: a / np.maximum(b, 1e-12) - 1
        dd = dict(m=rel(O1['m'], O0['m']), w=rel(O1['w'], O0['w']), jpt=rel(O1['jpt'], O0['jpt']), sumpt=rel(O1['sumpt'], O0['sumpt']),
                  mA=rel(O1['mA'], O0['mA']), D2=S1['D2'] - S0['D2'], tau21=S1['tau21'] - S0['tau21'],
                  jeta=O1['jeta'] - O0['jeta'], jphi=H.wrap(O1['jphi'] - O0['jphi']))
        e_res = dict(conv_frac=float(c.mean()), conv_by_class={classes[k]: float(c[y == k].mean()) for k in range(len(classes)) if (y == k).any()},
                     moved_pt_share=float(np.median(moved_share(T0, U)[c])) if c.any() else None,
                     displacement_median=float(np.median(displacement(G, U, T0['pt'])[c])) if c.any() else None,
                     f_median=float(np.median(inf['f'][c])) if c.any() else None, u_median=float(np.median(inf['u'][c])) if c.any() else None)
        e_res['change'] = {}
        for q, v in dd.items():
            st = {}
            for k in range(len(classes)):
                sel = c & (y == k)
                if sel.any(): st[classes[k]] = dict(median=float(np.median(v[sel])), absmax=float(np.max(np.abs(v[sel]))), mean=float(np.mean(v[sel])))
            e_res['change'][q] = st
        res['edits'][e] = e_res
        for q in ('m', 'w', 'jpt', 'mA', 'D2', 'tau21'): per[f'{e}__d{q}'] = dd[q].astype(np.float32)
        per[f'{e}__conv'] = c
        log(f'{e}: conv {c.mean():.3f} | median dm {np.median(dd["m"][c]):+.2e} dw {np.median(dd["w"][c]):+.4f} djpt max {np.abs(dd["jpt"][c]).max():.1e} '
            f'deta max {np.abs(dd["jeta"][c]).max():.1e} dmA {np.median(dd["mA"][c]):+.4f} dD2 {np.median(dd["D2"][c]):+.3f} dtau21 {np.median(dd["tau21"][c]):+.4f} '
            f'moved {e_res["moved_pt_share"]} disp {e_res["displacement_median"]} ({time.time() - t:.0f} s)')
    return res, per


def ref_feats(T):
    """features for the reference classifiers: log m, log w (E-scheme axis), log jet pT, log m_A, 32 EFPs"""
    from interp.b4_baselines_jedi import efps
    jpt, ee, ep, m = jet3(T['pt'], T['eta'], T['phi'], T['m']); w = width(T['pt'], T['eta'], T['phi'], ee, ep)
    real = T['pt'] > 0; nmax = int(real.sum(1).max()); o = np.argsort(-T['pt'], 1, kind='stable')[:, :nmax]
    tk = lambda a: np.take_along_axis(a, o, 1)
    x = np.stack([tk(T['pt']), tk(np.where(real, T['eta'] - ee[:, None], 0)), tk(np.where(real, H.wrap(T['phi'] - ep[:, None]), 0))], -1)
    return dict(m=m, w=w, jpt=jpt, mA=mA_stored(T), efp=efps(x, chunk=250))


def save_tables(ds, T0, y, out, extra=None):
    WORK.mkdir(parents=True, exist_ok=True); D = dict(y=y.astype(np.int16))
    for nm, T in [('U', T0)] + [(e, out[e][0]) for e in EDITS]:
        for k in ('pt', 'eta', 'phi', 'm'): D[f'{nm}__{k}'] = T[k].astype(np.float64 if k in ('eta', 'phi') else np.float32)
        D[f'{nm}__ax_eta'] = T['ax_eta']; D[f'{nm}__ax_phi'] = T['ax_phi']
    np.savez_compressed(WORK / f'{ds}_tables.npz', **D)
    if extra is not None:
        with open(WORK / f'{ds}_extra.pkl', 'wb') as fh: pickle.dump(extra, fh, protocol=4)
    log(f'wrote {WORK / f"{ds}_tables.npz"}')


def load_tables(ds):
    d = np.load(WORK / f'{ds}_tables.npz'); y = d['y'].astype(int); ext = {}
    extra = None
    if (WORK / f'{ds}_extra.pkl').exists():
        with open(WORK / f'{ds}_extra.pkl', 'rb') as fh: extra = pickle.load(fh)
        ext = extra.get('ext', {})
    Ts = {}
    for nm in ['U'] + EDITS:
        Ts[nm] = dict(pt=d[f'{nm}__pt'].astype(np.float64), eta=d[f'{nm}__eta'], phi=d[f'{nm}__phi'], m=d[f'{nm}__m'].astype(np.float64),
                      ax_eta=d[f'{nm}__ax_eta'], ax_phi=d[f'{nm}__ax_phi'], ext=ext)
    # jets on which the solve diverged (excluded from every comparison) get their unedited particles, so no non-finite or far
    # out-of-range input reaches a network
    U = Ts['U']
    for nm in EDITS:
        T = Ts[nm]; real = T['pt'] > 0
        bad = ~np.all(np.isfinite(T['pt']) & np.isfinite(T['eta']) & np.isfinite(T['phi']) & np.isfinite(T['m']), 1)
        bad |= (np.where(real, np.abs(T['eta'] - T['ax_eta'][:, None]), 0).max(1) > 3) | (np.where(real, T['pt'], 0).max(1) > 5 * U['pt'].max(1))
        for k in ('pt', 'eta', 'phi', 'm'): T[k] = np.where(bad[:, None], U[k], T[k])
        if bad.any(): log(f'{nm}: {int(bad.sum())} diverged jets replaced by the unedited jet')
    return Ts, y, extra


def sort_first(T):
    """pT-sort so that real particles come first (the edit geometry needs it); ext sorted alongside"""
    return H.sortT(T)


# ---------------------------------------------------------------------------------------------- dataset samples
def hls_sample(seed, per=2000, per_ref=3000):
    from jetdistill.config import RESULTS
    from interp.h1_irc_panel import hls_table
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); yall = np.asarray(d['y'])
    rng = np.random.default_rng(seed)    # identical draw to interp.jedi_core.sample(per_class=per, seed=seed)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(yall == c), per, replace=False) for c in range(5)]))
    r2 = np.random.default_rng(seed + 10); used = np.zeros(len(yall), bool); used[idx] = True
    ref = np.sort(np.concatenate([r2.choice(np.flatnonzero((yall == c) & ~used), per_ref, replace=False) for c in range(5)]))
    x = np.asarray(d['x'], np.float32)
    mk = lambda ii: {k: v for k, v in hls_table(x[ii]).items() if k != 'sgn'}
    return mk(idx), yall[idx].astype(int), mk(ref), yall[ref].astype(int), idx


def jc_sample4(per, seed):
    import uproot
    from interp.part_core import read_root, fof, cat, sub
    rng = np.random.default_rng(seed); Js = []
    for c in JC_CL:
        f = fof(c); n = uproot.open(f)['tree'].num_entries; idx = np.sort(rng.choice(n, per, replace=False)); parts = []
        for a in range(0, n, 20000):
            loc = idx[(idx >= a) & (idx < a + 20000)] - a
            if len(loc): parts.append(sub(read_root(f, start=a, stop=min(a + 20000, n)), loc))
        Js.append(cat(parts)); log(f'JetClass sample {c}: {per} random of {n} rows of {Path(f).name}')
    return cat(Js)


def part_edits(ds):
    seed = SEEDS[ds]
    if ds == 'hls':
        T0, y, Tp, yp, idx = hls_sample(seed); classes = H.HLS_CL; ntop = y == 4; extra = None
        info = f'interp.jedi_core.sample(per_class=2000, seed={seed}); reference 3,000 per class (rng {seed + 10})'
    elif ds == 'jetclass':
        from interp.h1_irc_panel import jc_table
        from interp.part_core import sub, CL
        Jall = jc_sample4(5000, seed); yall = Jall['label'].astype(int); r = np.random.default_rng(seed + 10); ev, rf = [], []
        for c in JC_CL:
            ii = r.permutation(np.flatnonzero(yall == CL.index(c))); ev.append(ii[:2000]); rf.append(ii[2000:])
        ev, rf = np.sort(np.concatenate(ev)), np.sort(np.concatenate(rf))
        tab = lambda Jx: (lambda Tj: dict(pt=Tj['pt'], eta=Tj['eta'], phi=Tj['phi'], m=Tj['m'], ax_eta=Tj['ax_eta'], ax_phi=Tj['ax_phi'], ext=Tj['ext']))(jc_table(Jx))
        J0 = sub(Jall, ev); y = yall[ev]; T0 = tab(J0); Tp = tab(sub(Jall, rf)); Tp['ext'] = {}; yp = yall[rf]; del Jall
        classes = list(CL); ntop = y == CL.index('Tbqq')
        info = f'JetClass first test file of Wqq, Zqq, Tbqq, QCD: 5,000 random rows per class (rng {seed}); 2,000 evaluated + 3,000 reference per class (rng {seed + 10})'
    else:
        from interp import data_top as dt
        d = dt.DST / 'test'; ylab = np.load(d / 'label.npy')
        ev = H._rows(ylab, 5000, seed); used = np.zeros(len(ylab), bool); used[ev] = True; r = np.random.default_rng(seed + 10)
        rf = np.sort(np.concatenate([r.choice(np.flatnonzero((ylab == c) & ~used), 5000, replace=False) for c in (0, 1)]))
        Fm = np.load(d / 'full.npy', mmap_mode='r'); jet = np.load(d / 'jet.npy')
        tab = lambda ii: H.table_from_full(np.asarray(Fm[ii]), np.minimum(jet[ii, 5].astype(np.int64), Fm.shape[1]))
        T0, Tp = tab(ev), tab(rf); y, yp = ylab[ev].astype(int), ylab[rf].astype(int); classes = ['QCD', 'top']; ntop = y == 1; extra = None
        info = f'Top Landscape test: 5,000 top + 5,000 QCD random rows (rng {seed}); reference 5,000 + 5,000 disjoint rows (rng {seed + 10})'
    T0 = sort_first(T0)
    if ds == 'jetclass':
        extra = dict(J0=J0, ext=T0['ext'])
    good = (T0['pt'] > 0).sum(1) >= 3
    log(f'{ds}: {len(y)} evaluated jets ({np.bincount(y).tolist()}), {len(yp)} reference jets; jets with < 3 particles: {int((~good).sum())}')
    out, G = make_edits(T0, y, ntop, seed)
    res, per = edit_stats(T0, G, out, y, classes); res['sample'] = info; res['N'] = int(len(y)); res['lever_diag'] = lever_diag(T0, G, y, classes)
    (OUT / f'h67_edits_{ds}.json').write_text(json.dumps(res, indent=1, default=float)); log(f'wrote h67_edits_{ds}.json')
    save_tables(ds, T0, y, out, extra)
    t = time.time(); RF = dict(ref=ref_feats(sort_first(Tp)), yref=yp)
    for nm, T in [('U', T0)] + [(e, out[e][0]) for e in EDITS]: RF[nm] = ref_feats(T)
    RF['perjet'] = per; RF['classes'] = classes
    with open(WORK / f'{ds}_feats.pkl', 'wb') as fh: pickle.dump(RF, fh, protocol=4)
    log(f'reference features written ({time.time() - t:.0f} s)')
    if ds == 'hls': fig_jet(T0, y, out, G)


# ============================================================================================================== eval step
def null_torch(module, seed):
    import torch
    m2 = copy.deepcopy(module); g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m2.parameters():
            v = p.data.detach().cpu().float().reshape(-1); p.data.copy_(v[torch.randperm(v.numel(), generator=g)].reshape(p.shape).to(p.dtype).to(p.device))
    return m2.eval()


def run_models(ds, Ts, models, prep):
    L = {}
    for nm, T in Ts.items():
        t = time.time(); X = prep(H.sortT(T))
        for mn, fn in models.items(): L[f'{mn}__{nm}'] = np.asarray(fn(X), np.float32)
        log(f'{ds} {nm}: {len(models)} models ({time.time() - t:.0f} s)')
    np.savez_compressed(WORK / f'{ds}_logits.npz', **L); log(f'wrote {WORK / f"{ds}_logits.npz"}')
    return L


def part_eval(ds):
    if ds in ('hls_jedi', 'hls_small'):
        Ts, y, _ = load_tables('hls')
        if ds == 'hls_jedi':
            from interp.jedi_core import Jedi
            from jetdistill import network as N
            nets = dict(jedi_n64=Jedi(64, '3-feature')); orig = N.redraw_untrained; N.redraw_untrained = H.permute_weights
            try:
                for s in range(6): nets[f'jedi_n64_perm{s}'] = Jedi(64, '3-feature', untrained=True, seed=11 + s)
            finally: N.redraw_untrained = orig
            models = {k: (lambda X, n=n: n.run(X)['logits']) for k, n in nets.items()}
        else:
            import torch
            from interp.h1_irc_panel import load_small, torch_logits
            torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1'))); models = {}
            for tag, run in (('dsets', 'dsets-d64-n64'), ('tf_d64', 'tf-2b-d64-n64-e100'), ('jedis', 'jedis-d64-n64')):
                net, sh, sc = load_small(V / 'models' / 'small' / run / 'seed0')
                models[tag] = (lambda X, net=net, sh=sh, sc=sc: torch_logits(net, X, sh, sc))
        run_models(ds, Ts, models, H.hls_prep)
    elif ds == 'jetclass':
        from interp.h1_irc_panel import pn_logits
        from interp.part_core import ParTNetwork, inputs, build
        from interp.p1v_verify_part import fwd
        from interp.pn_net import load_particlenet
        from jetdistill.config import MODELS
        dev = os.environ.get('JETDISTILL_DEVICE', 'cpu'); Ts, y, extra = load_tables('jetclass'); J0 = extra['J0']

        def prep(T):
            Jx = dict(J0); Jx['jet_eta'] = T['ax_eta'].astype(np.float32); Jx['jet_phi'] = T['ax_phi'].astype(np.float32)
            sg = np.where(Jx['jet_eta'] < 0, -1.0, 1.0)[:, None]; de, dp = H.rel(T)
            return build(Jx, T['pt'], sg * de, dp, T['m'], extra=T['ext'])
        pt_f = lambda net, n: (lambda J: fwd(net, *inputs(J, n))[1].astype(np.float64))
        models = {}; parts = {n: ParTNetwork(n) for n in ('full', 'kin')}
        for n, net in parts.items():
            models[f'ParT_{n}'] = pt_f(net, n)
            for s in range(3):
                pk = copy.copy(net); pk.model = null_torch(net.model, 11 + s); models[f'ParT_{n}_perm{s}'] = pt_f(pk, n)
        for n, dims in (('full', 17), ('kin', 7)):
            pn = load_particlenet(Path(MODELS).parent / 'particlenet' / f'ParticleNet_{n}.pt', dims, dev)
            models[f'PN_{n}'] = (lambda J, p=pn, n=n: pn_logits(p, J, n))
        run_models(ds, Ts, models, prep)
    else:
        import torch
        from interp.h4_top_wmass import WeaverNet, LNet, Flow, kin3_of
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'; Ts, y, _ = load_tables('top')

        def prep(T):
            pt, eta, phi, m = T['pt'], T['eta'], T['phi'], T['m']; real = pt > 0
            P4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)], -1) * real[..., None]
            return dict(P4=P4.astype(np.float32), nn=real.sum(1))
        two = lambda s: np.stack([np.zeros_like(s), s], 1); models = {}; wf = lambda net: (lambda X: two(net.score(X['P4'], X['nn'])))
        for kind, dname in (('ParT', 'part_kin'), ('PN', 'pnet_kin')):
            net = WeaverNet(kind, V / 'models/top' / dname / 'net_best_epoch_state.pt', dev); models[f'{kind}_kin'] = wf(net)
            if kind == 'ParT':
                for s in range(6):
                    n2 = copy.copy(net); n2.model = null_torch(net.model, 11 + s); models[f'ParT_kin_perm{s}'] = wf(n2)
        ln = LNet(dev, mass64=True); order = np.random.default_rng(SEEDS['top'] + 5).permutation(len(y))
        models['LorentzNet'] = (lambda X: two(ln.score(X['P4'], X['nn'], order)))
        for a in ('pfn', 'efn'):
            fl = Flow(a); models[a.upper()] = (lambda X, f_=fl: two(f_.score(kin3_of(X['P4'], X['nn']))))
        run_models(ds, Ts, models, prep)


# ============================================================================================================== summary
KNOWN = {'hls|jedi_n64': '0.8066 (H64, seed 6401)', 'hls|dsets': '0.8019 (H64)', 'hls|tf_d64': '0.8104 (H64)', 'hls|jedis': 'test.json',
         'jetclass|ParT_full': 'per class H64: Wqq 0.79, Zqq 0.69, Tbqq 0.96, QCD 0.77', 'jetclass|ParT_kin': 'H64: Wqq 0.69, Zqq 0.56, Tbqq 0.91, QCD 0.71',
         'jetclass|PN_full': 'H64: Wqq 0.77, Zqq 0.64, Tbqq 0.95, QCD 0.75', 'jetclass|PN_kin': 'H64: Wqq 0.67, Zqq 0.53, Tbqq 0.91, QCD 0.69',
         'top|ParT_kin': '0.937 (H64)', 'top|PN_kin': '0.940 (H64)', 'top|LorentzNet': '0.943 (H64)', 'top|PFN': '0.929 (H64)', 'top|EFN': '0.923 (H64)'}
REFS = {'MLP(m,pT)': ('m', 'jpt'), 'MLP(w,pT)': ('w', 'jpt'), 'MLP(m,w,pT)': ('m', 'w', 'jpt'), 'MLP(m_A,pT)': ('mA', 'jpt')}


def logodds_all(Lg):
    Lg = np.clip(np.nan_to_num(np.asarray(Lg, np.float64), neginf=-60.0, posinf=60.0), -1e6, 1e6)
    ls = H.logsoft(Lg); C = ls.shape[1]; out = np.empty_like(ls)
    for k in range(C):
        o = np.delete(ls, k, 1); mx = o.max(1, keepdims=True); out[:, k] = ls[:, k] - (mx[:, 0] + np.log(np.exp(o - mx).sum(1)))
    return out


def pnorm(d):
    d = np.asarray(d, float); n = len(d)
    if n < 2: return 1.0
    se = d.std(ddof=1) / math.sqrt(n)
    if se == 0: return 1.0 if d.mean() == 0 else 0.0
    return float(math.erfc(abs(d.mean()) / se / math.sqrt(2)))


def ref_logits(RF, classes_used):
    """train the reference classifiers on the reference jets; return {ref: {edit: logits (J, C_used)}}"""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    yr = RF['yref']; lab = {c: i for i, c in enumerate(classes_used)}; yr = np.array([lab[v] for v in yr])
    X = lambda F, keys: np.nan_to_num(np.stack([np.log(np.maximum(F[k], 1e-6)) for k in keys], 1), nan=0.0, posinf=20.0, neginf=-20.0)   # diverged jets (excluded) only
    out = {}; acc = {}
    for nm, keys in REFS.items():
        clf = make_pipeline(StandardScaler(), MLPClassifier((32, 32), max_iter=400, early_stopping=True, random_state=0)).fit(X(RF['ref'], keys), yr)
        out[nm] = {e: clf.predict_log_proba(X(RF[e], keys)) for e in ['U'] + EDITS}
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(RF['ref']['efp'], yr)
    out['GBDT(EFP)'] = {e: np.log(np.clip(clf.predict_proba(np.nan_to_num(RF[e]['efp'], nan=0.0, posinf=0.0, neginf=0.0)), 1e-12, 1)) for e in ['U'] + EDITS}
    return out


def class_stats(Ds, conv, yk, rng):
    """Ds: {edit: per-jet Delta s (own class)} for the jets of one class; conv: {edit: bool}"""
    r = {}
    for sgn, (a, b) in PAIRS.items():
        sel = conv[a] & conv[b] & conv['crot']; n = int(sel.sum())
        if n < 20: r[sgn] = dict(n=n); continue
        A, B, R0 = np.abs(Ds[a][sel]), np.abs(Ds[b][sel]), np.abs(Ds['crot'][sel])
        I = rng.integers(0, n, (NB, n)); As, Bs, Rs = A[I].sum(1), B[I].sum(1), R0[I].sum(1)
        with np.errstate(divide='ignore', invalid='ignore'):
            rho_b = As / Bs; phi_b = As / (As + Bs); phx_b = np.maximum(As - Rs, 0) / (np.maximum(As - Rs, 0) + np.maximum(Bs - Rs, 0))
        sA, sB, sR = A.sum(), B.sum(), R0.sum()
        r[sgn] = dict(n=n, mean_abs=dict(Ew=float(A.mean()), Em=float(B.mean()), crot=float(R0.mean())),
                      mean_abs_ci=dict(Ew=np.percentile(As / n, [2.5, 97.5]).tolist(), Em=np.percentile(Bs / n, [2.5, 97.5]).tolist(),
                                       crot=np.percentile(Rs / n, [2.5, 97.5]).tolist()),
                      mean_signed=dict(Ew=float(Ds[a][sel].mean()), Em=float(Ds[b][sel].mean())),
                      rho=float(sA / sB) if sB > 0 else None, rho_ci=np.nanpercentile(rho_b, [2.5, 97.5]).tolist() if sB > 0 else None,
                      phi=float(sA / (sA + sB)) if sA + sB > 0 else None, phi_ci=np.nanpercentile(phi_b, [2.5, 97.5]).tolist() if sA + sB > 0 else None,
                      phi_excess=float(max(sA - sR, 0) / (max(sA - sR, 0) + max(sB - sR, 0))) if (max(sA - sR, 0) + max(sB - sR, 0)) > 0 else None,
                      phi_excess_ci=np.nanpercentile(phx_b, [2.5, 97.5]).tolist() if np.isfinite(phx_b).any() else None,
                      p_Ew_eq_crot=pnorm(A - R0), p_Em_eq_crot=pnorm(B - R0))
    for e in EDITS:
        sel = conv[e]
        if sel.sum() >= 20:
            x = Ds[e][sel]; I = rng.integers(0, len(x), (NB, len(x)))
            r[e] = dict(n=int(sel.sum()), mean=float(x.mean()), ci=np.percentile(x[I].mean(1), [2.5, 97.5]).tolist(),
                        mean_abs=float(np.abs(x).mean()), mean_abs_ci=np.percentile(np.abs(x)[I].mean(1), [2.5, 97.5]).tolist(),
                        p_abs_vs_crot=pnorm(np.abs(x) - np.abs(Ds['crot'][sel])) if e != 'crot' else None)
    return r


def summarize_ds(ds, logit_files, model_role, classes_eval):
    with open(WORK / f'{ds}_feats.pkl', 'rb') as fh: RF = pickle.load(fh)
    d = np.load(WORK / f'{ds}_tables.npz'); y = d['y'].astype(int); classes = RF['classes']; per = RF['perjet']
    conv = {e: per[f'{e}__conv'] for e in EDITS}; rng = np.random.default_rng(SEEDS[ds] + 7)
    Lall = {}
    for f in logit_files:
        if (WORK / f).exists():
            z = np.load(WORK / f)
            for k in z.files:
                mn, e = k.split('__'); Lall.setdefault(mn, {})[e] = z[k]
        else: log(f'missing {f}')
    used = sorted(set(y.tolist())); cu = [classes[k] for k in used]
    R = ref_logits(RF, used)
    yl = np.array([used.index(v) for v in y])
    out = dict(models={}, classes=[classes[k] for k in classes_eval])
    allm = {**{m: ('trained' if model_role(m) == 'trained' else 'null', L) for m, L in Lall.items()}, **{m: ('ref', L) for m, L in R.items()}}
    for mn, (role, L) in allm.items():
        yy = yl if role == 'ref' else y
        lo = {e: logodds_all(L[e]) for e in L}; c = L['U'].argmax(1)
        acc = float((c == yy).mean()); pc = {classes[k]: float((c[y == k] == yy[y == k]).mean()) for k in classes_eval}
        r = dict(role=role, acc=acc, acc_ci=H.boot_frac(c == yy, rng), per_class_acc=pc, known=KNOWN.get(f'{ds.split("_")[0]}|{mn}'), cls={}, wp={})
        for k in classes_eval:
            kk = used.index(k) if role == 'ref' else k; sel = y == k
            Ds = {e: (lo[e][:, kk] - lo['U'][:, kk])[sel] for e in EDITS}; cv = {e: conv[e][sel] for e in EDITS}
            r['cls'][classes[k]] = class_stats(Ds, cv, k, rng)
            thr = float(np.median(lo['U'][sel, kk])); c0 = lo['U'][:, kk] > thr; wp = {}
            chg = {e: ((lo[e][:, kk] > thr) != c0) for e in EDITS}
            for e in EDITS:
                if e == 'crot': continue
                s2 = conv[e] & conv['crot']
                wp[e] = dict(changed=float(chg[e][s2].mean()), changed_crot=float(chg['crot'][s2].mean()), mcnemar=H.mcnemar(chg[e][s2], chg['crot'][s2]))
            r['wp'][classes[k]] = dict(threshold=thr, edits=wp)
        # class-pooled phi (sum over classes of |Delta s|)
        for sgn in PAIRS:
            As = Bs = 0.0
            for k in classes_eval:
                st = r['cls'][classes[k]].get(sgn, {})
                if 'mean_abs' in st: As += st['mean_abs']['Ew'] * st['n']; Bs += st['mean_abs']['Em'] * st['n']
            r[f'phi_pooled{sgn}'] = float(As / (As + Bs)) if As + Bs > 0 else None
        out['models'][mn] = r
        log(f'{ds} {mn} [{role}] acc {acc:.4f} (known {r["known"]}) | ' + ' '.join(
            f'{cls}: phi_uneq {r["cls"][cls].get("uneq", {}).get("phi")} phi_eq {r["cls"][cls].get("eq", {}).get("phi")}' for cls in out['classes']))
    return out


def part_summary():
    S = dict(datasets={}, holm={})
    cfg = dict(hls=(['hls_jedi_logits.npz', 'hls_small_logits.npz'], list(range(5))),
               jetclass=(['jetclass_logits.npz'], None), top=(['top_logits.npz'], [0, 1]))
    for ds, (files, ce) in cfg.items():
        if not (WORK / f'{ds}_feats.pkl').exists(): log(f'{ds}: no features'); continue
        if ce is None:
            from interp.part_core import CL
            ce = [CL.index(c) for c in JC_CL]
        S['datasets'][ds] = summarize_ds(ds, files, lambda m: 'null' if 'perm' in m else 'trained', ce)
        S['datasets'][ds]['edits'] = json.loads((OUT / f'h67_edits_{ds}.json').read_text())
    # Holm over trained models x edits x classes (|Delta s| vs crot) and McNemar (calls changed vs crot)
    for fam in ('abs_vs_crot', 'mcnemar'):
        keys, ps = [], []
        for ds, D in S['datasets'].items():
            for mn, r in D['models'].items():
                if r['role'] != 'trained': continue
                for c, st in r['cls'].items():
                    for e in EDITS:
                        if e == 'crot': continue
                        if fam == 'abs_vs_crot' and e in st and st[e].get('p_abs_vs_crot') is not None: keys.append(f'{ds}|{mn}|{c}|{e}'); ps.append(st[e]['p_abs_vs_crot'])
                        if fam == 'mcnemar' and e in r['wp'][c]['edits']: keys.append(f'{ds}|{mn}|{c}|{e}'); ps.append(r['wp'][c]['edits'][e]['mcnemar']['p'])
        hp = H.holm(ps) if ps else []
        S['holm'][fam] = dict(n_tests=len(ps), n_below_0p05=int(np.sum(np.array(hp) < 0.05)), p=dict(zip(keys, hp)))
    # assignment: references whose phi lies inside the network's 95 % interval
    S['assignment'] = {}
    for ds, D in S['datasets'].items():
        refs = {m: r for m, r in D['models'].items() if r['role'] == 'ref'}
        for mn, r in D['models'].items():
            if r['role'] == 'ref': continue
            for c, st in r['cls'].items():
                for sgn in PAIRS:
                    s = st.get(sgn, {})
                    if s.get('phi_ci') is None: continue
                    lo, hi = s['phi_ci']; inside = [rn for rn, rr in refs.items() if rr['cls'][c].get(sgn, {}).get('phi') is not None and lo <= rr['cls'][c][sgn]['phi'] <= hi]
                    S['assignment'][f'{ds}|{mn}|{c}|{sgn}'] = dict(phi=s['phi'], ci=s['phi_ci'], refs_inside=inside,
                                                                   ref_phi={rn: rr['cls'][c].get(sgn, {}).get('phi') for rn, rr in refs.items()})
    (OUT / 'h67_summary.json').write_text(json.dumps(S, indent=1, default=float)); log('wrote h67_summary.json')
    fig_phi(S)
    try:
        Ts, y, _ = load_tables('hls'); G = geometry(Ts['U'], y == 4); G['w0'] = width(Ts['U']['pt'], Ts['U']['eta'], Ts['U']['phi'], G['ee'], G['ep'])
        with open(WORK / 'hls_feats.pkl', 'rb') as fh: per = pickle.load(fh)['perjet']
        fig_jet(Ts['U'], y, {e: (Ts[e], dict(conv=per[f'{e}__conv'])) for e in EDITS}, G)
    except Exception as ex:
        log(f'jet figure not redrawn: {ex!r}')


# ============================================================================================================== figures
def fig_jet(T0, y, out, G):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    O0 = obs(T0, G['ee'], G['ep'])
    ok = out['Ew-10'][1]['conv'] & out['Em+10'][1]['conv']
    w = np.flatnonzero((y == 2) & ok & (np.abs(O0['m'] - 80) < 8) & ((T0['pt'] > 0).sum(1) >= 15))
    if not len(w): log('no W jet for the figure'); return
    j = w[len(w) // 2]; sub = lambda T: {k: (v[j:j + 1] if isinstance(v, np.ndarray) else v) for k, v in T.items()}
    sets = ((sub(T0), 'unedited W jet'), (sub(out['Ew-10'][0]), 'E_w: width -10 %, mass fixed'), (sub(out['Em+10'][0]), 'E_m: mass +10 %, width fixed'))
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6), sharex=True, sharey=True)
    for a, (T, ttl) in zip(ax, sets):
        O = obs(T, G['ee'][j:j + 1], G['ep'][j:j + 1]); real = T['pt'][0] > 0
        de = T['eta'][0] - G['ee'][j]; dp = H.wrap(T['phi'][0] - G['ep'][j]); lab = G['lab'][j]
        Pk = [T['pt'][0][lab == k].sum() for k in range(2)]; hard = int(np.argmax(Pk))
        for k, col in ((hard, '#2e5eaa'), (1 - hard, '#d1495b')):
            msk = real & (lab == k); lb = f'{"harder" if k == hard else "softer"} prong: pT share {Pk[k] / sum(Pk):.2f}'
            a.scatter(de[msk], dp[msk], s=6 + 300 * T['pt'][0, msk] / T['pt'][0].max(), c=col, alpha=0.65, edgecolors='k', linewidths=0.3, label=lb)
        a.set_title(f'{ttl}\nm = {O["m"][0]:.1f} GeV, w = {O["w"][0]:.4f}, jet pT = {O["jpt"][0]:.1f} GeV', fontsize=9.5)
        a.axhline(0, color='0.85', lw=0.5); a.axvline(0, color='0.85', lw=0.5); a.set_xlabel('Δη to jet axis'); a.legend(fontsize=7.5, frameon=False, loc='lower left')
    ax[0].set_ylabel('Δφ to jet axis')
    fig.suptitle('Width and mass moved separately: shift pT between the two prongs and change their opening angle\n'
                 '(one hls4ml W jet; jet pT, η, φ exact; marker area proportional to particle pT)', fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(OUT / 'h67_jet.png', dpi=130); log(f'wrote {OUT / "h67_jet.png"}')


def fig_phi(S):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    name = {'hls': 'hls4ml', 'jetclass': 'JetClass', 'top': 'Top'}
    rows = []
    for ds, D in S['datasets'].items():
        for mn, r in D['models'].items():
            if r['role'] == 'null' and not mn.endswith('perm0'): continue
            for c in D['classes']:
                if ds == 'hls' and c not in ('W', 'Z', 't', 'g'): continue
                rows.append((ds, mn, r['role'], c))
    order = {'trained': 0, 'ref': 1, 'null': 2}
    rows.sort(key=lambda x: (list(name).index(x[0]), order[x[2]], x[1], x[3]))
    fig, ax = plt.subplots(figsize=(8.5, 0.2 * len(rows) + 1.8)); cols = {'uneq': '#2e5eaa', 'eq': '#d1495b'}
    for i, (ds, mn, role, c) in enumerate(rows):
        for dl, sgn in ((-0.18, 'uneq'), (0.18, 'eq')):
            s = S['datasets'][ds]['models'][mn]['cls'][c].get(sgn, {})
            if s.get('phi') is None: continue
            x = s['phi']; lo, hi = s['phi_ci']
            ax.errorbar(x, i + dl, xerr=[[max(x - lo, 0)], [max(hi - x, 0)]], fmt='o', color=cols[sgn], ms=3.2, lw=0.9,
                        label={'uneq': 'w -10 % vs m +10 % (prong sharing more unequal)', 'eq': 'w +10 % vs m -10 % (sharing more equal)'}[sgn] if i == 0 else None)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f'{name[d]} {("ref " if ro == "ref" else "null " if ro == "null" else "")}{m} | {c}' for d, m, ro, c in rows], fontsize=6.5)
    ax.invert_yaxis(); ax.axvline(0, color='k', lw=0.7, ls='--'); ax.axvline(1, color='k', lw=0.7, ls=':'); ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel('φ = Σ|Δs| under the width edit / (Σ|Δs| width edit + Σ|Δs| mass edit)\n0: responds only to the mass edit (as MLP(m, pT)); '
                  '1: only to the width edit (as MLP(w, pT))', fontsize=8.5)
    ax.set_title('Response to a width change at fixed mass vs a mass change at fixed width (own-class log-odds, 95 % bootstrap)', fontsize=9.5)
    ax.legend(fontsize=8, frameon=False, loc='lower right'); fig.tight_layout(); fig.savefig(OUT / 'h67_phi.png', dpi=130); log(f'wrote {OUT / "h67_phi.png"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', required=True, choices=['edits', 'eval', 'summary'])
    ap.add_argument('--ds', default=None, choices=['hls', 'jetclass', 'top', 'hls_jedi', 'hls_small'])
    a = ap.parse_args(); log(f'H67 part {a.part} ds {a.ds}')
    if a.part == 'edits': part_edits(a.ds)
    elif a.part == 'eval': part_eval(a.ds)
    else: part_summary()
    log('done')
