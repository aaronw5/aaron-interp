"""H69 (loop D; claim D25 v2; follows H67 / H67v): which shape variable do the taggers read beside the jet mass?  Separate the
pT-weighted width w, the prong pT sharing s, D2 and tau21 with exact per-jet edits at fixed jet mass m and jet pT.

Definitions (about the E-scheme axis = direction of the summed constituent three-momentum of the unedited jet)
  w     = sum_i pT_i dR_i / sum_i pT_i, dR_i = (eta, phi) distance of particle i to the axis.
  s     = 1 - sum_k (P_k / P)^2, P_k = pT of prong k, P = sum pT. Prongs = exclusive-kT subjets of the unedited jet (3 for top-class
          jets, 2 otherwise, interp.h41_colour_flow.ktn_batch); the prong labels stay fixed through every edit, so s is a function of the
          prong pTs only. Two prongs: s = 2 z (1 - z), z = softer prong's pT share. (References also use s2 / s3 re-clustered on each jet.)
  D2    = e3 / e2^3 (beta = 1) on all constituents (interp.h45_prong_sharing.d2_batch).
  tau21 = tau2 / tau1 (beta = 1, exclusive-kT WTA axes; interp.h45_prong_sharing.tau21_batch).
  Small-angle identity that motivates the third lever: m^2 / pT^2 ~ <r^2> (pT-weighted mean of r^2 about the axis) and w = <r>, and for
  two point-like prongs w = sqrt(2 s) m / pT. At fixed m, pT and s, w can only move through the radial pT profile inside the prongs.

Edit map apply3(f, lu, c, cs) (particle count and order kept; particle masses scale with their pT so each particle keeps its direction):
  1 c  (all prongs; the 'width lever'): pT_i x (1 + c h_i). h_i is, inside each prong, the distance r_i of the particle to the jet axis
       made orthogonal (pT-weighted least squares inside the prong) to {1, x_eta, x_phi, r^2}, scaled to max |h| = 1 in the prong. So for
       every c, exactly: prong pT P_k, prong pT-weighted centroid and prong sum pT r^2 are unchanged (the prong four-momentum and the jet
       mass are unchanged to first order), and sum pT r (the width) changes. It moves pT between the inner (core) and outer (edge) part
       of each prong, with the weights chosen so that the prong four-momentum is held.  [Plan: 'core vs edge (0.05 of the prong axis)'
       with the prong four-momentum fixed to first order. A two-set core/edge reweighting cannot hold P_k, the centroid and sum pT r^2
       at once (H67 lever_diag: it moves m and w together), so the orthogonalised radial profile is used.]
  2 cs (softest prong only; the 'D2 lever'): core particles (within 0.05 of the prong centroid; the pT-median split if the prong has
       < 2 % or > 98 % of its pT inside 0.05) x (1 - cs a_E), edge particles x (1 + cs a_C) (a_C / a_E = core / edge pT share of the
       prong): prong pT unchanged, prong mass changes; prong recentred on its centroid.
  3 f  (prong sharing): prong k x (1 + f (P/K - P_k) / P_k) (H67 lever 'z'): f = 1 equal sharing, f < 0 more unequal.
  4 lu (opening): prong centroids scaled by u = exp(lu) about the axis (rigid prong translations).
  5 exact axis restore (4 passes) and jet pT restore (every four-vector x const).
  Per jet, the free levers are solved by a damped Newton iteration (finite-difference Jacobian, least-squares step) for the targets.
Edits (targets in log; a jet enters a comparison only if the solve converged: |residual| < 1e-5 for m, w, s and < 1e-4 for D2, tau21)
  Ew+-10 / Ew+-5   w x (1 +- 0.10 / 0.05) at fixed m, pT, s        levers f, lu, c
  Es+-10 / Es+-5   s x (1 +- 0.10 / 0.05) at fixed m, pT, w        levers f, lu, c
  ED2+-10 / ED2+-5 D2 x (1 +- 0.10 / 0.05) at fixed m, pT, s       levers f, lu, cs   (w free, reported)
  Et+-10           tau21 x (1 +- 0.10) at fixed m, pT, s, D2       levers f, lu, c, cs (w free, reported)
  crot_dm          every prong rotated about its centroid by one angle per jet whose particle displacement equals the mean displacement
                   of the converged 10 % edits on that jet, random sign; then m, w, s restored exactly (levers f, lu, c)
  crand            every particle displaced by the same distance (that matched displacement) in a random direction, prongs recentred,
                   then m and s restored (levers f, lu; w free)
Response: own-class one-vs-rest log-odds; Delta s = log-odds(edited) - log-odds(unedited) per jet.
  R_E = mean |Delta s(E)| / mean |Delta s(crot_dm)| on the jets where both converged.
  phi_X = M_X / (M_w + M_s + M_D2), M_X = mean of mean|Delta s| over the two signs of the 10 % edit of X (and the same with 5 %).
References (sklearn MLP(32, 32) / HistGradientBoosting on train rows of the same dataset): MLP(m,w,pT), MLP(m,s,pT) (s2, s3),
  MLP(m,D2,pT), MLP(m,tau21,pT), GBDT(EFP), evaluated on the same edited jets.
Statistics: bootstrap over jets (1,000), paired sign-flip permutation test |Delta s(E)| vs |Delta s(crot_dm)| (2,000 flips), McNemar on
  calls at eps_S = 0.5 (threshold = median unedited own-class log-odds of the class), Holm over trained models x classes x edits.
Samples: hls4ml jedi_core.sample(per_class=1000, seed=6901) (reference: 3,000 per class from train_ptetaphi_f16, rng 6911);
  JetClass val_5M <sample>_122.root, 1,000 random rows per class of Wqq, Zqq, Tbqq, Hbb, QCD (rng 6902) (reference: train_100M
  <sample>_005.root, 3,000 random rows per class, rng 6912); Top Landscape test 2,000 + 2,000 random rows (rng 6903) (reference:
  train split 3,000 + 3,000 random rows, rng 6913).
Run
  AI_PY=conda ./job.sh d-h69-ed-<ds> 2 6 'python -m interp.h69_width_sharing_d2 --part edits --ds <hls|jetclass|top>'
  ./job.sh d-h69-ev-jedi 'python -m interp.h69_width_sharing_d2 --part eval --ds hls_jedi'
  ./part_job.sh d-h69-ev-gpu 'python -m interp.h69_width_sharing_d2 --part eval --ds jetclass && python -m interp.h69_width_sharing_d2 --part eval --ds top'
  AI_PY=conda ./job.sh d-h69-sum 2 6 'python -m interp.h69_width_sharing_d2 --part summary'
  -> $AI_RESULTS/physics/h69_work/*, h69_edits_<ds>.json, h69_summary.json, h69_jet.png, h69_response.png
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
from interp import h67_mass_vs_width as M  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
SMOKE = int(os.environ.get('H69_SMOKE', '0'))      # > 0: that many jets per class, outputs under physics/h69_smoke/
OUT = RES / 'physics' / ('h69_smoke' if SMOKE else '')
WORK = OUT / 'h69_work'
M.WORK = WORK
V = Path('/j-jepa-vol/aaron-interp')
JCV = Path('/j-jepa-vol/JetClass/Pythia')
T00 = time.time()
SEEDS = dict(hls=6901, jetclass=6902, top=6903)
NB = 1000
NPERM = 2000
RCORE = 0.05
TOL, TOL2 = 1e-5, 1e-4
JC_CL = ['Wqq', 'Zqq', 'Tbqq', 'Hbb', 'QCD']
L = math.log
EDITS = ['Ew+10', 'Ew-10', 'Es+10', 'Es-10', 'ED2+10', 'ED2-10', 'Ew+5', 'Ew-5', 'Es+5', 'Es-5', 'ED2+5', 'ED2-5', 'Et+10', 'Et-10',
         'crot_dm', 'crand']
SPEC = {}
for sz in (10, 5):
    a = 1 + sz / 100
    for sg, v in (('+', L(a)), ('-', L(2 - a))):
        SPEC[f'Ew{sg}{sz}'] = (('f', 'lu', 'c'), (('m', 0.0), ('w', v), ('s', 0.0)))
        SPEC[f'Es{sg}{sz}'] = (('f', 'lu', 'c'), (('m', 0.0), ('w', 0.0), ('s', v)))
        SPEC[f'ED2{sg}{sz}'] = (('f', 'lu', 'cs'), (('m', 0.0), ('s', 0.0), ('D2', v)))
SPEC['Et+10'] = (('f', 'lu', 'c', 'cs'), (('m', 0.0), ('s', 0.0), ('D2', 0.0), ('tau21', L(1.1))))
SPEC['Et-10'] = (('f', 'lu', 'c', 'cs'), (('m', 0.0), ('s', 0.0), ('D2', 0.0), ('tau21', L(0.9))))
SPEC['crot_dm'] = (('f', 'lu', 'c'), (('m', 0.0), ('w', 0.0), ('s', 0.0)))
SPEC['crand'] = (('f', 'lu'), (('m', 0.0), ('s', 0.0)))
GROUPS = {10: {'w': ('Ew+10', 'Ew-10'), 's': ('Es+10', 'Es-10'), 'D2': ('ED2+10', 'ED2-10'), 'tau21': ('Et+10', 'Et-10')},
          5: {'w': ('Ew+5', 'Ew-5'), 's': ('Es+5', 'Es-5'), 'D2': ('ED2+5', 'ED2-5')}}
PHI_SETS = {'wsD2': ('w', 's', 'D2'), 'sD2': ('s', 'D2'), 'sD2tau21': ('s', 'D2', 'tau21')}
MAIN10 = ['Ew+10', 'Ew-10', 'Es+10', 'Es-10', 'ED2+10', 'ED2-10']


def log(*a):
    print(f'[{time.time() - T00:7.0f}s]', *a, flush=True)


gather, prong_sum, prong_centroid, jet3, width = M.gather, M.prong_sum, M.prong_centroid, M.jet3, M.width


# ============================================================================================================== geometry
def geometry69(T, ntop):
    G = M.geometry(T, ntop); real, lab, K, pt = G['real'], G['lab'], G['K'], T['pt']
    c05 = real & (G['rho'] <= RCORE)
    P = prong_sum(pt, lab, K); a = prong_sum(np.where(c05, pt, 0), lab, K) / np.maximum(P, 1e-300)
    bad = (P > 0) & ((a < 0.02) | (a > 0.98))
    G['core'] = np.where(gather(bad, lab), G['core'], c05) & real
    G['core_fallback_frac'] = float(bad.sum() / max((P > 0).sum(), 1))
    G['soft'] = np.argmin(np.where(P > 0, P, np.inf), 1)
    G['w0'] = width(T['pt'], T['eta'], T['phi'], G['ee'], G['ep'])
    G['o0'] = observe(T, G, ('m', 'w', 's', 'D2', 'tau21'))
    return G


def hfun(G, pt0, x):
    """width-lever profile h (J, L): inside each prong, r = |x| orthogonalised (pT-weighted) to {1, x_eta, x_phi, r^2}, max |h| = 1"""
    real, lab, K = G['real'], G['lab'], G['K']; r = np.hypot(x[..., 0], x[..., 1])
    c = gather(prong_centroid(pt0, x, lab, K), lab); d = x - c
    sig = np.sqrt(np.maximum(gather(prong_sum(pt0 * (d ** 2).sum(-1), lab, K) / np.maximum(prong_sum(pt0, lab, K), 1e-300), lab), 1e-12))
    Pk = np.maximum(prong_sum(pt0, lab, K), 1e-300)
    r2c = gather(prong_sum(pt0 * r ** 2, lab, K) / Pk, lab); r1c = gather(prong_sum(pt0 * r, lab, K) / Pk, lab)
    B = np.stack([np.ones_like(r), d[..., 0] / sig, d[..., 1] / sig, (r ** 2 - r2c) / np.maximum(sig * r1c, 1e-12)], -1)
    B = np.where(real[..., None], B, 0)
    Wk = np.stack([np.where(real & (lab == k), pt0, 0) for k in range(K)], -1)          # (J, L, K)
    Gm = np.einsum('jlk,jla,jlb->jkab', Wk, B, B); rhs = np.einsum('jlk,jla,jl->jka', Wk, B, r)
    tr = np.trace(Gm, axis1=2, axis2=3)[..., None, None]
    coef = np.linalg.solve(Gm + (1e-10 * tr + 1e-30) * np.eye(4), rhs[..., None])[..., 0]   # (J, K, 4)
    h = r - np.einsum('jla,jla->jl', B, gather(coef, lab))
    h = np.where(real, h, 0)
    nk = prong_sum(real.astype(float), lab, K)
    mx = np.stack([np.where(lab == k, np.abs(h), 0).max(1) for k in range(K)], 1)
    ok = (nk >= 5) & (mx > 0)
    h = np.where(gather(ok, lab), h / np.maximum(gather(mx, lab), 1e-300), 0)
    return np.where(real, h, 0)


def c_bounds(h, real):
    hp = np.where(real, h, 0).max(1); hn = np.where(real, -h, 0).max(1)
    return np.where(hp > 0, -0.95 / np.maximum(hp, 1e-12), -5.0), np.where(hn > 0, 0.95 / np.maximum(hn, 1e-12), 5.0)


def soft_shares(G, pt):
    lab, K = G['lab'], G['K']; P = prong_sum(pt, lab, K); PC = prong_sum(np.where(G['core'], pt, 0), lab, K)
    i = G['soft'][:, None]; Ps = np.take_along_axis(P, i, 1)[:, 0]; Cs = np.take_along_axis(PC, i, 1)[:, 0]
    aC = Cs / np.maximum(Ps, 1e-300); return aC, 1 - aC


def apply3(T0, G, f, lu, c, cs, h, base_pt=None, base_x=None):
    real, lab, K = G['real'], G['lab'], G['K']
    pt0 = T0['pt'] if base_pt is None else base_pt; x = (G['x0'] if base_x is None else base_x).copy()
    mfac0 = np.where(real, pt0 / np.maximum(T0['pt'], 1e-300), 0)
    cen = prong_centroid(pt0, x, lab, K); x = x - gather(cen - G['c'], lab)
    g = np.where(real, 1 + c[:, None] * h, 1.0)
    pt = np.where(real, pt0 * g, 0)
    aC, aE = soft_shares(G, pt)
    csx = np.clip(cs, -0.95 / np.maximum(aC, 1e-6), 0.95 / np.maximum(aE, 1e-6))
    soft = real & (lab == G['soft'][:, None])
    gs = np.where(soft, np.where(G['core'], 1 - (csx * aE)[:, None], 1 + (csx * aC)[:, None]), 1.0)
    pt = pt * gs; g = g * gs
    cen = prong_centroid(pt, x, lab, K); x = x - gather(cen - G['c'], lab)
    gf = M.lever_factors(G, pt, f, 'z'); gf = np.where(real, gf, 0); pt = pt * gf; g = g * gf
    x = x + (np.exp(lu)[:, None, None] - 1) * gather(G['c'], lab)
    mm = np.where(real, T0['m'] * mfac0 * g, 0)
    eta = G['ee'][:, None] + x[..., 0]; phi = G['ep'][:, None] + x[..., 1]
    for _ in range(4):
        _, je, jp, _ = jet3(pt, eta, phi, mm)
        eta = eta + (G['ee'] - je)[:, None]; phi = phi + H.wrap(G['ep'] - jp)[:, None]
    jpt, _, _, _ = jet3(pt, eta, phi, mm); kk = G['jpt0'] / np.maximum(jpt, 1e-300)
    pt = pt * kk[:, None]; mm = mm * kk[:, None]
    return dict(pt=np.where(real, pt, 0), eta=np.where(real, eta, 0), phi=np.where(real, H.wrap(phi), 0), m=np.where(real, mm, 0),
                ax_eta=T0['ax_eta'], ax_phi=T0['ax_phi'], ext=T0['ext'])


def observe(U, G, need):
    from interp.h45_prong_sharing import tau21_batch, d2_batch
    o = {}; lg = lambda a: np.log(np.maximum(a, 1e-12))
    if 'm' in need or 'w' in need:
        _, _, _, m = jet3(U['pt'], U['eta'], U['phi'], U['m']); o['m'] = lg(m); o['w'] = lg(width(U['pt'], U['eta'], U['phi'], G['ee'], G['ep']))
    if 's' in need:
        P = prong_sum(U['pt'], G['lab'], G['K']); S = P.sum(1); o['s'] = lg(1 - ((P / np.maximum(S, 1e-300)[:, None]) ** 2).sum(1))
    if 'D2' in need or 'tau21' in need:
        F = M.F4(U); nn = (U['pt'] > 0).sum(1).astype(np.int64)
        if 'D2' in need: o['D2'] = lg(d2_batch(F, nn))
        if 'tau21' in need: o['tau21'] = lg(tau21_batch(F, nn))
    return o


def solve3(T0, G, params, targets, base_pt=None, base_x=None, iters=80, hstep=1e-4):
    J = len(G['ee']); v = {k: np.zeros(J) for k in ('f', 'lu', 'c', 'cs')}
    x = G['x0'] if base_x is None else base_x; pt0 = T0['pt'] if base_pt is None else base_pt
    h = hfun(G, pt0, x - gather(prong_centroid(pt0, x, G['lab'], G['K']) - G['c'], G['lab']))
    fl, fh = M.f_bounds(G, base_pt, T0); cl, ch = c_bounds(h, G['real'])
    aC, aE = soft_shares(G, pt0)
    bnd = dict(f=(fl, fh), lu=(math.log(0.5), math.log(2.0)), c=(cl, ch), cs=(-0.95 / np.maximum(aC, 1e-6), 0.95 / np.maximum(aE, 1e-6)))
    need = tuple(t for t, _ in targets); tol = np.array([TOL if t in ('m', 'w', 's') else TOL2 for t in need])
    o0 = G['o0']

    def res(vv):
        U = apply3(T0, G, vv['f'], vv['lu'], vv['c'], vv['cs'], h, base_pt, base_x); o = observe(U, G, need)
        return np.nan_to_num(np.stack([o[t] - o0[t] - d for t, d in targets], 1), nan=1e3, posinf=1e3, neginf=-1e3)
    P = len(params)
    for it in range(iters):
        r = res(v); done = np.all(np.abs(r) < 0.05 * tol, 1)
        if done.all(): break
        cols = []
        for p in params:
            vv = dict(v); vv[p] = v[p] + hstep; cols.append((res(vv) - r) / hstep)
        A = np.stack(cols, 2); AtA = np.einsum('jtp,jtq->jpq', A, A); Atr = np.einsum('jtp,jt->jp', A, r)
        tr = np.trace(AtA, axis1=1, axis2=2)[:, None, None]
        step = -np.linalg.solve(AtA + (1e-10 * tr + 1e-20) * np.eye(P), Atr[..., None])[..., 0]
        step = np.nan_to_num(step)
        sc = np.minimum(1, 0.25 / np.maximum(np.abs(step).max(1), 1e-12)); sc = np.where(done, 0, sc)
        for i, p in enumerate(params): v[p] = np.clip(v[p] + sc * step[:, i], *bnd[p])
    r = res(v); conv = np.all(np.abs(r) < tol, 1)
    U = apply3(T0, G, v['f'], v['lu'], v['c'], v['cs'], h, base_pt, base_x)
    return U, dict(conv=conv, resid=np.max(np.abs(r), 1), **{k: v[k].copy() for k in v}, iters=it + 1)


# ============================================================================================================== edits
def rotate_prongs(G, theta):
    c = gather(G['c'], G['lab']); d = G['x0'] - c; tp = theta[:, None]
    return np.where(G['real'][..., None], c + np.stack([np.cos(tp) * d[..., 0] - np.sin(tp) * d[..., 1], np.sin(tp) * d[..., 0] + np.cos(tp) * d[..., 1]], -1), 0)


def make_edits(T0, y, ntop, seed):
    t = time.time(); G = geometry69(T0, ntop); J = len(y)
    log(f'geometry: {J} jets, prongs 3 for {int(ntop.sum())}; core fallback (pT-median split) for {G["core_fallback_frac"]:.3f} of prongs ({time.time() - t:.0f} s)')
    z = np.zeros(J); h0 = hfun(G, T0['pt'], G['x0'])
    U = apply3(T0, G, z, z, z, z, h0)
    log(f'round trip apply3(0): max |dpt| {np.abs(U["pt"] - T0["pt"]).max():.2e}, |deta| {np.abs(U["eta"] - T0["eta"]).max():.2e}')
    # lever reach at fixed opening (diagnostic): d log m, d log w, d log s, d log D2 for c = +-0.5 and cs = +-0.5
    diag = {}
    for nm, (c, cs) in {'c+0.5': (0.5, 0), 'c-0.5': (-0.5, 0), 'cs+0.5': (0, 0.5), 'cs-0.5': (0, -0.5)}.items():
        cl, ch = c_bounds(h0, G['real']); cc = np.clip(z + c, cl, ch)
        o = observe(apply3(T0, G, z, z, cc, z + cs, h0), G, ('m', 'w', 's', 'D2'))
        diag[nm] = {k: float(np.median(o[k] - G['o0'][k])) for k in o}
        log(f'lever {nm}: median ' + ' '.join(f'dlog{k} {v:+.4f}' for k, v in diag[nm].items()))
    out = {}
    for e in EDITS[:-2]:
        t = time.time(); pr, tg = SPEC[e]; U, inf = solve3(T0, G, pr, tg); out[e] = (U, inf)
        c = inf['conv']
        log(f'{e}: converged {c.mean():.3f} (' + ' '.join(f'{k}={c[y == k].mean():.2f}' for k in sorted(set(y.tolist()))) + ') ' +
            ' '.join(f'{p} med {np.median(inf[p][c]):+.3f}' for p in pr if c.any()) + f' ({time.time() - t:.0f} s, {inf["iters"]} it)')
    # displacement to match: mean over the converged 10 % edits on each jet
    Dm = np.stack([np.where(out[e][1]['conv'], M.displacement(G, out[e][0], T0['pt']), np.nan) for e in MAIN10], 1)
    with np.errstate(all='ignore'): D = np.nanmean(Dm, 1)
    D = np.where(np.isfinite(D), D, np.nanmedian(D)); G['Dmatch'] = D
    rng = np.random.default_rng(seed + 2000); pt = T0['pt']
    rbar = (pt * G['rho']).sum(1) / np.maximum(pt.sum(1), 1e-300)
    th = 2 * np.arcsin(np.clip(D / np.maximum(2 * rbar, 1e-12), 0, 1)) * rng.choice([-1.0, 1.0], J)
    U, inf = solve3(T0, G, *SPEC['crot_dm'], base_x=rotate_prongs(G, th)); out['crot_dm'] = (U, inf)
    log(f'crot_dm: converged {inf["conv"].mean():.3f}; displacement median {np.median(M.displacement(G, U, pt)[inf["conv"]]):.4f} vs matched {np.median(D):.4f}')
    a = rng.uniform(0, 2 * np.pi, pt.shape); xr = np.where(G['real'][..., None], G['x0'] + D[:, None, None] * np.stack([np.cos(a), np.sin(a)], -1), 0)
    U, inf = solve3(T0, G, *SPEC['crand'], base_x=xr); out['crand'] = (U, inf)
    log(f'crand: converged {inf["conv"].mean():.3f}; displacement median {np.median(M.displacement(G, U, pt)[inf["conv"]]):.4f}')
    return out, G, diag


def edit_stats(T0, G, out, y, classes):
    o0 = observe(T0, G, ('m', 'w', 's', 'D2', 'tau21')); res = dict(unedited={}, edits={})
    _, _, _, m0 = jet3(T0['pt'], T0['eta'], T0['phi'], T0['m'])
    for k, v in o0.items(): res['unedited'][f'log_{k}'] = dict(median=float(np.median(v)), q5_95=np.percentile(v, [5, 95]).tolist())
    # identity check w ~ sqrt(2 s) m / pT (two-prong jets)
    two = ~G['ntop']; pred = np.sqrt(2 * np.exp(o0['s'])) * m0 / G['jpt0']; rw = np.exp(o0['w']) / np.maximum(pred, 1e-12)
    res['identity_w_vs_sqrt2s_m_over_pt'] = dict(two_prong_median_ratio=float(np.median(rw[two])) if two.any() else None,
                                                 two_prong_corr_log=float(np.corrcoef(np.log(rw[two] * pred[two]), np.log(pred[two]))[0, 1]) if two.sum() > 2 else None,
                                                 three_prong_median_ratio=float(np.median(rw[~two])) if (~two).any() else None)
    per = {f'{k}0': v.astype(np.float32) for k, v in o0.items()}
    for e, (U, inf) in out.items():
        t = time.time(); c = inf['conv']; o1 = observe(U, G, ('m', 'w', 's', 'D2', 'tau21'))
        jpt, je, jp, _ = jet3(U['pt'], U['eta'], U['phi'], U['m'])
        dd = {k: o1[k] - o0[k] for k in o1}; dd['jpt'] = np.log(jpt / G['jpt0']); dd['axis'] = np.maximum(np.abs(je - G['ee']), np.abs(H.wrap(jp - G['ep'])))
        disp = M.displacement(G, U, T0['pt']); mv = M.moved_share(T0, U)
        er = dict(conv_frac=float(c.mean()), conv_by_class={classes[k]: float(c[y == k].mean()) for k in sorted(set(y.tolist()))},
                  displacement_median=float(np.median(disp[c])) if c.any() else None, moved_pt_share_median=float(np.median(mv[c])) if c.any() else None,
                  lever_medians={p: float(np.median(inf[p][c])) for p in ('f', 'lu', 'c', 'cs')} if c.any() else None, change={})
        for q, v in dd.items():
            er['change'][q] = {classes[k]: dict(median=float(np.median(v[c & (y == k)])), q5_95=np.percentile(v[c & (y == k)], [5, 95]).tolist())
                               for k in sorted(set(y.tolist())) if (c & (y == k)).sum() > 0}
        res['edits'][e] = er
        for q in ('m', 'w', 's', 'D2', 'tau21', 'jpt'): per[f'{e}__dlog{q}'] = dd[q].astype(np.float32)
        per[f'{e}__conv'] = c; per[f'{e}__disp'] = disp.astype(np.float32)
        log(f'{e}: conv {c.mean():.3f} | median dlog m {np.median(dd["m"][c]) if c.any() else 0:+.1e} w {np.median(dd["w"][c]) if c.any() else 0:+.4f} '
            f's {np.median(dd["s"][c]) if c.any() else 0:+.4f} D2 {np.median(dd["D2"][c]) if c.any() else 0:+.4f} tau21 {np.median(dd["tau21"][c]) if c.any() else 0:+.4f} '
            f'| max |dlog pT| {np.abs(dd["jpt"][c]).max() if c.any() else 0:.1e} axis {dd["axis"][c].max() if c.any() else 0:.1e} ({time.time() - t:.0f} s)')
    return res, per


def sharing(T):
    from interp.h41_colour_flow import ktn_batch
    F = M.F4(T); nn = (T['pt'] > 0).sum(1).astype(np.int64); out = {}
    for K in (2, 3):
        lab = ktn_batch(np.ascontiguousarray(F), nn, K); P = np.stack([np.where(lab == k, T['pt'], 0).sum(1) for k in range(K)], 1)
        S = P.sum(1); out[f's{K}'] = np.where(S > 0, 1 - ((P / np.maximum(S, 1e-300)[:, None]) ** 2).sum(1), np.nan)
    return out


def ref_feats(T):
    ok = np.all(np.isfinite(T['pt']) & np.isfinite(T['eta']) & np.isfinite(T['phi']) & np.isfinite(T['m']), 1) & ((T['pt'] > 0).sum(1) >= 3)
    g = int(np.flatnonzero(ok)[0]); T2 = {k: (np.where(ok[:, None], T[k], T[k][g:g + 1]) if k in ('pt', 'eta', 'phi', 'm') else T[k]) for k in T}
    R = M.ref_feats(T2); S = M.shapes(T2); S.update(sharing(T2))
    for k, v in S.items(): R[k] = v
    for k in list(R): R[k] = np.where(ok if R[k].ndim == 1 else ok[:, None], R[k], np.nan)
    return R


# ---------------------------------------------------------------------------------------------- samples
def hls_ref(seed, per):
    import h5py
    from jetdistill.config import DATA
    from interp.h1_irc_panel import hls_table
    Xt = np.load(Path(DATA) / 'train_ptetaphi_f16.npy', mmap_mode='r')
    with h5py.File(Path(DATA) / 'train_meta.h5', 'r') as f: yt = np.asarray(f['label'][:]).astype(int)
    if yt.ndim > 1: yt = yt.argmax(1)
    rng = np.random.default_rng(seed); idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(yt == c), per, replace=False) for c in range(5)]))
    x = np.asarray(Xt[idx], np.float32); o = np.argsort(-x[..., 0], 1, kind='stable'); x = np.take_along_axis(x, o[..., None], 1)[:, :64]
    T = {k: v for k, v in hls_table(x).items() if k != 'sgn'}
    return T, yt[idx]


def jc_rows(path, per, rng):
    from interp.part_core import read_root, cat, sub
    import uproot
    n = uproot.open(path)['tree'].num_entries; idx = np.sort(rng.choice(n, per, replace=False)); parts = []
    for a in range(0, n, 20000):
        loc = idx[(idx >= a) & (idx < a + 20000)] - a
        if len(loc): parts.append(sub(read_root(str(path), start=a, stop=min(a + 20000, n)), loc))
    log(f'JetClass {per} random of {n} rows of {path}'); return cat(parts)


def part_edits(ds):
    seed = SEEDS[ds]; t0 = time.time()
    if ds == 'hls':
        T0, y, _, _, _ = M.hls_sample(seed, per=1000, per_ref=1); classes = H.HLS_CL; ntop = y == 4; extra = None
        try:
            Tp, yp = hls_ref(seed + 10, 3000); rinfo = f'train_ptetaphi_f16 3,000 random rows per class (rng {seed + 10}), 64 hardest particles'
        except Exception as ex:
            log(f'hls train rows unavailable ({ex!r}); disjoint test rows used'); _, _, Tp, yp, _ = M.hls_sample(seed, per=1000, per_ref=3000)
            rinfo = f'n64 test, 3,000 rows per class disjoint from the evaluated jets (rng {seed + 10})'
        info = f'interp.jedi_core.sample(per_class=1000, seed={seed}); reference: {rinfo}'
    elif ds == 'jetclass':
        from interp.h1_irc_panel import jc_table
        from interp.part_core import cat, CL, PRE
        rng = np.random.default_rng(seed); r2 = np.random.default_rng(seed + 10)
        J0 = cat([jc_rows(JCV / 'val_5M' / f'{PRE[c]}_122.root', 1000, rng) for c in JC_CL])
        Jr = cat([jc_rows(JCV / 'train_100M' / f'{PRE[c]}_005.root', 3000, r2) for c in JC_CL])
        tab = lambda Jx: (lambda Tj: dict(pt=Tj['pt'], eta=Tj['eta'], phi=Tj['phi'], m=Tj['m'], ax_eta=Tj['ax_eta'], ax_phi=Tj['ax_phi'], ext=Tj['ext']))(jc_table(Jx))
        T0 = tab(J0); y = J0['label'].astype(int); Tp = tab(Jr); Tp['ext'] = {}; yp = Jr['label'].astype(int); del Jr
        classes = list(CL); ntop = y == CL.index('Tbqq')
        info = (f'JetClass val_5M <sample>_122.root, 1,000 random rows per class of {JC_CL} (rng {seed}); reference: train_100M '
                f'<sample>_005.root, 3,000 random rows per class (rng {seed + 10})')
    else:
        from interp import data_top as dt
        d = dt.DST / 'test'; ylab = np.load(d / 'label.npy'); ev = H._rows(ylab, 2000, seed)
        Fm = np.load(d / 'full.npy', mmap_mode='r'); jet = np.load(d / 'jet.npy', mmap_mode='r')
        T0 = H.table_from_full(np.asarray(Fm[ev]), np.minimum(np.asarray(jet[ev, 5]).astype(np.int64), Fm.shape[1])); y = ylab[ev].astype(int)
        dtr = dt.DST / 'train'; ytr = np.load(dtr / 'label.npy'); rf = H._rows(ytr, 3000, seed + 10)
        k3 = np.asarray(np.load(dtr / 'kin3.npy', mmap_mode='r')[rf], np.float64); jt = np.asarray(np.load(dtr / 'jet.npy', mmap_mode='r')[rf], np.float64)
        real = k3[..., 0] > 0
        Tp = dict(pt=np.where(real, k3[..., 0], 0), eta=np.where(real, jt[:, 1:2] + k3[..., 1], 0), phi=np.where(real, jt[:, 2:3] + k3[..., 2], 0),
                  m=np.zeros(k3.shape[:2]), ax_eta=jt[:, 1], ax_phi=jt[:, 2], ext={}); yp = ytr[rf].astype(int)
        classes = ['QCD', 'top']; ntop = y == 1; extra = None
        info = f'Top Landscape test: 2,000 top + 2,000 QCD random rows (rng {seed}); reference: train split 3,000 + 3,000 random rows (rng {seed + 10}), massless constituents from kin3'
    if SMOKE:
        rs = np.random.default_rng(1); ii = np.sort(np.concatenate([rs.choice(np.flatnonzero(y == c), SMOKE, replace=False) for c in sorted(set(y.tolist()))]))
        jj = np.sort(np.concatenate([rs.choice(np.flatnonzero(yp == c), 5 * SMOKE, replace=False) for c in sorted(set(yp.tolist()))]))
        cut = lambda T, i: {k: (v[i] if isinstance(v, np.ndarray) else ({a: b[i] for a, b in v.items()} if k == 'ext' else v)) for k, v in T.items()}
        T0, y, ntop, Tp, yp = cut(T0, ii), y[ii], ntop[ii], cut(Tp, jj), yp[jj]
        if ds == 'jetclass':
            from interp.part_core import sub
            J0 = sub(J0, ii)
    T0 = M.sort_first(T0)
    if ds == 'jetclass': extra = dict(J0=J0, ext=T0['ext'])
    log(f'{ds}: {len(y)} evaluated jets {np.bincount(y).tolist()}, {len(yp)} reference jets; < 3 particles: {int(((T0["pt"] > 0).sum(1) < 3).sum())} ({time.time() - t0:.0f} s)')
    out, G, diag = make_edits(T0, y, ntop, seed)
    res, per = edit_stats(T0, G, out, y, classes)
    res.update(sample=info, N=int(len(y)), lever_diag=diag, core_fallback_frac=G['core_fallback_frac'])
    OUT.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)
    (OUT / f'h69_edits_{ds}.json').write_text(json.dumps(res, indent=1, default=float)); log(f'wrote h69_edits_{ds}.json')
    D = dict(y=y.astype(np.int16), ntop=ntop)
    for nm, T in [('U', T0)] + [(e, out[e][0]) for e in EDITS]:
        for k in ('pt', 'eta', 'phi', 'm'): D[f'{nm}__{k}'] = T[k].astype(np.float64 if k in ('eta', 'phi') else np.float32)
        D[f'{nm}__ax_eta'] = T['ax_eta']; D[f'{nm}__ax_phi'] = T['ax_phi']
    np.savez_compressed(WORK / f'{ds}_tables.npz', **D)
    if extra is not None:
        with open(WORK / f'{ds}_extra.pkl', 'wb') as fh: pickle.dump(extra, fh, protocol=4)
    t = time.time(); RF = dict(ref=ref_feats(M.sort_first(Tp)), yref=yp, classes=classes, perjet=per)
    for nm, T in [('U', T0)] + [(e, out[e][0]) for e in EDITS]: RF[nm] = ref_feats(T)
    with open(WORK / f'{ds}_feats.pkl', 'wb') as fh: pickle.dump(RF, fh, protocol=4)
    log(f'reference features written ({time.time() - t:.0f} s)')


def load_tables(ds):
    d = np.load(WORK / f'{ds}_tables.npz'); y = d['y'].astype(int); extra = None; ext = {}
    if (WORK / f'{ds}_extra.pkl').exists():
        with open(WORK / f'{ds}_extra.pkl', 'rb') as fh: extra = pickle.load(fh)
        ext = extra.get('ext', {})
    Ts = {nm: dict(pt=d[f'{nm}__pt'].astype(np.float64), eta=d[f'{nm}__eta'], phi=d[f'{nm}__phi'], m=d[f'{nm}__m'].astype(np.float64),
                   ax_eta=d[f'{nm}__ax_eta'], ax_phi=d[f'{nm}__ax_phi'], ext=ext) for nm in ['U'] + EDITS}
    U = Ts['U']
    for nm in EDITS:
        T = Ts[nm]; real = T['pt'] > 0
        bad = ~np.all(np.isfinite(T['pt']) & np.isfinite(T['eta']) & np.isfinite(T['phi']) & np.isfinite(T['m']), 1)
        bad |= (np.where(real, np.abs(T['eta'] - T['ax_eta'][:, None]), 0).max(1) > 3) | (np.where(real, T['pt'], 0).max(1) > 5 * U['pt'].max(1))
        for k in ('pt', 'eta', 'phi', 'm'): T[k] = np.where(bad[:, None], U[k], T[k])
        if bad.any(): log(f'{nm}: {int(bad.sum())} diverged jets replaced by the unedited jet')
    return Ts, y, extra, d['ntop']


# ============================================================================================================== eval
def permute_flow(fl, seed):
    f2 = copy.copy(fl); rng = np.random.default_rng(seed)
    pm = lambda Ls: [(rng.permutation(K.ravel()).reshape(K.shape), rng.permutation(b.ravel()).reshape(b.shape)) for K, b in Ls]
    f2.phi, f2.F = pm(fl.phi), pm(fl.F); f2.out = pm([fl.out])[0]; return f2


def part_eval(ds):
    if ds == 'hls_jedi':
        Ts, y, _, _ = load_tables('hls')
        from interp.jedi_core import Jedi
        from jetdistill import network as N
        nets = dict(JEDI=Jedi(64, '3-feature')); orig = N.redraw_untrained; N.redraw_untrained = H.permute_weights
        try: nets['JEDI_perm0'] = Jedi(64, '3-feature', untrained=True, seed=11)
        finally: N.redraw_untrained = orig
        M.run_models(ds, Ts, {k: (lambda X, n=n: n.run(X)['logits']) for k, n in nets.items()}, H.hls_prep)
    elif ds == 'jetclass':
        from interp.h1_irc_panel import pn_logits
        from interp.part_core import ParTNetwork, inputs, build
        from interp.p1v_verify_part import fwd
        from interp.pn_net import load_particlenet
        from jetdistill.config import MODELS
        dev = os.environ.get('JETDISTILL_DEVICE', 'cpu'); Ts, y, extra, _ = load_tables('jetclass'); J0 = extra['J0']

        def prep(T):
            Jx = dict(J0); Jx['jet_eta'] = T['ax_eta'].astype(np.float32); Jx['jet_phi'] = T['ax_phi'].astype(np.float32)
            sg = np.where(Jx['jet_eta'] < 0, -1.0, 1.0)[:, None]; de, dp = H.rel(T)
            return build(Jx, T['pt'], sg * de, dp, T['m'], extra=T['ext'])
        pt_f = lambda net, n: (lambda J: fwd(net, *inputs(J, n))[1].astype(np.float64))
        models = {}
        for n in ('full', 'kin'):
            net = ParTNetwork(n); models[f'ParT_{n}'] = pt_f(net, n)
            if n == 'full':
                pk = copy.copy(net); pk.model = M.null_torch(net.model, 11); models['ParT_full_perm0'] = pt_f(pk, n)
        for n, dims in (('full', 17), ('kin', 7)):
            pn = load_particlenet(Path(MODELS).parent / 'particlenet' / f'ParticleNet_{n}.pt', dims, dev)
            models[f'PN_{n}'] = (lambda J, p=pn, n=n: pn_logits(p, J, n))
            if n == 'full':
                pnn = M.null_torch(pn, 11); models['PN_full_perm0'] = (lambda J, p=pnn: pn_logits(p, J, 'full'))
        M.run_models(ds, Ts, models, prep)
    else:
        import torch
        from interp.h4_top_wmass import WeaverNet, LNet, Flow, kin3_of
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'; Ts, y, _, _ = load_tables('top')

        def prep(T):
            pt, eta, phi, m = T['pt'], T['eta'], T['phi'], T['m']; real = pt > 0
            P4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)], -1) * real[..., None]
            return dict(P4=P4.astype(np.float32), nn=real.sum(1))
        two = lambda s: np.stack([np.zeros_like(s), s], 1); models = {}; wf = lambda net: (lambda X: two(net.score(X['P4'], X['nn'])))
        for kind, dname in (('ParT', 'part_kin'), ('PN', 'pnet_kin')):
            net = WeaverNet(kind, V / 'models/top' / dname / 'net_best_epoch_state.pt', dev); models[kind] = wf(net)
            n2 = copy.copy(net); n2.model = M.null_torch(net.model, 11); models[f'{kind}_perm0'] = wf(n2)
        ln = LNet(dev, mass64=True); order = np.random.default_rng(SEEDS['top'] + 5).permutation(len(y))
        models['LorentzNet'] = (lambda X: two(ln.score(X['P4'], X['nn'], order)))
        l2 = copy.copy(ln); l2.model = M.null_torch(ln.model, 11); models['LorentzNet_perm0'] = (lambda X: two(l2.score(X['P4'], X['nn'], order)))
        for a in ('pfn', 'efn'):
            fl = Flow(a); models[a.upper()] = (lambda X, f_=fl: two(f_.score(kin3_of(X['P4'], X['nn']))))
            fp = permute_flow(fl, 11); models[f'{a.upper()}_perm0'] = (lambda X, f_=fp: two(f_.score(kin3_of(X['P4'], X['nn']))))
        M.run_models(ds, Ts, models, prep)


# ============================================================================================================== summary
KNOWN = {'hls|JEDI': '0.8066 (H64, 5 classes)', 'jetclass|ParT_full': 'H64 test file: Wqq 0.79, Zqq 0.69, Tbqq 0.96, QCD 0.77',
         'jetclass|ParT_kin': 'H64: Wqq 0.69, Zqq 0.56, Tbqq 0.91, QCD 0.71', 'jetclass|PN_full': 'H64: Wqq 0.77, Zqq 0.64, Tbqq 0.95, QCD 0.75',
         'jetclass|PN_kin': 'H64: Wqq 0.67, Zqq 0.53, Tbqq 0.91, QCD 0.69', 'top|ParT': '0.937 (H64)', 'top|PN': '0.940 (H64)',
         'top|LorentzNet': '0.943 (H64)', 'top|PFN': '0.929 (H64)', 'top|EFN': '0.923 (H64)'}
REFS = {'MLP(m,w,pT)': ('m', 'w', 'jpt'), 'MLP(m,s,pT)': ('m', 's2', 's3', 'jpt'), 'MLP(m,D2,pT)': ('m', 'D2', 'jpt'), 'MLP(m,tau21,pT)': ('m', 'tau21', 'jpt')}


def ref_logits(RF, used):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    lab = {c: i for i, c in enumerate(used)}; keep = np.isin(RF['yref'], used); yr = np.array([lab[v] for v in RF['yref'][keep]])
    X = lambda F, keys: np.nan_to_num(np.stack([np.log(np.maximum(F[k], 1e-6)) for k in keys], 1), nan=0.0, posinf=20.0, neginf=-20.0)
    out = {}
    for nm, keys in REFS.items():
        Xr = X(RF['ref'], keys)[keep]; ok = np.all(np.isfinite(Xr), 1)
        clf = make_pipeline(StandardScaler(), MLPClassifier((32, 32), max_iter=400, early_stopping=True, random_state=0)).fit(Xr[ok], yr[ok])
        out[nm] = {e: clf.predict_log_proba(X(RF[e], keys)) for e in ['U'] + EDITS}; log(f'reference {nm} trained on {int(ok.sum())} jets')
    E = lambda F: np.nan_to_num(F['efp'], nan=0.0, posinf=0.0, neginf=0.0)
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(E(RF['ref'])[keep], yr)
    out['GBDT(EFP)'] = {e: np.log(np.clip(clf.predict_proba(E(RF[e])), 1e-12, 1)) for e in ['U'] + EDITS}
    return out


def perm_p(d, rng, n=NPERM):
    d = np.asarray(d, float); m = len(d)
    if m < 2: return 1.0
    S = rng.choice([-1.0, 1.0], (n, m)); null = np.abs((S * d).mean(1)); return float((1 + (null >= abs(d.mean()) - 1e-15).sum()) / (n + 1))


def class_stats(Ds, conv, thrcalls, rng):
    """Ds: {edit: per-jet Delta s} (jets of one class), conv: {edit: bool}; thrcalls: {edit: bool call (edited)} and 'U'"""
    r = dict(edits={}, phi={})
    for e in EDITS:
        if e == 'crot_dm': continue
        sel = conv[e] & conv['crot_dm']; n = int(sel.sum())
        if n < 20: r['edits'][e] = dict(n=n); continue
        A, R0 = np.abs(Ds[e][sel]), np.abs(Ds['crot_dm'][sel]); I = rng.integers(0, n, (NB, n))
        Ab, Rb = A[I].mean(1), R0[I].mean(1)
        with np.errstate(divide='ignore', invalid='ignore'): Rr = Ab / Rb
        ch_e = thrcalls[e][sel] != thrcalls['U'][sel]; ch_c = thrcalls['crot_dm'][sel] != thrcalls['U'][sel]
        r['edits'][e] = dict(n=n, mean_abs=float(A.mean()), mean_abs_ci=np.percentile(Ab, [2.5, 97.5]).tolist(),
                             mean_abs_crot_dm=float(R0.mean()), mean_abs_crot_dm_ci=np.percentile(Rb, [2.5, 97.5]).tolist(),
                             R=float(A.mean() / R0.mean()) if R0.mean() > 0 else None, R_ci=np.nanpercentile(Rr, [2.5, 97.5]).tolist(),
                             mean_signed=float(Ds[e][sel].mean()), mean_signed_ci=np.percentile(Ds[e][sel][I].mean(1), [2.5, 97.5]).tolist(),
                             diff=float((A - R0).mean()), diff_ci=np.percentile((A - R0)[I].mean(1), [2.5, 97.5]).tolist(), p_perm=perm_p(A - R0, rng),
                             p_signflip_asym=float(math.erfc(abs((A - R0).mean()) / max(math.sqrt(((A - R0) ** 2).sum()) / n, 1e-300) / math.sqrt(2))),
                             calls_changed=float(ch_e.mean()), calls_changed_crot_dm=float(ch_c.mean()), mcnemar=H.mcnemar(ch_e, ch_c))
    n = len(Ds['crot_dm']); I = rng.integers(0, n, (NB, n))
    for sz, grp in GROUPS.items():
        Mx, Mb = {}, {}
        for X, es in grp.items():
            vals, boots = [], []
            for e in es:
                c = conv[e]
                if c.sum() < 20: continue
                A = np.where(c, np.abs(Ds[e]), 0); vals.append(A.sum() / c.sum())
                with np.errstate(divide='ignore', invalid='ignore'): boots.append(A[I].sum(1) / c[I].sum(1))
            if vals: Mx[X] = float(np.mean(vals)); Mb[X] = np.nanmean(np.stack(boots), 0)
        c = conv['crot_dm']; A = np.where(c, np.abs(Ds['crot_dm']), 0)
        with np.errstate(divide='ignore', invalid='ignore'): Cb = A[I].sum(1) / c[I].sum(1)
        Cm = float(A.sum() / max(c.sum(), 1))
        g = dict(M=Mx, M_ci={X: np.nanpercentile(Mb[X], [2.5, 97.5]).tolist() for X in Mx}, M_crot_dm=Cm, M_crot_dm_ci=np.nanpercentile(Cb, [2.5, 97.5]).tolist(),
                 R={X: (Mx[X] / Cm if Cm > 0 else None) for X in Mx}, phi={})
        with np.errstate(divide='ignore', invalid='ignore'):
            g['R_ci'] = {X: np.nanpercentile(Mb[X] / Cb, [2.5, 97.5]).tolist() for X in Mx}
            for ps, keys in PHI_SETS.items():
                if not all(X in Mx for X in keys): continue
                tot = sum(Mx[X] for X in keys); totb = sum(Mb[X] for X in keys)
                if not tot > 0: continue
                exc = {X: max(Mx[X] - Cm, 0) for X in keys}; te = sum(exc.values()); excb = {X: np.maximum(Mb[X] - Cb, 0) for X in keys}; teb = sum(excb.values())
                g['phi'][ps] = dict(phi={X: Mx[X] / tot for X in keys}, phi_ci={X: np.nanpercentile(Mb[X] / totb, [2.5, 97.5]).tolist() for X in keys},
                                    phi_excess={X: (exc[X] / te if te > 0 else None) for X in keys},
                                    phi_excess_ci={X: np.nanpercentile(excb[X] / teb, [2.5, 97.5]).tolist() for X in keys})
        r['phi'][sz] = g
    return r


def summarize_ds(ds, files, classes_eval):
    with open(WORK / f'{ds}_feats.pkl', 'rb') as fh: RF = pickle.load(fh)
    d = np.load(WORK / f'{ds}_tables.npz'); y = d['y'].astype(int); classes = RF['classes']; per = RF['perjet']
    conv = {e: per[f'{e}__conv'] for e in EDITS}; rng = np.random.default_rng(SEEDS[ds] + 7)
    Lall = {}
    for f in files:
        if (WORK / f).exists():
            z = np.load(WORK / f)
            for k in z.files: mn, e = k.split('__'); Lall.setdefault(mn, {})[e] = z[k]
        else: log(f'missing {f}')
    used = sorted(set(y.tolist())); R = ref_logits(RF, used); yl = np.array([used.index(v) for v in y])
    out = dict(models={}, classes=[classes[k] for k in classes_eval])
    allm = {**{m: ('null' if 'perm' in m else 'trained', Lg) for m, Lg in Lall.items()}, **{m: ('ref', Lg) for m, Lg in R.items()}}
    DS = {}
    for mn, (role, Lg) in allm.items():
        yy = yl if role == 'ref' else y; lo = {e: M.logodds_all(Lg[e]) for e in Lg}; c = Lg['U'].argmax(1)
        acc = float((c == yy).mean())
        r = dict(role=role, acc=acc, acc_ci=H.boot_frac(c == yy, rng), per_class_acc={classes[k]: float((c[y == k] == yy[y == k]).mean()) for k in classes_eval},
                 known=KNOWN.get(f'{ds}|{mn}'), cls={})
        DS[mn] = {}
        for k in classes_eval:
            kk = used.index(k) if role == 'ref' else k; sel = y == k
            Ds = {e: (lo[e][:, kk] - lo['U'][:, kk])[sel] for e in EDITS}; DS[mn][classes[k]] = Ds
            thr = float(np.median(lo['U'][sel, kk])); calls = {e: lo[e][sel, kk] > thr for e in ['U'] + EDITS}
            r['cls'][classes[k]] = class_stats(Ds, {e: conv[e][sel] for e in EDITS}, calls, rng); r['cls'][classes[k]]['threshold_eps0.5'] = thr
        out['models'][mn] = r
        ph = {cl: st['phi'].get(10, {}).get('R', {}) for cl, st in r['cls'].items()}
        log(f'{ds} {mn} [{role}] acc {acc:.4f} (known {r["known"]}) per class {r["per_class_acc"]} | R10 ' +
            '; '.join(f'{cl}: ' + (' '.join(f'{X} {v:.2f}' for X, v in p.items() if v is not None) if p else '-') for cl, p in ph.items()))
    # per-jet agreement of signed Delta s between each network and each reference (pooled over the six 10 % edits)
    refs = [m for m, (ro, _) in allm.items() if ro == 'ref']
    for mn, (role, _) in allm.items():
        if role == 'ref': continue
        for k in classes_eval:
            cl = classes[k]; sel = y == k; agg = {}
            for rn in refs:
                a, b = [], []
                for e in MAIN10:
                    cc = conv[e][sel]; a.append(DS[mn][cl][e][cc]); b.append(DS[rn][cl][e][cc])
                a, b = np.concatenate(a), np.concatenate(b)
                if len(a) < 20 or a.std() == 0 or b.std() == 0: continue
                I = rng.integers(0, len(a), (NB, len(a)))
                cb = np.array([np.corrcoef(a[i], b[i])[0, 1] for i in I[:300]])
                agg[rn] = dict(n=int(len(a)), corr=float(np.corrcoef(a, b)[0, 1]), corr_ci=np.percentile(cb, [2.5, 97.5]).tolist())
            out['models'][mn]['cls'][cl]['corr_refs'] = agg
            # phi-vector distance to every reference (10 % and 5 % sets)
            st = out['models'][mn]['cls'][cl]['phi']; dist = {}
            for sz in (10, 5):
                for ps in PHI_SETS:
                    if ps not in st.get(sz, {}).get('phi', {}): continue
                    a_ = st[sz]['phi'][ps]; dk = f'{sz}|{ps}'; dist[dk] = {}
                    for rn in refs:
                        rs = out['models'][rn]['cls'][cl]['phi'].get(sz, {}).get('phi', {}).get(ps)
                        if rs is None: continue
                        dist[dk][rn] = dict(L1=float(sum(abs(a_['phi'][X] - rs['phi'][X]) for X in a_['phi'])), ref_phi=rs['phi'],
                                            inside={X: bool(a_['phi_ci'][X][0] <= rs['phi'][X] <= a_['phi_ci'][X][1]) for X in a_['phi']})
            out['models'][mn]['cls'][cl]['ref_distance'] = dist
    return out


def part_summary():
    S = dict(datasets={}, holm={})
    cfg = dict(hls=(['hls_jedi_logits.npz'], [2, 3, 4]), jetclass=(['jetclass_logits.npz'], None), top=(['top_logits.npz'], [0, 1]))
    for ds, (files, ce) in cfg.items():
        if not (WORK / f'{ds}_feats.pkl').exists(): log(f'{ds}: no features'); continue
        if ce is None:
            with open(WORK / f'{ds}_feats.pkl', 'rb') as fh: cls_ = pickle.load(fh)['classes']
            ce = [cls_.index(c) for c in JC_CL]
        S['datasets'][ds] = summarize_ds(ds, files, ce); S['datasets'][ds]['edits'] = json.loads((OUT / f'h69_edits_{ds}.json').read_text())
    # the sign-flip permutation p with 2,000 flips has a floor of 1/2001; Holm over ~400 tests uses its large-sample (normal) form
    for fam, key in (('signflip_vs_crot_dm', 'p_signflip_asym'), ('perm2000_vs_crot_dm', 'p_perm'), ('mcnemar_vs_crot_dm', 'mcnemar')):
        keys, ps = [], []
        for ds, D in S['datasets'].items():
            for mn, r in D['models'].items():
                if r['role'] != 'trained': continue
                for cl, st in r['cls'].items():
                    for e, v in st['edits'].items():
                        if key not in v: continue
                        keys.append(f'{ds}|{mn}|{cl}|{e}'); ps.append(v[key]['p'] if key == 'mcnemar' else v[key])
        hp = H.holm(ps) if ps else []
        S['holm'][fam] = dict(n_tests=len(ps), n_below_0p05=int(np.sum(np.array(hp) < 0.05)), p=dict(zip(keys, [float(v) for v in hp])))
        log(f'Holm {fam}: {S["holm"][fam]["n_below_0p05"]} of {len(ps)} below 0.05')
    (OUT / 'h69_summary.json').write_text(json.dumps(S, indent=1, default=float)); log('wrote h69_summary.json')
    try: fig_response(S)
    except Exception as ex: log(f'response figure failed: {ex!r}')
    try: fig_jet()
    except Exception as ex: log(f'jet figure failed: {ex!r}')


# ============================================================================================================== figures
def fig_jet():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    Ts, y, _, ntop = load_tables('top')
    with open(WORK / 'top_feats.pkl', 'rb') as fh: per = pickle.load(fh)['perjet']
    G = geometry69(Ts['U'], ntop.astype(bool))
    opts = [('E_w', ['Ew-10', 'Ew+10', 'Ew-5', 'Ew+5']), ('E_s', ['Es-10', 'Es+10', 'Es-5', 'Es+5']), ('E_D2', ['ED2+10', 'ED2-10', 'ED2+5', 'ED2-5']),
            ('E_t', ['Et+10', 'Et-10']), ('crot_dm', ['crot_dm'])]
    desc = {'Ew': 'E_w: width w {} (m, pT, s fixed)', 'Es': 'E_s: prong sharing s {} (m, pT, w fixed)', 'ED2': 'E_D2: D2 {} (m, pT, s fixed)',
            'Et': 'E_τ21: τ21 {} (m, pT, s, D2 fixed)'}
    ok = (y == 1) & ((Ts['U']['pt'] > 0).sum(1) >= 25) & (np.abs(np.exp(G['o0']['m']) - 172) < 25)
    score = sum(np.any([per[f'{e}__conv'] for e in es], 0).astype(int) * (10 if es[0].endswith('10') else 1) for _, es in opts)
    w = np.flatnonzero(ok)
    if not len(w): log('no top jet for the figure'); return
    j = w[np.argmax(score[w])]
    show = [('U', 'unedited top jet')]
    for nm, es in opts:
        e = next((e for e in es if per[f'{e}__conv'][j]), None)
        if e is None: show.append((None, f'{nm}: no exact solution on this jet')); continue
        if e == 'crot_dm': show.append((e, 'control crot_dm: prongs rotated\n(m, pT, w, s fixed)')); continue
        k, sz = (e[:3], e[3:]) if e.startswith('ED2') else (e[:2], e[2:])
        show.append((e, desc[k].format(sz.replace('-', '−') + ' %').replace(' (', '\n(')))
    cols = ['#0072B2', '#D55E00', '#009E73']
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 9.4), sharex=True, sharey=True); ax = ax.ravel()
    sub = lambda T: {k: (v[j:j + 1] if isinstance(v, np.ndarray) else v) for k, v in T.items()}
    Gj = {k: (v[j:j + 1] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(y) else v) for k, v in G.items()}
    for a, (e, ttl) in zip(ax, show):
        if e is None: a.set_title(ttl, fontsize=8.6); continue
        T = sub(Ts[e]); o = observe(T, Gj, ('m', 'w', 's', 'D2', 'tau21')); real = T['pt'][0] > 0; lab = G['lab'][j]; core = G['core'][j]
        de = T['eta'][0] - G['ee'][j]; dp = H.wrap(T['phi'][0] - G['ep'][j]); sz = 8 + 420 * T['pt'][0] / Ts['U']['pt'][j].max()
        for k in range(3):
            mk = real & (lab == k)
            if not mk.any(): continue
            a.scatter(de[mk & core], dp[mk & core], s=sz[mk & core], c=cols[k], alpha=0.75, edgecolors='k', linewidths=0.3)
            a.scatter(de[mk & ~core], dp[mk & ~core], s=sz[mk & ~core], facecolors='none', edgecolors=cols[k], linewidths=1.0)
            pk = T['pt'][0][mk]; cx, cy = (pk * de[mk]).sum() / pk.sum(), (pk * dp[mk]).sum() / pk.sum()
            a.add_patch(Circle((cx, cy), RCORE, fill=False, ls='--', lw=0.8, color=cols[k]))
        sh = np.exp(o['s'][0])
        a.set_title(f'{ttl}\nm = {np.exp(o["m"][0]):.1f} GeV, w = {np.exp(o["w"][0]):.4f}, s = {sh:.3f}, D2 = {np.exp(o["D2"][0]):.2f}, τ21 = {np.exp(o["tau21"][0]):.3f}', fontsize=8.6)
        a.axhline(0, color='0.85', lw=0.5); a.axvline(0, color='0.85', lw=0.5); a.set_aspect('equal')
    for a in ax[3:]: a.set_xlabel('Δη to jet axis')
    for a in ax[::3]: a.set_ylabel('Δφ to jet axis [rad]')
    ax[0].set_xlim(-0.75, 0.75); ax[0].set_ylim(-0.75, 0.75)
    fig.suptitle(f'Shape edits at fixed jet mass and jet pT on one Top Landscape top jet (pT = {G["jpt0"][j]:.0f} GeV). Three kT prongs in three colours;\n'
                 f'filled markers = prong core (within {RCORE} of the prong axis, dashed circle), open markers = prong edge; marker area ∝ particle pT', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0); fig.savefig(OUT / 'h69_jet.png', dpi=120); log(f'wrote {OUT / "h69_jet.png"} (jet row {j})')


def fig_response(S):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    name = {'hls': 'hls4ml', 'jetclass': 'JetClass', 'top': 'Top'}; sig = {'hls': ['W', 'Z', 't'], 'jetclass': ['Wqq', 'Zqq', 'Tbqq', 'Hbb'], 'top': ['top']}
    rows = []
    for ds, D in S['datasets'].items():
        for role in ('trained', 'null'):
            for mn, r in D['models'].items():
                if r['role'] != role: continue
                for c in sig[ds]:
                    if c in r['cls']: rows.append((ds, mn, role, c))
    X = [('w', 'E_w: width ±5 % (m, pT, s fixed)', 5), ('s', 'E_s: prong sharing ±5 % (m, pT, w fixed)', 5), ('D2', 'E_D2: D2 ±5 % (m, pT, s fixed)', 5),
         ('tau21', 'E_τ21: τ21 ±10 % (m, pT, s, D2 fixed)', 10)]
    cols = {'w': '#0072B2', 's': '#D55E00', 'D2': '#009E73', 'tau21': '#CC79A7'}
    fig, ax = plt.subplots(figsize=(8.8, 0.24 * len(rows) + 2.0))
    for i, (ds, mn, role, c) in enumerate(rows):
        for dy, (x, lb, sz) in zip((-0.3, -0.1, 0.1, 0.3), X):
            ph_ = S['datasets'][ds]['models'][mn]['cls'][c]['phi']; st = ph_.get(sz, ph_.get(str(sz), {}))
            v = st.get('R', {}).get(x); ci = st.get('R_ci', {}).get(x)
            if v is None or not v > 0: continue
            ax.errorbar(v, i + dy, xerr=[[max(v - ci[0], 0)], [max(ci[1] - v, 0)]], fmt='o', ms=3.4, lw=0.9, color=cols[x], label=lb if i == 0 else None)
    ax.axvline(1, color='k', lw=0.8, ls='--'); ax.set_xscale('log'); ax.set_xlim(0.2, 20)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f'{name[d]} {"ref " if ro == "ref" else "null " if ro == "null" else ""}{m} | {c}' for d, m, ro, c in rows], fontsize=6.6)
    ax.invert_yaxis(); ax.set_xlabel('R = mean |Δ own-class log-odds| under the edit / under the displacement-matched prong rotation crot_dm\n'
                                     '(R = 1: the edit moves the score no more than the control; 95 % bootstrap over jets; missing dot: < 20 jets solved)', fontsize=8.5)
    ax.set_title('Trained taggers respond to prong sharing, D2 and τ21 edits at fixed jet mass and pT\n(rows marked null: same architecture with permuted weights)', fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc='lower right'); fig.tight_layout(); fig.savefig(OUT / 'h69_response.png', dpi=125); log(f'wrote {OUT / "h69_response.png"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', required=True, choices=['edits', 'eval', 'summary', 'figs', 'figr'])
    ap.add_argument('--ds', default=None, choices=['hls', 'jetclass', 'top', 'hls_jedi'])
    a = ap.parse_args(); log(f'H69 part {a.part} ds {a.ds}'); WORK.mkdir(parents=True, exist_ok=True)
    if a.part == 'edits': part_edits(a.ds)
    elif a.part == 'eval': part_eval(a.ds)
    elif a.part == 'summary': part_summary()
    elif a.part == 'figr': fig_response(json.loads((OUT / 'h69_summary.json').read_text()))
    else: fig_jet()
    log('done')
