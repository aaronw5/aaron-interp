"""E42x (loop A, round 21, stage 4; claim A20t; parent E42v): does JEDI-linear's top score use the W-pair mass when the
subjet shapes are held rigid?

E42v found that E42's W edit narrows the subjets (common internal-spread factor h, median 0.78 at 60 GeV) and lowers tau21,
while its control does not, and that a classifier that never sees the pair mass gives the same W - N.  Here every edit
moves whole subjets rigidly with h = 1 (no subjet is resized), so subjet widths are fixed by construction.

PLAN (fixed before the first run)
Sample: jedi_core.sample(5000, 4421) (25,000 jets; fresh seed vs E42 4201/4202 and E42v 4302; overlap not removed, counted).
  Massless four-vectors (eta = deta, phi = dphi of the stored arrays); three exclusive-kT subjets (E-scheme; h4_top_wmass);
  W pair (a, b) = the pair with mass closest to 80.4 GeV, c = the third subjet ("b-subjet").  Top window 105 < mSD < 210 GeV.
  Sets: T60 / T100 = tops with 65 < mW < 95 GeV, W pair moved to 60 / 100 GeV; T80 (onto-peak) = all tops with
  60 <= mW < 70 or 90 < mW <= 100 GeV, W pair moved to 80.4 GeV.
Edits (every subjet translated rigidly in (eta, phi); pT of every particle, the subjet masses, widths and pT^D unchanged):
  Wr  (W-rigid) the W-pair opening is scaled by f about the pair's pT-weighted centroid cW (bisection: m_ab = t); c is placed
      at cW + g R(theta) (r_c - cW): g (common rescale of c's separation) restores the jet mass for each theta; theta = 0 is
      used when tau32 is already within 0.5 % (solver, Jet.tau32), otherwise the smallest |theta| (scan 0.05 rad steps to
      0.8 rad, then bisection) that returns tau32 to its unedited value.  (A pure g rescale fixes one constraint; the jet
      mass and tau32 together need the second coordinate theta.)
  Nf  (N-matched, same opening change) the non-W pair (d, c), d = harder W prong (softer if it fails), gets its opening
      scaled by the same factor f: c is placed on the circle of radius f |r_c - r_d| about d, at the azimuth that keeps the
      jet mass; the W pair is not moved, so m_W is unchanged.  With m and m_W fixed and rigid subjets the three-subjet
      triangle has one free coordinate, so tau32 cannot also be restored (the only solution that also keeps tau32 is the
      identity); tau32, tau21 and mSD of Nf are measured and handled as side effects (reference classifiers, regression,
      matched subset).
  ND  (N-matched, same displacement) as Nf with the opening factor f' chosen (bisection) so that the pT-weighted mean
      displacement of the constituents relative to the jet's pT-weighted centroid equals that of Wr.
  Ws / Ns  (W-scaled) the original E42 edit and its control (h4_verify.edit_set, h free), T60 / T100 only, as a reference.
Acceptance (reclustered observables, top_obs): Wr |dm|/m < 1 %, |dtau32|/tau32 < 1 %, |m_W - t| < 5 GeV (T80: < 2 GeV);
  Nf, ND |dm|/m < 1 %, |m_W - m_W0| < 2 GeV; Ws/Ns as E42.  Analysis set per set = jets with Wr and Nf accepted (ND on the
  subset where it is also accepted); Ws-Ns on its own accepted set.
Network input: every variant rebuilt about its own E-scheme axis (as E42); accuracy check on the stored inputs.
Side effects per edit (variant - unedited, recomputed after reclustering): m, mSD, tau21, tau32, m_W, three subjet widths
  (pT-weighted mean Delta R of a subjet's constituents to its axis, subjets sorted by pT), pT^D, displacement D.
References (fit on the 25,000 natural sample jets, top vs the other four classes, 2-fold cross-fit: a jet's variants are
  scored by the model not trained on it): BLIND = HistGBDT(m, mSD, tau21, tau32, n, three subjet widths, pT^D) (no pair
  mass); AWARE = BLIND + the three pair masses (m_W-candidate, the other two sorted) (positive control).
Networks: official 3-feature n64 (jedi_core.Jedi) + the 7 E39 EDIT_NETS (e33 Model) + one untrained redraw (seed 0).
Score: top log-odds s = L_t - logsumexp(other four).  Per jet D_v = s_v - s_id.  Contrasts: W - N for (Wr, Nf), (Wr, ND),
  (Ws, Ns); mean, 95 % jet bootstrap CI (2,000), sign-flip permutation p (10,000); McNemar (exact) on top calls W vs N.
  H0: (W - N)_net = (W - N)_BLIND (raw log-odds; per-jet paired difference, bootstrap CI, permutation p); also with BLIND
  scaled by k_DIL (net's / BLIND's antisymmetric response to a real mass change: all angles x 1.08 vs x 0.92).  H1: the
  net's W - N is further in the m_W-reader direction (negative at T60 / T100, positive at T80).  Holm over 8 trained
  networks x 3 sets, separately for Wr - ND (primary, changed after the smoke run: Nf's opening-matched control moved tau32
  by +0.07 and mSD by +10 GeV relative to Wr at 60 GeV, ND by -0.01 and +0.7 GeV) and Wr - Nf; theta search widened to
  +-pi after the smoke run (Wr solved for 40 / 186 smoke tops at 60 GeV).  Per-jet regression of (D_Wr - D_Nf) on the W - N differences of
  tau21, tau32, mSD, D (intercept with bootstrap CI) and a matched subset (|dtau21| < 0.02, |dtau32| < 0.02, |dmSD| < 3 GeV
  between Wr and Nf).
Output: $AI_RESULTS/jedi/e42x.json, e42x_edit.json, e42x_scores.npz, e42x_wpair_rigid.png.
Run: AI_PY=conda ./job.sh e42x-run 'python -m interp.e42x_top_wpair_matched_jedi --part edit &&
     /j-jepa-vol/phat-jet-aaron/env/bin/python -m interp.e42x_top_wpair_matched_jedi --part jedi'  (E42X_SMOKE=<n> per class)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

os.environ.setdefault('KERAS_BACKEND', 'jax')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
T00 = time.time()
SMOKE = int(os.environ.get('E42X_SMOKE', '0'))
PER = SMOKE or 5000
SEED = 4421
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
TAG = 'e42x_smoke' if SMOKE else 'e42x'
TMP = Path(os.environ.get('E42X_TMP', '/tmp/e42x'))
SETS = ('T60', 'T100', 'T80')
TARGET = {'T60': 60.0, 'T100': 100.0, 'T80': 80.4}
TOLW = {'T60': 5.0, 'T100': 5.0, 'T80': 2.0}
DIRN = {'T60': -1, 'T100': -1, 'T80': 1}       # sign of W - N expected from an m_W reader
LAMS = (0.92, 1.08)
OBSK = ('m', 'mSD', 'tau21', 'tau32', 'mW', 'n', 'w1', 'w2', 'w3', 'ptd', 'mWc', 'mo1', 'mo2')
FBLIND = ('m', 'mSD', 'tau21', 'tau32', 'n', 'w1', 'w2', 'w3', 'ptd')
FAWARE = FBLIND + ('mWc', 'mo1', 'mo2')
EDIT_NETS = ['official_n32', 'perminv_n64', 'hgq150_s0', 'hgq150_s1', 'hgq150_s2', 'float_s0', 'float_s1']
PAIRS = {'T60': (('Wr', 'Nf'), ('Wr', 'ND'), ('Ws', 'Ns')), 'T100': (('Wr', 'Nf'), ('Wr', 'ND'), ('Ws', 'Ns')),
         'T80': (('Wr', 'Nf'), ('Wr', 'ND'))}


def log(*a):
    print(f'[{time.time() - T00:7.0f}s]', *a, flush=True)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else round(float(o), 6)
    return o


# ================================================================================================ edits (conda env)
def minv(v):
    return float(np.sqrt(max(v[3] ** 2 - v[0] ** 2 - v[1] ** 2 - v[2] ** 2, 0.0)))


class Fast:
    """closed-form four-vector of a rigidly translated subjet of massless constituents:
    shift (de, dp): px = c A - s B, py = s A + c B, pz = ch C + sh D, E = ch D + sh C with A = sum pt cos phi, B = sum pt sin phi,
    C = sum pt sinh eta, D = sum pt cosh eta"""

    def __init__(self, jt):
        self.S = []
        for s in jt.sel:
            pt = jt.pt[s]; eta = jt.eJ + jt.x[s]; phi = jt.pJ + jt.y[s]
            self.S.append(((pt * np.cos(phi)).sum(), (pt * np.sin(phi)).sum(), (pt * np.sinh(eta)).sum(), (pt * np.cosh(eta)).sum()))

    def v(self, k, d):
        A, B, C, D = self.S[k]; c, s = np.cos(d[1]), np.sin(d[1]); ch, sh = np.cosh(d[0]), np.sinh(d[0])
        return np.array([c * A - s * B, s * A + c * B, ch * C + sh * D, ch * D + sh * C])


def disp(jt, s):
    """pT-weighted mean displacement of the constituents relative to the pT-weighted mean shift (axis-independent)"""
    s = np.asarray(s, float); w = jt.ptk / jt.ptk.sum(); sb = (w[:, None] * s).sum(0)
    return float((w * np.hypot(*(s - sb).T)).sum())


def bisect(fn, lo, hi, tol, it=60):
    flo, fhi = fn(lo), fn(hi)
    if flo is None or fhi is None or np.sign(flo) == np.sign(fhi): return None
    for _ in range(it):
        mid = 0.5 * (lo + hi); fm = fn(mid)
        if fm is None: return None
        if abs(fm) < tol: return mid
        if np.sign(fm) == np.sign(flo): lo, flo = mid, fm
        else: hi = mid
    return 0.5 * (lo + hi)


def rot(u, th):
    c, s = np.cos(th), np.sin(th); return np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]])


def w_rigid(jt, fj, t, tau0, m0, a, b, c):
    r = jt.r; pa, pb = jt.ptk[a], jt.ptk[b]; cW = (pa * r[a] + pb * r[b]) / (pa + pb); u = r[c] - cW

    def sab(f): return (f - 1) * (r[a] - cW), (f - 1) * (r[b] - cW)
    f = bisect(lambda f: minv(fj.v(a, sab(f)[0]) + fj.v(b, sab(f)[1])) - t, 0.05, 4.0, 0.01)
    if f is None: return None
    s_a, s_b = sab(f); Vab = fj.v(a, s_a) + fj.v(b, s_b)

    def shifts(g, th):
        s = np.zeros((3, 2)); s[a] = s_a; s[b] = s_b; s[c] = cW + g * rot(u, th) - r[c]; return s

    def gsol(th): return bisect(lambda g: minv(Vab + fj.v(c, cW + g * rot(u, th) - r[c])) - m0, 0.0, 6.0, 0.01)
    cache = {}

    def T(th):
        if th not in cache:
            g = gsol(th); cache[th] = None if g is None else (jt.tau32(shifts(g, th)) - tau0, g)
        return cache[th]
    tol = 0.005 * tau0
    t0 = T(0.0)
    if t0 is not None and abs(t0[0]) < tol: return dict(s=shifts(t0[1], 0.0), f=f, g=t0[1], th=0.0)
    cands = []                       # brackets / near-roots, tried in order of |theta|
    for side in (1, -1):
        prev = 0.0
        for k in range(1, 63):
            th = side * 0.05 * k; cur = T(th)
            if cur is None: continue
            pv = T(prev)
            if abs(cur[0]) < tol: cands.append((abs(th), th, th))
            elif pv is not None and np.sign(cur[0]) != np.sign(pv[0]): cands.append((abs(th), prev, th))
            prev = th
    for _, lo, hi in sorted(cands)[:6]:
        th = lo if lo == hi else bisect(lambda th: None if T(th) is None else T(th)[0], lo, hi, tol, it=12)
        if th is None or T(th) is None or abs(T(th)[0]) > 2 * tol: continue     # sign change at a tau32 jump, not a root
        g = T(th)[1]; return dict(s=shifts(g, th), f=f, g=g, th=th)
    return None


def n_place(jt, fj, d, e, c, fac, m0):
    """c on the circle of radius fac |r_c - r_d| about d, azimuth nearest the original that keeps the jet mass"""
    r = jt.r; u = r[c] - r[d]; rho = fac * np.hypot(*u); ph0 = np.arctan2(u[1], u[0]); Vde = fj.v(d, (0, 0)) + fj.v(e, (0, 0))

    def pos(dp): return r[d] + rho * np.array([np.cos(ph0 + dp), np.sin(ph0 + dp)]) - r[c]

    def F(dp): return minv(Vde + fj.v(c, pos(dp))) - m0
    f0 = F(0.0)
    if abs(f0) < 0.01: dp = 0.0
    else:
        dp = None
        for k in range(1, 80):
            for side in (1, -1):
                x1 = side * 0.02 * k; x0 = side * 0.02 * (k - 1)
                if np.sign(F(x1)) != np.sign(F(x0)):
                    dp = bisect(F, min(x0, x1), max(x0, x1), 0.01); break
            if dp is not None: break
        if dp is None: return None
    s = np.zeros((3, 2)); s[c] = pos(dp); return s


def n_matched(jt, fj, a, b, c, fac, m0, D_target=None):
    for d in sorted((a, b), key=lambda k: -jt.ptk[k]):
        e = a + b - d
        if D_target is None:
            s = n_place(jt, fj, d, e, c, fac, m0)
            if s is not None: return dict(s=s, fac=fac, d=d)
            continue
        G = lambda q: None if (ss := n_place(jt, fj, d, e, c, q, m0)) is None else disp(jt, ss) - D_target
        for k in (2.0, 4.0, 8.0):
            q_ext = float(np.clip(1 + k * (fac - 1), 0.05, 4.0)); g_ext = G(q_ext)
            if g_ext is not None and g_ext > 0: break
        else:
            continue
        q = bisect(lambda q: -D_target if q == 1.0 else G(q), 1.0, q_ext, 0.02 * D_target + 1e-5, it=40)
        if q is None: continue
        s = n_place(jt, fj, d, e, c, q, m0)
        if s is not None: return dict(s=s, fac=q, d=d)
    return None


def edit_one(P, lab, targets):
    from interp import h4_top_wmass as H
    jt = H.Jet(P, lab); fj = Fast(jt); tau0 = jt.tau32(np.zeros((3, 2)))
    pm0, m0 = jt.pair_masses(); w = min(pm0, key=lambda p: abs(pm0[p] - 80.4)); a, b = w; c = 3 - a - b
    out = dict(mW0=pm0[w], m0=m0, tau0=tau0)
    for t in targets:
        R = {}
        W = w_rigid(jt, fj, t, tau0, m0, a, b, c)
        if W is not None:
            DW = disp(jt, W['s']); R['Wr'] = dict(P=jt.build(W['s'])[0].astype(np.float32), f=W['f'], g=W['g'], th=W['th'], D=DW)
            for nm, Dt in (('Nf', None), ('ND', DW)):
                N = n_matched(jt, fj, a, b, c, W['f'], m0, Dt)
                if N is not None:
                    R[nm] = dict(P=jt.build(N['s'])[0].astype(np.float32), f=N['fac'], D=disp(jt, N['s']), d=int(N['d']))
        out[t] = R
    return out


def subjet_feats(P4, nn):
    """three exclusive-kT subjets (E-scheme) of each jet: widths (pT-weighted mean dR to the subjet axis, subjets sorted by pT),
    the three pair masses (m_W candidate = closest to 80.4, the other two sorted), pT^D"""
    from interp import h4_top_wmass as H
    J = len(nn); F = np.zeros((J, P4.shape[1], 4), np.float64); F[:] = P4; LAB = H.kt3_batch(F, nn)
    o = {k: np.zeros(J) for k in ('w1', 'w2', 'w3', 'ptd', 'mWc', 'mo1', 'mo2')}
    for j in range(J):
        n = nn[j]; P = F[j, :n]; pt = np.hypot(P[:, 0], P[:, 1]); o['ptd'][j] = np.sqrt((pt ** 2).sum()) / pt.sum()
        if n < 3: continue
        eta = np.arcsinh(P[:, 2] / np.maximum(pt, 1e-12)); phi = np.arctan2(P[:, 1], P[:, 0]); lab = LAB[j, :n]
        V = [P[lab == k].sum(0) for k in range(3)]; ptk = [np.hypot(v[0], v[1]) for v in V]; ws = []
        for k in range(3):
            s = lab == k; ek = np.arcsinh(V[k][2] / max(ptk[k], 1e-12)); pk = np.arctan2(V[k][1], V[k][0])
            dr = np.hypot(eta[s] - ek, (phi[s] - pk + np.pi) % (2 * np.pi) - np.pi); ws.append((pt[s] * dr).sum() / max(pt[s].sum(), 1e-12))
        order = np.argsort(ptk)[::-1]; o['w1'][j], o['w2'][j], o['w3'][j] = (ws[k] for k in order)
        pm = [minv(V[x] + V[y]) for x, y in ((0, 1), (0, 2), (1, 2))]; i = int(np.argmin([abs(m - 80.4) for m in pm]))
        rest = sorted(pm[:i] + pm[i + 1:]); o['mWc'][j] = pm[i]; o['mo1'][j], o['mo2'][j] = rest
    return o


def allobs(P4, nn):
    from interp import h4_top_wmass as H
    O = H._obs(P4, nn); O = {k: O[k] for k in ('m', 'mSD', 'tau21', 'tau32', 'mW', 'n')}; O.update(subjet_feats(P4, nn)); return O


def boot_ci(x, seed, B=2000):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]); return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def part_edit():
    from interp import h4_top_wmass as H
    from interp import h4_verify as HV
    from interp.h29_jedi_wpair_rebuild import p4_of, axis, model_x
    from interp.jedi_core import sample
    from interp.e42_top_wpair_jedi import dilate
    from interp.e42_verify import sample_idx
    TMP.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    D, meta = {}, dict(node='E42x', smoke=SMOKE, per_class=PER, seed=SEED, sets={})
    x, y, _ = sample(PER, SEED); idx = sample_idx(PER, SEED)
    old = np.concatenate([sample_idx(2000, 4201), sample_idx(2000, 4202), sample_idx(3000, 4302)])
    meta['overlap_with_E42_E42v_rows'] = int(np.isin(idx, old).sum())
    real = x[..., 0] > 0; nn = real.sum(1).astype(np.int64)
    assert all(real[j, :nn[j]].all() for j in range(len(nn))), 'padding not contiguous'
    P4 = H._identity(p4_of(x[..., 0], x[..., 1], x[..., 2]), nn).astype(np.float32)
    meta['identity_rebuild_max_abs'] = float(np.abs(model_x(P4, nn) - x).max())
    log(f'sample {SEED}: N {len(y)}; overlap with E42/E42v rows {meta["overlap_with_E42_E42v_rows"]}; identity rebuild max |d| {meta["identity_rebuild_max_abs"]:.2e}')
    D['y'] = y; D['U_st'] = x; D['U_rc'] = model_x(P4, nn, axis(P4, nn))
    O = allobs(P4, nn); log('observables + subjet features on all jets')
    for k in OBSK: D[f'NAT__{k}'] = O[k]
    top = y == 4; win = top & (O['mSD'] > 105) & (O['mSD'] < 210) & (nn >= 3)
    eT = win & (O['mW'] > 65) & (O['mW'] < 95); e80 = win & (((O['mW'] >= 60) & (O['mW'] < 70)) | ((O['mW'] > 90) & (O['mW'] <= 100)))
    meta['eligible'] = dict(tops=int(top.sum()), window=int(win.sum()), T60_T100=int(eT.sum()), T80=int(e80.sum()))
    log(f'eligible: {meta["eligible"]}')
    # ------------------------------------------------ rigid edits (Wr, Nf, ND)
    from interp.h4_top_wmass import _tau32
    z = np.array([[1.0, 0, 5, 6], [0, 1.0, 2, 3], [0.5, 0.5, 1, 2]]); _tau32(z[:, 0].copy(), z[:, 1].copy(), z[:, 2].copy(), z[:, 3].copy(), 3)
    rowsU = np.flatnonzero(eT | e80); F = np.zeros((len(rowsU), P4.shape[1], 4)); F[:] = P4[rowsU]; LAB = H.kt3_batch(F, nn[rowsU])
    res = {}; t1 = time.time()
    for ii, j in enumerate(rowsU):
        tg = ([60.0, 100.0] if eT[j] else []) + ([80.4] if e80[j] else [])
        try:
            res[j] = edit_one(P4[j, :nn[j]], LAB[ii, :nn[j]], tg)
        except Exception as ex:  # noqa: BLE001
            res[j] = dict(error=repr(ex))
        if (ii + 1) % 200 == 0: log(f'  rigid edits {ii + 1}/{len(rowsU)} ({time.time() - t1:.0f}s)')
    meta['rigid_edit_seconds'] = round(time.time() - t1); meta['rigid_edit_errors'] = sum('error' in r for r in res.values())
    log(f'rigid edits done in {meta["rigid_edit_seconds"]}s; errors {meta["rigid_edit_errors"]}')
    VAR = {}   # (key, v) -> (rows, P4)
    PAR = {}
    for key in SETS:
        t = TARGET[key]; rows = np.flatnonzero(e80 if key == 'T80' else eT)
        for v in ('Wr', 'Nf', 'ND'):
            rr = [j for j in rows if 'error' not in res[j] and v in res[j][t]]
            Pv = np.zeros((len(rr), P4.shape[1], 4), np.float32)
            for i, j in enumerate(rr): Pv[i, :nn[j]] = res[j][t][v]['P']
            VAR[(key, v)] = (np.array(rr, np.int64), Pv)
            PAR[(key, v)] = {j: {k: res[j][t][v][k] for k in res[j][t][v] if k != 'P'} for j in rr}
    # ------------------------------------------------ W-scaled (E42 edit) on T60 / T100
    for key in ('T60', 'T100'):
        t1 = time.time(); ed = HV.edit_set(P4, nn, np.flatnonzero(eT), TARGET[key])
        for v, k in (('Ws', 'W'), ('Ns', 'N')): VAR[(key, v)] = (ed[k]['rows'], ed[k]['P']); PAR[(key, v)] = {int(j): dict(h=float(h)) for j, h in zip(ed[k]['rows'], ed[k]['h'])}
        log(f'W-scaled {key}: {len(ed["W"]["rows"])} W, {len(ed["N"]["rows"])} N solved ({time.time() - t1:.0f}s)')
    # ------------------------------------------------ acceptance
    ACC = {}
    for (key, v), (rr, Pv) in VAR.items():
        if len(rr) == 0: ACC[(key, v)] = (rr, Pv, None); continue
        Ov = allobs(Pv, nn[rr]); dm = np.abs(Ov['m'] / O['m'][rr] - 1); dtau = np.abs(Ov['tau32'] / O['tau32'][rr] - 1); t = TARGET[key]
        if v == 'Wr': ok = (dm < 0.01) & (dtau < 0.01) & (np.abs(Ov['mW'] - t) < TOLW[key])
        elif v in ('Nf', 'ND'): ok = (dm < 0.01) & (np.abs(Ov['mW'] - O['mW'][rr]) < 2)
        elif v == 'Ws': ok = (np.abs(Ov['m'] - O['m'][rr]) < 0.5) & (np.abs(Ov['tau32'] - O['tau32'][rr]) < 0.02) & (np.abs(Ov['mW'] - t) < 5)
        else: ok = (np.abs(Ov['m'] - O['m'][rr]) < 0.5) & (np.abs(Ov['tau32'] - O['tau32'][rr]) < 0.02) & (np.abs(Ov['mW'] - O['mW'][rr]) < 2)
        ACC[(key, v)] = (rr[ok], Pv[ok], {k: Ov[k][ok] for k in Ov})
    # ------------------------------------------------ analysis sets and model inputs
    for key in SETS:
        cnt = dict(eligible=int((e80 if key == 'T80' else eT).sum()), **{f'{v}_solved': int(len(VAR[(key, v)][0])) for v in ('Wr', 'Nf', 'ND', 'Ws', 'Ns') if (key, v) in VAR},
                   **{f'{v}_accepted': int(len(ACC[(key, v)][0])) for v in ('Wr', 'Nf', 'ND', 'Ws', 'Ns') if (key, v) in ACC})
        meta['sets'][key] = dict(counts=cnt, pairs={}); log(f'{key}: {cnt}')
        for (Wv, Nv) in PAIRS[key]:
            rw, rn = ACC[(key, Wv)][0], ACC[(key, Nv)][0]; ar = np.intersect1d(rw, rn)
            nm = f'{Wv}-{Nv}'; meta['sets'][key]['pairs'][nm] = dict(n=int(len(ar)))
            if len(ar) < 5: continue
            D[f'rows__{key}__{nm}'] = ar; n0 = nn[ar]
            Pid = P4[ar]; D[f'X__{key}__{nm}__id'] = model_x(Pid, n0, axis(Pid, n0)); Oid = {k: O[k][ar] for k in O}
            se = {}
            Xid = D[f'X__{key}__{nm}__id']
            for v in (Wv, Nv):
                rr, Pv, Ov = ACC[(key, v)]; ii = np.searchsorted(rr, ar); Pv = Pv[ii]; Ov = {k: Ov[k][ii] for k in Ov}
                Xv = model_x(Pv, n0, axis(Pv, n0)); D[f'X__{key}__{nm}__{v}'] = Xv
                for k in OBSK: D[f'O__{key}__{nm}__{v}__{k}'] = Ov[k]
                ptw = Xid[..., 0].astype(np.float64); Dm = (ptw * np.hypot(Xv[..., 1] - Xid[..., 1], Xv[..., 2] - Xid[..., 2])).sum(1) / ptw.sum(1)
                D[f'O__{key}__{nm}__{v}__D'] = Dm
                se[v] = {k: dict(mean=float((Ov[k] - Oid[k]).mean()), ci=boot_ci(Ov[k] - Oid[k], 4430), median_after=float(np.median(Ov[k])))
                         for k in ('m', 'mSD', 'tau21', 'tau32', 'mW', 'w1', 'w2', 'w3', 'ptd', 'mWc', 'mo1', 'mo2')}
                se[v]['D'] = dict(mean=float(Dm.mean()), ci=boot_ci(Dm, 4431))
                pr = [PAR[(key, v)][int(j)] for j in ar]
                for pk in ('f', 'g', 'th', 'h'):
                    vals = [p[pk] for p in pr if pk in p]
                    if vals: se[v][f'param_{pk}_median'] = float(np.median(vals)); se[v][f'param_{pk}_q10_q90'] = [float(np.quantile(vals, .1)), float(np.quantile(vals, .9))]
            for k in ('m', 'mSD', 'tau21', 'tau32', 'mW', 'w1', 'w2', 'w3', 'D'):
                dd = D[f'O__{key}__{nm}__{Wv}__{k}'] - D[f'O__{key}__{nm}__{Nv}__{k}']
                se[f'W_minus_N__{k}'] = dict(mean=float(dd.mean()), ci=boot_ci(dd, 4432))
            se['unedited_median'] = {k: float(np.median(Oid[k])) for k in ('m', 'mSD', 'tau21', 'tau32', 'mW', 'w1', 'w2', 'w3')}
            for k in OBSK: D[f'O__{key}__{nm}__id__{k}'] = Oid[k]
            meta['sets'][key]['pairs'][nm]['side_effects'] = se
            log(f'{key} {nm} n {len(ar)}: W-N dtau21 {se["W_minus_N__tau21"]["mean"]:+.4f} dtau32 {se["W_minus_N__tau32"]["mean"]:+.4f} '
                f'dmSD {se["W_minus_N__mSD"]["mean"]:+.2f} dw1 {se["W_minus_N__w1"]["mean"]:+.4f} D W/N {se[Wv]["D"]["mean"]:.4f}/{se[Nv]["D"]["mean"]:.4f} '
                f'mW med after {se[Wv]["mW"]["median_after"]:.1f}/{se[Nv]["mW"]["median_after"]:.1f}')
            if key == 'T60' and nm == 'Wr-ND':   # example jet: median m_W0
                j = int(np.argmin(np.abs(Oid['mW'] - np.median(Oid['mW'])))); D['EX__row'] = ar[j]
                for v in ('id', Wv, Nv): D[f'EX__{v}'] = D[f'X__{key}__{nm}__{v}'][j]
                D['EX__lab'] = LAB[np.searchsorted(rowsU, ar[j])]
                meta['example'] = dict(row=int(ar[j]), **{f'{v}_{k}': float(D[f'O__{key}__{nm}__{v}__{k}'][j]) for v in ('id', Wv, Nv) for k in ('m', 'mW', 'mWc', 'mo1', 'mo2', 'tau32', 'tau21')})
    # ------------------------------------------------ DIL (real mass change) on eligible tops, for k_DIL
    rows = np.flatnonzero(eT); n0 = nn[rows]; D['rows__DIL'] = rows; D['X__DIL__id'] = model_x(P4[rows], n0, axis(P4[rows], n0))
    for k in OBSK: D[f'O__DIL__id__{k}'] = O[k][rows]
    for lam in LAMS:
        v = f'L{lam:g}'; P = dilate(P4[rows], n0, lam); D[f'X__DIL__{v}'] = model_x(P, n0, axis(P, n0)); Ov = allobs(P, n0)
        for k in OBSK: D[f'O__DIL__{v}__{k}'] = Ov[k]
    log(f'DIL on {len(rows)} eligible tops')
    # ------------------------------------------------ references: HistGBDT top vs rest, 2-fold cross-fit on the natural sample
    from sklearn.ensemble import HistGradientBoostingClassifier
    from interp.h4_top_wmass import _auc
    fold = np.random.default_rng(4422).permutation(len(y)) % 2; yb = (y == 4).astype(int); meta['ref'] = {}
    lo = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    for nm, fs in (('BLIND', FBLIND), ('AWARE', FAWARE)):
        X = np.column_stack([O[k] for k in fs]); mods = []; oof = np.zeros(len(y))
        for fo in (0, 1):
            m_ = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(X[fold != fo], yb[fold != fo]); mods.append(m_)
            oof[fold == fo] = m_.predict_proba(X[fold == fo])[:, 1]
        meta['ref'][nm] = dict(features=list(fs), fit_jets=int(len(y)), auc_top_vs_rest_oof=float(_auc(oof, yb)))
        log(f'reference {nm}: {meta["ref"][nm]}')
        for kk in [k for k in D if k.startswith('X__')]:
            parts = kk.split('__'); base = '__'.join(parts[1:]); rowk = f'rows__{parts[1]}' + (f'__{parts[2]}' if parts[1] != 'DIL' else '')
            rr = D[rowk]; Xv = np.column_stack([D[f'O__{base}__{k}'] for k in fs]); p = np.zeros(len(rr))
            for fo in (0, 1):
                sel = fold[rr] == fo
                if sel.any(): p[sel] = mods[1 - fo].predict_proba(Xv[sel])[:, 1]   # model not trained on the jet's fold
            D[f'{nm}__{base}'] = lo(p)
    np.savez_compressed(TMP / 'inputs.npz', **D)
    (OUT / f'{TAG}_edit.json').write_text(json.dumps(clean(meta), indent=1)); log(f'wrote {TMP}/inputs.npz ({len(D)} arrays), {TAG}_edit.json')


# ================================================================================================ networks (jax env)
def part_jedi():
    from interp.jedi_core import Jedi
    from interp.e42_top_wpair_jedi import score, call, Boot, perm_p, mcnemar, holm
    rng = np.random.default_rng(4423)
    Z = np.load(TMP / 'inputs.npz'); meta = json.loads((OUT / f'{TAG}_edit.json').read_text())
    R = dict(node='E42x', smoke=SMOKE, plan='module docstring (fixed before the first run)', edit=meta, nets={})
    Xk = [k for k in Z.files if k.startswith('X__')]; y = Z['y']

    def scores(run):
        sc, lg = {}, {}
        for k in Xk:
            L = run(np.asarray(Z[k], np.float32)); b = k[3:]; lg[b] = L; sc[b] = score(L)
        return sc, lg

    def kdil(sc, ref):
        a, b = (f'DIL__L{l:g}' for l in LAMS)
        num = sc[b] - sc[a]; den = Z[f'{ref}__{b}'] - Z[f'{ref}__{a}']; bt = Boot(len(num), 4424)
        return float(num.mean() / den.mean()), bt.ratio_ci(num, den), dict(dnet_092=float((sc[a] - sc['DIL__id']).mean()), dnet_108=float((sc[b] - sc['DIL__id']).mean()),
                                                                          dref_092=float((Z[f'{ref}__{a}'] - Z[f'{ref}__DIL__id']).mean()), dref_108=float((Z[f'{ref}__{b}'] - Z[f'{ref}__DIL__id']).mean()))

    def contrasts(sc, lg, kB):
        out = {}
        for key in SETS:
            for (Wv, Nv) in PAIRS[key]:
                nm = f'{Wv}-{Nv}'; b0 = f'{key}__{nm}'
                if f'{b0}__id' not in sc: continue
                s0, sW, sN = sc[f'{b0}__id'], sc[f'{b0}__{Wv}'], sc[f'{b0}__{Nv}']; n = len(s0); bt = Boot(n, 4425 + n)
                dW, dN = sW - s0, sN - s0; x = dW - dN
                c0, cW, cN = (call(lg[f'{b0}__{v}']) for v in ('id', Wv, Nv))
                M = dict(n=n, dW=float(dW.mean()), dW_ci=bt.ci(dW), dN=float(dN.mean()), dN_ci=bt.ci(dN), W_minus_N=float(x.mean()), W_minus_N_ci=bt.ci(x),
                         p_perm_WN=perm_p(x, rng), calls=dict(id=float(c0.mean()), W=float(cW.mean()), N=float(cN.mean()), changed_W=int((cW != c0).sum()),
                                                               changed_N=int((cN != c0).sum()), mcnemar_W_vs_N=mcnemar(cW, cN)), refs={})
                for rf in ('BLIND', 'AWARE'):
                    r = (Z[f'{rf}__{b0}__{Wv}'] - Z[f'{rf}__{b0}__id']) - (Z[f'{rf}__{b0}__{Nv}'] - Z[f'{rf}__{b0}__id'])
                    q = dict(W_minus_N=float(r.mean()), W_minus_N_ci=bt.ci(r), p_perm=perm_p(r, rng))
                    if rf == 'BLIND':
                        dd = x - r; q.update(net_minus_ref=float(dd.mean()), net_minus_ref_ci=bt.ci(dd), p_perm_net_minus_ref=perm_p(dd, rng))
                        if kB is not None:
                            dk = x - kB * r; q.update(k_dil=kB, net_minus_kref=float(dk.mean()), net_minus_kref_ci=bt.ci(dk), p_perm_net_minus_kref=perm_p(dk, rng))
                    M['refs'][rf] = q
                out[f'{key}|{nm}'] = M
        return out

    def regression(sc, key, nm, Wv, Nv):
        b0 = f'{key}__{nm}'; x = (sc[f'{b0}__{Wv}'] - sc[f'{b0}__id']) - (sc[f'{b0}__{Nv}'] - sc[f'{b0}__id'])
        cov = {k: Z[f'O__{b0}__{Wv}__{k}'] - Z[f'O__{b0}__{Nv}__{k}'] for k in ('tau21', 'tau32', 'mSD', 'D')}
        names = list(cov); C = np.column_stack([cov[k] for k in names]); sd = C.std(0); sd[sd < 1e-12] = 1
        A = np.column_stack([np.ones(len(x)), C / sd]); beta, *_ = np.linalg.lstsq(A, x, rcond=None); bi = np.random.default_rng(4426); ints = []
        for _ in range(2000):
            ii = bi.integers(0, len(x), len(x)); b_, *_ = np.linalg.lstsq(A[ii], x[ii], rcond=None); ints.append(b_[0])
        # intercept = W - N at zero difference in the four side effects (linear adjustment)
        msk = (np.abs(cov['tau21']) < 0.02) & (np.abs(cov['tau32']) < 0.02) & (np.abs(cov['mSD']) < 3)
        mm = dict(n=int(msk.sum()))
        if msk.sum() >= 10:
            xm = x[msk]; mm.update(W_minus_N=float(xm.mean()), W_minus_N_ci=Boot(len(xm), 4427).ci(xm), p_perm=perm_p(xm, rng))
        return dict(n=int(len(x)), covariates=names, intercept=float(beta[0]), intercept_ci=[float(np.quantile(ints, .025)), float(np.quantile(ints, .975))],
                    coef_std=dict(zip(names, map(float, beta[1:]))), raw=float(x.mean()), matched_subset=mm)

    # ------------------------------------------------ reference contrasts (network-free)
    refsc = {rf: {k[len(rf) + 2:]: Z[k] for k in Z.files if k.startswith(f'{rf}__')} for rf in ('BLIND', 'AWARE')}
    R['ref_regression'] = {rf: {f'{key}|{pn}': regression(refsc[rf], key, pn, *pn.split('-')) for key in SETS for pn in ('Wr-Nf', 'Wr-ND') if f'{key}__{pn}__id' in refsc[rf]} for rf in refsc}
    log(f'reference regressions: {json.dumps(clean(R["ref_regression"]))}')
    # ------------------------------------------------ networks
    nets = [('official_n64', None)] + [(n, n) for n in EDIT_NETS] + [('untrained_s0', 'untrained')]
    for name, kind in nets:
        t1 = time.time()
        try:
            if kind is None: M_ = Jedi(64, '3-feature'); run = lambda x, M_=M_: M_.run(x)['logits']
            elif kind == 'untrained': M_ = Jedi(64, '3-feature', untrained=True, seed=0); run = lambda x, M_=M_: M_.run(x)['logits']
            else:
                from interp.e33_program_recurrence_jedi import Model
                M_ = Model(kind); run = lambda x, M_=M_: M_.run(x)['logits']
        except Exception as e:  # noqa: BLE001
            log(f'{name}: load failed {e!r}'); R['nets'][name] = dict(error=repr(e)); continue
        Lst = run(np.asarray(Z['U_st'], np.float32)); Lrc = run(np.asarray(Z['U_rc'], np.float32))
        acc = dict(N=int(len(y)), acc_stored=float((Lst.argmax(1) == y).mean()), acc_recentred=float((Lrc.argmax(1) == y).mean()))
        sc, lg = scores(run); kB, kBci, kper = kdil(sc, 'BLIND')
        res = dict(unedited=acc, k_dil_blind=kB, k_dil_blind_ci=kBci, dil=kper, sets=contrasts(sc, lg, kB))
        res['regression'] = {f'{key}|{pn}': regression(sc, key, pn, *pn.split('-')) for key in SETS for pn in ('Wr-Nf', 'Wr-ND') if f'{key}__{pn}__id' in sc}
        R['nets'][name] = res
        log(f'{name}: acc {acc}; k_DIL(BLIND) {kB:+.3f} {np.round(kBci, 3)} ({time.time() - t1:.0f}s)')
        for k, M in res['sets'].items():
            b = M['refs']['BLIND']
            log(f'  {k} n {M["n"]}: dW {M["dW"]:+.3f} dN {M["dN"]:+.3f} W-N {M["W_minus_N"]:+.3f} {np.round(M["W_minus_N_ci"], 3)} p {M["p_perm_WN"]:.3g}; '
                f'BLIND W-N {b["W_minus_N"]:+.3f} {np.round(b["W_minus_N_ci"], 3)}; net-BLIND {b["net_minus_ref"]:+.3f} p {b["p_perm_net_minus_ref"]:.3g}; '
                f'AWARE W-N {M["refs"]["AWARE"]["W_minus_N"]:+.3f}; calls id/W/N {M["calls"]["id"]:.3f}/{M["calls"]["W"]:.3f}/{M["calls"]["N"]:.3f} McN p {M["calls"]["mcnemar_W_vs_N"]["p"]:.3g}')
        for k, g in res['regression'].items(): log(f'  regression {k}: {g}')
        if name == 'official_n64':
            np.savez_compressed(OUT / f'{TAG}_scores.npz', **{f'official__{k}': v for k, v in sc.items()}, **{k: Z[k] for k in Z.files if k.startswith(('BLIND__', 'AWARE__', 'rows__', 'EX__'))})
        del M_
    # ------------------------------------------------ Holm over trained networks x sets (primary Wr - Nf)
    trained = [n for n, _ in nets if n != 'untrained_s0' and 'error' not in R['nets'].get(n, {'error': 1})]
    for stat, get in (('W_minus_N', lambda M: M['p_perm_WN']), ('net_minus_BLIND', lambda M: M['refs']['BLIND']['p_perm_net_minus_ref']),
                      ('net_minus_kBLIND', lambda M: M['refs']['BLIND'].get('p_perm_net_minus_kref', 1.0)), ('McNemar', lambda M: M['calls']['mcnemar_W_vs_N']['p'])):
        for pn in ('Wr-ND', 'Wr-Nf'):
            fam = [(n, f'{k}|{pn}') for n in trained for k in SETS if f'{k}|{pn}' in R['nets'][n]['sets']]
            hp = holm([get(R['nets'][n]['sets'][k]) for n, k in fam])
            for (n, k), h in zip(fam, hp): R['nets'][n]['sets'][k][f'p_holm_{stat}'] = float(h)
            R[f'holm_{stat}_{pn}_family_size'] = len(fam)
    (OUT / f'{TAG}.json').write_text(json.dumps(clean(R), indent=1)); log(f'wrote {OUT}/{TAG}.json')
    try:
        figure(R, Z)
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc(); log(f'figure failed: {e!r}')


def figure(R, Z):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[1, 1.6]))
    a = ax[0]
    if 'EX__id' in Z.files:
        lab = Z['EX__lab']; ex = R['edit'].get('example', {})
        for v, mk, nm in (('id', 'o', 'unedited'), ('Wr', '^', 'W-rigid (W pair -> 60 GeV)'), ('ND', 's', 'N-matched (non-W pair, same displacement)')):
            X = Z[f'EX__{v}']; real = X[:, 0] > 0; n = real.sum(); col = np.array(['C0', 'C1', 'C2'])[np.clip(lab[:n], 0, 2)]
            a.scatter(X[real, 1], X[real, 2], s=8 + 6 * X[real, 0] ** 0.7, c=col if v == 'id' else 'none', edgecolors=col, marker=mk, alpha=0.6 if v == 'id' else 0.8,
                      label=f'{nm}: m_W {ex.get(v + "_mWc", np.nan):.0f} GeV, m {ex.get(v + "_m", np.nan):.0f} GeV, tau32 {ex.get(v + "_tau32", np.nan):.2f}')
        a.set_xlabel('Delta eta (about the jet axis)'); a.set_ylabel('Delta phi (rad)'); a.legend(fontsize=7, loc='lower left')
        a.set_title('a  one top jet: subjets moved rigidly (colour = subjet, marker size ~ pT)', fontsize=9, loc='left')
    b = ax[1]; names = [n for n in R['nets'] if 'error' not in R['nets'][n]]
    keys = [k for k in ('T60|Wr-ND', 'T100|Wr-ND', 'T80|Wr-ND') if all(k in R['nets'][n]['sets'] for n in names)]
    rows = names + ['BLIND GBDT (no pair mass)', 'AWARE GBDT (+ pair masses)']; yy = np.arange(len(rows))[::-1]; off = {k: (i - 1) * 0.25 for i, k in enumerate(keys)}
    lbl = {'T60|Wr-ND': 'm_W -> 60 GeV', 'T100|Wr-ND': 'm_W -> 100 GeV', 'T80|Wr-ND': 'off-peak m_W -> 80 GeV'}
    for i, k in enumerate(keys):
        vals = []
        for n in names: M = R['nets'][n]['sets'][k]; vals.append((M['W_minus_N'], M['W_minus_N_ci']))
        M0 = R['nets']['official_n64']['sets'][k]
        for rf in ('BLIND', 'AWARE'): q = M0['refs'][rf]; vals.append((q['W_minus_N'], q['W_minus_N_ci']))
        v = np.array([q[0] for q in vals]); lo_ = np.array([q[1][0] for q in vals]); hi = np.array([q[1][1] for q in vals])
        b.errorbar(v, yy + off[k], xerr=[v - lo_, hi - v], fmt='o', ms=4, color=f'C{i + 3}', label=lbl[k], capsize=2)
    b.axvline(0, color='k', lw=0.7); b.set_yticks(yy); b.set_yticklabels(rows, fontsize=8)
    b.set_xlabel('W - N: change in top log-odds under the W-pair edit minus under the matched non-W edit')
    b.set_title('b  W - N per network vs pair-blind / pair-aware classifiers (95 % bootstrap CI)', fontsize=9, loc='left'); b.legend(fontsize=8)
    fig.tight_layout(); p = OUT / f'{TAG}_wpair_rigid.png'; fig.savefig(p, dpi=130); log(f'wrote {p}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['edit', 'jedi'])
    a = ap.parse_args(); rc = 0
    try:
        part_edit() if a.part == 'edit' else part_jedi()
    except BaseException:  # noqa: BLE001
        import traceback; traceback.print_exc(); rc = 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
