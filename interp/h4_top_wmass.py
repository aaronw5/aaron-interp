"""H4 (loop D, claim D6): do trained top taggers use the W-pair mass inside the top beyond total mass and tau32?
Edit pre-registered in H2 (inbox_d/merged/H2_report.md, "Pre-registered edits for the next round").

Sample: interp.data_top.sample(5000, 601) on the Top Landscape test split (5,000 top + 5,000 QCD, random rows) and, for the
background rejection, 100,000 random QCD test jets (sample(100000, 602, classes=(0,))).

Subjets: three exclusive-kT subjets (E-scheme, all constituents, rapidity-azimuth distances; the H2 definition). W pair = the
subjet pair whose mass is closest to 80.4 GeV, third subjet c = the other one.
Edits (each subjet moved rigidly in (eta, phi): every constituent of a subjet gets the same shift, pT and constituent masses kept,
so the subjet pT fractions and subjet masses do not change):
  W<t>  rescale the opening of the W pair about its pT-weighted centroid by f (bisection) so that the W-pair mass is t GeV; then
        move c along the line from the W centroid (factor g, bisection) so that the jet mass is restored (|dm| < 0.05 GeV in the
        solve). Top jets: t = 60 and 100 GeV (from 65 < mW < 95). QCD jets in the mSD window: t = 80.4 GeV (from 50 < mW < 70 or 90 < mW < 115).
  N<t>  control, non-W pair: the pair (d, c) with d = the harder W daughter is given the same relative mass change
        m_dc -> m_dc * t / mW0, by a rigid 2-d shift of c (Newton) that also keeps the jet mass; the W pair is not moved, so its
        mass is unchanged. If it does not converge with the harder daughter, the softer one is used.
  R<t>  control, rotation: all constituents rotated about the jet axis in the (deta, dphi) plane by an angle of random sign whose
        pT-weighted mean displacement equals that of the W<t> edit on the same jet.
  id    identity: the same rebuild (pT, eta, phi, m -> px, py, pz, E) without any shift; every comparison is to id.
Acceptance (recomputed with the H2 observables, reclustering after the edit): W<t>: |dm| < 0.5 GeV, |d tau32| < 0.02,
|mW - t| < 5 GeV; N<t>: |dm| < 0.5 GeV, |d tau32| < 0.02, |mW - mW0| < 2 GeV. Analysis set for target t = jets where both
W<t> and N<t> are accepted (R<t> on the same jets).
Models: ParT and ParticleNet trained here on Top Landscape kin features (weaver checkpoints of jobs d-h2-top-part / -pnet, the
best-epoch state at run time, interim unless the training has finished), LorentzNet (released top weights, input = constituent
four-momenta plus two beam particles, as the authors' collate_fn); classical: LR2 = LR(mSD, tau32) (negative control),
LR3 = LR(mSD, tau32, mW) and MLP3 = a small MLP on (mSD, tau32, mW) (positive controls), fitted on 100k val jets (H2).
Score: log-odds of top (network logit difference; log likelihood ratio for LR; MLP logit).
Steps: edit (CPU, numba), classical (CPU), c2st (CPU), nets (GPU), stats, fig. Results: $AI_RESULTS/physics/h4_*.
  AI_PY=conda ./job.sh d-h4-ctrl 4 12 'python -m interp.h4_top_wmass --step edit,classical,c2st'
  ./part_job.sh d-h4-run 'python -m interp.h4_top_wmass --step nets,stats,fig'
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
import numpy as np

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
OUT2 = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
SMOKE = bool(os.environ.get('H4_SMOKE'))          # H4_SMOKE=1: 300 + 300 sample jets, 3,000 QCD, outputs in physics/h4_smoke/
OUT = OUT2 / 'h4_smoke' if SMOKE else OUT2
V = Path('/j-jepa-vol/aaron-interp')
MWREF = 80.4
TOP_TARGETS = (60.0, 100.0)
QCD_TARGET = 80.4
B = 2000

try:
    from numba import njit, prange
except ImportError:  # local import check only
    njit = lambda *a, **k: (lambda f: f); prange = range
try:
    from interp.h2_onesignal_setup import _cluster as _h2_cluster
except Exception:  # local import check only
    _h2_cluster = None


# ---------------------------------------------------------------------------------------------------------------- subjets
@njit(cache=False)
def _kt3_labels(px, py, pz, E, n):
    """exclusive kT (p = 1, E-scheme, rapidity-azimuth dR) down to 3 objects; returns each particle's subjet 0/1/2"""
    vx = px.copy(); vy = py.copy(); vz = pz.copy(); vE = E.copy()
    pt = np.hypot(vx, vy); ph = np.arctan2(vy, vx); yy = np.zeros(n)
    for i in range(n):
        e = max(vE[i], abs(vz[i]) + 1e-9); yy[i] = 0.5 * np.log((e + vz[i]) / (e - vz[i]))
    alive = np.ones(n, np.bool_); lab = np.arange(n); cnt = n
    while cnt > 3:
        best = 1e300; bi = -1; bj = -1
        for a in range(n):
            if not alive[a]: continue
            for b in range(a + 1, n):
                if not alive[b]: continue
                dphi = abs(ph[a] - ph[b])
                if dphi > np.pi: dphi = 2 * np.pi - dphi
                d = min(pt[a] ** 2, pt[b] ** 2) * ((yy[a] - yy[b]) ** 2 + dphi ** 2)
                if d < best: best = d; bi = a; bj = b
        vx[bi] += vx[bj]; vy[bi] += vy[bj]; vz[bi] += vz[bj]; vE[bi] += vE[bj]; alive[bj] = False
        pt[bi] = np.hypot(vx[bi], vy[bi]); ph[bi] = np.arctan2(vy[bi], vx[bi])
        e = max(vE[bi], abs(vz[bi]) + 1e-9); yy[bi] = 0.5 * np.log((e + vz[bi]) / (e - vz[bi]))
        for i in range(n):
            if lab[i] == bj: lab[i] = bi
        cnt -= 1
    out = -np.ones(n, np.int64); k = 0
    for a in range(n):
        if alive[a]:
            for i in range(n):
                if lab[i] == a: out[i] = k
            k += 1
    return out


@njit(parallel=True, cache=False)
def kt3_batch(F, nn):
    J = F.shape[0]; L = F.shape[1]; out = -np.ones((J, L), np.int64)
    for j in prange(J):
        n = nn[j]
        if n < 3: continue
        out[j, :n] = _kt3_labels(F[j, :n, 0].astype(np.float64), F[j, :n, 1].astype(np.float64), F[j, :n, 2].astype(np.float64),
                                 F[j, :n, 3].astype(np.float64), n)
    return out


def _minv(v):
    return float(np.sqrt(max(v[3] ** 2 - v[0] ** 2 - v[1] ** 2 - v[2] ** 2, 0.0)))


@njit(cache=False)
def _tau32(px, py, pz, E, n):
    """tau3/tau2 of one jet, the H2 definition (beta = 1, R0 = 0.8, exclusive-kT axes with WTA recombination)"""
    rw, _, _, _, _, _, _, _, _ = _h2_cluster(px, py, pz, E, n, 1, True, 1, False)
    ptp = np.hypot(px, py); php = np.arctan2(py, px); yp = np.zeros(n)
    for i in range(n):
        e = max(E[i], abs(pz[i]) + 1e-9); yp[i] = 0.5 * np.log((e + pz[i]) / (e - pz[i]))
    t = np.zeros(2)
    for q in range(2):
        N = 2 + q; base = 3 if q == 0 else 0
        for i in range(n):
            dm = 1e9
            for k in range(N):
                dphi = abs(php[i] - rw[base + k, 6])
                if dphi > np.pi: dphi = 2 * np.pi - dphi
                d = np.sqrt((yp[i] - rw[base + k, 5]) ** 2 + dphi ** 2)
                if d < dm: dm = d
            t[q] += ptp[i] * dm
    return t[1] / max(t[0], 1e-12)


class Jet:
    """one jet's constituents in (pt, eta, phi, m) with relative coordinates (x, y) = (eta - eta_J, phi - phi_J) about the
    jet axis (constituent sum); subjet labels lab (0/1/2), subjet pT-weighted centroids r. A configuration is (shifts (3, 2),
    h): constituent i of subjet k sits at r_k + shift_k + h (x_i - r_k), i.e. subjets translated rigidly and their internal
    spread scaled by h (pT, constituent masses and so the subjet pT fractions are unchanged)"""

    def __init__(self, P, lab):
        P = np.asarray(P, np.float64); self.pt = np.hypot(P[:, 0], P[:, 1]); self.m = np.sqrt(np.maximum(P[:, 3] ** 2 - (P[:, :3] ** 2).sum(1), 0))
        self.eta = np.arcsinh(P[:, 2] / np.maximum(self.pt, 1e-12)); self.phi = np.arctan2(P[:, 1], P[:, 0])
        S = P.sum(0); self.eJ = np.arcsinh(S[2] / np.hypot(S[0], S[1])); self.pJ = np.arctan2(S[1], S[0])
        self.x = self.eta - self.eJ; self.y = (self.phi - self.pJ + np.pi) % (2 * np.pi) - np.pi
        self.lab = lab; self.sel = [lab == k for k in range(3)]
        self.ptk = np.array([self.pt[s].sum() for s in self.sel])
        self.r = np.array([[(self.pt[s] * self.x[s]).sum() / self.ptk[k], (self.pt[s] * self.y[s]).sum() / self.ptk[k]] for k, s in enumerate(self.sel)])

    def p4(self, x, y, s=slice(None)):
        pt, m = self.pt[s], self.m[s]; eta = self.eJ + x; phi = self.pJ + y
        return np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)], -1)

    def xy(self, shifts, h=1.0):
        rr = self.r[self.lab]; sh = np.asarray(shifts)[self.lab]
        return rr[:, 0] + sh[:, 0] + h * (self.x - rr[:, 0]), rr[:, 1] + sh[:, 1] + h * (self.y - rr[:, 1])

    def V(self, k, shift, h=1.0):
        s = self.sel[k]; r = self.r[k]
        return self.p4(r[0] + shift[0] + h * (self.x[s] - r[0]), r[1] + shift[1] + h * (self.y[s] - r[1]), s).sum(0)

    def build(self, shifts=None, h=1.0, rot=None):
        """(n, 4) px, py, pz, E for a configuration (or a rotation by angle rot about the jet axis), and the pT-weighted mean
        displacement D of the constituents in (eta, phi)"""
        if rot is not None:
            c, s_ = np.cos(rot), np.sin(rot); x, y = c * self.x - s_ * self.y, s_ * self.x + c * self.y
        else:
            x, y = self.xy(np.zeros((3, 2)) if shifts is None else shifts, h)
        D = float((self.pt * np.hypot(x - self.x, y - self.y)).sum() / self.pt.sum())
        return self.p4(x, y), D

    def tau32(self, shifts, h=1.0):
        P, _ = self.build(shifts, h); return _tau32(P[:, 0].copy(), P[:, 1].copy(), P[:, 2].copy(), P[:, 3].copy(), len(P))

    def pair_masses(self, shifts=np.zeros((3, 2)), h=1.0):
        Vk = [self.V(k, shifts[k], h) for k in range(3)]
        return {(a, b): _minv(Vk[a] + Vk[b]) for a, b in ((0, 1), (0, 2), (1, 2))}, _minv(Vk[0] + Vk[1] + Vk[2])


def _bisect(fn, lo, hi, tol=0.01, it=60):
    flo, fhi = fn(lo), fn(hi)
    if flo is None or fhi is None or np.sign(flo) == np.sign(fhi): return None
    for _ in range(it):
        mid = 0.5 * (lo + hi); fm = fn(mid)
        if fm is None: return None
        if abs(fm) < tol: return mid
        if np.sign(fm) == np.sign(flo): lo, flo = mid, fm
        else: hi = mid
    return 0.5 * (lo + hi)


def _solve_h(inner, jt, tau0, tol=0.004):
    """outer 1-d solve on the internal-spread scale h so that tau32 returns to tau0; inner(h) -> shifts or None"""
    def F(h):
        sh = inner(h)
        return None if sh is None else jt.tau32(sh, h) - tau0
    f1 = F(1.0)
    if f1 is None: return None
    if abs(f1) < tol: return 1.0
    lo, hi = (0.5, 1.0) if f1 > 0 else (1.0, 1.6)
    h = _bisect(F, lo, hi, tol=tol, it=14)
    return h


def edit_jet(jt, target, tau0):
    """W edit and non-W control on one jet (see the module docstring); tau32 restored to tau0 through h.
    Returns P4 arrays, D, convergence flags and parameters"""
    pm0, m0 = jt.pair_masses(); w = min(pm0, key=lambda p: abs(pm0[p] - MWREF)); a, b = w; c = 3 - a - b; mW0 = pm0[w]
    r = jt.r; cW = (jt.ptk[a] * r[a] + jt.ptk[b] * r[b]) / (jt.ptk[a] + jt.ptk[b])
    res = dict(mW0=mW0, m0=m0, w=(a, b), c=c)

    def sh(f, g):
        s = np.zeros((3, 2)); s[a] = (f - 1) * (r[a] - cW); s[b] = (f - 1) * (r[b] - cW); s[c] = (g - 1) * (r[c] - cW); return s

    def inner_W(h):
        f = _bisect(lambda f: _minv(jt.V(a, sh(f, 1)[a], h) + jt.V(b, sh(f, 1)[b], h)) - target, 0.05, 4.0)
        if f is None: return None
        Va, Vb = jt.V(a, sh(f, 1)[a], h), jt.V(b, sh(f, 1)[b], h)
        g = _bisect(lambda g: _minv(Va + Vb + jt.V(c, sh(f, g)[c], h)) - m0, 0.0, 6.0)
        return None if g is None else sh(f, g)
    h = _solve_h(inner_W, jt, tau0)
    s = None if h is None else inner_W(h)
    if s is None: res['W_ok'] = False
    else:
        P, D = jt.build(s, h); pm, mt = jt.pair_masses(s, h); res.update(W_ok=True, W_P=P, W_D=D, W_h=h, W_pairs=pm, W_m=mt)
    # non-W control: the pair (d, c) gets the same relative mass change by a rigid 2-d shift t of c; the W pair is not
    # translated (its mass moves only through h); jet mass kept; tau32 restored through h
    ratio = target / mW0; res['N_ok'] = False
    for d in sorted((a, b), key=lambda k: -jt.ptk[k]):
        tgt = ratio * pm0[tuple(sorted((d, c)))]

        def inner_N(h):
            Va, Vb, Vd = jt.V(a, (0, 0), h), jt.V(b, (0, 0), h), jt.V(d, (0, 0), h)
            F = lambda t: np.array([_minv(Vd + jt.V(c, t, h)) - tgt, _minv(Va + Vb + jt.V(c, t, h)) - m0])
            t = (ratio - 1) * (r[c] - r[d])
            for _ in range(40):
                Ft = F(t)
                if np.max(np.abs(Ft)) < 0.02:
                    if np.hypot(*t) > 1.0: return None
                    s = np.zeros((3, 2)); s[c] = t; return s
                e = 1e-4; Jm = np.stack([(F(t + np.array([e, 0])) - Ft) / e, (F(t + np.array([0, e])) - Ft) / e], 1)
                try: step = np.linalg.solve(Jm, -Ft)
                except np.linalg.LinAlgError: return None
                nrm = np.hypot(*step)
                if nrm > 0.1: step *= 0.1 / nrm
                t = t + step
            return None
        h = _solve_h(inner_N, jt, tau0)
        s = None if h is None else inner_N(h)
        if s is not None:
            P, D = jt.build(s, h); pm, mt = jt.pair_masses(s, h)
            res.update(N_ok=True, N_P=P, N_D=D, N_d=d, N_h=h, N_pairs=pm, N_m=mt); break
    return res


# ------------------------------------------------------------------------------------------------------------------- data
def _identity(P4, nn):
    """rebuild every jet through (pt, eta, phi, m) -> (px, py, pz, E) with no shift (the same path as the edits)"""
    out = np.zeros_like(P4)
    for j in range(len(nn)):
        n = nn[j]; P = P4[j, :n].astype(np.float64); pt = np.hypot(P[:, 0], P[:, 1]); eta = np.arcsinh(P[:, 2] / np.maximum(pt, 1e-12))
        phi = np.arctan2(P[:, 1], P[:, 0]); m = np.sqrt(np.maximum(P[:, 3] ** 2 - (P[:, :3] ** 2).sum(1), 0))
        out[j, :n] = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)], -1)
    return out


def _obs(P4, nn):
    from interp.h2_onesignal_setup import top_obs
    F = np.zeros(P4.shape[:2] + (6,), np.float32); F[..., :4] = P4
    o = []
    for s in range(0, len(nn), 20000):
        o.append(top_obs(np.ascontiguousarray(F[s:s + 20000]), np.ascontiguousarray(nn[s:s + 20000].astype(np.int64)), 0.1, 0.8))
    o = np.concatenate(o) if o else np.zeros((0, 10))
    return dict(m=o[:, 0], mSD=o[:, 1], tau32=o[:, 4] / np.maximum(o[:, 3], 1e-12), tau21=o[:, 3] / np.maximum(o[:, 2], 1e-12),
                tau1=o[:, 2], tau2=o[:, 3], tau3=o[:, 4], mW=o[:, 5], m12min=o[:, 6], m123=o[:, 7], n=o[:, 8])


_FULL = {}


def _full_test():
    """the whole test full.npy read sequentially once (random-row memmap reads on CephFS ran at ~3 s per row, 2026-10-19)"""
    if 'f' not in _FULL:
        from interp import data_top as dt
        t = time.time(); _FULL['f'] = np.load(dt.DST / 'test' / 'full.npy'); log(f'read test full.npy {_FULL["f"].shape} in {time.time() - t:.0f}s')
    return _FULL['f']


def _sample(per_class, seed, classes=(0, 1)):
    """interp.data_top.sample(per_class, seed, 'test', classes=...) with the same rows (same rng calls), from the in-memory array"""
    from interp import data_top as dt
    d = dt.DST / 'test'; y = np.load(d / 'label.npy'); rng = np.random.default_rng(seed)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in classes]))
    return dict(idx=idx, label=y[idx], jet=np.load(d / 'jet.npy')[idx], full=_full_test()[idx])


def load_sets():
    T = _sample(300 if SMOKE else 5000, 601); Q = _sample(3000 if SMOKE else 100000, 602, classes=(0,))
    sets = {}
    for name, S in (('T', T), ('Q', Q)):
        nn = np.minimum(S['jet'][:, 5].astype(np.int64), S['full'].shape[1])
        sets[name] = dict(idx=S['idx'], label=S['label'].astype(int), nn=nn, P4_stored=S['full'][..., :4].astype(np.float32),
                          deta_stored=S['full'][..., 4], dphi_stored=S['full'][..., 5], jet=S['jet'])
    return sets


VARS = {'T': ['id'] + [f'{k}{int(t)}' for t in TOP_TARGETS for k in 'WNR'], 'Q': ['id', 'W80', 'N80', 'R80']}


NCPU = int(os.environ.get('OMP_NUM_THREADS', '4'))


def _work(arg):
    """one jet, one target: W edit, non-W control, rotation (sign from rng(603 + row))"""
    j, P, lab, t = arg
    jt = Jet(P, lab); tau0 = jt.tau32(np.zeros((3, 2))); e = edit_jet(jt, t, tau0); out = {}
    rec = dict(row=j, mW0=e['mW0'], m0=e['m0'], tau0=tau0)
    if e['W_ok']:
        out['W'] = dict(P=e['W_P'].astype(np.float32), meta=dict(rec, D=e['W_D'], h=e['W_h'], pairW=e['W_pairs'][tuple(sorted(e['w']))], m=e['W_m']))
        alpha = 2 * np.arcsin(min(1.0, e['W_D'] / (2 * (jt.pt * np.hypot(jt.x, jt.y)).sum() / jt.pt.sum())))
        P, D = jt.build(rot=alpha * np.random.default_rng(603 + j).choice([-1, 1])); out['R'] = dict(P=P.astype(np.float32), meta=dict(rec, D=D, alpha=alpha))
    if e['N_ok']:
        out['N'] = dict(P=e['N_P'].astype(np.float32), meta=dict(rec, D=e['N_D'], h=e['N_h'], pairW=e['N_pairs'][tuple(sorted(e['w']))], m=e['N_m']))
    return out


def step_edit():
    import multiprocessing as mp
    z = np.array([[1.0, 0, 5, 6], [0, 1.0, 2, 3], [0.5, 0.5, 1, 2]]); _tau32(z[:, 0].copy(), z[:, 1].copy(), z[:, 2].copy(), z[:, 3].copy(), 3)  # compile before fork
    sets = load_sets(); save = {}; info = {}
    for name, S in sets.items():
        nn = S['nn']; log(f'set {name}: {len(nn)} jets; identity rebuild')
        P4id = _identity(S['P4_stored'], nn).astype(np.float32)
        log(f'  identity max |dp| vs stored {np.max(np.abs(P4id - S["P4_stored"])):.3g} GeV; observables ...')
        O = _obs(P4id, nn)
        save[f'{name}_idx'] = S['idx']; save[f'{name}_label'] = S['label']; save[f'{name}_nn'] = nn; save[f'{name}_P4_id'] = P4id
        if name == 'T': save['T_P4_stored'] = S['P4_stored']; save['T_deta_stored'] = S['deta_stored']; save['T_dphi_stored'] = S['dphi_stored']
        for k, v in O.items(): save[f'{name}_id_obs_{k}'] = v
        win = (O['mSD'] > 105) & (O['mSD'] < 210) & (nn >= 3)
        if name == 'T':
            elig = win & (S['label'] == 1) & (O['mW'] > 65) & (O['mW'] < 95); targets = TOP_TARGETS
        else:
            elig = win & (((O['mW'] > 50) & (O['mW'] < 70)) | ((O['mW'] > 90) & (O['mW'] < 115))); targets = (QCD_TARGET,)
        rows = np.flatnonzero(elig); log(f'  eligible {len(rows)} of {len(nn)}; kT subjets ...')
        F = np.zeros((len(rows), P4id.shape[1], 4), np.float64); F[:] = P4id[rows]
        LAB = kt3_batch(F, nn[rows]); log('  subjets done; editing')
        for t in targets:
            tag = str(int(t)) if t != QCD_TARGET else '80'
            jobs = [(int(j), P4id[j, :nn[j]], LAB[ii, :nn[j]], float(t)) for ii, j in enumerate(rows)]
            with mp.get_context('fork').Pool(NCPU) as pool:
                results = pool.map(_work, jobs, chunksize=8)
            log(f'    target {t}: {len(results)} jets edited')
            out = {k: [] for k in ('W', 'N', 'R')}; meta = {k: [] for k in ('W', 'N', 'R')}
            for j, rr_ in zip(rows, results):
                for k in 'WNR':
                    if rr_.get(k) is not None:
                        out[k].append((j, rr_[k]['P'])); meta[k].append(rr_[k]['meta'])
            for k in 'WNR':
                v = f'{k}{tag}'; rr = np.array([o[0] for o in out[k]], np.int64)
                P = np.zeros((len(rr), P4id.shape[1], 4), np.float32)
                for i, (_, Pj) in enumerate(out[k]): P[i, :len(Pj)] = Pj
                Ov = _obs(P, nn[rr]); save[f'{name}_{v}_rows'] = rr; save[f'{name}_{v}_P4'] = P
                for kk, vv in Ov.items(): save[f'{name}_{v}_obs_{kk}'] = vv
                for kk in ('D', 'mW0', 'm0'): save[f'{name}_{v}_{kk}'] = np.array([m[kk] for m in meta[k]])
                if k in 'WN':
                    for kk in ('pairW', 'h'): save[f'{name}_{v}_{kk}'] = np.array([m[kk] for m in meta[k]])
                info[f'{name}_{v}'] = dict(n_eligible=int(len(rows)), n_converged=int(len(rr)))
                log(f'  {name} {v}: converged {len(rr)}/{len(rows)}; median D {np.median(save[f"{name}_{v}_D"]) if len(rr) else np.nan:.4f}')
    np.savez(OUT / 'h4_edits.npz', **save)
    (OUT / 'h4_edit_info.json').write_text(json.dumps(info, indent=1)); log('wrote h4_edits.npz')


def load_edits():
    return dict(np.load(OUT / 'h4_edits.npz'))


def obs_of(E, name, v):
    return {k[len(f'{name}_{v}_obs_'):]: E[k] for k in E if k.startswith(f'{name}_{v}_obs_')}


def analysis_sets(E):
    """per set / target: accepted rows (W and N both accepted) and the positions of those rows in each variant's arrays"""
    A = {}
    for name, tags in (('T', [str(int(t)) for t in TOP_TARGETS]), ('Q', ['80'])):
        O0 = obs_of(E, name, 'id')
        for tag in tags:
            t = float(tag) if name == 'T' else QCD_TARGET; acc = {}
            for k in 'WN':
                rr = E[f'{name}_{k}{tag}_rows']; O = obs_of(E, name, f'{k}{tag}')
                dm = np.abs(O['m'] - O0['m'][rr]); dt = np.abs(O['tau32'] - O0['tau32'][rr])
                okW = np.abs(O['mW'] - t) < 5 if k == 'W' else np.abs(O['mW'] - O0['mW'][rr]) < 2
                acc[k] = set(rr[(dm < 0.5) & (dt < 0.02) & okW].tolist())
            rows = np.array(sorted(acc['W'] & acc['N']), np.int64)
            pos = {k: np.searchsorted(E[f'{name}_{k}{tag}_rows'], rows) for k in 'WNR'}
            A[f'{name}{tag}'] = dict(set=name, tag=tag, target=t, rows=rows, pos=pos,
                                    n_eligible=int(len(set(E[f'{name}_W{tag}_rows'].tolist()) | set(E[f'{name}_N{tag}_rows'].tolist()))),
                                    n_W_acc=len(acc['W']), n_N_acc=len(acc['N']))
    return A


# --------------------------------------------------------------------------------------------------------------- classical
def _lr_fit(X, y, edges):
    hs, _ = np.histogramdd(X[y == 1], bins=edges); hb, _ = np.histogramdd(X[y == 0], bins=edges)
    return np.log((hs + 0.5) / (hs.sum() + 0.5 * hs.size)) - np.log((hb + 0.5) / (hb.sum() + 0.5 * hb.size))


def _lr_eval(H, X, edges):
    ix = tuple(np.clip(np.digitize(X[:, k], edges[k]) - 1, 0, len(edges[k]) - 2) for k in range(X.shape[1]))
    return H[ix]


def step_classical():
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    Vv = dict(np.load(OUT2 / 'h2_top_obs_val.npz')); Vv['tau32'] = Vv['tau3'] / np.maximum(Vv['tau2'], 1e-12); yv = Vv['label'].astype(int)
    e2 = [np.linspace(0, 300, 31), np.linspace(0, 1, 26)]; e3 = e2 + [np.linspace(0, 200, 21)]
    H2 = _lr_fit(np.stack([Vv['mSD'], Vv['tau32']], 1), yv, e2); H3 = _lr_fit(np.stack([Vv['mSD'], Vv['tau32'], Vv['mW']], 1), yv, e3)
    X3 = np.stack([Vv['mSD'], Vv['tau32'], Vv['mW']], 1); sc = StandardScaler().fit(X3)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 64), early_stopping=True, random_state=0, max_iter=200).fit(sc.transform(X3), yv)
    log(f'MLP3 fitted on {len(yv)} val jets: {mlp.n_iter_} iterations, val score {mlp.best_validation_score_:.4f}')
    # tau21 controls (H4 follow-up: the W edit also lowers tau21 and the subjet masses): LR / MLP on (mSD, tau32, tau21), no mW
    Vv['tau21'] = Vv['tau2'] / np.maximum(Vv['tau1'], 1e-12); e3t = e2 + [np.linspace(0, 1, 26)]
    H3t = _lr_fit(np.stack([Vv['mSD'], Vv['tau32'], Vv['tau21']], 1), yv, e3t)
    X3t = np.stack([Vv['mSD'], Vv['tau32'], Vv['tau21']], 1); sct = StandardScaler().fit(X3t)
    mlpt = MLPClassifier(hidden_layer_sizes=(64, 64), early_stopping=True, random_state=0, max_iter=200).fit(sct.transform(X3t), yv)
    log(f'MLP3t fitted: {mlpt.n_iter_} iterations, val score {mlpt.best_validation_score_:.4f}')
    E = load_edits(); out = {}
    for name in ('T', 'Q'):
        for v in VARS[name]:
            O = obs_of(E, name, v); X2 = np.stack([O['mSD'], O['tau32']], 1); X3e = np.stack([O['mSD'], O['tau32'], O['mW']], 1)
            p = np.clip(mlp.predict_proba(sc.transform(X3e))[:, 1], 1e-7, 1 - 1e-7)
            X3te = np.stack([O['mSD'], O['tau32'], O['tau21']], 1); pt_ = np.clip(mlpt.predict_proba(sct.transform(X3te))[:, 1], 1e-7, 1 - 1e-7)
            sc_v = {'LR2': _lr_eval(H2, X2, e2), 'LR3': _lr_eval(H3, X3e, e3), 'MLP3': np.log(p / (1 - p)),
                    'LR3t': _lr_eval(H3t, X3te, e3t), 'MLP3t': np.log(pt_ / (1 - pt_))}
            for k, x in sc_v.items():   # full-length arrays like the network scores: edited rows substituted into the id scores
                if v == 'id': out[f'{k}|{name}|{v}'] = x
                else: full = out[f'{k}|{name}|id'].copy(); full[E[f'{name}_{v}_rows']] = x; out[f'{k}|{name}|{v}'] = full
    np.savez(OUT / 'h4_scores_classical.npz', **out); log('wrote h4_scores_classical.npz')


# -------------------------------------------------------------------------------------------------------------------- C2ST
@njit(parallel=True, cache=False)
def _subfeats(F, nn, LAB):
    """per jet: 3 pair masses sorted, subjet pT fractions sorted, subjet masses sorted, pair dR (rapidity-azimuth) sorted"""
    J = F.shape[0]; out = np.zeros((J, 12))
    for j in prange(J):
        n = nn[j]
        if n < 3: continue
        S = np.zeros((3, 4))
        for i in range(n):
            k = LAB[j, i]
            for c in range(4): S[k, c] += F[j, i, c]
        pt = np.zeros(3); yy = np.zeros(3); ph = np.zeros(3); ms = np.zeros(3)
        for k in range(3):
            pt[k] = np.hypot(S[k, 0], S[k, 1]); ph[k] = np.arctan2(S[k, 1], S[k, 0])
            e = max(S[k, 3], abs(S[k, 2]) + 1e-9); yy[k] = 0.5 * np.log((e + S[k, 2]) / (e - S[k, 2]))
            ms[k] = np.sqrt(max(S[k, 3] ** 2 - S[k, 0] ** 2 - S[k, 1] ** 2 - S[k, 2] ** 2, 0.0))
        pm = np.zeros(3); dr = np.zeros(3); q = 0
        for a in range(3):
            for b in range(a + 1, 3):
                v = S[a] + S[b]; pm[q] = np.sqrt(max(v[3] ** 2 - v[0] ** 2 - v[1] ** 2 - v[2] ** 2, 0.0))
                dphi = abs(ph[a] - ph[b])
                if dphi > np.pi: dphi = 2 * np.pi - dphi
                dr[q] = np.sqrt((yy[a] - yy[b]) ** 2 + dphi ** 2); q += 1
        out[j, 0:3] = np.sort(pm); out[j, 3:6] = np.sort(pt / pt.sum()); out[j, 6:9] = np.sort(ms); out[j, 9:12] = np.sort(dr)
    return out


def _features(P4, nn):
    O = _obs(P4, nn); F = np.ascontiguousarray(P4.astype(np.float64)); LAB = kt3_batch(F, nn); sf = _subfeats(F, nn, LAB)
    X = np.column_stack([O['mSD'], O['tau1'], O['tau2'], O['tau3'], O['tau21'], O['m12min'], O['m123'], O['n'], sf])
    return X, O


def step_c2st():
    """classifier two-sample test: edited top jets vs natural test jets matched in (m, tau32, mW); features = the other
    observables (mSD, tau1-3, tau21, smallest pair mass, three-subjet mass, multiplicity, sorted pair masses, subjet pT
    fractions, subjet masses, pair dR). Gradient-boosted trees, 5-fold out-of-fold AUC; null: AUC of the same procedure with
    shuffled labels (10 permutations) and the Mann-Whitney normal approximation."""
    from scipy.spatial import cKDTree
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from scipy.stats import norm
    from interp import data_top as dt
    E = load_edits(); A = analysis_sets(E)
    pool = dict(np.load(OUT2 / 'h2_top_obs_test.npz')); pool['tau32'] = pool['tau3'] / np.maximum(pool['tau2'], 1e-12)
    used = np.zeros(len(pool['label']), bool); used[E['T_idx']] = True; used[E['Q_idx']] = True
    full = _full_test(); res = {}
    rng = np.random.default_rng(604)

    def oof_auc(X, y, seed=0):
        from sklearn.metrics import roc_auc_score
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            p[te] = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        return roc_auc_score(y, p)

    for tag in [str(int(t)) for t in TOP_TARGETS]:
        a = A[f'T{tag}']
        for v in ('id', f'W{tag}', f'N{tag}', f'R{tag}'):
            if v == 'id': P = E['T_P4_id'][a['rows']]
            else: P = E[f'T_{v}_P4'][a['pos'][v[0]]]
            nn = E['T_nn'][a['rows']]; Xe, Oe = _features(P, nn)
            for pool_kind in ('top', 'all'):
                cand = np.flatnonzero(~used & (pool['n'] >= 3) & ((pool['label'] == 1) if pool_kind == 'top' else True))
                Z = lambda m, t3, w: np.column_stack([m / 2.0, t3 / 0.02, w / 2.0])
                tree = cKDTree(Z(pool['m'][cand], pool['tau32'][cand], pool['mW'][cand]))
                dist, nb = tree.query(Z(Oe['m'], Oe['tau32'], Oe['mW']), k=1); ridx = cand[nb]
                Pn = full[ridx][..., :4].astype(np.float32)
                nnn = np.minimum(pool['n'][ridx].astype(np.int64), Pn.shape[1])
                Xn, On = _features(Pn, nnn)
                X = np.vstack([Xe, Xn]); y = np.r_[np.ones(len(Xe)), np.zeros(len(Xn))]
                auc = oof_auc(X, y); n1 = n0 = len(Xe); se = np.sqrt((n1 + n0 + 1) / (12 * n1 * n0))
                perm = [oof_auc(X, rng.permutation(y), seed=s) for s in range(10)]
                res[f'T{tag}|{v}|{pool_kind}'] = dict(n=int(n1), auc=float(auc), perm_auc_mean=float(np.mean(perm)), perm_auc_max=float(np.max(perm)),
                                                      p_normal=float(norm.sf((auc - 0.5) / se)),
                                                      match_dist_median=float(np.median(dist)), match_dist_p90=float(np.percentile(dist, 90)),
                                                      matched_top_fraction=float(pool['label'][ridx].mean()),
                                                      d_m_median=float(np.median(np.abs(Oe['m'] - On['m']))), d_mW_median=float(np.median(np.abs(Oe['mW'] - On['mW']))),
                                                      d_tau32_median=float(np.median(np.abs(Oe['tau32'] - On['tau32']))))
                log('c2st', f'T{tag}|{v}|{pool_kind}', res[f'T{tag}|{v}|{pool_kind}'])
    (OUT / 'h4_c2st.json').write_text(json.dumps(res, indent=1)); log('wrote h4_c2st.json')


FEATS = ['mSD', 'tau1', 'tau2', 'tau3', 'tau21', 'm12min', 'm123', 'n', 'pairm_lo', 'pairm_mid', 'pairm_hi', 'ptfrac_lo', 'ptfrac_mid',
         'ptfrac_hi', 'subm_lo', 'subm_mid', 'subm_hi', 'pairdR_lo', 'pairdR_mid', 'pairdR_hi']


def step_c2stsub():
    """off-manifold follow-up for the W edits: (1) per-feature AUC edited vs matched natural top jets (which observables
    separate them); (2) out-of-fold C2ST probability per edited jet; the half of edited jets with the lowest probability of
    being edited ('most natural-looking') is re-analysed: C2ST AUC of that half vs all matched natural jets, and the
    W - N score change of every model on that half (bootstrap 95 % CI, one-sided p in the LR3 direction)"""
    from scipy.spatial import cKDTree
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    E = load_edits(); A = analysis_sets(E); S = _all_scores()
    pool = dict(np.load(OUT2 / 'h2_top_obs_test.npz')); pool['tau32'] = pool['tau3'] / np.maximum(pool['tau2'], 1e-12)
    used = np.zeros(len(pool['label']), bool); used[E['T_idx']] = True; used[E['Q_idx']] = True
    full = _full_test(); res = {}
    for tag in [str(int(t)) for t in TOP_TARGETS]:
        a = A[f'T{tag}']; rows = a['rows']; P = E[f'T_W{tag}_P4'][a['pos']['W']]; nn = E['T_nn'][rows]; Xe, Oe = _features(P, nn)
        cand = np.flatnonzero(~used & (pool['n'] >= 3) & (pool['label'] == 1))
        Z = lambda m, t3, w: np.column_stack([m / 2.0, t3 / 0.02, w / 2.0])
        _, nb = cKDTree(Z(pool['m'][cand], pool['tau32'][cand], pool['mW'][cand])).query(Z(Oe['m'], Oe['tau32'], Oe['mW']), k=1); ridx = cand[nb]
        Pn = full[ridx][..., :4].astype(np.float32); Xn, _ = _features(Pn, np.minimum(pool['n'][ridx].astype(np.int64), Pn.shape[1]))
        X = np.vstack([Xe, Xn]); y = np.r_[np.ones(len(Xe)), np.zeros(len(Xn))]; p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            p[te] = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        fa = {f: float(roc_auc_score(y, X[:, i])) for i, f in enumerate(FEATS)}
        fa = dict(sorted(fa.items(), key=lambda kv: -abs(kv[1] - 0.5)))
        pe = p[:len(Xe)]; half = pe <= np.median(pe); r = dict(n=int(len(rows)), auc_all=float(roc_auc_score(y, p)),
                                                                feature_auc=fa, n_half=int(half.sum()),
                                                                auc_half_vs_natural=float(roc_auc_score(np.r_[np.ones(half.sum()), np.zeros(len(Xn))], np.r_[pe[half], p[len(Xe):]])),
                                                                feature_medians={f: dict(edited=float(np.median(Xe[:, i])), natural=float(np.median(Xn[:, i])), id=None) for i, f in enumerate(FEATS[:11])},
                                                                models={})
        Xid, _ = _features(E['T_P4_id'][rows], nn)
        for i, f in enumerate(FEATS[:11]): r['feature_medians'][f]['id'] = float(np.median(Xid[:, i]))
        sub = rows[half]
        for k in MODELS:
            if f'{k}|T|id' not in S: continue
            s0 = S[f'{k}|T|id'][sub]; dW = S[f'{k}|T|W{tag}'][sub] - s0; dN = S[f'{k}|T|N{tag}'][sub] - s0
            rng = np.random.default_rng(8); bi = rng.integers(0, len(sub), (B, len(sub))); diff = (dW - dN)[bi].mean(1)
            r['models'][k] = dict(dW=float(dW.mean()), dN=float(dN.mean()), W_minus_N=float((dW - dN).mean()), ci95=np.percentile(diff, [2.5, 97.5]).tolist(),
                                  p_one_sided=float((np.sum(diff >= 0) + 1) / (B + 1)))
        res[f'T{tag}'] = r; log('c2stsub', f'T{tag}', {k: v for k, v in r.items() if k != 'models'})
        for k, m in r['models'].items(): log(f'   {k:5s} W-N {m["W_minus_N"]:+.3f} {np.round(m["ci95"], 3)} (all-jet analysis set: see h4_results)')
    (OUT / 'h4_c2stsub.json').write_text(json.dumps(res, indent=1)); log('wrote h4_c2stsub.json')


# -------------------------------------------------------------------------------------------------------------------- nets
KINF = [(1.7, 0.7), (2.0, 0.7), (-4.7, 0.7), (-4.7, 0.7), (0.2, 4.0)]


def jet_inputs(P4, nn):
    """Top Landscape conversion (particle_transformer utils/convert_top_datasets.py) from (px, py, pz, E): jet = constituent sum,
    part_deta = (eta_i - jet_eta) * sign(jet_eta), part_dphi = phi_i - phi_jet wrapped"""
    P = P4.astype(np.float64); L = P.shape[1]; mask = np.arange(L)[None, :] < nn[:, None]
    S = (P * mask[..., None]).sum(1); jpt = np.hypot(S[:, 0], S[:, 1]); jeta = np.arcsinh(S[:, 2] / jpt); jphi = np.arctan2(S[:, 1], S[:, 0])
    pt = np.hypot(P[..., 0], P[..., 1]); ptc = np.where(mask, pt, 1.0)
    eta = np.arcsinh(np.where(mask, P[..., 2], 0) / ptc); phi = np.arctan2(P[..., 1], P[..., 0])
    sg = np.where(jeta < 0, -1.0, 1.0)[:, None]
    deta = np.where(mask, (eta - jeta[:, None]) * sg, 0); dphi = np.where(mask, (phi - jphi[:, None] + np.pi) % (2 * np.pi) - np.pi, 0)
    return dict(px=P[..., 0], py=P[..., 1], pz=P[..., 2], energy=P[..., 3], pt=np.where(mask, pt, 0), deta=deta, dphi=dphi, mask=mask,
                jet_pt=jpt, jet_energy=S[:, 3])


def weaver_inputs(J, P128=128):
    """(points (N,2,P), features (N,7,P), vectors (N,4,P), mask (N,1,P)) as top_kin.yaml: first 128 constituents,
    clip((x - c) * s, -5, 5) for the five log/dR features, raw deta/dphi, wrap padding (masked)"""
    m = J['mask'][:, :P128]; P = int(m.sum(1).max()); m = m[:, :P]; cnt = m.sum(1)
    wrap = np.arange(P)[None, :] % np.maximum(cnt, 1)[:, None]
    fill = lambda a: np.where(m, a[:, :P], np.take_along_axis(a[:, :P], wrap, 1))
    one = lambda a: np.where(J['mask'], a, 1.0)
    raw = [np.log(one(J['pt'])), np.log(one(J['energy'])), np.log(one(J['pt']) / J['jet_pt'][:, None]),
           np.log(one(J['energy']) / J['jet_energy'][:, None]), np.hypot(J['deta'], J['dphi'])]
    feats = [fill(np.where(J['mask'], np.clip((x - c) * s, -5, 5), 0)[:, :P128]) for x, (c, s) in zip(raw, KINF)]
    feats += [fill(J['deta'][:, :P128]), fill(J['dphi'][:, :P128])]
    vec = np.stack([fill(J[k][:, :P128]) for k in ('px', 'py', 'pz', 'energy')], 1)
    pts = np.stack([fill(J['deta'][:, :P128]), fill(J['dphi'][:, :P128])], 1)
    f32 = lambda a: np.ascontiguousarray(a, np.float32)
    return f32(pts), f32(np.stack(feats, 1)), f32(vec), f32(m[:, None, :])


def _strip(sd):
    out = {}
    for k, v in sd.items():
        for p in ('module.', 'mod.'):
            if k.startswith(p): k = k[len(p):]
        out[k] = v
    return out


class WeaverNet:
    def __init__(self, kind, ckpt, dev):
        import torch
        self.torch, self.dev, self.kind = torch, dev, kind
        if kind == 'ParT':
            from weaver.nn.model.ParticleTransformer import ParticleTransformer
            self.model = ParticleTransformer(input_dim=7, num_classes=2, pair_input_dim=4, use_pre_activation_pair=False, embed_dims=[128, 512, 128],
                                             pair_embed_dims=[64, 64, 64], num_heads=8, num_layers=8, num_cls_layers=2, block_params=None,
                                             cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0}, fc_params=[],
                                             activation='gelu', trim=True, for_inference=False)
        else:
            from weaver.nn.model.ParticleNet import ParticleNet
            self.model = ParticleNet(input_dims=7, num_classes=2, conv_params=[(16, (64, 64, 64)), (16, (128, 128, 128)), (16, (256, 256, 256))],
                                     fc_params=[(256, 0.1)], use_fusion=False, use_fts_bn=True, use_counts=True, for_inference=False)
        self.model.load_state_dict(_strip(torch.load(ckpt, map_location='cpu')), strict=True); self.model.eval().to(dev)

    def score(self, P4, nn, batch=512):
        torch = self.torch; out = np.zeros(len(nn))
        with torch.no_grad():
            for s in range(0, len(nn), batch):
                J = jet_inputs(P4[s:s + batch], nn[s:s + batch]); pts, x, v, m = weaver_inputs(J)
                d = lambda a: torch.from_numpy(a).to(self.dev)
                o = (self.model(d(x), v=d(v), mask=d(m)) if self.kind == 'ParT' else self.model(d(pts), d(x), d(m))).float().cpu().numpy()
                out[s:s + len(o)] = o[:, 0] - o[:, 1]          # class 0 = jet_isTop (top_kin.yaml label order)
        return out


class LNet:
    """released LorentzNet top model (n_scalar 2, n_hidden 72, n_layers 6, c_weight 0.005), inputs as the authors' collate_fn
    (add_beams=True, beam_mass=1, scale=1): beams (sqrt2, 0, 0, +-1) first, constituents (E, px, py, pz); node scalars
    psi(one-hot(beam label) * |p^2|^0.5); fully connected edges between real nodes; batches of 32 in a fixed random order
    (the network averages node features over the batch's padded length, so batch composition is kept fixed across variants)"""

    def __init__(self, dev, mass64=True):
        import importlib.util, torch
        spec = importlib.util.spec_from_file_location('lnet_models', str(V / 'ext/LorentzNet-release/models.py'))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        self.torch, self.dev, self.mod, self.mass64 = torch, dev, mod, mass64
        self.model = mod.LorentzNet(n_scalar=2, n_hidden=72, n_class=2, dropout=0.2, n_layers=6, c_weight=0.005)
        self.model.load_state_dict(_strip(torch.load(V / 'models/top/lorentznet/best-val-model.pt', map_location='cpu')), strict=True)
        self.model.eval().to(dev)

    def score(self, P4, nn, order, batch=32):
        torch = self.torch; out = np.zeros(len(nn))
        with torch.no_grad():
            for s in range(0, len(nn), batch):
                ix = order[s:s + batch]; nmax = int(nn[ix].max()); b = len(ix); N = nmax + 2
                Pm = np.zeros((b, N, 4), np.float32)
                Pm[:, 0] = [np.sqrt(2.0), 0, 0, 1]; Pm[:, 1] = [np.sqrt(2.0), 0, 0, -1]
                Pm[:, 2:, 0] = P4[ix, :nmax, 3]; Pm[:, 2:, 1:] = P4[ix, :nmax, :3]
                Pt = torch.from_numpy(Pm)
                Pm2 = Pt.double() if self.mass64 else Pt
                mass = self.mod.normsq4(Pm2).abs().sqrt().float()
                nodes = torch.zeros(b, N, 2); nodes[:, :2, 1] = mass[:, :2]; nodes[:, 2:, 0] = mass[:, 2:]
                nodes = self.mod.psi(nodes)
                am = Pt[..., 0] != 0; em = am[:, :, None] & am[:, None, :] & ~torch.eye(N, dtype=torch.bool)[None]
                bi, ii, jj = torch.nonzero(em, as_tuple=True); edges = [(bi * N + ii).to(self.dev), (bi * N + jj).to(self.dev)]
                pred = self.model(scalars=nodes.view(b * N, -1).to(self.dev), x=Pt.view(b * N, -1).to(self.dev), edges=edges,
                                  node_mask=am.view(b * N, -1).to(self.dev).float(), edge_mask=em.reshape(b * N * N, -1).to(self.dev), n_nodes=N)
                o = pred.float().cpu().numpy(); out[ix] = o[:, 1] - o[:, 0]     # LorentzNet label 1 = top (is_signal)
        return out


def _ckpt_info(d):
    tl = (d / 'train.log').read_text(errors='ignore') if (d / 'train.log').exists() else ''
    ep = re.findall(r'Epoch #(\d+): Current validation metric: ([0-9.]+) \(best: ([0-9.]+)\)', tl)
    best = [int(e) for e, v, b in ep if v == b]
    done = 'Test metric' in tl or (d / 'predict_output').exists() or 'Epoch #19: Current validation metric' in tl
    return dict(epochs_validated=len(ep), best_epoch=best[-1] if best else None, best_val_acc=float(ep[-1][2]) if ep else None,
                training_finished=bool(done))


def step_nets():
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    E = load_edits(); scores = {}; info = {}
    nets = {}
    for kind, dname in (('ParT', 'part_kin'), ('PN', 'pnet_kin')):
        d = V / 'models/top' / dname; fin = d / 'net_best_epoch_state.pt'
        snap = OUT / f'h4_ckpt_{dname}.pt'; shutil.copy(fin, snap)
        info[kind] = dict(_ckpt_info(d), checkpoint=str(fin), md5=hashlib.md5(snap.read_bytes()).hexdigest())
        nets[kind] = WeaverNet(kind, snap, dev); log(kind, info[kind])
    nets['LNet'] = LNet(dev, mass64=True); info['LNet'] = dict(checkpoint=str(V / 'models/top/lorentznet/best-val-model.pt'), released=True)
    rng = np.random.default_rng(605)
    for name in ('T', 'Q'):
        nn = E[f'{name}_nn']; P4id = E[f'{name}_P4_id']; order = rng.permutation(len(nn))
        for v in VARS[name]:
            P = P4id.copy()
            if v != 'id': P[E[f'{name}_{v}_rows']] = E[f'{name}_{v}_P4']
            for k, net in nets.items():
                t = time.time()
                scores[f'{k}|{name}|{v}'] = net.score(P, nn, order) if k == 'LNet' else net.score(P, nn)
                log(f'  {k} {name} {v}: {len(nn)} jets, {time.time() - t:.0f}s')
        if name == 'T':   # input-path checks on the 10k sample: stored arrays (no rebuild) and LorentzNet with float32 masses
            Pst = E['T_P4_stored']
            for k in ('ParT', 'PN'): scores[f'{k}|T|stored'] = nets[k].score(Pst, nn)
            scores['LNet|T|stored'] = nets['LNet'].score(Pst, nn, order)
            scores['LNet32|T|id'] = LNet(dev, mass64=False).score(P4id, nn, order)
            J = jet_inputs(Pst, nn); info['deta_check_max_abs'] = float(np.max(np.abs(J['deta'] - E['T_deta_stored'])))
            info['dphi_check_max_abs'] = float(np.max(np.abs(J['dphi'] - E['T_dphi_stored'])))
            log('input checks', {k: info[k] for k in ('deta_check_max_abs', 'dphi_check_max_abs')})
    np.savez(OUT / 'h4_scores_nets.npz', **scores); (OUT / 'h4_nets_info.json').write_text(json.dumps(info, indent=1))
    log('wrote h4_scores_nets.npz')


# --------------------------------------------------------------------------------------------------------------- EFN / PFN
class Flow:
    """numpy forward pass of the H2 energyflow EFN / PFN (models/top/<arch>/weights.npz, keys '<layer>_<name>_<k>'):
    Phi = 3 ReLU dense layers per particle, latent = sum_i z_i Phi(deta_i, dphi_i) (EFN) or sum_i Phi(z_i, deta_i, dphi_i)
    over particles with any nonzero input (PFN, energyflow masking), F = 3 ReLU dense layers, linear output (softmax(2)).
    Inputs as interp.h2_efnpfn_train.feats on kin3 (float16 pT, deta, dphi of the 128 hardest constituents)."""

    def __init__(self, arch):
        self.arch = arch; W = np.load(V / 'models/top' / arch / 'weights.npz'); keys = sorted(W.files)
        layers = sorted({k.rsplit('_', 1)[0] for k in keys}, key=lambda s: int(s.split('_')[0]))
        Ls = [(W[f'{l}_0'].astype(np.float32), W[f'{l}_1'].astype(np.float32)) for l in layers]
        self.phi, self.F, self.out = Ls[:3], Ls[3:6], Ls[6]
        self.info = json.loads((V / 'models/top' / arch / 'info.json').read_text())

    def score(self, k3, batch=2000):
        out = np.zeros(len(k3))
        for s in range(0, len(k3), batch):
            x = np.asarray(k3[s:s + batch], np.float32); pt = x[..., 0]; real = pt > 0
            z = pt / np.maximum(pt.sum(1, keepdims=True), 1e-12); ang = x[..., 1:3] * real[..., None]
            h = ang if self.arch == 'efn' else np.concatenate([z[..., None], ang], -1)
            mask = z if self.arch == 'efn' else np.any(h != 0, -1).astype(np.float32)
            for K, b in self.phi: h = np.maximum(h @ K + b, 0)
            L = np.einsum('jn,jnk->jk', mask, h)
            for K, b in self.F: L = np.maximum(L @ K + b, 0)
            lo = L @ self.out[0] + self.out[1]; out[s:s + len(lo)] = lo[:, 1] - lo[:, 0]     # label 1 = top
        return out


def kin3_of(P4, nn, L=128):
    J = jet_inputs(P4, nn)
    return np.stack([J['pt'][:, :L], J['deta'][:, :L], J['dphi'][:, :L]], -1).astype(np.float16).astype(np.float32)


def step_ef():
    """EFN / PFN scores (final H2 models; CPU numpy) for every variant: edited rows scored, others = identity scores"""
    E = load_edits(); flows = {a.upper(): Flow(a) for a in ('pfn', 'efn')}; out = {}; info = {}
    from interp import data_top as dt
    k3st = np.asarray(np.load(dt.DST / 'test' / 'kin3.npy', mmap_mode='r')[E['T_idx']], np.float32)
    for k, f in flows.items():
        ref = np.load(V / 'models/top' / k.lower() / 'test_prob.npy')[E['T_idx']].astype(np.float64)
        ref = np.log(np.clip(ref[:, 1], 1e-7, 1)) - np.log(np.clip(ref[:, 0], 1e-7, 1))
        s_st = f.score(k3st); info[k] = dict(f.info, max_abs_dlogit_vs_saved_test_prob=float(np.max(np.abs(np.clip(s_st, -15, 15) - np.clip(ref, -15, 15)))))
        out[f'{k}|T|stored'] = s_st; log(k, 'check vs saved test_prob on the 10k sample:', info[k]['max_abs_dlogit_vs_saved_test_prob'])
    for name in ('T', 'Q'):
        nn = E[f'{name}_nn']
        for v in VARS[name]:
            P = E[f'{name}_P4_id'] if v == 'id' else E[f'{name}_{v}_P4']; n_ = nn if v == 'id' else nn[E[f'{name}_{v}_rows']]
            k3 = np.concatenate([kin3_of(P[s:s + 5000], n_[s:s + 5000]) for s in range(0, len(n_), 5000)]) if len(n_) else np.zeros((0, 128, 3), np.float32)
            for k, f in flows.items():
                x = f.score(k3)
                if v == 'id': out[f'{k}|{name}|id'] = x
                else: full = out[f'{k}|{name}|id'].copy(); full[E[f'{name}_{v}_rows']] = x; out[f'{k}|{name}|{v}'] = full
            log(f'  EF {name} {v}: {len(n_)} jets scored')
    np.savez(OUT / 'h4_scores_ef.npz', **out); (OUT / 'h4_ef_info.json').write_text(json.dumps(info, indent=1, default=float)); log('wrote h4_scores_ef.npz')


# ------------------------------------------------------------------------------------------------------------------- stats
MODELS = ['ParT', 'PN', 'LNet', 'PFN', 'EFN', 'LR2', 'LR3', 'MLP3', 'LR3t', 'MLP3t']


def _all_scores():
    S = {**dict(np.load(OUT / 'h4_scores_classical.npz')), **dict(np.load(OUT / 'h4_scores_nets.npz'))}
    if (OUT / 'h4_scores_ef.npz').exists(): S.update(dict(np.load(OUT / 'h4_scores_ef.npz')))
    return S


def _auc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _boot_mean(x, rng, b=B):
    i = rng.integers(0, len(x), (b, len(x))); m = x[i].mean(1)
    return float(x.mean()), np.percentile(m, [2.5, 97.5]).tolist(), m


def _mcnemar(a, b):
    """paired calls a (before/control) vs b (edit): n01 = pass->fail, n10 = fail->pass; exact two-sided binomial p"""
    from scipy.stats import binomtest
    n01 = int((a & ~b).sum()); n10 = int((~a & b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return dict(pass_to_fail=n01, fail_to_pass=n10, p=float(p))


def _holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1.0, ps[i] * (m - r))); adj[i] = run
    return adj


def step_stats():
    E = load_edits(); A = analysis_sets(E); S = _all_scores()
    ninfo = json.loads((OUT / 'h4_nets_info.json').read_text()) if (OUT / 'h4_nets_info.json').exists() else {}
    rng = np.random.default_rng(606); R = dict(models=MODELS, nets_info=ninfo, edit_info=json.loads((OUT / 'h4_edit_info.json').read_text()))
    yT = E['T_label']; topT = yT == 1
    # unedited performance: AUC / acc on the 10k sample; R30 / R50 with 5,000 tops (thresholds) and 100k QCD; thresholds
    perf, thr = {}, {}
    for k in MODELS + ['LNet32']:
        if f'{k}|T|id' not in S: continue
        s = S[f'{k}|T|id']; sq = S.get(f'{k}|Q|id')
        t30, t50 = np.quantile(s[topT], 0.7), np.quantile(s[topT], 0.5)
        P = dict(AUC_sample=float(_auc(s, yT)), acc_sample=float(((s > 0) == topT).mean()) if k not in ('LR2', 'LR3', 'LR3t') else None)
        bs = []
        for _ in range(200):
            i = rng.integers(0, len(s), len(s)); bs.append(_auc(s[i], yT[i]))
        P['AUC_ci95'] = np.percentile(bs, [2.5, 97.5]).tolist()
        if sq is not None:
            P['R30'] = float(1 / max((sq >= t30).mean(), 1e-9)); P['R50'] = float(1 / max((sq >= t50).mean(), 1e-9))
            tB = np.quantile(sq, 0.99); P['eps_S_at_epsB_1pct'] = float((s[topT] >= tB).mean())
            bb = [(1 / max((sq[i] >= t30).mean(), 1e-9), 1 / max((sq[i] >= t50).mean(), 1e-9)) for i in (rng.integers(0, len(sq), len(sq)) for _ in range(200))]
            P['R30_ci95'] = np.percentile([b[0] for b in bb], [2.5, 97.5]).tolist(); P['R50_ci95'] = np.percentile([b[1] for b in bb], [2.5, 97.5]).tolist()
            thr[k] = dict(eS30=t30, eS50=t50, eB1=tB)
        if f'{k}|T|stored' in S:
            P['max_abs_dlogit_stored_vs_identity'] = float(np.max(np.abs(S[f'{k}|T|stored'] - s)))
            P['calls_changed_stored_vs_identity'] = int(((S[f'{k}|T|stored'] > 0) != (s > 0)).sum())
        perf[k] = P
    R['unedited'] = perf; R['thresholds'] = {k: {a: float(b) for a, b in v.items()} for k, v in thr.items()}
    R['published'] = {'ParT': dict(AUC=0.9858, R30=1602, R50=413), 'PN': dict(AUC=0.9858, R30=1615, R50=397), 'LNet': dict(AUC=0.9868, R30=2195, R50=498, released_score_file_AUC=0.9870)}
    if (OUT / 'h4_ef_info.json').exists():
        for k, v in json.loads((OUT / 'h4_ef_info.json').read_text()).items(): R['published'][k] = dict(own_full_test=v['test'], check=v['max_abs_dlogit_vs_saved_test_prob'])
    # edits
    E_res = {}; tests = []
    for key, a in A.items():
        name, tag, rows = a['set'], a['tag'], a['rows']; n = len(rows)
        if n == 0: continue
        O0 = obs_of(E, name, 'id'); sub = {}
        for v in 'WNR':
            O = obs_of(E, name, f'{v}{tag}'); p = a['pos'][v]
            sub[v] = {q: dict(median_before=float(np.median(O0[q][rows])), median_after=float(np.median(O[q][p])),
                              median_abs_change=float(np.median(np.abs(O[q][p] - O0[q][rows]))), p90_abs_change=float(np.percentile(np.abs(O[q][p] - O0[q][rows]), 90)))
                      for q in ('m', 'mSD', 'tau32', 'tau21', 'tau2', 'mW', 'm12min', 'm123')}
            sub[v]['D_median'] = float(np.median(E[f'{name}_{v}{tag}_D'][p]))
        sub['N_pairW_change_max'] = float(np.max(np.abs(E[f'{name}_N{tag}_pairW'][a['pos']['N']] - E[f'{name}_N{tag}_mW0'][a['pos']['N']])))
        res = dict(n_eligible=a['n_eligible'], n_W_accepted=a['n_W_acc'], n_N_accepted=a['n_N_acc'], n_analysis=int(n), observables=sub, models={})
        sign = -1 if name == 'T' else 1               # direction of the LR3 response (W moved off 80: down; W put at 80 in QCD: up)
        for k in MODELS:
            if f'{k}|{name}|id' not in S: continue
            s0 = S[f'{k}|{name}|id'][rows]; d = {}
            for v in 'WNR':
                sv = S[f'{k}|{name}|{v}{tag}'][rows]; mu, ci, bm = _boot_mean(sv - s0, np.random.default_rng(7)); d[v] = dict(mean=mu, ci95=ci, _b=bm)
            M = dict(delta_score={v: dict(mean=d[v]['mean'], ci95=d[v]['ci95']) for v in 'WNR'})
            for c in 'NR':
                diff = d['W']['_b'] - d[c]['_b']; mu = d['W']['mean'] - d[c]['mean']
                p = (np.sum(sign * diff <= 0) + 1) / (len(diff) + 1)
                M[f'W_minus_{c}'] = dict(mean=float(mu), ci95=np.percentile(diff, [2.5, 97.5]).tolist(), p_one_sided=float(p))
            sd0 = float(np.std(S[f'{k}|T|id'])); M['score_sd_on_sample'] = sd0
            M['W_minus_N_in_sd'] = M['W_minus_N']['mean'] / sd0
            if k in thr:
                wp = {}
                for wn in ('eS30', 'eS50', 'eB1'):
                    t = thr[k][wn]; c0 = s0 >= t; cW = S[f'{k}|{name}|W{tag}'][rows] >= t; cN = S[f'{k}|{name}|N{tag}'][rows] >= t; cR = S[f'{k}|{name}|R{tag}'][rows] >= t
                    w = dict(pass_before=float(c0.mean()), pass_W=float(cW.mean()), pass_N=float(cN.mean()), pass_R=float(cR.mean()),
                             mcnemar_W_vs_before=_mcnemar(c0, cW), mcnemar_W_vs_N=_mcnemar(cN, cW), mcnemar_N_vs_before=_mcnemar(c0, cN))
                    if name == 'Q':   # background rejection over all 100k QCD jets, edited rows substituted
                        sq0 = S[f'{k}|Q|id']; full = {}
                        for v in ('id', 'W', 'N', 'R'):
                            sv = sq0.copy()
                            if v != 'id': sv[rows] = S[f'{k}|Q|{v}{tag}'][rows]
                            full[v] = sv >= t
                        bb = rng.integers(0, len(sq0), (300, len(sq0)))
                        for v in full:
                            eb = full[v].mean(); w[f'invEpsB_{v}'] = float(1 / eb) if eb > 0 else None
                            ebb = full[v][bb].mean(1); w[f'invEpsB_{v}_ci95'] = np.percentile(1 / np.maximum(ebb, 1e-9), [2.5, 97.5]).tolist()
                        w['n_QCD_passing_before'] = int(full['id'].sum())
                    wp[wn] = w
                M['working_points'] = wp
            res['models'][k] = M; tests.append((key, k, M['W_minus_N']['p_one_sided']))
        E_res[key] = res
    adj = _holm([t[2] for t in tests])
    for (key, k, _), pa in zip(tests, adj): E_res[key]['models'][k]['W_minus_N']['p_holm'] = float(pa)
    R['n_tests_holm'] = len(tests); R['edits'] = E_res
    if (OUT / 'h4_c2st.json').exists(): R['c2st'] = json.loads((OUT / 'h4_c2st.json').read_text())
    (OUT / 'h4_results.json').write_text(json.dumps(R, indent=1, default=float)); log('wrote h4_results.json')
    for key, r in E_res.items():
        log(key, f"n {r['n_analysis']}/{r['n_eligible']}")
        for k, M in r['models'].items():
            log(f"  {k:5s} dW {M['delta_score']['W']['mean']:+.3f} dN {M['delta_score']['N']['mean']:+.3f} dR {M['delta_score']['R']['mean']:+.3f} "
                f"W-N {M['W_minus_N']['mean']:+.3f} {np.round(M['W_minus_N']['ci95'], 3)} p_holm {M['W_minus_N']['p_holm']:.3g}")


def step_fig():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = json.loads((OUT / 'h4_results.json').read_text()); keys = [k for k in ('T60', 'T100', 'Q80') if k in R['edits']]
    titles = {'T60': 'top jets: W pair 80 → 60 GeV', 'T100': 'top jets: W pair 80 → 100 GeV', 'Q80': 'QCD jets: best pair → 80.4 GeV'}
    lab = {'ParT': 'ParT', 'PN': 'PNet', 'LNet': 'LNet', 'PFN': 'PFN', 'EFN': 'EFN', 'LR2': 'LR2\n(neg.)', 'LR3': 'LR3\n(pos.)',
           'MLP3': 'MLP3\n(pos.)', 'LR3t': 'LR3t', 'MLP3t': 'MLP\nτ21'}
    fig, ax = plt.subplots(1, len(keys), figsize=(6.4 * len(keys), 4.6), sharey=False)
    for a, key in zip(np.atleast_1d(ax), keys):
        r = R['edits'][key]; ms = [m for m in R['models'] if m in r['models'] and m != 'LR3t']; x = np.arange(len(ms))
        for off, (v, col, nm) in zip((-0.27, 0, 0.27), (('W', 'C3', 'W-pair edit'), ('N', 'C0', 'non-W pair control'), ('R', 'C7', 'rotation control'))):
            vals = [100 * (r['models'][m]['working_points']['eS50'][f'pass_{v}'] - r['models'][m]['working_points']['eS50']['pass_before']) for m in ms]
            a.bar(x + off, vals, 0.26, color=col, label=nm)
        a.axhline(0, color='k', lw=0.6); a.set_xticks(x); a.set_xticklabels([lab[m] for m in ms], fontsize=8)
        a.set_title(f"{titles[key]} (n = {r['n_analysis']})", fontsize=9)
        a.set_ylabel('change in fraction passing the\n$\\varepsilon_S$ = 0.5 cut [percentage points]', fontsize=8)
    np.atleast_1d(ax)[0].legend(frameon=False, fontsize=7)
    fig.suptitle('Top Landscape: moving the W-pair mass at fixed jet mass and $\\tau_{32}$ (red) changes the network calls in the direction of LR3 / MLP3, more than the non-W control (blue)\n'
                 'LR2 = LR($m_{SD}$, $\\tau_{32}$); LR3 / MLP3 = LR / MLP on ($m_{SD}$, $\\tau_{32}$, $m_W$); MLP $\\tau_{21}$ = MLP on ($m_{SD}$, $\\tau_{32}$, $\\tau_{21}$)', fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / 'h4_fig_wmass.png', dpi=120); log('wrote h4_fig_wmass.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--step', default='edit,classical,c2st'); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for st in a.step.split(','):
        log(f'== step {st}'); globals()[f'step_{st}']()
    log('done')
