"""H50 (loop D, claim D18; = proposed H47): separate the prong momentum fraction z from the prong separation Delta R, D2 and the
reclustered prong masses.

Background. H45 / H45v: every tagger moves its W/Z score when momentum is moved between the two prongs at fixed jet four-momentum
(edit D). At fixed jet mass m and pT, m^2 ~ m_a^2 / z + m_b^2 / (1 - z) + z (1 - z) pT^2 Delta R^2, so z and Delta R cannot vary
independently at fixed four-momentum, and H45v found that D also moves the RECLUSTERED prong masses by -3.5 to +6.6 GeV. This node
uses three arms.

Prongs: two exclusive-kT subjets (p = 1, E-scheme) of all constituents (H45.kt2_labels). z = vector-pT fraction of the softer
prong; Delta R = rapidity-azimuth distance of the two prong four-vectors; m_hi, m_lo = heavier / lighter prong mass. Every edit
below keeps each constituent's mass and its slot (track / PID features stay with it). Coordinates of an edit: (pT, rapidity y,
azimuth phi, m) per constituent; a rigid translation of one prong in (y, phi) is a longitudinal boost plus a rotation about the
beam, so it leaves that prong's mass and the constituent pT exactly unchanged.

Arm A (z and Delta R move together, prong masses held):
  D(dc)   H45 edit, dc in DCS_A: rest-frame rotation of the decay axis, |cos theta*| -> |cos theta*| + dc, jet four-momentum exact.
  A(dc)   D(dc), then recluster (kT-2) and restore each reclustered prong's mass to the mass of the original prong it overlaps
          most in pT (= the original reclustered prong mass, which D keeps for the original particle assignment): the
          constituent offsets of prong k from its axis in (y, phi) are scaled by s_k (bisection) and the prong is translated back
          onto its axis; then the two prongs are moved rigidly along their joining line (scale lambda about the pT-weighted
          centre, bisection) so that the jet mass equals the original. Reclustered again; repeated (<= NROUND rounds) until the
          reclustering reproduces the partition that was restored. Asserted: prong masses and jet mass equal their targets to
          1e-6 of the jet energy (float64) on every accepted jet.
  control cphi(dc) (H45): rest-frame rotation by the same angle about the jet direction (z, Delta R unchanged).
Arm B (Delta R at fixed z and fixed prong masses; jet mass changes):
  B(lam)  both prongs translated rigidly in (y, phi) along their joining line about the pT-weighted centre, Delta R -> lam * Delta R
          (exact); constituent pT (so z) and prong masses exact; jet mass changes by dm_B.
  controls with the same jet mass change and Delta R, z fixed:
    MBs   every constituent pT scaled by a common f (bisection to m_B) at fixed y, phi, m: z, Delta R exact; pT and prong masses
          scale by ~f.
    MBp   offsets of both prongs from their own axes scaled by a common s (bisection to m_B), prongs translated back onto their
          axes: Delta R exact, z ~ fixed, prong masses change (pT fixed).
Arm C (z at fixed Delta R and fixed prong masses; jet mass changes):
  C(dz)   z -> z1 = clip(z + dz, ZMIN, 0.5): constituent pT of the softer prong scaled by z1 W / w_s and of the harder by
          (1 - z1) W / w_h (W = w_s + w_h, prong vector pTs; jet scalar pT kept), each prong's offsets rescaled (bisection) so its
          mass equals the original, each prong translated back onto its original axis (Delta R exact). Jet mass changes by dm_C.
  controls: BC (arm-B edit with lam solved so that the jet mass equals m_C: Delta R moves instead of z), MBsC, MBpC (as above,
          to m_C).
Acceptance: for every variant, the particles that change prong when the edited jet is reclustered (kT-2) carry < MIGMAX = 1 % of
  the jet scalar pT (arm A: the same criterion defines convergence of the restoration rounds; D and cphi: H45 acceptance only), and
  both prong centres within Delta R 0.75 of the jet axis. Contrasts use rows accepted for both members.
Samples: JetClass test WToQQ_101 / ZToQQ_101 / ZJetsToNuNu_101, 500 < pT < 1000 GeV, |eta| < 2, >= 3 particles, PER_JC random rows
  per class (rng 5001 / 5002 / 5003; h41.read_window); hls4ml jedi_core.sample(per_class=PER_HLS, seed=5002) (tops only in the
  5-class accuracy).
Models: jc ParT_full, ParT_kin, PN_full, PN_kin; nulls: 3 ParT_full with permuted weight tensors (torch seeds 25-27).
  hls JEDI-linear official n64 3-feature, dsets-d64-n64, jedis-d64-n64, tf-2b-d32-n64 (seed0); nulls: 2 redrawn JEDI (seeds 20,
  21), 2 permuted dsets (torch seeds 27, 28).
References (trained on unedited train jets; NSEED retrains; MLP(64, 64), early stopping, background resampled to the signal count):
  MLP0 (m, pT, N); MLP0+dR; MLP0+pm (m_hi, m_lo); MLP0+D2; MLP0+z; MLPx = H45 reference (m, pT, tau21, D2, N, dR); MLPx+z;
  MLPx+pm; GBDT (HistGradientBoosting) on all of m, pT, N, tau21, tau32, D2, dR, z, z_g, m_hi, m_lo, m_SD.
  jc train rows: train_100M *_002 files, NREF per class (rng 5016-5018); hls train archive without the dev split (rng 5019).
Metric: per jet Delta s = s(variant) - s(unedited), s = log-odds (H45 scores); per arm net contrast edit - control; mean, 95 %
  bootstrap CI over jets (B), t-test p, Holm over arms x contrasts x models x cells x signs (all datasets); ratio net / reference
  (bootstrap over jets jointly); McNemar on calls (threshold = median unedited score of the signal class); W-vs-Z cells per dataset.
Parts: --part jc (env-part python, CPU) | hls (conda: small nets too) | hjedi (jax env) | nets (GPU) | stats (conda) | fig
       --selftest (synthetic jet: every invariant of every arm)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h45_prong_sharing as H45   # noqa: E402

T00 = time.time()
SMOKE = int(os.environ.get('H50_SMOKE', '0'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
V = Path('/j-jepa-vol/aaron-interp')
TAG = 'h50_smoke' if SMOKE else 'h50'
PER_JC = SMOKE or 3000
PER_HLS = SMOKE or 2000
NREF = 1500 if SMOKE else 10000
NSEED = 2
B = 300 if SMOKE else 2000
NPROC = int(os.environ.get('OMP_NUM_THREADS', '4'))
DCS_A = (-0.3, 0.3)
LAMS = (0.85, 1.15)
DZS = (-0.08, 0.08)
ZMIN = 0.03
NROUND = 4
DRMAX = 0.75
TOL = 1e-6
MIGMAX = 0.01   # acceptance: particles that change prong on reclustering carry < 1 % of the jet scalar pT
ARMS = {  # arm -> list of (edit, control, label of the contrast) per parameter value; contrasts are formatted with the value
    'A': [('A', 'cphi'), ('D', 'cphi'), ('A', 'D')],
    'B': [('B', 'MBs'), ('B', 'MBp')],
    'C': [('C', 'MBsC'), ('C', 'MBpC'), ('C', 'BC')],
}
PARAMS = {'A': DCS_A, 'B': LAMS, 'C': DZS}
PRIMARY = {('A', 'cphi'), ('A', 'D'), ('B', 'MBs'), ('C', 'MBsC'), ('C', 'BC')}
VARIANTS = [('D', DCS_A), ('cphi', DCS_A), ('A', DCS_A), ('B', LAMS), ('MBs', LAMS), ('MBp', LAMS),
            ('C', DZS), ('BC', DZS), ('MBsC', DZS), ('MBpC', DZS)]
VNAMES = [f'{k}|{p}' for k, ps in VARIANTS for p in ps]
REFS = {'MLP0': ['m', 'pt', 'N'], 'MLP0+dR': ['m', 'pt', 'N', 'dR'], 'MLP0+pm': ['m', 'pt', 'N', 'msub_hi', 'msub_lo'],
        'MLP0+D2': ['m', 'pt', 'N', 'D2'], 'MLP0+z': ['m', 'pt', 'N', 'z'], 'MLPx': ['m', 'pt', 'tau21', 'D2', 'N', 'dR'],
        'MLPx+z': ['m', 'pt', 'tau21', 'D2', 'N', 'dR', 'z'], 'MLPx+pm': ['m', 'pt', 'tau21', 'D2', 'N', 'dR', 'msub_hi', 'msub_lo'],
        'GBDT': ['m', 'pt', 'N', 'tau21', 'tau32', 'D2', 'dR', 'z', 'zg', 'msub_hi', 'msub_lo', 'mSD']}
JC_NETS = ['ParT_full', 'ParT_kin', 'PN_full', 'PN_kin']
HLS_NETS = ['jedi', 'dsets', 'jedis', 'tf']


def log(*a):
    print(f'[{time.time() - T00:7.0f}s]', *a, flush=True)


H45.log = log
jsonable = H45.jsonable
wrap = H45.wrap
rap = H45.rap


# ================================================================================================================ per-jet geometry
def kin(P):
    pt = np.hypot(P[:, 0], P[:, 1]); phi = np.arctan2(P[:, 1], P[:, 0])
    m = np.sqrt(np.maximum(P[:, 3] ** 2 - (P[:, :3] ** 2).sum(1), 0)); return pt, rap(P), phi, m


def build(pt, y, phi, m):
    mt = np.sqrt(pt ** 2 + m ** 2)
    return np.stack([pt * np.cos(phi), pt * np.sin(phi), mt * np.sinh(y), mt * np.cosh(y)], -1)


def mass4(V):
    return float(np.sqrt(max(V[3] ** 2 - V[0] ** 2 - V[1] ** 2 - V[2] ** 2, 0.0)))


def paxis(pt, y, phi, m, sel):
    V = build(pt[sel], y[sel], phi[sel], m[sel]).sum(0); return float(rap(V)), float(np.arctan2(V[1], V[0])), V


def scale_prong(pt, y, phi, m, sel, s, target_axis=None):
    """offsets of prong `sel` from its four-vector axis scaled by s in (y, phi), then the prong translated so that its axis is at
    target_axis (default: the axis before scaling). Translation in (y, phi) = longitudinal boost + azimuthal rotation of the
    prong, so it keeps the prong mass; returns new (y, phi)"""
    ya, pa, _ = paxis(pt, y, phi, m, sel); ty, tp = (ya, pa) if target_axis is None else target_axis
    y2 = y.copy(); p2 = phi.copy()
    y2[sel] = ya + s * (y[sel] - ya); p2[sel] = pa + s * wrap(phi[sel] - pa)
    yb, pb, _ = paxis(pt, y2, p2, m, sel); y2[sel] += ty - yb; p2[sel] += wrap(tp - pb); return y2, p2


def separate(pt, y, phi, m, lab, lam):
    """both prongs translated rigidly along their joining line about the pT-weighted centre: Delta R -> lam * Delta R exactly"""
    a = [paxis(pt, y, phi, m, lab == k) for k in (0, 1)]; w = np.array([np.hypot(v[2][0], v[2][1]) for v in a])
    c0 = np.array([a[0][0], a[0][1]]); d = np.array([a[1][0] - a[0][0], wrap(a[1][1] - a[0][1])]); C = c0 + d * w[1] / w.sum()
    y2 = y.copy(); p2 = phi.copy()
    for k, ck in ((0, c0), (1, c0 + d)):
        sh = (lam - 1.0) * (ck - C); s = lab == k; y2[s] += sh[0]; p2[s] += sh[1]
    return y2, p2


def bisect(f, lo, hi, target, n=70):
    flo, fhi = f(lo) - target, f(hi) - target
    if not (flo <= 0 <= fhi): return None
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if f(mid) - target < 0: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def jm(pt, y, phi, m):
    return mass4(build(pt, y, phi, m).sum(0))


def pmass(pt, y, phi, m, sel):
    return mass4(build(pt[sel], y[sel], phi[sel], m[sel]).sum(0))


def restore_pm(pt, y, phi, m, lab, targets):
    """prong k offsets scaled so that its mass = targets[k]; prong axis kept"""
    for k in (0, 1):
        s = lab == k
        if s.sum() < 2:
            if abs(pmass(pt, y, phi, m, s) - targets[k]) > 1e-3: return None
            continue
        sk = bisect(lambda q: pmass(pt, *scale_prong(pt, y, phi, m, s, q), m, s), 0.02, 6.0, targets[k])
        if sk is None: return None
        y, phi = scale_prong(pt, y, phi, m, s, sk)
    return y, phi


def mass_by_sep(pt, y, phi, m, lab, target):
    lam = bisect(lambda q: jm(pt, *separate(pt, y, phi, m, lab, q), m), 0.05, 3.0, target)
    if lam is None: return None, None
    return separate(pt, y, phi, m, lab, lam), lam


def mbs(pt, y, phi, m, target):
    f = bisect(lambda q: jm(pt * q, y, phi, m), 0.2, 5.0, target)
    return None if f is None else (build(pt * f, y, phi, m), f)


def mbp(pt, y, phi, m, lab, target):
    def mk(q):
        yy, pp = y, phi
        for k in (0, 1):
            if (lab == k).sum() >= 2: yy, pp = scale_prong(pt, yy, pp, m, lab == k, q)
        return yy, pp
    s = bisect(lambda q: jm(pt, *mk(q), m), 0.02, 6.0, target)
    return None if s is None else (build(pt, *mk(s), m), s)


def arm_c(pt, y, phi, m, lab, dz, M0):
    a = [paxis(pt, y, phi, m, lab == k) for k in (0, 1)]; w = np.array([np.hypot(v[2][0], v[2][1]) for v in a])
    s_ = int(np.argmin(w)); h_ = 1 - s_; W = w.sum(); z0 = w[s_] / W; z1 = float(np.clip(z0 + dz, ZMIN, 0.5))
    if abs(z1 - z0) < 2e-3: return None, dict(reason='z_at_limit')
    pt2 = pt.copy(); pt2[lab == s_] *= z1 * W / w[s_]; pt2[lab == h_] *= (1 - z1) * W / w[h_]
    r = restore_pm(pt2, y, phi, m, lab, M0)
    if r is None: return None, dict(reason='pm_bisect')
    y2, p2 = r
    for k in (0, 1):   # back onto the original prong axes: Delta R exact
        s = lab == k; yb, pb, _ = paxis(pt2, y2, p2, m, s); y2 = y2.copy(); p2 = p2.copy(); y2[s] += a[k][0] - yb; p2[s] += wrap(a[k][1] - pb)
    return (pt2, y2, p2), dict(reason='ok', z0=z0, z1=z1)


def jet_edits(args):
    """all non-iterative variants for one jet: D, cphi (H45.edit_one), B, MBs, MBp, C, BC, MBsC, MBpC"""
    j, P, lab, seed = args
    pt, y, phi, m = kin(P); E = P[:, 3].sum(); out = {}; info = {}
    M0 = [pmass(pt, y, phi, m, lab == k) for k in (0, 1)]; mj0 = jm(pt, y, phi, m)
    rng = np.random.default_rng([seed, int(j)])
    for dc in DCS_A:
        r, inf = H45.edit_one(P, lab, dc, rng)
        if r is not None: out[f'D|{dc}'] = r['D']; out[f'cphi|{dc}'] = r['cphi']
        info[f'D|{dc}'] = inf.get('reason')
    for lam in LAMS:
        yb, pb = separate(pt, y, phi, m, lab, lam); QB = build(pt, yb, pb, m); out[f'B|{lam}'] = QB; mB = mass4(QB.sum(0))
        info[f'dm|B|{lam}'] = mB - mj0
        r = mbs(pt, y, phi, m, mB)
        if r: out[f'MBs|{lam}'] = r[0]; info[f'f|MBs|{lam}'] = r[1]
        r = mbp(pt, y, phi, m, lab, mB)
        if r: out[f'MBp|{lam}'] = r[0]; info[f's|MBp|{lam}'] = r[1]
    for dz in DZS:
        r, inf = arm_c(pt, y, phi, m, lab, dz, M0); info[f'C|{dz}'] = inf['reason']
        if r is None: continue
        QC = build(*r, m); mC = mass4(QC.sum(0)); out[f'C|{dz}'] = QC; info[f'dm|C|{dz}'] = mC - mj0
        # assert: prong masses and Delta R exactly kept
        Mc = [pmass(r[0], r[1], r[2], m, lab == k) for k in (0, 1)]
        assert max(abs(Mc[0] - M0[0]), abs(Mc[1] - M0[1])) < TOL * E, (j, Mc, M0)
        rs, lam = mass_by_sep(pt, y, phi, m, lab, mC)
        if rs is not None: out[f'BC|{dz}'] = build(pt, *rs, m); info[f'lam|BC|{dz}'] = lam
        r2 = mbs(pt, y, phi, m, mC)
        if r2: out[f'MBsC|{dz}'] = r2[0]
        r2 = mbp(pt, y, phi, m, lab, mC)
        if r2: out[f'MBpC|{dz}'] = r2[0]
    return j, out, info


def restore_job(args):
    """arm A step: jet Q (after D, possibly already restored) with reclustered labels L -> prong masses restored to the masses of
    the matched original prongs, then jet mass restored by the prong separation"""
    j, Q, L, LAB0, pt0, M0, mj0 = args
    if (L == 0).sum() < 1 or (L == 1).sum() < 1: return j, None, dict(reason='empty')
    O = np.array([[pt0[(L == k) & (LAB0 == kk)].sum() for kk in (0, 1)] for k in (0, 1)])
    mp = (0, 1) if O[0, 0] + O[1, 1] >= O[0, 1] + O[1, 0] else (1, 0); tg = [M0[mp[0]], M0[mp[1]]]
    pt, y, phi, m = kin(Q); E = Q[:, 3].sum()
    r = restore_pm(pt, y, phi, m, L, tg)
    if r is None: return j, None, dict(reason='pm_bisect')
    rs, lam = mass_by_sep(pt, r[0], r[1], m, L, mj0)
    if rs is None: return j, None, dict(reason='jm_bisect')
    Qn = build(pt, *rs, m); Mn = [pmass(pt, rs[0], rs[1], m, L == k) for k in (0, 1)]
    err = max(abs(Mn[0] - tg[0]), abs(Mn[1] - tg[1]), abs(mass4(Qn.sum(0)) - mj0))
    assert err < TOL * E, (j, err, Mn, tg)
    return j, Qn, dict(reason='ok', lam=lam, err=err / E)


def same_partition(a, b):
    return bool((a == b).all() or (a != b).all())


def migration(a, b, pt):
    """scalar-pT fraction of the particles whose prong changes between partitions a and b (label swap allowed)"""
    d = pt[a != b].sum(); s = pt[a == b].sum(); return float(min(d, s) / max(pt.sum(), 1e-12))


def pool_map(fn, args):
    import multiprocessing as mp
    if NPROC <= 1 or len(args) < 50: return [fn(a) for a in args]
    with mp.get_context('fork').Pool(NPROC) as p: return p.map(fn, args, chunksize=max(1, len(args) // (NPROC * 8)))


def run_edits(P4, nn, rows, seed, tag):
    J, L = P4.shape[:2]; LAB = H45.kt2_labels(P4, nn)
    VAR = {v: P4.copy() for v in VNAMES}; ACC = {v: np.zeros(J, bool) for v in VNAMES}; INFO = {}
    t = time.time()
    res = pool_map(jet_edits, [(j, P4[j, :nn[j]].astype(np.float64), LAB[j, :nn[j]], seed) for j in rows])
    reasons = {}
    for j, out, info in res:
        for k, Q in out.items(): VAR[k][j, :nn[j]] = Q.astype(np.float32); ACC[k][j] = True
        for k, v in info.items():
            if isinstance(v, str): reasons.setdefault(k, {}); reasons[k][v] = reasons[k].get(v, 0) + 1
            else: INFO.setdefault(k, np.full(J, np.nan))[j] = v
    log(f'[{tag}] edits (D, cphi, B, MBs, MBp, C, BC, MBsC, MBpC) on {len(rows)} jets in {time.time() - t:.0f}s; {json.dumps(reasons)}')
    # ---- acceptance: reclustering reproduces the edit's partition (D / cphi: H45 rule only) and prong centres within DRMAX
    stab = {}
    for v in VNAMES:
        if v.startswith('A|'): continue
        Lv = H45.kt2_labels(VAR[v], nn); ok = ACC[v].copy(); same = np.zeros(J, bool); pok = np.zeros(J, bool); mig = np.full(J, np.nan)
        for j in np.flatnonzero(ok):
            n = nn[j]; same[j] = same_partition(Lv[j, :n], LAB[j, :n]); pok[j] = H45.prong_offsets_ok(VAR[v][j, :n].astype(np.float64), LAB[j, :n])
            mig[j] = migration(Lv[j, :n], LAB[j, :n], np.hypot(VAR[v][j, :n, 0].astype(np.float64), VAR[v][j, :n, 1].astype(np.float64)))
        stab[v] = dict(built=int(ok.sum()), same_partition=int((ok & same).sum()), migration_lt_max=int((ok & (mig < MIGMAX)).sum()), prong_ok=int((ok & pok).sum()))
        INFO[f'mig|{v}'] = mig
        ACC[v] = ok & pok if v.split('|')[0] in ('D', 'cphi') else ok & pok & (mig < MIGMAX)
        if v.startswith('D|'): INFO[f'LAB_{v}'] = Lv
    log(f'[{tag}] acceptance (built, same partition, migration < {MIGMAX}, prong centres ok) ' + json.dumps({v: (s['built'], s['same_partition'], s['migration_lt_max'], s['prong_ok']) for v, s in stab.items()}))
    # ---- arm A: iterative restoration of the reclustered prong masses
    pt0all = np.hypot(P4[..., 0].astype(np.float64), P4[..., 1].astype(np.float64))
    M0 = {}; mj0 = {}
    for j in rows:
        n = nn[j]; pt, y, phi, m = kin(P4[j, :n].astype(np.float64)); M0[j] = [pmass(pt, y, phi, m, LAB[j, :n] == k) for k in (0, 1)]; mj0[j] = jm(pt, y, phi, m)
    INFO['A_rounds'] = {}
    for dc in DCS_A:
        v = f'A|{dc}'; src = f'D|{dc}'; cur = {j: VAR[src][j, :nn[j]].astype(np.float64) for j in np.flatnonzero(ACC[src])}
        Lc = INFO[f'LAB_{src}']; labs = {j: Lc[j, :nn[j]] for j in cur}; done = {}; why = {}
        lam = np.full(J, np.nan); rounds = np.zeros(J, int)
        for rd in range(NROUND):
            if not cur: break
            res = pool_map(restore_job, [(j, Q, labs[j], LAB[j, :nn[j]], pt0all[j, :nn[j]], M0[j], mj0[j]) for j, Q in cur.items()])
            nxt = {}; Ptmp = np.zeros((J, L, 4), np.float32); ids = []
            for j, Qn, inf in res:
                if Qn is None: why[inf['reason']] = why.get(inf['reason'], 0) + 1; continue
                Ptmp[j, :nn[j]] = Qn.astype(np.float32); ids.append(j); nxt[j] = (Qn, inf)
            if not ids: break
            ids = np.array(ids); Lr = H45.kt2_labels(Ptmp[ids], nn[ids]); cur2 = {}
            for i, j in enumerate(ids):
                Ln = Lr[i, :nn[j]]; Qn_ = nxt[j][0]
                if migration(Ln, labs[j], np.hypot(Qn_[:, 0], Qn_[:, 1])) < MIGMAX: done[j] = (nxt[j][0], labs[j]); lam[j] = nxt[j][1]['lam']; rounds[j] = rd + 1
                else: cur2[j] = nxt[j][0]; labs[j] = Ln
            why[f'round{rd + 1}_converged'] = int(len(ids) - len(cur2)); cur = cur2
        why['not_converged'] = len(cur)
        for j, (Q, Lf) in done.items():
            VAR[v][j, :nn[j]] = Q.astype(np.float32); ACC[v][j] = H45.prong_offsets_ok(Q, Lf)
        INFO[f'lam|A|{dc}'] = lam; INFO['A_rounds'][str(dc)] = why
        log(f'[{tag}] arm A dc={dc:+.2f}: {why}; accepted {int(ACC[v].sum())}')
    OBS = {'id': H45.observables(P4, nn)}
    for v in VNAMES: OBS[v] = H45.observables(VAR[v], nn)
    log(f'[{tag}] observables done')
    return VAR, ACC, OBS, INFO, LAB, dict(acceptance=stab, reasons=reasons, A_rounds=INFO['A_rounds'])


# ================================================================================================================ references
def fit_refs(Op, yp, scores, cls_of):
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from interp.h4_top_wmass import _auc
    R, info = {}, {}
    for sc, (sig, bkg) in scores.items():
        sel = np.isin(yp, [cls_of[sig]] + [cls_of[b] for b in bkg]); y = (yp[sel] == cls_of[sig]).astype(int)
        tr = np.random.default_rng(5020).random(len(y)) < 0.8; it = np.flatnonzero(tr)
        for kind, feats in REFS.items():
            Xf = np.column_stack([Op[q][sel] for q in feats]).astype(np.float64)
            for s in range(1 if kind == 'GBDT' else NSEED):
                rs = np.random.default_rng(5040 + s); pos, neg = it[y[it] == 1], it[y[it] == 0]
                neg = rs.choice(neg, min(len(neg), len(pos)), replace=False); ii = np.concatenate([pos, neg])
                scl = StandardScaler().fit(Xf[ii])
                if kind == 'GBDT': mdl = HistGradientBoostingClassifier(max_iter=300, random_state=s).fit(scl.transform(Xf[ii]), y[ii])
                else: mdl = MLPClassifier(hidden_layer_sizes=(64, 64), early_stopping=True, random_state=s, max_iter=300).fit(scl.transform(Xf[ii]), y[ii])
                pr = mdl.predict_proba(scl.transform(Xf[~tr]))[:, 1]
                R[(sc, kind, s)] = (mdl, scl, feats); info[f'{sc}|{kind}|{s}'] = float(_auc(pr, y[~tr]))
        log(f'  refs {sc}: held-out AUC ' + ', '.join(f'{k}={v:.3f}' for k, v in info.items() if k.startswith(sc + '|')))
    return R, info


def ref_scores(R, O):
    lo = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return {f'{kind}@{s}|{sc}': lo(mdl.predict_proba(scl.transform(np.column_stack([O[q] for q in feats]).astype(np.float64)))[:, 1]).astype(np.float32)
            for (sc, kind, s), (mdl, scl, feats) in R.items()}


def save_obs(path, OBS, ACC, INFO, y, Rsc, extra):
    D = {'y': y}
    for nm, O in OBS.items():
        for q in H45.OBSK: D[f'O|{nm}|{q}'] = np.asarray(O[q], np.float32)
    for v, a in ACC.items(): D[f'acc|{v}'] = a
    for k, v in INFO.items():
        if isinstance(v, np.ndarray) and v.ndim == 1: D[f'info|{k}'] = v.astype(np.float32)
    for nm, d in Rsc.items():
        for k, v in d.items(): D[f'R|{nm}|{k}'] = v
    D.update(extra); np.savez_compressed(path, **D)


def ref_pool_jc():
    from interp import h4_top_wmass as H
    from interp import h19_wpair_cross as X
    from interp.h41_colour_flow import read_window
    Os, ys, meta = [], [], {}
    for k, c in enumerate(H45.JC_CLASSES):
        Jr, inf = read_window(H45.TRAIN / f'{H45.JC_PRE[c]}_002.root', NREF, 5016 + k); meta[c] = inf
        Pr, nr = X.jc_P4(Jr); Pr = H._identity(Pr, nr).astype(np.float32); Os.append(H45.observables(Pr, nr)); ys.append(Jr['label'].astype(int))
    return {q: np.concatenate([o[q] for o in Os]) for q in H45.OBSK}, np.concatenate(ys), meta


def ref_pool_hls():
    import h5py
    from interp import h4_top_wmass as H
    with h5py.File(V / 'data' / 'train_meta.h5', 'r') as f: ya = f['label'][:].astype(np.int64)
    dev = np.asarray(np.load(V / 'data' / 'splits.npz')['dev'], np.int64); isdev = np.zeros(len(ya), bool); isdev[dev] = True
    rr = np.random.default_rng(5019); it = np.sort(np.concatenate([rr.choice(np.flatnonzero(~isdev & (ya == c)), NREF, replace=False) for c in range(4)]))
    xr = np.asarray(np.load(V / 'data' / 'train_ptetaphi_f16.npy', mmap_mode='r')[it, :64], np.float32); yr = ya[it]
    rl = xr[..., 0] > 0; nr = rl.sum(1).astype(np.int64)
    Pr = H._identity((H45.p4(xr[..., 0].astype(np.float64), xr[..., 1], xr[..., 2], 0.0) * rl[..., None]).astype(np.float32), nr).astype(np.float32)
    return H45.observables(Pr, nr), yr, dict(rows=int(len(it)), seed=5019)


# ================================================================================================================ parts
def part_jc():
    from interp import h4_top_wmass as H
    from interp import h19_wpair_cross as X
    from interp.h41_colour_flow import read_window
    import interp.h41_colour_flow as H41
    from interp.part_core import cat, kin as pkin, PK
    H41.log = log
    Js, meta = [], dict(node='H50', part='jc', sample={})
    for k, c in enumerate(H45.JC_CLASSES):
        J, inf = read_window(X.jc_file(c, 101), PER_JC, 5001 + k); Js.append(J); meta['sample'][c] = inf
    J = cat(Js); y = J['label'].astype(int); P4raw, nn = X.jc_P4(J); P4 = H._identity(P4raw, nn).astype(np.float32)
    J['pmass'] = pkin(J)[3].astype(np.float32)
    log(f'jc sample {len(nn)} jets; labels {np.bincount(y, minlength=10).tolist()}; median pT {np.median(J["jet_pt"]):.0f} GeV')
    rows = np.flatnonzero(nn >= 3)
    VAR, ACC, OBS, INFO, LAB, emeta = run_edits(P4, nn, rows, 5010, 'jc')
    D = {'nn': nn, 'y': y, 'LAB': LAB.astype(np.int16), 'P|id': P4}
    for v in VNAMES: D[f'P|{v}'] = VAR[v]
    for k in PK + ['pmass', 'jet_eta', 'jet_phi', 'jet_pt', 'label']: D[f'base|{k}'] = J[k]
    RES.mkdir(parents=True, exist_ok=True); np.savez_compressed(RES / f'{TAG}_jc_inputs.npz', **D); log('wrote jc inputs')
    Op, yp, meta['ref_rows'] = ref_pool_jc()
    R, meta['ref_auc'] = fit_refs(Op, yp, H45.JC_SCORES, H45.JC_CLASSES)
    save_obs(RES / f'{TAG}_jc_obs.npz', OBS, ACC, INFO, y, {nm: ref_scores(R, O) for nm, O in OBS.items()}, {})
    meta['edit'] = emeta; (RES / f'{TAG}_jc_meta.json').write_text(json.dumps(jsonable(meta), indent=1)); log('jc part written')


def part_hls():
    import torch
    from interp import h4_top_wmass as H
    from interp.jedi_core import sample
    from interp.h1_irc_panel import load_small, torch_logits
    x, y, _ = sample(per_class=PER_HLS, seed=5002); real = x[..., 0] > 0; nn = real.sum(1).astype(np.int64)
    P4 = H._identity(H45.p4(x[..., 0].astype(np.float64), x[..., 1], x[..., 2], 0.0) * real[..., None], nn).astype(np.float32)
    log(f'hls sample {len(y)} jets (jedi_core.sample seed 5002), per class {np.bincount(y).tolist()}')
    rows = np.flatnonzero((nn >= 3) & (y != 4))
    VAR, ACC, OBS, INFO, LAB, emeta = run_edits(P4, nn, rows, 5011, 'hls')
    D = {'y': y, 'nn': nn, 'LAB': LAB.astype(np.int16), 'P|id': P4, 'X|id': H45.model_x(P4, nn, H45.axis(P4, nn))}
    for v in VNAMES: D[f'X|{v}'] = H45.model_x(VAR[v], nn, H45.axis(VAR[v], nn)); D[f'P|{v}'] = VAR[v]
    RES.mkdir(parents=True, exist_ok=True); np.savez_compressed(RES / f'{TAG}_hls_inputs.npz', **D); log('wrote hls inputs')
    Op, yp, rmeta = ref_pool_hls()
    R, auc = fit_refs(Op, yp, H45.HLS_SCORES, H45.HLS_CLASSES)
    save_obs(RES / f'{TAG}_hls_obs.npz', OBS, ACC, INFO, y, {nm: ref_scores(R, O) for nm, O in OBS.items()}, {})
    (RES / f'{TAG}_hls_meta.json').write_text(json.dumps(jsonable(dict(node='H50', part='hls', ref_rows=rmeta, ref_auc=auc, edit=emeta)), indent=1))
    torch.set_num_threads(NPROC)
    models, known = {}, {}
    for tag, run in (('dsets', 'dsets-d64-n64'), ('jedis', 'jedis-d64-n64'), ('tf', 'tf-2b-d32-n64')):
        d = V / 'models' / 'small' / run / 'seed0'; models[tag] = load_small(d)
        tj = json.loads((d / 'test.json').read_text()); known[tag] = tj.get('accuracy', tj.get('acc'))
    for s in range(2):
        netp, sh, sc = load_small(V / 'models' / 'small' / 'dsets-d64-n64' / 'seed0'); g = torch.Generator().manual_seed(27 + s)
        with torch.no_grad():
            for p in netp.parameters():
                f = p.data.flatten(); p.data.copy_(f[torch.randperm(f.numel(), generator=g)].view_as(p))
        models[f'dsets_perm{s}'] = (netp, sh, sc)
    Zi = np.load(RES / f'{TAG}_hls_inputs.npz'); out = {}; acc = {}
    for name, (n_, sh, sc) in models.items():
        for k in [k for k in Zi.files if k.startswith('X|')]: out[f'L|{k[2:]}|{name}'] = torch_logits(n_, np.asarray(Zi[k], np.float32).copy(), sh, sc).astype(np.float32)
        acc[name] = float((out[f'L|id|{name}'].argmax(1) == y).mean()); log(f'{name}: 5-class accuracy {acc[name]:.4f} (known test {known.get(name)})')
    np.savez_compressed(RES / f'{TAG}_hls_logits_small.npz', **out)
    (RES / f'{TAG}_hls_models_small.json').write_text(json.dumps(jsonable(dict(acc=acc, known=known)), indent=1))


def part_hjedi():
    from interp.jedi_core import Jedi
    known = json.loads((V / 'data' / 'official_acc.json').read_text())['3-feature']['64']
    nets = {'jedi': Jedi(64, '3-feature')}
    for s in (20, 21): nets[f'jedi_redrawn{s - 20}'] = Jedi(64, '3-feature', untrained=True, seed=s)
    Zi = np.load(RES / f'{TAG}_hls_inputs.npz'); y = Zi['y']; out = {}; acc = {}
    for name, n_ in nets.items():
        for k in [k for k in Zi.files if k.startswith('X|')]: out[f'L|{k[2:]}|{name}'] = np.asarray(n_.run(np.asarray(Zi[k], np.float32))['logits'], np.float32)
        acc[name] = float((out[f'L|id|{name}'].argmax(1) == y).mean()); log(f'{name}: 5-class accuracy {acc[name]:.4f} (known {known})')
    np.savez_compressed(RES / f'{TAG}_hls_logits_jedi.npz', **out)
    (RES / f'{TAG}_hls_models_jedi.json').write_text(json.dumps(jsonable(dict(acc=acc, known=known)), indent=1))


def part_nets():
    import torch
    from interp.h23_axis_audit import Nets
    from interp.part_core import ParTNetwork
    import interp.h23_axis_audit as H23
    H23.log = log
    Z = np.load(RES / f'{TAG}_jc_inputs.npz'); nn = Z['nn']; base = {k.split('|', 1)[1]: Z[k] for k in Z.files if k.startswith('base|')}
    N = Nets(top=False); names = list(JC_NETS)
    for s in range(3):
        netp = ParTNetwork('full'); g = torch.Generator().manual_seed(25 + s)
        with torch.no_grad():
            for p in netp.model.parameters():
                if p.numel() > 1: p.copy_(p.flatten()[torch.randperm(p.numel(), generator=g).to(p.device)].view_as(p))
        N.jc[f'ParT_perm{s}'] = ('part', 'full', netp); names.append(f'ParT_perm{s}')
    out = {}
    for k in [k for k in Z.files if k.startswith('P|')]:
        P = Z[k]; lg = {}
        N.score('jc', P, nn, base, H45.axis(P, nn), None, jc_names=names, top=False, logits=lg)
        for m, L in lg.items(): out[f'L|{k[2:]}|{m}'] = L.astype(np.float32)
        log(f'scored {k[2:]}')
    y = Z['y']; acc = {m: float((out[f'L|id|{m}'].argmax(1) == y).mean()) for m in names}
    for m in names: log(f'{m}: 10-class accuracy on the sample {acc[m]:.4f}')
    np.savez_compressed(RES / f'{TAG}_jc_logits.npz', **out); (RES / f'{TAG}_jc_models.json').write_text(json.dumps(acc, indent=1))


# ================================================================================================================ statistics
JC_CELLS = H45.JC_CELLS
HLS_CELLS = [('W', 'WQG'), ('W', 'WZ'), ('Z', 'ZQG'), ('Z', 'WZ'), ('q', 'WQG'), ('q', 'ZQG'), ('g', 'WQG'), ('g', 'ZQG')]


def ratio_boot(x, r, bi):
    mr = r.mean()
    if abs(mr) < 0.1: return None, None
    br = r[bi].mean(1); bx = x[bi].mean(1); ok = np.abs(br) > 1e-9
    return float(x.mean() / mr), np.percentile(bx[ok] / br[ok], [2.5, 97.5]).tolist()


def stats_ds(ds):
    from interp.h4_top_wmass import _auc
    Zo = dict(np.load(RES / f'{TAG}_{ds}_obs.npz')); y = Zo['y']
    if ds == 'jc':
        Zl = dict(np.load(RES / f'{TAG}_jc_logits.npz')); cls_of, scores, cells, nets = H45.JC_CLASSES, H45.JC_SCORES, JC_CELLS, JC_NETS
        nulls = [f'ParT_perm{s}' for s in range(3)]
    else:
        Zl = dict(np.load(RES / f'{TAG}_hls_logits_small.npz')); Zl.update(dict(np.load(RES / f'{TAG}_hls_logits_jedi.npz')))
        cls_of, scores, cells, nets = H45.HLS_CLASSES, H45.HLS_SCORES, HLS_CELLS, HLS_NETS
        nulls = ['jedi_redrawn0', 'jedi_redrawn1', 'dsets_perm0', 'dsets_perm1']
    names = {v: k for k, v in cls_of.items()}; cname = np.array([names.get(int(v), str(v)) for v in y])
    refk = sorted({k.split('|')[2] for k in Zo if k.startswith('R|id|')})
    refbase = sorted({r.split('@')[0] for r in refk})

    def S(m, v, sc):
        if '@' in m: return Zo[f'R|{v}|{m}|{sc}'].astype(np.float64)
        return H45.score_of(ds, Zl[f'L|{v}|{m}'].astype(np.float64), sc)
    out = dict(dataset=ds, nets=nets, nulls=nulls, refs=refbase, unedited={}, ref_unedited_AUC={}, obs={}, cells={}, tests=[])
    for m in nets + nulls:
        L = Zl[f'L|id|{m}']; d = dict(acc=float((L.argmax(1) == y).mean()))
        if ds == 'hls': d['acc_WZqg_only'] = float((L[y != 4].argmax(1) == y[y != 4]).mean())
        for sc, (sig, bkg) in scores.items():
            sel = np.isin(cname, [sig] + bkg); d[f'AUC_{sc}'] = float(_auc(S(m, 'id', sc)[sel], (cname[sel] == sig).astype(int)))
        out['unedited'][m] = d
    for r in refk:
        for sc, (sig, bkg) in scores.items():
            sel = np.isin(cname, [sig] + bkg); out['ref_unedited_AUC'][f'{r}|{sc}'] = float(_auc(S(r, 'id', sc)[sel], (cname[sel] == sig).astype(int)))
    log(f'[{ds}] unedited ' + json.dumps({m: round(d['acc'], 4) for m, d in out['unedited'].items()}))
    rng = np.random.default_rng(5050)
    # ---- achieved observable changes per variant and class (rows accepted for that variant)
    OQ = ('z', 'dR', 'msub_hi', 'msub_lo', 'm', 'pt', 'D2', 'tau21', 'zg', 'N')
    for c in cls_of:
        if c == 't' or not (cname == c).any(): continue
        for v in VNAMES:
            rr_ = np.flatnonzero(Zo[f'acc|{v}'] & (cname == c))
            if len(rr_) < 5: continue
            d = dict(n=int(len(rr_)), n_class=int((cname == c).sum()))
            for q in OQ:
                dq = Zo[f'O|{v}|{q}'][rr_].astype(np.float64) - Zo[f'O|id|{q}'][rr_].astype(np.float64)
                d[q] = dict(mean=float(dq.mean()), median_abs=float(np.median(np.abs(dq))), id_mean=float(Zo[f'O|id|{q}'][rr_].mean()))
            out['obs'][f'{c}|{v}'] = d
        log(f'[{ds}] {c}: ' + ' | '.join(f"{v} n{out['obs'][f'{c}|{v}']['n']} dz {out['obs'][f'{c}|{v}']['z']['mean']:+.3f} dDR {out['obs'][f'{c}|{v}']['dR']['mean']:+.3f} "
                                          f"dmhi {out['obs'][f'{c}|{v}']['msub_hi']['mean']:+.2f} dmlo {out['obs'][f'{c}|{v}']['msub_lo']['mean']:+.2f} dm {out['obs'][f'{c}|{v}']['m']['mean']:+.2f} "
                                          f"dpt {out['obs'][f'{c}|{v}']['pt']['mean']:+.1f} dD2 {out['obs'][f'{c}|{v}']['D2']['mean']:+.2f}"
                                          for v in VNAMES if f'{c}|{v}' in out['obs']))
    thr = {(m, sc): float(np.median(S(m, 'id', sc)[cname == scores[sc][0]])) for m in nets + nulls for sc in scores}
    for (c, sc) in cells:
        for arm, cons in ARMS.items():
            for p in PARAMS[arm]:
                for (e, ctl) in cons:
                    ve, vc = f'{e}|{p}', f'{ctl}|{p}'
                    rows = np.flatnonzero(Zo[f'acc|{ve}'] & Zo[f'acc|{vc}'] & (cname == c))
                    key = f'{c}|{sc}|{arm}|{e}-{ctl}|{p}'; Rr = dict(n=int(len(rows)), models={})
                    if len(rows) < 20: out['cells'][key] = Rr; continue
                    bi = rng.integers(0, len(rows), (B, len(rows))); dl = {}
                    for m in nets + nulls + refk:
                        x = (S(m, ve, sc) - S(m, vc, sc))[rows]; dl[m] = x; Mm = {}
                        Mm['mean'], Mm['ci95'] = H45.boot(x, bi)
                        Mm['d_edit'] = float((S(m, ve, sc) - S(m, 'id', sc))[rows].mean()); Mm['d_control'] = float((S(m, vc, sc) - S(m, 'id', sc))[rows].mean())
                        if m in nets:
                            Mm['tp'] = H45.tp(x)
                            if (e, ctl) in PRIMARY: out['tests'].append((key, m))
                            t_ = thr[(m, sc)]; Mm['mcnemar'] = H45.mcnemar(S(m, vc, sc)[rows] >= t_, S(m, ve, sc)[rows] >= t_)
                        Rr['models'][m] = Mm
                    for rb in refbase:
                        rs_ = [r for r in refk if r.startswith(rb + '@')]; rmean = np.mean([dl[r] for r in rs_], 0)
                        Rr[f'ref_{rb}'] = dict(mean=float(rmean.mean()), range=[float(min(Rr['models'][r]['mean'] for r in rs_)), float(max(Rr['models'][r]['mean'] for r in rs_))],
                                                ci95=H45.boot(rmean, bi)[1])
                        for m in nets:
                            rt, rci = ratio_boot(dl[m], rmean, bi); Rr['models'][m][f'ratio_{rb}'] = rt; Rr['models'][m][f'ratio_{rb}_ci95'] = rci
                            dd = dl[m] - rmean; Rr['models'][m][f'minus_{rb}'], Rr['models'][m][f'minus_{rb}_ci95'] = H45.boot(dd, bi)
                    v_ = [Rr['models'][m]['mean'] for m in nulls]; Rr['null_band'] = [float(min(v_)), float(max(v_))]
                    out['cells'][key] = Rr
                    log(f'[{ds}] {key} n {len(rows)}: ' + ' '.join(f"{m} {Rr['models'][m]['mean']:+.2f}" for m in nets)
                        + ' | ' + ' '.join(f"{rb} {Rr[f'ref_{rb}']['mean']:+.2f}" for rb in refbase) + f" | null {Rr['null_band'][0]:+.2f}..{Rr['null_band'][1]:+.2f}")
    # ---- arm A collapse ratio (A - cphi) / (D - cphi), per net and reference, on rows accepted for A, D, cphi
    out['collapse'] = {}
    for (c, sc) in cells:
        for p in DCS_A:
            rows = np.flatnonzero(Zo[f'acc|A|{p}'] & Zo[f'acc|D|{p}'] & Zo[f'acc|cphi|{p}'] & (cname == c))
            if len(rows) < 20: continue
            bi = rng.integers(0, len(rows), (B, len(rows))); d = dict(n=int(len(rows)))
            for m in nets + [r for r in refk if r.endswith('@0')]:
                xa = (S(m, f'A|{p}', sc) - S(m, f'cphi|{p}', sc))[rows]; xd = (S(m, f'D|{p}', sc) - S(m, f'cphi|{p}', sc))[rows]
                rt, rci = ratio_boot(xa, xd, bi); d[m] = dict(A_cphi=float(xa.mean()), D_cphi=float(xd.mean()), ratio=rt, ratio_ci95=rci)
            out['collapse'][f'{c}|{sc}|{p}'] = d
    if ds == 'hls':   # JEDI-linear input moments M0 = sum pT, m_A = sqrt(M0 M2)
        Zi = np.load(RES / f'{TAG}_hls_inputs.npz')

        def mom(x):
            x = x.astype(np.float64); pt = x[..., 0]; M0 = pt.sum(1); M2 = (pt * (x[..., 1] ** 2 + x[..., 2] ** 2)).sum(1); return M0, np.sqrt(M0 * M2)
        M0i, mAi = mom(Zi['X|id']); out['moments'] = {}
        for v in VNAMES:
            M0v, mAv = mom(Zi[f'X|{v}'])
            for c in ('W', 'Z', 'q', 'g'):
                rr_ = np.flatnonzero(Zo[f'acc|{v}'] & (cname == c))
                if len(rr_) < 5: continue
                out['moments'][f'{v}|{c}'] = dict(dlog_mA_mean=float(np.log(mAv[rr_] / mAi[rr_]).mean()), dM0_rel_mean=float((M0v[rr_] / M0i[rr_] - 1).mean()))
    return out


def part_stats():
    R = dict(node='H50', smoke=SMOKE, per_class=dict(jc=PER_JC, hls=PER_HLS), n_ref=NREF, n_retrains=NSEED, bootstrap=B,
             dcs_A=DCS_A, lams=LAMS, dzs=DZS, sets={})
    for ds in ('jc', 'hls'):
        try:
            R['sets'][ds] = stats_ds(ds)
            mf = RES / f'{TAG}_{ds}_meta.json'
            if mf.exists(): R['sets'][ds]['meta'] = json.loads(mf.read_text())
        except FileNotFoundError as e:
            log(f'[{ds}] missing input {e}')
    lst = [(ds, k, m) for ds in R['sets'] for (k, m) in R['sets'][ds]['tests']]
    adj = H45.holm([R['sets'][ds]['cells'][k]['models'][m]['tp'] for ds, k, m in lst])
    for (ds, k, m), a in zip(lst, adj): R['sets'][ds]['cells'][k]['models'][m]['tp_holm'] = float(a)
    R['holm'] = dict(n_tests=len(lst), n_lt_0p05=int((adj < 0.05).sum()),
                     by_contrast={c: [int(sum(1 for (ds, k, m), a in zip(lst, adj) if k.split('|')[3] == c and a < 0.05)),
                                      int(sum(1 for (ds, k, m) in lst if k.split('|')[3] == c))] for c in sorted({k.split('|')[3] for _, k, _ in lst})})
    log(f'Holm {R["holm"]}')
    (RES / f'{TAG}_z_vs_dr.json').write_text(json.dumps(jsonable(R), indent=1)); log(f'wrote {TAG}_z_vs_dr.json')
    try:
        figures(R)
    except Exception:
        import traceback; traceback.print_exc()


# ================================================================================================================ figures
def figures(R=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if R is None: R = json.loads((RES / f'{TAG}_z_vs_dr.json').read_text())
    Z = np.load(RES / f'{TAG}_jc_inputs.npz'); Zo = np.load(RES / f'{TAG}_jc_obs.npz'); y = Z['y']; nn = Z['nn']; LAB = Z['LAB']
    ok = Zo['acc|A|0.3'] & Zo['acc|B|1.15'] & Zo['acc|C|0.08'] & (y == 7) & (nn >= 20)
    mm = Zo['O|id|m']; zz = Zo['O|id|z']; cand = np.flatnonzero(ok & (mm > 72) & (mm < 88) & (zz > 0.2) & (zz < 0.35))
    j = int(cand[0]) if len(cand) else int(np.flatnonzero(ok)[0])
    panels = [('unedited W jet', 'id'), ('arm A: D(+0.3), prong masses restored', 'A|0.3'), ('arm B: ΔR × 1.15 at fixed z', 'B|1.15'),
              ('arm C: z + 0.08 at fixed ΔR', 'C|0.08')]
    fig, axs = plt.subplots(1, 4, figsize=(17, 5.0), sharex=True, sharey=True)
    P0 = Z['P|id'][j, :nn[j]].astype(np.float64); T = P0.sum(0); eJ, pJ = np.arcsinh(T[2] / np.hypot(T[0], T[1])), np.arctan2(T[1], T[0])
    for ax, (title, key) in zip(axs, panels):
        P = Z[f'P|{key}'][j, :nn[j]].astype(np.float64); pt, eta, phi, _ = H45.ptetaphim(P)
        for k, col in ((0, '#1f77b4'), (1, '#d62728')):
            s = LAB[j, :nn[j]] == k
            ax.scatter(eta[s] - eJ, wrap(phi[s] - pJ), s=4 + 300 * pt[s] / pt.max(), c=col, alpha=0.65, edgecolors='none', label=f'prong {k + 1}: pT {pt[s].sum():.0f} GeV')
        g = lambda q: float(Zo[f'O|{key}|{q}'][j])
        ax.set_title(f'{title}\nz = {g("z"):.3f}, ΔR = {g("dR"):.3f}, m = {g("m"):.1f} GeV\nprong masses {g("msub_hi"):.1f} / {g("msub_lo"):.1f} GeV, D2 = {g("D2"):.2f}', fontsize=9.5)
        ax.add_patch(plt.Circle((0, 0), 0.8, fill=False, ls=':', color='grey')); ax.set_aspect('equal'); ax.legend(fontsize=8, loc='lower left')
        ax.set_xlabel('Δη from jet axis'); ax.grid(alpha=0.2)
    axs[0].set_ylabel('Δφ from jet axis [rad]'); axs[0].set_xlim(-0.85, 0.85); axs[0].set_ylim(-0.85, 0.85)
    fig.suptitle(f'H50: the three arms on one JetClass W→qq jet (pT {np.hypot(T[0], T[1]):.0f} GeV); marker area ∝ particle pT; prongs = exclusive-kT subjets', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(RES / f'{TAG}_edit.png', dpi=130); plt.close(fig); log(f'wrote {TAG}_edit.png (row {j})')
    # ---- response figure: W jets, W-vs-background score, net contrast per arm, nets vs references
    pans = [('jc', 'Wqq|WQ', 'JetClass W jets, log p(W)/p(QCD)'), ('hls', 'W|WQG', 'hls4ml W jets, log p(W)/p(q or g)')]
    pans = [p for p in pans if p[0] in R['sets']]
    cons = [('A', 'A-cphi', 0.3, 'A: D(+0.3) with\nprong masses restored\n− cφ'), ('A', 'D-cphi', 0.3, 'D(+0.3) − cφ\n(H45 edit)'),
            ('B', 'B-MBs', 1.15, 'B: ΔR×1.15\n− mass-matched\npT scaling'), ('C', 'C-MBsC', 0.08, 'C: z+0.08\n− mass-matched\npT scaling'),
            ('C', 'C-BC', 0.08, 'C: z+0.08\n− mass-matched\nΔR edit')]
    fig, axs = plt.subplots(1, len(pans), figsize=(7.5 * len(pans), 4.8), squeeze=False); axs = axs[0]
    rcol = {'MLP0+dR': '#8c564b', 'MLP0+z': '#2ca02c', 'MLP0+pm': '#9467bd', 'GBDT': '#7f7f7f'}
    for ax, (ds, cell, title) in zip(axs, pans):
        S_ = R['sets'][ds]; nets = S_['nets']; xs = np.arange(len(cons)); wdt = 0.8 / (len(nets) + len(rcol))
        for i, m in enumerate(nets):
            mu, lo, hi = [], [], []
            for arm, cn, p, _ in cons:
                cl = S_['cells'].get(f'{cell}|{arm}|{cn}|{p}', {}).get('models', {}).get(m)
                mu.append(cl['mean'] if cl else np.nan); lo.append(cl['ci95'][0] if cl else np.nan); hi.append(cl['ci95'][1] if cl else np.nan)
            mu = np.array(mu); ax.bar(xs + (i - 0.5 * (len(nets) + len(rcol) - 1)) * wdt, mu, wdt, yerr=[mu - np.array(lo), np.array(hi) - mu], capsize=2, label=m)
        for i, (rb, col) in enumerate(rcol.items()):
            mu = [S_['cells'].get(f'{cell}|{arm}|{cn}|{p}', {}).get(f'ref_{rb}', {}).get('mean', np.nan) for arm, cn, p, _ in cons]
            ax.bar(xs + (len(nets) + i - 0.5 * (len(nets) + len(rcol) - 1)) * wdt, mu, wdt, color=col, alpha=0.55, hatch='//', label=f'reference {rb}')
        ax.axhline(0, color='k', lw=0.6); ax.set_xticks(xs); ax.set_xticklabels([c[3] for c in cons], fontsize=8); ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.2, axis='y'); ax.legend(fontsize=7, ncol=2)
    axs[0].set_ylabel('mean Δ log-odds, edit − control (95 % CI)')
    fig.suptitle('H50: W-jet score response per arm; hatched = reference classifiers trained on unedited jets', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(RES / f'{TAG}_response.png', dpi=130); plt.close(fig); log(f'wrote {TAG}_response.png')


# ================================================================================================================ self-test
def selftest():
    rng = np.random.default_rng(0); P = []
    for (e, p, pt, nk) in ((0.0, 0.0, 400.0, 12), (0.20, 0.10, 150.0, 9)):
        for _ in range(nk):
            ee, pp, q = e + rng.normal(0, 0.025), p + rng.normal(0, 0.025), pt / nk * rng.uniform(0.5, 1.5)
            P.append([q * np.cos(pp), q * np.sin(pp), q * np.sinh(ee), np.sqrt((q * np.cosh(ee)) ** 2 + 0.14 ** 2)])
    P = np.array(P); lab = np.repeat([0, 1], [12, 9]); nn = np.array([len(P)])
    pv = lambda Q: H45.prong_vars(Q[None], nn, lab[None]); v0 = pv(P); pt, y, phi, m = kin(P); mj0 = jm(pt, y, phi, m)
    M0 = [pmass(pt, y, phi, m, lab == k) for k in (0, 1)]
    assert np.abs(build(pt, y, phi, m) - P).max() < 1e-9
    j, out, info = jet_edits((0, P, lab, 1))
    for k, Q in out.items():
        q = pv(Q); mq = [pmass(*kin(Q), lab == kk) for kk in (0, 1)]
        log(f'selftest {k:>10s}: z {v0["z"][0]:.4f}->{q["z"][0]:.4f}  dR {v0["dR"][0]:.4f}->{q["dR"][0]:.4f}  prong m {M0[0]:.3f}/{M0[1]:.3f}->{mq[0]:.3f}/{mq[1]:.3f}  '
            f'jet m {mj0:.3f}->{mass4(Q.sum(0)):.3f}  jet pT {np.hypot(*P.sum(0)[:2]):.2f}->{np.hypot(*Q.sum(0)[:2]):.2f}  constituent pT max change {np.abs(np.hypot(Q[:, 0], Q[:, 1]) - pt).max():.2e}')
        if k.startswith('B|'):
            lam = float(k.split('|')[1]); assert abs(q['dR'][0] / v0['dR'][0] - lam) < 1e-9 and abs(q['z'][0] - v0['z'][0]) < 1e-12
            assert max(abs(mq[0] - M0[0]), abs(mq[1] - M0[1])) < 1e-9
        if k.startswith('C|'): assert abs(q['dR'][0] - v0['dR'][0]) < 1e-9 and max(abs(mq[0] - M0[0]), abs(mq[1] - M0[1])) < 1e-6
        if k.startswith(('MBs', 'MBp', 'BC')):
            tgt = mass4(out[('B|' if k.startswith(('MBs|', 'MBp|')) else 'C|') + k.split('|')[1]].sum(0)); assert abs(mass4(Q.sum(0)) - tgt) < 1e-6
            assert abs(q['dR'][0] - v0['dR'][0]) < 1e-4 or k.startswith('BC')
    for dc in DCS_A:
        Q = out[f'D|{dc}']; L1 = H45.kt2_labels(Q[None].astype(np.float32), nn)[0]
        _, Qn, inf = restore_job((0, Q, L1, lab, pt, M0, mj0))
        mq = [pmass(*kin(Qn), L1 == kk) for kk in (0, 1)]; q = pv(Qn)
        log(f'selftest A|{dc}: restore {inf}; reclustered-label prong m {mq[0]:.4f}/{mq[1]:.4f} (targets {M0}); jet m {mass4(Qn.sum(0)):.4f}; z {q["z"][0]:.4f}; dR {q["dR"][0]:.4f}')
    log('selftest passed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', default=None); ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
    if a.selftest: selftest()
    else:
        log(f'H50 part {a.part}; smoke {SMOKE}; per class jc {PER_JC} hls {PER_HLS}; NPROC {NPROC}')
        if a.part in ('jc', 'hls'): selftest()
        dict(jc=part_jc, hls=part_hls, hjedi=part_hjedi, nets=part_nets, stats=part_stats, fig=figures)[a.part](); log('done')
