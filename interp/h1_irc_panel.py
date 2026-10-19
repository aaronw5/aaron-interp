"""H1 (loop D, new claim D5): IRC-safety panel. Do jet taggers change their calls under an exact collinear split of one particle
into two and under the emission of one soft particle, compared with count-fixed controls, an IRC-safe reference model (EFN) and
IRC-safe / IRC-unsafe classical observables?

Models
  hls4ml (5 classes g q W Z t; sample = interp.jedi_core.sample(per_class=2000, seed=401), 10,000 jets, 64 slots):
    --part jedi  (jax env)   JEDI-linear official n64 3-feature (fixed point, jedi_core); C4 = the same network with every weight
                             redrawn (jedi_core.Jedi(untrained=True, seed=0), "weights scrambled").
    --part small (conda env) Deep Sets dsets-d64, Linformer linf-2b-d32-k8, transformer tf-2b-d64-e100 (seed0 best.pt, models/small);
                             C3 = an Energy Flow Network trained here (latent L = sum_i z_i Phi(deta_i, dphi_i), z_i = pT_i / sum pT,
                             Phi one hidden layer 2-128-64, F 64-128-128-5; train archive minus splits dev, early stop on dev);
                             C4 = Deep Sets with the entries of every weight tensor permuted (seed 7).
  JetClass (10 classes; random rows of each class's first test file, seed 402, 1,000 per class = 10,000 jets, 128 slots; build
  with eta = jet_eta + sign(jet_eta) deta):
    --part jetclass (GPU)    ParT_full, ParT_kin, ParticleNet_full, ParticleNet_kin; C4 = ParT_full with weights redrawn
                             (ParTNetwork(untrained=True, seed=0)).
Edits (on the real particles; 3-momenta conserved exactly; at dR = 0 four-momenta conserved exactly):
  split  r{0,1,4,rnd} z{0.5,0.2,0.05} dR{0,1e-3,1e-2}: the particle of pT rank r (rnd: a random real particle) -> two daughters
         p1 = z p + k u, p2 = (1 - z) p - k u, u a unit vector perpendicular to p at a random azimuth in the (phi-hat, theta-hat)
         plane, k = dR z (1 - z) pT (daughter separation dR in (eta, phi)); daughter masses z m, (1 - z) m; both daughters keep the
         parent's PID / charge / impact parameters (JetClass). For dR > 0 the energy rises by ~ z(1-z) pT dR^2 / 2 (recorded as dm).
  multi  k{2,4,8}: the k hardest particles each split (z 0.5, dR 0).
  soft   eps{1e-4,1e-3,1e-2}: one particle of pT eps * pT_jet added at a uniform point of the disc R < 0.8 about the jet axis, its
         3-momentum taken from the hardest particle (recoil); JetClass: massless neutral hadron (isNeutralHadron 1, charge 0, d0 = dz = 0).
  *_fix  the same edit, then the softest real particle removed (multiplicity fixed): split_r0_z0.5_dR0_fix, soft_eps*_fix;
         drop_softest alone as its control.
  cn     (ParT/PN full only) split_r0_z0.5_dR0 of a charged-hadron hardest particle with daughter 2 relabelled a neutral hadron
         (charge 0, d0 = dz = 0); control: the same split with both daughters charged, on the same jets.
Controls
  C1 reshuffle_r{..}_f{0.5,0.2,0.05}: the particle of rank r gives the 3-momentum fraction f to its nearest real neighbour
     (multiplicity fixed; the IRC-safe observables change by O(f pT dR_nn)); matched to split(r, z = f). Rates are given on all jets
     and on the jets with dR_nn < 1e-2.
  recoil_eps*: the recoil of the soft edit without the added particle (matched control of soft).
  C2 merge: the closest real pair merged into one particle (four-vector sum; multiplicity -1).
  C3 EFN (IRC safe by construction; split at dR = 0 leaves it unchanged unless a particle leaves the 64 slots).
  C4 scrambled-weight networks (generic input sensitivity).
Padding / truncation: after an edit the particles are re-sorted by pT and cut to 64 (hls4ml) / 128 (JetClass) slots; the jets where
  a real particle leaves the window are recorded (overflow) and every rate is also given on the jets without overflow.
Metrics per model x condition: % calls changed with 95 % bootstrap interval over jets; accuracy; class flow matrix and McNemar for the
  dominant off-diagonal pair; net flow per class; mean delta log-softmax per class with CI; relative logit response; background
  rejection at the 50 % signal-efficiency working point (threshold frozen on unedited jets) for W/Z/t vs QCD; paired McNemar vs the
  matched count-fixed control (split vs reshuffle, soft vs recoil); Holm over all tests of the part (and over all parts in --summary).
Classical observables (1,000 jets, rows [::10]) on edited jets: N, pT^D, m, m_SD (C/A, z_cut 0.1, beta 0, all real particles), tau21,
  tau32 (k-means axes, jetdistill.observables.compute), D2 (beta 1, all real particles): mean change with CI, median |relative change|,
  fraction crossing the unedited sample median.
Run
  AI_PY=conda ./job.sh d-h1-small 1 2 'python -m interp.h1_irc_panel --part small'
  ./job.sh d-h1-jedi 1 2 'python -m interp.h1_irc_panel --part jedi'
  ./part_job.sh d-h1-jc 'python -m interp.h1_irc_panel --part jetclass'
  ./job.sh d-h1-sum 'python -m interp.h1_irc_panel --part summary'
  -> $AI_RESULTS/physics/h1_{jedi,small,jetclass,summary}.json, h1_*_calls.npz, h1_panel.png; EFN weights in physics/h1_efn/
  --smoke N: N jets per class, writes h1_<part>_smoke.json
"""
import argparse
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
OUT = RES / 'physics'
SMALL = Path(os.environ.get('AI_SMALL_MODELS', '/j-jepa-vol/aaron-interp/models/small'))
T0 = time.time()
NBOOT = 1000


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


# ================================================================================================================ particle table
# T = dict(pt, eta, phi, m: (J, W) float64; ext: {key: (J, W)}; ax_eta, ax_phi, sgn: (J,) the jet axis (hls4ml: 0, 0, 1))
def p3(pt, eta, phi):
    return pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)


def from3(px, py, pz):
    pt = np.hypot(px, py); eta = np.arcsinh(pz / np.maximum(pt, 1e-300)); phi = np.arctan2(py, px)
    return pt, eta, phi


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def copyT(T, extra_cols=0):
    J = len(T['pt'])
    pad = lambda a: np.concatenate([a, np.zeros((J, extra_cols), a.dtype)], 1) if extra_cols else a.copy()
    U = {k: pad(T[k]) for k in ('pt', 'eta', 'phi', 'm')}
    U['ext'] = {k: pad(v) for k, v in T['ext'].items()}
    for k in ('ax_eta', 'ax_phi', 'sgn'): U[k] = T[k]
    return U


def setp(T, rows, col, pt, eta, phi, m):
    T['pt'][rows, col] = pt; T['eta'][rows, col] = eta; T['phi'][rows, col] = phi; T['m'][rows, col] = m


def rank_slot(T, r, rng):
    """slot index of pT rank r (T sorted) or of a random real particle (r = 'rnd'); -1 where the jet has no such particle"""
    n = (T['pt'] > 0).sum(1)
    if r == 'rnd':
        return np.where(n > 0, np.floor(rng.random(len(n)) * n).astype(int), -1)
    return np.where(n > r, r, -1)


def split(T, slot, z, dR, rng, neutral2=False):
    """slot (J,) -> daughters in slot and in one appended column (see the module docstring)"""
    J = len(slot); rows = np.arange(J); ok = slot >= 0; s = np.maximum(slot, 0)
    U = copyT(T, 1); c = U['pt'].shape[1] - 1
    pt0, eta0, phi0, m0 = (T[k][rows, s] for k in ('pt', 'eta', 'phi', 'm'))
    px, py, pz = p3(pt0, eta0, phi0); p = np.sqrt(px ** 2 + py ** 2 + pz ** 2); p = np.maximum(p, 1e-300)
    cth, sth = pz / p, pt0 / p
    a = rng.random(J) * 2 * np.pi
    u = np.stack([np.cos(a) * -np.sin(phi0) + np.sin(a) * cth * np.cos(phi0),
                  np.cos(a) * np.cos(phi0) + np.sin(a) * cth * np.sin(phi0),
                  np.sin(a) * -sth], -1)
    k = dR * z * (1 - z) * pt0
    P = np.stack([px, py, pz], -1)
    d1 = z * P + k[:, None] * u; d2 = (1 - z) * P - k[:, None] * u
    r = rows[ok]
    q1 = from3(*d1[ok].T); q2 = from3(*d2[ok].T)
    setp(U, r, s[ok], *q1, z * m0[ok]); setp(U, r, c, *q2, (1 - z) * m0[ok])
    for key in U['ext']:
        U['ext'][key][r, c] = T['ext'][key][r, s[ok]]
    if neutral2 and U['ext']:
        for key in U['ext']:
            if key in ('charge', 'isChargedHadron', 'isPhoton', 'isElectron', 'isMuon', 'd0val', 'dzval', 'd0err', 'dzerr'):
                U['ext'][key][r, c] = 0
        U['ext']['isNeutralHadron'][r, c] = 1
    return U, ok


def soft(T, eps, R, rng, add=True):
    """one particle pT = eps * pT_jet uniform in the disc R about the axis, 3-momentum taken from slot 0 (add=False: recoil only)"""
    J = len(T['pt']); rows = np.arange(J)
    px, py, pz = p3(T['pt'], T['eta'], T['phi']); real = T['pt'] > 0
    ptj = np.hypot((px * real).sum(1), (py * real).sum(1))
    rr = R * np.sqrt(rng.random(J)); th = rng.random(J) * 2 * np.pi
    de, dp = rr * np.cos(th), rr * np.sin(th)
    eta_s = T['ax_eta'] + T['sgn'] * de; phi_s = T['ax_phi'] + dp
    qx, qy, qz = p3(eps * ptj, eta_s, phi_s)
    U = copyT(T, 1 if add else 0)
    h = from3(px[:, 0] - qx, py[:, 0] - qy, pz[:, 0] - qz)
    setp(U, rows, 0, *h, T['m'][:, 0])
    if add:
        c = U['pt'].shape[1] - 1
        setp(U, rows, c, eps * ptj, eta_s, phi_s, np.zeros(J))
        if U['ext']:
            for key in U['ext']: U['ext'][key][:, c] = 0
            U['ext']['isNeutralHadron'][:, c] = 1; U['ext']['mask'][:, c] = 1
    return U


def drop_softest(T):
    U = copyT(T); real = U['pt'] > 0; J = len(real); rows = np.arange(J)
    j = np.where(real, U['pt'], np.inf).argmin(1); ok = real.sum(1) > 1
    U['pt'][rows[ok], j[ok]] = 0.0
    return U


def dR_rows(T, i, j, rows):
    de = T['eta'][rows, i] - T['eta'][rows, j]; dp = wrap(T['phi'][rows, i] - T['phi'][rows, j])
    return np.hypot(de, dp)


def nearest(T, slot):
    """nearest real neighbour of slot (by dR in (eta, phi)); -1 if none"""
    J = len(slot); rows = np.arange(J); s = np.maximum(slot, 0)
    de = T['eta'] - T['eta'][rows, s][:, None]; dp = wrap(T['phi'] - T['phi'][rows, s][:, None])
    d = np.hypot(de, dp); d[T['pt'] <= 0] = np.inf; d[rows, s] = np.inf
    nb = d.argmin(1); dmin = d[rows, nb]
    ok = (slot >= 0) & np.isfinite(dmin)
    return np.where(ok, nb, -1), np.where(ok, dmin, np.nan)


def reshuffle(T, slot, f):
    """C1: slot gives the 3-momentum fraction f to its nearest real neighbour"""
    nb, dnn = nearest(T, slot); ok = nb >= 0; r = np.arange(len(slot))[ok]; s, b = slot[ok], nb[ok]
    U = copyT(T)
    P = lambda i: np.stack(p3(T['pt'][r, i], T['eta'][r, i], T['phi'][r, i]), -1)
    Ps, Pb = P(s), P(b); q = f * Ps
    setp(U, r, s, *from3(*(Ps - q).T), (1 - f) * T['m'][r, s])
    setp(U, r, b, *from3(*(Pb + q).T), T['m'][r, b])
    return U, ok, dnn


def merge_closest(T, chunk=500):
    """C2: the closest real pair merged (four-vector sum) into the harder slot"""
    J, W = T['pt'].shape; I = np.full(J, -1); K = np.full(J, -1); dmin = np.full(J, np.nan)
    for a in range(0, J, chunk):
        sl = slice(a, a + chunk); real = T['pt'][sl] > 0
        de = T['eta'][sl][:, :, None] - T['eta'][sl][:, None]; dp = wrap(T['phi'][sl][:, :, None] - T['phi'][sl][:, None])
        d = np.hypot(de, dp); bad = ~(real[:, :, None] & real[:, None]) | np.eye(W, dtype=bool)[None]
        d[bad] = np.inf; f = d.reshape(len(d), -1).argmin(1); i, k = np.divmod(f, W)
        dm = d.reshape(len(d), -1)[np.arange(len(d)), f]; ok = np.isfinite(dm)
        I[sl] = np.where(ok, np.minimum(i, k), -1); K[sl] = np.where(ok, np.maximum(i, k), -1); dmin[sl] = np.where(ok, dm, np.nan)
    ok = I >= 0; r = np.arange(J)[ok]; i, k = I[ok], K[ok]
    U = copyT(T)
    def P4(j):
        px, py, pz = p3(T['pt'][r, j], T['eta'][r, j], T['phi'][r, j])
        E = np.sqrt((T['pt'][r, j] * np.cosh(T['eta'][r, j])) ** 2 + T['m'][r, j] ** 2)
        return px, py, pz, E
    a4, b4 = P4(i), P4(k); s4 = [x + y for x, y in zip(a4, b4)]
    mm = np.sqrt(np.maximum(s4[3] ** 2 - s4[0] ** 2 - s4[1] ** 2 - s4[2] ** 2, 0))
    setp(U, r, i, *from3(*s4[:3]), mm); U['pt'][r, k] = 0.0
    return U, ok, dmin


def finalize(T, cap):
    """sort by pT, cut to cap slots, zero the padding; returns (table, overflow (J,) bool)"""
    o = np.argsort(-T['pt'], 1, kind='stable'); tk = lambda a: np.take_along_axis(a, o, 1)[:, :cap]
    nreal = (T['pt'] > 0).sum(1); over = nreal > cap
    U = {k: tk(T[k]) for k in ('pt', 'eta', 'phi', 'm')}; U['ext'] = {k: tk(v) for k, v in T['ext'].items()}
    real = U['pt'] > 0
    for k in ('eta', 'phi', 'm'): U[k] = np.where(real, U[k], 0.0)
    U['pt'] = np.where(real, U['pt'], 0.0)
    for k in U['ext']: U['ext'][k] = np.where(real, U['ext'][k], 0.0)
    if 'mask' in U['ext']: U['ext']['mask'] = real.astype(np.float64)
    for k in ('ax_eta', 'ax_phi', 'sgn'): U[k] = T[k]
    return U, over


def jet4(T):
    real = T['pt'] > 0
    px, py, pz = p3(T['pt'], T['eta'], T['phi']); E = np.sqrt((T['pt'] * np.cosh(T['eta'])) ** 2 + T['m'] ** 2)
    return np.stack([(a * real).sum(1) for a in (px, py, pz, E)], 1)


# ================================================================================================================ conditions
Z3, DR3, EPS3, RANKS = (0.5, 0.2, 0.05), (0.0, 1e-3, 1e-2), (1e-4, 1e-3, 1e-2), (0, 1, 4, 'rnd')


def condition_list(jetclass_full=False):
    C = []
    for r in RANKS:
        for z in Z3:
            for d in DR3: C.append(dict(name=f'split_r{r}_z{z}_dR{d:g}', kind='split', r=r, z=z, dR=d, control=f'reshuffle_r{r}_f{z}'))
    for k in (2, 4, 8): C.append(dict(name=f'multi_k{k}', kind='multi', k=k))
    for e in EPS3: C.append(dict(name=f'soft_eps{e:g}', kind='soft', eps=e, control=f'recoil_eps{e:g}'))
    for e in EPS3: C.append(dict(name=f'recoil_eps{e:g}', kind='recoil', eps=e))
    C.append(dict(name='split_r0_z0.5_dR0_fix', kind='split', r=0, z=0.5, dR=0.0, fix=True, control='drop_softest'))
    for e in EPS3: C.append(dict(name=f'soft_eps{e:g}_fix', kind='soft', eps=e, fix=True, control='drop_softest'))
    C.append(dict(name='drop_softest', kind='drop'))
    for r in RANKS:
        for f in Z3: C.append(dict(name=f'reshuffle_r{r}_f{f}', kind='reshuffle', r=r, f=f))
    C.append(dict(name='merge_closest', kind='merge'))
    if jetclass_full:
        C.append(dict(name='cn_split_charged_neutral', kind='cn', neutral=True, control='cn_split_charged_charged'))
        C.append(dict(name='cn_split_charged_charged', kind='cn', neutral=False))
    return C


OBS_CONDS = ('split_r0_z0.5_dR0', 'split_r0_z0.05_dR0', 'split_r0_z0.5_dR0.01', 'split_rrnd_z0.5_dR0', 'multi_k8', 'soft_eps0.0001',
             'soft_eps0.01', 'reshuffle_r0_f0.5', 'merge_closest', 'drop_softest')


def apply(T0, c, cap, R, seed):
    """-> (edited finalized table, applied mask, info dict)"""
    rng = np.random.default_rng(seed + zlib.crc32(c['name'].encode()) % 100000)
    J = len(T0['pt']); info = {}; applied = np.ones(J, bool)
    if c['kind'] == 'split':
        slot = rank_slot(T0, c['r'], rng); U, applied = split(T0, slot, c['z'], c['dR'], rng)
    elif c['kind'] == 'multi':
        U = T0
        for r in range(c['k']):
            U, ok = split(U, rank_slot(T0, r, rng), 0.5, 0.0, rng); applied &= ok
    elif c['kind'] in ('soft', 'recoil'):
        U = soft(T0, c['eps'], R, rng, add=c['kind'] == 'soft')
    elif c['kind'] == 'drop':
        U = drop_softest(T0)
    elif c['kind'] == 'reshuffle':
        slot = rank_slot(T0, c['r'], rng); U, applied, dnn = reshuffle(T0, slot, c['f'])
        info['dR_nn'] = dnn
    elif c['kind'] == 'merge':
        U, applied, dmin = merge_closest(T0); info['dR_pair'] = dmin
    elif c['kind'] == 'cn':
        ch = T0['ext']['isChargedHadron'][:, 0] > 0.5
        slot = np.where(ch & (T0['pt'][:, 0] > 0), 0, -1)
        U, applied = split(T0, slot, 0.5, 0.0, rng, neutral2=c['neutral'])
    else:
        raise ValueError(c)
    if c.get('fix'):
        U = drop_softest(U)
    U, over = finalize(U, cap)
    p4a, p4b = jet4(T0), jet4(U)
    info['overflow'] = over
    info['dp3_rel'] = np.abs(p4b[:, :3] - p4a[:, :3]).max(1) / np.maximum(np.hypot(p4a[:, 0], p4a[:, 1]), 1e-9)
    ma = np.sqrt(np.maximum(p4a[:, 3] ** 2 - (p4a[:, :3] ** 2).sum(1), 0)); mb = np.sqrt(np.maximum(p4b[:, 3] ** 2 - (p4b[:, :3] ** 2).sum(1), 0))
    info['dm'] = mb - ma
    info['dN'] = (U['pt'] > 0).sum(1) - (T0['pt'] > 0).sum(1)
    return U, applied, info


# ================================================================================================================ statistics
def boot_binary(x, rng, B=NBOOT):
    x = np.asarray(x, bool); n = len(x)
    if n == 0: return [float('nan')] * 2
    b = rng.binomial(n, x.mean(), B) / n
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def boot_mean_cols(X, rng, B=500):
    n = len(X)
    if n == 0: return np.full((X.shape[1], 2), np.nan)
    idx = rng.integers(0, n, (B, n)); M = np.stack([X[i].mean(0) for i in idx])
    return np.percentile(M, [2.5, 97.5], 0).T


def binom_two_sided(k, n):
    """exact two-sided binomial test p for k of n at 1/2 (McNemar exact)"""
    if n == 0: return 1.0
    k = min(k, n - k)
    lp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    m = max(lp); s = sum(math.exp(v - m) for v in lp)
    return float(min(1.0, 2 * math.exp(m) * s))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); out = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out.tolist()


def logsoftmax(L):
    L = L - L.max(1, keepdims=True); return L - np.log(np.exp(L).sum(1, keepdims=True))


def rel_response(L0, L1):
    a = L0 - L0.mean(1, keepdims=True); b = L1 - L1.mean(1, keepdims=True)
    return float(np.linalg.norm(b - a, axis=1).mean() / np.sqrt(((a - a.mean(0)) ** 2).sum(1).mean()))


def rejection(L0, L1, y, sig, bkg):
    P0, P1 = np.exp(logsoftmax(L0)), np.exp(logsoftmax(L1)); bk = np.isin(y, bkg); sg = y == sig
    if sg.sum() < 10 or bk.sum() < 10: return None
    s0 = P0[:, sig] / (P0[:, sig] + P0[:, bkg].sum(1)); s1 = P1[:, sig] / (P1[:, sig] + P1[:, bkg].sum(1))
    t = np.median(s0[sg]); nb = int(bk.sum())
    f0, f1 = int((s0[bk] > t).sum()), int((s1[bk] > t).sum())
    return dict(threshold=float(t), sig_eff_after=float((s1[sg] > t).mean()), n_bkg=nb, n_bkg_pass_before=f0, n_bkg_pass_after=f1,
                rej_before=float(nb / f0) if f0 else None, rej_after=float(nb / f1) if f1 else None)


def cond_stats(L0, L1, y, applied, over, rng, classes, sig_bkg):
    c0, c1 = L0.argmax(1), L1.argmax(1); C = len(classes)
    def rate(sel):
        ch = (c0 != c1)[sel]
        return dict(n=int(sel.sum()), changed=float(ch.mean()) if sel.any() else None, ci=boot_binary(ch, rng),
                    acc_before=float((c0 == y)[sel].mean()) if sel.any() else None, acc_after=float((c1 == y)[sel].mean()) if sel.any() else None)
    sel = applied
    out = dict(all=rate(sel), no_overflow=rate(sel & ~over), overflow_frac=float(over[sel].mean()) if sel.any() else None)
    F = np.zeros((C, C), int); np.add.at(F, (c0[sel], c1[sel]), 1); out['flow'] = F.tolist()
    off = F + F.T; np.fill_diagonal(off, 0); a, b = np.unravel_index(off.argmax(), off.shape)
    if F[b, a] > F[a, b]: a, b = b, a
    out['dominant'] = None if off.max() == 0 else dict(frm=classes[a], to=classes[b], n_ab=int(F[a, b]), n_ba=int(F[b, a]),
                                                       p_mcnemar=binom_two_sided(int(F[a, b]), int(F[a, b] + F[b, a])))
    out['net_flow'] = {classes[k]: int(F[:, k].sum() - F[k, :].sum()) for k in range(C)}
    D = (logsoftmax(L1) - logsoftmax(L0))[sel]; ci = boot_mean_cols(D, rng)
    out['dlogp'] = {classes[k]: dict(mean=float(D[:, k].mean()), ci=ci[k].tolist()) for k in range(C)}
    out['rel_response'] = rel_response(L0[sel], L1[sel])
    out['rejection'] = {f'{classes[s]}_vs_{"+".join(classes[b] for b in bk)}': rejection(L0[sel], L1[sel], y[sel], s, bk) for s, bk in sig_bkg}
    return out


def paired(ch_a, ch_b, sel, rng):
    a, b = ch_a[sel], ch_b[sel]; n10, n01 = int((a & ~b).sum()), int((~a & b).sum()); n = int(sel.sum())
    if n == 0: return dict(n=0)
    idx = rng.integers(0, n, (NBOOT, n)); d = (a[idx].mean(1) - b[idx].mean(1))
    return dict(n=n, edit=float(a.mean()), control=float(b.mean()), diff=float(a.mean() - b.mean()),
                diff_ci=np.percentile(d, [2.5, 97.5]).tolist(), n_edit_only=n10, n_control_only=n01,
                p_mcnemar=binom_two_sided(n10, n10 + n01))


# ================================================================================================================ observables
def observables(T):
    import importlib
    OC = importlib.import_module('jetdistill.observables.compute')
    pt, eta, phi, m = T['pt'], T['eta'], T['phi'], T['m']; real = pt > 0
    S = pt.sum(1); z = pt / np.maximum(S, 1e-300)[:, None]
    p4 = jet4(T); mass = np.sqrt(np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1), 0))
    O = dict(N=real.sum(1).astype(float), ptD=np.sqrt((pt ** 2).sum(1)) / np.maximum(S, 1e-300), m=mass)
    from interp.e19_common import kt3
    tau = {k: np.zeros(len(pt)) for k in (1, 2, 3)}
    for j in range(len(pt)):                      # exclusive-kT axes (IRC safe), tau_N = sum_i z_i min_k dR_ik / R0
        r = real[j]; e, f, w = eta[j, r], phi[j, r], z[j, r]
        for k in (1, 2, 3):
            ax = kt3(pt[j, r], e, f, k)[1]; ax = ax[ax[:, 0] > 0]
            d = np.hypot(e[:, None] - ax[None, :, 1], wrap(f[:, None] - ax[None, :, 2])).min(1)
            tau[k][j] = (w * d).sum() / OC.R0
    O['tau21'] = tau[2] / np.maximum(tau[1], 1e-12); O['tau32'] = tau[3] / np.maximum(tau[2], 1e-12)
    O['D2'] = np.zeros(len(pt)); order = np.argsort(real.sum(1), kind='stable'); a = 0
    while a < len(pt):                            # D2 (beta 1) on all real particles, chunks sized by multiplicity
        nr = real.sum(1)[order]; nb = 1                # sorted ascending: the last jet of a chunk has its largest multiplicity
        while a + nb < len(pt) and (nb + 1) * math.comb(max(int(nr[a + nb]), 3), 3) <= 2e6: nb += 1
        idx = order[a:a + nb]; w = max(int(real[idx].sum(1).max()), 3)
        e2, _, e3, *_ = OC._ecf(z[idx, :w], eta[idx, :w], phi[idx, :w]); O['D2'][idx] = e3 / np.maximum(e2 ** 3, 1e-12); a += len(idx)
    msd = np.zeros(len(pt)); order = np.argsort(real.sum(1), kind='stable')
    for a in range(0, len(pt), 100):
        idx = order[a:a + 100]; w = max(int(real[idx].sum(1).max()), 2)
        msd[idx] = OC._softdrop(pt[idx, :w], eta[idx, :w], phi[idx, :w])[0]
    O['mSD'] = msd
    return O


def obs_stats(O0, O1, rng):
    out = {}
    for k in O0:
        a, b = O0[k], O1[k]; d = b - a; med = np.median(a)
        cross = (a > med) != (b > med)
        rel = np.abs(d) / np.maximum(np.abs(a), 1e-12)
        idx = rng.integers(0, len(d), (NBOOT, len(d))); bm = d[idx].mean(1)
        out[k] = dict(mean_delta=float(d.mean()), ci=np.percentile(bm, [2.5, 97.5]).tolist(), median_abs_rel=float(np.median(rel)),
                      frac_changed=float((np.abs(d) > 1e-9 * np.maximum(np.abs(a), 1)).mean()),
                      cross_median=float(cross.mean()), cross_ci=boot_binary(cross, rng))
    return out


# ================================================================================================================ driver
def subT(T, rows):
    U = {k: T[k][rows] for k in ('pt', 'eta', 'phi', 'm')}; U['ext'] = {}
    return U


def run_panel(name, T0, y, models, prep, classes, sig_bkg, cap, R, seed, conds, obs_rows, smoke):
    """models: {name: f(prep(T)) -> logits}; returns the result dict and the calls npz dict"""
    rng = np.random.default_rng(seed)
    res = dict(part=name, N=int(len(y)), seed=seed, cap=cap, R=R, classes=classes, labels=np.bincount(y, minlength=len(classes)).tolist(),
               n_real=dict(mean=float((T0['pt'] > 0).sum(1).mean()), frac_at_cap=float(((T0['pt'] > 0).sum(1) >= cap).mean())),
               models={}, conditions={}, observables={})
    L0 = {}; NPZ = dict(y=y); X0 = prep(T0)
    for mn, f in models.items():
        t = time.time(); L0[mn] = f(X0); c0 = L0[mn].argmax(1)
        res['models'][mn] = dict(acc=float((c0 == y).mean()), acc_ci=boot_binary(c0 == y, rng),
                                 per_class={classes[k]: float((c0[y == k] == k).mean()) for k in range(len(classes))}, results={})
        NPZ[f'{mn}__base'] = c0.astype(np.int8)
        log(f'{name} {mn}: unedited accuracy {res["models"][mn]["acc"]:.4f} ({time.time() - t:.0f} s)')
    O0 = observables(subT(T0, obs_rows)) if obs_rows is not None else None
    CH = {}
    for c in conds:
        t = time.time(); U, applied, info = apply(T0, c, cap, R, seed)
        ci = dict(n_applied=int(applied.sum()), overflow=float(info['overflow'][applied].mean()) if applied.any() else None,
                  dN_mean=float(info['dN'][applied].mean()) if applied.any() else None,
                  dp3_rel_max=float(np.nanmax(info['dp3_rel'][applied])) if applied.any() else None,
                  dm_median=float(np.median(info['dm'][applied])) if applied.any() else None,
                  dm_absmax=float(np.abs(info['dm'][applied]).max()) if applied.any() else None)
        for k in ('dR_nn', 'dR_pair'):
            if k in info:
                v = info[k][applied]; ci[k] = dict(median=float(np.nanmedian(v)), frac_below_1em2=float(np.nanmean(v < 1e-2)))
        res['conditions'][c['name']] = ci
        sub_nn = (info['dR_nn'] < 1e-2) & applied if 'dR_nn' in info else None
        XU = prep(U)
        for mn, f in models.items():
            L1 = f(XU); st = cond_stats(L0[mn], L1, y, applied, info['overflow'], rng, classes, sig_bkg)
            if sub_nn is not None:
                st['dRnn_below_1e-2'] = dict(n=int(sub_nn.sum()), changed=float((L0[mn].argmax(1) != L1.argmax(1))[sub_nn].mean()) if sub_nn.any() else None,
                                             ci=boot_binary((L0[mn].argmax(1) != L1.argmax(1))[sub_nn], rng))
            res['models'][mn]['results'][c['name']] = st
            ch = L0[mn].argmax(1) != L1.argmax(1); CH[(mn, c['name'])] = (ch, applied, sub_nn)
            NPZ[f'{mn}__{c["name"]}'] = L1.argmax(1).astype(np.int8)
        log(f'{name} {c["name"]}: applied {ci["n_applied"]} overflow {ci["overflow"]} dN {ci["dN_mean"]} dp3 {ci["dp3_rel_max"]:.1e} '
            f'dm med {ci["dm_median"]:.2e} | ' + ' '.join(f'{mn} {res["models"][mn]["results"][c["name"]]["all"]["changed"]:.4f}' for mn in models)
            + f' ({time.time() - t:.0f} s)')
        if O0 is not None and c['name'] in OBS_CONDS:
            res['observables'][c['name']] = obs_stats(O0, observables(subT(U, obs_rows)), rng)
            log(f'  observables {c["name"]}: ' + ' '.join(f'{k} {v["mean_delta"]:+.3g} x{v["cross_median"]:.3f}' for k, v in res['observables'][c['name']].items()))
    # paired tests: edit vs matched count-fixed control
    tests = []
    for mn in models:
        for c in conds:
            if 'control' not in c or (mn, c['control']) not in CH: continue
            a, ap, _ = CH[(mn, c['name'])]; b, bp, bsub = CH[(mn, c['control'])]
            sel = ap & bp; T = dict(control=c['control'], all=paired(a, b, sel, rng))
            if bsub is not None: T['dRnn_below_1e-2'] = paired(a, b, sel & bsub, rng)
            res['models'][mn]['results'][c['name']]['vs_control'] = T
            tests.append((mn, c['name'], T['all'].get('p_mcnemar', 1.0)))
    hp = holm([t[2] for t in tests])
    for (mn, cn, _), h in zip(tests, hp): res['models'][mn]['results'][cn]['vs_control']['holm_p_part'] = h
    res['n_tests_part'] = len(tests)
    return res, NPZ


def save(res, NPZ, part, smoke):
    OUT.mkdir(parents=True, exist_ok=True); sfx = '_smoke' if smoke else ''
    p = OUT / f'h1_{part}{sfx}.json'; p.write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed(OUT / f'h1_{part}{sfx}_calls.npz', **NPZ); log(f'wrote {p}')


# ------------------------------------------------------------------------------------------------ hls4ml
HLS_CL = ['g', 'q', 'W', 'Z', 't']; HLS_SB = [(2, [0, 1]), (3, [0, 1]), (4, [0, 1])]


def hls_table(x):
    x = np.asarray(x, np.float64); J = len(x); real = x[..., 0] > 0
    return dict(pt=np.where(real, x[..., 0], 0), eta=np.where(real, x[..., 1], 0), phi=np.where(real, x[..., 2], 0), m=np.zeros(x.shape[:2]),
                ext={}, ax_eta=np.zeros(J), ax_phi=np.zeros(J), sgn=np.ones(J))


def hls_x(T):
    return np.stack([T['pt'], T['eta'], T['phi']], -1).astype(np.float32)


def hls_sample(per, seed=401):
    """the jets of interp.jedi_core.sample(per_class=per, seed=seed) (same indices), without loading the Q_ observables"""
    from jetdistill.config import RESULTS
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')
    y = np.asarray(d['y']); rng = np.random.default_rng(seed)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per, replace=False) for c in range(5)]))
    x = np.asarray(d['x'], np.float32)[idx]
    log(f'hls4ml sample: {len(idx)} jets (seed {seed})')
    return hls_table(x), y[idx].astype(int)


def part_jedi(smoke):
    from interp.jedi_core import Jedi
    T0, y = hls_sample(smoke or 2000)
    nets = dict(jedi=Jedi(64, '3-feature'), jedi_scrambled=Jedi(64, '3-feature', untrained=True, seed=0))
    models = {k: (lambda x, n=n: n.run(x)['logits']) for k, n in nets.items()}
    rows = np.arange(len(y))[::10]
    res, NPZ = run_panel('jedi', T0, y, models, hls_x, HLS_CL, HLS_SB, 64, 0.8, 401, condition_list(), rows, smoke)
    res['sample'] = 'interp.jedi_core.sample(per_class=2000, seed=401)'
    save(res, NPZ, 'jedi', smoke)


# --- small torch models and the EFN
def load_small(run_dir):
    import importlib.util, torch
    spec = importlib.util.spec_from_file_location('small_models', ROOT / 'train' / 'models.py')
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    arch = json.loads((run_dir / 'arch.json').read_text())
    net = M.build_model(arch['spec'], arch['n_particles'], cls_token=False, final_ln=arch['final_ln'])
    net.load_state_dict(torch.load(run_dir / 'best.pt', map_location='cpu')); net.eval()
    return net, np.asarray(arch['inputs']['shift'], np.float32), np.asarray(arch['inputs']['scale'], np.float32)


def torch_logits(net, x, shift, scale, batch=2000):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = x[i:i + batch]; m = xb[..., 0] > 0; xn = (xb - shift) / scale; xn[~m] = 0
            out.append(net(torch.from_numpy(xn), torch.from_numpy(m)).double().numpy())
    return np.concatenate(out)


def efn_class():
    import torch.nn as nn

    class EFN(nn.Module):
        """latent L = sum_i z_i Phi(deta_i / 0.1, dphi_i / 0.1), z_i = pT_i / sum_j pT_j; F 64-128-128-5"""
        def __init__(self, hid=128, lat=64):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(2, hid), nn.ReLU(), nn.Linear(hid, lat), nn.ReLU())
            self.F = nn.Sequential(nn.Linear(lat, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 5))

        def forward(self, x, mask=None):            # x raw (B, n, 3): pT, deta, dphi; padding pT = 0
            pt = x[..., 0].clamp(min=0); z = pt / pt.sum(1, keepdim=True).clamp(min=1e-12)
            return self.F((z.unsqueeze(-1) * self.phi(x[..., 1:3] / 0.1)).sum(1))
    return EFN


def efn_train(d, max_epochs=12, patience=3):
    import torch, torch.nn.functional as Fn
    torch.manual_seed(0); torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT / 'train'))
    from train_small import load_archive, dev_indices
    DATA = os.environ['JETDISTILL_DATA']
    n_tot = len(np.load(os.path.join(DATA, 'train_ptetaphi_f16.npy'), mmap_mode='r'))
    dev, _ = dev_indices(DATA, n_tot); tr = np.setdiff1d(np.arange(n_tot), dev)
    xtr, ytr = load_archive(DATA, 'train', 64, tr); xdv, ydv = load_archive(DATA, 'train', 64, dev)
    log(f'EFN train: {len(ytr)} train, {len(ydv)} dev jets')
    net = efn_class()(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    steps = max_epochs * (len(ytr) // 1024); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps + 10)
    best, bad, hist = 1e9, 0, []
    xdv_t = torch.from_numpy(xdv.astype(np.float32)); ydv_t = torch.from_numpy(ydv)
    for ep in range(max_epochs):
        net.train(); perm = np.random.default_rng(ep).permutation(len(ytr)); tl = 0.0; t = time.time()
        for i in range(0, len(perm) - 1023, 1024):
            b = np.sort(perm[i:i + 1024]); xb = torch.from_numpy(xtr[b].astype(np.float32)); yb = torch.from_numpy(ytr[b])
            loss = Fn.cross_entropy(net(xb), yb); opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tl += loss.item()
        net.eval()
        with torch.no_grad():
            lo = torch.cat([net(xdv_t[i:i + 5000]) for i in range(0, len(ydv), 5000)])
            vl = Fn.cross_entropy(lo, ydv_t).item(); va = (lo.argmax(1) == ydv_t).float().mean().item()
        hist.append(dict(epoch=ep + 1, train_loss=tl / (len(perm) // 1024), val_loss=vl, val_acc=va)); log(f'EFN epoch {ep + 1}: {hist[-1]} ({time.time() - t:.0f} s)')
        if vl < best: best, bad = vl, 0; torch.save(net.state_dict(), d / 'best.pt')
        else:
            bad += 1
            if bad >= patience: break
    xte, yte = load_archive(DATA, 'val', 64); net.load_state_dict(torch.load(d / 'best.pt')); net.eval()
    with torch.no_grad():
        c = torch.cat([net(torch.from_numpy(xte[i:i + 5000].astype(np.float32))).argmax(1) for i in range(0, len(yte), 5000)]).numpy()
    info = dict(arch='EFN: Phi 2-128-64 ReLU on (deta, dphi)/0.1, latent sum_i z_i Phi, F 64-128-128-5', history=hist,
                test_acc=float((c == yte).mean()), n_test=int(len(yte)), best_val_loss=best)
    (d / 'info.json').write_text(json.dumps(info, indent=1)); log(f'EFN test accuracy (val archive, 260k) {info["test_acc"]:.4f}')
    return info


def part_small(smoke):
    import torch
    torch.set_num_threads(1)
    d = OUT / ('h1_efn_smoke' if smoke else 'h1_efn'); d.mkdir(parents=True, exist_ok=True)
    efn_info = json.loads((d / 'info.json').read_text()) if (d / 'info.json').exists() else efn_train(d, max_epochs=1 if smoke else 12)
    T0, y = hls_sample(smoke or 2000)
    models = {}
    for tag, run in (('dsets', 'dsets-d64-n64'), ('linf', 'linf-2b-d32-k8-n64'), ('tf_d64', 'tf-2b-d64-n64-e100')):
        net, sh, sc = load_small(SMALL / run / 'seed0'); models[tag] = (lambda x, net=net, sh=sh, sc=sc: torch_logits(net, x, sh, sc))
    net, sh, sc = load_small(SMALL / 'dsets-d64-n64' / 'seed0'); g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in net.parameters():
            f = p.data.flatten(); p.data.copy_(f[torch.randperm(f.numel(), generator=g)].view_as(p))
    models['dsets_scrambled'] = (lambda x, net=net, sh=sh, sc=sc: torch_logits(net, x, sh, sc))
    efn = efn_class()(); efn.load_state_dict(torch.load(d / 'best.pt')); efn.eval()
    models['efn'] = (lambda x: torch_logits(efn, x, np.zeros(3, np.float32), np.ones(3, np.float32)))
    res, NPZ = run_panel('small', T0, y, models, hls_x, HLS_CL, HLS_SB, 64, 0.8, 401, condition_list(), None, smoke)
    res['sample'] = 'interp.jedi_core.sample(per_class=2000, seed=401)'; res['efn'] = efn_info
    res['known_acc'] = {k: json.loads((SMALL / r / 'seed0' / 'test.json').read_text())['accuracy'] for k, r in
                        (('dsets', 'dsets-d64-n64'), ('linf', 'linf-2b-d32-k8-n64'), ('tf_d64', 'tf-2b-d64-n64-e100'))}
    save(res, NPZ, 'small', smoke)


# ------------------------------------------------------------------------------------------------ JetClass
EXT_KEYS = ['d0val', 'd0err', 'dzval', 'dzerr', 'charge', 'isChargedHadron', 'isNeutralHadron', 'isPhoton', 'isElectron', 'isMuon', 'mask']


def jc_sample(per, seed=402):
    import uproot
    from interp.part_core import read_root, fof, cat, sub, CL
    rng = np.random.default_rng(seed); Js = []
    for c in CL:
        f = fof(c); n = uproot.open(f)['tree'].num_entries; idx = np.sort(rng.choice(n, per, replace=False)); parts = []
        for a in range(0, n, 20000):
            loc = idx[(idx >= a) & (idx < a + 20000)] - a
            if len(loc): parts.append(sub(read_root(f, start=a, stop=min(a + 20000, n)), loc))
        Js.append(cat(parts)); log(f'JetClass sample {c}: {per} of {n} rows of {Path(f).name}')
    return cat(Js)


def jc_table(J0):
    from interp.part_core import kin
    pt, de, dp, ms = (np.asarray(a, np.float64) for a in kin(J0))
    sgn = np.where(J0['jet_eta'] < 0, -1.0, 1.0).astype(np.float64); je, jp = J0['jet_eta'].astype(np.float64), J0['jet_phi'].astype(np.float64)
    real = pt > 0
    return dict(pt=pt, eta=np.where(real, je[:, None] + sgn[:, None] * de, 0), phi=np.where(real, jp[:, None] + dp, 0), m=ms,
                ext={k: np.asarray(J0[k], np.float64) for k in EXT_KEYS}, ax_eta=je, ax_phi=jp, sgn=sgn)


def jc_jets(J0, T):
    from interp.part_core import build
    real = T['pt'] > 0
    de = np.where(real, T['sgn'][:, None] * (T['eta'] - T['ax_eta'][:, None]), 0); dp = np.where(real, wrap(T['phi'] - T['ax_phi'][:, None]), 0)
    return build(J0, T['pt'], de, dp, T['m'], extra=T['ext'])


def jc_rel(T):
    real = T['pt'] > 0
    return dict(pt=T['pt'], eta=np.where(real, T['sgn'][:, None] * (T['eta'] - T['ax_eta'][:, None]), 0),
                phi=np.where(real, wrap(T['phi'] - T['ax_phi'][:, None]), 0), m=T['m'], ext={})


def pn_logits(model, J, n, batch=256):
    import torch
    from interp.part_core import inputs
    x, v, m = inputs(J, n); N = len(x); order = np.argsort(m[:, 0].sum(1), kind='stable'); L = np.zeros((N, 10))
    dev = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, N, batch):
            idx = order[i:i + batch]; P = min(max(int(m[idx, 0].sum(1).max()), 17), x.shape[2])   # kNN needs >= k + 1 = 17 slots (weaver pads to 128)
            fx = torch.from_numpy(np.ascontiguousarray(x[idx, :, :P])).to(dev); mk = torch.from_numpy(np.ascontiguousarray(m[idx, :, :P])).to(dev)
            pts = fx[:, -2:].clone()
            L[idx] = model(pts, fx.clone(), mk).double().cpu().numpy()
    return L


def part_jetclass(smoke):
    from interp.part_core import ParTNetwork, CL
    from interp.p1v_verify_part import fwd
    from interp.part_core import inputs
    from interp.pn_net import load_particlenet
    from jetdistill.config import MODELS
    dev = os.environ.get('JETDISTILL_DEVICE', 'cpu')
    J0 = jc_sample(smoke or 1000); y = J0['label'].astype(int); T0 = jc_table(J0)
    Jb = jc_jets(J0, T0)
    from interp.part_core import build, kin
    Jref = build(J0, *kin(J0))
    rt = max(float(np.abs(Jb[k] - Jref[k]).max()) for k in ('px', 'py', 'pz', 'energy', 'deta', 'dphi'))
    log(f'round trip table -> build vs build(kin(J0)): max abs diff {rt:.2e}')
    models = {}
    for n in ('full', 'kin'):
        net = ParTNetwork(n); models[f'ParT_{n}'] = (lambda J, net=net, n=n: fwd(net, *inputs(J, n))[1])
    net = ParTNetwork('full', untrained=True, seed=0); models['ParT_full_scrambled'] = (lambda J, net=net: fwd(net, *inputs(J, 'full'))[1])
    for n, dims in (('full', 17), ('kin', 7)):
        pn = load_particlenet(Path(MODELS).parent / 'particlenet' / f'ParticleNet_{n}.pt', dims, dev)
        models[f'PN_{n}'] = (lambda J, pn=pn, n=n: pn_logits(pn, J, n))
    sb = [(CL.index('Wqq'), [0]), (CL.index('Zqq'), [0]), (CL.index('Tbqq'), [0]), (CL.index('Hbb'), [0])]
    rows = np.arange(len(y))[::10]
    T0r = dict(T0)
    res, NPZ = run_panel('jetclass', T0, y, models, lambda T: jc_jets(J0, T), list(CL), sb, 128, 0.8, 402, condition_list(jetclass_full=True), None, smoke)
    res['sample'] = f'random rows (seed 402) of each class first test file, {smoke or 1000} per class'; res['roundtrip_maxdiff'] = rt
    res['known_acc'] = dict(ParT_full='0.859 (D4, rows 8000-9000), 0.861 published', ParT_kin='0.745 (D4)', PN_full='0.844 published')
    save(res, NPZ, 'jetclass', smoke)
    # observables on relative coordinates (a second pass over the observable conditions, same edits: same seeds)
    try:
        rng = np.random.default_rng(402); sub_ = lambda T: subT(jc_rel(T), rows)
        O0 = observables(sub_(T0r))
        for c in condition_list(jetclass_full=True):
            if c['name'] not in OBS_CONDS: continue
            U, _, _ = apply(T0, c, 128, 0.8, 402); res['observables'][c['name']] = obs_stats(O0, observables(sub_(U)), rng)
            log(f'  observables {c["name"]}: ' + ' '.join(f'{k} {v["mean_delta"]:+.3g} x{v["cross_median"]:.3f}' for k, v in res['observables'][c['name']].items()))
    except Exception as e:
        import traceback; traceback.print_exc(); res['observables_error'] = repr(e)
    save(res, NPZ, 'jetclass', smoke)


# ------------------------------------------------------------------------------------------------ summary + figure
def part_summary():
    R = {p: json.loads((OUT / f'h1_{p}.json').read_text()) for p in ('jedi', 'small', 'jetclass') if (OUT / f'h1_{p}.json').exists()}
    tests = []
    for p, r in R.items():
        for mn, m in r['models'].items():
            for cn, st in m['results'].items():
                if 'vs_control' in st: tests.append((p, mn, cn, st['vs_control']['all'].get('p_mcnemar', 1.0)))
    hp = holm([t[3] for t in tests]); S = dict(n_tests=len(tests), tests={})
    for (p, mn, cn, pv), h in zip(tests, hp):
        st = R[p]['models'][mn]['results'][cn]['vs_control']['all']
        S['tests'][f'{mn}|{cn}'] = dict(p=pv, holm=h, diff=st.get('diff'), diff_ci=st.get('diff_ci'), edit=st.get('edit'), control=st.get('control'), n=st.get('n'))
    S['accuracy'] = {mn: m['acc'] for r in R.values() for mn, m in r['models'].items()}
    (OUT / 'h1_summary.json').write_text(json.dumps(S, indent=1)); log(f'summary: {len(tests)} tests')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    order = [('jedi', 'jedi', 'JEDI-linear'), ('small', 'dsets', 'Deep Sets'), ('small', 'linf', 'Linformer'), ('small', 'tf_d64', 'Transformer'),
             ('small', 'efn', 'EFN (IRC safe)'), ('jetclass', 'ParT_full', 'ParT full'), ('jetclass', 'ParT_kin', 'ParT kin'),
             ('jetclass', 'PN_full', 'ParticleNet full'), ('jetclass', 'PN_kin', 'ParticleNet kin'),
             ('jedi', 'jedi_scrambled', 'JEDI scrambled'), ('small', 'dsets_scrambled', 'Deep Sets scrambled'), ('jetclass', 'ParT_full_scrambled', 'ParT scrambled')]
    order = [o for o in order if o[0] in R and o[1] in R[o[0]]['models']]
    panels = [('Collinear split of the hardest particle (5 % / 95 %, ΔR = 0)', 'split_r0_z0.05_dR0', 'reshuffle_r0_f0.05',
               'control: 5 % of its pT moved to its nearest neighbour'),
              ('Soft particle added (pT = 10⁻⁴ × jet pT, ΔR < 0.8)', 'soft_eps0.0001', 'recoil_eps0.0001',
               'control: the same momentum taken from the hardest particle, nothing added')]
    trained = [o for o in order if 'scrambled' not in o[1]]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw=dict(width_ratios=[1.3, 1.3, 1]))
    w = 0.38
    for a, (title, e, c, clab) in zip(ax[:2], panels):
        for b, (cn, lab, col) in enumerate(((e, 'edit', '#3b6fb6'), (c, clab, '#bbbbbb'))):
            v, lo, hi = [], [], []
            for p, mn, _ in trained:
                st = R[p]['models'][mn]['results'][cn]['all']; v.append(100 * st['changed']); lo.append(100 * st['ci'][0]); hi.append(100 * st['ci'][1])
            xs = np.arange(len(trained)) + (b - 0.5) * w
            a.bar(xs, v, w, color=col, label=lab)
            a.errorbar(xs, v, [np.subtract(v, lo), np.subtract(hi, v)], fmt='none', ecolor='k', lw=0.7)
        a.set_xticks(np.arange(len(trained))); a.set_xticklabels([o[2] for o in trained], rotation=35, ha='right', fontsize=8)
        a.set_ylabel('calls changed (% of 10,000 jets)'); a.set_title(title, fontsize=9.5); a.legend(fontsize=7.5, frameon=False, loc='upper left')
    for p, mn, lab in trained:
        r = R[p]['models'][mn]['results']; ks = [1, 2, 4, 8]
        v = [100 * r['split_r0_z0.5_dR0' if k == 1 else f'multi_k{k}']['all']['changed'] for k in ks]
        ax[2].plot(ks, v, 'o-', label=lab, lw=1.2, ms=3, color='#888888' if mn == 'efn' else None, ls='--' if mn == 'efn' else '-')
    ax[2].set_xscale('log', base=2); ax[2].set_xticks([1, 2, 4, 8]); ax[2].set_xticklabels(['1', '2', '4', '8'])
    ax[2].set_xlabel('number of hardest particles each split in two (ΔR = 0)'); ax[2].set_ylabel('calls changed (%)')
    ax[2].set_title('More collinear splits, more changed calls', fontsize=9.5); ax[2].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / 'h1_panel.png', dpi=130); log(f'wrote {OUT / "h1_panel.png"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['jedi', 'small', 'jetclass', 'summary'])
    ap.add_argument('--smoke', type=int, default=0); a = ap.parse_args()
    log(f'H1 part {a.part} smoke {a.smoke}')
    dict(jedi=part_jedi, small=part_small, jetclass=part_jetclass)[a.part](a.smoke) if a.part != 'summary' else part_summary()
    log('done')
