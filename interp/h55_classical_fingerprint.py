"""H55 (loop D, round D11, exploratory): classical-tagger fingerprint panel. One library of physics edits is applied to the jets
of all four datasets; every trained network, weight-permuted copies of each architecture (null) and classical taggers at
published working points are scored on the same edited jets. Each tagger's responses over the edits form a vector
("fingerprint"); each network's vector is compared with each classical tagger's by cosine similarity, against the similarity
of the permuted networks.

Samples (random rows, never first-k)
  hls  interp.jedi_core.sample(per_class=2000, seed=5501) jets (h1_irc_panel.hls_sample, same indices): 10,000 jets g q W Z t,
       64 slots, massless particles at (eta, phi) = (deta, dphi), stored axis (0, 0).
  jc   JetClass first test file of Hbb, Wqq, Zqq, Tbqq, QCD (ZJetsToNuNu): 2,000 random rows each (default_rng 5502), 128 slots,
       stored axis jet_eta / jet_phi, eta = jet_eta + sign(jet_eta) deta.
  top  Top Landscape test split: 2,000 top + 2,000 QCD random rows (default_rng 5503), 200 slots; axis = E-scheme sum of the
       particles (as the conversion), recomputed after every particle edit.
  qg   Komiske-Metodiev-Thaler Pythia quark/gluon test split: 2,000 quark + 2,000 gluon random rows (default_rng 5504);
       massless; axis = E-scheme sum, recomputed after every particle edit.
Edits (each produces a new particle table; particles re-sorted by pT and cut to the slot limit; 'ok' marks the jets the edit
applies to; every metric is on ok jets)
  split   collinear split of the hardest particle into two with momentum fractions 0.5 / 0.5 in the same direction (dR = 0):
          the four-momentum is conserved exactly, every IRC-safe observable is unchanged, N -> N + 1 (H1 split r0 z0.5 dR0).
          Daughters keep the parent's PID / charge / impact parameters.
  ksplit3 three distinct random particles of the softer half of the jet each split into two halves in the same direction
          (H18v sp<k> with delta = 0): N -> N + 3, every IRC-safe observable unchanged exactly. Jets with fewer than 6
          particles: not ok.
  (split / ksplit3 / soft: jets where a real particle would be pushed out of the slot window are not ok, so these edits are
  exact on every ok jet.)
  soft    one massless neutral particle of pT = 1e-3 pT_jet added at a uniformly random point of the annulus R/2 < dR < R about
          the axis (R = 0.8; QG 0.4), its three-momentum taken from the hardest particle (jet three-momentum fixed) (H1 / H5).
  wta     axis convention: the inputs (and girth) measured about the winner-take-all axis (C/A clustering with WTA pT
          recombination: the direction of one particle; H36 wta_index) instead of the stored / E-scheme axis. Particles fixed.
  ctrl    control for wta: the axis moved by a vector of the same length as the WTA offset in a uniformly random direction
          (equivalently a rigid translation of the inputs). Particles fixed.
  gin     (hls, jc, top) three massless neutral particles with total pT 2 % of the jet pT placed beyond the farther soft-drop
          prong, where Cambridge/Aachen clusters them into that prong before the prongs merge: soft drop keeps them (H49 G_in).
  gout    (hls, jc, top) the same three particles at the same ungroomed-mass change placed away from both prongs, where they
          join after the prongs merge: soft drop removes them (H49 G_out). Both: H49 matched set (|dm_SD(gout)| < 0.5 GeV,
          dm_SD(gin) >= 0.5 dm(gout), |m(gin) - m(gout)| < 1 GeV).
  decay   two exclusive-kT prongs; every particle boosted to the jet rest frame, rotated so that |cos theta*| of the prong axis
          rises by 0.3 (more asymmetric momentum sharing), boosted back (H45 D(+0.3)). One Lorentz transformation for all
          particles: jet four-momentum, jet mass, N and every pairwise mass exactly fixed. ok: both prong centres within 0.75
          (QG 0.35) of the jet axis.
  zboost  longitudinal boost along the beam (z) axis with rapidity shift dy = 0.5 sign(eta_jet) applied to every four-vector:
          jet mass, pT, rapidity differences and azimuth exactly unchanged; particle energies and pseudorapidities change. The
          stored axis is moved with it (eta + dy, the axis being massless). hls4ml inputs (pT, deta, dphi of massless particles)
          are exactly invariant by construction.
Classical taggers (computed on each edited jet; observables: m = invariant mass; m_SD = soft drop beta 0, z_cut 0.1, C/A;
  tau21 / tau32 = N-subjettiness ratios with exclusive-kT axes, beta 1 (h2 top_obs); D2 = e3 / e2^3, beta 1 (h45 d2_batch);
  N = number of particles; pT^D = sqrt(sum pT^2) / sum pT; girth = sum pT dR / sum pT about the axis in use)
  V_tau21   65 < m_SD < 105 GeV and tau21 < 0.45 (CMS 2016 W/Z tagger, JME-16-003 high-purity working point)
  V_DDT     65 < m_SD < 105 GeV and tau21' = tau21 + 0.063 rho' < 0.55, rho' = ln(m_SD^2 / (pT x 1 GeV)) (CMS DDT, EXO-17-001)
  V_D2      65 < m_SD < 105 GeV and D2 < c, c = median D2 of the task's unedited signal jets inside the window (50 % in-window
            signal efficiency on this sample; no published hls4ml / JetClass value)
  mass_V    65 < m < 105 GeV, ungroomed
  top_tau32 105 < m_SD < 210 GeV and tau32 < 0.54 (CMS 2016 top tagger working point, as H49)
  mass_top  105 < m < 210 GeV, ungroomed
  LR_N, LR_NpTD, LR_girth (q vs g): histogram likelihood ratios of N (integer bins), (N, pT^D) (10 x 10 quantile bins), girth (40
            quantile bins), 2-fold cross-fit on the unedited sample (fold by default_rng 5505); call = quark when the ratio is above
            the threshold giving 50 % quark efficiency on unedited jets.
Networks: call = signal when the task log-odds (log p(sig) - log sum p(bkg)) exceeds its median on unedited signal jets
  (50 % signal efficiency; the same definition for every network and the LR taggers).
  hls  JEDI-linear official n64 3-feature (jax env) + 6 weight-permuted copies (H12 permute_weights, seeds 11-16);
       dsets-d64-n64, linf-2b-d32-k8-n64, tf-2b-d64-n64-e100, jedis-d64-n64 (seed0) + 6 permuted copies each (torch seeds 1-6)
  jc   ParT_full, ParT_kin, PN_full, PN_kin (released) + 6 permuted ParT_full and 6 permuted PN_full
  top  ParT_kin, PN_kin (H2 weaver), LorentzNet (released), PFN, EFN (H2 energyflow) + 6 permuted ParT, PN, PFN, EFN
  qg   part_kin, pnet_kin (H5 weaver), pfn_k, efn_k (H2 energyflow, ported), lorentznet (released) + 6 permuted part, pnet, pfn, efn
Metrics (--part stats), per row (network / permuted network / classical tagger) x task x edit, on the task's jets (signal +
  background classes) where the edit applies: chg = fraction of calls changed; flow = (calls gained - calls lost) / jets;
  networks: mean Delta log-odds (edit - unedited) with 95 % bootstrap CI and paired (edit - ctrl); exact McNemar on the call
  flow (calls gained vs lost, H0: the edit does not shift the call rate) per edit, and for wta the McNemar of the changed
  indicators wta vs ctrl; Holm within each family. Fingerprint = (chg_e, flow_e) over the dataset's edits, each component
  divided by its RMS over the trained + classical rows of that task; cosine similarity network vs classical tagger with 95 %
  bootstrap CI over jets (B = 1000, the standardisation recomputed per draw); null: cosine of each permuted network to the same
  tagger; test statistic Delta = cos(net, tagger) - mean_perm cos(perm, tagger); one-sided z = Delta / sqrt(sd_boot^2 + sd_perm^2)
  (sd_boot: jet-bootstrap SD of the network's cosine; sd_perm: SD of the cosine across permuted networks), p = 1 - Phi(z);
  also the bootstrap p = P(Delta_b <= 0) (floor 1/1001); Holm on the z p over networks x taggers (all datasets); nearest tagger per network and its bootstrap frequency.
Run
  AI_PY=conda ./job.sh d-h55-edit 4 12 'python -m interp.h55_classical_fingerprint --part edits --ds hls && JETDISTILL_TAGGER=part
     JETDISTILL_DATA=/j-jepa-vol/aaron-interp/code/cluster/data_part python -m interp.h55_classical_fingerprint --part edits --ds jc
     && python -m interp.h55_classical_fingerprint --part edits --ds top && python -m interp.h55_classical_fingerprint --part edits --ds qg'
  ./job.sh d-h55-jedi 1 4 'python -m interp.h55_classical_fingerprint --part nets --ds hls_jedi'
  AI_PY=conda ./job.sh d-h55-small 1 4 'python -m interp.h55_classical_fingerprint --part nets --ds hls_small'
  ./part_job.sh d-h55-gpu 'python -m interp.h55_classical_fingerprint --part nets --ds jc && ... --ds top && ... --ds qg'
  AI_PY=conda ./job.sh d-h55-stats 1 4 'python -m interp.h55_classical_fingerprint --part stats'
  -> $AI_RESULTS/physics/h55_<ds>_tab.npz, h55_<ds>_obs.npz, h55_<ds>_<group>_logits.npz / _acc.json, h55_fingerprint.json,
     h55_heatmap.png, h55_edits.png. --smoke N: N jets per class (file suffix _smoke).
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp.h1_irc_panel import copyT, setp, finalize, p3, from3, wrap, split as h1_split  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
OUT = RES / 'physics'
V = Path('/j-jepa-vol/aaron-interp')
T00 = time.time()
SEEDS = dict(hls=5501, jc=5502, top=5503, qg=5504)
PER = dict(hls=2000, jc=2000, top=2000, qg=2000)
CAP = dict(hls=64, jc=128, top=200, qg=None)
RJ = dict(hls=0.8, jc=0.8, top=0.8, qg=0.4)
JC_CL = ('Hbb', 'Wqq', 'Zqq', 'Tbqq', 'QCD')
EDITS = ['split', 'ksplit3', 'soft', 'wta', 'ctrl', 'gin', 'gout', 'decay', 'zboost']
CONDS = ['base'] + EDITS
DS_EDITS = {ds: [e for e in EDITS if not (ds == 'qg' and e in ('gin', 'gout'))] for ds in SEEDS}
NPERM = 6
NBOOT = 1000
WIN_V, WIN_T = (65.0, 105.0), (105.0, 210.0)
TAU21_CUT, DDT_CUT, DDT_M, TAU32_CUT = 0.45, 0.55, 0.063, 0.54
EDIT_WORDS = dict(
    split='hardest particle split into two equal collinear halves (exact; N+1)',
    ksplit3='3 soft-half particles each split into two collinear halves (exact; N+3)',
    soft='one neutral particle, pT = 0.1 % of jet pT, at R/2 < dR < R (recoil from hardest)',
    wta='inputs about the winner-take-all axis instead of the stored / E-scheme axis',
    ctrl='axis moved by the WTA offset length in a random direction (control for wta)',
    gin='3 particles (2 % of jet pT) kept by soft drop (beyond the farther prong)',
    gout='3 particles (2 % of jet pT) removed by soft drop (away from both prongs), same ungroomed dm as gin',
    decay='rest-frame rotation: |cos theta*| of the 2-prong axis +0.3 at fixed jet four-momentum',
    zboost='boost along the beam axis, rapidity shift 0.5 (m, pT, dy, dphi fixed)')


def log(*a):
    print(f'[{time.time() - T00:7.0f}s]', *a, flush=True)


def sfx(smoke):
    return '_smoke' if smoke else ''


def nreal(T):
    return (T['pt'] > 0).sum(1)


def eaxis(T):
    real = T['pt'] > 0; px, py, pz = p3(T['pt'], T['eta'], T['phi'])
    _, e, p = from3((px * real).sum(1), (py * real).sum(1), (pz * real).sum(1))
    return e, p


def reaxis(T):
    """axis = E-scheme sum of the particles (Top / QG convention)"""
    T['ax_eta'], T['ax_phi'] = eaxis(T); T['sgn'] = np.where(T['ax_eta'] < 0, -1.0, 1.0); return T


def neutral(U, W, on):
    """appended columns W.. (on (J, K)) become neutral hadrons in the dataset's type fields (h18_softcount_cross.neutral)"""
    K = on.shape[1]; e = U['ext']
    for k in e: e[k][:, W:W + K] = 0.0
    if 'isNeutralHadron' in e: e['isNeutralHadron'][:, W:W + K] = on
    if 'mask' in e: e['mask'][:, W:W + K] = on
    if 'pid' in e: e['pid'][:, W:W + K] = np.where(on, 130.0, 0.0)


def four(T):
    real = T['pt'] > 0; pt, eta, phi, m = T['pt'], T['eta'], T['phi'], T['m']
    P = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)], -1)
    return P * real[..., None]


# ================================================================================================================ edits
def e_split(T, rng):
    slot = np.where(nreal(T) > 0, 0, -1)
    U, ok = h1_split(T, slot, 0.5, 0.0, rng); return U, ok


def e_ksplit(T, k, rng):
    n = nreal(T); J, W = T['pt'].shape; h = n // 2; ok = h >= k; rows = np.arange(J)
    col = np.arange(W)[None, :]; soft = (col >= (n - h)[:, None]) & (col < n[:, None])
    order = np.argsort(np.where(soft, rng.random((J, W)), 2.0), 1)[:, :k]
    U = copyT(T, k); r = rows[ok]
    for j in range(k):
        c = order[ok, j]
        for q in ('pt', 'm'): U[q][r, c] = T[q][r, c] / 2; U[q][r, W + j] = T[q][r, c] / 2
        for q in ('eta', 'phi'): U[q][r, W + j] = T[q][r, c]
        for q in U['ext']: U['ext'][q][r, W + j] = T['ext'][q][r, c]
    return U, ok


def e_soft(T, eps, R, rng):
    J = len(T['pt']); rows = np.arange(J); real = T['pt'] > 0
    px, py, pz = p3(T['pt'], T['eta'], T['phi'])
    ptj = np.hypot((px * real).sum(1), (py * real).sum(1))
    rr = R * np.sqrt(rng.uniform(0.25, 1.0, J)); th = rng.uniform(0, 2 * np.pi, J)
    es = T['ax_eta'] + T['sgn'] * rr * np.cos(th); fs = wrap(T['ax_phi'] + rr * np.sin(th)); q = eps * ptj
    qx, qy, qz = p3(q, es, fs)
    U = copyT(T, 1); c = U['pt'].shape[1] - 1
    setp(U, rows, 0, *from3(px[:, 0] - qx, py[:, 0] - qy, pz[:, 0] - qz), T['m'][:, 0])
    setp(U, rows, c, q, es, fs, np.zeros(J)); neutral(U, c, np.ones((J, 1), bool))
    return U, nreal(T) > 0


def wta_index(eta, phi, pt):
    """C/A clustering to one jet with WTA pT recombination (h36_axis_top_qg.wta_index): index of the axis particle"""
    n = len(pt)
    if n == 1: return 0
    D = (eta[:, None] - eta[None]) ** 2 + wrap(phi[:, None] - phi[None]) ** 2; np.fill_diagonal(D, np.inf)
    w = pt.astype(np.float64).copy(); i = 0
    for _ in range(n - 1):
        k = int(np.argmin(D)); i, j = divmod(k, n)
        if w[j] > w[i]: i, j = j, i
        w[i] += w[j]; D[j, :] = np.inf; D[:, j] = np.inf
    return i


def e_wta(T):
    J = len(T['pt']); ae, ap = np.array(T['ax_eta']), np.array(T['ax_phi'])
    for j in range(J):
        r = T['pt'][j] > 0
        if r.sum() == 0: continue
        e, f, p = T['eta'][j, r], T['phi'][j, r], T['pt'][j, r]; i = wta_index(e, f, p); ae[j], ap[j] = e[i], f[i]
    U = copyT(T); U['ax_eta'], U['ax_phi'] = ae, ap; U['sgn'] = T['sgn']
    return U, nreal(T) > 0


def e_ctrl(T, W, rng):
    d = np.hypot(W['ax_eta'] - T['ax_eta'], wrap(W['ax_phi'] - T['ax_phi'])); th = rng.uniform(0, 2 * np.pi, len(d))
    U = copyT(T); U['ax_eta'] = T['ax_eta'] + d * np.cos(th); U['ax_phi'] = wrap(T['ax_phi'] + d * np.sin(th)); U['sgn'] = T['sgn']
    return U, nreal(T) > 0, d


def e_decay(T, dc, rmax, seed):
    from interp import h45_prong_sharing as H45
    P4 = four(T); nn = nreal(T).astype(np.int64); J = len(nn)
    LAB = H45.kt2_labels(P4, nn); U = copyT(T); ok = np.zeros(J, bool); why = {}
    for j in range(J):
        n = nn[j]
        if n < 3: why['n<3'] = why.get('n<3', 0) + 1; continue
        P = P4[j, :n].astype(np.float64); lab = LAB[j, :n]
        r, inf = H45.edit_one(P, lab, dc, np.random.default_rng([seed, j]))
        if r is None: why[inf['reason']] = why.get(inf['reason'], 0) + 1; continue
        D = r['D']; Tt = D.sum(0); ej, pj = H45.rap(Tt), np.arctan2(Tt[1], Tt[0]); good = True
        for k in (0, 1):
            Vk = D[lab == k].sum(0); good &= bool(np.hypot(H45.rap(Vk) - ej, wrap(np.arctan2(Vk[1], Vk[0]) - pj)) <= rmax)
        if not good: why['prong_outside'] = why.get('prong_outside', 0) + 1; continue
        pt, eta, phi, m = H45.ptetaphim(D)
        U['pt'][j, :n], U['eta'][j, :n], U['phi'][j, :n], U['m'][j, :n] = pt, eta, phi, m; ok[j] = True
    why['ok'] = int(ok.sum()); log(f'  decay edit: {why}')
    return U, ok


def e_zboost(T, dy0):
    U = copyT(T); real = T['pt'] > 0; dy = dy0 * T['sgn']
    pt, eta, m = T['pt'], T['eta'], T['m']; mt = np.sqrt(pt ** 2 + m ** 2)
    E = np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2); pz = pt * np.sinh(eta)
    y = 0.5 * np.log(np.maximum(E + pz, 1e-300) / np.maximum(E - pz, 1e-300)) + dy[:, None]
    U['eta'] = np.where(real, np.arcsinh(mt * np.sinh(y) / np.maximum(pt, 1e-300)), 0.0)
    U['ax_eta'] = T['ax_eta'] + dy; return U, nreal(T) > 0


def make_conds(T0, ds, seed):
    """dict cond -> (table, ok, info); tables finalized (pT-sorted, capped)"""
    cap = CAP[ds] or T0['pt'].shape[1] + 8; rng = np.random.default_rng(seed + 100); C = {}
    own_axis = ds in ('top', 'qg')
    fin = lambda U: finalize(U, cap)
    B, _ = fin(T0); C['base'] = (B, np.ones(len(B['pt']), bool), {})
    for name, fn in (('split', lambda: e_split(B, rng)), ('ksplit3', lambda: e_ksplit(B, 3, rng)),
                     ('soft', lambda: e_soft(B, 1e-3, RJ[ds], rng))):
        U, ok = fn(); U, over = fin(U); ok = ok & ~over       # jets where a real particle would leave the slot window: not ok
        if own_axis: reaxis(U)
        C[name] = (U, ok, dict(overflow=float(over.mean()))); log(f'  {name}: ok {ok.mean():.3f}, slot overflow {over.mean():.3f}')
    W, ok = e_wta(B); C['wta'] = (W, ok, {})
    U, ok, d = e_ctrl(B, W, rng); C['ctrl'] = (U, ok, dict(offset_median=float(np.median(d)), offset_q=np.percentile(d, [25, 75]).tolist()))
    log(f'  wta / ctrl: median axis offset {np.median(d):.4f}')
    if ds != 'qg':
        from interp import h49_groomed_mass as H49
        H49.SEEDS['_cur'] = seed
        G, conds = H49.geometry(B, cap)
        okg = G['found'] & G['kept'] & G['bracket'] & (G['Rg'] >= H49.RG_MIN) & (G['rin'] <= H49.R_MAX) & (G['rstar'] <= H49.R_MAX)
        for nm, key in (('gin', 'Gin'), ('gout', 'Gout')):
            U = conds[key]
            if own_axis: reaxis(U)
            C[nm] = (U, okg.copy(), dict(geometry_ok=float(okg.mean())))
        log(f'  gin / gout: geometry ok {okg.mean():.3f}')
    U, ok = e_decay(B, 0.3, 0.75 if ds != 'qg' else 0.35, seed + 7); U, _ = fin(U)
    if own_axis: reaxis(U)
    C['decay'] = (U, ok, {})
    U, ok = e_zboost(B, 0.5); U, _ = fin(U)
    if own_axis: reaxis(U)
    C['zboost'] = (U, ok, {})
    return C


# ================================================================================================================ observables
def observables(T, ds, base=None):
    """classical observables on a table; base given -> only girth recomputed (axis-only edits)"""
    real = T['pt'] > 0; pt = T['pt']; S = pt.sum(1)
    de = np.where(real, T['eta'] - T['ax_eta'][:, None], 0); dp = np.where(real, wrap(T['phi'] - T['ax_phi'][:, None]), 0)
    girth = (pt * np.hypot(de, dp)).sum(1) / np.maximum(S, 1e-12)
    if base is not None:
        O = {k: v.copy() for k, v in base.items()}; O['girth'] = girth; return O
    from interp import h4_top_wmass as H4
    P4 = four(T).astype(np.float32); nn = nreal(T).astype(np.int64)
    o = H4._obs(P4, nn); P = four(T).sum(1)
    O = dict(m=o['m'], msd=o['mSD'], tau21=o['tau21'], tau32=o['tau32'], N=nn.astype(float),
             ptd=np.sqrt((pt ** 2).sum(1)) / np.maximum(S, 1e-12), girth=girth, ptj=np.hypot(P[:, 0], P[:, 1]))
    if ds in ('hls', 'jc'):
        from interp import h45_prong_sharing as H45
        O['d2'] = H45.d2_batch(np.ascontiguousarray(four(T).astype(np.float64)), nn)
    return O


# ================================================================================================================ samples
def _rows(y, per, seed, classes=(0, 1)):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per, replace=False) for c in classes]))


def load_ds(ds, per):
    seed = SEEDS[ds]
    if ds == 'hls':
        from interp.h1_irc_panel import hls_sample
        T, y = hls_sample(per, seed); return T, y, dict(sample=f'interp.jedi_core.sample(per_class={per}, seed={seed}) jets', classes=['g', 'q', 'W', 'Z', 't'])
    if ds == 'jc':
        import uproot
        from interp.part_core import read_root, fof, cat, sub, CL
        from interp.h1_irc_panel import jc_table
        rng = np.random.default_rng(seed); Js = []
        for c in JC_CL:
            f = fof(c); n = uproot.open(f)['tree'].num_entries; idx = np.sort(rng.choice(n, per, replace=False)); parts = []
            for a in range(0, n, 20000):
                loc = idx[(idx >= a) & (idx < a + 20000)] - a
                if len(loc): parts.append(sub(read_root(f, start=a, stop=min(a + 20000, n)), loc))
            Js.append(cat(parts)); log(f'JetClass {c}: {per} random rows of {n} ({Path(f).name})')
        J0 = cat(Js); T = jc_table(J0); y = J0['label'].astype(int)
        return T, y, dict(sample=f'JetClass first test file per class {JC_CL}, {per} random rows each (default_rng {seed})', classes=list(CL))
    from interp.h12_boost_panel import table_from_full
    if ds == 'top':
        from interp import data_top as D
    else:
        from interp import data_qg as D
    d = D.DST / 'test'; ylab = np.load(d / 'label.npy'); idx = _rows(ylab, per, seed)
    t = time.time(); F = np.load(d / 'full.npy')[idx]; jet = np.load(d / 'jet.npy')[idx]
    log(f'{ds} test rows {len(idx)} read ({time.time() - t:.0f} s)')
    nn = np.minimum(jet[:, 5].astype(np.int64), F.shape[1]); y = ylab[idx].astype(int); T = table_from_full(F, nn); del F
    if ds == 'qg':
        T['m'] = np.zeros_like(T['m']); pid = np.load(d / 'pid.npy')[idx].astype(np.float64); T['ext'] = dict(pid=pid * (T['pt'] > 0))
        CAP['qg'] = T['pt'].shape[1]
    T['sgn'] = np.where(T['ax_eta'] < 0, -1.0, 1.0)
    return T, y, dict(sample=f'{ds} test split, {per} per class random rows (default_rng {seed})', classes=['QCD', 'top'] if ds == 'top' else ['gluon', 'quark'],
                      idx=idx.tolist()[:5])


# ================================================================================================================ table io
def save_tab(path, C, y, info):
    Z = dict(y=y.astype(np.int16))
    for c, (T, ok, inf) in C.items():
        for k in ('pt', 'eta', 'phi', 'm'): Z[f'{c}|{k}'] = T[k]
        for k in ('ax_eta', 'ax_phi', 'sgn'): Z[f'{c}|{k}'] = np.asarray(T[k], np.float64)
        for k, v in T['ext'].items(): Z[f'{c}|ext|{k}'] = v.astype(np.float32)
        Z[f'{c}|ok'] = ok
    np.savez(path, **Z); (path.with_suffix('.json')).write_text(json.dumps(dict(info=info, cond_info={c: v[2] for c, v in C.items()}), indent=1, default=float))


def load_tab(path, conds=None):
    Z = np.load(path); out = {}; y = Z['y'].astype(int)
    keys = Z.files; cs = conds or sorted({k.split('|')[0] for k in keys if '|' in k}, key=lambda c: CONDS.index(c))
    for c in cs:
        T = {k: Z[f'{c}|{k}'] for k in ('pt', 'eta', 'phi', 'm', 'ax_eta', 'ax_phi', 'sgn')}
        T['ext'] = {k.split('|')[2]: Z[k].astype(np.float64) for k in keys if k.startswith(f'{c}|ext|')}
        out[c] = (T, Z[f'{c}|ok'])
    return out, y


# ================================================================================================================ part: edits
def part_edits(ds, smoke):
    per = smoke or PER[ds]; T0, y, info = load_ds(ds, per); log(f'[{ds}] {len(y)} jets, label counts {np.bincount(y).tolist()}')
    C = make_conds(T0, ds, SEEDS[ds]); OUT.mkdir(parents=True, exist_ok=True)
    O = {}
    for c, (T, ok, _) in C.items():
        t = time.time(); O[c] = observables(T, ds, O['base'] if c in ('wta', 'ctrl') else None)
        log(f'  observables {c} ({time.time() - t:.0f} s): mean m {O[c]["m"].mean():.2f}, m_SD {O[c]["msd"].mean():.2f}, N {O[c]["N"].mean():.2f}')
    # H49 matched-set requirement on the groomed / ungroomed mass changes
    if 'gin' in C:
        dmsd_out = O['gout']['msd'] - O['base']['msd']; dm_out = O['gout']['m'] - O['base']['m']; dmsd_in = O['gin']['msd'] - O['base']['msd']
        match = (np.abs(dmsd_out) < 0.5) & (dmsd_in >= 0.5 * dm_out) & (np.abs(O['gin']['m'] - O['gout']['m']) < 1.0)
        for c in ('gin', 'gout'):
            T, ok, inf = C[c]; ok2 = ok & match; inf['matched'] = float(ok2.mean()); C[c] = (T, ok2, inf)
        log(f'  gin / gout matched fraction {C["gin"][1].mean():.3f}')
    for c, (T, ok, inf) in C.items():
        if c == 'base': continue
        d = {k: (O[c][k] - O['base'][k])[ok] for k in O[c]}
        inf['obs_change'] = {k: dict(mean=float(v.mean()) if len(v) else None, max_abs=float(np.abs(v).max()) if len(v) else None) for k, v in d.items()}
    save_tab(OUT / f'h55_{ds}_tab{sfx(smoke)}.npz', C, y, info)
    np.savez(OUT / f'h55_{ds}_obs{sfx(smoke)}.npz', **{f'{c}|{k}': v for c, o in O.items() for k, v in o.items()})
    log(f'[{ds}] wrote tables and observables')


# ================================================================================================================ part: nets
def permute_torch(model, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            f = p.data.view(-1); idx = torch.randperm(f.numel(), generator=g).to(f.device); f.copy_(f[idx].clone())
    return model


def rel(T):
    real = T['pt'] > 0
    return np.where(real, T['eta'] - T['ax_eta'][:, None], 0), np.where(real, wrap(T['phi'] - T['ax_phi'][:, None]), 0)


def hls_x(T):
    de, dp = rel(T); real = T['pt'] > 0
    return np.stack([np.where(real, T['pt'], 0), de, dp], -1)[:, :64].astype(np.float32)


def run_nets(ds, group, models, prep, smoke, extra=None):
    """models: {name: f(prepped) -> logits (J, C)}"""
    tabs, y = load_tab(OUT / f'h55_{ds}_tab{sfx(smoke)}.npz'); Z = {}; acc = dict(extra or {})
    for c, (T, ok) in tabs.items():
        t = time.time(); X = prep(T)
        for mn, f in models.items(): Z[f'{mn}|{c}'] = np.asarray(f(X), np.float32)
        log(f'  [{group}] {c}: {len(models)} models scored ({time.time() - t:.0f} s)')
    for mn in models:
        L = Z[f'{mn}|base']; c0 = L.argmax(1)
        acc[mn] = dict(acc=float((c0 == y).mean()) if L.shape[1] > 2 else float(((L[:, 1] > L[:, 0]).astype(int) == y).mean()),
                       per_class={int(k): float((c0[y == k] == k).mean()) for k in np.unique(y)} if L.shape[1] > 2 else None)
        if L.shape[1] == 2:
            from sklearn.metrics import roc_auc_score
            acc[mn]['auc'] = float(roc_auc_score(y, L[:, 1] - L[:, 0]))
        log(f'  [{group}] {mn}: unedited accuracy {acc[mn]["acc"]:.4f}')
    np.savez(OUT / f'h55_{ds}_{group}_logits{sfx(smoke)}.npz', **Z)
    (OUT / f'h55_{ds}_{group}_acc{sfx(smoke)}.json').write_text(json.dumps(acc, indent=1, default=float))


def nets_hls_jedi(smoke):
    from interp.jedi_core import Jedi
    from jetdistill import network as N
    from interp.h12_boost_panel import permute_weights
    nets = {'JEDI': Jedi(64, '3-feature')}
    orig = N.redraw_untrained; N.redraw_untrained = permute_weights
    try:
        for s in range(NPERM): nets[f'JEDI_perm{s}'] = Jedi(64, '3-feature', untrained=True, seed=11 + s)
    finally: N.redraw_untrained = orig
    models = {k: (lambda x, n=n: np.asarray(n.run(x)['logits'], np.float64)) for k, n in nets.items()}
    run_nets('hls', 'jedi', models, hls_x, smoke, dict(_known=dict(JEDI='5-class acc 0.8092 (260k test); 0.808 (H13)')))


def nets_hls_small(smoke):
    import torch
    from interp.h1_irc_panel import load_small, torch_logits
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    SMALL = V / 'models' / 'small'; models = {}; known = {}
    for tag, run in (('dsets', 'dsets-d64-n64'), ('linf', 'linf-2b-d32-k8-n64'), ('tf', 'tf-2b-d64-n64-e100'), ('jedis', 'jedis-d64-n64')):
        net, sh, sc = load_small(SMALL / run / 'seed0'); models[tag] = (lambda x, net=net, sh=sh, sc=sc: torch_logits(net, x, sh, sc))
        known[tag] = json.loads((SMALL / run / 'seed0' / 'test.json').read_text())['accuracy']
        for s in range(NPERM):
            n2, _, _ = load_small(SMALL / run / 'seed0'); permute_torch(n2, 1 + s)
            models[f'{tag}_perm{s}'] = (lambda x, net=n2, sh=sh, sc=sc: torch_logits(net, x, sh, sc))
    run_nets('hls', 'small', models, hls_x, smoke, dict(_known=known))


def nets_jc(smoke):
    from interp.h1_irc_panel import pn_logits
    from interp.part_core import ParTNetwork, inputs, build
    from interp.p1v_verify_part import fwd
    from interp.pn_net import load_particlenet
    from jetdistill.config import MODELS
    dev = os.environ.get('JETDISTILL_DEVICE', 'cpu'); models = {}
    for n in ('full', 'kin'):
        net = ParTNetwork(n); models[f'ParT_{n}'] = (lambda J, net=net, n=n: fwd(net, *inputs(J, n))[1])
    for s in range(NPERM):
        net = ParTNetwork('full'); permute_torch(net.model, 1 + s); models[f'ParT_full_perm{s}'] = (lambda J, net=net: fwd(net, *inputs(J, 'full'))[1])
    pth = lambda n: Path(MODELS).parent / 'particlenet' / f'ParticleNet_{n}.pt'
    for n, dims in (('full', 17), ('kin', 7)):
        pn = load_particlenet(pth(n), dims, dev); models[f'PN_{n}'] = (lambda J, pn=pn, n=n: pn_logits(pn, J, n))
    for s in range(NPERM):
        pn = permute_torch(load_particlenet(pth('full'), 17, dev), 1 + s); models[f'PN_full_perm{s}'] = (lambda J, pn=pn: pn_logits(pn, J, 'full'))

    def prep(T):
        sg = np.where(T['ax_eta'] < 0, -1.0, 1.0)[:, None]; de, dp = rel(T)
        Jx = dict(jet_eta=T['ax_eta'].astype(np.float32), jet_phi=T['ax_phi'].astype(np.float32), label=np.zeros(len(de), np.int64))
        return build(Jx, T['pt'], sg * de, dp, T['m'], extra=T['ext'])
    run_nets('jc', 'nets', models, prep, smoke, dict(_known=dict(
        ParT_full='per class (H18/H49) QCD 0.783 Zqq 0.689 Wqq 0.805 Tbqq 0.956; 10-class 0.859 (D4)', ParT_kin='QCD 0.718 Zqq 0.550 Wqq 0.702 Tbqq 0.912',
        PN_full='QCD 0.764 Zqq 0.652 Wqq 0.784 Tbqq 0.949', PN_kin='QCD 0.693 Zqq 0.528 Wqq 0.680 Tbqq 0.913')))


def nets_top(smoke):
    import torch
    from interp import h4_top_wmass as H4
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    W = {}
    for kind, dname in (('ParT', 'part_kin'), ('PN', 'pnet_kin')):
        ck = V / 'models/top' / dname / 'net_best_epoch_state.pt'; W[f'{kind}_kin'] = H4.WeaverNet(kind, ck, dev)
        for s in range(NPERM):
            w2 = H4.WeaverNet(kind, ck, dev); permute_torch(w2.model, 1 + s); W[f'{kind}_kin_perm{s}'] = w2
    FL = {a.upper(): H4.Flow(a) for a in ('pfn', 'efn')}
    for a in ('PFN', 'EFN'):
        for s in range(NPERM):
            f2 = copy.deepcopy(FL[a]); rng = np.random.default_rng(1 + s)
            perm = lambda L: [(rng.permutation(K.ravel()).reshape(K.shape), rng.permutation(b.ravel()).reshape(b.shape)) for K, b in L]
            f2.phi, f2.F = perm(f2.phi), perm(f2.F); f2.out = perm([f2.out])[0]; FL[f'{a}_perm{s}'] = f2
    ln = H4.LNet(dev, mass64=True)

    def prep(T):
        P4 = four(T).astype(np.float32); nn = nreal(T); return dict(P4=P4, nn=nn, ae=T['ax_eta'], ap=T['ax_phi'])

    def jets(X):
        J = H4.jet_inputs(X['P4'], X['nn']); m = J['mask']; P = X['P4'].astype(np.float64)
        ptc = np.where(m, J['pt'], 1.0); eta = np.arcsinh(np.where(m, P[..., 2], 0) / ptc); phi = np.arctan2(P[..., 1], P[..., 0])
        S = (P * m[..., None]).sum(1); jeta = np.arcsinh(S[:, 2] / np.hypot(S[:, 0], S[:, 1])); sg = np.where(jeta < 0, -1.0, 1.0)
        J['deta'] = np.where(m, (eta - X['ae'][:, None]) * sg[:, None], 0); J['dphi'] = np.where(m, wrap(phi - X['ap'][:, None]), 0)
        return J
    cache = {}

    def weaver_all(X):
        if cache.get('X') is X: return cache['out']
        J = jets(X); out = {k: np.zeros(len(X['nn'])) for k in W}
        for b in range(0, len(X['nn']), 512):
            Jb = {k: (v[b:b + 512] if np.ndim(v) else v) for k, v in J.items()}; pts, x, v, mk = H4.weaver_inputs(Jb)
            d = lambda a: torch.from_numpy(a).to(dev)
            with torch.no_grad():
                for name, wn in W.items():
                    o = (wn.model(d(x), v=d(v), mask=d(mk)) if wn.kind == 'ParT' else wn.model(d(pts), d(x), d(mk))).float().cpu().numpy()
                    out[name][b:b + len(o)] = o[:, 0] - o[:, 1]          # class 0 = top
        k3 = np.stack([J['pt'][:, :128], J['deta'][:, :128], J['dphi'][:, :128]], -1).astype(np.float16).astype(np.float32)
        for name, f in FL.items(): out[name] = f.score(k3)
        cache['X'] = X; cache['out'] = out; return out
    two = lambda s: np.stack([np.zeros_like(s), s], 1)
    order = np.random.default_rng(SEEDS['top'] + 5).permutation(2 * (smoke or PER['top']))
    models = {k: (lambda X, k=k: two(weaver_all(X)[k])) for k in list(W) + list(FL)}
    models['LorentzNet'] = lambda X: two(ln.score(X['P4'], X['nn'], order))
    known = dict(ParT_kin='0.940 (weaver test, H2)', PN_kin='0.940 (weaver test, H2)', PFN=0.9283, EFN=0.9218)
    try: known['LorentzNet'] = json.loads((V / 'models/top/lorentznet/test-result.json').read_text()).get('test_acc')
    except Exception: pass
    run_nets('top', 'nets', models, prep, smoke, dict(_known=known))


def nets_qg(smoke):
    import torch
    from interp import h36_axis_top_qg  # noqa: F401  (patches h5_qg_counting.kinematics: inputs about Jt['_cond'] axis)
    from interp import h5_qg_counting as H5
    from interp import h14_qg_generator as H14
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; M = H5.Models(dev)
    for name, arch in (('part_kin', 'part'), ('pnet_kin', 'pn')): M.add_weaver(name, arch, False)
    for kind in ('pfn', 'efn'): H14.add_keras_ef(M, kind)
    M.add_lnet()
    for src in ('part_kin', 'pnet_kin', 'pfn_k', 'efn_k'):
        kind, arch, pidfeat, net = M.m[src]
        for s in range(NPERM):
            n2 = permute_torch(copy.deepcopy(net), 1 + s); M.m[f'{src}_perm{s}'] = (kind, arch, pidfeat, n2.eval())

    def prep(T):
        real = T['pt'] > 0
        return dict(pt=T['pt'], eta=T['eta'], phi=T['phi'], pid=(T['ext']['pid'] * real).astype(np.int64), n=real.sum(1),
                    _cond=('ax', np.asarray(T['ax_eta']), np.asarray(T['ax_phi'])))
    two = lambda s: np.stack([np.zeros_like(s), s], 1)     # class 1 = quark; Models.logit = log p(g) - log p(q)
    models = {k: (lambda X, k=k: two(-M.logit(k, X))) for k in M.m}
    run_nets('qg', 'nets', models, prep, smoke, dict(_known=dict(part_kin='0.827 (weaver test, H2)', pnet_kin='0.827 (weaver test, H2)'), ckpt=M.info))


# ================================================================================================================ part: stats
DS_TASKS = dict(
    hls={'W_vs_QCD': (2, [0, 1]), 'Z_vs_QCD': (3, [0, 1]), 't_vs_QCD': (4, [0, 1]), 'q_vs_g': (1, [0])},
    jc=None, top={'top_vs_QCD': (1, [0])}, qg={'q_vs_g': (1, [0])})
TASK_TAGGERS = {'W_vs_QCD': ['V_tau21', 'V_DDT', 'V_D2', 'mass_V'], 'Z_vs_QCD': ['V_tau21', 'V_DDT', 'V_D2', 'mass_V'],
                't_vs_QCD': ['top_tau32', 'mass_top'], 'Top_vs_QCD': ['top_tau32', 'mass_top'], 'top_vs_QCD': ['top_tau32', 'mass_top'],
                'q_vs_g': ['LR_N', 'LR_NpTD', 'LR_girth'], 'Hbb_vs_QCD': []}


def jc_tasks():
    CL = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']; i = CL.index
    return {'W_vs_QCD': (i('Wqq'), [i('QCD')]), 'Z_vs_QCD': (i('Zqq'), [i('QCD')]), 'Top_vs_QCD': (i('Tbqq'), [i('QCD')]), 'Hbb_vs_QCD': (i('Hbb'), [i('QCD')])}


def logodds(L, sig, bkg):
    L = L.astype(np.float64); L = L - L.max(1, keepdims=True); ls = L - np.log(np.exp(L).sum(1, keepdims=True))
    b = ls[:, bkg]; mb = b.max(1)
    return ls[:, sig] - (mb + np.log(np.exp(b - mb[:, None]).sum(1)))


def binom_p(k, n):
    if n == 0: return 1.0
    k = min(k, n - k)
    lp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    m = max(lp); return float(min(1.0, 2 * math.exp(m) * sum(math.exp(v - m) for v in lp)))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); out = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out.tolist()


def lr_tables(O0, yq, fold, kind):
    """histogram log-likelihood ratio (quark-positive), 2-fold; returns f(O) -> score using the other fold's table per jet"""
    def feats(O):
        if kind == 'LR_N': return [O['N']]
        if kind == 'LR_NpTD': return [O['N'], O['ptd']]
        return [O['girth']]
    X0 = feats(O0)
    if kind == 'LR_N': edges = [np.arange(-0.5, 400.5, 1.0)]
    elif kind == 'LR_NpTD': edges = [np.unique(np.quantile(x, np.linspace(0, 1, 11))) for x in X0]
    else: edges = [np.unique(np.quantile(X0[0], np.linspace(0, 1, 41)))]
    for e in edges: e[0], e[-1] = -1e9, 1e9
    tabs = []
    for f in (0, 1):
        tr = fold == f
        hq = np.histogramdd(np.stack([x[tr & (yq == 1)] for x in X0], 1), edges)[0] + 0.5
        hg = np.histogramdd(np.stack([x[tr & (yq == 0)] for x in X0], 1), edges)[0] + 0.5
        tabs.append(np.log(hq / hq.sum()) - np.log(hg / hg.sum()))

    def score(O):
        X = feats(O); ix = tuple(np.clip(np.searchsorted(e, x, side='right') - 1, 0, len(e) - 2) for e, x in zip(edges, X))
        s0, s1 = tabs[1][ix], tabs[0][ix]           # fold-0 jets scored by the fold-1 table and vice versa
        return np.where(fold == 0, s0, s1)
    return score


def classical_calls(ds, task, sig, bkg, O, y, rows):
    """{tagger: {cond: bool calls on rows}} plus efficiencies on unedited jets"""
    out, info = {}, {}
    ysig = y[rows] == sig
    for tg in TASK_TAGGERS[task]:
        calls = {}
        if tg.startswith('V_') or tg == 'mass_V' or tg.startswith('top_') or tg == 'mass_top':
            lo, hi = WIN_T if 'top' in tg else WIN_V
            if tg == 'V_D2':
                b = O['base']; inw = (b['msd'][rows] > lo) & (b['msd'][rows] < hi) & ysig; c = float(np.median(b['d2'][rows][inw])); info['V_D2_cut'] = c
            for cnd, o in O.items():
                ms, m = o['msd'][rows], o['m'][rows]
                if tg == 'mass_V' or tg == 'mass_top': calls[cnd] = (m > lo) & (m < hi)
                elif tg == 'V_tau21': calls[cnd] = (ms > lo) & (ms < hi) & (o['tau21'][rows] < TAU21_CUT)
                elif tg == 'V_DDT':
                    rho = np.log(np.maximum(ms, 1e-3) ** 2 / np.maximum(o['ptj'][rows], 1e-3))
                    calls[cnd] = (ms > lo) & (ms < hi) & (o['tau21'][rows] + DDT_M * rho < DDT_CUT)
                elif tg == 'V_D2': calls[cnd] = (ms > lo) & (ms < hi) & (o['d2'][rows] < c)
                elif tg == 'top_tau32': calls[cnd] = (ms > lo) & (ms < hi) & (o['tau32'][rows] < TAU32_CUT)
        else:
            fold = np.random.default_rng(5505).integers(0, 2, len(rows)); yq = ysig.astype(int)
            sub = lambda o: {k: v[rows] for k, v in o.items()}
            f = lr_tables(sub(O['base']), yq, fold, tg); s0 = f(sub(O['base']))
            cand = np.unique(s0); effs = np.array([(s0[ysig] >= t).mean() for t in cand]); thr = cand[np.argmin(np.abs(effs - 0.5))]
            for cnd, o in O.items(): calls[cnd] = f(sub(o)) >= thr
        c0 = calls['base']; info[tg] = dict(eff_sig=float(c0[ysig].mean()), eff_bkg=float(c0[~ysig].mean()))
        out[tg] = calls
    return out, info


def response_arrays(calls, oks, edits):
    """per-jet changed (n, E) and flow (n, E) indicators and ok masks"""
    c0 = calls['base'].astype(np.int8)
    ch = np.stack([(calls[e].astype(np.int8) != c0) & oks[e] for e in edits], 1).astype(np.float64)
    fl = np.stack([(calls[e].astype(np.int8) - c0) * oks[e] for e in edits], 1).astype(np.float64)
    ok = np.stack([oks[e] for e in edits], 1).astype(np.float64)
    return ch, fl, ok


def cosmat(F, ref_rows):
    """F (R, D) fingerprints; scale columns by RMS over ref_rows; drop zero columns; cosine matrix (R, R)"""
    s = np.sqrt((F[ref_rows] ** 2).mean(0)); keep = s > 1e-12
    G = F[:, keep] / s[keep]; nrm = np.linalg.norm(G, axis=1); nrm = np.where(nrm > 0, nrm, np.inf)
    G = G / nrm[:, None]; return G @ G.T


def part_stats(smoke):
    import matplotlib; matplotlib.use('Agg')
    rng = np.random.default_rng(55)
    R = dict(edits=EDIT_WORDS, datasets={}, tests=[])
    ALLHEAT = []
    for ds in ('hls', 'jc', 'top', 'qg'):
        p = OUT / f'h55_{ds}_obs{sfx(smoke)}.npz'
        if not p.exists(): log(f'[{ds}] no observables, skipped'); continue
        Zo = np.load(p); O = {}
        for k in Zo.files:
            c, q = k.split('|'); O.setdefault(c, {})[q] = Zo[k]
        Zt = np.load(OUT / f'h55_{ds}_tab{sfx(smoke)}.npz'); y = Zt['y'].astype(int); oks = {c: Zt[f'{c}|ok'] for c in CONDS if f'{c}|ok' in Zt.files}
        tinfo = json.loads((OUT / f'h55_{ds}_tab{sfx(smoke)}.json').read_text())
        L = {}; acc = {}
        for g in ('jedi', 'small', 'nets'):
            q = OUT / f'h55_{ds}_{g}_logits{sfx(smoke)}.npz'
            if q.exists():
                Zl = np.load(q)
                for k in Zl.files:
                    mn, c = k.split('|'); L.setdefault(mn, {})[c] = Zl[k]
                acc.update(json.loads((OUT / f'h55_{ds}_{g}_acc{sfx(smoke)}.json').read_text()))
        edits = [e for e in DS_EDITS[ds] if e in oks]
        tasks = DS_TASKS[ds] or jc_tasks()
        D = dict(N=int(len(y)), labels=np.bincount(y).tolist(), sample=tinfo['info'].get('sample'), cond_info=tinfo['cond_info'],
                 accuracy=acc, tasks={})
        for task, (sig, bkg) in tasks.items():
            rows = np.flatnonzero(np.isin(y, [sig] + bkg)); ysig = y[rows] == sig; n = len(rows)
            okr = {c: oks[c][rows] for c in oks}
            Ocl = {c: O[c] for c in ['base'] + edits}
            ccalls, cinfo = classical_calls(ds, task, sig, bkg, Ocl, y, rows)
            T = dict(n=n, n_sig=int(ysig.sum()), classical=cinfo, rows={}, edits=edits)
            names, kinds, CH, FL, OK = [], [], [], [], []
            for tg, calls in ccalls.items():
                ch, fl, ok = response_arrays(calls, okr, edits); names.append(tg); kinds.append('classical'); CH.append(ch); FL.append(fl); OK.append(ok)
            for mn, lc in L.items():
                s = {c: logodds(lc[c][rows], sig, bkg) for c in ['base'] + edits}
                thr = float(np.median(s['base'][ysig])); calls = {c: v > thr for c, v in s.items()}
                ch, fl, ok = response_arrays(calls, okr, edits)
                kind = 'perm' if '_perm' in mn else 'net'; names.append(mn); kinds.append(kind); CH.append(ch); FL.append(fl); OK.append(ok)
                r = dict(kind=kind, eff_sig=float(calls['base'][ysig].mean()), eff_bkg=float(calls['base'][~ysig].mean()), per_edit={})
                if kind == 'net':
                    from sklearn.metrics import roc_auc_score
                    r['auc'] = float(roc_auc_score(ysig, s['base']))
                    chc = calls['ctrl'] != calls['base']
                    for j, e in enumerate(edits):
                        o = okr[e].astype(bool); d = (s[e] - s['base'])[o]; dc = (s[e] - s['ctrl'])[o]
                        bi = rng.integers(0, max(o.sum(), 1), (NBOOT, max(o.sum(), 1)))
                        che = (calls[e] != calls['base'])[o]
                        n10, n01 = int((che & ~chc[o]).sum()), int((~che & chc[o]).sum())
                        r['per_edit'][e] = dict(n_ok=int(o.sum()), chg=float(che.mean()) if o.any() else None,
                                                flow=float((calls[e].astype(int) - calls['base'].astype(int))[o].mean()) if o.any() else None,
                                                dlo=float(d.mean()) if o.any() else None,
                                                dlo_ci=np.percentile(d[bi].mean(1), [2.5, 97.5]).tolist() if o.any() else None,
                                                dlo_minus_ctrl=float(dc.mean()) if o.any() else None,
                                                dlo_minus_ctrl_ci=np.percentile(dc[bi].mean(1), [2.5, 97.5]).tolist() if o.any() else None,
                                                mcnemar_vs_ctrl=dict(edit_only=n10, ctrl_only=n01, p=binom_p(n10, n10 + n01)))
                        g_, l_ = int((calls[e] & ~calls['base'])[o].sum()), int((~calls[e] & calls['base'])[o].sum())
                        r['per_edit'][e]['mcnemar_flow'] = dict(gained=g_, lost=l_, p=binom_p(g_, g_ + l_))
                        R['tests'].append(dict(ds=ds, task=task, model=mn, edit=e, p=r['per_edit'][e]['mcnemar_flow']['p'], family='flow'))
                        if e == 'wta': R['tests'].append(dict(ds=ds, task=task, model=mn, edit=e, p=r['per_edit'][e]['mcnemar_vs_ctrl']['p'], family='wta_vs_ctrl'))
                T['rows'][mn] = r
            for i, nm in enumerate(names):
                if kinds[i] == 'classical':
                    T['rows'][nm] = dict(kind='classical', eff_sig=cinfo[nm]['eff_sig'], eff_bkg=cinfo[nm]['eff_bkg'], per_edit={
                        e: dict(n_ok=int(OK[i][:, j].sum()), chg=float(CH[i][:, j].sum() / max(OK[i][:, j].sum(), 1)),
                                flow=float(FL[i][:, j].sum() / max(OK[i][:, j].sum(), 1))) for j, e in enumerate(edits)})
            # ------------------------------------------------ fingerprints + bootstrap
            CH, FL, OK = np.stack(CH), np.stack(FL), np.stack(OK)          # (R, n, E)
            kinds = np.array(kinds); ref = np.flatnonzero(kinds != 'perm')
            fp = lambda w: np.concatenate([np.einsum('n,rne->re', w, CH), np.einsum('n,rne->re', w, FL)], 1) / \
                np.maximum(np.concatenate([np.einsum('n,rne->re', w, OK)] * 2, 1), 1e-12)
            C0 = cosmat(fp(np.ones(n)), ref)
            Wb = rng.multinomial(n, np.full(n, 1.0 / n), size=NBOOT).astype(np.float64)
            Cb = np.stack([cosmat(fp(w), ref) for w in Wb])
            T['fingerprint_columns'] = [f'chg|{e}' for e in edits] + [f'flow|{e}' for e in edits]
            T['fingerprint'] = {nm: fp(np.ones(n))[i].tolist() for i, nm in enumerate(names)}
            cl = [i for i, k in enumerate(kinds) if k == 'classical']; nets = [i for i, k in enumerate(kinds) if k == 'net']; perms = [i for i, k in enumerate(kinds) if k == 'perm']
            T['cosine'] = {}
            for i in nets:
                T['cosine'][names[i]] = {}
                for t in cl:
                    pc = C0[perms, t] if perms else np.array([np.nan]); dlt = C0[i, t] - np.nanmean(pc)
                    db = Cb[:, i, t] - np.nanmean(Cb[:, perms, t], 1) if perms else np.full(NBOOT, np.nan)
                    pb = float((1 + (db <= 0).sum()) / (NBOOT + 1))
                    # z test with both the jet-bootstrap spread of the network's cosine and the spread across permuted networks
                    sdb = float(np.std(Cb[:, i, t])); sdp = float(np.nanstd(pc, ddof=1)) if len(perms) > 1 else 0.0
                    z = dlt / max(math.hypot(sdb, sdp), 1e-12); pz = float(0.5 * math.erfc(z / math.sqrt(2)))
                    T['cosine'][names[i]][names[t]] = dict(cos=float(C0[i, t]), ci=np.percentile(Cb[:, i, t], [2.5, 97.5]).tolist(),
                                                           perm_mean=float(np.nanmean(pc)), perm_range=[float(np.nanmin(pc)), float(np.nanmax(pc))],
                                                           delta=float(dlt), delta_ci=np.percentile(db, [2.5, 97.5]).tolist(), p_boot=pb,
                                                           sd_boot=sdb, sd_perm=sdp, z=float(z), p_z=pz,
                                                           rank_above_perms=int((C0[i, t] > pc).sum()), n_perm=len(perms))
                    R['tests'].append(dict(ds=ds, task=task, model=names[i], tagger=names[t], p=pz, p_boot=pb, family='cosine'))
                if cl:
                    nb = np.argmax(Cb[:, i, cl], 1); best = int(np.argmax(C0[i, cl]))
                    T['cosine'][names[i]]['_nearest'] = dict(tagger=names[cl[best]], boot_freq=float((nb == best).mean()))
            # perm-to-tagger cosines (null distribution) and tagger-to-tagger
            T['cosine_perm'] = {names[i]: {names[t]: float(C0[i, t]) for t in cl} for i in perms}
            T['cosine_classical'] = {names[a]: {names[b]: float(C0[a, b]) for b in cl} for a in cl}
            D['tasks'][task] = T
            heat = dict(ds=ds, task=task, edits=edits, rows=[])
            for i, nm in enumerate(names):
                if kinds[i] == 'perm': continue
                heat['rows'].append((nm, kinds[i], [float(CH[i][:, j].sum() / max(OK[i][:, j].sum(), 1)) for j in range(len(edits))],
                                     [float(FL[i][:, j].sum() / max(OK[i][:, j].sum(), 1)) for j in range(len(edits))]))
            if perms:
                pm = np.median([[CH[i][:, j].sum() / max(OK[i][:, j].sum(), 1) for j in range(len(edits))] for i in perms], 0)
                heat['rows'].append(('permuted (median)', 'perm', pm.tolist(), [0.0] * len(edits)))
            ALLHEAT.append(heat)
            log(f'[{ds}] {task}: {len(nets)} nets, {len(perms)} permuted, {len(cl)} classical')
        R['datasets'][ds] = D
    # Holm per family
    for fam in ('cosine', 'flow', 'wta_vs_ctrl'):
        idx = [i for i, t in enumerate(R['tests']) if t.get('family') == fam]
        hp = holm([R['tests'][i]['p'] for i in idx])
        for i, h in zip(idx, hp): R['tests'][i]['p_holm'] = h
    R['heat'] = ALLHEAT
    (OUT / f'h55_fingerprint{sfx(smoke)}.json').write_text(json.dumps(R, indent=1, default=float))
    log('wrote fingerprint json')
    try: fig_heat(ALLHEAT, smoke)
    except Exception as e: import traceback; traceback.print_exc(); log('heat figure failed', repr(e))
    try: fig_edits(smoke)
    except Exception as e: import traceback; traceback.print_exc(); log('edit figure failed', repr(e))


LABELS = {'JEDI': 'JEDI-linear', 'dsets': 'Deep Sets', 'linf': 'Linformer', 'tf': 'Transformer', 'jedis': 'JEDI-style float',
          'V_tau21': 'm_SD + τ21<0.45', 'V_DDT': 'm_SD + τ21DDT<0.55', 'V_D2': 'm_SD + D2 (50 %)', 'mass_V': 'plain m 65-105',
          'top_tau32': 'm_SD + τ32<0.54', 'mass_top': 'plain m 105-210', 'LR_N': 'LR(N)', 'LR_NpTD': 'LR(N, pT^D)', 'LR_girth': 'LR(girth)'}
SHORT = dict(split='split\nhardest', ksplit3='split 3\nsoft', soft='soft\nwide', wta='WTA\naxis', ctrl='random\nshift',
             gin='soft kept\nby SD', gout='soft removed\nby SD', decay='decay-axis\nrotation', zboost='beam\nboost')


def fig_heat(H, smoke):
    import matplotlib.pyplot as plt
    show = [h for h in H if h['task'] in ('W_vs_QCD', 't_vs_QCD', 'q_vs_g', 'Top_vs_QCD', 'top_vs_QCD')]
    nr = sum(len(h['rows']) for h in show); cols = EDITS
    fig, ax = plt.subplots(figsize=(11, 0.28 * nr + 1.6)); M = np.full((nr, len(cols)), np.nan); ylab = []; seps = []; r = 0
    for h in show:
        for nm, kind, ch, fl in h['rows']:
            for j, e in enumerate(h['edits']): M[r, cols.index(e)] = 100 * ch[j]
            ylab.append(f'{h["ds"]} {h["task"]}: ' + ('[classical] ' if kind == 'classical' else '') + LABELS.get(nm, nm)); r += 1
        seps.append(r)
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('0.85')
    im = ax.imshow(np.sqrt(M), aspect='auto', cmap=cmap, vmin=0, vmax=np.sqrt(60))
    for i in range(nr):
        for j in range(len(cols)):
            if np.isfinite(M[i, j]): ax.text(j, i, f'{M[i, j]:.0f}' if M[i, j] >= 1 else (f'{M[i, j]:.1f}' if M[i, j] > 0 else '0'),
                                             ha='center', va='center', fontsize=6.5, color='w' if M[i, j] < 20 else 'k')
    for s in seps[:-1]: ax.axhline(s - 0.5, color='w', lw=2)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([SHORT[c] for c in cols], fontsize=8)
    ax.set_yticks(range(nr)); ax.set_yticklabels(ylab, fontsize=6.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.025); t = np.array([0, 1, 5, 10, 20, 40, 60]); cb.set_ticks(np.sqrt(t)); cb.set_ticklabels([str(v) for v in t])
    cb.set_label('% of calls changed (50 % signal-efficiency working point; classical: published cut)', fontsize=7)
    ax.set_title('Which edits change which tagger\'s calls: % of jets whose call flips (grey = edit not applied)', fontsize=9)
    fig.tight_layout(); p = OUT / f'h55_heatmap{sfx(smoke)}.png'; fig.savefig(p, dpi=150); log(f'wrote {p}')


def fig_edits(smoke):
    import matplotlib.pyplot as plt
    p = OUT / f'h55_jc_tab{sfx(smoke)}.npz'
    if not p.exists(): return
    Z = np.load(p); y = Z['y']; oks = {c: Z[f'{c}|ok'] for c in CONDS if f'{c}|ok' in Z.files}
    good = np.flatnonzero((y == 7) & np.all(np.stack([oks[c] for c in oks]), 0))
    j = int(good[0]) if len(good) else int(np.flatnonzero(y == 7)[0])
    O = np.load(OUT / f'h55_jc_obs{sfx(smoke)}.npz')
    fig, axs = plt.subplots(2, 5, figsize=(15, 6.4)); axs = axs.ravel()
    ae0, ap0 = Z['base|ax_eta'][j], Z['base|ax_phi'][j]; sg = 1.0 if ae0 >= 0 else -1.0
    for k, c in enumerate(CONDS):
        a = axs[k]; pt, eta, phi = Z[f'{c}|pt'][j], Z[f'{c}|eta'][j], Z[f'{c}|phi'][j]; r = pt > 0
        de, dp = sg * (eta[r] - ae0), wrap(phi[r] - ap0)
        if c == 'zboost': de = sg * (eta[r] - Z['zboost|ax_eta'][j])
        a.scatter(de, dp, s=4 + 400 * pt[r] / pt[r].max(), c='tab:blue', alpha=0.6, lw=0)
        if c != 'base':
            b = Z['base|pt'][j]; br = b > 0; bd = sg * (Z['base|eta'][j][br] - ae0); bp = wrap(Z['base|phi'][j][br] - ap0)
            a.scatter(bd, bp, s=6, facecolors='none', edgecolors='k', lw=0.4)
        a.plot([0], [0], 'k+', ms=10)
        if c in ('wta', 'ctrl'): a.plot([sg * (Z[f'{c}|ax_eta'][j] - ae0)], [wrap(Z[f'{c}|ax_phi'][j] - ap0)], 'rx', ms=10, mew=2)
        m0, m1 = O[f'base|m'][j], O[f'{c}|m'][j]; s0, s1 = O['base|msd'][j], O[f'{c}|msd'][j]; n1 = int(O[f'{c}|N'][j])
        a.set_title(f'{c}: {EDIT_WORDS.get(c, "unedited")[:48]}\nm {m1:.1f} (Δ{m1 - m0:+.2f}), m_SD {s1:.1f} (Δ{s1 - s0:+.2f}) GeV, N {n1}', fontsize=6.5)
        a.set_xlim(-0.8, 0.8); a.set_ylim(-0.8, 0.8); a.set_aspect('equal'); a.tick_params(labelsize=6)
        a.set_xlabel('Δη to the unedited axis', fontsize=7); a.set_ylabel('Δφ [rad]', fontsize=7)
    fig.suptitle(f'One JetClass W→qq jet (row {j}) under each edit. Blue: edited particles (area ∝ pT); black rings: unedited; '
                 '+ unedited axis; red x: axis used (wta / ctrl)', fontsize=9)
    fig.tight_layout(); q = OUT / f'h55_edits{sfx(smoke)}.png'; fig.savefig(q, dpi=140); log(f'wrote {q}')


# ================================================================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', required=True, choices=['edits', 'nets', 'stats'])
    ap.add_argument('--ds', default=None)
    ap.add_argument('--smoke', type=int, default=0)
    a = ap.parse_args()
    if a.part == 'edits': part_edits(a.ds, a.smoke)
    elif a.part == 'nets':
        dict(hls_jedi=nets_hls_jedi, hls_small=nets_hls_small, jc=nets_jc, top=nets_top, qg=nets_qg)[a.ds](a.smoke)
    else: part_stats(a.smoke)
    log('done')


if __name__ == '__main__':
    main()
