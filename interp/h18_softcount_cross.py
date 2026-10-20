"""H18 (loop D; claims D7 vs D10, open question b of round D3): is the different soft-particle count response of the quark-gluon
networks (H5v / H14: one added on-manifold soft particle moves the QG logit by -0.024..-0.002, vs LR(N) +0.09) and of the
JetClass / Top networks (H13v c4: W / top calls follow the number of added diffuse particles) a property of the edits or of the
models? The same edit family is applied to every dataset's networks, on their own jets, with count-aware and count-blind
classical references trained on the same dataset.

Samples (seed 4181, random rows; never first-k)
  qgpy  Komiske-Metodiev-Thaler Pythia test split, 5,000 quark + 5,000 gluon jets (interp.h14_qg_generator.load_sample('py'))
  qghw  Herwig 7.1 (Zenodo 3066475 files 18-19), 5,000 + 5,000 (load_sample('hw')); Pythia-trained networks
  jc    JetClass first test file per class: 3,334 random rows of QCD (ZJetsToNuNu), Wqq, Tbqq (the edit sample, 10,002 jets)
        and 1,000 of each other class (unedited 10-class accuracy only, reported as the class-balanced accuracy)
  top   Top Landscape test split, 5,000 top + 5,000 QCD (interp.data_top.sample)
  hls   hls4ml n64 test jets, 2,000 per class (h1_irc_panel.hls_sample)
Binary task scores s (log-odds toward the signal; Delta s > 0 = more signal-like)
  qg*  quark vs gluon: s = log p(q) - log p(g)                 (signal = quark; H5 / H14 logit with the sign flipped)
  jc   W vs QCD: s = logit_Wqq - logit_QCD (on QCD + Wqq jets); top vs QCD: s = logit_Tbqq - logit_QCD (on QCD + Tbqq jets)
  top  top vs QCD: the Top Landscape networks' top log-odds (as H13)
  hls  W vs rest: s = log p(W) - log(1 - p(W)) (all five classes)
Models
  qg   part_kinpid (ParT), part_kin (ParT_kin), pnet_kinpid (PN), pnet_kin (PN_kin), lorentznet (released), pfn_t (H5 torch
       PFN), efn_k (H2 energyflow EFN, IRC safe: zero control), efn_t (H5 torch EFN), part_kin_perm (permuted weights, null)
  jc   ParT_full, ParT_kin, PN_full, PN_kin (released), ParT_full_redrawn (null)
  top  ParT (kin), PN (kin), LNet (released LorentzNet), PFN, EFN (IRC safe), PFN_permuted (null)  (interp.h13_pileup_panel.TopModels)
  hls  JEDI-linear official n64 3-feature (--part model, jax env); Deep Sets dsets-d64-n64 seed0 and the H1 EFN (IRC safe)
       (--part hlss, conda env, read the saved hls jets)
References (--part refs, venv-ef; trained on the dataset's training split, 30,000 jets per class; hls: 12,000 per class of the
  'fit' split; jc: 20,000 random rows of the first train file of QCD, Wqq, Tbqq): LR_N = histogram log-likelihood ratio of the
  multiplicity the model sees (count only); EFPN = GBDT on log EFPs (d <= 5, beta 1, connected) + N; EFP = GBDT on the EFPs only
  (count blind, IRC safe). qghw additionally: LR_N_hw from the Herwig training files (0-17).
Edits (each on every jet; particles re-sorted and cut to the model's slot limit; inputs about the stored / unedited axis for
  jc, top, hls; the QG input code recomputes the E-scheme axis from the particles, as H5 / H14)
  e1        one soft particle on the data manifold (H5v generator): pT = the jet's softest pT, at a random existing particle's
            position + 0.05 in a random direction, that particle's type / track fields copied
  k1..k20   k = 1, 2, 5, 10, 20 soft particles drawn from the jet's own soft half: each copies a random particle of the softer
            half of the jet (pT, mass, type fields) and is placed at that particle's distance r from the axis at a random azimuth
  pu5/pu20  H13 diffuse disc: Poisson(rho A / <q>) particles, rho = 5 / 20 GeV per unit area, uniform in the disc of radius R
            (0.4 QG, 0.8 otherwise) about the axis, q from the sample's softest-quartile constituent spectrum, neutral hadrons
  c4_5/c4_20 the pu particles with their displacement from the axis divided by 20 (count and sum pT kept, mass change ~1/400)
  c5_5/c5_20 one neutral particle carrying the kept sum pT of pu and placed at the radius that gives the same kept-particle mass
  c0        count-fixed null: the (eta, phi) positions of the jet's softer-half particles randomly permuted among themselves
Metrics (--part stats): per model x task x edit, mean Delta s over jets with 95 % bootstrap interval (B = 1000), mean change of
  the multiplicity the model sees (Delta N_seen), response per added particle = mean Delta s / mean Delta N_seen; slope of mean
  Delta s in k (k1..k20); % calls changed, signal and background efficiency at the threshold giving eps_S = 0.5 on unedited
  jets (McNemar); sign-flip permutation p (P = 2000) for mean Delta s = 0, Holm over all networks x edits x tasks; paired
  bootstrap of slope_net - slope_EFP (H0: no counting), slope_net - slope_LR_N, slope_net - slope_EFPN; C2ST AUC per edit.
Run
  ./part_job.sh d-h18-qg  'python -m interp.h18_softcount_cross --part model --ds qg'
  ./part_job.sh d-h18-jc  'python -m interp.h18_softcount_cross --part model --ds jc'
  ./part_job.sh d-h18-top 'python -m interp.h18_softcount_cross --part model --ds top'
  ./job.sh d-h18-hls 1 2 'python -m interp.h18_softcount_cross --part model --ds hls'
  AI_PY=conda ./job.sh d-h18-hlss 1 2 'python -m interp.h18_softcount_cross --part hlss'
  ./job.sh d-h18-refs-<ds> 4 8 '/j-jepa-vol/aaron-interp/venv-ef/bin/python -m interp.h18_softcount_cross --part refs --ds <ds>'
  ./job.sh d-h18-stats 1 2 'python -m interp.h18_softcount_cross --part stats'
  -> $AI_RESULTS/physics/h18_<ds>_{model,jets,train,refs}.npz, h18_<ds>.json, h18_summary.json, h18_softcount.png
  --smoke N: N jets per class, files with suffix _smoke
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
from interp.h1_irc_panel import copyT, finalize, jet4, wrap, holm  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
OUT = RES / 'physics'
V = Path('/j-jepa-vol/aaron-interp')
SEED = 4181
KS = (1, 2, 5, 10, 20)
RHOS = (5, 20)
CONDS = ['base', 'e1'] + [f'k{k}' for k in KS] + [f'pu{r}' for r in RHOS] + [f'c4_{r}' for r in RHOS] + [f'c5_{r}' for r in RHOS] + ['c0']
EDITS = CONDS[1:]
DSETS = ('qgpy', 'qghw', 'jc', 'top', 'hls')
TASKS = {'qgpy': [('q_vs_g', [1], [0])], 'qghw': [('q_vs_g', [1], [0])], 'jc': [('W_vs_QCD', [7], [0]), ('top_vs_QCD', [8], [0])],
         'top': [('t_vs_QCD', [1], [0])], 'hls': [('W_vs_rest', [2], [0, 1, 3, 4])]}
CAP = dict(qgpy=200, qghw=200, jc=128, top=200, hls=64)
NSEEN_CAP = dict(qgpy=128, qghw=128, jc=128, top=128, hls=64)     # slots of the count-sensitive input (weaver 128, JEDI 64)
RDISC = dict(qgpy=0.4, qghw=0.4, jc=0.8, top=0.8, hls=0.8)
NETS = dict(qgpy=['part_kinpid', 'part_kin', 'pnet_kinpid', 'pnet_kin', 'lorentznet', 'pfn_t', 'efn_k', 'efn_t', 'part_kin_perm'],
            jc=['ParT_full', 'ParT_kin', 'PN_full', 'PN_kin', 'ParT_full_redrawn'],
            top=['ParT', 'PN', 'LNet', 'PFN', 'EFN', 'PFN_permuted'], hls=['JEDI', 'dsets', 'efn_hls'])
NETS['qghw'] = NETS['qgpy']
NULLS = {'part_kin_perm', 'ParT_full_redrawn', 'PFN_permuted'}
IRC_SAFE = {'efn_k', 'efn_t', 'EFN', 'efn_hls'}
LABEL = {'part_kinpid': 'ParT', 'part_kin': 'ParT_kin', 'pnet_kinpid': 'PN', 'pnet_kin': 'PN_kin', 'lorentznet': 'LorentzNet',
         'pfn_t': 'PFN', 'efn_k': 'EFN', 'efn_t': 'EFN (torch)', 'part_kin_perm': 'ParT_kin permuted', 'ParT_full_redrawn': 'ParT_full redrawn',
         'LNet': 'LorentzNet', 'PFN_permuted': 'PFN permuted', 'JEDI': 'JEDI-linear', 'dsets': 'Deep Sets', 'efn_hls': 'EFN'}
KNOWN = dict(qgpy=dict(part_kin='AUC 0.9013', part_kinpid='AUC 0.9123', pnet_kin='AUC 0.9015', pnet_kinpid='AUC 0.906 (H5v)',
                       lorentznet='AUC 0.917 (H5v)', pfn_t='AUC 0.883 (H5v)', efn_k='AUC 0.8835 (H14), 0.8831 (H2 test)'),
             qghw=dict(part_kin='AUC 0.8079 (H14)', part_kinpid='AUC 0.8049 (H14)', pnet_kin='AUC 0.8080 (H14)', efn_k='AUC 0.8023 (H14)'),
             jc=dict(ParT_full='10-class acc 0.859 (D4), 0.856-0.863', ParT_kin='0.745 (D4), 0.736-0.742', PN_full='0.839-0.846',
                     PN_kin='0.695-0.704'),
             top=dict(ParT='AUC 0.9858', PN='AUC 0.9858', LNet='AUC 0.9870', PFN='AUC 0.9819', EFN='AUC ~0.98'),
             hls=dict(JEDI='5-class acc 0.8092 (260k test); 0.808 (H13)'))
SENT = 50.0
T0_ = time.time()


def log(*a):
    print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


def sfx(smoke):
    return '_smoke' if smoke else ''


# ======================================================================================================================= edits
def nreal(T):
    return (T['pt'] > 0).sum(1)


def eaxis(T):
    from interp.h1_irc_panel import p3, from3
    real = T['pt'] > 0; px, py, pz = p3(T['pt'], T['eta'], T['phi'])
    _, e, p = from3((px * real).sum(1), (py * real).sum(1), (pz * real).sum(1))
    return e, p


def neutral(U, W, on):
    """the appended columns W .. W + K - 1 (on: (J, K) bool) become neutral hadrons in the dataset's type fields"""
    K = on.shape[1]; e = U['ext']
    for k in e: e[k][:, W:W + K] = 0.0
    if 'isNeutralHadron' in e: e['isNeutralHadron'][:, W:W + K] = on
    if 'mask' in e: e['mask'][:, W:W + K] = on
    if 'pid' in e: e['pid'][:, W:W + K] = np.where(on, 130.0, 0.0)


def copy_fields(U, col, T, src):
    rows = np.arange(len(src))
    U['m'][:, col] = T['m'][rows, src]
    for k in U['ext']: U['ext'][k][:, col] = T['ext'][k][rows, src]


def e_onman(T, rng):
    """H5v add_soft: softest pT, at a random particle's position + 0.05 in a random direction, that particle's fields"""
    n = nreal(T); J = len(n); rows = np.arange(J)
    i = np.minimum((rng.random(J) * n).astype(int), n - 1); th = 2 * np.pi * rng.random(J)
    U = copyT(T, 1); c = U['pt'].shape[1] - 1
    U['pt'][:, c] = T['pt'][rows, n - 1]
    U['eta'][:, c] = T['eta'][rows, i] + 0.05 * np.cos(th); U['phi'][:, c] = wrap(T['phi'][rows, i] + 0.05 * np.sin(th))
    copy_fields(U, c, T, i)
    return U


def e_ksoft(T, k, rng):
    """k particles copied from random members of the jet's softer half (pT, mass, type fields), each at its donor's distance
    from the axis and a random azimuth"""
    n = nreal(T); J = len(n); rows = np.arange(J); h = np.maximum(n // 2, 1)
    U = copyT(T, k); W = T['pt'].shape[1]
    for j in range(k):
        d = n - h + np.minimum((rng.random(J) * h).astype(int), h - 1); th = 2 * np.pi * rng.random(J)
        r = np.hypot(T['eta'][rows, d] - T['ax_eta'], wrap(T['phi'][rows, d] - T['ax_phi']))
        U['pt'][:, W + j] = T['pt'][rows, d]
        U['eta'][:, W + j] = T['ax_eta'] + r * np.cos(th); U['phi'][:, W + j] = wrap(T['ax_phi'] + r * np.sin(th))
        copy_fields(U, W + j, T, d)
    return U


def spectrum(T):
    pts = T['pt'][T['pt'] > 0]; q25 = float(np.percentile(pts, 25)); pool = pts[pts < q25]
    return dict(q25=q25, pool=pool, mean=float(pool.mean()), n_pool=int(len(pool)))


def e_disc(T, rho, R, spec, rng):
    """H13 injection with disc radius R (neutral hadrons, massless)"""
    J, W = T['pt'].shape; A = math.pi * R * R
    n = rng.poisson(rho * A / spec['mean'], J); K = max(int(n.max()), 1)
    U = copyT(T, K); on = np.arange(K)[None, :] < n[:, None]
    q = rng.choice(spec['pool'], (J, K)); r = R * np.sqrt(rng.random((J, K))); th = 2 * np.pi * rng.random((J, K))
    U['pt'][:, W:] = np.where(on, q, 0.0)
    U['eta'][:, W:] = np.where(on, T['ax_eta'][:, None] + r * np.cos(th), 0.0)
    U['phi'][:, W:] = np.where(on, wrap(T['ax_phi'][:, None] + r * np.sin(th)), 0.0)
    U['m'][:, W:] = 0.0; neutral(U, W, on)
    return U, W, n


def core(U, T0, W, f=20.0):
    Vv = copyT(U); on = Vv['pt'][:, W:] > 0
    de = Vv['eta'][:, W:] - T0['ax_eta'][:, None]; dp = wrap(Vv['phi'][:, W:] - T0['ax_phi'][:, None])
    Vv['eta'][:, W:] = np.where(on, T0['ax_eta'][:, None] + de / f, 0.0)
    Vv['phi'][:, W:] = np.where(on, wrap(T0['ax_phi'][:, None] + dp / f), 0.0)
    return Vv


def mass_all(T):
    p4 = jet4(T); return np.sqrt(np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1), 0))


def one_particle(T0, Q, r, th):
    U = copyT(T0, 1); c = U['pt'].shape[1] - 1; ok = Q > 0
    U['pt'][:, c] = np.where(ok, Q, 0.0); U['m'][:, c] = 0.0
    U['eta'][:, c] = np.where(ok, T0['ax_eta'] + r * np.cos(th), 0.0); U['phi'][:, c] = np.where(ok, wrap(T0['ax_phi'] + r * np.sin(th)), 0.0)
    neutral(U, c, ok[:, None])
    return U


def match_radius(T0, Q, th, target, cap, rmax, it=50):
    """(H13v c5) per-jet radius of one added particle of pT Q such that the kept-particle mass equals target (bisection)"""
    U = copyT(T0, 1); c = U['pt'].shape[1] - 1; ok = Q > 0
    U['pt'][:, c] = np.where(ok, Q, 0.0); U['eta'][:, c] = np.where(ok, SENT, 0.0); U['m'][:, c] = 0.0
    Uf, _ = finalize(U, cap); col = np.argmax(Uf['eta'] == SENT, 1); has = (Uf['eta'] == SENT).any(1)
    rows = np.arange(len(Q)); Uf['pt'][rows, col] = np.where(has, 0.0, Uf['pt'][rows, col])
    P = jet4(Uf)

    def m_of(r):
        e = T0['ax_eta'] + r * np.cos(th); p = T0['ax_phi'] + r * np.sin(th); q = np.where(ok & has, Q, 0.0)
        v = P + np.stack([q * np.cos(p), q * np.sin(p), q * np.sinh(e), q * np.cosh(e)], 1)
        return np.sqrt(np.maximum(v[:, 3] ** 2 - (v[:, :3] ** 2).sum(1), 0))
    lo, hi = np.zeros(len(Q)), np.full(len(Q), rmax)
    for _ in range(it):
        mid = 0.5 * (lo + hi); up = m_of(mid) < target; lo = np.where(up, mid, lo); hi = np.where(up, hi, mid)
    r = 0.5 * (lo + hi)
    return r, dict(abs_dm_median=float(np.median(np.abs(m_of(r) - target)[ok])) if ok.any() else None,
                   at_rmax=float((r > rmax * 0.999)[ok].mean()) if ok.any() else None, r_median=float(np.median(r[ok])) if ok.any() else None)


def c0_perm(T, rng):
    U = copyT(T); n = nreal(T)
    for j in range(len(n)):
        h = n[j] // 2
        if h < 2: continue
        idx = np.arange(n[j] - h, n[j]); p = rng.permutation(idx)
        U['eta'][j, idx] = T['eta'][j, p]; U['phi'][j, idx] = T['phi'][j, p]
    return U


def conditions(T0, ds, cap):
    """yields (name, unfinalized table, info); T0 sorted and capped"""
    spec = spectrum(T0); R = RDISC[ds]
    log(f'{ds}: soft spectrum q25 {spec["q25"]:.3f} GeV, mean of pool {spec["mean"]:.3f} GeV ({spec["n_pool"]} particles); disc R {R}')
    yield 'base', T0, {}
    yield 'e1', e_onman(T0, np.random.default_rng([SEED, 1])), {}
    for k in KS: yield f'k{k}', e_ksoft(T0, k, np.random.default_rng([SEED, 2, k])), {}
    Uf0 = finalize(T0, cap)[0]
    for rho in RHOS:
        U, W, n = e_disc(T0, rho, R, spec, np.random.default_rng([SEED, 3, rho]))
        Uf = finalize(U, cap)[0]
        yield f'pu{rho}', U, dict(n_injected_mean=float(n.mean()), injected_sumpt_mean=float((U['pt'][:, W:]).sum(1).mean()))
        yield f'c4_{rho}', core(U, T0, W), {}
        Q = np.maximum(Uf['pt'].sum(1) - Uf0['pt'].sum(1), 0.0); th = 2 * np.pi * np.random.default_rng([SEED, 4, rho]).random(len(Q))
        r, inf = match_radius(T0, Q, th, mass_all(Uf), cap, rmax=2.5 * R)
        yield f'c5_{rho}', one_particle(T0, Q, r, th), dict(radius_match=inf, added_pt_median=float(np.median(Q)))
    yield 'c0', c0_perm(T0, np.random.default_rng([SEED, 5])), {}


def ragged(Uf, ae, ap):
    real = Uf['pt'] > 0; n = real.sum(1)
    de = Uf['eta'] - ae[:, None]; dp = wrap(Uf['phi'] - ap[:, None])
    return np.r_[0, np.cumsum(n)].astype(np.int64), np.stack([Uf['pt'][real], de[real], dp[real]], 1).astype(np.float32)


def unragged(off, p, width=None):
    n = np.diff(off); J = len(n); L = int(width or n.max()); x = np.zeros((J, L, 3), np.float32)
    rows = np.repeat(np.arange(J), n); pos = np.arange(len(p)) - np.repeat(off[:-1], n); k = pos < L
    x[rows[k], pos[k]] = p[k]
    return x


# ======================================================================================================================= metrics
def auc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def task_rows(ds, y, task):
    _, sig, bkg = task
    rows = np.isin(y, sig + bkg); return rows, np.isin(y, sig)


def logsm(L):
    L = np.asarray(L, np.float64); m = L.max(1, keepdims=True); return L - m - np.log(np.exp(L - m).sum(1, keepdims=True))


# ======================================================================================================================= samples
def jc_sample(counts, seed):
    import uproot
    from interp.part_core import read_root, fof, cat, sub, CL
    rng = np.random.default_rng(seed); Js = []
    for c in CL:
        f = fof(c); n = uproot.open(f)['tree'].num_entries; idx = np.sort(rng.choice(n, counts[c], replace=False)); parts = []
        for a in range(0, n, 20000):
            loc = idx[(idx >= a) & (idx < a + 20000)] - a
            if len(loc): parts.append(sub(read_root(f, start=a, stop=min(a + 20000, n)), loc))
        Js.append(cat(parts)); log(f'JetClass {c}: {counts[c]} random rows of {n} ({Path(f).name})')
    return cat(Js)


def jc_train(per, seed):
    """random rows of the first train file of QCD, Wqq, Tbqq -> ragged (pT, deta, dphi), label (class index)"""
    import uproot
    from interp.part_core import read_root, kin, sub, cat, PRE, CL
    rng = np.random.default_rng(seed); out = []
    for c in ('QCD', 'Wqq', 'Tbqq'):
        f = Path('/j-jepa-vol/JetClass/Pythia/train_100M') / f'{PRE[c]}_000.root'; n = uproot.open(f)['tree'].num_entries
        idx = np.sort(rng.choice(n, per, replace=False)); parts = []
        for a in range(0, n, 20000):
            loc = idx[(idx >= a) & (idx < a + 20000)] - a
            if len(loc): parts.append(sub(read_root(f, start=a, stop=min(a + 20000, n)), loc))
        J = cat(parts); out.append(J); log(f'JetClass train {c}: {per} random rows of {n} ({f.name})')
    J = cat(out); pt, de, dp, _ = (np.asarray(a, np.float64) for a in kin(J))
    T = dict(pt=pt, eta=de, phi=dp); return ragged_simple(T), J['label'].astype(int)


def ragged_simple(T):
    real = T['pt'] > 0; n = real.sum(1)
    return np.r_[0, np.cumsum(n)].astype(np.int64), np.stack([T['pt'][real], T['eta'][real], T['phi'][real]], 1).astype(np.float32)


def qg_table(L0):
    """list of (n, 4) [pT, eta, phi, pid] -> h1-style table (massless), axis = E-scheme axis of all particles"""
    n = np.array([len(a) for a in L0]); W = int(n.max()); J = len(L0)
    T = {k: np.zeros((J, W)) for k in ('pt', 'eta', 'phi', 'm')}; pid = np.zeros((J, W))
    for j, a in enumerate(L0):
        a = a[np.argsort(-a[:, 0], kind='stable')]
        T['pt'][j, :len(a)], T['eta'][j, :len(a)], T['phi'][j, :len(a)], pid[j, :len(a)] = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    T['ext'] = dict(pid=pid); T['ax_eta'], T['ax_phi'] = eaxis(T); T['sgn'] = np.where(T['ax_eta'] < 0, -1.0, 1.0)
    return T


# ======================================================================================================================= model part
def run_edits(ds, T0, y, scorer, cap, smoke, extra_info):
    """all conditions -> per model x task scores, N_seen, ragged jets; files h18_<ds>_model.npz, _jets.npz, _model.json"""
    ae, ap = T0['ax_eta'], T0['ax_phi']; NPZ = dict(y=y); JETS = {}; info = dict(conditions={}, **extra_info)
    n0 = None; m0 = None; s0 = None
    for name, U, inf in conditions(T0, ds, cap):
        t = time.time(); Uf, over = finalize(U, cap)
        ns = np.minimum((Uf['pt'] > 0).sum(1), NSEEN_CAP[ds]); NPZ[f'nseen|{name}'] = ns.astype(np.int16)
        S = scorer(Uf, ae, ap)
        for mn, d in S.items():
            for tn, s in d.items(): NPZ[f's|{mn}|{tn}|{name}'] = np.asarray(s, np.float32)
        off, p = ragged(Uf, ae, ap); JETS[f'off|{name}'] = off; JETS[f'p|{name}'] = p
        m = mass_all(Uf); sp = Uf['pt'].sum(1)
        if name == 'base': n0, m0, s0 = ns, m, sp
        ax2 = eaxis(Uf); off_mrad = 1e3 * np.hypot(ax2[0] - ae, wrap(ax2[1] - ap))
        inf = dict(inf, dN_seen_mean=float((ns - n0).mean()), dm_mean=float((m - m0).mean()), dm_median=float(np.median(m - m0)),
                   dsumpt_mean=float((sp - s0).mean()), overflow_frac=float(over.mean()),
                   escheme_axis_offset_mrad_median=float(np.median(off_mrad)))
        info['conditions'][name] = inf
        tn0 = TASKS[ds][0][0]
        log(f'{ds} {name}: dN_seen {inf["dN_seen_mean"]:+.2f} dm {inf["dm_mean"]:+.2f} GeV dsumpt {inf["dsumpt_mean"]:+.2f} GeV '
            f'axis offset {inf["escheme_axis_offset_mrad_median"]:.2f} mrad | ' +
            ' '.join(f'{mn} {np.mean(NPZ[f"s|{mn}|{tn0}|{name}"] - NPZ[f"s|{mn}|{tn0}|base"]):+.4f}' for mn in S) + f' ({time.time() - t:.0f} s)')
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f'h18_{ds}_model{sfx(smoke)}.npz', **NPZ)
    np.savez_compressed(OUT / f'h18_{ds}_jets{sfx(smoke)}.npz', **JETS)
    # unedited performance on the sample
    un = {}
    for mn in {k.split('|')[1] for k in NPZ if k.startswith('s|')}:
        un[mn] = {}
        for task in TASKS[ds]:
            rows, sig = task_rows(ds, y, task); s = NPZ[f's|{mn}|{task[0]}|base'][rows].astype(np.float64); ys = sig[rows].astype(int)
            un[mn][task[0]] = dict(AUC=auc(s, ys), acc_s_gt_0=float(((s > 0).astype(int) == ys).mean()), n=int(rows.sum()))
    info['unedited'] = dict(un, **info.get('unedited', {}))
    for mn, v in info['unedited'].items(): log(f'{ds} unedited {mn}: {v}  known: {KNOWN.get(ds, {}).get(mn, "-")}')
    (OUT / f'h18_{ds}_model{sfx(smoke)}.json').write_text(json.dumps(info, indent=1, default=float))
    log(f'wrote h18_{ds}_model{sfx(smoke)}.npz / .json and h18_{ds}_jets{sfx(smoke)}.npz')


def part_qg(smoke):
    import torch
    from interp import h5_qg_counting as H
    from interp import h14_qg_generator as G
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    M = H.load_models(dev); G.add_keras_ef(M, 'efn'); names = [n for n in NETS['qgpy'] if n in M.m]
    log('QG models', names, 'device', dev)

    def scorer(Uf, ae, ap):
        Jt = dict(pt=Uf['pt'], eta=Uf['eta'], phi=Uf['phi'], pid=Uf['ext']['pid'].astype(np.int64), n=(Uf['pt'] > 0).sum(1))
        return {nm: {'q_vs_g': -M.logit(nm, Jt)} for nm in names}
    for gen in ('py', 'hw'):
        L0, y, inf = G.load_sample(gen, smoke or 5000, SEED); y = np.asarray(y).astype(int)
        T0 = finalize(qg_table(L0), CAP['qgpy'])[0]
        log(f'QG {gen}: {len(y)} jets, quark frac {y.mean():.3f}, <N> q {nreal(T0)[y == 1].mean():.2f} g {nreal(T0)[y == 0].mean():.2f}')
        run_edits('qg' + gen, T0, y, scorer, CAP['qgpy'], smoke, dict(sample=f'{inf["source"]}, {smoke or 5000} per class, seed {SEED}',
                                                                     idx=np.asarray(inf['idx']).tolist(), models=M.info))
    # training dump for the references: random non-val Pythia train rows, 128 hardest (pT, deta, dphi)
    rng = np.random.default_rng(SEED + 2); d = H.QG / 'train'; yt = np.load(d / 'label.npy'); isv = np.load(d / 'is_val.npy')
    per = 2000 if smoke else 30000
    ix = np.sort(np.concatenate([rng.choice(np.flatnonzero((yt == c) & ~isv), per, replace=False) for c in (0, 1)]))
    k3 = np.asarray(np.load(d / 'kin3.npy', mmap_mode='r')[ix], np.float64)
    off, p = ragged_simple(dict(pt=k3[..., 0], eta=k3[..., 1], phi=k3[..., 2]))
    np.savez_compressed(OUT / f'h18_qgpy_train{sfx(smoke)}.npz', off=off, p=p, y=yt[ix].astype(int), idx=ix)
    log(f'wrote QG train dump ({len(ix)} jets)')


def part_jc(smoke):
    from interp.part_core import ParTNetwork, CL, inputs, sub
    from interp.p1v_verify_part import fwd
    from interp.pn_net import load_particlenet
    from interp.h1_irc_panel import pn_logits, jc_table
    from interp.h13_pileup_panel import jc_maker
    from jetdistill.config import MODELS
    dev = os.environ.get('JETDISTILL_DEVICE', 'cpu')
    big = smoke or 3334; small = max((smoke or 3334) * 3 // 10, 10) if smoke else 1000
    counts = {c: (big if c in ('QCD', 'Wqq', 'Tbqq') else small) for c in CL}
    Jall = jc_sample(counts, SEED); yall = Jall['label'].astype(int)
    nets = {}
    for n in ('full', 'kin'):
        net = ParTNetwork(n); nets[f'ParT_{n}'] = (lambda J, net=net, n=n: np.asarray(fwd(net, *inputs(J, n))[1], np.float64))
    net = ParTNetwork('full', untrained=True, seed=0); nets['ParT_full_redrawn'] = (lambda J, net=net: np.asarray(fwd(net, *inputs(J, 'full'))[1], np.float64))
    for n, dims in (('full', 17), ('kin', 7)):
        pn = load_particlenet(Path(MODELS).parent / 'particlenet' / f'ParticleNet_{n}.pt', dims, dev)
        nets[f'PN_{n}'] = (lambda J, pn=pn, n=n: pn_logits(pn, J, n))
    # unedited 10-class accuracy (class-balanced) on the whole sample
    Tall = jc_table(Jall); mk = jc_maker(Jall); Jb = mk(finalize(Tall, 128)[0], Tall['ax_eta'], Tall['ax_phi'])
    acc10 = {}
    for mn, f in nets.items():
        c = f(Jb).argmax(1); acc10[mn] = dict(balanced_acc_10class=float(np.mean([(c[yall == k] == k).mean() for k in range(len(CL))])),
                                             per_class={CL[k]: float((c[yall == k] == k).mean()) for k in range(len(CL))})
        log(f'jc {mn}: 10-class balanced accuracy {acc10[mn]["balanced_acc_10class"]:.4f}')
    rows = np.flatnonzero(np.isin(yall, [0, 7, 8])); J0 = sub(Jall, rows); y = yall[rows]
    T0 = finalize(jc_table(J0), 128)[0]; make = jc_maker(J0)

    def scorer(Uf, ae, ap):
        J = make(Uf, ae, ap); out = {}
        for mn, f in nets.items():
            L = f(J); out[mn] = {'W_vs_QCD': L[:, 7] - L[:, 0], 'top_vs_QCD': L[:, 8] - L[:, 0]}
        return out
    run_edits('jc', T0, y, scorer, 128, smoke, dict(sample=f'JetClass first test file per class; QCD/Wqq/Tbqq {big} random rows each '
                                                             f'(edit sample), other classes {small} (accuracy only); seed {SEED}',
                                                      unedited={f'{k}__10class': v for k, v in acc10.items()}))
    (off, p), yt = jc_train(1500 if smoke else 20000, SEED + 2)
    np.savez_compressed(OUT / f'h18_jc_train{sfx(smoke)}.npz', off=off, p=p, y=yt)
    log(f'wrote JetClass train dump ({len(yt)} jets)')


def part_top(smoke):
    import torch
    from interp import data_top as dt
    from interp.h13_pileup_panel import TopModels, top_J, top_table
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    S = dt.sample(smoke or 5000, SEED); y = S['label'].astype(int); T0 = finalize(top_table(S['full']), 200)[0]
    Mo = TopModels(dev); order = np.random.default_rng(SEED + 7).permutation(len(y))
    log(f'Top: {len(y)} jets, <N> top {nreal(T0)[y == 1].mean():.1f} QCD {nreal(T0)[y == 0].mean():.1f}')

    def scorer(Uf, ae, ap):
        sc = Mo.score(top_J(Uf, ae, ap), order); return {k: {'t_vs_QCD': v} for k, v in sc.items()}
    run_edits('top', T0, y, scorer, 200, smoke, dict(sample=f'Top Landscape test, {smoke or 5000} per class, seed {SEED}', idx=S['idx'].tolist()))
    rng = np.random.default_rng(SEED + 2); d = dt.DST / 'train'; yt = np.load(d / 'label.npy'); per = 2000 if smoke else 30000
    ix = np.sort(np.concatenate([rng.choice(np.flatnonzero(yt == c), per, replace=False) for c in (0, 1)]))
    k3 = np.asarray(np.load(d / 'kin3.npy', mmap_mode='r')[ix], np.float64)
    off, p = ragged_simple(dict(pt=k3[..., 0], eta=k3[..., 1], phi=k3[..., 2]))
    np.savez_compressed(OUT / f'h18_top_train{sfx(smoke)}.npz', off=off, p=p, y=yt[ix].astype(int), idx=ix)
    log(f'wrote Top train dump ({len(ix)} jets)')


def hls_W(L):
    ls = logsm(L); pW = np.exp(ls[:, 2]); return ls[:, 2] - np.log(np.maximum(1 - pW, 1e-300))


def part_hls(smoke):
    from interp.jedi_core import Jedi
    from interp.h1_irc_panel import hls_sample
    from interp.h13_pileup_panel import hls_in
    T0, y = hls_sample(smoke or 2000, SEED); T0 = finalize(T0, 64)[0]
    jedi = Jedi(64, '3-feature'); acc = {}
    L = np.asarray(jedi.run(hls_in(T0, T0['ax_eta'], T0['ax_phi']))['logits']); acc['JEDI__5class'] = dict(acc_5class=float((L.argmax(1) == y).mean()))
    log(f'hls JEDI 5-class accuracy {acc["JEDI__5class"]["acc_5class"]:.4f}; at 64 particles: {(nreal(T0) >= 64).mean():.3f} of jets')

    def scorer(Uf, ae, ap):
        return {'JEDI': {'W_vs_rest': hls_W(jedi.run(hls_in(Uf, ae, ap))['logits'])}}
    run_edits('hls', T0, y, scorer, 64, smoke, dict(sample=f'hls4ml n64 test (jedi_core.sample indices), {smoke or 2000} per class, seed {SEED}',
                                                   unedited=acc, frac_at_64=float((nreal(T0) >= 64).mean())))
    from jetdistill.data import particles, labels
    yt = labels('fit'); rng = np.random.default_rng(SEED + 2); per = 1500 if smoke else 12000
    ix = np.sort(np.concatenate([rng.choice(np.flatnonzero(yt == c), per, replace=False) for c in range(5)]))
    X = particles('fit', 64).astype(np.float64)[ix]
    off, p = ragged_simple(dict(pt=X[..., 0], eta=X[..., 1], phi=X[..., 2]))
    np.savez_compressed(OUT / f'h18_hls_train{sfx(smoke)}.npz', off=off, p=p, y=yt[ix].astype(int), idx=ix)
    log(f'wrote hls4ml train dump ({len(ix)} jets)')


def part_hlss(smoke):
    """Deep Sets and the H1 EFN (conda torch) on the jets the JEDI job saved"""
    import torch
    from interp.h1_irc_panel import load_small, torch_logits, efn_class
    torch.set_num_threads(1)
    SM = Path(os.environ.get('AI_SMALL_MODELS', '/j-jepa-vol/aaron-interp/models/small'))
    net, sh, sc = load_small(SM / 'dsets-d64-n64' / 'seed0')
    efn = efn_class()(); efn.load_state_dict(torch.load(OUT / 'h1_efn' / 'best.pt', map_location='cpu')); efn.eval()
    f = dict(dsets=lambda x: torch_logits(net, x, sh, sc), efn_hls=lambda x: torch_logits(efn, x, np.zeros(3, np.float32), np.ones(3, np.float32)))
    Jz = np.load(OUT / f'h18_hls_jets{sfx(smoke)}.npz'); Z = np.load(OUT / f'h18_hls_model{sfx(smoke)}.npz'); y = Z['y']
    NPZ = dict(y=y); info = dict(unedited={}, known=dict(dsets=json.loads((SM / 'dsets-d64-n64' / 'seed0' / 'test.json').read_text())['accuracy'],
                                                         efn_hls=json.loads((OUT / 'h1_efn' / 'info.json').read_text())['test_acc']))
    for c in CONDS:
        x = unragged(Jz[f'off|{c}'], Jz[f'p|{c}'], 64)
        for mn, fn in f.items():
            L = fn(x); NPZ[f's|{mn}|W_vs_rest|{c}'] = hls_W(L).astype(np.float32)
            if c == 'base':
                rows = np.ones(len(y), bool); s = NPZ[f's|{mn}|W_vs_rest|base']
                info['unedited'][mn] = dict(W_vs_rest=dict(AUC=auc(s, (y == 2).astype(int)), acc_s_gt_0=float(((s > 0) == (y == 2)).mean())))
                info['unedited'][f'{mn}__5class'] = dict(acc_5class=float((L.argmax(1) == y).mean()))
        log(f'hlss {c}: ' + ' '.join(f'{mn} {np.mean(NPZ[f"s|{mn}|W_vs_rest|{c}"] - NPZ[f"s|{mn}|W_vs_rest|base"]):+.4f}' for mn in f))
    for k, v in info['unedited'].items(): log('hlss unedited', k, v, 'known', info['known'].get(k.split('__')[0]))
    np.savez_compressed(OUT / f'h18_hls_small{sfx(smoke)}.npz', **NPZ)
    (OUT / f'h18_hls_small{sfx(smoke)}.json').write_text(json.dumps(info, indent=1))
    log('wrote h18_hls_small')


# ======================================================================================================================= references
def part_refs(ds, smoke, ncpu):
    from multiprocessing import Pool
    from sklearn.ensemble import HistGradientBoostingClassifier
    from interp import h14_qg_generator as G
    from interp import h5_qg_counting as H
    tds = 'qgpy' if ds.startswith('qg') else ds
    Tr = np.load(OUT / f'h18_{tds}_train{sfx(smoke)}.npz'); Jz = np.load(OUT / f'h18_{ds}_jets{sfx(smoke)}.npz')
    Z = np.load(OUT / f'h18_{ds}_model{sfx(smoke)}.npz'); y = Z['y']
    cap = NSEEN_CAP[ds]
    tolist = lambda off, p: [p[off[i]:off[i + 1]].astype(np.float64) for i in range(len(off) - 1)]
    pool = Pool(ncpu); t = time.time()
    Lt = tolist(Tr['off'], Tr['p']); yt = Tr['y']; Nt = np.minimum(np.diff(Tr['off']), cap).astype(float)
    Ft = G.efps([a[:cap] for a in Lt], pool); log(f'{ds}: train EFPs {Ft.shape} ({time.time() - t:.0f} s)')
    feat = lambda F, N, wn: np.c_[np.log(np.maximum(F, 1e-12)), N] if wn else np.log(np.maximum(F, 1e-12))
    refs, info = {}, dict(train=dict(n=int(len(yt)), per_class={int(c): int((yt == c).sum()) for c in np.unique(yt)}), lr_n={})
    for tn, sig, bkg in TASKS[ds]:
        rows = np.isin(yt, sig + bkg); g = np.isin(yt, sig)[rows].astype(int)
        for wn in (True, False):
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True, validation_fraction=0.15,
                                               random_state=0).fit(feat(Ft[rows], Nt[rows], wn), g)
            refs[('EFPN' if wn else 'EFP', tn)] = ('gbdt', m, wn)
        Ns = Nt[rows].astype(int); M = cap + 1
        hs = np.bincount(Ns[g == 1], minlength=M)[:M] + 0.5; hb = np.bincount(Ns[g == 0], minlength=M)[:M] + 0.5
        tab = np.log(hs / hs.sum()) - np.log(hb / hb.sum()); refs[('LR_N', tn)] = ('lrn', tab, None)
        info['lr_n'][tn] = dict(mean_N_sig=float(Ns[g == 1].mean()), mean_N_bkg=float(Ns[g == 0].mean()))
        if ds == 'qghw':
            m = np.load(G.HW / 'meta.npz'); tr = m['fid'] < 18; n = np.minimum(m['N'][tr].astype(int), cap); lab = m['label'][tr].astype(int)
            hs = np.bincount(n[lab == 1], minlength=M)[:M] + 0.5; hb = np.bincount(n[lab == 0], minlength=M)[:M] + 0.5
            refs[('LR_N_hw', tn)] = ('lrn', np.log(hs / hs.sum()) - np.log(hb / hb.sum()), None)
            info['lr_n']['herwig_train'] = dict(mean_N_q=float(n[lab == 1].mean()), mean_N_g=float(n[lab == 0].mean()))
        log(f'{ds} {tn}: references trained; LR(N) means {info["lr_n"][tn]}')
    OUTZ = dict(y=y); c2 = {}; sub = np.sort(np.random.default_rng(SEED + 9).choice(len(y), min(4000, len(y)), replace=False))
    L0 = None
    for c in CONDS:
        t = time.time(); L = tolist(Jz[f'off|{c}'], Jz[f'p|{c}']); N = np.array([min(len(a), cap) for a in L], float)
        F = G.efps([a[:cap] for a in L], pool)
        for (rn, tn), (kind, m, wn) in refs.items():
            OUTZ[f's|{rn}|{tn}|{c}'] = (m.decision_function(feat(F, N, wn)) if kind == 'gbdt' else m[np.minimum(N.astype(int), cap)]).astype(np.float32)
        if c == 'base': L0 = [L[i] for i in sub]
        else: c2[c] = H.c2st(L0, [L[i] for i in sub])
        log(f'{ds} {c}: refs ' + ' '.join(f'{rn}/{tn} {np.mean(OUTZ[f"s|{rn}|{tn}|{c}"] - OUTZ[f"s|{rn}|{tn}|base"]):+.4f}' for rn, tn in refs) +
            (f' | C2ST {c2[c]["auc"]:.3f}' if c in c2 else '') + f' ({time.time() - t:.0f} s)')
    c2['none'] = H.c2st(L0, L0)
    info['c2st'] = c2
    for rn, tn in refs:
        rows, sig = task_rows(ds, y, next(x for x in TASKS[ds] if x[0] == tn))
        info.setdefault('unedited', {}).setdefault(rn, {})[tn] = dict(AUC=auc(OUTZ[f's|{rn}|{tn}|base'][rows], sig[rows].astype(int)))
    log(f'{ds} references unedited: {info["unedited"]}')
    np.savez_compressed(OUT / f'h18_{ds}_refs{sfx(smoke)}.npz', **OUTZ)
    (OUT / f'h18_{ds}_refs{sfx(smoke)}.json').write_text(json.dumps(info, indent=1, default=float))
    pool.close(); log(f'wrote h18_{ds}_refs')


# ======================================================================================================================= statistics
class Boot:
    def __init__(self, n, B=1000, P=2000, seed=SEED):
        r = np.random.default_rng(seed); self.idx = r.integers(0, n, (B, n), dtype=np.int32)
        self.sg = r.choice(np.array([-1.0, 1.0], np.float32), (P, n))

    def means(self, d):
        return d[self.idx].mean(1)

    def signflip(self, d):
        obs = abs(d.mean()); return float((int((np.abs(self.sg @ d / len(d)) >= obs - 1e-12).sum()) + 1) / (len(self.sg) + 1))

    def signflip_normal(self, d):
        """two-sided p of the sign-flip null in its normal limit (null sd = sqrt(sum d^2) / n); resolves p below 1 / (P + 1)"""
        from scipy.stats import norm
        sd = np.sqrt((d ** 2).sum()) / len(d)
        return float(2 * norm.sf(abs(d.mean()) / sd)) if sd > 0 else 1.0


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def p_boot(v, est=None):
    """two-sided bootstrap p: normal approximation from the bootstrap s.e. (est = point estimate); the percentile p is floored at 1/B"""
    from scipy.stats import norm
    if est is not None and v.std() > 0: return float(2 * norm.sf(abs(est) / v.std()))
    return float(min(1.0, 2 * max(min((v <= 0).mean(), (v >= 0).mean()), 1 / len(v))))


def mcnemar(c0, c1):
    from scipy.stats import binomtest
    a = int((c0 & ~c1).sum()); b = int((~c0 & c1).sum())
    return dict(lost=a, gained=b, p=float(binomtest(a, a + b, 0.5).pvalue) if a + b else 1.0)


def load_ds(ds, smoke):
    S = {}; info = {}
    for f in (f'h18_{ds}_model', f'h18_{ds}_refs') + ((f'h18_hls_small',) if ds == 'hls' else ()):
        p = OUT / f'{f}{sfx(smoke)}.npz'
        if not p.exists(): log('missing', p); continue
        Z = np.load(p); y = Z['y']
        for k in Z.files:
            if k.startswith('s|'): S[tuple(k.split('|')[1:])] = Z[k].astype(np.float64)
        if 'nseen|base' in Z.files: NS = {c: Z[f'nseen|{c}'].astype(float) for c in CONDS}
        j = OUT / f'{f}{sfx(smoke)}.json'
        if j.exists(): info[f.split('_', 2)[-1]] = json.loads(j.read_text())
    return y, S, NS, info


def stats_ds(ds, smoke, tests):
    y, S, NS, info = load_ds(ds, smoke)
    models = sorted({k[0] for k in S}); res = dict(ds=ds, tasks={}, info={k: {kk: vv for kk, vv in v.items() if kk != 'idx'} for k, v in info.items()})
    kx = np.array(KS, float)
    for task in TASKS[ds]:
        tn = task[0]; rows, sig = task_rows(ds, y, task); ix = np.flatnonzero(rows); sg = sig[ix]; bt = Boot(len(ix))
        R = dict(n=int(len(ix)), n_sig=int(sg.sum()), models={})
        dN = {c: NS[c][ix] - NS['base'][ix] for c in CONDS}
        R['dN_seen'] = {c: float(dN[c].mean()) for c in EDITS}
        Dk = {}
        for mn in models:
            if (mn, tn, 'base') not in S: continue
            s0 = S[(mn, tn, 'base')][ix]; thr = float(np.quantile(s0[sg], 0.5)); c0 = s0 > thr
            M = dict(unedited=dict(AUC=auc(s0, sg.astype(int))), edits={})
            for c in EDITS:
                if (mn, tn, c) not in S: continue
                d = S[(mn, tn, c)][ix] - s0; bm = bt.means(d); bn = bt.means(dN[c]); c1 = S[(mn, tn, c)][ix] > thr
                pp = bm / np.where(np.abs(bn) > 1e-9, bn, np.nan)
                r = dict(mean=float(d.mean()), ci=ci(bm), dN=float(dN[c].mean()),
                         per_particle=float(d.mean() / dN[c].mean()) if abs(dN[c].mean()) > 1e-9 else None,
                         per_particle_ci=ci(pp[np.isfinite(pp)]) if np.isfinite(pp).sum() > 100 else None,
                         p_signflip=bt.signflip_normal(d), p_signflip_mc=bt.signflip(d), calls_changed_pct=float(100 * (c0 != c1).mean()),
                         eff_sig_after=float(c1[sg].mean()), eff_bkg_before=float(c0[~sg].mean()), eff_bkg_after=float(c1[~sg].mean()),
                         mcnemar=mcnemar(c0, c1), by_class=dict(sig=float(d[sg].mean()), bkg=float(d[~sg].mean())))
                M['edits'][c] = r
                if mn in NETS[ds] and mn not in NULLS: tests.append([ds, tn, mn, c, r['p_signflip']])
            if all((mn, tn, f'k{k}') in S for k in KS):
                D = np.stack([S[(mn, tn, f'k{k}')][ix] - s0 for k in KS], 1); Dk[mn] = D
                mu = D.mean(0); bmu = np.stack([bt.means(D[:, q]) for q in range(len(KS))], 1)   # (B, 5)
                sl = np.polyfit(kx, mu, 1)[0]; bsl = np.polyfit(kx, bmu.T, 1)[0]
                M['k_slope'] = dict(means=mu.tolist(), slope=float(sl), ci=ci(bsl), p_boot=p_boot(bsl, sl),
                                    within_pm002=bool(ci(bsl)[0] > -0.02 and ci(bsl)[1] < 0.02))
                M['_bsl'] = bsl
            R['models'][mn] = M
        # paired slope comparisons with the references (same jets, same bootstrap resamples)
        for mn, M in R['models'].items():
            if '_bsl' not in M: continue
            M['slope_vs'] = {}
            for rn in ('EFP', 'EFPN', 'LR_N'):
                if rn not in R['models'] or '_bsl' not in R['models'][rn] or rn == mn: continue
                dv = M['_bsl'] - R['models'][rn]['_bsl']
                df = float(M['k_slope']['slope'] - R['models'][rn]['k_slope']['slope'])
                M['slope_vs'][rn] = dict(diff=df, ci=ci(dv), p_boot=p_boot(dv, df))
                if mn in NETS[ds] and mn not in NULLS: tests.append([ds, tn, mn, f'slope-{rn}', M['slope_vs'][rn]['p_boot']])
            # fraction of the count-aware reference slope: slope_net / slope_LR_N with bootstrap CI
            if 'LR_N' in R['models'] and '_bsl' in R['models']['LR_N']:
                b = M['_bsl'] / R['models']['LR_N']['_bsl']
                M['slope_over_LRN'] = dict(value=float(M['k_slope']['slope'] / R['models']['LR_N']['k_slope']['slope']), ci=ci(b[np.isfinite(b)]))
        for M in R['models'].values(): M.pop('_bsl', None)
        res['tasks'][tn] = R
        lr = R['models'].get('LR_N', {}).get('k_slope', {}).get('slope')
        log(f'{ds} {tn}: k-slope per particle (LR_N {lr}): ' + '; '.join(f'{mn} {M["k_slope"]["slope"]:+.4f} {np.round(M["k_slope"]["ci"], 4).tolist()}'
                                                                      for mn, M in R['models'].items() if 'k_slope' in M))
    return res


def part_stats(smoke):
    tests = []; ALL = {}
    for ds in DSETS:
        if not (OUT / f'h18_{ds}_model{sfx(smoke)}.npz').exists(): log('skip', ds); continue
        ALL[ds] = stats_ds(ds, smoke, tests)
    h = holm([t[4] for t in tests]) if tests else []
    for t, hh in zip(tests, h): t.append(float(hh))
    for ds, R in ALL.items():
        R['holm'] = [t for t in tests if t[0] == ds]
        (OUT / f'h18_{ds}{sfx(smoke)}.json').write_text(json.dumps(R, indent=1, default=float))
    # summary table: per network, k-slope, e1 / pu5 / pu20 / c4_20 / c5_20 / c0 per-particle responses, outcome flags
    summ = dict(n_tests=len(tests), n_holm_lt_05=int(sum(x < 0.05 for x in h)), rows=[])
    for ds, R in ALL.items():
        for tn, T in R['tasks'].items():
            lr = T['models'].get('LR_N', {}).get('k_slope', {})
            for mn, M in T['models'].items():
                if 'k_slope' not in M: continue
                E = M['edits']; g = lambda c, k='per_particle': E.get(c, {}).get(k)
                hp = {t[3]: t[5] for t in R['holm'] if t[1] == tn and t[2] == mn}
                summ['rows'].append(dict(ds=ds, task=tn, model=mn, kind='net' if mn in NETS[ds] else 'ref', null=mn in NULLS, irc_safe=mn in IRC_SAFE,
                                         AUC=M['unedited']['AUC'], k_slope=M['k_slope']['slope'], k_slope_ci=M['k_slope']['ci'],
                                         within_pm002=M['k_slope']['within_pm002'], LR_N_slope=lr.get('slope'),
                                         slope_over_LRN=M.get('slope_over_LRN', {}).get('value'), slope_over_LRN_ci=M.get('slope_over_LRN', {}).get('ci'),
                                         e1=g('e1', 'mean'), e1_ci=g('e1', 'ci'), e1_holm=hp.get('e1'),
                                         pu5_pp=g('pu5'), pu5_pp_ci=g('pu5', 'per_particle_ci'), pu20_pp=g('pu20'), pu20_pp_ci=g('pu20', 'per_particle_ci'),
                                         c4_20_pp=g('c4_20'), c5_20=g('c5_20', 'mean'), c0=g('c0', 'mean'), c0_ci=g('c0', 'ci'),
                                         k20_calls_pct=g('k20', 'calls_changed_pct'), pu20_calls_pct=g('pu20', 'calls_changed_pct'),
                                         k20_eff_sig=g('k20', 'eff_sig_after'), pu20_eff_sig=g('pu20', 'eff_sig_after'),
                                         slope_vs=M.get('slope_vs')))
    (OUT / f'h18_summary{sfx(smoke)}.json').write_text(json.dumps(summ, indent=1, default=float))
    for r in summ['rows']:
        log(f'{r["ds"]:5s} {r["task"]:10s} {r["model"]:18s} AUC {r["AUC"]:.4f} k-slope {r["k_slope"]:+.4f} {np.round(r["k_slope_ci"], 4).tolist()} '
            f'LR_N {r["LR_N_slope"]} e1 {r["e1"]} pu20/particle {r["pu20_pp"]} c4_20/particle {r["c4_20_pp"]} c0 {r["c0"]}')
    log(f'wrote h18_summary ({len(tests)} tests, {summ["n_holm_lt_05"]} Holm < 0.05)')
    try:
        part_fig(smoke)
    except Exception as e:
        log('figure failed', repr(e))


def part_fig(smoke):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    panels = [('qgpy', 'q_vs_g', 'Quark-gluon, Pythia jets\nquark vs gluon'), ('qghw', 'q_vs_g', 'Quark-gluon, Herwig jets\nquark vs gluon'),
              ('jc', 'W_vs_QCD', 'JetClass\nW vs QCD'), ('jc', 'top_vs_QCD', 'JetClass\ntop vs QCD'), ('top', 't_vs_QCD', 'Top Landscape\ntop vs QCD'),
              ('hls', 'W_vs_rest', 'hls4ml\nW vs rest')]
    panels = [p for p in panels if (OUT / f'h18_{p[0]}{sfx(smoke)}.json').exists()]
    fig, ax = plt.subplots(2, len(panels), figsize=(3.6 * len(panels), 8.2), squeeze=False)
    cm = plt.get_cmap('tab10'); refst = dict(LR_N=('k', '--'), EFPN=('0.45', '-.'), EFP=('0.65', ':'))
    pc = ['e1', 'c4_20']; pl = ['one soft particle\non the data manifold (e1)', 'ρ = 20 disc particles\nmoved to r/20 (c4)']
    for j, (ds, tn, title) in enumerate(panels):
        R = json.loads((OUT / f'h18_{ds}{sfx(smoke)}.json').read_text())['tasks'][tn]; a = ax[0, j]; b = ax[1, j]
        nets = [m for m in NETS[ds] if m in R['models'] and m not in NULLS]
        for i, m in enumerate(nets):
            M = R['models'][m]; mu = np.array(M['k_slope']['means'])
            cis = np.array([M['edits'][f'k{k}']['ci'] for k in KS])
            a.errorbar(KS, mu, yerr=[mu - cis[:, 0], cis[:, 1] - mu], color=cm(i), marker='o', ms=3, lw=1.2, label=LABEL.get(m, m))
        for rn, (col, ls) in refst.items():
            if rn in R['models'] and 'k_slope' in R['models'][rn]:
                a.plot(KS, R['models'][rn]['k_slope']['means'], color=col, ls=ls, lw=1.6, label={'LR_N': 'LR(N) (count only)', 'EFPN': 'GBDT(EFPs + N)', 'EFP': 'GBDT(EFPs), count-blind'}[rn])
        a.axhline(0, color='grey', lw=0.5); a.set_title(title, fontsize=10); a.set_xlabel('k added soft particles (from the jet\'s own soft half)', fontsize=8)
        if j == 0: a.set_ylabel('mean Δ(signal log-odds)', fontsize=9)
        a.legend(fontsize=6.5, loc='best')
        bars = nets + ['EFP', 'LR_N']; w = 0.8 / len(bars)
        for i, m in enumerate(bars):
            if m not in R['models']: continue
            E = R['models'][m]['edits']
            v = [E[c]['per_particle'] for c in pc]
            lo = [E[c]['per_particle_ci'] for c in pc]
            v = np.array([x if x is not None else np.nan for x in v]); lo = np.array([x if x is not None else [np.nan, np.nan] for x in lo])
            col = 'k' if m == 'LR_N' else ('0.65' if m == 'EFP' else cm(i))
            b.bar(np.arange(len(pc)) + (i - (len(bars) - 1) / 2) * w, v, width=w, color=col,
                  yerr=[v - lo[:, 0], lo[:, 1] - v], error_kw=dict(lw=0.6),
                  label={'LR_N': 'LR(N)', 'EFP': 'GBDT(EFPs)'}.get(m, LABEL.get(m, m)))
        b.axhline(0, color='grey', lw=0.5); b.set_xticks(range(len(pc))); b.set_xticklabels(pl, fontsize=7.5)
        b.legend(fontsize=6, loc='lower left', ncol=2)
        if j == 0: b.set_ylabel('Δ(signal log-odds) per added particle\n(count changed, IRC-safe content ~fixed)', fontsize=8.5)
    fig.suptitle('Response of jet taggers to added soft particles, by dataset (signal log-odds; 95 % bootstrap intervals; N = 10,000 jets per dataset)', fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / f'h18_softcount{sfx(smoke)}.png', dpi=110); log('wrote', OUT / f'h18_softcount{sfx(smoke)}.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', required=True, choices=['model', 'hlss', 'refs', 'stats', 'fig'])
    ap.add_argument('--ds', default=None); ap.add_argument('--smoke', type=int, default=0); ap.add_argument('--cpu', type=int, default=4)
    a = ap.parse_args(); log(f'H18 part {a.part} ds {a.ds} smoke {a.smoke}')
    if a.part == 'model': dict(qg=part_qg, jc=part_jc, top=part_top, hls=part_hls)[a.ds](a.smoke)
    elif a.part == 'hlss': part_hlss(a.smoke)
    elif a.part == 'refs': part_refs(a.ds, a.smoke, a.cpu)
    elif a.part == 'stats': part_stats(a.smoke)
    else: part_fig(a.smoke)
    log('done')
