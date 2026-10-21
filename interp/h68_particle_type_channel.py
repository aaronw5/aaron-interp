"""H68 (loop D, round D14, exploratory; claim D10 v2 / new D claim on particle-type composition): which particle-type channel
(charged hadron, photon, neutral hadron) the full-input JetClass taggers read when the particles' four-momenta are held fixed,
plus pileup-like soft ghosts with the JetClass particle-type composition.

Arm 1 (relabel, --part jc). Sample: all 10 JetClass classes, the first test file of each class (test_20M/<sample>_100.root),
  1,000 random rows per class (numpy default_rng(6801); never first-k). N = 10,000 jets.
  Every edit keeps each particle's (px, py, pz, E) and changes only its type fields. A particle made neutral gets charge 0 and
  d0val = d0err = dzval = dzerr = 0, which is how Delphes stores neutral particles in JetClass (checked on the sample and
  logged). Edits:
    nh2ph        every neutral hadron relabelled as a photon
    ph2nh        every photon relabelled as a neutral hadron
    ch2nh2      every charged hadron with pT < 2 GeV relabelled as a neutral hadron (track fields removed)
    ch2g2       the same particles relabelled as photons (separates "track removed" from "neutral-hadron flag")
    fr05/10/20  energy-fraction-matched: charged hadrons taken in a random order (rng [6801, 2]) until their summed energy is
                closest to f x (jet energy), f = 5 / 10 / 20 %, relabelled as neutral hadrons; the achieved charged-energy
                fraction change Delta f_ch is recorded per jet
    frg10       the same particle set as fr10 relabelled as photons
  Controls on the same particle sets (fr10 and ch2nh2):
    csame_<set> within-category relabel at the same particles: charges of equal numbers of + and - charged hadrons in the set
                are flipped (type stays charged hadron, net jet charge unchanged, track fields unchanged); JetClass has no pi/K
                flag, so the charge sign is the only within-category label available
    ckin_<set>  the same particles keep their labels and their four-momenta are scaled by (1 + eps), eps ~ N(0, 0.005)
                per particle (0.5 % jitter); jet_pt / jet_energy updated by the change of the particle sums
  Kinematics-only models (ParT_kin, PN_kin) see no type field, so their response to every relabel edit must be exactly 0.
Arm 2 (ghosts, same sample). Soft particles added uniformly in the disc of radius 0.8 about the stored jet axis, with the
  JetClass composition: the type (charged hadron / photon / neutral hadron) drawn with the proportions measured on the
  sample's real particles with pT < 2 GeV (leptons excluded), pT drawn from that type's measured pT spectrum below 2 GeV,
  mass = that type's median measured mass, charged ghosts get charge +-1 and the (d0val, d0err, dzval, dzerr) of a random
  real soft charged hadron of the sample. Count Poisson with mean rho * pi * 0.8^2 / <pT_soft>, rho = added pT per unit
  eta-phi area in GeV; one master set at rho = 20 thinned to 5 and 10 (nested). Conditions (all through part_core.build):
    gbase       the unedited jet passed through build (the reference for this arm)
    gmix5/10/20 composition ghosts at rho 5, 10, 20
    gnh10 / gph10 / gch10   the rho = 10 ghosts (same pT and positions) all labelled neutral hadron (H65's label) / photon /
                charged hadron
    gstack10    the rho = 10 composition ghosts all 0.002 rad from the axis (H65's c2: count and summed pT matched, mass ~0)
    gclus10     the physical replacement for the stack: the same ghosts in groups of 5; each group at a random azimuth about
                the axis, its members at radius uniform in [0.03, 0.1] rad and within +-0.3 rad of the group azimuth
                (count and summed pT matched, added mass a few GeV at most)
Arm 3 (--part top): Top Landscape test, 1,000 top + 1,000 QCD random rows (rng 6803), H65 ghosts (massless, pT 0.5-1 GeV,
  rho = 10): pu10 uniform, c2_10 on the axis, cl10 clustered as gclus10. Models ParT, PN, LorentzNet (LNet), PFN, EFN,
  PFN_permuted (h13_pileup_panel.TopModels). Gives the clustered replacement for LorentzNet, which has no JetClass model.
Models (arm 1-2): ParT_full, PN_full (type inputs); ParT_kin, PN_kin (no type inputs); six ParT_full copies with every weight
  tensor's entries permuted (torch seeds 1..6) as the label-artefact null.
Score: s_c = logit_c - logit_QCD on jets of true class c (c = 1..9); for QCD jets s_QCD = logit_QCD - logsumexp(other logits).
Metric: Delta s per class and edit (mean over the class's 1,000 jets) with 95 % bootstrap interval (B = 1000 jet resamples);
  p (normal approximation from the bootstrap s.e.); slope dDelta s / dDelta f_ch through the origin pooled over fr05/10/20
  (bootstrap); McNemar exact p on 10-class correct -> wrong vs wrong -> correct; Holm across classes x edits x models.
References (JetClass train_100M/<sample>_000.root, 2,000 random rows per class, rng 6802, disjoint from the test sample):
  LR(f_ch), LR(f_nh) = class-vs-QCD histogram likelihood ratios (20 bins, +1 smoothing); GBDT_frac = HistGradientBoosting on
  (m, tau21, tau32, N, f_ch, f_nh, f_gamma) and GBDT_kin on (m, tau21, tau32, N), class vs QCD. Under each relabel edit
  m, tau and N are unchanged, so GBDT_kin's Delta is 0 and GBDT_frac's Delta log-odds is the reference response.
  f_ch = energy of charged particles / summed particle energy; f_nh, f_gamma likewise for neutral hadrons, photons.
Run
  ./part_job.sh d-h68-run 'python -m interp.h68_particle_type_channel --part jc && python -m interp.h68_particle_type_channel --part top && python -m interp.h68_particle_type_channel --part fig'
  -> $AI_RESULTS/physics/h68_jc.json, h68_top.json, h68_particle_type.png ; --smoke N: N jets per class
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

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
OUT = RES / 'physics'
SEED, SEED_REF, SEED_TOP = 6801, 6802, 6803
CLS = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
KNOWN = dict(ParT_full=0.8593, ParT_kin=0.7447, PN_full=0.8457, PN_kin=0.7034)
TRAIN = os.environ.get('H68_TRAIN', '/j-jepa-vol/JetClass/Pythia/train_100M')
RJ, B = 0.8, 1000
TRK = ('d0val', 'd0err', 'dzval', 'dzerr')
T0_ = time.time()


def log(*a):
    print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


def ci(v):
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else [float('nan')] * 2


def p_z(est, v):
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]; se = v.std(ddof=1) if v.size > 2 else 0.0
    if se <= 0: return 1.0 if est == 0 else 0.0
    return float(math.erfc(abs(est) / se / math.sqrt(2)))


# ================================================================================================================ data
def read_rows(path, idx, extra=('jet_tau1', 'jet_tau2', 'jet_tau3', 'jet_sdmass')):
    import uproot
    from interp.part_core import read_root, cat, sub
    n = uproot.open(path)['tree'].num_entries; parts = []
    for a in range(0, n, 20000):
        loc = idx[(idx >= a) & (idx < a + 20000)] - a
        if len(loc): parts.append(sub(read_root(path, start=a, stop=min(a + 20000, n)), loc))
    J = cat(parts); X = uproot.open(path)['tree'].arrays(list(extra), library='np')
    for k in extra: J[k] = np.asarray(X[k])[idx].astype(np.float32)
    return J


def sample(per, seed, src='test'):
    import uproot
    from interp.part_core import fof, cat, PRE
    rng = np.random.default_rng(seed); Js = []
    for c in CLS:
        f = fof(c) if src == 'test' else Path(TRAIN) / f'{PRE[c]}_000.root'
        n = uproot.open(f)['tree'].num_entries; idx = np.sort(rng.choice(n, per, replace=False))
        Js.append(read_rows(f, idx)); log(f'{src} {c}: {per} random rows of {n} ({Path(f).name})')
    return cat(Js)


def fractions(J):
    m = J['mask'] > 0; E = J['energy'] * m; tot = np.maximum(E.sum(1), 1e-9)
    return dict(f_ch=(E * (np.abs(J['charge']) > 0)).sum(1) / tot, f_nh=(E * (J['isNeutralHadron'] > 0)).sum(1) / tot,
                f_g=(E * (J['isPhoton'] > 0)).sum(1) / tot)


def shape_feats(J):
    from interp.part_core import jmass
    t1, t2, t3 = (np.maximum(J[k], 1e-9) for k in ('jet_tau1', 'jet_tau2', 'jet_tau3'))
    return np.stack([jmass(J), J['jet_tau2'] / t1, J['jet_tau3'] / t2, J['mask'].sum(1)], 1).astype(np.float64)


# ================================================================================================================ edits
def copyJ(J):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in J.items()}


def relabel(J, sel, to):
    U = copyJ(J); s = sel & (J['mask'] > 0)
    for k in ('isChargedHadron', 'isNeutralHadron', 'isPhoton', 'isElectron', 'isMuon', 'charge') + TRK:
        U[k] = np.where(s, 0.0, U[k]).astype(np.float32)
    U['isNeutralHadron' if to == 'nh' else 'isPhoton'] = np.where(s, 1.0, U['isNeutralHadron' if to == 'nh' else 'isPhoton']).astype(np.float32)
    return U


def frac_set(J, f, rng):
    """charged hadrons in a random order; the prefix whose summed energy is closest to f x (summed particle energy)"""
    ch = (J['isChargedHadron'] > 0) & (J['mask'] > 0); E = J['energy'] * (J['mask'] > 0); tot = E.sum(1)
    key = np.where(ch, rng.random(ch.shape), np.inf); o = np.argsort(key, 1)
    Eo = np.where(np.take_along_axis(ch, o, 1), np.take_along_axis(E, o, 1), 0.0); cum = np.concatenate([np.zeros((len(E), 1)), np.cumsum(Eo, 1)], 1)
    k = np.abs(cum - f * tot[:, None]).argmin(1)            # number of particles taken (0 .. n_ch)
    rank = np.empty_like(o); np.put_along_axis(rank, o, np.arange(o.shape[1])[None].repeat(len(o), 0), 1)
    return ch & (rank < k[:, None])


def csame(J, sel, rng):
    """flip the charge of equal numbers of + and - charged hadrons in sel (net charge kept)"""
    U = copyJ(J); s = sel & (J['isChargedHadron'] > 0)
    key = rng.random(s.shape); q = J['charge']
    pk = np.where(s & (q > 0), key, np.inf); mk = np.where(s & (q < 0), key, np.inf)
    rp = np.argsort(np.argsort(pk, 1), 1); rm = np.argsort(np.argsort(mk, 1), 1)
    m = np.minimum((s & (q > 0)).sum(1), (s & (q < 0)).sum(1))[:, None]
    flip = (s & (q > 0) & (rp < m)) | (s & (q < 0) & (rm < m))
    U['charge'] = np.where(flip, -q, q).astype(np.float32)
    return U, flip


def ckin(J, sel, rng):
    U = copyJ(J); s = sel & (J['mask'] > 0); eps = np.where(s, rng.normal(0, 0.005, s.shape), 0.0); g = (1 + eps).astype(np.float32)
    for k in ('px', 'py', 'pz', 'energy'): U[k] = (J[k] * g).astype(np.float32)
    m = J['mask'] > 0; sx = lambda D, k: (D[k] * m).sum(1)
    pt0 = np.hypot(sx(J, 'px'), sx(J, 'py')); pt1 = np.hypot(sx(U, 'px'), sx(U, 'py'))
    U['jet_pt'] = (J['jet_pt'] * pt1 / np.maximum(pt0, 1e-9)).astype(np.float32)
    U['jet_energy'] = (J['jet_energy'] + sx(U, 'energy') - sx(J, 'energy')).astype(np.float32)
    return U


# ---------------------------------------------------------------------------------------------------------------- ghosts
def soft_pool(J):
    from interp.part_core import kin
    pt, de, dp, ms = kin(J); m = (J['mask'] > 0) & (pt < 2.0) & (pt > 0)
    types = dict(ch=m & (J['isChargedHadron'] > 0), ph=m & (J['isPhoton'] > 0), nh=m & (J['isNeutralHadron'] > 0))
    n = {k: int(v.sum()) for k, v in types.items()}; tot = sum(n.values())
    P = dict(prop={k: n[k] / tot for k in n}, pt={k: pt[v].astype(np.float64) for k, v in types.items()},
             mass={k: float(np.median(ms[v])) for k, v in types.items()}, trk=np.stack([J[k][types['ch']] for k in TRK], 1).astype(np.float64))
    P['mean_pt'] = float(sum(P['prop'][k] * P['pt'][k].mean() for k in n))
    return P


def ghost_master(Nj, P, seed, rhomax=20.0):
    rng = np.random.default_rng([seed, 11]); A = math.pi * RJ ** 2
    n = rng.poisson(rhomax * A / P['mean_pt'], Nj); K = int(n.max()); on = np.arange(K)[None] < n[:, None]
    keys = ['ch', 'ph', 'nh']; t = rng.choice(3, (Nj, K), p=[P['prop'][k] for k in keys])
    pt = np.zeros((Nj, K)); ms = np.zeros((Nj, K))
    for i, k in enumerate(keys):
        sel = t == i; pt[sel] = rng.choice(P['pt'][k], int(sel.sum())); ms[sel] = P['mass'][k]
    r = RJ * np.sqrt(rng.random((Nj, K))); th = 2 * np.pi * rng.random((Nj, K))
    trk = P['trk'][rng.integers(0, len(P['trk']), (Nj, K))]; q = np.where(rng.random((Nj, K)) < 0.5, -1.0, 1.0)
    # clustered layout: groups of 5 consecutive ghosts share an azimuth; members at radius U[0.03, 0.1], azimuth +- 0.3
    grp = np.arange(K) // 5; gaz = 2 * np.pi * rng.random((Nj, grp.max() + 1))[:, grp]
    rc = rng.uniform(0.03, 0.1, (Nj, K)); ac = gaz + rng.uniform(-0.3, 0.3, (Nj, K))
    ast = 2 * np.pi * rng.random((Nj, K))
    return dict(on=on, type=t, pt=pt, m=ms, de=r * np.cos(th), dp=r * np.sin(th), de_cl=rc * np.cos(ac), dp_cl=rc * np.sin(ac),
                de_st=0.002 * np.cos(ast), dp_st=0.002 * np.sin(ast), trk=trk, q=q, v=rng.random((Nj, K)), n20=n)


def with_ghosts(J, G, rho, layout='uni', label='mix'):
    """part_core.build of the jet plus the ghosts kept at density rho (nested thinning), positions per layout, types per label"""
    from interp.part_core import kin, build, PK
    pt, de, dp, ms = (np.asarray(a, np.float64) for a in kin(J)); on = G['on'] & (G['v'] < rho / 20.0) if rho > 0 else np.zeros_like(G['on'])
    gde, gdp = dict(uni=(G['de'], G['dp']), cl=(G['de_cl'], G['dp_cl']), st=(G['de_st'], G['dp_st']))[layout]
    t = dict(mix=G['type'], nh=np.full_like(G['type'], 2), ph=np.full_like(G['type'], 1), ch=np.zeros_like(G['type']))[label]
    gm = np.where(t == 0, G['m'] * 0 + 0.13957, 0.0) if label != 'mix' else G['m']
    if label == 'nh': gm = np.zeros_like(G['m'])
    gpt = np.where(on, G['pt'], 0.0)
    ex = {}
    for k in PK:
        if k in ('px', 'py', 'pz', 'energy', 'deta', 'dphi'): continue
        a = np.zeros_like(gpt)
        if k == 'mask': a = on.astype(np.float64)
        elif k == 'isChargedHadron': a = on & (t == 0)
        elif k == 'isPhoton': a = on & (t == 1)
        elif k == 'isNeutralHadron': a = on & (t == 2)
        elif k == 'charge': a = np.where(on & (t == 0), G['q'], 0.0)
        elif k in TRK: a = np.where(on & (t == 0), G['trk'][..., TRK.index(k)], 0.0)
        ex[k] = np.concatenate([np.asarray(J[k], np.float64), np.asarray(a, np.float64)], 1)
    cat2 = lambda a, b: np.concatenate([a, np.where(on, b, 0.0)], 1)
    return build(J, cat2(pt, gpt), cat2(de, gde), cat2(dp, gdp), cat2(ms, gm), extra=ex), on.sum(1), (gpt).sum(1)


# ================================================================================================================ models
def jc_models(nperm):
    import torch
    from interp.part_core import ParTNetwork, inputs
    from interp.p1v_verify_part import fwd
    from interp.pn_net import load_particlenet
    from interp.h1_irc_panel import pn_logits
    from jetdistill.config import MODELS
    dev = os.environ.get('JETDISTILL_DEVICE', 'cpu'); nets = {}
    for n in ('full', 'kin'):
        net = ParTNetwork(n); nets[f'ParT_{n}'] = (lambda J, net=net, n=n: fwd(net, *inputs(J, n))[1])
        pn = load_particlenet(Path(MODELS).parent / 'particlenet' / f'ParticleNet_{n}.pt', 17 if n == 'full' else 7, dev)
        nets[f'PN_{n}'] = (lambda J, pn=pn, n=n: pn_logits(pn, J, n))
    for s in range(1, nperm + 1):
        netp = ParTNetwork('full'); g = torch.Generator().manual_seed(s)
        with torch.no_grad():
            for p in netp.model.parameters():
                if p.numel() > 1:
                    f = p.data.flatten().cpu(); p.data.copy_(f[torch.randperm(f.numel(), generator=g)].view_as(p).to(p.device))
        nets[f'ParT_full_perm{s}'] = (lambda J, net=netp: fwd(net, *inputs(J, 'full'))[1])
    return nets


def scores(L, y):
    L = np.asarray(L, np.float64); s = L[np.arange(len(y)), y] - L[:, 0]
    oth = np.logaddexp.reduce(L[:, 1:], axis=1); return np.where(y == 0, L[:, 0] - oth, s)


# ================================================================================================================ references
def references(Jr, Jt_conds, yr):
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    Fr = fractions(Jr); Sr = shape_feats(Jr); out = {}; models = {}
    Xk = Sr; Xf = np.concatenate([Sr, np.stack([Fr['f_ch'], Fr['f_nh'], Fr['f_g']], 1)], 1)
    rng = np.random.default_rng([SEED_REF, 3]); fold = rng.random(len(yr)) < 0.5
    for c in range(1, 10):
        r = np.isin(yr, [0, c]); yy = (yr[r] == c).astype(int); A = {}
        for nm, X in (('kin', Xk[r]), ('frac', Xf[r])):
            te = fold[r]; m = HGB(max_iter=200, learning_rate=0.1, random_state=0).fit(X[~te], yy[~te])
            from interp.h18_softcount_cross import auc
            A[nm] = auc(m.predict_proba(X[te])[:, 1], yy[te]); models[(c, nm)] = HGB(max_iter=200, learning_rate=0.1, random_state=0).fit(X, yy)
        lr = {}
        for fk in ('f_ch', 'f_nh'):
            e = np.linspace(0, 1, 21); hs = np.histogram(Fr[fk][r][yy == 1], e)[0] + 1.0; hb = np.histogram(Fr[fk][r][yy == 0], e)[0] + 1.0
            lr[fk] = np.log(hs / hs.sum()) - np.log(hb / hb.sum())
        out[CLS[c]] = dict(auc_gbdt_kin=float(A['kin']), auc_gbdt_frac=float(A['frac']), lr=lr,
                           mean_f={k: [float(Fr[k][r][yy == 0].mean()), float(Fr[k][r][yy == 1].mean())] for k in Fr})
        log(f'ref {CLS[c]}: AUC GBDT kin {A["kin"]:.4f} frac {A["frac"]:.4f}; mean f_ch QCD {out[CLS[c]]["mean_f"]["f_ch"][0]:.3f} vs {out[CLS[c]]["mean_f"]["f_ch"][1]:.3f}')
    return out, models


def ref_scores(models, refs, J, y):
    """per jet, reference log-odds of its own class vs QCD (QCD jets: vs every class, reported separately): GBDT_frac, LR f_ch, LR f_nh"""
    F = fractions(J); S = shape_feats(J); X = np.concatenate([S, np.stack([F['f_ch'], F['f_nh'], F['f_g']], 1)], 1)
    R = {k: np.zeros(len(y)) for k in ('gbdt', 'lr_fch', 'lr_fnh')}; Q = {}
    bins = lambda v: np.clip((v * 20).astype(int), 0, 19)
    for c in range(1, 10):
        p = np.clip(models[(c, 'frac')].predict_proba(X)[:, 1], 1e-6, 1 - 1e-6); lo = np.log(p / (1 - p))
        lc = refs[CLS[c]]['lr']['f_ch'][bins(F['f_ch'])]; ln = refs[CLS[c]]['lr']['f_nh'][bins(F['f_nh'])]
        r = y == c; R['gbdt'][r] = lo[r]; R['lr_fch'][r] = lc[r]; R['lr_fnh'][r] = ln[r]
        Q[CLS[c]] = dict(gbdt=lo[y == 0], lr_fch=lc[y == 0], lr_fnh=ln[y == 0])
    return R, Q, F


# ================================================================================================================ run JetClass
def run_jc(smoke):
    from interp.h1_irc_panel import holm, binom_two_sided
    OUT.mkdir(parents=True, exist_ok=True); per = smoke or 1000
    J = sample(per, SEED); y = J['label'].astype(int); Nj = len(y)
    assert np.bincount(y, minlength=10).tolist() == [per] * 10, np.bincount(y)
    m = J['mask'] > 0; neu = m & ((J['isNeutralHadron'] > 0) | (J['isPhoton'] > 0))
    chk = {k: float(np.abs(J[k][neu]).max()) for k in ('charge',) + TRK}; log('max |field| on real neutral particles (Delphes convention check):', chk)
    F0 = fractions(J); log('sample mean f_ch / f_nh / f_g:', {k: round(float(v.mean()), 4) for k, v in F0.items()})
    Jr = sample(smoke or 2000, SEED_REF, 'train'); yr = Jr['label'].astype(int)
    refs, rmodels = references(Jr, None, yr); del Jr
    rng = np.random.default_rng([SEED, 2]); pt = np.hypot(J['px'], J['py'])
    sets = dict(fr05=frac_set(J, 0.05, np.random.default_rng([SEED, 2, 5])), fr10=frac_set(J, 0.10, np.random.default_rng([SEED, 2, 10])),
                fr20=frac_set(J, 0.20, np.random.default_rng([SEED, 2, 20])), ch2=(J['isChargedHadron'] > 0) & m & (pt < 2.0))
    C = {'base': J, 'nh2ph': relabel(J, J['isNeutralHadron'] > 0, 'g'), 'ph2nh': relabel(J, J['isPhoton'] > 0, 'nh'),
         'ch2nh2': relabel(J, sets['ch2'], 'nh'), 'ch2g2': relabel(J, sets['ch2'], 'g'),
         'fr05': relabel(J, sets['fr05'], 'nh'), 'fr10': relabel(J, sets['fr10'], 'nh'), 'fr20': relabel(J, sets['fr20'], 'nh'),
         'frg10': relabel(J, sets['fr10'], 'g')}
    nflip = {}
    for sn in ('fr10', 'ch2'):
        C[f'csame_{sn}'], fl = csame(J, sets[sn], np.random.default_rng([SEED, 4, len(sn)])); nflip[sn] = [float(fl.sum(1).mean()), float(sets[sn].sum(1).mean())]
        C[f'ckin_{sn}'] = ckin(J, sets[sn], np.random.default_rng([SEED, 5, len(sn)]))
    log('csame: mean flipped / mean set size per jet', nflip)
    # ghost arm
    P = soft_pool(J); log('soft (pT < 2 GeV) composition', {k: round(v, 4) for k, v in P['prop'].items()}, 'mean pT', round(P['mean_pt'], 3),
                          'median masses', P['mass'])
    G = ghost_master(Nj, P, SEED); log(f'ghost master: mean count at rho 20 {G["n20"].mean():.1f}')
    GH = {}
    for name, rho, lay, lab in (('gbase', 0, 'uni', 'mix'), ('gmix5', 5, 'uni', 'mix'), ('gmix10', 10, 'uni', 'mix'), ('gmix20', 20, 'uni', 'mix'),
                                ('gnh10', 10, 'uni', 'nh'), ('gph10', 10, 'uni', 'ph'), ('gch10', 10, 'uni', 'ch'),
                                ('gstack10', 10, 'st', 'mix'), ('gclus10', 10, 'cl', 'mix')):
        U, ng, gq = with_ghosts(J, G, rho, lay, lab); C[name] = U; GH[name] = dict(n_added=float(ng.mean()), pt_added=float(gq.mean()))
    from interp.part_core import jmass
    M0 = jmass(C['gbase'])
    for g in GH: GH[g]['dm'] = float((jmass(C[g]) - M0).mean()); GH[g]['dN'] = float((C[g]['mask'].sum(1) - C['gbase']['mask'].sum(1)).mean())
    log('ghost conditions', GH)
    rel_conds = [c for c in C if c not in GH and c != 'base']; gh_conds = [c for c in GH if c != 'gbase']
    dfch = {c: fractions(C[c])['f_ch'] - F0['f_ch'] for c in rel_conds}
    log('mean Delta f_ch per edit', {c: round(float(dfch[c].mean()), 4) for c in rel_conds})
    # references on each condition
    R0, Q0, _ = ref_scores(rmodels, refs, J, y); REF = {}
    for c in rel_conds:
        R1, Q1, _ = ref_scores(rmodels, refs, C[c], y)
        REF[c] = {CLS[k]: {r: float((R1[r] - R0[r])[y == k].mean()) for r in R1} for k in range(1, 10)}
    nets = jc_models(6 if not smoke else 2); S = {}; acc = {}; conf = {}
    for mn, f in nets.items():
        t = time.time(); perm = 'perm' in mn
        for c in C:
            if perm and c in GH and c not in ('gbase', 'gmix10', 'gnh10', 'gph10'): continue
            L = np.asarray(f(C[c]), np.float64); S[(mn, c)] = (scores(L, y), L.argmax(1))
            if c in ('base', 'gbase'):
                a = L.argmax(1); acc[(mn, c)] = float((a == y).mean())
                if c == 'base': conf[mn] = np.bincount(y * 10 + a, minlength=100).reshape(10, 10).tolist()
        log(f'{mn}: 10-class acc base {acc[(mn, "base")]:.4f} gbase {acc[(mn, "gbase")]:.4f} (known {KNOWN.get(mn, "-")}); per class '
            + ' '.join(f'{CLS[k]}:{(S[(mn, "base")][1][y == k] == k).mean():.3f}' for k in range(10)) + f' ({time.time() - t:.0f} s)')
    # statistics
    out = dict(sample=f'JetClass test_20M <sample>_100.root, {per} random rows per class (default_rng({SEED}))', n=Nj,
               reference_sample=f'train_100M <sample>_000.root, {smoke or 2000} random rows per class (default_rng({SEED_REF}))',
               delphes_check=chk, mean_fractions={k: float(v.mean()) for k, v in F0.items()}, soft_composition=P['prop'],
               soft_mean_pt=P['mean_pt'], soft_mass=P['mass'], ghosts=GH, csame_flips=nflip, refs={k: {kk: vv for kk, vv in v.items() if kk != 'lr'} for k, v in refs.items()},
               ref_delta=REF, accuracy={f'{a}|{b}': v for (a, b), v in acc.items()}, confusion=conf, known=KNOWN,
               dfch={c: {CLS[k]: float(dfch[c][y == k].mean()) for k in range(10)} for c in rel_conds},
               nrel={c: float((C[c]['isNeutralHadron'] != J['isNeutralHadron']).sum(1).mean() + (C[c]['isPhoton'] != J['isPhoton']).sum(1).mean()) for c in rel_conds},
               cells={}, slopes={}, tests={})
    idx = {k: np.random.default_rng([SEED, 99, k]).integers(0, per, (B, per)) for k in range(10)}
    fam_s, fam_m = [], []
    for (mn, c), (s, a) in S.items():
        if c in ('base', 'gbase'): continue
        b0 = 'gbase' if c in GH else 'base'; s0, a0 = S[(mn, b0)]; d = s - s0; cor0, cor1 = a0 == y, a == y
        for k in range(10):
            r = np.flatnonzero(y == k); dk = d[r]; bs = dk[idx[k]].mean(1)
            n01 = int((cor0[r] & ~cor1[r]).sum()); n10 = int((~cor0[r] & cor1[r]).sum()); pm = binom_two_sided(n01, n01 + n10)
            cell = dict(ds=float(dk.mean()), ci=ci(bs), p=p_z(float(dk.mean()), bs), right_to_wrong=n01, wrong_to_right=n10, p_mcnemar=pm,
                        call_change=float((a0[r] != a[r]).mean()))
            out['cells'][f'{mn}|{c}|{CLS[k]}'] = cell
            if 'perm' not in mn and mn.endswith('full'): fam_s.append((f'{mn}|{c}|{CLS[k]}', cell['p'])); fam_m.append((f'{mn}|{c}|{CLS[k]}', pm))
    for nm, fam in (('holm_ds', fam_s), ('holm_mcnemar', fam_m)):
        h = holm([p for _, p in fam]); out[nm] = {k: float(v) for (k, _), v in zip(fam, h)}
        log(f'{nm}: {len(fam)} tests, {sum(v < 0.05 for v in h)} with p_Holm < 0.05')
    # slopes dDs/dDf_ch pooled over fr05/10/20 (through origin), per model x class; and paired tests
    for mn in [m_ for m_ in nets if 'perm' not in m_] + ['ParT_full_perm1']:
        for k in range(10):
            r = np.flatnonzero(y == k)
            X = np.stack([dfch[c][r] for c in ('fr05', 'fr10', 'fr20')], 1); Y = np.stack([(S[(mn, c)][0] - S[(mn, 'base')][0])[r] for c in ('fr05', 'fr10', 'fr20')], 1)
            sl = lambda ii: (X[ii] * Y[ii]).sum((-1, -2)) / np.maximum((X[ii] ** 2).sum((-1, -2)), 1e-12)
            est = float(sl(np.arange(per)[None])[0]); bs = np.array([sl(ii[None])[0] for ii in idx[k][:B]])
            out['slopes'][f'{mn}|{CLS[k]}'] = dict(slope=est, ci=ci(bs), p=p_z(est, bs))
            for a_, b_ in (('fr10', 'frg10'), ('ch2nh2', 'ch2g2'), ('fr10', 'csame_fr10'), ('fr10', 'ckin_fr10'), ('gnh10', 'gph10'), ('gmix10', 'gnh10'),
                           ('gclus10', 'gmix10'), ('gstack10', 'gmix10')):
                if (mn, a_) not in S or (mn, b_) not in S: continue
                dd = (S[(mn, a_)][0] - S[(mn, b_)][0])[r]; bs = dd[idx[k]].mean(1)
                out['tests'][f'{mn}|{a_} - {b_}|{CLS[k]}'] = dict(diff=float(dd.mean()), ci=ci(bs), p=p_z(float(dd.mean()), bs))
    # perm band
    for c in rel_conds + gh_conds:
        v = [abs(out['cells'][f'ParT_full_perm{s}|{c}|{CLS[k]}']['ds']) for s in range(1, 7 if not smoke else 3) for k in range(10)
             if f'ParT_full_perm{s}|{c}|{CLS[k]}' in out['cells']]
        if v: out.setdefault('perm_max_abs', {})[c] = float(max(v))
    # sign agreement with GBDT reference
    agree = {}
    for mn in ('ParT_full', 'PN_full'):
        for c in rel_conds:
            sg = [(np.sign(out['cells'][f'{mn}|{c}|{CLS[k]}']['ds']), np.sign(REF[c][CLS[k]]['gbdt'])) for k in range(1, 10)]
            agree[f'{mn}|{c}'] = int(sum(a == b for a, b in sg))
    out['sign_agree_gbdt'] = agree
    for mn in ('ParT_full', 'PN_full', 'ParT_kin', 'PN_kin', 'ParT_full_perm1'):
        log(f'{mn} Ds per class [' + ' '.join(CLS) + ']')
        for c in rel_conds + gh_conds:
            if f'{mn}|{c}|QCD' in out['cells']:
                log(f'   {c:12s} ' + ' '.join(f'{out["cells"][f"{mn}|{c}|{CLS[k]}"]["ds"]:+.2f}' for k in range(10)))
    for c in rel_conds: log(f'GBDT ref {c:12s} ' + ' '.join(f'{REF[c][CLS[k]]["gbdt"]:+.2f}' for k in range(1, 10)))
    # example jet for the figure: an Hbb jet near median multiplicity
    rh = np.flatnonzero(y == 1); nb = J['mask'].sum(1); j = int(rh[np.argmin(np.abs(nb[rh] - np.median(nb[rh])))])
    from interp.part_core import kin
    k0 = [a[j] for a in kin(J)]; mm = J['mask'][j] > 0
    out['example'] = dict(j=j, pt=k0[0][mm].tolist(), deta=k0[1][mm].tolist(), dphi=k0[2][mm].tolist(),
                          type=np.where(J['isChargedHadron'][j][mm] > 0, 'ch', np.where(J['isPhoton'][j][mm] > 0, 'ph', np.where(J['isNeutralHadron'][j][mm] > 0, 'nh', 'lep'))).tolist(),
                          fr10=sets['fr10'][j][mm].tolist())
    s = '_smoke' if smoke else ''
    (OUT / f'h68_jc{s}.json').write_text(json.dumps(out, indent=1, default=float)); log(f'wrote h68_jc{s}.json')


# ================================================================================================================ run Top
def run_top(smoke):
    from interp import h65_pileup_dose as H
    H.SEEDS['top'] = SEED_TOP; H.PER['top'] = smoke or 1000
    T0, y, info, _ = H.load('top', smoke or 1000); T0 = H.finalize(T0, H.CAP['top'])[0]; W = T0['pt'].shape[1]
    G = H.ghosts(T0, 'top', SEED_TOP); rng = np.random.default_rng([SEED_TOP, 11]); Nj, K = G['q'].shape
    grp = np.arange(K) // 5; gaz = 2 * np.pi * rng.random((Nj, grp.max() + 1))[:, grp]
    rc = rng.uniform(0.03, 0.1, (Nj, K)); ac = gaz + rng.uniform(-0.3, 0.3, (Nj, K))
    m = G['on'] & (G['v'] < 10 / H.RHOMAX); scorer, _ = H.make_scorer('top', None); ae, ap = T0['ax_eta'], T0['ax_phi']
    conds = dict(base=T0, pu10=H.with_ghosts(T0, G['de'], G['dp'], G['q'], m),
                 c2_10=H.with_ghosts(T0, H.R_AX * np.cos(G['thax']), H.R_AX * np.sin(G['thax']), G['q'], m),
                 cl10=H.with_ghosts(T0, rc * np.cos(ac), rc * np.sin(ac), G['q'], m))
    S = {}; dm = {}
    from interp.h1_irc_panel import jet4
    from interp.h49_groomed_mass import _m4
    for c, U in conds.items():
        Uf = H.fin(U, W, H.CAP['top'])[0]; dm[c] = _m4(jet4(Uf)); L = scorer(Uf, ae, ap)
        for mn, l in L.items(): S[(mn, c)] = np.asarray(l, np.float64).ravel()
        log(f'top {c}: dm {np.mean(dm[c] - dm["base"]):+.2f} GeV; ' + ' '.join(f'{mn} {np.mean(S[(mn, c)] - S[(mn, "base")]):+.3f}' for mn in L))
    out = dict(sample=info['sample'], n=int(len(y)), dm={c: float((dm[c] - dm['base']).mean()) for c in conds}, cells={}, unedited={})
    idx = np.random.default_rng([SEED_TOP, 99]).integers(0, len(y), (B, len(y)))
    for mn in {k[0] for k in S}:
        out['unedited'][mn] = dict(auc=H.auc(S[(mn, 'base')], y))
        for c in ('pu10', 'c2_10', 'cl10'):
            d = S[(mn, c)] - S[(mn, 'base')]
            for lab, sel in (('top', y == 1), ('QCD', y == 0)):
                r = np.flatnonzero(sel); bs = d[r][idx[:, :len(r)] % len(r)].mean(1)
                out['cells'][f'{mn}|{c}|{lab}'] = dict(ds=float(d[r].mean()), ci=ci(bs), p=p_z(float(d[r].mean()), bs))
        log(f'top {mn}: AUC {out["unedited"][mn]["auc"]:.4f}; top-jet Ds pu10 / c2 / cl10 ' +
            ' / '.join(f'{out["cells"][f"{mn}|{c}|top"]["ds"]:+.2f}' for c in ('pu10', 'c2_10', 'cl10')))
    s = '_smoke' if smoke else ''
    (OUT / f'h68_top{s}.json').write_text(json.dumps(out, indent=1, default=float)); log(f'wrote h68_top{s}.json')


# ================================================================================================================ figure
def fig(smoke):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    s = '_smoke' if smoke else ''; D = json.loads((OUT / f'h68_jc{s}.json').read_text())
    f, ax = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw=dict(width_ratios=[1, 1.6]))
    e = D['example']; col = dict(ch='tab:blue', ph='tab:orange', nh='tab:green', lep='k')
    for t in col:
        ii = [i for i, tt in enumerate(e['type']) if tt == t and not e['fr10'][i]]
        ax[0].scatter([e['deta'][i] for i in ii], [e['dphi'][i] for i in ii], s=[8 + 30 * math.sqrt(e['pt'][i]) for i in ii], c=col[t], alpha=0.6,
                      label={'ch': 'charged hadron', 'ph': 'photon', 'nh': 'neutral hadron', 'lep': 'lepton'}[t])
    ii = [i for i in range(len(e['pt'])) if e['fr10'][i]]
    ax[0].scatter([e['deta'][i] for i in ii], [e['dphi'][i] for i in ii], s=[8 + 30 * math.sqrt(e['pt'][i]) for i in ii], facecolors='none',
                  edgecolors='tab:red', linewidths=1.5, label='charged -> neutral hadron (10 % of E)')
    ax[0].set_xlabel('Δη to jet axis'); ax[0].set_ylabel('Δφ to jet axis [rad]'); ax[0].set_title('H→bb jet: particles relabelled in edit fr10\n(four-momenta unchanged; marker area ∝ √pT)', fontsize=10)
    ax[0].legend(fontsize=8, loc='lower left'); ax[0].set_aspect('equal')
    cond = [('ph2nh', 'all photons -> neutral hadrons', 'tab:green'), ('nh2ph', 'all neutral hadrons -> photons', 'tab:orange'),
            ('fr10', '10 % of E: charged -> neutral hadron', 'tab:purple'), ('frg10', 'same particles: charged -> photon', 'tab:brown'),
            ('csame_fr10', 'control: charge sign swap, same particles', 'tab:gray')]
    x = np.arange(10); w = 0.16
    for i, (c, lab, cc) in enumerate(cond):
        v = np.array([D['cells'][f'ParT_full|{c}|{k}']['ds'] for k in CLS]); lo = np.array([D['cells'][f'ParT_full|{c}|{k}']['ci'] for k in CLS])
        ax[1].bar(x + (i - 2) * w, v, w, yerr=np.abs(lo.T - v), label=lab, color=cc, capsize=1)
    ax[1].axhline(0, c='k', lw=0.6); ax[1].set_xticks(x); ax[1].set_xticklabels(CLS, fontsize=9)
    ax[1].set_ylabel('Δ log-odds (own class vs QCD), mean over jets'); ax[1].set_title('ParT_full: relabel particle types, four-momenta fixed\n(1,000 jets per class, 95 % bootstrap CI)', fontsize=10)
    ax[1].legend(fontsize=8, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.08))
    f.tight_layout(); p = OUT / f'h68_particle_type{s}.png'; f.savefig(p, dpi=110); log('wrote', p.name)


if __name__ == '__main__':
    ap_ = argparse.ArgumentParser(); ap_.add_argument('--part', choices=['jc', 'top', 'fig'], required=True); ap_.add_argument('--smoke', type=int, default=0)
    a = ap_.parse_args()
    dict(jc=run_jc, top=run_top, fig=fig)[a.part](a.smoke)
