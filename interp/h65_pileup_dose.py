"""H65 (loop D, round D13; new D claim linking D21, D23 v2, D5): how do jet taggers' calls change with the density of uniform
soft contamination (pileup / underlying-event-like ghosts), compared with a plain-mass window, a groomed-mass (m_SD) window, an
area-subtracted-mass window, a multiplicity likelihood ratio LR(N) and an EFN, on all four datasets?

Edit ('pu<rho>'): ghost particles placed uniformly in the disc of radius R (the jet radius: 0.8 for hls4ml / JetClass / Top,
  0.4 for quark-gluon) about the stored jet axis, in (eta, phi). Each ghost is massless (JetClass: neutral hadron, d0 = dz = 0;
  QG: pid 130), with pT drawn uniformly in [0.5, 1] GeV. Their number is Poisson with mean rho * pi R^2 / 0.75 GeV, so the added
  pT per unit eta-phi area is rho, rho in {2, 5, 10, 20} GeV. One master set (rho = 20) is drawn per jet; the set at a lower rho
  keeps each ghost with probability rho / 20 (Poisson thinning), so the sets are nested and the same positions are used for
  every model. After adding, the particles are pT-sorted and cut to the model's slot limit (hls4ml 64, JetClass 128, Top 200,
  QG all), and every input is taken about the stored (unedited) axis, so the edit does not move the input axis.
Controls (same ghost pT values within each rho)
  c1_<rho>  one massless particle carrying the summed ghost pT, placed 0.002 from the axis: summed pT matched, count +1,
            mass change ~ 0 (a collinear addition)
  c2_<rho>  the same ghosts (same number, same pT values) all placed 0.002 from the axis: count and summed pT matched, mass ~ 0
  in<rho>   the same ghosts placed inside the soft-drop-kept core: uniformly within R_g / 2 of one of the two soft-drop prong
            centres (prong chosen at random per ghost); C/A clustering attaches them to a prong, soft drop keeps them
  out<rho>  the same ghosts placed only outside the core: uniform in the jet disc at > 1.1 R_g from both prong centres;
            C/A attaches them after the prongs merge and soft drop (z_cut 0.1, beta 0) removes them.
            The in / out pair is the H58v r_g split as a density scan (count and summed pT matched; plain-mass change larger
            for out). Valid on jets with a soft-drop split, R_g >= 0.02, both prong centres inside R and >= 5 % of the disc
            area outside the core.
Samples (random rows; seeds 6501-6504)
  hls  jedi_core.sample(2000, 6501) jets (h1_irc_panel.hls_sample), 10,000 jets, 5 classes
  jc   JetClass first test file per class, QCD (ZJetsToNuNu), Zqq, Wqq, Tbqq: 2,000 random rows each (seed 6502)
  top  Top Landscape test split, 2,000 top + 2,000 QCD random rows (seed 6503)
  qg   Komiske-Metodiev-Thaler Pythia test split (data_qg), 2,000 quark + 2,000 gluon random rows (seed 6504)
Models (nulls: weight-redrawn / weight-permuted)
  hls  JEDI-linear official n64 3-feature; JEDI-linear with redrawn (untrained) weights
  jc   ParT_full, ParT_kin, PN_full, PN_kin; ParT_full permuted (every weight tensor's entries permuted, seed 7); ParT_full untrained
  top  ParT (kin), PN (kin), LorentzNet, PFN, EFN; PFN permuted (h13_pileup_panel.TopModels)
  qg   part_kin, pnet_kin (H2/H5 weaver), pfn_k, efn_k (H2 energyflow), lorentznet (released); part_kin permuted
Scores: log-odds of signal class c vs QCD (hls: log p(c) - log(p(q) + p(g)); jc: logit_c - logit_QCD; top: top log-odds;
  qg: log p(quark) - log p(gluon)).
References, computed on the particles the model sees (after the slot cut), all at 50 % signal efficiency on unedited jets:
  win_m     plain jet mass m in the shortest interval containing 50 % of unedited signal jets (QG: m < median quark m)
  win_msd   soft-drop mass (C/A, z_cut 0.1, beta 0) window, same construction
  win_msub  area-subtracted mass m_sub = |p_J - rho_seen A_mu| (4-vector area subtraction, Soyez-style, rho_m = 0): A_mu = the
            4-momentum of a uniform unit-pT-density disc of radius R about the stored axis; rho_seen = rho x (ghost pT kept by
            the slot cut / ghost pT added), i.e. the true density of the ghosts present in the input
  LR_N      histogram likelihood ratio of the particle count N the model sees (2-fold cross-fit on the unedited sample)
  EFN       (Top, QG) the trained IRC-safe energy flow network
Metrics (--part summary): per tagger x class x condition, the fraction of calls changed (call = score above the unedited 50 %
  signal-efficiency threshold, on signal + QCD jets of the task) and the mean delta log-odds, 95 % bootstrap intervals over jets
  (B = 1000, the same resamples for every tagger); slope vs rho through the origin (per GeV of rho); pu - c1 and pu - c2 at each
  rho (paired bootstrap; McNemar exact binomial p on discordant call changes); r_core(rho) = calls changed by 'out' / by 'in';
  nearest reference by L2 distance between call-change curves over rho in {2, 5, 10, 20} (bootstrap frequency); Holm over
  models x rho x classes (all datasets) per family.
Run
  ./job.sh d-h65-hls 1 4 'python -m interp.h65_pileup_dose --part hls'                       (jax env: obs + JEDI)
  ./part_job.sh d-h65-gpu 'python -m interp.h65_pileup_dose --part nets --ds jc && ... --ds top && ... --ds qg'
  AI_PY=conda ./job.sh d-h65-obs 1 4 'python -m interp.h65_pileup_dose --part obs --ds top && ... --ds qg &&
      JETDISTILL_TAGGER=part JETDISTILL_DATA=/j-jepa-vol/aaron-interp/code/cluster/data_part python -m interp.h65_pileup_dose --part obs --ds jc'
  ./job.sh d-h65-sum 1 4 'python -m interp.h65_pileup_dose --part summary'
  -> $AI_RESULTS/physics/h65_<ds>_{obs,nets}.npz / .json, h65_summary.json, h65_pileup.png ; --smoke N: N jets per class
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
from interp.h1_irc_panel import finalize, jet4, wrap, holm, binom_two_sided  # noqa: E402
from interp.h18_softcount_cross import nreal, auc, logsm  # noqa: E402
from interp.h49_groomed_mass import add_parts, softdrop, _m4  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
OUT = RES / 'physics'
SEEDS = dict(hls=6501, jc=6502, top=6503, qg=6504)
PER = dict(hls=2000, jc=2000, top=2000, qg=2000)
RHOS = (2, 5, 10, 20)
RHOMAX = 20.0
QLO, QHI = 0.5, 1.0
RJ = dict(hls=0.8, jc=0.8, top=0.8, qg=0.4)
CAP = dict(hls=64, jc=128, top=200, qg=400)
R_AX = 0.002
SIG = dict(hls={'W': 2, 'Z': 3, 't': 4}, jc={'W': 7, 'Z': 6, 't': 8}, top={'t': 1}, qg={'q': 1})
QCD = dict(hls=[0, 1], jc=[0], top=[0], qg=[0])
EDITS = [f'{e}{r}' for r in RHOS for e in ('pu', 'c1_', 'c2_', 'in', 'out')]
CONDS = ['base'] + EDITS
NULLS = {'JEDI_redrawn', 'ParT_full_permuted', 'ParT_full_untrained', 'PFN_permuted', 'part_kin_perm'}
EFN = dict(top='EFN', qg='efn_k')
KNOWN = dict(JEDI='5-class acc 0.8092 (260k test); 0.8091 (H49)', JEDI_redrawn='0.200 (chance)',
             ParT_full='per class (H18) QCD 0.783 Zqq 0.689 Wqq 0.805 Tbqq 0.956', ParT_kin='QCD 0.718 Zqq 0.550 Wqq 0.702 Tbqq 0.912',
             PN_full='QCD 0.764 Zqq 0.652 Wqq 0.784 Tbqq 0.949', PN_kin='QCD 0.693 Zqq 0.528 Wqq 0.680 Tbqq 0.913',
             ParT='AUC 0.9858', PN='AUC 0.9858', LNet='AUC 0.9870', PFN='AUC 0.9819', EFN='AUC 0.975-0.979',
             part_kin='AUC 0.9013', pnet_kin='AUC 0.9015', lorentznet='AUC 0.917', pfn_k='AUC ~0.88-0.90', efn_k='AUC 0.8835')
T0_ = time.time()


def log(*a):
    print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


def sfx(smoke):
    return '_smoke' if smoke else ''


# ===================================================================================================================== samples
def load(ds, smoke):
    per = smoke or PER[ds]; seed = SEEDS[ds]
    if ds == 'hls':
        from interp.h1_irc_panel import hls_sample
        T, y = hls_sample(per, seed)
        return T, y, dict(sample=f'jets of jedi_core.sample(per_class={per}, seed={seed})'), None
    if ds == 'jc':
        from interp.h49_groomed_mass import jc_sample4
        from interp.h1_irc_panel import jc_table
        J0 = jc_sample4(per, seed); y = J0['label'].astype(int)
        return jc_table(J0), y, dict(sample=f'JetClass first test file per class; QCD, Zqq, Wqq, Tbqq {per} random rows each, seed {seed}'), J0
    from interp.h12_boost_panel import table_from_full
    if ds == 'top':
        from interp import data_top as D
    else:
        from interp import data_qg as D
    d = D.DST / 'test'; ylab = np.load(d / 'label.npy'); rng = np.random.default_rng(seed)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(ylab == c), per, replace=False) for c in (0, 1)]))
    F = np.asarray(np.load(d / 'full.npy', mmap_mode='r')[idx]); jet = np.asarray(np.load(d / 'jet.npy', mmap_mode='r')[idx])
    nn = np.minimum(jet[:, 5].astype(np.int64), F.shape[1]); y = ylab[idx].astype(int); T = table_from_full(F, nn); del F
    if ds == 'qg':
        T['m'] = np.zeros_like(T['m']); pid = np.asarray(np.load(d / 'pid.npy', mmap_mode='r')[idx]).astype(np.float64)
        T['ext'] = dict(pid=pid * (T['pt'] > 0))
    T['sgn'] = np.where(T['ax_eta'] < 0, -1.0, 1.0)
    log(f'{ds}: {len(y)} test rows (seed {seed}), labels {np.bincount(y).tolist()}')
    return T, y, dict(sample=f'{ds} test split, {per} per class random rows (default_rng {seed})', idx=idx.tolist()), None


# ===================================================================================================================== edits
def ghosts(T0, ds, seed):
    """master ghost set (rho = 20) per jet plus thinning numbers, axis azimuths for c1 / c2"""
    J = len(T0['pt']); R = RJ[ds]; A = math.pi * R * R; rng = np.random.default_rng([seed, 1])
    n = rng.poisson(RHOMAX * A / (0.5 * (QLO + QHI)), J); K = int(n.max())
    on = np.arange(K)[None, :] < n[:, None]
    r = R * np.sqrt(rng.random((J, K))); th = 2 * np.pi * rng.random((J, K))
    G = dict(on=on, de=r * np.cos(th), dp=r * np.sin(th), q=rng.uniform(QLO, QHI, (J, K)), v=rng.random((J, K)),
             thax=2 * np.pi * rng.random((J, K)), n20=n)
    log(f'{ds}: ghosts at rho 20: mean count {n.mean():.1f} (expected {RHOMAX * A / 0.75:.1f}), max {K}')
    return G


def core_positions(G, T0, sd, ds, seed):
    """in-core / out-of-core ghost positions (same ghosts, positions only) and the C3 validity mask"""
    J, K = G['q'].shape; R = RJ[ds]; ae, ap = T0['ax_eta'], T0['ax_phi']; rng = np.random.default_rng([seed, 3])
    pe = lambda v: (np.arcsinh(v[:, 2] / np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-300)), np.arctan2(v[:, 1], v[:, 0]))
    e1, f1 = pe(sd['v1']); e2, f2 = pe(sd['v2'])
    c1 = np.stack([e1 - ae, wrap(f1 - ap)], 1); c2 = np.stack([e2 - ae, wrap(f2 - ap)], 1)
    Rg = np.hypot(*(c1 - c2).T)
    pr = rng.random((J, K)) < 0.5; rr = 0.5 * Rg[:, None] * np.sqrt(rng.random((J, K))); th = 2 * np.pi * rng.random((J, K))
    cen = np.where(pr[..., None], c1[:, None, :], c2[:, None, :])
    de_in = cen[..., 0] + rr * np.cos(th); dp_in = cen[..., 1] + rr * np.sin(th)
    ok = sd['found'] & (Rg >= 0.02) & (np.hypot(*c1.T) < R) & (np.hypot(*c2.T) < R)
    de_out = G['de'].copy(); dp_out = G['dp'].copy(); frac = np.zeros(J)
    for j in np.flatnonzero(ok):
        need = int(G['n20'][j]); got_e, got_p = [], []; drawn = acc = 0
        for _ in range(60):
            m = max(4 * need, 64); r = R * np.sqrt(rng.random(m)); t = 2 * np.pi * rng.random(m); x, z = r * np.cos(t), r * np.sin(t)
            g = (np.hypot(x - c1[j, 0], z - c1[j, 1]) > 1.1 * Rg[j]) & (np.hypot(x - c2[j, 0], z - c2[j, 1]) > 1.1 * Rg[j])
            drawn += m; acc += int(g.sum()); got_e.extend(x[g].tolist()); got_p.extend(z[g].tolist())
            if len(got_e) >= need: break
        frac[j] = acc / max(drawn, 1)
        if len(got_e) >= need and frac[j] >= 0.05:
            de_out[j, :need] = got_e[:need]; dp_out[j, :need] = got_p[:need]
        else:
            ok[j] = False
    ok &= frac >= 0.05
    G.update(de_in=de_in, dp_in=dp_in, de_out=de_out, dp_out=dp_out, c3ok=ok, Rg=Rg, zg=sd['zg'], out_area_frac=frac)
    log(f'{ds}: C3 valid on {ok.mean():.3f} of jets; median R_g {np.median(Rg[ok]) if ok.any() else float("nan"):.3f}')


def with_ghosts(T0, de, dp, q, mask):
    ae, ap = T0['ax_eta'], T0['ax_phi']
    return add_parts(T0, np.where(mask, q, 0.0), ae[:, None] + de, ap[:, None] + dp)


def conditions(T0, G):
    yield 'base', T0, np.zeros(len(T0['pt'])), np.zeros(len(T0['pt']))
    for r in RHOS:
        m = G['on'] & (G['v'] < r / RHOMAX); Q = (G['q'] * m).sum(1); nG = m.sum(1)
        yield f'pu{r}', with_ghosts(T0, G['de'], G['dp'], G['q'], m), Q, nG
        a = G['thax'][:, :1]
        yield f'c1_{r}', with_ghosts(T0, R_AX * np.cos(a), R_AX * np.sin(a), Q[:, None], (Q > 0)[:, None]), Q, (Q > 0).astype(float)
        yield f'c2_{r}', with_ghosts(T0, R_AX * np.cos(G['thax']), R_AX * np.sin(G['thax']), G['q'], m), Q, nG
        yield f'in{r}', with_ghosts(T0, G['de_in'], G['dp_in'], G['q'], m), Q, nG
        yield f'out{r}', with_ghosts(T0, G['de_out'], G['dp_out'], G['q'], m), Q, nG


def fin(U, W, cap):
    """slot cut as finalize does; returns (table, ghosts kept count, ghost pT kept) (appended columns >= W are the added ones)"""
    o = np.argsort(-U['pt'], 1, kind='stable')[:, :cap]; pt = np.take_along_axis(U['pt'], o, 1)
    isg = (o >= W) & (pt > 0)
    return finalize(U, cap)[0], isg.sum(1), (pt * isg).sum(1)


def area4(T0, ds, nr=24, nt=48):
    """4-momentum of a unit-pT-density uniform disc of radius R about the stored axis (A_mu per GeV of rho)"""
    R = RJ[ds]; ed = np.linspace(0, R, nr + 1); rc = 0.5 * (ed[1:] + ed[:-1]); th = (np.arange(nt) + 0.5) * 2 * np.pi / nt
    w = np.repeat(0.5 * (ed[1:] ** 2 - ed[:-1] ** 2) * (2 * np.pi / nt), nt)
    de = (rc[:, None] * np.cos(th)[None]).ravel(); dp = (rc[:, None] * np.sin(th)[None]).ravel()
    eta = T0['ax_eta'][:, None] + de[None]; phi = T0['ax_phi'][:, None] + dp[None]
    return np.stack([(w * np.cos(phi)).sum(1), (w * np.sin(phi)).sum(1), (w * np.sinh(eta)).sum(1), (w * np.cosh(eta)).sum(1)], 1)


def rho_of(name):
    return 0.0 if name == 'base' else float(name.replace('pu', '').replace('c1_', '').replace('c2_', '').replace('in', '').replace('out', ''))


# ===================================================================================================================== scorers
def make_scorer(ds, J0):
    if ds == 'hls':
        from interp.jedi_core import Jedi
        from interp.h13_pileup_panel import hls_in
        nets = dict(JEDI=Jedi(64, '3-feature'), JEDI_redrawn=Jedi(64, '3-feature', untrained=True, seed=0))
        return lambda Uf, ae, ap: {k: np.asarray(n.run(hls_in(Uf, ae, ap))['logits'], np.float64) for k, n in nets.items()}, {}
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; log('device', dev)
    if ds == 'jc':
        from interp.part_core import ParTNetwork, inputs
        from interp.p1v_verify_part import fwd
        from interp.pn_net import load_particlenet
        from interp.h1_irc_panel import pn_logits
        from interp.h13_pileup_panel import jc_maker
        from jetdistill.config import MODELS
        nets = {}
        for n in ('full', 'kin'):
            net = ParTNetwork(n); nets[f'ParT_{n}'] = (lambda J, net=net, n=n: np.asarray(fwd(net, *inputs(J, n))[1], np.float64))
        netp = ParTNetwork('full'); g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for p in netp.model.parameters():
                if p.numel() > 1:
                    f = p.data.flatten().cpu(); p.data.copy_(f[torch.randperm(f.numel(), generator=g)].view_as(p).to(p.device))
        nets['ParT_full_permuted'] = (lambda J, net=netp: np.asarray(fwd(net, *inputs(J, 'full'))[1], np.float64))
        netu = ParTNetwork('full', untrained=True, seed=0)
        nets['ParT_full_untrained'] = (lambda J, net=netu: np.asarray(fwd(net, *inputs(J, 'full'))[1], np.float64))
        for n, dims in (('full', 17), ('kin', 7)):
            pn = load_particlenet(Path(MODELS).parent / 'particlenet' / f'ParticleNet_{n}.pt', dims, os.environ.get('JETDISTILL_DEVICE', dev))
            nets[f'PN_{n}'] = (lambda J, pn=pn, n=n: pn_logits(pn, J, n))
        make = jc_maker(J0)

        def sc(Uf, ae, ap):
            J = make(Uf, ae, ap); return {mn: f(J) for mn, f in nets.items()}
        return sc, {}
    if ds == 'top':
        from interp.h13_pileup_panel import TopModels, top_J
        Mo = TopModels(dev); st = {}

        def sc(Uf, ae, ap):
            if 'order' not in st: st['order'] = np.random.default_rng(SEEDS['top'] + 7).permutation(len(ae))
            return Mo.score(top_J(Uf, ae, ap), st['order'])
        return sc, {}
    from interp import h36_axis_top_qg  # noqa: F401  (patches h5_qg_counting.kinematics: inputs about Jt['_cond'] axis)
    from interp import h5_qg_counting as H5
    from interp import h14_qg_generator as H14
    M = H5.Models(dev)
    for name, arch in (('part_kin', 'part'), ('pnet_kin', 'pn')): M.add_weaver(name, arch, False)
    for kind in ('pfn', 'efn'): H14.add_keras_ef(M, kind)
    M.add_lnet(); M.add_permuted('part_kin_perm', 'part_kin')
    names = list(M.m); log('QG models', names)

    def sc(Uf, ae, ap):
        real = Uf['pt'] > 0
        X = dict(pt=Uf['pt'], eta=Uf['eta'], phi=Uf['phi'], pid=(Uf['ext']['pid'] * real).astype(np.int64), n=real.sum(1),
                 _cond=('ax', np.asarray(ae), np.asarray(ap)))
        return {k: -np.asarray(M.logit(k, X), np.float64) for k in names}      # log p(q) - log p(g)
    return sc, M.info


def class_scores(L, ds):
    L = np.asarray(L, np.float64)
    if L.ndim == 1: return {list(SIG[ds])[0]: L}
    ls = logsm(L); qcd = np.logaddexp.reduce(np.stack([ls[:, k] for k in QCD[ds]], 1), axis=1)
    return {c: ls[:, k] - qcd for c, k in SIG[ds].items()}


# ===================================================================================================================== run
def run(ds, part, smoke):
    OUT.mkdir(parents=True, exist_ok=True)
    T0, y, info, J0 = load(ds, smoke); T0 = finalize(T0, CAP[ds])[0]; W = T0['pt'].shape[1]
    log(f'{ds}: {len(y)} jets, slot limit {CAP[ds]}, width {W}, <N> {nreal(T0).mean():.1f}, at limit {(nreal(T0) >= CAP[ds]).mean():.3f}')
    t = time.time(); sd0 = softdrop(T0); log(f'{ds}: base soft drop ({time.time() - t:.0f} s), split found {sd0["found"].mean():.3f}')
    G = ghosts(T0, ds, SEEDS[ds]); core_positions(G, T0, sd0, ds, SEEDS[ds])
    do_obs = part in ('obs', 'hls'); do_nets = part in ('nets', 'hls')
    scorer, minfo = make_scorer(ds, J0) if do_nets else (None, {})
    A4 = area4(T0, ds) if do_obs else None
    ae, ap = T0['ax_eta'], T0['ax_phi']
    OBS = dict(y=y, c3ok=G['c3ok'], Rg=G['Rg'], zg=G['zg'], out_area_frac=G['out_area_frac'], n20=G['n20'])
    NETS = dict(y=y); un = {}
    for name, U, Q, nG in conditions(T0, G):
        t = time.time(); Uf, gk, gpt = fin(U, W, CAP[ds]); n = nreal(Uf)
        OBS[f'n|{name}'] = n.astype(np.int16); NETS[f'n|{name}'] = n.astype(np.int16)
        OBS[f'gadd|{name}'] = nG.astype(np.int16); OBS[f'gkept|{name}'] = gk.astype(np.int16)
        OBS[f'qadd|{name}'] = Q.astype(np.float32); OBS[f'qkept|{name}'] = gpt.astype(np.float32)
        msg = f'{ds} {name}: ghosts added {nG.mean():.1f} kept {gk.mean():.1f}, pT added {Q.mean():.2f} kept {gpt.mean():.2f} GeV'
        if do_obs:
            P = jet4(Uf); rs = rho_of(name) * np.where(Q > 0, gpt / np.maximum(Q, 1e-12), 0.0)
            OBS[f'm|{name}'] = _m4(P).astype(np.float32); OBS[f'msub|{name}'] = _m4(P - rs[:, None] * A4).astype(np.float32)
            OBS[f'msd|{name}'] = softdrop(Uf)['msd'].astype(np.float32); OBS[f'sumpt|{name}'] = Uf['pt'].sum(1).astype(np.float32)
            msg += (f' | dm {np.mean(OBS[f"m|{name}"] - OBS["m|base"]):+.2f} dmsd {np.mean(OBS[f"msd|{name}"] - OBS["msd|base"]):+.2f}'
                    f' dmsub {np.mean(OBS[f"msub|{name}"] - OBS["msub|base"]):+.2f} GeV')
        if do_nets:
            S = scorer(Uf, ae, ap)
            for mn, L in S.items():
                L = np.asarray(L, np.float64)
                for cls, s in class_scores(L, ds).items(): NETS[f's|{mn}|{cls}|{name}'] = s.astype(np.float32)
                if name == 'base':
                    un[mn] = {}
                    if L.ndim == 2:
                        a = L.argmax(1)
                        if ds == 'hls': un[mn]['acc_5class'] = float((a == y).mean())
                        un[mn]['per_class_acc'] = {str(k): float((a[y == k] == k).mean()) for k in sorted(set(y.tolist()))}
                    for cls, s in class_scores(L, ds).items():
                        rows = np.isin(y, [SIG[ds][cls]] + QCD[ds]); ys = (y[rows] == SIG[ds][cls]).astype(int)
                        un[mn][f'AUC_{cls}_vs_QCD'] = auc(s[rows], ys); un[mn][f'acc_{cls}_s>0'] = float(((s[rows] > 0) == ys).mean())
                    log(f'{ds} unedited {mn}: {un[mn]}  known: {KNOWN.get(mn, "-")}')
            c0 = list(SIG[ds])[0]
            msg += ' | ' + ' '.join(f'{mn} {np.mean(NETS[f"s|{mn}|{c0}|{name}"] - NETS[f"s|{mn}|{c0}|base"]):+.3f}' for mn in S)
        log(msg + f' ({time.time() - t:.0f} s)')
        if do_obs and ds == 'jc' and name == 'pu10':
            kw = SIG['jc']['W']; rr = np.flatnonzero((y == kw) & G['c3ok']); nb = nreal(T0)
            j = int(rr[np.argmin(np.abs(nb[rr] - np.median(nb[rr])) + 0.01 * np.abs(OBS['m|base'][rr] - 80.4))])
            b = T0['pt'][j] > 0; m = G['on'][j] & (G['v'][j] < 10 / RHOMAX)
            OBS.update({'ex|j': np.array(j), 'ex|pt': T0['pt'][j, b], 'ex|deta': T0['eta'][j, b] - ae[j], 'ex|dphi': wrap(T0['phi'][j, b] - ap[j]),
                        'ex|gpt': G['q'][j, m], 'ex|gdeta': G['de'][j, m], 'ex|gdphi': G['dp'][j, m]})
    s = sfx(smoke)
    if do_obs:
        np.savez_compressed(OUT / f'h65_{ds}_obs{s}.npz', **OBS)
        (OUT / f'h65_{ds}_obs{s}.json').write_text(json.dumps(dict(info, n=int(len(y))), indent=1, default=float))
        log(f'wrote h65_{ds}_obs{s}.npz')
    if do_nets:
        np.savez_compressed(OUT / f'h65_{ds}_nets{s}.npz', **NETS)
        (OUT / f'h65_{ds}_nets{s}.json').write_text(json.dumps(dict(info, n=int(len(y)), unedited=un, models=minfo,
                                                                     known={k: KNOWN.get(k, '-') for k in un}), indent=1, default=str))
        log(f'wrote h65_{ds}_nets{s}.npz')


# ===================================================================================================================== statistics
def shortest50(x):
    xs = np.sort(x); n = len(xs); k = int(math.ceil(0.5 * n)); w = xs[k - 1:] - xs[:n - k + 1]; i = int(np.argmin(w))
    return float(xs[i]), float(xs[i + k - 1])


def lr_n(nb, ycls, seed):
    """2-fold cross-fit histogram log-likelihood ratio of N in 20 quantile bins of the unedited N (both classes); an N beyond
    the outermost bin edge takes the outermost bin's value (constant extrapolation); returns a function of N arrays"""
    fold = np.random.default_rng([seed, 21]).permutation(len(nb)) % 2
    edges = np.unique(np.quantile(nb, np.linspace(0, 1, 21))[1:-1]); tabs = {}
    for f in (0, 1):
        tr = fold != f; b = np.searchsorted(edges, nb[tr], side='right'); nbins = len(edges) + 1
        hs = np.bincount(b[ycls[tr]], minlength=nbins) + 0.5; hb = np.bincount(b[~ycls[tr]], minlength=nbins) + 0.5
        tabs[f] = np.log(hs / hs.sum()) - np.log(hb / hb.sum())
    return lambda n: np.where(fold == 0, tabs[0][np.searchsorted(edges, n, side='right')], tabs[1][np.searchsorted(edges, n, side='right')])


def ci(v):
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else [None, None]


def p_boot(v):
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]; B = len(v)
    if not B: return None
    return float(min(1.0, 2 * min(((v <= 0).sum() + 1) / (B + 1), ((v >= 0).sum() + 1) / (B + 1))))


def p_z(est, v):
    """two-sided normal-approximation p for H0: effect = 0, from the estimate and its bootstrap standard error"""
    from math import erfc, sqrt
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]; se = v.std(ddof=1) if v.size > 2 else 0.0
    if se <= 0: return 1.0 if est == 0 else 0.0
    return float(erfc(abs(est) / se / sqrt(2)))


def summary(smoke, B=1000):
    s = sfx(smoke); RESU = dict(datasets={}, conds=CONDS, rhos=list(RHOS)); fam = dict(pu=[], ctrl=[])
    for ds in ('hls', 'jc', 'top', 'qg'):
        fo, fn = OUT / f'h65_{ds}_obs{s}.npz', OUT / f'h65_{ds}_nets{s}.npz'
        if not (fo.exists() and fn.exists()): log(f'{ds}: missing {fo.name} or {fn.name}'); continue
        O = dict(np.load(fo)); S = dict(np.load(fn)); y = O['y']
        assert (S['y'] == y).all()
        mism = {c: int((O[f'n|{c}'] != S[f'n|{c}']).sum()) for c in CONDS}
        log(f'{ds}: N-alignment mismatches obs vs nets per condition: {sum(mism.values())}')
        jinfo = json.loads((OUT / f'h65_{ds}_nets{s}.json').read_text())
        models = sorted({k.split('|')[1] for k in S if k.startswith('s|')})
        D = dict(n=int(len(y)), unedited=jinfo['unedited'], known=jinfo.get('known'), alignment_mismatch=mism, classes={},
                 c3_valid_frac=float(O['c3ok'].mean()), ghosts={})
        for c in CONDS:
            D['ghosts'][c] = dict(added_n=float(O[f'gadd|{c}'].mean()), kept_n=float(O[f'gkept|{c}'].mean()),
                                  added_pt=float(O[f'qadd|{c}'].mean()), kept_pt=float(O[f'qkept|{c}'].mean()))
        for cls, k in SIG[ds].items():
            rows = np.isin(y, [k] + QCD[ds]); sig = (y == k)[rows]; nr = int(rows.sum()); c3 = O['c3ok'][rows]
            idx = np.random.default_rng([SEEDS[ds], 99, k]).integers(0, nr, (B, nr))
            jit = np.random.default_rng([SEEDS[ds], 98]).random(nr) * 1e-6
            taggers = {}
            for mn in models:
                sc = {c: S[f's|{mn}|{cls}|{c}'][rows].astype(np.float64) for c in CONDS}
                taggers[mn] = ('net', sc)
            fnlr = lr_n(O['n|base'][rows].astype(int), sig, SEEDS[ds])
            taggers['LR_N'] = ('lr', {c: fnlr(O[f'n|{c}'][rows].astype(int)) + jit for c in CONDS})
            for ob in ('m', 'msd', 'msub'):
                X = {c: O[f'{ob}|{c}'][rows].astype(np.float64) for c in CONDS}
                if ds == 'qg':
                    t = float(np.median(X['base'][sig])); taggers[f'win_{ob}'] = ('win', {c: X[c] < t for c in CONDS}, (None, t))
                else:
                    lo, hi = shortest50(X['base'][sig]); taggers[f'win_{ob}'] = ('win', {c: (X[c] >= lo) & (X[c] <= hi) for c in CONDS}, (lo, hi))
            C = dict(n_rows=nr, n_sig=int(sig.sum()), n_c3=int(c3.sum()), taggers={}, observables={})
            for c in CONDS:
                sel = c3 if (c.startswith('in') or c.startswith('out')) else np.ones(nr, bool)
                C['observables'][c] = {f'd{ob}_mean': float((O[f'{ob}|{c}'][rows] - O[f'{ob}|base'][rows])[sel].mean()) for ob in ('m', 'msd', 'msub', 'n')}
                C['observables'][c].update({f'd{ob}_mean_sig': float((O[f'{ob}|{c}'][rows] - O[f'{ob}|base'][rows])[sel & sig].mean()) for ob in ('m', 'msd', 'msub')})
            curves = {}; effc = {}
            for tn, tv in taggers.items():
                kind = tv[0]
                if kind == 'win':
                    calls = tv[1]; dsc = None; wp = tv[2]
                else:
                    sc = tv[1]; thr = float(np.median(sc['base'][sig])); calls = {c: sc[c] > thr for c in CONDS}
                    dsc = {c: sc[c] - sc['base'] for c in CONDS}; wp = thr
                R = dict(kind=kind, wp=wp, effS0=float(calls['base'][sig].mean()), effB0=float(calls['base'][~sig].mean()), cond={})
                chg_b = {}; ds_b = {}; eS_b = {}
                for c in EDITS:
                    sel = c3 if (c.startswith('in') or c.startswith('out')) else np.ones(nr, bool)
                    ch = (calls[c] != calls['base']).astype(np.float64)
                    cb = (ch * sel)[idx].sum(1) / np.maximum(sel[idx].sum(1), 1); chg_b[c] = cb
                    ss = (sig & sel).astype(np.float64); eS_b[c] = (calls[c] * ss)[idx].sum(1) / np.maximum(ss[idx].sum(1), 1)
                    e = dict(chg=float(ch[sel].mean()), chg_ci=ci(cb), effS=float(calls[c][sig & sel].mean()), effB=float(calls[c][~sig & sel].mean()))
                    if dsc is not None:
                        d = dsc[c]; db = (d * sel)[idx].sum(1) / np.maximum(sel[idx].sum(1), 1); ds_b[c] = db
                        e.update(ds=float(d[sel].mean()), ds_ci=ci(db), ds_p_boot=p_boot(db), ds_p=p_z(float(d[sel].mean()), db),
                                 ds_sig=float(d[sel & sig].mean()), ds_bkg=float(d[sel & ~sig].mean()))
                        if kind == 'net' and c.startswith('pu') and tn not in NULLS:
                            fam['pu'].append((f'{ds}|{cls}|{tn}|{c}', e['ds_p']))
                    R['cond'][c] = e
                x = np.array(RHOS, float)
                M = np.stack([chg_b[f'pu{r}'] for r in RHOS], 1); curves[tn] = M
                effc[tn] = np.stack([eS_b[f'pu{r}'] for r in RHOS], 1)
                R['chg_slope_per_GeV'] = float((x * np.array([R['cond'][f'pu{r}']['chg'] for r in RHOS])).sum() / (x * x).sum())
                R['chg_slope_ci'] = ci((M * x).sum(1) / (x * x).sum())
                if dsc is not None:
                    Md = np.stack([ds_b[f'pu{r}'] for r in RHOS], 1)
                    R['ds_slope_per_GeV'] = float((x * np.array([R['cond'][f'pu{r}']['ds'] for r in RHOS])).sum() / (x * x).sum())
                    R['ds_slope_ci'] = ci((Md * x).sum(1) / (x * x).sum())
                # network minus controls, r_core
                R['vs_ctrl'] = {}; R['r_core'] = {}
                for r in RHOS:
                    chp = calls[f'pu{r}'] != calls['base']
                    for cc in ('c1_', 'c2_'):
                        chc = calls[f'{cc}{r}'] != calls['base']; b_ = int((chp & ~chc).sum()); c_ = int((~chp & chc).sum())
                        p = binom_two_sided(b_, b_ + c_) if b_ + c_ else 1.0
                        db = chg_b[f'pu{r}'] - chg_b[f'{cc}{r}']
                        e = dict(dchg=float(chp.mean() - chc.mean()), dchg_ci=ci(db), mcnemar_b=b_, mcnemar_c=c_, p=float(p))
                        if dsc is not None:
                            dd = ds_b[f'pu{r}'] - ds_b[f'{cc}{r}']; est = float((dsc[f'pu{r}'] - dsc[f'{cc}{r}']).mean())
                            e.update(dds=est, dds_ci=ci(dd), dds_p=p_z(est, dd))
                        R['vs_ctrl'][f'{cc}{r}'] = e
                        if kind == 'net' and tn not in NULLS: fam['ctrl'].append((f'{ds}|{cls}|{tn}|pu{r}-{cc}{r}', float(p)))
                    ci_, co_ = R['cond'][f'in{r}']['chg'], R['cond'][f'out{r}']['chg']
                    rb = chg_b[f'out{r}'] / np.where(chg_b[f'in{r}'] > 0, chg_b[f'in{r}'], np.nan)
                    R['r_core'][str(r)] = dict(r=(co_ / ci_ if ci_ > 0 else None), ci=ci(rb), chg_in=ci_, chg_out=co_)
                    if dsc is not None:
                        di, do = R['cond'][f'in{r}']['ds'], R['cond'][f'out{r}']['ds']
                        rbs = ds_b[f'out{r}'] / np.where(np.abs(ds_b[f'in{r}']) > 1e-9, ds_b[f'in{r}'], np.nan)
                        R['r_core'][str(r)].update(r_ds=(do / di if abs(di) > 1e-9 else None), r_ds_ci=ci(rbs), ds_in=di, ds_out=do)
                C['taggers'][tn] = R
            refs = ['win_m', 'win_msd', 'win_msub', 'LR_N'] + ([EFN[ds]] if ds in EFN else [])
            for tn in taggers:
                if taggers[tn][0] != 'net' or tn == EFN.get(ds): continue
                dist = {rf: np.sqrt(((curves[tn] - curves[rf]) ** 2).sum(1)) for rf in refs}
                pt_ = {rf: float(np.sqrt(sum((C['taggers'][tn]['cond'][f'pu{r}']['chg'] - C['taggers'][rf]['cond'][f'pu{r}']['chg']) ** 2 for r in RHOS))) for rf in refs}
                Dm = np.stack([dist[rf] for rf in refs], 1); am = Dm.argmin(1)
                C['taggers'][tn]['nearest'] = dict(dist=pt_, dist_ci={rf: ci(dist[rf]) for rf in refs}, nearest=min(pt_, key=pt_.get),
                                                   boot_freq={rf: float((am == i).mean()) for i, rf in enumerate(refs)})
                # same on the signal-efficiency curve eps_S(rho) (fraction of signal jets still called signal; does not saturate)
                distE = {rf: np.sqrt(((effc[tn] - effc[rf]) ** 2).sum(1)) for rf in refs}
                ptE = {rf: float(np.sqrt(sum((C['taggers'][tn]['cond'][f'pu{r}']['effS'] - C['taggers'][rf]['cond'][f'pu{r}']['effS']) ** 2 for r in RHOS))) for rf in refs}
                amE = np.stack([distE[rf] for rf in refs], 1).argmin(1)
                C['taggers'][tn]['nearest_effS'] = dict(dist=ptE, dist_ci={rf: ci(distE[rf]) for rf in refs}, nearest=min(ptE, key=ptE.get),
                                                        boot_freq={rf: float((amE == i).mean()) for i, rf in enumerate(refs)})
            D['classes'][cls] = C
            log(f'{ds} {cls}: ' + '; '.join(f'{tn} chg pu ' + '/'.join(f'{C["taggers"][tn]["cond"][f"pu{r}"]["chg"]:.3f}' for r in RHOS) for tn in C['taggers']))
        RESU['datasets'][ds] = D
    for f, L in fam.items():
        if not L: continue
        ps = holm([p if p is not None else 1.0 for _, p in L])
        RESU[f'holm_{f}'] = {k: dict(p=p, p_holm=float(h)) for (k, p), h in zip(L, ps)}
        log(f'Holm family {f}: {len(L)} tests, {sum(h < 0.05 for h in ps)} with p_holm < 0.05')
    (OUT / f'h65_summary{s}.json').write_text(json.dumps(RESU, indent=1, default=float))
    log(f'wrote h65_summary{s}.json')
    try:
        figure(RESU, s)
    except Exception as e:  # noqa: BLE001
        log('figure failed', repr(e))


def figure(RESU, s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    O = np.load(OUT / f'h65_jc_obs{s}.npz')
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.0))
    a = ax[0]; th = np.linspace(0, 2 * np.pi, 200)
    a.plot(0.8 * np.cos(th), 0.8 * np.sin(th), color='0.6', lw=1, ls='--')
    a.scatter(O['ex|deta'], O['ex|dphi'], s=4 + 300 * O['ex|pt'] / O['ex|pt'].max(), color='#2b6cb0', alpha=0.75, label=f'jet particles ({len(O["ex|pt"])})')
    a.scatter(O['ex|gdeta'], O['ex|gdphi'], s=12, color='#dd6b20', marker='x', label=f'ghosts, rho = 10 GeV ({len(O["ex|gpt"])}, sum {O["ex|gpt"].sum():.0f} GeV)')
    a.set_xlim(-0.9, 0.9); a.set_ylim(-0.9, 0.9); a.set_aspect('equal'); a.set_xlabel('Δη from jet axis'); a.set_ylabel('Δφ from jet axis')
    j = int(O['ex|j']); a.set_title(f'JetClass W jet, m = {float(O["m|base"][j]):.0f} → {float(O["m|pu10"][j]):.0f} GeV, '
                                    f'm_SD = {float(O["msd|base"][j]):.0f} → {float(O["msd|pu10"][j]):.0f} GeV', fontsize=10)
    a.legend(loc='lower left', fontsize=8, frameon=False)
    b = ax[1]; C = RESU['datasets']['jc']['classes']['W']['taggers']; x = [0] + list(RHOS)
    sty = [('ParT_full', 'pu', '#2b6cb0', '-', 'ParT_full, ghosts over the jet'),
           ('ParT_full', 'c2_', '#2b6cb0', '--', 'ParT_full, same ghosts at the axis (mass ~ fixed)'),
           ('win_m', 'pu', '#c53030', '-', 'plain-mass window'), ('win_msd', 'pu', '#2f855a', '-', 'soft-drop mass window'),
           ('win_msub', 'pu', '#805ad5', '-', 'area-subtracted mass window'), ('LR_N', 'pu', '#d69e2e', '-', 'LR(N), particle count only')]
    for tn, cp, col, ls, lab in sty:
        if tn not in C: continue
        e0 = 100 * C[tn]['effS0']; yv = [e0] + [100 * C[tn]['cond'][f'{cp}{r}']['effS'] for r in RHOS]
        b.plot(x, yv, color=col, ls=ls, marker='o', ms=3, label=lab)
    b.set_xlabel('ghost pT density ρ (GeV per unit η–φ area)'); b.set_ylabel('W jets still called W (%)')
    b.set_ylim(-2, 55); b.set_title('JetClass W vs QCD, threshold fixed at 50 % W efficiency', fontsize=10)
    b.legend(fontsize=7.5, frameon=False, loc='lower left'); b.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / f'h65_pileup{s}.png', dpi=130); log(f'wrote h65_pileup{s}.png')


if __name__ == '__main__':
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--part', required=True, choices=['hls', 'nets', 'obs', 'summary'])
    ap_.add_argument('--ds', default=None, choices=['hls', 'jc', 'top', 'qg'])
    ap_.add_argument('--smoke', type=int, default=0)
    a = ap_.parse_args()
    if a.part == 'summary': summary(a.smoke)
    elif a.part == 'hls': run('hls', 'hls', a.smoke)
    else: run(a.ds, a.part, a.smoke)
