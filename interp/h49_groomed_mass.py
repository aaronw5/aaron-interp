"""H49 (loop D, round D10; new claim D21, refines D10): under soft contamination, does each tagger's W / Z / top call follow
the soft-drop (groomed) mass m_SD or the plain (ungroomed) jet mass m, once the number of added particles, their summed pT
and the ungroomed mass change are held equal?

Edit pair (per jet, on the capped particle list each model sees)
  Prongs: Cambridge/Aachen (C/A) reclustering of the jet's particles (E-scheme recombination, rapidity-azimuth distance) and
  soft drop (beta = 0, z_cut = 0.1; with beta = 0 the R0 = 0.8 parameter drops out): decluster, following the harder branch,
  until min(pT1, pT2) / (pT1 + pT2) > z_cut. The two branches at that step are the prongs A, B; R_g = their distance in
  (rapidity, azimuth). F = the prong farther from the jet's E-scheme axis, r_F its distance from the axis.
  Added particles: k = 3 massless neutral particles, each with pT q = 0.02 pT_J / 3 (total 2 % of the jet pT), all on the
  circle of radius r* = r_F + d about the jet axis, d = min(0.7 R_g, 0.78 - r_F) (>= 0.3 R_g, else the jet is not matched), spaced by 0.15 R_g along the circle.
  G_out: centred on one of 9 directions 60-300 deg from F (30 deg steps): >= 1.1 R_g from both prong centres, least pT within
         0.5 R_g, so C/A attaches them after A and B have merged and soft drop removes them: m rises, m_SD should not.
  G_in:  centred on the direction of prong F, i.e. d beyond the centre of F on the outer side; C/A attaches them to F
         before A and B merge, so soft drop keeps them: m and m_SD both rise.
  Because Delta(m^2) = 2 P_J . p_added ~ q pT_J (r^2 + <r^2>) to leading order in the angles, equal radius gives equal
  ungroomed Delta m. The G_in radius is then rescaled per jet (bisection) so that its ungroomed m equals G_out's to 0.01 GeV.
  Matched set (all statistics): prongs found with R_g >= 0.02; r_in, r* <= 0.8; |Delta m_SD(G_out)| < 0.5 GeV;
  Delta m_SD(G_in) >= 0.5 Delta m(G_out); |m(G_in) - m(G_out)| < 1 GeV; all added particles inside the slot limit.
Controls
  c0    count-only: the same 3 particles at the jet axis (radius 0.002) and their total pT 3q removed from the jet's softest
        particles (pT and mass scaled by 1 - 3q/S over the softest set whose pT sum S reaches 6q): count +3, sum pT and
        mass about fixed.
  rand  the same 3 particles at uniformly random positions in the disc of radius 0.8 about the axis.
  TN    (own-axis build only) the unedited particles with inputs taken about the E-scheme axis of G_in / G_out: a pure
        translation by the axis shift that each edit induces.
Builds: 'st' inputs about the stored axis (hls4ml (0, 0), JetClass jet_eta / jet_phi, Top the unedited E-scheme axis);
        'rc' inputs about each (edited) jet's own E-scheme axis. JetClass eta = jet_eta + sign(jet_eta) * deta.
Samples (random rows; never first-k)
  hls  jedi_core.sample(2000, 4901) jets (h1_irc_panel.hls_sample: same indices), 10,000 jets, 5 classes, 64 slots
  jc   JetClass first test file per class, QCD (ZJetsToNuNu), Wqq, Zqq, Tbqq: 2,000 random rows each (seed 4902), 128 slots
  top  Top Landscape test: data_top.sample(3000, 4903), 3,000 top + 3,000 QCD, 200 slots
Models
  hls  JEDI-linear official n64 3-feature, JEDI-linear with redrawn weights (null); --part hlss (conda): dsets-d64-n64,
       linf-2b-d32-k8-n64, tf-2b-d32-n64 (seed 0)
  jc   ParT_full, ParT_kin, PN_full, PN_kin; ParT_full with every weight tensor's entries permuted (null)
  top  ParT, PN (kin), LorentzNet, PFN, EFN; PFN with permuted weights (null) (interp.h13_pileup_panel.TopModels)
Scores: log-odds of signal class c against QCD. hls: log p(c) - log(p(q) + p(g)); jc: logit_c - logit_QCD; top: top log-odds.
References (computed on the same edited jets): windows W/Z: 65 < m < 105 GeV and 65 < m_SD < 105 GeV; top: 105 < m_SD <
  210 GeV and tau32 < 0.54 (and the same on ungroomed m); tau32 = tau3 / tau2 with exclusive-kT axes, beta 1, all particles.
  MLP(m), MLP(m_SD): numpy MLP (1 -> 32 -> 32 -> classes, tanh, Adam) on the unedited observable, 2-fold cross-fit on the sample.
Run
  ./job.sh d-h49-hls 1 4 'python -m interp.h49_groomed_mass --part hls'
  AI_PY=conda ./job.sh d-h49-hlss 1 2 'python -m interp.h49_groomed_mass --part hlss'
  ./part_job.sh d-h49-jc  'python -m interp.h49_groomed_mass --part jc'
  ./part_job.sh d-h49-top 'python -m interp.h49_groomed_mass --part top'
  ./job.sh d-h49-sum 1 4 'python -m interp.h49_groomed_mass --part summary'
  -> $AI_RESULTS/physics/h49_{hls,hlss,jc,top}.npz / .json, h49_summary.json, h49_groomed.png. --smoke N: N jets per class.
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
from interp.h1_irc_panel import copyT, finalize, jet4, wrap, holm, p3  # noqa: E402
from interp.h18_softcount_cross import neutral, eaxis, nreal, logsm, auc  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
OUT = RES / 'physics'
V = Path('/j-jepa-vol/aaron-interp')
SEEDS = dict(hls=4901, jc=4902, top=4903)
K, FRAC, ZCUT, D_IN, SPACE = 3, 0.02, 0.1, 0.7, 0.15
RG_MIN, R_MAX = 0.02, 0.8
EPS_B = 0.05
WIN_WZ, WIN_T, TAU32_CUT = (65.0, 105.0), (105.0, 210.0), 0.54
CAP = dict(hls=64, jc=128, top=200)
EDITS = ['Gin', 'Gout', 'c0', 'rand']
OBS_CONDS = ['base'] + EDITS
SCORE_CONDS = [f'{c}_{b}' for c in OBS_CONDS for b in ('st', 'rc')] + ['TNin_rc', 'TNout_rc']
SIG = dict(hls={'W': 2, 'Z': 3, 't': 4}, jc={'W': 7, 'Z': 6, 't': 8}, top={'t': 1})
QCD = dict(hls=[0, 1], jc=[0], top=[0])
NULLS = {'JEDI_redrawn', 'ParT_full_permuted', 'PFN_permuted'}
LABEL = {'JEDI': 'JEDI-linear', 'JEDI_redrawn': 'JEDI-linear redrawn', 'dsets': 'Deep Sets d64', 'linf': 'Lin-former 2b-d32-k8',
         'tf': 'Transformer 2b-d32', 'ParT_full': 'ParT_full', 'ParT_kin': 'ParT_kin', 'PN_full': 'PN_full', 'PN_kin': 'PN_kin',
         'ParT_full_permuted': 'ParT_full permuted', 'ParT': 'ParT (Top)', 'PN': 'PN (Top)', 'LNet': 'LorentzNet', 'PFN': 'PFN',
         'EFN': 'EFN', 'PFN_permuted': 'PFN permuted', 'MLPm': 'MLP(m)', 'MLPmsd': 'MLP(m_SD)'}
KNOWN = dict(JEDI='5-class acc 0.8092 (260k test); 0.808 (H13), 0.811 (H13v)', JEDI_redrawn='0.200 (chance)',
             dsets='0.8064 (test)', linf='0.8106 (test)', tf='0.7619 (test)',
             ParT_full='per class (H18) QCD 0.783 Zqq 0.689 Wqq 0.805 Tbqq 0.956', ParT_kin='QCD 0.718 Zqq 0.550 Wqq 0.702 Tbqq 0.912',
             PN_full='QCD 0.764 Zqq 0.652 Wqq 0.784 Tbqq 0.949', PN_kin='QCD 0.693 Zqq 0.528 Wqq 0.680 Tbqq 0.913',
             ParT='AUC 0.9858', PN='AUC 0.9858', LNet='AUC 0.9870', PFN='AUC 0.9819', EFN='AUC 0.975-0.979')
T0_ = time.time()


def log(*a):
    print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


def sfx(smoke):
    return '_smoke' if smoke else ''


# =========================================================================================================== soft drop
def _yphi4(v):
    pt = np.hypot(v[..., 0], v[..., 1]); E, pz = v[..., 3], v[..., 2]
    y = 0.5 * np.log(np.maximum(E + pz, 1e-300) / np.maximum(E - pz, 1e-300))
    return pt, y, np.arctan2(v[..., 1], v[..., 0])


def _m4(v):
    return np.sqrt(np.maximum(v[..., 3] ** 2 - (v[..., :3] ** 2).sum(-1), 0.0))


def _sd_chunk(pt, eta, phi, m):
    """C/A (E-scheme, rapidity-azimuth distance) + soft drop (beta 0, z_cut 0.1) on (J, w) particles.
    Returns m_SD, prong four-vectors v1 (harder), v2 at the stopping split, z_g, R_g, found (a split passed z_cut)."""
    J, w = pt.shape; rows = np.arange(J); real = pt > 0
    px, py, pz = p3(pt, eta, phi); E = np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)
    V = np.zeros((J, 2 * w, 4)); V[:, :w] = np.stack([px, py, pz, E], -1) * real[..., None]
    _, Y, PH = _yphi4(V[:, :w]); Y = np.where(real, Y, 0.0); PH = np.where(real, PH, 0.0)
    act = real.copy(); INF = np.inf
    D = (Y[:, :, None] - Y[:, None]) ** 2 + wrap(PH[:, :, None] - PH[:, None]) ** 2
    D = np.where(act[:, :, None] & act[:, None, :], D, INF); D[:, np.arange(w), np.arange(w)] = INF
    kids = -np.ones((J, 2 * w, 2), int); slot = np.tile(np.arange(w), (J, 1))
    for s in range(w - 1):
        g = np.flatnonzero(act.sum(1) >= 2)
        if not len(g): break
        f = D[g].reshape(len(g), -1).argmin(1); a, b = f // w, f % w
        na, nb = slot[g, a], slot[g, b]; new = w + s
        V[g, new] = V[g, na] + V[g, nb]; kids[g, new, 0] = na; kids[g, new, 1] = nb
        slot[g, a] = new; act[g, b] = False
        _, yn, pn = _yphi4(V[g, new]); Y[g, a] = yn; PH[g, a] = pn
        D[g, b, :] = INF; D[g, :, b] = INF
        d = (Y[g] - yn[:, None]) ** 2 + wrap(PH[g] - pn[:, None]) ** 2
        d = np.where(act[g], d, INF); d[np.arange(len(g)), a] = INF
        D[g, a, :] = d; D[g, :, a] = d
    has = real.any(1); cur = np.where(has, slot[rows, act.argmax(1)], 0)
    done = ~has; msd = np.zeros(J); v1o = np.zeros((J, 4)); v2o = np.zeros((J, 4)); zg = np.zeros(J); rg = np.zeros(J)
    found = np.zeros(J, bool)
    for _ in range(2 * w):
        c1, c2 = kids[rows, cur, 0], kids[rows, cur, 1]; leaf = (c1 < 0) & ~done
        msd = np.where(leaf, _m4(V[rows, cur]), msd); done |= leaf
        if done.all(): break
        u1, u2 = V[rows, np.maximum(c1, 0)], V[rows, np.maximum(c2, 0)]
        p1, y1, f1 = _yphi4(u1); p2, y2, f2 = _yphi4(u2)
        zz = np.minimum(p1, p2) / np.maximum(p1 + p2, 1e-300); st = ~done & (zz > ZCUT)
        msd = np.where(st, _m4(V[rows, cur]), msd); h = (p1 >= p2)[:, None]
        v1o = np.where(st[:, None], np.where(h, u1, u2), v1o); v2o = np.where(st[:, None], np.where(h, u2, u1), v2o)
        zg = np.where(st, zz, zg); rg = np.where(st, np.hypot(y1 - y2, wrap(f1 - f2)), rg)
        found |= st; done |= st
        cur = np.where(~done, np.where(p1 >= p2, c1, c2), cur)
    return msd, v1o, v2o, zg, rg, found


def softdrop(T, chunk=200):
    J = len(T['pt']); n = nreal(T); order = np.argsort(n, kind='stable')
    out = dict(msd=np.zeros(J), v1=np.zeros((J, 4)), v2=np.zeros((J, 4)), zg=np.zeros(J), rg=np.zeros(J), found=np.zeros(J, bool))
    for a in range(0, J, chunk):
        idx = order[a:a + chunk]; w = max(int(n[idx].max()), 2)
        # particles are pT-sorted after finalize, so the first w slots hold every real particle
        r = _sd_chunk(T['pt'][idx, :w], T['eta'][idx, :w], T['phi'][idx, :w], T['m'][idx, :w])
        for k, v in zip(('msd', 'v1', 'v2', 'zg', 'rg', 'found'), r): out[k][idx] = v
    return out


def mass(T):
    return _m4(jet4(T))


def tau32(T, ae, ap):
    """tau3 / tau2 (exclusive-kT axes, beta 1, all real particles), per jet"""
    from interp.e19_common import kt3
    J = len(T['pt']); out = np.zeros(J)
    for j in range(J):
        r = T['pt'][j] > 0; pt = T['pt'][j, r]; e = T['eta'][j, r] - ae[j]; f = wrap(T['phi'][j, r] - ap[j])
        if len(pt) < 4: out[j] = 1.0; continue
        z = pt / pt.sum(); tau = {}
        for k in (2, 3):
            ax = kt3(pt, e, f, k)[1]; ax = ax[ax[:, 0] > 0]
            tau[k] = (z * np.hypot(e[:, None] - ax[None, :, 1], wrap(f[:, None] - ax[None, :, 2])).min(1)).sum()
        out[j] = tau[3] / max(tau[2], 1e-12)
    return out


# =========================================================================================================== edits
def add_parts(T, pt, eta, phi):
    """append K particles (pt, eta, phi: (J, K); pt 0 = none) as massless neutral hadrons"""
    W = T['pt'].shape[1]; Kk = pt.shape[1]; U = copyT(T, Kk); on = pt > 0
    U['pt'][:, W:] = np.where(on, pt, 0.0); U['eta'][:, W:] = np.where(on, eta, 0.0); U['phi'][:, W:] = np.where(on, wrap(phi), 0.0)
    U['m'][:, W:] = 0.0; neutral(U, W, on)
    return U


def p4q(q, eta, phi):
    """four-vectors (J, K, 4) of massless particles"""
    return np.stack([q * np.cos(phi), q * np.sin(phi), q * np.sinh(eta), q * np.cosh(eta)], -1)


def geometry(T0, cap):
    """prongs, radius r*, the G_in / G_out / c0 / rand particle positions; returns (G dict, conditions dict of tables)"""
    J = len(T0['pt']); rows = np.arange(J)
    ae, ap = eaxis(T0); pJ = np.hypot(*jet4(T0)[:, :2].T); q = FRAC * pJ / K
    sd = softdrop(T0); log(f'base soft drop: split found in {sd["found"].mean():.3f} of jets, median R_g {np.median(sd["rg"][sd["found"]]):.3f}')
    pe = lambda v: (np.arcsinh(v[:, 2] / np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-300)), np.arctan2(v[:, 1], v[:, 0]))
    e1, f1 = pe(sd['v1']); e2, f2 = pe(sd['v2'])
    d1 = np.stack([e1 - ae, wrap(f1 - ap)], 1); d2 = np.stack([e2 - ae, wrap(f2 - ap)], 1)
    r1, r2 = np.hypot(*d1.T), np.hypot(*d2.T); far2 = r2 > r1
    dF = np.where(far2[:, None], d2, d1); dN = np.where(far2[:, None], d1, d2); rF = np.maximum(r1, r2)
    Rg = np.hypot(*(d1 - d2).T)                               # pseudorapidity-azimuth prong distance (for the geometry)
    thF = np.where(rF > 1e-3, np.arctan2(dF[:, 1], dF[:, 0]), np.arctan2(dF[:, 1] - dN[:, 1], dF[:, 0] - dN[:, 0]))
    dIn = np.clip(0.78 - rF, 0.3 * Rg, D_IN * Rg)            # offset beyond F: 0.7 R_g, reduced (>= 0.3 R_g) to keep r* <= 0.78
    rstar = rF + dIn
    dth = SPACE * Rg / np.maximum(rstar, 1e-6); offs = np.array([-1.0, 0.0, 1.0])
    # G_out direction: of 9 candidate angles thF + 60..300 deg (30 deg steps) on the circle r*, those whose centre is >= 1.1 R_g
    # from both prong centres; among them the one with the least unedited pT within 0.5 R_g of the centre (all 9 if none qualify)
    real = T0['pt'] > 0; de = T0['eta'] - ae[:, None]; dp = wrap(T0['phi'] - ap[:, None])
    cand = np.deg2rad(np.arange(60, 301, 30)); costs = []
    for c in cand:
        th = thF + c; ce, cp = rstar * np.cos(th), rstar * np.sin(th)
        near = real & (np.hypot(de - ce[:, None], dp - cp[:, None]) < 0.5 * Rg[:, None]); npt = (T0['pt'] * near).sum(1)
        dmin = np.minimum(np.hypot(ce - dF[:, 0], cp - dF[:, 1]), np.hypot(ce - dN[:, 0], cp - dN[:, 1]))
        costs.append(np.where(dmin >= 1.1 * Rg, npt, 1e6 + npt - dmin))
    thO = thF + cand[np.argmin(np.stack(costs, 1), 1)]
    angO = thO[:, None] + offs[None] * dth[:, None]; angI = thF[:, None] + offs[None] * dth[:, None]
    Q = np.repeat(q[:, None], K, 1)
    eO, pO = ae[:, None] + rstar[:, None] * np.cos(angO), ap[:, None] + rstar[:, None] * np.sin(angO)
    Uo = add_parts(T0, Q, eO, pO); Uof, over = finalize(Uo, cap)
    kept = (np.isclose(Uof['pt'][:, :, None], Q[:, None, :1], rtol=0, atol=1e-9).sum(1)[:, 0] >= K)
    P_rest = jet4(Uof) - p4q(Q, eO, pO).sum(1); m_out = _m4(jet4(Uof))
    m_in_of = lambda rho: _m4(P_rest + p4q(Q, ae[:, None] + rho[:, None] * rstar[:, None] * np.cos(angI),
                                           ap[:, None] + rho[:, None] * rstar[:, None] * np.sin(angI)).sum(1))
    lo, hi = np.full(J, 0.2), np.full(J, 2.0)
    bracket = (m_in_of(lo) <= m_out) & (m_in_of(hi) >= m_out)
    for _ in range(60):
        mid = 0.5 * (lo + hi); up = m_in_of(mid) < m_out; lo = np.where(up, mid, lo); hi = np.where(up, hi, mid)
    rho = 0.5 * (lo + hi); rin = rho * rstar
    eI, pI = ae[:, None] + rin[:, None] * np.cos(angI), ap[:, None] + rin[:, None] * np.sin(angI)
    Ui = add_parts(T0, Q, eI, pI); Uif, _ = finalize(Ui, cap)
    # c0: 3 particles at the axis, their total pT removed from the softest particles
    Uc = copyT(T0); ptm = np.where(real, T0['pt'], np.inf); o = np.argsort(ptm, 1, kind='stable')
    srt = np.take_along_axis(np.where(real, T0['pt'], 0.0), o, 1); cs = np.cumsum(srt, 1); need = 2 * K * q
    inset_s = (cs - srt) < need[:, None]; inset_s &= np.take_along_axis(real, o, 1)
    inset = np.zeros_like(real); np.put_along_axis(inset, o, inset_s, 1)
    S = (T0['pt'] * inset).sum(1); fac = 1 - K * q / np.maximum(S, 1e-300)
    Uc['pt'] = np.where(inset, T0['pt'] * fac[:, None], T0['pt']); Uc['m'] = np.where(inset, T0['m'] * fac[:, None], T0['m'])
    ang0 = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])[None]
    Uc = add_parts(Uc, Q, ae[:, None] + 0.002 * np.cos(ang0), ap[:, None] + 0.002 * np.sin(ang0))
    # rand
    rng = np.random.default_rng([SEEDS.get('_cur', 0), 11]); rr = R_MAX * np.sqrt(rng.random((J, K))); th = 2 * np.pi * rng.random((J, K))
    Ur = add_parts(T0, Q, ae[:, None] + rr * np.cos(th), ap[:, None] + rr * np.sin(th))
    G = dict(ae=ae, ap=ap, q=q, pJ=pJ, rstar=rstar, rin=rin, rho=rho, Rg=Rg, rF=rF, zg=sd['zg'], found=sd['found'], bracket=bracket,
             kept=kept, overflow=over, c0_frac_scaled=inset.sum(1) / np.maximum(real.sum(1), 1), c0_fac=fac,
             dF=dF, dN=dN, eO=eO - ae[:, None], pO=wrap(pO - ap[:, None]), eI=eI - ae[:, None], pI=wrap(pI - ap[:, None]),
             m_out_check=m_out, msd_base_sd=sd['msd'])
    conds = dict(base=finalize(T0, cap)[0], Gin=Uif, Gout=Uof, c0=finalize(Uc, cap)[0], rand=finalize(Ur, cap)[0])   # OBS_CONDS order
    return G, conds


def observables(conds, ae, ap, do_tau=True):
    O = {}
    for c, U in conds.items():
        t = time.time(); O[f'm|{c}'] = mass(U); O[f'msd|{c}'] = softdrop(U)['msd']; O[f'n|{c}'] = nreal(U).astype(float)
        O[f'sumpt|{c}'] = U['pt'].sum(1)
        if do_tau: O[f'tau32|{c}'] = tau32(U, ae, ap)
        log(f'observables {c}: median m {np.median(O[f"m|{c}"]):.1f} m_SD {np.median(O[f"msd|{c}"]):.1f} GeV ({time.time() - t:.0f} s)')
    return O


def matched(G, O):
    dm_out = O['m|Gout'] - O['m|base']; dm_in = O['m|Gin'] - O['m|base']
    dsd_out = O['msd|Gout'] - O['msd|base']; dsd_in = O['msd|Gin'] - O['msd|base']
    crit = dict(prongs=G['found'] & (G['Rg'] >= RG_MIN), radius=(G['rstar'] - G['rF'] >= 0.3 * G['Rg'] - 1e-9) & (G['rstar'] <= R_MAX) & (G['rin'] <= R_MAX), slots=G['kept'],
                bracket=G['bracket'], groomed_out=np.abs(dsd_out) < 0.5, kept_in=dsd_in >= 0.5 * dm_out, dm_match=np.abs(dm_in - dm_out) < 1.0)
    valid = np.ones(len(dm_out), bool)
    for v in crit.values(): valid &= v
    return valid, crit


def axes_of(conds, T0):
    """stored axis per dataset (T0 ax_eta / ax_phi) and each condition's own E-scheme axis"""
    st = (T0['ax_eta'], T0['ax_phi']); A = {}
    for c, U in conds.items(): A[c] = eaxis(U)
    return st, A


def score_all(conds, T0, scorer):
    """scorer(Uf, ae, ap) -> {model: logits or score (J,) / (J, C)}; returns {cond_build: {model: array}}"""
    st, A = axes_of(conds, T0); S = {}
    for c, U in conds.items():
        for b, ax in (('st', st), ('rc', A[c])):
            t = time.time(); S[f'{c}_{b}'] = scorer(U, *ax); log(f'scored {c}_{b} ({time.time() - t:.0f} s)')
    S['TNin_rc'] = scorer(conds['base'], *A['Gin']); S['TNout_rc'] = scorer(conds['base'], *A['Gout'])
    off = {c: 1e3 * np.hypot(A[c][0] - A['base'][0], wrap(A[c][1] - A['base'][1])) for c in conds}
    return S, off


# =========================================================================================================== numpy MLP reference
def mlp_fit(x, y, C, seed, hidden=32, steps=1500, lr=1e-2):
    rng = np.random.default_rng(seed); mu, sd = x.mean(), x.std() + 1e-9; X = ((x - mu) / sd)[:, None]
    Ws = [rng.normal(0, 1.0, (1, hidden)), rng.normal(0, 1 / math.sqrt(hidden), (hidden, hidden)), rng.normal(0, 1 / math.sqrt(hidden), (hidden, C))]
    bs = [np.zeros(hidden), np.zeros(hidden), np.zeros(C)]; P = Ws + bs
    mo = [np.zeros_like(p) for p in P]; ve = [np.zeros_like(p) for p in P]; Y = np.eye(C)[y]; n = len(y)

    def fwd(X):
        h1 = np.tanh(X @ Ws[0] + bs[0]); h2 = np.tanh(h1 @ Ws[1] + bs[1]); return h1, h2, h2 @ Ws[2] + bs[2]
    for t in range(1, steps + 1):
        h1, h2, L = fwd(X); Pr = np.exp(logsm(L)); dL = (Pr - Y) / n
        gW2 = h2.T @ dL; gb2 = dL.sum(0); dh2 = dL @ Ws[2].T * (1 - h2 ** 2)
        gW1 = h1.T @ dh2; gb1 = dh2.sum(0); dh1 = dh2 @ Ws[1].T * (1 - h1 ** 2)
        gW0 = X.T @ dh1; gb0 = dh1.sum(0)
        grads = [gW0, gW1, gW2, gb0, gb1, gb2]
        for i, (p, g) in enumerate(zip(P, grads)):
            mo[i] = 0.9 * mo[i] + 0.1 * g; ve[i] = 0.999 * ve[i] + 0.001 * g * g
            p -= lr * (mo[i] / (1 - 0.9 ** t)) / (np.sqrt(ve[i] / (1 - 0.999 ** t)) + 1e-8)
    return lambda xn: logsm(fwd(((np.asarray(xn) - mu) / sd)[:, None])[2])


def mlp_refs(O, y, ds, seed):
    """MLP(m) and MLP(m_SD): 2-fold cross-fit on unedited jets; log-probabilities per condition"""
    classes = sorted(set(y.tolist())); cid = np.searchsorted(classes, y); C = len(classes)
    fold = np.random.default_rng([seed, 21]).permutation(len(y)) % 2; out = {}
    for name, key in (('MLPm', 'm'), ('MLPmsd', 'msd')):
        LP = {c: np.zeros((len(y), C)) for c in OBS_CONDS}
        for f in (0, 1):
            tr = fold != f; te = fold == f
            fn = mlp_fit(O[f'{key}|base'][tr] / 100.0, cid[tr], C, seed + f)
            for c in OBS_CONDS: LP[c][te] = fn(O[f'{key}|{c}'][te] / 100.0)
        out[name] = (LP, classes)
    return out


def class_scores(L, ds, classes=None):
    """log-odds per signal class vs QCD from logits / log-probs L (J, C) in the dataset's label space (classes: label ids of columns)"""
    out = {}
    if L.ndim == 1: return {'t': L}
    ls = logsm(L); col = (lambda k: k) if classes is None else (lambda k: classes.index(k))
    qcd = np.logaddexp.reduce(np.stack([ls[:, col(k)] for k in QCD[ds]], 1), axis=1)
    for c, k in SIG[ds].items(): out[c] = ls[:, col(k)] - qcd
    return out


# =========================================================================================================== run one dataset
def run_dataset(ds, T0, y, scorer, smoke, info, do_tau=True):
    SEEDS['_cur'] = SEEDS[ds]
    G, conds = geometry(T0, CAP[ds])
    ae, ap = G['ae'], G['ap']
    O = observables(conds, ae, ap, do_tau)
    valid, crit = matched(G, O)
    acc = {name: {cls: float(v[y == k].mean()) for cls, k in list(SIG[ds].items()) + [(f'QCD{i}', i) for i in QCD[ds]]}
           for name, v in list(crit.items()) + [('ALL', valid)]}
    log(f'{ds}: matched set {valid.sum()} / {len(y)}; per class {acc["ALL"]}')
    S, off = score_all(conds, T0, scorer)
    NPZ = dict(y=y, valid=valid, **{f'crit|{k}': v for k, v in crit.items()}, **{f'G|{k}': np.asarray(v) for k, v in G.items()}, **O)
    for c, d in S.items():
        for mn, L in d.items():
            L = np.asarray(L, np.float64)
            for cls, s in class_scores(L, ds).items(): NPZ[f's|{mn}|{cls}|{c}'] = s.astype(np.float32)
            if L.ndim == 2: NPZ[f'arg|{mn}|{c}'] = L.argmax(1).astype(np.int8)
    for c, v in off.items(): NPZ[f'axoff_mrad|{c}'] = v
    R = mlp_refs(O, y, ds, SEEDS[ds])
    for name, (LP, classes) in R.items():
        for c in OBS_CONDS:
            for cls, s in class_scores(LP[c], ds, classes).items():
                NPZ[f's|{name}|{cls}|{c}_st'] = s.astype(np.float32); NPZ[f's|{name}|{cls}|{c}_rc'] = s.astype(np.float32)
    # unedited performance on the whole sample
    un = {}
    for mn, L in S['base_st'].items():
        L = np.asarray(L, np.float64); un[mn] = {}
        if L.ndim == 2:
            a = L.argmax(1)
            if ds == 'hls': un[mn]['acc_5class'] = float((a == y).mean())
            un[mn]['per_class_acc'] = {str(k): float((a[y == k] == k).mean()) for k in sorted(set(y.tolist()))}
        for cls, s in class_scores(L, ds).items():
            rows = np.isin(y, [SIG[ds][cls]] + QCD[ds]); un[mn][f'AUC_{cls}_vs_QCD'] = auc(s[rows], (y[rows] == SIG[ds][cls]).astype(int))
        log(f'{ds} unedited {mn}: {un[mn]}  known: {KNOWN.get(mn, "-")}')
    for name in R:
        un[name] = {}
        for cls in SIG[ds]:
            s = NPZ[f's|{name}|{cls}|base_st']; rows = np.isin(y, [SIG[ds][cls]] + QCD[ds])
            un[name][f'AUC_{cls}_vs_QCD'] = auc(s[rows], (y[rows] == SIG[ds][cls]).astype(int))
    # geometry summary on the matched set
    gs = {}
    for cls, k in list(SIG[ds].items()) + [('QCD', -1)]:
        r = valid & (np.isin(y, QCD[ds]) if k < 0 else (y == k))
        if not r.any(): continue
        med = lambda v: float(np.median(v[r]))
        gs[cls] = dict(n=int(r.sum()), Rg=med(G['Rg']), rstar=med(G['rstar']), rin_over_rstar=med(G['rho']), q_GeV=med(G['q']),
                       dm_out=med(O['m|Gout'] - O['m|base']), dm_in=med(O['m|Gin'] - O['m|base']),
                       dmsd_out=med(O['msd|Gout'] - O['msd|base']), dmsd_in=med(O['msd|Gin'] - O['msd|base']),
                       dm_c0=med(O['m|c0'] - O['m|base']), dmsd_c0=med(O['msd|c0'] - O['msd|base']),
                       dm_rand=med(O['m|rand'] - O['m|base']), dmsd_rand=med(O['msd|rand'] - O['msd|base']),
                       dm_out_mean=float((O['m|Gout'] - O['m|base'])[r].mean()), dmsd_in_mean=float((O['msd|Gin'] - O['msd|base'])[r].mean()),
                       dn={c: float((O[f'n|{c}'] - O['n|base'])[r].mean()) for c in EDITS},
                       dsumpt={c: float((O[f'sumpt|{c}'] - O['sumpt|base'])[r].mean()) for c in EDITS},
                       axis_offset_mrad_median={c: float(np.median(off[c][r])) for c in EDITS},
                       m_base=med(O['m|base']), msd_base=med(O['msd|base']))
        log(f'{ds} geometry {cls}: {gs[cls]}')
    # example jet for the figure: a matched signal jet (W where present) with R_g near the class median
    kx = SIG[ds].get('W', SIG[ds].get('t')); rr = np.flatnonzero(valid & (y == kx))
    if len(rr):
        j = int(rr[np.argmin(np.abs(G['Rg'][rr] - np.median(G['Rg'][rr])) + 0.002 * np.abs(nreal(T0)[rr] - np.median(nreal(T0)[rr])))])
        b = conds['base']; r = b['pt'][j] > 0
        NPZ.update({'ex|j': np.array(j), 'ex|pt': b['pt'][j, r], 'ex|deta': b['eta'][j, r] - ae[j], 'ex|dphi': wrap(b['phi'][j, r] - ap[j])})
    info.update(matched_fraction=acc, geometry=gs, unedited=un, n=int(len(y)))
    np.savez_compressed(OUT / f'h49_{ds}{sfx(smoke)}.npz', **NPZ)
    (OUT / f'h49_{ds}{sfx(smoke)}.json').write_text(json.dumps(info, indent=1, default=float))
    log(f'wrote h49_{ds}{sfx(smoke)}.npz / .json')
    return conds, G, valid


# =========================================================================================================== dataset parts
def part_hls(smoke):
    from interp.jedi_core import Jedi
    from interp.h1_irc_panel import hls_sample
    from interp.h13_pileup_panel import hls_in
    T0, y = hls_sample(smoke or 2000, SEEDS['hls']); T0 = finalize(T0, 64)[0]
    nets = dict(JEDI=Jedi(64, '3-feature'), JEDI_redrawn=Jedi(64, '3-feature', untrained=True, seed=0))
    X = {}

    def scorer(Uf, ae, ap):
        x = hls_in(Uf, ae, ap); X[len(X)] = x
        return {k: np.asarray(n.run(x)['logits'], np.float64) for k, n in nets.items()}
    info = dict(sample=f'jets of jedi_core.sample(per_class={smoke or 2000}, seed={SEEDS["hls"]})', frac_at_64=float((nreal(T0) >= 64).mean()))
    run_dataset('hls', T0, y, scorer, smoke, info)
    assert len(X) == len(SCORE_CONDS), (len(X), SCORE_CONDS)
    np.savez_compressed(OUT / f'h49_hls_x{sfx(smoke)}.npz', y=y, **{SCORE_CONDS[i]: X[i] for i in range(len(SCORE_CONDS))})
    log('wrote h49_hls_x (inputs per condition, for --part hlss)')


def part_hlss(smoke):
    import torch
    from interp.h1_irc_panel import load_small, torch_logits
    torch.set_num_threads(1)
    SM = Path(os.environ.get('AI_SMALL_MODELS', '/j-jepa-vol/aaron-interp/models/small'))
    runs = dict(dsets='dsets-d64-n64', linf='linf-2b-d32-k8-n64', tf='tf-2b-d32-n64')
    nets = {k: load_small(SM / d / 'seed0') for k, d in runs.items()}
    Z = np.load(OUT / f'h49_hls_x{sfx(smoke)}.npz'); y = Z['y']; NPZ = dict(y=y); info = dict(unedited={}, known={})
    for k, d in runs.items():
        info['known'][k] = json.loads((SM / d / 'seed0' / 'test.json').read_text())['accuracy']
    for c in SCORE_CONDS:
        x = Z[c]
        for mn, (net, sh, sc) in nets.items():
            L = torch_logits(net, x, sh, sc)
            for cls, s in class_scores(L, 'hls').items(): NPZ[f's|{mn}|{cls}|{c}'] = s.astype(np.float32)
            NPZ[f'arg|{mn}|{c}'] = L.argmax(1).astype(np.int8)
            if c == 'base_st':
                info['unedited'][mn] = dict(acc_5class=float((L.argmax(1) == y).mean()))
                log(f'hlss unedited {mn}: acc {info["unedited"][mn]["acc_5class"]:.4f} known {info["known"][mn]}')
        log(f'hlss {c} done')
    np.savez_compressed(OUT / f'h49_hlss{sfx(smoke)}.npz', **NPZ)
    (OUT / f'h49_hlss{sfx(smoke)}.json').write_text(json.dumps(info, indent=1))
    log('wrote h49_hlss')


def jc_sample4(per, seed):
    import uproot
    from interp.part_core import read_root, fof, cat, sub
    rng = np.random.default_rng(seed); Js = []
    for c in ('QCD', 'Zqq', 'Wqq', 'Tbqq'):
        f = fof(c); n = uproot.open(f)['tree'].num_entries; idx = np.sort(rng.choice(n, per, replace=False)); parts = []
        for a in range(0, n, 20000):
            loc = idx[(idx >= a) & (idx < a + 20000)] - a
            if len(loc): parts.append(sub(read_root(f, start=a, stop=min(a + 20000, n)), loc))
        Js.append(cat(parts)); log(f'JetClass {c}: {per} random rows of {n} ({Path(f).name})')
    return cat(Js)


def part_jc(smoke):
    import torch
    from interp.part_core import ParTNetwork, inputs
    from interp.p1v_verify_part import fwd
    from interp.pn_net import load_particlenet
    from interp.h1_irc_panel import pn_logits, jc_table
    from interp.h13_pileup_panel import jc_maker
    from jetdistill.config import MODELS
    dev = os.environ.get('JETDISTILL_DEVICE', 'cpu')
    J0 = jc_sample4(smoke or 2000, SEEDS['jc']); y = J0['label'].astype(int)
    T0 = finalize(jc_table(J0), 128)[0]; make = jc_maker(J0)
    nets = {}
    for n in ('full', 'kin'):
        net = ParTNetwork(n); nets[f'ParT_{n}'] = (lambda J, net=net, n=n: np.asarray(fwd(net, *inputs(J, n))[1], np.float64))
    netp = ParTNetwork('full'); g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in netp.model.parameters():
            if p.numel() > 1:
                f = p.data.flatten().cpu(); p.data.copy_(f[torch.randperm(f.numel(), generator=g)].view_as(p).to(p.device))
    nets['ParT_full_permuted'] = (lambda J, net=netp: np.asarray(fwd(net, *inputs(J, 'full'))[1], np.float64))
    for n, dims in (('full', 17), ('kin', 7)):
        pn = load_particlenet(Path(MODELS).parent / 'particlenet' / f'ParticleNet_{n}.pt', dims, dev)
        nets[f'PN_{n}'] = (lambda J, pn=pn, n=n: pn_logits(pn, J, n))

    def scorer(Uf, ae, ap):
        J = make(Uf, ae, ap); return {mn: f(J) for mn, f in nets.items()}
    info = dict(sample=f'JetClass first test file per class; QCD, Zqq, Wqq, Tbqq {smoke or 2000} random rows each, seed {SEEDS["jc"]}')
    run_dataset('jc', T0, y, scorer, smoke, info)


def part_top(smoke):
    import torch
    from interp import data_top as dt
    from interp.h13_pileup_panel import TopModels, top_J, top_table
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    S = dt.sample(smoke or 3000, SEEDS['top']); y = S['label'].astype(int); T0 = finalize(top_table(S['full']), 200)[0]
    Mo = TopModels(dev); order = np.random.default_rng(SEEDS['top'] + 7).permutation(len(y))

    def scorer(Uf, ae, ap):
        return Mo.score(top_J(Uf, ae, ap), order)
    info = dict(sample=f'Top Landscape test, data_top.sample({smoke or 3000}, {SEEDS["top"]})', idx=S['idx'].tolist())
    run_dataset('top', T0, y, scorer, smoke, info)


# =========================================================================================================== statistics
class Boot:
    def __init__(self, n, B=2000, seed=0):
        self.idx = np.random.default_rng(seed).integers(0, n, (B, n)) if n else None

    def mean(self, x):
        return x[self.idx].mean(1)


def ci(v):
    v = np.asarray(v, np.float64)
    if v.size == 0: return None
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def signflip_p(d, P=10000, seed=1):
    d = np.asarray(d, np.float64); n = len(d)
    if n == 0 or np.all(d == 0): return 1.0
    rng = np.random.default_rng(seed); obs = abs(d.mean()); cnt = 0
    for a in range(0, P, 1000):
        s = rng.choice([-1.0, 1.0], (min(1000, P - a), n)); cnt += int((np.abs((s * d).mean(1)) >= obs - 1e-15).sum())
    return (cnt + 1) / (P + 1)


def mcnemar(a, b):
    """exact two-sided McNemar on paired binary calls a, b"""
    from scipy.stats import binomtest
    n01 = int((a & ~b).sum()); n10 = int((~a & b).sum())
    p = 1.0 if n01 + n10 == 0 else float(binomtest(n01, n01 + n10, 0.5).pvalue)
    return dict(in_only=n01, out_only=n10, p=p)


def eff_at(ss, sb, eps=EPS_B):
    t = np.quantile(sb, 1 - eps); return float((ss > t).mean()), t


def stats_score(Z, mn, cls, b, ds, rng_seed):
    y = Z['y']; v = Z['valid']; sig = v & (y == SIG[ds][cls]); qcd = v & np.isin(y, QCD[ds])
    g = lambda c, rows: Z[f's|{mn}|{cls}|{c}_{b}' if c not in ('TNin', 'TNout') else f's|{mn}|{cls}|{c}_rc'][rows].astype(np.float64)
    out = dict(n_sig=int(sig.sum()), n_qcd=int(qcd.sum()))
    bs = Boot(int(sig.sum()), seed=rng_seed); s0 = g('base', sig)
    conds = EDITS + (['TNin', 'TNout'] if b == 'rc' and f's|{mn}|{cls}|TNin_rc' in Z else [])
    ds_ = {c: g(c, sig) - s0 for c in conds}
    for c in conds:
        out[f'ds_{c}'] = float(ds_[c].mean()); out[f'ds_{c}_ci'] = ci(bs.mean(ds_[c]))
    D = ds_['Gin'] - ds_['Gout']; out['D'] = float(D.mean()); out['D_ci'] = ci(bs.mean(D)); out['D_p'] = signflip_p(D, seed=rng_seed)
    out['D_over_sd_base'] = float(D.mean() / max(s0.std(), 1e-12))
    for ref in ('MLPmsd',):            # MLP(m) has D = 0 by construction (m matched), so no ratio
        k = f's|{ref}|{cls}|'
        if k + 'Gin_st' in Z and not mn.startswith('MLP'):
            Dr = Z[k + 'Gin_st'][sig].astype(np.float64) - Z[k + 'Gout_st'][sig].astype(np.float64)
            out[f'ratio_to_{ref}'] = float(D.mean() / Dr.mean()) if abs(Dr.mean()) > 1e-9 else None
            rb_ = bs.mean(D) / bs.mean(Dr); out[f'ratio_to_{ref}_ci'] = ci(rb_[np.isfinite(rb_)])
    mi, mo_ = bs.mean(ds_['Gin']), bs.mean(ds_['Gout'])
    out['gindex'] = float(1 - ds_['Gout'].mean() / ds_['Gin'].mean()) if abs(ds_['Gin'].mean()) > 1e-9 else None
    gb = 1 - mo_ / np.where(np.abs(mi) > 1e-12, mi, np.nan); out['gindex_ci'] = ci(gb[np.isfinite(gb)]) if np.isfinite(gb).sum() > 100 else None
    if 'TNin' in conds:
        Dt = (ds_['Gin'] - ds_['TNin']) - (ds_['Gout'] - ds_['TNout']); out['D_minus_TN'] = float(Dt.mean()); out['D_minus_TN_ci'] = ci(bs.mean(Dt))
    q0 = g('base', qcd); bq = Boot(int(qcd.sum()), seed=rng_seed + 1)
    Dq = (g('Gin', qcd) - q0) - (g('Gout', qcd) - q0); out['D_qcd'] = float(Dq.mean()); out['D_qcd_ci'] = ci(bq.mean(Dq))
    # signal efficiency at eps_B = 5 % (threshold re-set on each condition's QCD jets) and frozen-threshold calls
    E = {}; t0 = None
    for c in ['base'] + EDITS:
        e, t = eff_at(g(c, sig), g(c, qcd)); E[c] = e
        if c == 'base': t0 = t
    out['effS_reset'] = E
    rb = np.random.default_rng(rng_seed + 2); dd = []
    ssi, sso, sqi, sqo, ss0, sq0 = g('Gin', sig), g('Gout', sig), g('Gin', qcd), g('Gout', qcd), s0, q0
    for _ in range(500):
        i = rb.integers(0, len(ss0), len(ss0)); j = rb.integers(0, len(sq0), len(sq0))
        dd.append(eff_at(ssi[i], sqi[j])[0] - eff_at(sso[i], sqo[j])[0])
    out['effS_in_minus_out'] = E['Gin'] - E['Gout']; out['effS_in_minus_out_ci'] = ci(dd)
    calls = {c: g(c, sig) > t0 for c in ['base'] + EDITS}
    out['effS_frozen'] = {c: float(v.mean()) for c, v in calls.items()}
    out['mcnemar_in_vs_out'] = mcnemar(calls['Gin'], calls['Gout'])
    return out


def stats_window(Z, cls, ds, kind, rng_seed):
    """kind 'm' or 'msd'; W/Z window 65-105 GeV; top 105-210 GeV and tau32 < 0.54"""
    y = Z['y']; v = Z['valid']; sig = v & (y == SIG[ds][cls]); qcd = v & np.isin(y, QCD[ds])
    lo, hi = WIN_T if cls == 't' else WIN_WZ

    def pas(c):
        x = Z[f'{kind}|{c}']; p = (x > lo) & (x < hi)
        if cls == 't': p &= Z[f'tau32|{c}'] < TAU32_CUT
        return p
    P = {c: pas(c) for c in ['base'] + EDITS}
    out = dict(effS={c: float(P[c][sig].mean()) for c in P}, effB={c: float(P[c][qcd].mean()) for c in P})
    d = P['Gin'][sig].astype(float) - P['Gout'][sig].astype(float); bs = Boot(int(sig.sum()), seed=rng_seed)
    out['D_effS'] = float(d.mean()); out['D_effS_ci'] = ci(bs.mean(d)); out['mcnemar_in_vs_out'] = mcnemar(P['Gin'][sig], P['Gout'][sig])
    for c in EDITS:
        e = P[c][sig].astype(float) - P['base'][sig].astype(float); out[f'deffS_{c}'] = float(e.mean()); out[f'deffS_{c}_ci'] = ci(bs.mean(e))
    return out


def load_ds(ds, smoke):
    p = OUT / f'h49_{ds}{sfx(smoke)}.npz'
    if not p.exists(): return None
    Z = dict(np.load(p))
    if ds == 'hls' and (OUT / f'h49_hlss{sfx(smoke)}.npz').exists():
        Zs = np.load(OUT / f'h49_hlss{sfx(smoke)}.npz'); Z.update({k: Zs[k] for k in Zs.files if k != 'y'})
    return Z


def part_summary(smoke):
    res = dict(datasets={}, tests=[]); seed = 100
    for ds in ('hls', 'jc', 'top'):
        Z = load_ds(ds, smoke)
        if Z is None: log(f'{ds}: missing'); continue
        models = sorted({k.split('|')[1] for k in Z if k.startswith('s|')})
        R = dict(models={}, windows={}, calls_changed={})
        for mn in models:
            R['models'][mn] = {}
            for cls in SIG[ds]:
                for b in ('st', 'rc'):
                    if mn.startswith('MLP') and b == 'rc': continue
                    seed += 1; st = stats_score(Z, mn, cls, b, ds, seed); R['models'][mn][f'{cls}_{b}'] = st
                    log(f'{ds} {mn} {cls} {b}: D {st["D"]:+.3f} {st["D_ci"]} p {st["D_p"]:.2g}; ds_in {st["ds_Gin"]:+.3f} out {st["ds_Gout"]:+.3f} '
                        f'c0 {st["ds_c0"]:+.3f} rand {st["ds_rand"]:+.3f}; g {st["gindex"]}; dEffS {st["effS_in_minus_out"]:+.3f} McN p {st["mcnemar_in_vs_out"]["p"]:.2g}')
                    if not mn.startswith('MLP'):
                        res['tests'].append(dict(ds=ds, model=mn, cls=cls, build=b, p_D=st['D_p'], p_mcn=st['mcnemar_in_vs_out']['p']))
            # argmax calls changed (multiclass models), matched set
            if f'arg|{mn}|base_st' in Z:
                v = Z['valid']; R['calls_changed'][mn] = {f'{c}_{b}': float((Z[f'arg|{mn}|{c}_{b}'][v] != Z[f'arg|{mn}|base_{b}'][v]).mean())
                                                          for c in EDITS for b in ('st', 'rc')}
        for cls in SIG[ds]:
            for kind in ('m', 'msd'):
                seed += 1; R['windows'][f'{cls}_{kind}'] = stats_window(Z, cls, ds, kind, seed)
                w = R['windows'][f'{cls}_{kind}']
                log(f'{ds} window {cls} {kind}: effS {w["effS"]} D {w["D_effS"]:+.3f} {w["D_effS_ci"]} McN p {w["mcnemar_in_vs_out"]["p"]:.2g}')
        R['axis_offset_mrad_median'] = {c: float(np.median(Z[f'axoff_mrad|{c}'][Z['valid']])) for c in EDITS}
        js = OUT / f'h49_{ds}{sfx(smoke)}.json'
        if js.exists(): R['info'] = json.loads(js.read_text())
        if ds == 'hls' and (OUT / f'h49_hlss{sfx(smoke)}.json').exists(): R['info_small'] = json.loads((OUT / f'h49_hlss{sfx(smoke)}.json').read_text())
        res['datasets'][ds] = R
    for b in ('rc', 'st'):
        T = [t for t in res['tests'] if t['build'] == b]
        for key in ('p_D', 'p_mcn'):
            h = holm([t[key] for t in T])
            for t, hv in zip(T, h): t[f'holm_{key}'] = hv
    res['holm_note'] = 'Holm within build (rc, st) over every network (nulls included) x signal class x dataset; separately for the D sign-flip test and McNemar'
    (OUT / f'h49_summary{sfx(smoke)}.json').write_text(json.dumps(res, indent=1, default=float))
    log(f'wrote h49_summary{sfx(smoke)}.json')
    try:
        figure(res, smoke)
    except Exception as e:  # noqa: BLE001
        log('figure failed:', repr(e))


def figure(res, smoke):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    Z = load_ds('jc', smoke) or load_ds('hls', smoke)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw=dict(width_ratios=[1, 1.35]))
    a = ax[0]
    if Z is not None and 'ex|j' in Z:
        j = int(Z['ex|j']); pt, de, dp = Z['ex|pt'], Z['ex|deta'], Z['ex|dphi']
        a.scatter(de, dp, s=600 * pt / pt.max(), c='0.55', alpha=0.6, lw=0, label='jet particles (area ∝ pT)')
        G = lambda k: Z[f'G|{k}'][j]
        for nm, dv, mk in (('far prong F', G('dF'), 's'), ('other prong', G('dN'), 'D')):
            a.scatter(*dv, marker=mk, s=90, facecolor='none', edgecolor='k', lw=1.5, label=f'{nm} (soft-drop branch axis)')
        a.scatter(G('eO'), G('pO'), marker='x', s=80, c='#c0392b', lw=2.2, label='G_out: removed by soft drop')
        a.scatter(G('eI'), G('pI'), marker='+', s=110, c='#1f6fb4', lw=2.4, label='G_in: kept by soft drop')
        t = np.linspace(0, 2 * np.pi, 200); r = G('rstar')
        a.plot(r * np.cos(t), r * np.sin(t), ls=':', c='0.3', lw=1); a.plot(0, 0, '+', c='k', ms=8)
        dm = Z['m|Gout'][j] - Z['m|base'][j]; dsi = Z['msd|Gin'][j] - Z['msd|base'][j]; dso = Z['msd|Gout'][j] - Z['msd|base'][j]
        a.set_title(f'One JetClass W jet, three particles of {G("q"):.1f} GeV each\n'
                    f'Δm = +{dm:.1f} GeV for both edits; Δm_SD: G_in +{dsi:.1f}, G_out {dso:+.1f} GeV', fontsize=10)
        L = max(0.45, 1.25 * r); a.set_xlim(-L, L); a.set_ylim(-L, L); a.set_aspect('equal')
        a.set_xlabel('Δη from jet axis'); a.set_ylabel('Δφ from jet axis [rad]'); a.legend(fontsize=7, loc='lower left')
    b = ax[1]; rows = []
    for ds, cls_list in (('hls', ['W', 'Z', 't']), ('jc', ['W', 'Z', 't']), ('top', ['t'])):
        R = res['datasets'].get(ds)
        if not R: continue
        for mn, d in R['models'].items():
            if mn in NULLS: continue
            for cls in cls_list:
                k = f'{cls}_st' if mn.startswith('MLP') else f'{cls}_rc'
                if k in d: rows.append((ds, mn, cls, d[k]))
    ylab = []; yy = 0
    for ds, mn, cls, st in rows:
        mi, mo = st['ds_Gin'], st['ds_Gout']; c0 = st['ds_c0']
        b.plot([mo, mi], [yy, yy], c='0.75', lw=1)
        b.scatter(mi, yy, marker='+', c='#1f6fb4', s=70, lw=2); b.scatter(mo, yy, marker='x', c='#c0392b', s=50, lw=2)
        b.scatter(c0, yy, marker='o', facecolor='none', edgecolor='0.3', s=30)
        ylab.append(f'{ {"hls": "hls4ml", "jc": "JetClass", "top": "TopLand."}[ds]} {LABEL.get(mn, mn)} {cls}'); yy += 1
    b.axvline(0, c='k', lw=0.6); b.set_yticks(range(len(ylab))); b.set_yticklabels(ylab, fontsize=6.5); b.invert_yaxis()
    b.set_xlabel('mean change of the signal-vs-QCD log-odds on signal jets\n(+ G_in kept by soft drop, × G_out removed, ○ c0 count-only)')
    b.set_title('Same added count, pT and ungroomed Δm: does the score\nseparate G_in from G_out like MLP(m_SD) does?', fontsize=10)
    fig.tight_layout(); p = OUT / f'h49_groomed{sfx(smoke)}.png'; fig.savefig(p, dpi=130); log(f'wrote {p}')


# =========================================================================================================== main
if __name__ == '__main__':
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--part', required=True, choices=['hls', 'hlss', 'jc', 'top', 'summary'])
    ap_.add_argument('--smoke', type=int, default=0)
    A = ap_.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    log(f'H49 part {A.part} smoke {A.smoke}')
    dict(hls=part_hls, hlss=part_hlss, jc=part_jc, top=part_top, summary=part_summary)[A.part](A.smoke)
    log('done')
