"""H81 (loop D, round D17; new claim D32, exploratory): does the training dataset or the network architecture set a tagger's
response to physics edits? One edit library, two common jet sets, ParT and ParticleNet trained on JetClass and on Top Landscape.

Jet sets (random rows, never first-k)
  top  Top Landscape test split, 2,000 top + 2,000 QCD (interp.h4_top_wmass._sample(2000, 8101) = data_top.sample rows);
       anti-kT R = 0.8, 550 < pT < 650 GeV, |eta| < 2, Delphes; constituents massless.
  jc   JetClass test TTBar_100 (TbqqP, label Tbqq) + ZJetsToNuNu_100 (QCD), stored 550 < jet_pt < 650 GeV, |jet_eta| < 2, >= 3
       particles, 2,000 random eligible rows per class (interp.h19_wpair_cross.jc_read, rng 8102); constituent masses as stored.
  Every jet is rebuilt through (pT, eta, phi, m) per particle (h4_top_wmass._identity path), sorted by pT, <= 256 slots.
Scores s (top-vs-QCD log-odds; Delta s > 0 = more top-like)
  JetClass models: s = logit_Tbqq - logit_QCD = log p(TbqqP) / p(QCD) (10-class softmax, the two-class log ratio).
  Top Landscape models: s = the two-class top log-odds (output[top] - output[QCD]).
Models (kinematic inputs only: pT, E, pT/pT_jet, E/E_jet, Delta R, Delta eta, Delta phi per particle; ParT also pair features)
  JC_ParT_rel  released JetClass ParT_kin (authors, full 100M training set)
  JC_ParT_s0..s2  ParT_kin retrained from scratch on the 2M JetClass subset, torch seeds 0..2 (P10, models/part_seeds/kin_s<k>)
  JC_PN_rel    released JetClass ParticleNet_kin
  Top_ParT_s0 / _s1  Top Landscape ParT_kin, authors' TopLandscape recipe (H4 / H39: models/top/part_kin, part_kin_s1)
  Top_PN_s0 / _s1    Top Landscape ParticleNet_kin (models/top/pnet_kin, pnet_kin_s1)
  references: Top_LNet (released LorentzNet), Top_PFN, Top_EFN (H2 energyflow), CMS = m_SD + tau32 working point call
  (105 < m_SD < 210 GeV and tau32 < 0.54; s = 1 if tagged else 0), LR_N = histogram log-likelihood ratio of the constituent
  count N (2-fold cross-fit on the unedited jets of the same jet set).
  Top-trained networks are given massless constituents (E = |p|, their training convention, as H19 / H31); JetClass-trained
  networks get the stored constituent masses (zero for Top Landscape jets).
Input axis: every edited jet is rebuilt about its own E-scheme axis (direction of the summed constituent 3-momenta), except E5 / C5.
Edit library (edit E_k and its standard control C_k; r_k uses the difference E_k - C_k)
  E1 exact collinear split of the 10 softest particles: each becomes two particles with half its pT and half its mass at the same
     (eta, phi). N +10; jet four-momentum and every IRC-safe observable (EFPs, m_SD, tau_N) unchanged exactly.
     C1 the same 10 particles each displaced by Delta R = 1e-3 in a random direction, no particle added (count fixed).
  E2 5 massless particles of pT 0.5 GeV each at uniform random positions in the disc Delta R < 0.8 about the axis.
     C2 the same 2.5 GeV added to the hardest particle's pT (direction and mass fixed): same summed pT, no new particle.
  E3 U(1.1) uniform energy scale: every constituent four-momentum x 1.1 (m, pT x 1.1; m / pT and all angles fixed).
     C3 UB(1.1): U(1.1) followed by a boost along the jet momentum back to the original |P| (H71): m x 1.1 at fixed pT.
  E4 B(1.1) mass-preserving boost along the jet momentum with |P| x 1.1 (H12): pT x 1.1, m fixed, angles x ~1/1.1.
     C4 P(1.1) pT-only scale (= U(1.1), H12 'P'): pT x 1.1 and m x 1.1 at fixed angles.
  E5 inputs about the winner-take-all (WTA, C/A clustering with winner-take-all recombination) axis instead of the E-scheme axis
     (every particle's (Delta eta, Delta phi) shifted by the E-scheme - WTA offset; the particles themselves unchanged).
     C5 inputs about an axis displaced from the E-scheme axis by the same length in a uniformly random direction.
  E6 W decay-angle rotation at fixed m_top and m_W (H31 'D', dc = +0.3): three exclusive-kT subjets, W pair = the pair with mass
     closest to 80.4 GeV; the W-pair constituents rotated rigidly in the W rest frame so that |cos theta*| -> |cos theta*| + 0.3
     (clipped at 0.995). Jet four-momentum, m_W (pair mass) and subjet internal structure fixed.
     C6 H31 'cphi': the same W system rotated in the W rest frame about the b-subjet direction u by the same angle |alpha|
     (random sign): |cos theta*|, all pair masses fixed; constituent positions move by a comparable amount.
  E7 H72 'rmm' at r_cut = 0.4: all particles beyond Delta R 0.4 of the axis removed, then jet m and pT restored (one momentum scale
     a and one angular scale b about the remaining jet's axis, 8 iterations; valid if |dm/m| < 2 %, |dpT/pT| < 1 %, b <= 5).
     C7 H72 'crm': a pT-matched set of random inside particles removed (same count, summed pT within 20 %), then the same m, pT
     restore. Analysis set of E7 = jets valid for both.
Response vector per model and jet set: r_k = mean over valid jets of (s(E_k) - s(C_k)) / SD(s on unedited jets of that set).

Steps
  --step run  [--models all|<comma list>] (GPU)  edits + scores; writes $AI_RESULTS/physics/h81_edits.npz, h81_s_<model>.npz,
              h81_run.json (unedited AUC / accuracy per model x jet set, edit diagnostics), h81_example.npz (figure jet).
  --step stats (CPU)  response vectors, bootstrap (B = 1000 over jets), cosine matrix, variance decomposition (dataset,
              architecture, interaction, seed), label permutation test, jet-set effect, Holm -> h81_stats.json
Run
  ./part_job.sh d-h81-run 'python -m interp.h81_dataset_vs_arch --step run'
  ./job.sh d-h81-stats 1 2 'python -m interp.h81_dataset_vs_arch --step stats && python -m interp.h81_plot'
  --smoke N: N jets per class, files with suffix _smoke
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

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
OUT = RES / 'physics'
V = Path('/j-jepa-vol/aaron-interp')
SEED_TOP, SEED_JC, SEED_EDIT = 8101, 8102, 8104
CAP = 256
EDITS = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7']
ENAME = dict(E1='collinear split x10', E2='5 soft particles', E3='energy scale U(1.1)', E4='boost B(1.1)', E5='WTA axis',
             E6='W decay angle', E7='periphery removal + m,pT restore')
CONDS = ['base'] + [f'{x}{k}' for k in range(1, 8) for x in ('E', 'C')]
JC_NETS = ['JC_ParT_rel', 'JC_ParT_s0', 'JC_ParT_s1', 'JC_ParT_s2', 'JC_PN_rel']
TOP_NETS = ['Top_ParT_s0', 'Top_ParT_s1', 'Top_PN_s0', 'Top_PN_s1']
MAIN = JC_NETS + TOP_NETS
REFN = ['Top_LNet', 'Top_PFN', 'Top_EFN', 'CMS', 'LR_N']
FAC = {m: ('JC' if m.startswith('JC') else 'Top', 'ParT' if '_ParT_' in m else 'PN') for m in MAIN}
KNOWN = dict(top=dict(Top_ParT_s0='AUC 0.9862 / acc 93.7 % (H32, Top sample)', Top_PN_s0='0.9845 / 93.8 %', Top_LNet='0.9867 / 94.3 %',
                      Top_PFN='0.9782 / 92.7 %', Top_EFN='0.9728 / 91.5 %', JC_ParT_rel='0.9815 / 92.4 %', JC_PN_rel='0.9821 / 92.3 %',
                      Top_PN_s1='Top test metric (acc) 0.9403 (H39)'),
             jc=dict(Top_ParT_s0='0.9849 / 94.0 % (H32, JetClass sample)', Top_PN_s0='0.9836 / 94.0 %', Top_LNet='0.9860 / 94.2 %',
                     Top_PFN='0.9792 / 92.0 %', Top_EFN='0.9773 / 92.7 %', JC_ParT_rel='0.9920 / 95.8 %', JC_PN_rel='0.9885 / 94.9 %',
                     JC_ParT_s0='10-class test acc 0.715 (P10)', JC_ParT_s1='0.708', JC_ParT_s2='0.713'))
T0_ = time.time()


def log(*a):
    print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


def sfx(smoke):
    return '_smoke' if smoke else ''


# ===================================================================================================================== tables
def table_from_P4(P4, nn):
    from interp.h18_softcount_cross import eaxis
    P = P4.astype(np.float64); L = P.shape[1]; real = np.arange(L)[None, :] < nn[:, None]
    pt = np.hypot(P[..., 0], P[..., 1]); real &= pt > 0; ptc = np.where(real, pt, 1.0)
    eta = np.arcsinh(np.where(real, P[..., 2], 0) / ptc); phi = np.arctan2(P[..., 1], P[..., 0])
    m = np.sqrt(np.maximum(P[..., 3] ** 2 - (P[..., :3] ** 2).sum(-1), 0))
    T = dict(pt=np.where(real, pt, 0), eta=np.where(real, eta, 0), phi=np.where(real, phi, 0), m=np.where(real, m, 0), ext={})
    T['ax_eta'], T['ax_phi'] = eaxis(T); T['sgn'] = np.where(T['ax_eta'] < 0, -1.0, 1.0)
    return T


def P4_of(T):
    from interp.h1_irc_panel import p3
    real = T['pt'] > 0; px, py, pz = p3(T['pt'], T['eta'], T['phi']); E = np.sqrt((T['pt'] * np.cosh(T['eta'])) ** 2 + T['m'] ** 2)
    P = np.stack([np.where(real, a, 0) for a in (px, py, pz, E)], -1)
    return P, real.sum(1).astype(np.int64)


def fin(U):
    from interp.h1_irc_panel import finalize
    U = dict(U); U.setdefault('sgn', np.where(U['ax_eta'] < 0, -1.0, 1.0)); U.setdefault('ext', {})
    Uf, over = finalize(U, CAP)
    return Uf, over


def own_axis(Uf):
    from interp.h18_softcount_cross import eaxis
    return eaxis(Uf)


# ===================================================================================================================== edits
def softest_slots(T, k):
    n = (T['pt'] > 0).sum(1)
    return [np.where(n - 1 - i >= 0, n - 1 - i, -1) for i in range(k)]


def e1_split(T):
    from interp.h1_irc_panel import copyT
    U = copyT(T, 10); W = T['pt'].shape[1]; rows = np.arange(len(T['pt']))
    for i, s in enumerate(softest_slots(T, 10)):
        ok = s >= 0; r = rows[ok]; ss = s[ok]
        for k in ('pt', 'm'):
            half = T[k][r, ss] / 2; U[k][r, ss] = half; U[k][r, W + i] = half
        U['eta'][r, W + i] = T['eta'][r, ss]; U['phi'][r, W + i] = T['phi'][r, ss]
    return U


def c1_jitter(T, rng, d=1e-3):
    from interp.h1_irc_panel import copyT, wrap
    U = copyT(T); rows = np.arange(len(T['pt']))
    for s in softest_slots(T, 10):
        ok = s >= 0; r = rows[ok]; ss = s[ok]; th = 2 * np.pi * rng.random(len(r))
        U['eta'][r, ss] = T['eta'][r, ss] + d * np.cos(th); U['phi'][r, ss] = wrap(T['phi'][r, ss] + d * np.sin(th))
    return U


def e2_soft(T, rng, k=5, q=0.5, R=0.8):
    from interp.h1_irc_panel import copyT, wrap
    J = len(T['pt']); U = copyT(T, k); W = T['pt'].shape[1]
    r = R * np.sqrt(rng.random((J, k))); th = 2 * np.pi * rng.random((J, k))
    U['pt'][:, W:] = q; U['m'][:, W:] = 0.0
    U['eta'][:, W:] = T['ax_eta'][:, None] + r * np.cos(th); U['phi'][:, W:] = wrap(T['ax_phi'][:, None] + r * np.sin(th))
    return U


def c2_hard(T, q=2.5):
    from interp.h1_irc_panel import copyT
    U = copyT(T); U['pt'][:, 0] = T['pt'][:, 0] + q; return U


def scaleU(T, lam):
    from interp.h1_irc_panel import copyT
    U = copyT(T); U['pt'] = T['pt'] * lam; U['m'] = T['m'] * lam; return U


def boostB(T, lam):
    from interp.h12_boost_panel import boost
    U, _ = boost(T, lam); U['sgn'] = np.where(U['ax_eta'] < 0, -1.0, 1.0); return U


def e6_rot(T, seed):
    """H31 edit_jet D (dc = +0.3) and cphi on every jet with >= 3 particles; returns (T_D, T_cphi, valid, info)"""
    from interp import h31_top_dalitz as D31
    from interp import h4_top_wmass as H
    P4, nn = P4_of(T); rows = np.flatnonzero(nn >= 3); F = np.ascontiguousarray(P4[rows])
    LAB = H.kt3_batch(F, nn[rows]); PD, PC = P4.copy(), P4.copy(); ok = np.zeros(len(nn), bool)
    al = np.full(len(nn), np.nan); c0 = np.full(len(nn), np.nan); c1 = np.full(len(nn), np.nan); reasons = {}
    for ii, j in enumerate(rows):
        n = nn[j]; r = D31.edit_jet(P4[j, :n], LAB[ii, :n], 0.3, seed * 100000 + int(j))
        rs = r['D'][1]['reason']; reasons[rs] = reasons.get(rs, 0) + 1
        if r['D'][0] is None or r['cphi'][0] is None: continue
        PD[j, :n] = r['D'][0]; PC[j, :n] = r['cphi'][0]; ok[j] = True
        al[j] = r['D'][1]['alpha']; c0[j] = r['D'][1]['c0']; c1[j] = r['D'][1]['c1']
    TD, TC = table_from_P4(PD, nn), table_from_P4(PC, nn)
    return TD, TC, ok, dict(reasons=reasons, alpha_median=float(np.nanmedian(np.abs(al))), c0_median=float(np.nanmedian(c0)),
                            c1_median=float(np.nanmedian(c1)), n_valid=int(ok.sum()))


def e7_remove(T, seed, r_cut=0.4):
    from interp.h72_angular_reach import geom, remove, restore, match_inside
    from interp.h1_irc_panel import jet4
    ee, ep, de, dp, dr, real = geom(T); P0 = jet4(T); out = real & (dr > r_cut)
    Urm = remove(T, out); Vr, ok_r, b_r = restore(Urm, P0)
    rem, okc, ratio = match_inside(T, out, np.random.default_rng([seed, 7]))
    Ucr = remove(T, rem); Vc, ok_c, b_c = restore(Ucr, P0)
    for X in (Vr, Vc): X['sgn'] = np.where(X['ax_eta'] < 0, -1.0, 1.0)
    valid = ok_r & okc & ok_c
    return Vr, Vc, valid, dict(n_out_mean=float(out.sum(1).mean()), frac_jets_with_outer=float(out.any(1).mean()),
                               fpt_out_mean=float(((T['pt'] * out).sum(1) / T['pt'].sum(1)).mean()), n_valid=int(valid.sum()),
                               b_rmm_median=float(np.median(b_r[valid])) if valid.any() else None)


def conditions(T0, ds):
    """yields (name, finalized table, axis (eta, phi), valid (J,), info); T0 finalized"""
    from interp.h13_pileup_panel import wta_axes
    from interp.h1_irc_panel import wrap
    J = len(T0['pt']); allv = np.ones(J, bool); sd = SEED_EDIT + (0 if ds == 'top' else 50)

    def own(U, valid=allv, info=None):
        Uf, over = fin(U); return Uf, own_axis(Uf), valid, dict(info or {}, overflow_frac=float(over.mean()))
    yield ('base',) + own(T0)
    yield ('E1',) + own(e1_split(T0))
    yield ('C1',) + own(c1_jitter(T0, np.random.default_rng([sd, 1])))
    yield ('E2',) + own(e2_soft(T0, np.random.default_rng([sd, 2])))
    yield ('C2',) + own(c2_hard(T0))
    yield ('E3',) + own(scaleU(T0, 1.1))
    yield ('C3',) + own(boostB(scaleU(T0, 1.1), 1 / 1.1))
    yield ('E4',) + own(boostB(T0, 1.1))
    yield ('C4',) + own(scaleU(T0, 1.1))
    ae, ap = own_axis(T0); we, wp = wta_axes(dict(T0, ax_eta=ae, ax_phi=ap))
    off = np.hypot(we - ae, wrap(wp - ap)); th = 2 * np.pi * np.random.default_rng([sd, 5]).random(J)
    inf5 = dict(offset_median=float(np.median(off)), offset_mean=float(off.mean()))
    yield 'E5', T0, (we, wp), allv, inf5
    yield 'C5', T0, (ae + off * np.cos(th), wrap(ap + off * np.sin(th))), allv, inf5
    TD, TC, ok6, inf6 = e6_rot(T0, sd)
    yield ('E6',) + own(TD, ok6, inf6)
    yield ('C6',) + own(TC, ok6, inf6)
    Vr, Vc, ok7, inf7 = e7_remove(T0, sd)
    yield ('E7',) + own(Vr, ok7, inf7)
    yield ('C7',) + own(Vc, ok7, inf7)


# ===================================================================================================================== models
class Nets:
    def __init__(self, names):
        import torch
        from interp import h4_top_wmass as H
        self.torch = torch; self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'; self.f = {}; self.missing = {}
        for nm in names:
            try:
                self.f[nm] = self._load(nm, H)
            except FileNotFoundError as e:
                self.missing[nm] = str(e); log(f'model {nm} not available: {e}')
        log(f'models on {self.dev}: {list(self.f)}')

    def _load(self, nm, H):
        torch = self.torch
        if nm.startswith('JC_ParT'):
            from interp.part_core import ParTNetwork
            net = ParTNetwork('kin')
            if not nm.endswith('rel'):
                p = V / 'models/part_seeds' / f'kin_s{nm[-1]}' / 'net_best_epoch_state.pt'
                if not p.exists(): raise FileNotFoundError(p)
                sd = torch.load(p, map_location='cpu'); sd = {k[4:] if k.startswith('mod.') else k: v for k, v in sd.items()}
                net.model.load_state_dict(sd, strict=True); net.model.eval()
            return ('jcpart', net)
        if nm == 'JC_PN_rel':
            from interp.pn_net import load_particlenet
            from jetdistill.config import MODELS
            return ('jcpn', load_particlenet(Path(MODELS).parent / 'particlenet' / 'ParticleNet_kin.pt', 7, self.dev))
        if nm.startswith('Top_ParT') or nm.startswith('Top_PN'):
            d = ('part_kin' if 'ParT' in nm else 'pnet_kin') + ('' if nm.endswith('s0') else '_s1')
            p = V / 'models/top' / d / 'net_best_epoch_state.pt'
            if not p.exists() or (nm.endswith('s1') and not (V / 'models/top' / d / 'net_epoch-19_state.pt').exists()):
                raise FileNotFoundError(f'{p} (or training not finished: no net_epoch-19_state.pt)')
            return ('weaver', H.WeaverNet('ParT' if 'ParT' in nm else 'PN', p, self.dev))
        if nm == 'Top_LNet': return ('lnet', H.LNet(self.dev, mass64=True))
        if nm in ('Top_PFN', 'Top_EFN'): return ('flow', H.Flow(nm[4:].lower()))
        raise KeyError(nm)

    def score(self, Uf, ae, ap, ds, order):
        from interp.part_core import build, inputs, PK
        from interp.p1v_verify_part import fwd
        from interp.h1_irc_panel import pn_logits, wrap
        from interp.h13_pileup_panel import top_J
        from interp.h4_top_wmass import weaver_inputs
        torch = self.torch; out = {}; real = Uf['pt'] > 0; J = len(ae)
        if any(k in ('jcpart', 'jcpn') for k, _ in self.f.values()):
            sg = np.where(ae < 0, -1.0, 1.0)[:, None]
            de = np.where(real, sg * (Uf['eta'] - ae[:, None]), 0); dp = np.where(real, wrap(Uf['phi'] - ap[:, None]), 0)
            Jf = dict(jet_eta=ae.astype(np.float32), jet_phi=ap.astype(np.float32), label=np.zeros(J, np.int32))
            ext = {k: np.zeros_like(Uf['pt']) for k in PK if k not in ('px', 'py', 'pz', 'energy', 'deta', 'dphi')}
            Jjc = build(Jf, Uf['pt'], de, dp, Uf['m'], extra=ext); xin = inputs(Jjc, 'kin')
        Um = dict(Uf); Um['m'] = np.zeros_like(Uf['m']) if ds == 'jc' else Uf['m']     # Top-trained nets: massless constituents
        Jt = top_J(Um, ae, ap); nnt = Jt['mask'].sum(1).astype(int)
        for nm, (kind, net) in self.f.items():
            if kind == 'jcpart':
                L = fwd(net, *xin)[1]; out[nm] = L[:, 8] - L[:, 0]
            elif kind == 'jcpn':
                L = pn_logits(net, Jjc, 'kin'); out[nm] = L[:, 8] - L[:, 0]
            elif kind == 'weaver':
                s = np.zeros(J)
                with torch.no_grad():
                    for a in range(0, J, 512):
                        Jb = {k: v[a:a + 512] for k, v in Jt.items()}; pts, x, v, m = weaver_inputs(Jb)
                        d = lambda t: torch.from_numpy(t).to(self.dev)
                        o = (net.model(d(x), v=d(v), mask=d(m)) if net.kind == 'ParT' else net.model(d(pts), d(x), d(m))).float().cpu().numpy()
                        s[a:a + len(o)] = o[:, 0] - o[:, 1]
                out[nm] = s
            elif kind == 'lnet':
                Lm = int(nnt.max()); P4 = np.stack([Jt[k][:, :Lm] for k in ('px', 'py', 'pz', 'energy')], -1).astype(np.float32)
                out[nm] = net.score(P4, nnt, order)
            elif kind == 'flow':
                k3 = np.stack([Jt['pt'][:, :128], Jt['deta'][:, :128], Jt['dphi'][:, :128]], -1).astype(np.float16).astype(np.float32)
                out[nm] = net.score(k3)
        return out


# ===================================================================================================================== samples
def load_sets(smoke):
    from interp import h4_top_wmass as H
    from interp import h19_wpair_cross as X
    from interp.part_core import cat
    per = smoke or 2000; S = {}
    St = H._sample(per, SEED_TOP); nn = np.minimum(St['jet'][:, 5].astype(np.int64), St['full'].shape[1])
    P4 = H._identity(St['full'][..., :4].astype(np.float32), nn); y = St['label'].astype(int)
    S['top'] = dict(P4=P4, nn=nn, y=y, info=dict(source='Top Landscape test', per_class=per, rng=SEED_TOP, idx_first=St['idx'][:5].tolist()))
    log(f'top set: {len(y)} jets ({y.sum()} top), <N> {nn.mean():.1f}')
    Jt, it, inft = X.jc_read('Tbqq', 100, per, SEED_JC); Jq, iq, infq = X.jc_read('QCD', 100, per, SEED_JC)
    J = cat([Jt, Jq]); P4r, nn = X.jc_P4(J); P4 = H._identity(P4r, nn); y = (J['label'] == 8).astype(int)
    S['jc'] = dict(P4=P4, nn=nn, y=y, info=dict(source='JetClass test TTBar_100 / ZJetsToNuNu_100, stored 550<pT<650, |eta|<2', Tbqq=inft,
                                                QCD=infq, rng=SEED_JC, stored_axis=(J['jet_eta'], J['jet_phi'])))
    log(f'jc set: {len(y)} jets ({y.sum()} top), <N> {nn.mean():.1f}')
    return S


def classical(Uf, nfit=None):
    from interp import h4_top_wmass as H
    P4, nn = P4_of(Uf); O = H._obs(P4.astype(np.float32), nn)
    return dict(m=O['m'], mSD=O['mSD'], tau32=O['tau32'], N=nn.astype(float), pt=np.hypot(P4[..., 0].sum(1), P4[..., 1].sum(1)))


def lrn_fit(N, y, folds):
    """2-fold cross-fit histogram LR of N: returns function(N_new) -> log-ratio per jet using the other fold's histogram"""
    bins = np.arange(0, 302)
    H = {}
    for f in (0, 1):
        tr = folds != f
        h1 = np.histogram(np.clip(N[tr & (y == 1)], 0, 300), bins)[0] + 1.0; h0 = np.histogram(np.clip(N[tr & (y == 0)], 0, 300), bins)[0] + 1.0
        H[f] = np.log(h1 / h1.sum()) - np.log(h0 / h0.sum())

    def apply(Nn):
        k = np.clip(Nn, 0, 300).astype(int); return np.where(folds == 0, H[0][k], H[1][k])
    return apply


def auc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def step_run(models, smoke):
    OUT.mkdir(parents=True, exist_ok=True)
    S = load_sets(smoke)
    names = MAIN + REFN[:3] if models == 'all' else models.split(',')
    nets = Nets(names); info = dict(node='H81', smoke=bool(smoke), models_missing=nets.missing, sets={}, unedited={}, conds={})
    E = {}; Sc = {nm: {} for nm in nets.f}; ex = {}
    for ds, D in S.items():
        T0 = fin(table_from_P4(D['P4'], D['nn']))[0]; y = D['y']; J = len(y)
        order = np.random.default_rng(SEED_EDIT + 7).permutation(J)
        if ds == 'jc':
            se, sp = D['info'].pop('stored_axis'); ae0, ap0 = own_axis(T0)
            from interp.h1_irc_panel import wrap
            dax = np.hypot(ae0 - se, wrap(ap0 - sp)); D['info']['own_vs_stored_axis_mrad_median'] = float(1e3 * np.median(dax))
            log(f'jc: E-scheme axis vs stored axis, median offset {1e3 * np.median(dax):.3f} mrad')
        info['sets'][ds] = D['info']; info['conds'][ds] = {}
        E[f'{ds}|y'] = y; folds = np.random.default_rng([SEED_EDIT, 9, len(ds)]).integers(0, 2, J); lr = None
        for name, Uf, (ae, ap), valid, inf in conditions(T0, ds):
            t = time.time(); sc = nets.score(Uf, ae, ap, ds, order)
            for nm, s in sc.items(): Sc[nm][f'{ds}|{name}'] = np.asarray(s, np.float32)
            O = classical(Uf)
            if name == 'base': lr = lrn_fit(O['N'], y, folds); O0 = O
            cms = ((O['mSD'] > 105) & (O['mSD'] < 210) & (O['tau32'] < 0.54)).astype(np.float32)
            E[f'{ds}|{name}|CMS'] = cms; E[f'{ds}|{name}|LR_N'] = lr(O['N']).astype(np.float32); E[f'{ds}|{name}|valid'] = valid
            for k in ('m', 'mSD', 'tau32', 'N', 'pt'): E[f'{ds}|{name}|obs_{k}'] = O[k].astype(np.float32)
            d = {k: float(np.mean((O[k] - O0[k])[valid])) for k in ('m', 'mSD', 'tau32', 'N', 'pt')}
            info['conds'][ds][name] = dict(inf, n_valid=int(valid.sum()), mean_change=d, CMS_tag_rate=float(cms.mean()))
            log(f'{ds} {name}: valid {valid.sum()}/{J} | dm {d["m"]:+.3f} dmSD {d["mSD"]:+.3f} dtau32 {d["tau32"]:+.4f} dN {d["N"]:+.2f} '
                f'dpT {d["pt"]:+.2f} | ' + ' '.join(f'{nm} {np.mean((sc[nm] - Sc[nm][f"{ds}|base"])[valid]):+.3f}' for nm in sc) +
                f' ({time.time() - t:.0f} s)')
            if ds == 'top' and name in ('base', 'E1', 'E6', 'C6'):
                if name == 'base':
                    nr = (T0['pt'] > 0).sum(1); cand = np.flatnonzero((y == 1) & (nr > 40) & (nr < 70))
                    ex['row'] = int(cand[0]) if len(cand) else 0
                j = ex.get('row', 0); r = Uf['pt'][j] > 0
                ex[f'{name}|pt'] = Uf['pt'][j, r]; ex[f'{name}|deta'] = Uf['eta'][j, r] - ae[j]
                ex[f'{name}|dphi'] = (Uf['phi'][j, r] - ap[j] + np.pi) % (2 * np.pi) - np.pi
        # unedited performance
        info['unedited'][ds] = {}
        for nm in nets.f:
            s = Sc[nm][f'{ds}|base'].astype(np.float64)
            info['unedited'][ds][nm] = dict(AUC=auc(s, y), acc_s_gt_0=float(((s > 0) == (y == 1)).mean()), sd_s=float(s.std()))
            log(f'{ds} unedited {nm}: AUC {info["unedited"][ds][nm]["AUC"]:.4f} acc {info["unedited"][ds][nm]["acc_s_gt_0"]:.4f}  known: {KNOWN[ds].get(nm, "-")}')
        for nm in ('CMS', 'LR_N'):
            s = E[f'{ds}|base|{nm}'].astype(np.float64)
            info['unedited'][ds][nm] = dict(AUC=auc(s, y), acc_s_gt_0=float(((s > (0.5 if nm == 'CMS' else 0)) == (y == 1)).mean()),
                                            eps_S=float(s[y == 1].mean()) if nm == 'CMS' else None, eps_B=float(s[y == 0].mean()) if nm == 'CMS' else None)
            log(f'{ds} unedited {nm}: {info["unedited"][ds][nm]}')
    s_ = sfx(smoke)
    for nm, d in Sc.items(): np.savez_compressed(OUT / f'h81_s_{nm}{s_}.npz', **d)
    if models == 'all':
        np.savez_compressed(OUT / f'h81_edits{s_}.npz', **E); np.savez_compressed(OUT / f'h81_example{s_}.npz', **ex)
        (OUT / f'h81_run{s_}.json').write_text(json.dumps(info, indent=1, default=float))
    else:
        p = OUT / f'h81_run{s_}.json'; R = json.loads(p.read_text()) if p.exists() else dict(unedited={})
        for ds in info['unedited']:
            R['unedited'].setdefault(ds, {}).update({k: v for k, v in info['unedited'][ds].items() if k in nets.f})
        R.setdefault('added_later', []).extend(list(nets.f)); p.write_text(json.dumps(R, indent=1, default=float))
    log('wrote', [f'h81_s_{nm}{s_}.npz' for nm in Sc], 'h81_run json')


# ===================================================================================================================== stats
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for i, k in enumerate(o):
        run = max(run, min(1.0, (m - i) * ps[k])); adj[k] = run
    return adj


def signflip(d, rng, P=2000):
    d = np.asarray(d, float)
    if d.size == 0 or np.allclose(d, 0): return 1.0
    obs = abs(d.mean()); sg = rng.choice([-1.0, 1.0], (P, len(d)))
    return float((1 + (np.abs((sg * d).mean(1)) >= obs - 1e-15).sum()) / (P + 1))


def ci(a):
    a = np.asarray(a, float); return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]


def cells_def(avail, jc_part='rel'):
    c = {('JC', 'ParT'): ['JC_ParT_rel'] if jc_part == 'rel' else [m for m in ('JC_ParT_s0', 'JC_ParT_s1', 'JC_ParT_s2') if m in avail],
         ('JC', 'PN'): ['JC_PN_rel'], ('Top', 'ParT'): [m for m in ('Top_ParT_s0', 'Top_ParT_s1') if m in avail],
         ('Top', 'PN'): [m for m in ('Top_PN_s0', 'Top_PN_s1') if m in avail]}
    return {k: v for k, v in c.items() if v}


def seed_groups(avail):
    g = [[m for m in ('JC_ParT_s0', 'JC_ParT_s1', 'JC_ParT_s2') if m in avail], [m for m in ('Top_ParT_s0', 'Top_ParT_s1') if m in avail],
         [m for m in ('Top_PN_s0', 'Top_PN_s1') if m in avail]]
    return [x for x in g if len(x) >= 2]


def decomp(R, cells, groups):
    """R: {model: (..., E) array}; returns dict of arrays (..., E): a (dataset), b (architecture), c (interaction) half-contrasts,
    s2 (pooled within-group seed variance), k (= sum 1/n_cell / 16), df"""
    mu = {k: np.mean([R[m] for m in v], 0) for k, v in cells.items()}
    a = (mu['JC', 'ParT'] + mu['JC', 'PN'] - mu['Top', 'ParT'] - mu['Top', 'PN']) / 4
    b = (mu['JC', 'ParT'] - mu['JC', 'PN'] + mu['Top', 'ParT'] - mu['Top', 'PN']) / 4
    c = (mu['JC', 'ParT'] - mu['JC', 'PN'] - mu['Top', 'ParT'] + mu['Top', 'PN']) / 4
    ss = 0; df = 0
    for g in groups:
        X = np.stack([R[m] for m in g]); ss = ss + ((X - X.mean(0)) ** 2).sum(0); df += len(g) - 1
    s2 = ss / max(df, 1); k = sum(1 / len(v) for v in cells.values()) / 16
    return dict(a=a, b=b, c=c, s2=s2, k=k, df=df)


def shares(dc, axis=None):
    """variance components (bias-corrected: a^2 - k s2, floored at 0) and shares; axis=-1 pools (sums) over edits"""
    k, s2 = dc['k'], dc['s2']
    comp = {f: np.maximum(dc[x] ** 2 - k * s2, 0) for f, x in (('dataset', 'a'), ('architecture', 'b'), ('interaction', 'c'))}
    comp['seed'] = s2
    raw = {f: dc[x] ** 2 for f, x in (('dataset', 'a'), ('architecture', 'b'), ('interaction', 'c'))}
    if axis is not None:
        comp = {f: v.sum(axis) for f, v in comp.items()}; raw = {f: v.sum(axis) for f, v in raw.items()}
    tot = sum(comp.values())
    return {f: v / np.where(tot > 0, tot, 1) for f, v in comp.items()}, comp, raw


def cosm(R, names, axis=-1):
    X = np.stack([R[m] for m in names]); n = np.linalg.norm(X, axis=axis, keepdims=True)
    Xn = X / np.where(n > 0, n, 1)
    return np.einsum('i...e,j...e->ij...', Xn, Xn)


def perm_T(C, labs, pairs_mask_fn):
    """T = mean cos over (same dataset, different arch) pairs - mean cos over (same arch, different dataset) pairs"""
    n = len(labs); sd, sa = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if labs[i][0] == labs[j][0] and labs[i][1] != labs[j][1]: sd.append(C[i, j])
            elif labs[i][1] == labs[j][1] and labs[i][0] != labs[j][0]: sa.append(C[i, j])
    return np.mean(sd, 0) - np.mean(sa, 0), np.mean(sd, 0), np.mean(sa, 0)


def step_stats(smoke, B=1000, P=10000):
    from scipy.stats import f as fdist
    s_ = sfx(smoke); rng = np.random.default_rng(8199)
    Ez = np.load(OUT / f'h81_edits{s_}.npz'); run = json.loads((OUT / f'h81_run{s_}.json').read_text())
    avail = [m for m in MAIN + REFN[:3] if (OUT / f'h81_s_{m}{s_}.npz').exists()]
    Sz = {m: np.load(OUT / f'h81_s_{m}{s_}.npz') for m in avail}
    main = [m for m in MAIN if m in avail]; refs = [m for m in REFN if m in avail or m in ('CMS', 'LR_N')]
    allm = main + refs
    log('models', allm)
    get = lambda m, ds, c: (Ez[f'{ds}|{c}|{m}'] if m in ('CMS', 'LR_N') else Sz[m][f'{ds}|{c}']).astype(np.float64)
    out = dict(node='H81', smoke=bool(smoke), B=B, models=allm, main=main, factor_labels={m: FAC[m] for m in main}, unedited=run['unedited'],
               conds=run.get('conds'), sets=run.get('sets'), per_set={})
    Rpt, Rb, Dj = {}, {}, {}
    for ds in ('jc', 'top'):
        y = Ez[f'{ds}|y']; J = len(y); idx = rng.integers(0, J, (B, J))
        Rpt[ds], Rb[ds], Dj[ds] = {}, {}, {}
        res = dict(r={}, r_ci={}, r_by_class={}, p_signflip={}, sd_base={})
        for m in allm:
            s0 = get(m, ds, 'base'); sd = s0.std(); sdb = s0[idx].std(1)
            rp, rb, pp, rc = [], [], [], {}
            for e in EDITS:
                k = e[1]; w = Ez[f'{ds}|E{k}|valid'].astype(float); d = (get(m, ds, f'E{k}') - get(m, ds, f'C{k}')) * w
                Dj[ds][m, e] = (d, w, sd)
                rp.append(d.sum() / w.sum() / sd); rb.append(d[idx].sum(1) / w[idx].sum(1) / sdb)
                pp.append(signflip(d[w > 0], rng)); rc[e] = {c: float(d[(y == c) & (w > 0)].mean() / sd) for c in (0, 1)}
            Rpt[ds][m] = np.array(rp); Rb[ds][m] = np.stack(rb, -1)            # (E,), (B, E)
            res['r'][m] = Rpt[ds][m].tolist(); res['r_ci'][m] = [ci(Rb[ds][m][:, i]) for i in range(len(EDITS))]
            res['p_signflip'][m] = pp; res['r_by_class'][m] = rc; res['sd_base'][m] = float(sd)
        out['per_set'][ds] = res
    # Holm over models x edits x sets for r != 0
    keys = [(ds, m, i) for ds in Rpt for m in allm for i in range(len(EDITS))]
    adj = holm([out['per_set'][ds]['p_signflip'][m][i] for ds, m, i in keys])
    for ds in Rpt:
        out['per_set'][ds]['p_holm'] = {m: [None] * len(EDITS) for m in allm}
    for (ds, m, i), pa in zip(keys, adj): out['per_set'][ds]['p_holm'][m][i] = float(pa)
    # concatenated (both jet sets) response vectors
    Rpt['both'] = {m: np.concatenate([Rpt['jc'][m], Rpt['top'][m]]) for m in allm}
    Rb['both'] = {m: np.concatenate([Rb['jc'][m], Rb['top'][m]], -1) for m in allm}
    # cosine matrices
    out['cosine'] = {}
    for ds in ('jc', 'top', 'both'):
        C = cosm(Rpt[ds], allm); Cb = cosm({m: Rb[ds][m] for m in allm}, allm)        # (M, M), (M, M, B)
        out['cosine'][ds] = dict(names=allm, C=C.tolist(), ci_lo=np.quantile(Cb, 0.025, -1).tolist(), ci_hi=np.quantile(Cb, 0.975, -1).tolist())
        if all(m in allm for m in ('JC_ParT_rel', 'JC_PN_rel', 'Top_ParT_s0')):
            i, j, k = allm.index('JC_ParT_rel'), allm.index('JC_PN_rel'), allm.index('Top_ParT_s0')
            dif = Cb[i, j] - Cb[i, k]
            out['cosine'][ds]['JCParT_JCPN_minus_JCParT_TopParT'] = dict(value=float(C[i, j] - C[i, k]), ci=ci(dif),
                                                                         cos_JCParT_JCPN=float(C[i, j]), cos_JCParT_TopParT=float(C[i, k]),
                                                                         p_boot_le0=float(np.mean(dif <= 0)))
    # permutation test on factor labels (main models only)
    labs = [FAC[m] for m in main]; out['perm_test'] = {}
    for ds in ('jc', 'top', 'both'):
        C = cosm(Rpt[ds], main); Tobs, wd, wa = perm_T(C, labs, None)
        Cb = cosm({m: Rb[ds][m] for m in main}, main); Tb, _, _ = perm_T(Cb, labs, None)
        cnt = 0
        for _ in range(P):
            pl = [labs[i] for i in rng.permutation(len(labs))]; Tp, _, _ = perm_T(C, pl, None)
            cnt += abs(Tp) >= abs(Tobs) - 1e-12
        out['perm_test'][ds] = dict(T=float(Tobs), T_ci=ci(Tb), within_dataset_cos=float(wd), within_arch_cos=float(wa), p_perm=float((cnt + 1) / (P + 1)),
                                    H0='mean cosine over same-dataset/different-architecture pairs = over same-architecture/different-dataset pairs',
                                    n_pairs=None)
        log(f'perm test {ds}: T {Tobs:+.3f} CI {ci(Tb)} within-dataset {wd:.3f} within-arch {wa:.3f} p {out["perm_test"][ds]["p_perm"]:.4f}')
    # variance decomposition
    out['decomp'] = {}
    for variant in ('primary', 'jc_part_retrained'):
        cells = cells_def(main, 'rel' if variant == 'primary' else 'seeds'); groups = seed_groups(main)
        if len(cells) < 4 or not groups:
            out['decomp'][variant] = dict(skipped=f'cells {list(cells)} groups {groups}'); continue
        out['decomp'][variant] = dict(cells={f'{a}-{b}': v for (a, b), v in cells.items()}, seed_groups=groups)
        Ftests = []
        for ds in ('jc', 'top'):
            dc = decomp(Rpt[ds], cells, groups); dcb = decomp({m: Rb[ds][m] for m in main}, cells, groups)
            sh, comp, raw = shares(dc); shb, compb, _ = shares(dcb)
            psh, pcomp, praw = shares(dc, -1); pshb, pcompb, _ = shares(dcb, -1)
            per_edit = {}
            for i, e in enumerate(EDITS):
                row = dict(dataset_effect_a=float(dc['a'][i]), arch_effect_b=float(dc['b'][i]), interaction_c=float(dc['c'][i]),
                           seed_var=float(dc['s2'][i]), a_ci=ci(dcb['a'][:, i]), b_ci=ci(dcb['b'][:, i]), c_ci=ci(dcb['c'][:, i]))
                for f in sh: row[f'share_{f}'] = float(sh[f][i]); row[f'share_{f}_ci'] = ci(shb[f][:, i])
                for f, x in (('dataset', 'a'), ('architecture', 'b'), ('interaction', 'c')):
                    F = dc[x][i] ** 2 / max(dc['k'] * dc['s2'][i], 1e-30); p = float(fdist.sf(F, 1, dc['df']))
                    row[f'F_{f}'] = float(F); row[f'p_F_{f}'] = p; Ftests.append((variant, ds, e, f, p))
                    # per-jet sign-flip test of the contrast for these fixed models
                    wts = {}
                    for (cd, ca), mem in cells.items():
                        sgn = dict(a=(1 if cd == 'JC' else -1), b=(1 if ca == 'ParT' else -1), c=(1 if (cd == 'JC') == (ca == 'ParT') else -1))[x]
                        for m in mem: wts[m] = wts.get(m, 0) + sgn / 4 / len(mem)
                    w = Dj[ds][main[0], e][1]; xj = sum(wt * Dj[ds][m, e][0] / Dj[ds][m, e][2] for m, wt in wts.items())
                    row[f'p_signflip_{f}'] = signflip(xj[w > 0], rng)
                per_edit[e] = row
            pooled = {f'share_{f}': float(psh[f]) for f in psh}
            pooled.update({f'share_{f}_ci': ci(pshb[f]) for f in pshb}); pooled.update({f'var_{f}': float(pcomp[f]) for f in pcomp})
            pooled.update({f'rawsq_{f}': float(praw[f]) for f in praw})
            pooled['dataset_gt_arch_boot_frac'] = float(np.mean(pcompb['dataset'] > pcompb['architecture']))
            pooled['seed_ge_both_boot_frac'] = float(np.mean((pcompb['seed'] >= pcompb['dataset']) & (pcompb['seed'] >= pcompb['architecture'])))
            out['decomp'][variant][ds] = dict(per_edit=per_edit, pooled=pooled, df_seed=dc['df'], k=dc['k'])
            log(f'decomp {variant} {ds}: pooled shares ' + ' '.join(f'{f} {psh[f]:.3f} {ci(pshb[f])}' for f in psh))
        adj = holm([t[-1] for t in Ftests])
        for (v, ds, e, f, _), pa in zip(Ftests, adj): out['decomp'][v][ds]['per_edit'][e][f'p_F_{f}_holm'] = float(pa)
        sf = [(ds, e, f) for ds in ('jc', 'top') for e in EDITS for f in ('dataset', 'architecture', 'interaction')]
        adj = holm([out['decomp'][variant][ds]['per_edit'][e][f'p_signflip_{f}'] for ds, e, f in sf])
        for (ds, e, f), pa in zip(sf, adj): out['decomp'][variant][ds]['per_edit'][e][f'p_signflip_{f}_holm'] = float(pa)
    # jet-set effect: same model, JetClass jets vs Top jets
    cells = cells_def(main, 'rel'); js = {}
    if len(cells) == 4:
        D = {m: (Rpt['jc'][m] - Rpt['top'][m]) / 2 for m in main}; Db = {m: (Rb['jc'][m] - Rb['top'][m]) / 2 for m in main}
        jsv = np.mean([np.mean([D[m] for m in v], 0) for v in cells.values()], 0)
        jsb = np.mean([np.mean([Db[m] for m in v], 0) for v in cells.values()], 0)
        aj = decomp(Rpt['jc'], cells, seed_groups(main))['a']; at = decomp(Rpt['top'], cells, seed_groups(main))['a']
        ajb = decomp({m: Rb['jc'][m] for m in main}, cells, seed_groups(main))['a']; atb = decomp({m: Rb['top'][m] for m in main}, cells, seed_groups(main))['a']
        VJ = (jsv ** 2).sum(); VD = ((aj ** 2 + at ** 2) / 2).sum(); VJb = (jsb ** 2).sum(-1); VDb = ((ajb ** 2 + atb ** 2) / 2).sum(-1)
        js = dict(jetset_effect_per_edit=jsv.tolist(), jetset_effect_ci=[ci(jsb[:, i]) for i in range(len(EDITS))],
                  V_jetset_pooled=float(VJ), V_jetset_ci=ci(VJb), V_dataset_pooled=float(VD), V_dataset_ci=ci(VDb),
                  ratio_jetset_over_dataset=float(VJ / max(VD, 1e-30)), ratio_ci=ci(VJb / np.maximum(VDb, 1e-30)),
                  frac_boot_jetset_gt_dataset=float(np.mean(VJb > VDb)),
                  per_model_cos_jc_vs_top={m: float(Rpt['jc'][m] @ Rpt['top'][m] / max(np.linalg.norm(Rpt['jc'][m]) * np.linalg.norm(Rpt['top'][m]), 1e-30)) for m in allm},
                  per_model_half_diff={m: ((Rpt['jc'][m] - Rpt['top'][m]) / 2).tolist() for m in allm})
        log(f'jet-set effect: V_jetset {VJ:.4f} {ci(VJb)} vs V_dataset {VD:.4f} {ci(VDb)}')
    out['jetset_effect'] = js
    out['edits'] = EDITS; out['edit_names'] = ENAME
    (OUT / f'h81_stats{s_}.json').write_text(json.dumps(out, indent=1, default=float)); log(f'wrote h81_stats{s_}.json')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--step', required=True, choices=['run', 'stats'])
    ap.add_argument('--models', default='all'); ap.add_argument('--smoke', type=int, default=0); a = ap.parse_args()
    log(f'H81 step {a.step} models {a.models} smoke {a.smoke}')
    if a.step == 'run': step_run(a.models, a.smoke)
    else: step_stats(a.smoke, B=200 if a.smoke else 1000, P=1000 if a.smoke else 10000)


if __name__ == '__main__':
    main()
