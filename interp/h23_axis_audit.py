"""H23 (loop D; claims D1, D6, D10, D11): axis-rebuild audit of the earlier W-pair (H3v, H4/H4v, H19) and soft-contamination
(H13) edits. Absorbs H20.

Question. H19v showed that H19 built every edited JetClass jet about the axis of the UNEDITED jet. Rigid subjet moves at fixed
jet mass shift the constituent-sum (E-scheme) axis, so the axis-relative inputs (deta, dphi, deltaR) of the JetClass networks were
translated as well as edited, and the JetClass networks respond to a pure translation. Here every earlier edit is built two ways:
  st  fixed axis, as in the original node (JetClass sample: the stored jet_eta / jet_phi; Top sample: the unedited jet's E-scheme
      axis; hls4ml: the stored (0, 0) axis);
  rc  recentred: each edited jet rebuilt about its own E-scheme axis (sum of the constituent four-momenta the model sees), with
      the JetClass convention eta = jet_eta + sign(jet_eta) * deta (part_core.build via h19_wpair_cross.jc_dict).
Translation-only control TN (TW): the unedited jet moved rigidly in (eta, phi) by the axis shift that edit N (W) induces, nothing
else; scored on the st path (and on the rc path for the dN decomposition).

Arms (--part jc, GPU):
  sample  JetClass TTBar_100 / ZJetsToNuNu_100 rows with 550 < jet_pt < 650 GeV, |jet_eta| < 2, >= 3 particles; 2,000 Tbqq and
          2,000 QCD drawn with default_rng(9523) (random rows, never first-k), excluding the H19 (rng 4191 / 4192) and H19v
          (rng 4391 / 4392) draws.
  H4      H4 / H19 W-pair edit (h4_verify.edit_set, three exclusive-kT subjets): T60, T100 (tops with 65 < mW < 95 -> 60 / 100 GeV),
          T80 (tops 6-15 GeV off 80.4 -> 80.4), Q80 (QCD in the top window -> 80.4); controls N (non-W pair, same relative mass
          change, W pair and m, tau32 fixed) and R (rotation); acceptance as H19.
  H3v     H3v W-pair edit (k-means subjets, h3_verify.wshift_all): W pair -> 60 / 100 GeV at fixed jet mass; control A (H3's, W pair
          free) and control B (W pair held at its mass, the non-W pair moved by the same dm).
  decomp  JetClass full models' residual recentred dN: dN on the rc path with the PID inputs (charge, 5 PID flags), the track
          inputs (tanh d0, d0err, tanh dz, dzerr) or both set to 0 at the input-feature level (both = kinematic inputs only),
          against the kin models and the rc translation-only control.
  H13jc   H13 soft contamination on 1,000 Wqq + 1,000 Zqq random rows of the first test files (default_rng(9523)): pu (rho 5 / 20 /
          40 GeV per unit area), c4 (same particles, displacement / 20), c5 (one wide particle, kept sum pT and mass matched),
          c2k (that particle on the axis), c3 (unedited jet about the pu axis: translation only); st and rc for each.
  models  ParT_full, ParT_kin, PN_full, PN_kin (JetClass); Top ParT, PN, LorentzNet, PFN, EFN, permuted PFN (h13_pileup_panel
          TopModels; LorentzNet takes four-vectors only, so st = rc for it).
  refs    MLPx_jc, MLPxW_jc, LR2_jc retrained as in H19 (TTBar_101 / ZJetsToNuNu_101, 25,000 per class).
(--part top, GPU): Top Landscape test, 2,000 t + 2,000 QCD (data_top.sample seed 9523); H4 arm (Top nets and JetClass kin nets,
  st / rc, TN) with MLPx_top (val split, 25,000 per class) and MLPx_jc; H13 arm with the Top nets.
(--part hls, CPU): hls4ml jets of jedi_core.sample(per_class=1000, seed=9523); H13 arm with JEDI-linear (official n64 3-feature)
  and a redrawn-weight JEDI-linear (null).
(--part summary): Holm across arms x models over the primary tests of the three JSONs; redraw of results/physics/h19_wpair.png.
Metrics: d_k = s_k - s_id (log-odds L_T - L_QCD; same path), W - N with 95 % bootstrap CI over jets (B = 2000), ratio to MLPx with
  CI, sign-flip permutation p (one-sided vs MLPx in the mW direction), McNemar W vs N calls at eps_S 0.3 / 0.5 (thresholds from the
  sample's unedited tops on that path), % calls changed; st - rc paired differences (two-sided sign-flip); (dN_st - dN_rc) - dTN.
Run: ./part_job.sh d-h23-audit 'python -m interp.h23_axis_audit --part jc'  (H23_SMOKE=<n>: n jets per class, small refs)
Outputs: $AI_RESULTS/physics/h23_{jc,top,hls,summary}.json, h23_*_perjet.npz, h19_wpair.png (summary)
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

T00 = time.time()
SMOKE = int(os.environ.get('H23_SMOKE', '0'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
TAG = 'h23_smoke' if SMOKE else 'h23'
SEED = 9523
PER = SMOKE or 2000
PER_WZ = (SMOKE // 2) if SMOKE else 1000
B = 400 if SMOKE else 2000
NPERM = 400 if SMOKE else 2000
NREF = 2000 if SMOKE else 25000
TBQQ, QCD = 8, 0
SETS = ('T60', 'T100', 'T80', 'Q80')
DIRECTION = {'T60': -1, 'T100': -1, 'T80': 1, 'Q80': 1}
RHOS = (5, 20, 40)
GROUPS = {'pid0': list(range(5, 11)), 'trk0': list(range(11, 15)), 'both0': list(range(5, 15))}   # ParT/PN 'full' feature rows


def log(*a):
    print(f'[{time.time() - T00:7.0f}s]', *a, flush=True)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return jsonable(o.tolist())
    return o


# ================================================================================================== statistics
def bci(x, bi):
    x = np.asarray(x, float); return np.percentile(x[bi].mean(1), [2.5, 97.5]).tolist()


def perm_p(x, sign, rng, n=NPERM):
    """sign-flip permutation p, H0 mean(x) = 0; one-sided toward sign * mean > 0 (sign = 0: two-sided)"""
    x = np.asarray(x, float)
    s = rng.choice([-1.0, 1.0], (n, len(x))); null = (s * x).mean(1)
    if sign == 0: return float((np.sum(np.abs(null) >= abs(x.mean())) + 1) / (n + 1))
    return float((np.sum(sign * null >= sign * x.mean()) + 1) / (n + 1))


def mcnemar(a, b):
    """a: calls of the reference condition, b: of the test condition; exact two-sided binomial p"""
    from interp.h1_irc_panel import binom_two_sided
    a, b = np.asarray(a, bool), np.asarray(b, bool); n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return dict(a=float(a.mean()), b=float(b.mean()), a_only=n10, b_only=n01, p=binom_two_sided(n10, n10 + n01))


def holm(ps):
    from interp.h1_irc_panel import holm as h
    return h(ps) if len(ps) else []


def bbin(x, rng):
    from interp.h1_irc_panel import boot_binary
    return boot_binary(x, rng, B)


# ================================================================================================== geometry
def axis(P4, nn):
    """E-scheme axis (pseudorapidity, azimuth of the summed constituent momentum) of the first nn particles"""
    P = P4.astype(np.float64); real = np.arange(P.shape[1])[None, :] < nn[:, None]
    S = (P * real[..., None]).sum(1); pt = np.hypot(S[:, 0], S[:, 1])
    return np.arcsinh(S[:, 2] / pt), np.arctan2(S[:, 1], S[:, 0])


def translate(P4, nn, de, dp):
    """every constituent moved by (de, dp) in (eta, phi); pT and mass kept (as interp.h19_verify.translate)"""
    P = P4.astype(np.float64); real = np.arange(P.shape[1])[None, :] < nn[:, None]
    pt = np.hypot(P[..., 0], P[..., 1]); ptc = np.where(real, pt, 1.0)
    eta = np.arcsinh(np.where(real, P[..., 2], 0) / ptc) + de[:, None]; phi = np.arctan2(P[..., 1], P[..., 0]) + dp[:, None]
    m = np.sqrt(np.maximum(P[..., 3] ** 2 - (P[..., :3] ** 2).sum(-1), 0))
    out = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)], -1)
    return (out * real[..., None]).astype(np.float32)


def massless(P4):
    P = P4.copy(); P[..., 3] = np.sqrt((P4[..., :3].astype(np.float64) ** 2).sum(-1)) * (P4[..., 3] != 0); return P


# ================================================================================================== models
def pn_logits_x(model, x, m, batch=256):
    import torch
    N = len(x); order = np.argsort(m[:, 0].sum(1), kind='stable'); L = np.zeros((N, 10)); dev = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, N, batch):
            idx = order[i:i + batch]; P = min(max(int(m[idx, 0].sum(1).max()), 17), x.shape[2])
            fx = torch.from_numpy(np.ascontiguousarray(x[idx, :, :P])).to(dev); mk = torch.from_numpy(np.ascontiguousarray(m[idx, :, :P])).to(dev)
            L[idx] = model(fx[:, -2:].clone(), fx.clone(), mk).double().cpu().numpy()
    return L


class Nets:
    def __init__(self, top=True):
        import torch
        from interp.part_core import ParTNetwork
        from interp.pn_net import load_particlenet
        from jetdistill.config import MODELS
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.jc = {f'ParT_{n}': ('part', n, ParTNetwork(n)) for n in ('full', 'kin')}
        for n, dims in (('full', 17), ('kin', 7)):
            self.jc[f'PN_{n}'] = ('pn', n, load_particlenet(Path(MODELS).parent / 'particlenet' / f'ParticleNet_{n}.pt', dims, self.dev))
        self.top = None
        if top:
            from interp.h13_pileup_panel import TopModels
            self.top = TopModels(self.dev)
        log(f'models loaded on {self.dev}: {list(self.jc)} + {self.top.names if self.top else []}')

    def jc_logits(self, name, Jd, zero=None):
        from interp.part_core import inputs
        from interp.p1v_verify_part import fwd
        kind, n, net = self.jc[name]; x, v, m = inputs(Jd, n)
        if zero is not None: x = x.copy(); x[:, zero] = 0.0
        return fwd(net, x, v, m)[1] if kind == 'part' else pn_logits_x(net, x, m)

    def score(self, ds, P4, nn, base, ax, order, jc_names=None, top=True, logits=None, zero=None):
        """scores (L_T - L_QCD for JetClass nets, top-vs-QCD log-odds for Top nets) with inputs built about axis ax = (eta, phi)"""
        from interp.h19_wpair_cross import jc_dict
        from interp.h13_pileup_panel import top_table, top_J
        r = {}; b = dict(base); b['jet_eta'] = ax[0].astype(np.float32); b['jet_phi'] = ax[1].astype(np.float32)
        Jd = jc_dict(b, P4, nn)
        for name in (jc_names if jc_names is not None else self.jc):
            if ds == 'top' and name.endswith('full'): continue
            L = self.jc_logits(name, Jd, zero); r[name] = L[:, TBQQ] - L[:, QCD]
            if logits is not None: logits[name] = L
        if top and self.top is not None:
            T = top_table(massless(P4) if ds == 'jc' else P4)
            r.update({f'{k}_top': v for k, v in self.top.score(top_J(T, ax[0], ax[1]), order).items()})
        return r


# ================================================================================================== samples
def jc_window(c, k, per, seed, excl):
    """per random rows of <class>_<k>.root in the H19 window, excluding earlier draws excl = [(seed, n), ...] (same rng calls)"""
    import uproot
    from interp.part_core import read_root, cat, sub
    from interp.h19_wpair_cross import jc_file
    f = jc_file(c, k); t = uproot.open(f)['tree']
    a = t.arrays(['jet_pt', 'jet_eta', 'jet_nparticles'], library='np')
    ok = (a['jet_pt'] > 550) & (a['jet_pt'] < 650) & (np.abs(a['jet_eta']) < 2) & (a['jet_nparticles'] >= 3)
    el = np.flatnonzero(ok); old = np.concatenate([np.random.default_rng(s).choice(el, min(n, len(el)), replace=False) for s, n in excl])
    el2 = np.setdiff1d(el, old); idx = np.sort(np.random.default_rng(seed).choice(el2, per, replace=False)); parts = []; N = t.num_entries
    for s in range(0, N, 20000):
        loc = idx[(idx >= s) & (idx < s + 20000)] - s
        if len(loc): parts.append(sub(read_root(f, start=s, stop=min(s + 20000, N)), loc))
    info = dict(file=f.name, eligible=int(len(el)), excluded=int(len(np.unique(old))), drawn=int(len(idx)),
                overlap_with_excluded=int(len(np.intersect1d(idx, old))), seed=seed)
    log(f'jc {c}: {info}')
    return cat(parts), info


def load_jc():
    from interp.part_core import cat, kin
    from interp import h4_top_wmass as H
    from interp.h19_wpair_cross import jc_P4
    Jt, it = jc_window('Tbqq', 100, PER, SEED, [(4191, 5000), (4391, 5000)])
    Jq, iq = jc_window('QCD', 100, PER, SEED + 1, [(4192, 5000), (4392, 3000)])
    J = cat([Jt, Jq]); P4raw, nn = jc_P4(J); P4 = H._identity(P4raw, nn).astype(np.float32); y = (J['label'] == TBQQ).astype(int)
    J['pmass'] = kin(J)[3].astype(np.float32)
    return dict(J=J, base=J, P4=P4, nn=nn, y=y, ax=(J['jet_eta'].astype(np.float64), J['jet_phi'].astype(np.float64)), info=dict(Tbqq=it, QCD=iq))


def load_top():
    from interp import data_top as dt
    from interp import h4_top_wmass as H
    from interp.h19_wpair_cross import top_base
    S = dt.sample(PER, SEED); full = S['full']; nn = np.minimum(S['jet'][:, 5].astype(np.int64), full.shape[1])
    P4 = H._identity(full[..., :4].astype(np.float32), nn).astype(np.float32); y = S['label'].astype(int)
    base = top_base(P4, nn); log(f'top sample: {len(y)} jets ({int(y.sum())} top), seed {SEED}')
    return dict(base=base, P4=P4, nn=nn, y=y, full=full, idx=S['idx'], ax=(base['jet_eta'].astype(np.float64), base['jet_phi'].astype(np.float64)),
                info=dict(n=int(len(y)), n_top=int(y.sum()), seed=SEED, note='data_top.sample(2000, 9523); earlier H4/H19 rows not excluded'))


def refs(which):
    from interp import h19_wpair_cross as X
    X.NREF = NREF; out = {}
    for w in which:
        Xr, Or, yr = X.ref_train_jc() if w == 'jc' else X.ref_train_top()
        out[w] = X.fit_refs(Xr, Or, yr, w)[0]
    return out


def ref_scores(REF, P4, nn):
    from interp import h4_top_wmass as H
    from interp.h19_wpair_cross import all_refs
    return all_refs(REF, *H._features(P4, nn))


# ================================================================================================== unedited performance
def perf_of(s, y, lg=None):
    from interp import h4_top_wmass as H
    out = {}
    for k, v in s.items():
        d = dict(AUC=float(H._auc(v, y)), acc_binary=float(((v > 0) == (y == 1)).mean()))
        if lg is not None and k in lg: d['acc_10class'] = float((lg[k].argmax(1) == np.where(y == 1, TBQQ, QCD)).mean())
        d['thr'] = {'eS30': float(np.quantile(v[y == 1], 0.7)), 'eS50': float(np.quantile(v[y == 1], 0.5))}
        out[k] = d
    return out


# ================================================================================================== H4 W-pair arm
def edits_h4(S, sets):
    from interp import h4_top_wmass as H
    from interp import h4_verify as HV
    nn, P4, y = S['nn'], S['P4'], S['y']; O = H._obs(P4, nn); S['O'] = O
    win = (O['mSD'] > 105) & (O['mSD'] < 210) & (nn >= 3)
    elig = {'T60': win & (y == 1) & (O['mW'] > 65) & (O['mW'] < 95)}; elig['T100'] = elig['T60']
    d80 = np.abs(O['mW'] - H.MWREF); elig['T80'] = win & (y == 1) & (d80 > 6) & (d80 < 15)
    elig['Q80'] = win & (y == 0) & (((O['mW'] > 50) & (O['mW'] < 70)) | ((O['mW'] > 90) & (O['mW'] < 115)))
    tgt = {'T60': (60.0, 5.0), 'T100': (100.0, 5.0), 'T80': (H.MWREF, 2.0), 'Q80': (H.MWREF, 5.0)}
    E = {}
    for key in sets:
        t, tol = tgt[key]; rows = np.flatnonzero(elig[key])
        if len(rows) == 0: continue
        ed = HV.edit_set(P4, nn, rows, t); acc = {}
        for k in 'WN':
            e = ed[k]; rr = e['rows']
            if len(rr) == 0: acc[k] = set(); continue
            Oe = H._obs(e['P'], nn[rr]); dm = np.abs(Oe['m'] - O['m'][rr]); dtau = np.abs(Oe['tau32'] - O['tau32'][rr])
            okW = np.abs(Oe['mW'] - t) < tol if k == 'W' else np.abs(Oe['mW'] - O['mW'][rr]) < 2
            acc[k] = set(rr[(dm < 0.5) & (dtau < 0.02) & okW].tolist())
        ar = np.array(sorted(acc['W'] & acc['N'] & set(ed['R']['rows'].tolist())), np.int64)
        E[key] = dict(rows=ar, P={k: ed[k]['P'][np.searchsorted(ed[k]['rows'], ar)] for k in 'WNR'},
                      counts=dict(eligible=int(len(rows)), W_acc=len(acc['W']), N_acc=len(acc['N']), analysis=int(len(ar))))
        log(f'H4 {key}: {E[key]["counts"]}')
    return E


def ratio_ci(x, r, bi):
    bx, br = x[bi].mean(1), r[bi].mean(1)
    return float(x.mean() / r.mean()), np.percentile(bx / br, [2.5, 97.5]).tolist()


def stats_variants(Ssc, nets, paths, ref_of, perf, key, sg, n, rng, contrasts, tests, decomp=None):
    """Ssc[f'{variant}|{path}'][net] per-jet scores; contrasts: list of (label, a, b): edit a vs control b"""
    bi = rng.integers(0, n, (B, n)); out = {}
    d = lambda m, v, p: Ssc[f'{v}|{p}'][m] - Ssc[f'id|{p}'][m]
    for lab, a, b in contrasts:
        R = {}
        refx = {}
        for rf in ('MLPx', 'MLPxW', 'LR2'):
            for tr in ('jc', 'top'):
                k = f'{rf}_{tr}'
                if k in Ssc['id|st']:
                    x = d(k, a, 'st') - d(k, b, 'st'); refx[k] = x; R[k] = dict(diff=float(x.mean()), ci95=bci(x, bi))
        for m in nets:
            for p in paths(m):
                if m not in Ssc[f'id|{p}']: continue
                da, db = d(m, a, p), d(m, b, p); x = da - db
                M = dict(d_edit=float(da.mean()), d_edit_ci95=bci(da, bi), d_ctrl=float(db.mean()), d_ctrl_ci95=bci(db, bi),
                         diff=float(x.mean()), ci95=bci(x, bi), p_diff_zero=perm_p(x, sg, rng))
                rk = ref_of(m)
                if rk in refx:
                    M['ratio_MLPx'], M['ratio_MLPx_ci95'] = ratio_ci(x, refx[rk], bi); M['ref'] = rk
                    mw = rk.replace('MLPx', 'MLPxW')
                    if mw in refx: M['ratio_MLPxW'], M['ratio_MLPxW_ci95'] = ratio_ci(x, refx[mw], bi)
                    M['p_vs_MLPx'] = perm_p(x - refx[rk], sg, rng); tests.append((key, lab, m, p, 'vs_MLPx', M['p_vs_MLPx']))
                for wp in ('eS30', 'eS50'):
                    thr = perf[f'{m}|{p}']['thr'][wp]; c0 = Ssc[f'id|{p}'][m] >= thr; ca = Ssc[f'{a}|{p}'][m] >= thr; cb = Ssc[f'{b}|{p}'][m] >= thr
                    M[f'calls_{wp}'] = dict(pass_id=float(c0.mean()), pass_edit=float(ca.mean()), pass_ctrl=float(cb.mean()),
                                            changed_edit=float((ca != c0).mean()), changed_ctrl=float((cb != c0).mean()), mcnemar_edit_vs_ctrl=mcnemar(cb, ca))
                    if wp == 'eS50': tests.append((key, lab, m, p, 'mcnemar_eS50', M[f'calls_{wp}']['mcnemar_edit_vs_ctrl']['p']))
                R[f'{m}|{p}'] = M
            if 'st' in paths(m) and 'rc' in paths(m) and f'{m}|rc' in R:
                xs = d(m, a, 'st') - d(m, b, 'st'); xr = d(m, a, 'rc') - d(m, b, 'rc'); dd = xs - xr
                R[f'{m}|st_minus_rc'] = dict(diff=float(dd.mean()), ci95=bci(dd, bi), p_two_sided=perm_p(dd, 0, rng))
                tests.append((key, lab, m, 'st-rc', 'st_eq_rc', R[f'{m}|st_minus_rc']['p_two_sided']))
                Tb = f'T{b}'
                if f'{Tb}|st' in Ssc:
                    dbs, dbr, dT = d(m, b, 'st'), d(m, b, 'rc'), d(m, Tb, 'st'); z = (dbs - dbr) - dT
                    R[f'{m}|axis_term'] = dict(ctrl_st_minus_rc=float((dbs - dbr).mean()), ctrl_st_minus_rc_ci95=bci(dbs - dbr, bi),
                                               dT_ctrl_st=float(dT.mean()), dT_ctrl_st_ci95=bci(dT, bi), residual=float(z.mean()), residual_ci95=bci(z, bi))
                    if f'{Tb}|rc' in Ssc and m in Ssc[f'{Tb}|rc']:
                        dTr = d(m, Tb, 'rc'); R[f'{m}|axis_term'].update(dT_ctrl_rc=float(dTr.mean()), dT_ctrl_rc_ci95=bci(dTr, bi))
                    Ta = f'T{a}'
                    if f'{Ta}|st' in Ssc:
                        dTa = d(m, Ta, 'st'); R[f'{m}|axis_term'].update(dT_edit_st=float(dTa.mean()), dT_edit_st_ci95=bci(dTa, bi))
        out[lab] = R
    return out


def run_h4(ds, S, N, REF, perf, PJ, tests, rng):
    from interp.h19_wpair_cross import subd
    E = edits_h4(S, SETS); res = {}
    jcn = [m for m in N.jc if not (ds == 'top' and m.endswith('full'))]
    topn = [f'{k}_top' for k in N.top.names] if N.top else []
    paths = lambda m: ('st', 'rc') if m != 'LNet_top' else ('rc',)
    for key, e in E.items():
        ar = e['rows']; n = len(ar)
        if n < 5: res[key] = dict(counts=e['counts']); continue
        P0, n0 = S['P4'][ar], S['nn'][ar]; base = subd(S['base'], ar); order = np.random.default_rng(606).permutation(n)
        st = (S['ax'][0][ar], S['ax'][1][ar]); a0 = axis(P0, n0)
        V = {'id': P0, **e['P']}; shift = {}
        for k in 'WNR':
            ak = axis(V[k], n0); shift[k] = (ak[0] - a0[0], wrap(ak[1] - a0[1]))
        for k in 'WN': V[f'T{k}'] = translate(P0, n0, *shift[k])
        Ssc = {}
        for v, P in V.items():
            Ssc[f'{v}|st'] = N.score(ds, P, n0, base, st, order); Ssc[f'{v}|rc'] = N.score(ds, P, n0, base, axis(P, n0), order)
            Ssc[f'{v}|st'].update(ref_scores(REF, P, n0))
            # LorentzNet has no axis: its st and rc scores are the same call; keep it under rc only
            Ssc[f'{v}|st'].pop('LNet_top', None)
        r = dict(counts=e['counts'], axis_shift={k: dict(median=float(np.median(np.hypot(*shift[k]))), p90=float(np.quantile(np.hypot(*shift[k]), 0.9))) for k in 'WNR'},
                 stored_vs_sum_axis_median=float(np.median(np.hypot(st[0] - a0[0], wrap(st[1] - a0[1])))))
        r['contrasts'] = stats_variants(Ssc, jcn + topn, paths, lambda m: ('MLPx_jc' if m in N.jc else ('MLPx_top' if ds == 'top' else 'MLPx_jc')),
                                        perf, key, DIRECTION[key], n, rng, [('W_vs_N', 'W', 'N'), ('W_vs_R', 'W', 'R')], tests)
        # decomposition of the full models' recentred dN (JetClass sample, T60 / T100)
        if ds == 'jc' and key in ('T60', 'T100'):
            bi = rng.integers(0, n, (B, n)); D = {}
            for m in ('ParT_full', 'PN_full'):
                D[m] = {}
                for g, rows_ in [('none', None)] + list(GROUPS.items()):
                    sc = {v: N.score(ds, V[v], n0, base, axis(V[v], n0), order, jc_names=[m], top=False, zero=rows_)[m] for v in ('id', 'W', 'N', 'TN')}
                    dN, dW, dT = sc['N'] - sc['id'], sc['W'] - sc['id'], sc['TN'] - sc['id']
                    D[m][g] = dict(dN_rc=float(dN.mean()), dN_rc_ci95=bci(dN, bi), dW_rc=float(dW.mean()), W_minus_N_rc=float((dW - dN).mean()),
                                   W_minus_N_rc_ci95=bci(dW - dN, bi), dTN_rc=float(dT.mean()), dTN_rc_ci95=bci(dT, bi))
                    PJ[f'{ds}|{key}|decomp|{m}|{g}|dN'] = dN.astype(np.float32)
                    if g != 'none':
                        x = PJ[f'{ds}|{key}|decomp|{m}|none|dN'] - dN
                        D[m][g]['dN_change_vs_none'] = float(x.mean()); D[m][g]['dN_change_ci95'] = bci(x, bi)
                        D[m][g]['p_two_sided'] = perm_p(x, 0, rng); tests.append((key, 'decomp', m, 'rc', g, D[m][g]['p_two_sided']))
                log(f'[{ds}] {key} decomp {m}: ' + ' | '.join(f"{g} dN {v['dN_rc']:+.2f} dTN {v['dTN_rc']:+.2f} W-N {v['W_minus_N_rc']:+.2f}" for g, v in D[m].items()))
            for m in ('ParT_kin', 'PN_kin'):
                dN = Ssc['N|rc'][m] - Ssc['id|rc'][m]; D[m] = dict(none=dict(dN_rc=float(dN.mean()), dN_rc_ci95=bci(dN, bi)))
            r['decomp'] = D
        for v, s in Ssc.items():
            for m, a in s.items(): PJ[f'{ds}|H4|{key}|{v}|{m}'] = np.asarray(a, np.float32)
        res[key] = r
        c = r['contrasts']['W_vs_N']
        log(f'[{ds}] {key} n {n} axis shift W {r["axis_shift"]["W"]["median"]:.3f} N {r["axis_shift"]["N"]["median"]:.3f}; MLPx W-N '
            + ' '.join(f'{k} {c[k]["diff"]:+.2f}' for k in ('MLPx_jc', 'MLPx_top', 'MLPxW_jc', 'MLPxW_top') if k in c))
        for m in jcn + topn:
            s_ = []
            for p in ('st', 'rc'):
                if f'{m}|{p}' in c: M = c[f'{m}|{p}']; s_.append(f"{p}: dN {M['d_ctrl']:+.2f} W-N {M['diff']:+.2f} {np.round(M['ci95'], 2)} x{M.get('ratio_MLPx', np.nan):.2f}")
            if f'{m}|axis_term' in c: s_.append(f"dTN {c[f'{m}|axis_term']['dT_ctrl_st']:+.2f} resid {c[f'{m}|axis_term']['residual']:+.2f}")
            log(f'   {m:13s} ' + ' | '.join(s_))
    return res


# ================================================================================================== H3v W-pair arm (JetClass)
def run_h3v(S, N, REF, perf, PJ, tests, rng):
    from interp import h3_verify as H3V
    from interp.h3_d3_fullstats import top_split
    from interp.part_core import kin, build, sub
    from interp.h19_wpair_cross import jc_P4, subd
    tops = np.flatnonzero(S['y'] == 1); Jt = sub(S['J'], tops)
    pt, de, dp, ms = (np.asarray(a, np.float64) for a in kin(Jt)); TS = top_split(pt, de, dp)
    log(f'H3v: tops with a W pair {int(TS["ok"].sum())}/{len(tops)}'); res = {}
    jcn = list(N.jc); topn = [f'{k}_top' for k in N.top.names] if N.top else []
    paths = lambda m: ('st', 'rc') if m != 'LNet_top' else ('rc',)
    for t in (60.0, 100.0):
        R = H3V.wshift_all(pt, de, dp, t, TS); m = R['ok'] & R['okE'] & R['okA'] & R['okB']; rows = np.flatnonzero(m); n = len(rows)
        key = f'H3v_{t:g}'; log(f'{key}: n {n}; pair masses unedited/edit/ctrlA/ctrlB ' + json.dumps({k: R['med'](R[k], m) for k in ('P0', 'PE', 'PA', 'PB')}))
        if n < 5: res[key] = dict(n=n); continue
        Jm = sub(Jt, rows); base = subd(S['base'], tops[rows]); order = np.random.default_rng(607).permutation(n)
        V = {}
        for v, (de_, dp_) in (('id', (de, dp)), ('E', R['edit']), ('A', R['ctrlA']), ('B', R['ctrlB'])):
            Jb = build(Jm, pt[rows], de_[rows], dp_[rows], ms[rows]); P4, nn_ = jc_P4(Jb); V[v] = P4
        n0 = S['nn'][tops[rows]]
        assert np.array_equal(n0, nn_), 'particle count changed in the H3v build'
        a0 = axis(V['id'], n0); shift = {}
        for k in ('E', 'B'):
            ak = axis(V[k], n0); shift[k] = (ak[0] - a0[0], wrap(ak[1] - a0[1])); V[f'T{k}'] = translate(V['id'], n0, *shift[k])
        st = (S['ax'][0][tops[rows]], S['ax'][1][tops[rows]]); Ssc = {}
        for v, P in V.items():
            Ssc[f'{v}|st'] = N.score('jc', P, n0, base, st, order); Ssc[f'{v}|rc'] = N.score('jc', P, n0, base, axis(P, n0), order)
            Ssc[f'{v}|st'].update(ref_scores(REF, P, n0)); Ssc[f'{v}|st'].pop('LNet_top', None)
        r = dict(n=n, masses={k: R['med'](R[k], m) for k in ('P0', 'PE', 'PA', 'PB')},
                 axis_shift={k: dict(median=float(np.median(np.hypot(*shift[k]))), p90=float(np.quantile(np.hypot(*shift[k]), 0.9))) for k in shift})
        sg = -1   # moving the W pair off 80 GeV lowers a W-mass-constrained top score more than the W-fixed control
        r['contrasts'] = stats_variants(Ssc, jcn + topn, paths, lambda mm: 'MLPx_jc', perf, key, sg, n, rng,
                                        [('E_vs_B', 'E', 'B'), ('E_vs_A', 'E', 'A')], tests)
        # 10-class top call (argmax == Tbqq), as H3v reported
        calls = {}
        for mm in ('ParT_full', 'ParT_kin', 'PN_full', 'PN_kin'):
            for p in ('st', 'rc'):
                C = {}
                for v in ('id', 'E', 'B', 'TB'):
                    from interp.h19_wpair_cross import jc_dict
                    P = V[v]; ax = st if p == 'st' else axis(P, n0); b = dict(base); b['jet_eta'] = ax[0].astype(np.float32); b['jet_phi'] = ax[1].astype(np.float32)
                    C[v] = N.jc_logits(mm, jc_dict(b, P, n0)).argmax(1) == TBQQ
                bi = rng.integers(0, n, (B, n)); x = C['E'].astype(float) - C['B']
                calls[f'{mm}|{p}'] = dict(top_call_id=float(C['id'].mean()), top_call_E=float(C['E'].mean()), top_call_B=float(C['B'].mean()),
                                          top_call_TB=float(C['TB'].mean()), E_minus_B_pp=float(100 * x.mean()), E_minus_B_ci95=[100 * q for q in bci(x, bi)],
                                          mcnemar_E_vs_B=mcnemar(C['B'], C['E']))
                tests.append((key, 'topcall_E_vs_B', mm, p, 'mcnemar', calls[f'{mm}|{p}']['mcnemar_E_vs_B']['p']))
        r['top_call_10class'] = calls
        for v, s in Ssc.items():
            for mm, a in s.items(): PJ[f'jc|{key}|{v}|{mm}'] = np.asarray(a, np.float32)
        res[key] = r
        c = r['contrasts']['E_vs_B']
        log(f'{key}: MLPx_jc E-B {c["MLPx_jc"]["diff"]:+.3f} {np.round(c["MLPx_jc"]["ci95"], 3)}; axis shift E {r["axis_shift"]["E"]["median"]:.3f} B {r["axis_shift"]["B"]["median"]:.3f}')
        for mm in jcn + topn:
            s_ = [f"{p}: E-B {c[f'{mm}|{p}']['diff']:+.2f} {np.round(c[f'{mm}|{p}']['ci95'], 2)} x{c[f'{mm}|{p}'].get('ratio_MLPx', np.nan):.2f}" for p in ('st', 'rc') if f'{mm}|{p}' in c]
            if f'{mm}|{"st"}' in calls or f'{mm}|st' in calls:
                s_.append('call E-B st/rc ' + '/'.join(f"{calls[f'{mm}|{p}']['E_minus_B_pp']:+.1f}" for p in ('st', 'rc')))
            log(f'   {mm:13s} ' + ' | '.join(s_))
    return res


# ================================================================================================== H13 soft-contamination arm
def h13_conditions(T0, cap, seed):
    """(name, finalized table) for id and, per rho, pu / c4 / c5 / c2k (h13_verify definitions about the stored axis)"""
    from interp.h1_irc_panel import finalize
    from interp.h13_pileup_panel import inject, spectrum, mass_all
    from interp.h13_verify import core, match_radius, add_one
    spec = spectrum(T0); W = T0['pt'].shape[1]
    Uf0, _ = finalize(T0, cap); yield 'id', Uf0
    for rho in RHOS:
        U, inf = inject(T0, rho, spec, np.random.default_rng([seed, rho, 0])); Uf, _ = finalize(U, cap); yield f'pu_{rho}', Uf
        C = core(U, T0, W); Cf, _ = finalize(C, cap); yield f'c4_{rho}', Cf
        Q = np.maximum(Uf['pt'].sum(1) - Uf0['pt'].sum(1), 0.0); th = 2 * np.pi * np.random.default_rng([seed, rho, 7]).random(len(Q))
        r, info = match_radius(T0, Q, th, mass_all(Uf), cap); log(f'  c5_{rho}: added pT median {np.median(Q):.2f} GeV; radius {info}')
        F5, _ = add_one(T0, Q, r, th, cap); yield f'c5_{rho}', F5
        F2, _ = add_one(T0, Q, np.zeros(len(Q)), th, cap); yield f'c2k_{rho}', F2


def run_h13(tag, T0, y, score_fn, cap, call_fn, seed, rng, tests, PJ, effs):
    """score_fn(Uf, ae, ap) -> {model: per-jet score array or logits}; call_fn(model, out, path) -> per-jet call (int/bool);
    effs: {name: (class mask, call value)} efficiencies to report"""
    from interp.h13_pileup_panel import eaxis
    st = (T0['ax_eta'], T0['ax_phi']); OUT = {}; AX = {}; base_rc = None; Uf0 = None
    for name, Uf in h13_conditions(T0, cap, seed):
        t = time.time(); rc = eaxis(Uf); AX[name] = rc
        OUT[f'{name}|st'] = score_fn(Uf, *st); OUT[f'{name}|rc'] = score_fn(Uf, *rc)
        if name == 'id': Uf0 = Uf
        if name.startswith('pu_'): OUT[f'c3_{name[3:]}|st'] = score_fn(Uf0, *rc)   # unedited jet about the pu axis: translation only
        if name.startswith('c5_'): OUT[f'c3c5_{name[3:]}|st'] = score_fn(Uf0, *rc)  # unedited jet about the c5 axis
        log(f'{tag} {name} scored ({time.time() - t:.0f} s); axis offset median {np.median(np.hypot(rc[0] - st[0], wrap(rc[1] - st[1]))):.4f}')
    models = list(OUT['id|st']); res = dict(N=int(len(y)), axis_offset={}, models={})
    for name, rc in AX.items():
        off = np.hypot(rc[0] - st[0], wrap(rc[1] - st[1])); res['axis_offset'][name] = dict(median=float(np.median(off)), p90=float(np.quantile(off, 0.9)))
    for mn in models:
        R = {}; CH = {}
        for path in ('st', 'rc'):
            c0 = call_fn(mn, OUT[f'id|{path}'][mn], path)
            for k in OUT:
                nm, p = k.split('|')
                if p != path: continue
                c1 = call_fn(mn, OUT[k][mn], path); ch = c1 != c0
                CH[k] = ch; d = dict(changed=float(ch.mean()), ci95=bbin(ch, rng))
                for en, (msk, val) in effs.items(): d[f'eff_{en}'] = float((c1[msk] == val).mean())
                R[k] = d
        T = {}
        for rho in RHOS:
            for path in ('st', 'rc'):
                for c in ('c4', 'c5', 'c2k'):
                    tt = mcnemar(CH[f'pu_{rho}|{path}'], CH[f'{c}_{rho}|{path}']); T[f'pu_vs_{c}_{rho}|{path}'] = tt
                    tests.append((tag, mn, rho, f'pu_vs_{c}|{path}', tt['p']))
                n = len(y); bi = rng.integers(0, n, (B, n)); a, b = CH[f'c4_{rho}|{path}'].astype(float), CH[f'pu_{rho}|{path}'].astype(float)
                T[f'ratio_c4_pu_{rho}|{path}'] = dict(value=float(a.mean() / max(b.mean(), 1e-12)), ci95=np.percentile(a[bi].mean(1) / np.maximum(b[bi].mean(1), 1e-12), [2.5, 97.5]).tolist())
                a = CH[f'c5_{rho}|{path}'].astype(float)
                T[f'ratio_c5_pu_{rho}|{path}'] = dict(value=float(a.mean() / max(b.mean(), 1e-12)), ci95=np.percentile(a[bi].mean(1) / np.maximum(b[bi].mean(1), 1e-12), [2.5, 97.5]).tolist())
            for c in ('pu', 'c4', 'c5', 'c2k'):
                # st vs rc: are the calls that change the same? (changed-call indicator, paired)
                tt = mcnemar(CH[f'{c}_{rho}|rc'], CH[f'{c}_{rho}|st']); T[f'{c}_{rho}|st_vs_rc'] = tt; tests.append((tag, mn, rho, f'{c}|st_vs_rc', tt['p']))
        R['tests'] = T; res['models'][mn] = R
        log(f'{tag} {mn}: changed st/rc ' + ' '.join(f"{c}_{rho} {R[f'{c}_{rho}|st']['changed']:.3f}/{R[f'{c}_{rho}|rc']['changed']:.3f}"
                                                  for rho in (5, 40) for c in ('pu', 'c4', 'c5', 'c2k')) +
            ' | c3 ' + ' '.join(f"{R[f'c3_{rho}|st']['changed']:.3f}" for rho in RHOS) + ' c3c5 ' + ' '.join(f"{R[f'c3c5_{rho}|st']['changed']:.3f}" for rho in RHOS))
    return res


def h13_jc(N, rng, tests, PJ):
    import uproot
    from interp.part_core import read_root, fof, cat, sub, CL
    from interp.h1_irc_panel import jc_table
    from interp.h13_pileup_panel import jc_maker
    Js = []; info = {}
    for i, c in enumerate(('Wqq', 'Zqq')):
        f = fof(c); n = uproot.open(f)['tree'].num_entries; idx = np.sort(np.random.default_rng([SEED, i]).choice(n, PER_WZ, replace=False)); parts = []
        for a in range(0, n, 20000):
            loc = idx[(idx >= a) & (idx < a + 20000)] - a
            if len(loc): parts.append(sub(read_root(f, start=a, stop=min(a + 20000, n)), loc))
        Js.append(cat(parts)); info[c] = dict(file=Path(f).name, rows=int(n), drawn=int(len(idx)), seed=[SEED, i])
    J0 = cat(Js); y = J0['label'].astype(int); T = jc_table(J0); make = jc_maker(J0); iW, iZ = CL.index('Wqq'), CL.index('Zqq')
    log(f'H13 jc sample {info}')
    score = lambda Uf, ae, ap: {m: N.jc_logits(m, make(Uf, ae, ap)) for m in N.jc}
    call = lambda m, L, p: L.argmax(1)
    res = run_h13('h13_jc', T, y, score, 128, call, SEED, rng, tests, PJ, {'W_on_W': (y == iW, iW), 'Z_on_Z': (y == iZ, iZ)})
    # unedited accuracy (10-class argmax) on these W / Z jets
    L0 = score(*h13_first(T, 128))
    res['unedited'] = {m: dict(acc_10class=float((L.argmax(1) == y).mean()), W_eff=float((L.argmax(1)[y == iW] == iW).mean()), Z_eff=float((L.argmax(1)[y == iZ] == iZ).mean())) for m, L in L0.items()}
    res['sample'] = info; log(f'H13 jc unedited {res["unedited"]}')
    return res


def h13_first(T, cap):
    from interp.h1_irc_panel import finalize
    Uf, _ = finalize(T, cap); return Uf, T['ax_eta'], T['ax_phi']


# ================================================================================================== parts
def part_jc():
    rng = np.random.default_rng(23); tests = []; PJ = {}
    S = load_jc(); N = Nets(); REF = refs(['jc'])
    ref_ = lambda P, n: ref_scores(REF, P, n)
    order = np.random.default_rng(605).permutation(len(S['nn']))
    lg_st, lg_rc = {}, {}
    s_st = N.score('jc', S['P4'], S['nn'], S['base'], S['ax'], order, logits=lg_st); s_st.update(ref_(S['P4'], S['nn']))
    s_rc = N.score('jc', S['P4'], S['nn'], S['base'], axis(S['P4'], S['nn']), order, logits=lg_rc)
    perf = {f'{k}|st': v for k, v in perf_of(s_st, S['y'], lg_st).items()}; perf.update({f'{k}|rc': v for k, v in perf_of(s_rc, S['y'], lg_rc).items()})
    for k in list(perf):
        if k.startswith('LNet_top|'): perf['LNet_top|st'] = perf['LNet_top|rc'] = perf[k]
    a0 = axis(S['P4'], S['nn']); dax = np.hypot(a0[0] - S['ax'][0], wrap(a0[1] - S['ax'][1]))
    log('unedited (jc sample) ' + ', '.join(f"{k} AUC {v['AUC']:.4f} acc {v['acc_binary']:.4f}" + (f" 10c {v['acc_10class']:.4f}" if 'acc_10class' in v else '') for k, v in perf.items()))
    out = dict(node='H23', part='jc', smoke=SMOKE, sample=S['info'], unedited=perf,
               stored_vs_sum_axis=dict(median=float(np.median(dax)), p99=float(np.quantile(dax, 0.99))))
    out['H4'] = run_h4('jc', S, N, REF, perf, PJ, tests, rng)
    out['H3v'] = run_h3v(S, N, REF, perf, PJ, tests, rng)
    out['H13'] = h13_jc(N, rng, tests, PJ)
    finish(out, tests, PJ, 'jc')


def part_top():
    from interp.h13_pileup_panel import top_table, top_J
    rng = np.random.default_rng(24); tests = []; PJ = {}
    S = load_top(); N = Nets(); REF = refs(['jc', 'top'])
    order = np.random.default_rng(605).permutation(len(S['nn']))
    s = N.score('top', S['P4'], S['nn'], S['base'], S['ax'], order); s.update(ref_scores(REF, S['P4'], S['nn']))
    perf = {f'{k}|st': v for k, v in perf_of(s, S['y']).items()}; perf.update({f'{k}|rc': v for k, v in perf_of(s, S['y']).items()})
    log('unedited (top sample; st = rc on unedited jets) ' + ', '.join(f"{k} AUC {v['AUC']:.4f} acc {v['acc_binary']:.4f}" for k, v in perf.items() if k.endswith('|st')))
    out = dict(node='H23', part='top', smoke=SMOKE, sample=S['info'], unedited=perf)
    out['H4'] = run_h4('top', S, N, REF, perf, PJ, tests, rng)
    # H13 on the Top Landscape sample, Top nets (cap 200 as H13)
    T0 = top_table(S['full']); y = S['y']; ordr = np.random.default_rng(305).permutation(len(y))
    score = lambda Uf, ae, ap: N.top.score(top_J(Uf, ae, ap), ordr)
    s0 = {p: score(*h13_first(T0, 200)) for p in ('st',)}
    from interp.h13_pileup_panel import eaxis
    Uf0, _, _ = h13_first(T0, 200); s0['rc'] = score(Uf0, *eaxis(Uf0))
    thr = {p: {k: float(np.quantile(v[y == 1], 0.5)) for k, v in s0[p].items()} for p in s0}
    call = lambda m, sc, p: sc > thr[p][m]
    res = run_h13('h13_top', T0, y, score, 200, call, SEED, rng, tests, PJ, {'tag_on_top': (y == 1, True), 'tag_on_qcd': (y == 0, True)})
    from interp import h4_top_wmass as H
    res['unedited'] = {k: dict(AUC=float(H._auc(v, y)), acc_binary=float(((v > 0) == (y == 1)).mean())) for k, v in s0['st'].items()}
    out['H13'] = res
    finish(out, tests, PJ, 'top')


def part_decomp():
    """JetClass full models' recentred non-W response, with the unedited accuracy of each input-zeroed model (same sample and
    edits as --part jc: load_jc and edits_h4 are deterministic)"""
    from interp.part_core import inputs
    rng = np.random.default_rng(26); tests = []
    S = load_jc(); N = Nets(top=False); y = S['y']; order = np.random.default_rng(605).permutation(len(y))
    rcax = axis(S['P4'], S['nn']); acc = {}
    for m in ('ParT_full', 'PN_full', 'ParT_kin', 'PN_kin'):
        for g, rows_ in [('none', None)] + (list(GROUPS.items()) if m.endswith('full') else []):
            lg = {}; s = N.score('jc', S['P4'], S['nn'], S['base'], rcax, order, jc_names=[m], top=False, logits=lg, zero=rows_)[m]
            acc[f'{m}|{g}'] = dict(**{k: v for k, v in perf_of({m: s}, y, lg)[m].items() if k != 'thr'}, mean_score_top=float(s[y == 1].mean()),
                                   mean_score_qcd=float(s[y == 0].mean()))
            log(f'decomp unedited {m} {g}: {acc[f"{m}|{g}"]}')
    E = edits_h4(S, ('T60', 'T100')); res = dict(node='H23', part='decomp', smoke=SMOKE, sample=S['info'], unedited_rc=acc, sets={})
    from interp.h19_wpair_cross import subd
    for key, e in E.items():
        ar = e['rows']; n = len(ar); P0, n0 = S['P4'][ar], S['nn'][ar]; base = subd(S['base'], ar); ordr = np.random.default_rng(606).permutation(n)
        a0 = axis(P0, n0); aN = axis(e['P']['N'], n0)
        V = {'id': P0, 'W': e['P']['W'], 'N': e['P']['N'], 'TN': translate(P0, n0, aN[0] - a0[0], wrap(aN[1] - a0[1]))}
        bi = rng.integers(0, n, (B, n)); D = {}
        for m in ('ParT_full', 'PN_full', 'ParT_kin', 'PN_kin'):
            D[m] = {}; base_dN = None
            for g, rows_ in [('none', None)] + (list(GROUPS.items()) if m.endswith('full') else []):
                sc = {v: N.score('jc', P, n0, base, axis(P, n0), ordr, jc_names=[m], top=False, zero=rows_)[m] for v, P in V.items()}
                dN, dW, dT = sc['N'] - sc['id'], sc['W'] - sc['id'], sc['TN'] - sc['id']
                r = dict(dW_rc=float(dW.mean()), dW_rc_ci95=bci(dW, bi), dN_rc=float(dN.mean()), dN_rc_ci95=bci(dN, bi),
                         W_minus_N_rc=float((dW - dN).mean()), W_minus_N_rc_ci95=bci(dW - dN, bi), dTN_rc=float(dT.mean()), dTN_rc_ci95=bci(dT, bi),
                         dN_over_dW=float(dN.mean() / dW.mean()), dN_over_dW_ci95=np.percentile(dN[bi].mean(1) / dW[bi].mean(1), [2.5, 97.5]).tolist())
                if g == 'none': base_dN = dN
                else:
                    x = base_dN - dN; r.update(dN_change_vs_none=float(x.mean()), dN_change_ci95=bci(x, bi), p_two_sided=perm_p(x, 0, rng))
                    tests.append((key, m, g, r['p_two_sided']))
                D[m][g] = r
            log(f'decomp {key} {m}: ' + ' | '.join(f"{g} dW {v['dW_rc']:+.2f} dN {v['dN_rc']:+.2f} {np.round(v['dN_rc_ci95'], 2)} dN/dW {v['dN_over_dW']:.2f} "
                                                   f"W-N {v['W_minus_N_rc']:+.2f} dTN {v['dTN_rc']:+.3f}" for g, v in D[m].items()))
        res['sets'][key] = dict(n=n, models=D)
    finish(res, tests, {}, 'decomp')


def part_hls():
    from interp.jedi_core import Jedi, sample
    from interp.h1_irc_panel import hls_table
    from interp.h13_pileup_panel import hls_in
    rng = np.random.default_rng(25); tests = []; PJ = {}
    x, y, _ = sample(per_class=PER_WZ, seed=SEED); y = np.asarray(y).astype(int); T = hls_table(x)
    nets = dict(jedi=Jedi(64, '3-feature'), jedi_redrawn=Jedi(64, '3-feature', untrained=True, seed=0))
    score = lambda Uf, ae, ap: {k: np.asarray(n.run(hls_in(Uf, ae, ap))['logits'], np.float64) for k, n in nets.items()}
    call = lambda m, L, p: L.argmax(1)
    L0 = score(*h13_first(T, 64))
    unedited = {k: dict(acc=float((L.argmax(1) == y).mean()), W_eff=float((L.argmax(1)[y == 2] == 2).mean())) for k, L in L0.items()}
    log(f'hls unedited {unedited} (known JEDI-linear 0.809 on the 260k test set)')
    res = run_h13('h13_hls', T, y, score, 64, call, SEED, rng, tests, PJ, {'W_on_W': (y == 2, 2), 'Z_on_Z': (y == 3, 3), 't_on_t': (y == 4, 4)})
    res['unedited'] = unedited
    finish(dict(node='H23', part='hls', smoke=SMOKE, sample=dict(per_class=PER_WZ, seed=SEED), H13=res), tests, PJ, 'hls')


def finish(out, tests, PJ, part):
    adj = holm([t[-1] for t in tests])
    out['tests'] = [list(t) + [float(a)] for t, a in zip(tests, adj)]
    out['holm_family'] = f'{len(tests)} tests in part {part} (Holm within part; the summary part applies Holm over all parts)'
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f'{TAG}_{part}.json').write_text(json.dumps(jsonable(out), indent=1))
    if PJ: np.savez_compressed(RES / f'{TAG}_{part}_perjet.npz', **{k.replace('|', '__'): v for k, v in PJ.items()})
    log(f'wrote {TAG}_{part}.json ({len(tests)} tests)')


# ================================================================================================== summary + figure
def part_summary():
    P = {p: json.loads((RES / f'{TAG}_{p}.json').read_text()) for p in ('jc', 'top', 'hls', 'decomp') if (RES / f'{TAG}_{p}.json').exists()}
    # Primary family, Holm across arms x models x paths. The sign-flip permutation p of the parts has a floor of 1/(NPERM + 1)
    # = 5e-4, which Holm over several hundred tests cannot resolve, so the per-jet arrays are re-tested here with a one-sample
    # t-test (n >= 150 jets per arm except Q80); McNemar p (exact) are taken from the part JSONs.
    from scipy import stats
    def tp(x, sign):
        x = np.asarray(x, float); t, p2 = stats.ttest_1samp(x, 0.0)
        if not np.isfinite(t): return 1.0
        return float(p2) if sign == 0 else float(p2 / 2 if sign * t > 0 else 1 - p2 / 2)
    fam = []
    for part in ('jc', 'top'):
        f = RES / f'{TAG}_{part}_perjet.npz'
        if not f.exists(): continue
        Z = np.load(f); g = lambda *k: Z['__'.join(k)]
        for key in SETS:
            r = P[part]['H4'].get(key, {})
            if 'contrasts' not in r: continue
            c = r['contrasts']['W_vs_N']
            for mp, M in c.items():
                if '|' not in mp or mp.split('|')[1] not in ('st', 'rc') or 'ref' not in M: continue
                m, p = mp.split('|'); d = lambda v, pp: g(part, 'H4', key, v, pp, m) - g(part, 'H4', key, 'id', pp, m)
                rf = lambda v: g(part, 'H4', key, v, 'st', M['ref']) - g(part, 'H4', key, 'id', 'st', M['ref'])
                x = d('W', p) - d('N', p); rx = rf('W') - rf('N')
                fam.append(dict(part=part, arm='H4', set=key, model=m, path=p, test='W-N vs MLPx (one-sided, mW direction)', effect=M['diff'],
                                ci95=M['ci95'], ratio=M['ratio_MLPx'], ratio_ci95=M['ratio_MLPx_ci95'], p=tp(x - rx, DIRECTION[key])))
                if p == 'rc' and f'{m}|st_minus_rc' in c:
                    q = c[f'{m}|st_minus_rc']; xs = d('W', 'st') - d('N', 'st')
                    fam.append(dict(part=part, arm='H4', set=key, model=m, path='st-rc', test='W-N fixed minus recentred (two-sided)', effect=q['diff'],
                                    ci95=q['ci95'], p=tp(xs - x, 0)))
        if part == 'jc':
            for key in ('H3v_60', 'H3v_100'):
                r = P[part]['H3v'].get(key, {})
                if 'contrasts' not in r: continue
                c = r['contrasts']['E_vs_B']
                for mp, M in c.items():
                    if '|' not in mp or mp.split('|')[1] not in ('st', 'rc') or 'ref' not in M: continue
                    m, p = mp.split('|'); d = lambda v, pp: g(part, key, v, pp, m) - g(part, key, 'id', pp, m)
                    rx = (g(part, key, 'E', 'st', 'MLPx_jc') - g(part, key, 'B', 'st', 'MLPx_jc')); x = d('E', p) - d('B', p)
                    fam.append(dict(part=part, arm='H3v', set=key, model=m, path=p, test='E-B vs MLPx (one-sided, down)', effect=M['diff'], ci95=M['ci95'],
                                    ratio=M['ratio_MLPx'], ratio_ci95=M['ratio_MLPx_ci95'], p=tp(x - rx, -1)))
                    if p == 'rc' and f'{m}|st_minus_rc' in c:
                        q = c[f'{m}|st_minus_rc']; xs = d('E', 'st') - d('B', 'st')
                        fam.append(dict(part=part, arm='H3v', set=key, model=m, path='st-rc', test='E-B fixed minus recentred (two-sided)', effect=q['diff'],
                                        ci95=q['ci95'], p=tp(xs - x, 0)))
                for mp, C in r['top_call_10class'].items():
                    m, p = mp.split('|')
                    fam.append(dict(part=part, arm='H3v', set=key, model=m, path=p, test='10-class top call E vs B (McNemar)', effect=C['E_minus_B_pp'],
                                    ci95=C['E_minus_B_ci95'], p=C['mcnemar_E_vs_B']['p']))
            for key in ('T60', 'T100'):
                for m in ('ParT_full', 'PN_full'):
                    for gname in GROUPS:
                        k0, k1 = '__'.join(('jc', key, 'decomp', m, 'none', 'dN')), '__'.join(('jc', key, 'decomp', m, gname, 'dN'))
                        if k0 in Z.files and k1 in Z.files:
                            x = Z[k0] - Z[k1]
                            fam.append(dict(part='jc', arm='decomp', set=key, model=m, path='rc', test=f'dN change when {gname} (two-sided)', effect=float(x.mean()), p=tp(x, 0)))
    for part in ('jc', 'top', 'hls'):
        if part not in P: continue
        for mn, R in P[part]['H13']['models'].items():
            for tn, tv in R['tests'].items():
                if tn.startswith('ratio'): continue
                fam.append(dict(part=part, arm='H13', set=tn.split('|')[0], model=mn, path=tn.split('|')[1], test=tn, effect=[tv['a'], tv['b']], p=tv['p']))
    adj = holm([t['p'] for t in fam])
    for t, a in zip(fam, adj): t['holm'] = float(a)
    S = dict(n_tests=len(fam), family='H4 (W-N vs MLPx and st-rc), H3v (E-B vs MLPx, st-rc, 10-class McNemar), decomp, H13 McNemar; '
                                       'Holm over all', n_holm_lt_05=int(sum(a < 0.05 for a in adj)), tests=fam)
    (RES / f'{TAG}_summary.json').write_text(json.dumps(jsonable(S), indent=1)); log(f'summary: {len(fam)} tests, {S["n_holm_lt_05"]} Holm < 0.05')
    for t in fam:
        if t['arm'] in ('H4', 'H3v', 'decomp') and t['path'] != 'st' or t['arm'] == 'H3v':
            log(f"  {t['part']} {t['arm']} {t['set']} {t['model']} {t['path']} {t['test']}: {t['effect'] if not isinstance(t['effect'], float) else round(t['effect'], 3)} p {t['p']:.2g} holm {t['holm']:.2g}")
    figure(P)


def figure(P):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nets = [('ParT_full', 'JetClass ParT full'), ('ParT_kin', 'JetClass ParT kin'), ('PN_full', 'JetClass PN full'), ('PN_kin', 'JetClass PN kin'),
            ('ParT_top', 'Top ParT'), ('PN_top', 'Top PN'), ('LNet_top', 'Top LorentzNet'), ('PFN_top', 'Top PFN'), ('EFN_top', 'Top EFN')]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, (part, title) in zip(axs, (('jc', 'JetClass tops, 550-650 GeV'), ('top', 'Top Landscape tops'))):
        if part not in P: continue
        H4 = P[part]['H4']; xs = np.arange(len(nets))
        for j, (key, col) in enumerate((('T60', '#2a6fdb'), ('T100', '#d9822b'))):
            if key not in H4 or 'contrasts' not in H4[key]: continue
            c = H4[key]['contrasts']['W_vs_N']; off = (j - 0.5) * 0.36
            for i, (m, _) in enumerate(nets):
                for p, mk, fc in (('rc', 'o', col), ('st', 'x', '0.55')):
                    M = c.get(f'{m}|{p}')
                    if not M or 'ratio_MLPx' not in M: continue
                    lo_, hi_ = M['ratio_MLPx_ci95']; v = M['ratio_MLPx']
                    ax.errorbar(i + off + (0.08 if p == 'st' else 0), v, yerr=[[max(v - lo_, 0)], [max(hi_ - v, 0)]], fmt=mk, color=fc, ms=6 if p == 'rc' else 5,
                                capsize=2, lw=1, label=f'{key} {"recentred" if p == "rc" else "fixed axis (as H19)"}' if i == 0 else None)
        ax.axhline(1, color='k', lw=0.8, ls='--'); ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(xs); ax.set_xticklabels([n for _, n in nets], rotation=40, ha='right', fontsize=8); ax.set_title(title, fontsize=10)
        ax.set_ylim(-1.5, 9)
    axs[0].set_ylabel('W-pair response / side-effect model\n(W − N log-odds ÷ MLPx W − N)')
    axs[0].legend(fontsize=7, loc='upper left')
    fig.suptitle('W-pair mass moved to 60 or 100 GeV at fixed jet mass and τ32 (edit W minus non-W control N): rebuilt about each edited jet\'s own axis,\n'
                 'ParT, ParticleNet and LorentzNet respond at 2.6–6.5× the side-effect model, PFN and EFN at 0.9–2.6× (dashed line = 1×)', fontsize=9.5)
    fig.tight_layout(); f = RES / 'h19_wpair.png'; fig.savefig(f, dpi=150); log(f'wrote {f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['jc', 'top', 'hls', 'decomp', 'summary']); a = ap.parse_args()
    log(f'H23 part {a.part}; smoke {SMOKE}')
    dict(jc=part_jc, top=part_top, hls=part_hls, decomp=part_decomp, summary=part_summary)[a.part]()
    log('done')
