"""H64 verification (skeptic, loop D). Fresh samples, the H64 edits, and three additions:
  1. independent boost check: an explicit 4x4 Lorentz matrix (rapidity from atanh(|P|/E), not the asinh(|P|/m) of h12.boost) applied
     to every constituent four-vector, compared with h12.boost; and the jet mass / jet pT recomputed from the arrays the model is
     actually fed (after prep), unedited vs B.
  2. in-range restriction: R_B^w recomputed on the jets whose unedited AND B-edited jet pT both lie inside the dataset's pT window
     (the same jet subset for B, P, A, C).
  3. composition control C(lam) = P(lam) o A(lam): every (deta, dphi) / lam, then every four-vector x lam. To first order it changes
     (log m, log pT, log angle) by (0, +log lam, -log lam), the same as B, but it is not a Lorentz transformation. Delta_B vs Delta_C
     asks whether the boost response is the response to (pT up, angles down at fixed m), or to something specific to the exact boost.
  4. extra references: MLP(w,pT) on [log w, log pT] with w = sqrt(M2/M0) (pT-weighted RMS angle about the stored axis; an angular-size +
     pT reader with no invariant-mass input), and MLP(m) on [log m] (an exact invariant-mass reader: Delta_B = 0 by construction).
Samples (fresh): hls4ml jedi_core-style draw seed 7461 (2,000 per class, + 3,000 per class reference rng 7471); JetClass jc_sample(3000,
  seed 7462) split 1,500 evaluated + 1,500 reference per class (rng 7472); Top test 2,000 + 2,000 (rng 7463) + 10,000 + 10,000 reference
  (rng 7473); QG test 2,000 + 2,000 (rng 7464) + 10,000 + 10,000 reference (rng 7474).
Run
  ./part_job.sh d-h64-verify 'python -m interp.h64_verify --part jetclass && python -m interp.h64_verify --part top && python -m interp.h64_verify --part qg'
  ./job.sh d-h64-verify-hls 'python -m interp.h64_verify --part hls'
  -> $AI_RESULTS/physics/h64_verify_<part>.json
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
from interp import h12_boost_panel as H  # noqa: E402
from interp import h64_axis_boost_panel as M  # noqa: E402

OUT = M.OUT
LAMS = (0.8, 0.9, 1.1, 1.25)
NB = 1000
T0_ = time.time()
SEEDS = dict(hls=7461, jetclass=7462, top=7463, qg=7464)
KINDS = ('B', 'P', 'A', 'C', 'R')
M.REF_FEATS['MLP(w,pT)'] = lambda O: np.stack([np.log(np.maximum(O['mA'] / np.maximum(O['sumpt'], 1e-9), 1e-5)), np.log(O['jpt'])], 1)
M.REF_FEATS['MLP(m)'] = lambda O: np.log(np.maximum(O['m'], 1e-3))[:, None]


def log(*a):
    print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


def explicit_boost_p4(T, lam):
    """independent implementation: (E, px, py, pz) of every constituent after the 4x4 boost along the jet 3-momentum, |P| -> lam |P|"""
    px, py, pz, E = H.four(T); P = np.stack([px.sum(1), py.sum(1), pz.sum(1)], 1); EJ = E.sum(1); pa = np.linalg.norm(P, axis=1)
    m = np.sqrt(np.maximum(EJ ** 2 - pa ** 2, 0)); E1 = np.sqrt(m ** 2 + (lam * pa) ** 2)
    zeta = np.arctanh(np.clip(lam * pa / E1, -1 + 1e-15, 1 - 1e-15)) - np.arctanh(np.clip(pa / EJ, -1 + 1e-15, 1 - 1e-15))
    n = P / pa[:, None]; g = np.cosh(zeta); gb = np.sinh(zeta); J = len(n)
    L = np.zeros((J, 4, 4)); L[:, 0, 0] = g; L[:, 0, 1:] = gb[:, None] * n; L[:, 1:, 0] = gb[:, None] * n
    L[:, 1:, 1:] = np.eye(3)[None] + (g - 1)[:, None, None] * n[:, :, None] * n[:, None, :]
    v = np.stack([E, px, py, pz], -1)
    return np.einsum('jab,jlb->jla', L, v) * (T['pt'] > 0)[..., None]


MFIX = False     # --mfix: P scales every constituent's pT (hence E, |p|) by lam at fixed constituent mass (default: whole four-vector x lam)


def pscale(T, lam):
    if not MFIX: return H.pscale(T, lam)[0]
    U = H.copyT(T); U['pt'] = T['pt'] * lam; return U


def compose_PA(T, lam):
    return pscale(H.ascale(T, lam)[0], lam)


def wsum(D, y, C, sel, s):
    """class-signed sum of per-class means on jets sel"""
    return sum(s[k] * D[sel & (y == k)].mean() for k in range(C) if (sel & (y == k)).any())


def ratio_boot(Dnum, Dden, y, C, sel, rng):
    """R^w = sum_k s_k mean_k(num) / sum_k s_k mean_k(den), s_k = sign(mean_k(den)); stratified bootstrap over jets in sel"""
    idx = {k: np.flatnonzero(sel & (y == k)) for k in range(C)}; idx = {k: v for k, v in idx.items() if len(v) >= 20}
    s = {k: np.sign(Dden[v].mean()) for k, v in idx.items()}
    num = sum(s[k] * Dnum[v].mean() for k, v in idx.items()); den = sum(s[k] * Dden[v].mean() for k, v in idx.items())
    bn, bd = np.zeros(NB), np.zeros(NB)
    for k, v in idx.items():
        I = v[rng.integers(0, len(v), (NB, len(v)))]; bn += s[k] * Dnum[I].mean(1); bd += s[k] * Dden[I].mean(1)
    ok = np.abs(den) >= 0.02
    return dict(value=float(num / den) if ok else None, ci=np.percentile(bn / bd, [2.5, 97.5]).tolist() if ok else None,
                num=float(num), den=float(den), n=int(sum(len(v) for v in idx.values())))


def diff_signed(Da, Db, y, C, sel, s, rng):
    """class-signed mean of (Da - Db) per jet, stratified bootstrap, normal-approx p (paired)"""
    idx = {k: np.flatnonzero(sel & (y == k)) for k in range(C)}; idx = {k: v for k, v in idx.items() if len(v) >= 20}
    d = Da - Db; val = sum(s[k] * d[v].mean() for k, v in idx.items())
    b = np.zeros(NB); var = 0.0
    for k, v in idx.items():
        b += s[k] * d[v[rng.integers(0, len(v), (NB, len(v)))]].mean(1); var += d[v].var(ddof=1) / len(v)
    p = float(math.erfc(abs(val) / math.sqrt(var) / math.sqrt(2))) if var > 0 else (1.0 if val == 0 else 0.0)
    return dict(mean=float(val), ci=np.percentile(b, [2.5, 97.5]).tolist(), p=p)


def vpanel(part, T0, y, Tp, yp, classes, models, prep, window, seed, input_mass):
    rng = np.random.default_rng(seed); C = len(classes); J = len(y)
    res = dict(part=part, N=J, labels=np.bincount(y, minlength=C).tolist(), classes=classes, window=window, models={}, checks={}, R={}, tests={})
    O0 = M.feats(T0); Op = M.feats(Tp); refs = M.train_refs(Op, yp)
    for k, clf in refs.items():
        models[k] = dict(f=(lambda O, clf=clf, k=k: clf.predict_log_proba(M.REF_FEATS[k](O))), src='table', role='ref')
    X0 = prep(H.sortT(T0)); L0 = {}
    for mn, Md in models.items():
        lg = Md['f'](X0 if Md['src'] == 'prep' else O0); c = lg.argmax(1); L0[mn] = M.logodds_all(lg)
        res['models'][mn] = dict(role=Md['role'], acc=float((c == y).mean()), acc_ci=H.boot_frac(c == y, rng), known=Md.get('known'))
        log(f'{part} {mn}: accuracy {res["models"][mn]["acc"]:.4f} (known {Md.get("known")})')
    m_in0, pt_in0 = input_mass(X0)
    DD = {mn: {} for mn in models}; JPT = {}; erng = np.random.default_rng(seed + 1000)
    for lam in LAMS:
        B, _ = H.boost(T0, lam); Dsp = H.displacement(T0, B)
        # independent boost check
        q = explicit_boost_p4(T0, lam); px, py, pz, E = H.four(B); qb = np.stack([E, px, py, pz], -1)
        scale = np.maximum(np.abs(q).max(-1, keepdims=True), 1e-9)
        Xb = prep(H.sortT(B)); m_in, pt_in = input_mass(Xb); good = O0['m'] > 1.0
        res['checks'][str(lam)] = dict(
            explicit_vs_h12_max_rel=float((np.abs(q - qb) / scale)[good].max()), n_jets_m_le_1GeV=int((~good).sum()),
            model_input_mass_rel_change_absmax=float(np.max(np.abs(m_in[good] / m_in0[good] - 1))),
            model_input_mass_rel_change_median=float(np.median(m_in[good] / m_in0[good] - 1)),
            model_input_jetpt_ratio_median=float(np.median(pt_in / pt_in0)), model_input_jetpt_ratio_q=np.percentile(pt_in / pt_in0, [0.5, 99.5]).tolist(),
            table_vs_input_mass_rel_absmax_unedited=float(np.max(np.abs(m_in0[good] / O0['m'][good] - 1))))
        log(f'{part} check lam {lam}: {res["checks"][str(lam)]}')
        for kd, U in (('B', B), ('P', pscale(T0, lam)), ('A', H.ascale(T0, lam)[0]), ('C', compose_PA(T0, lam)), ('R', H.rotate(T0, Dsp, erng)[0])):
            O1 = M.feats(U); X1 = Xb if kd == 'B' else prep(H.sortT(U)); JPT[(kd, lam)] = O1['jpt']
            if kd == 'C':
                res['checks'][str(lam)]['C_mass_rel_change_median'] = float(np.median(O1['m'][good] / O0['m'][good] - 1))
                res['checks'][str(lam)]['C_mass_rel_change_q'] = np.percentile(O1['m'][good] / O0['m'][good] - 1, [5, 95]).tolist()
            for mn, Md in models.items():
                lo = M.logodds_all(Md['f'](X1 if Md['src'] == 'prep' else O1)); DD[mn][(kd, lam)] = (lo - L0[mn])[np.arange(J), y]
        log(f'{part} lam {lam}: ' + ' '.join(f'{mn} B{DD[mn][("B", lam)].mean():+.3f} A{DD[mn][("A", lam)].mean():+.3f} C{DD[mn][("C", lam)].mean():+.3f}' for mn in models))
    inw = lambda v: (v >= window[0]) & (v <= window[1])
    allj = np.ones(J, bool); pvals, keys = [], []
    for mn in models:
        res['R'][mn] = {}
        for lam in LAMS:
            sel_in = inw(O0['jpt']) & inw(JPT[('B', lam)])
            D = DD[mn]; sA = {k: np.sign(D[('A', lam)][y == k].mean()) for k in range(C)}
            r = dict(R_B_w=ratio_boot(D[('B', lam)], D[('A', lam)], y, C, allj, rng),
                     R_B_w_inrange=ratio_boot(D[('B', lam)], D[('A', lam)], y, C, sel_in, rng),
                     R_C_w=ratio_boot(D[('C', lam)], D[('A', lam)], y, C, allj, rng),
                     inrange_frac=float(sel_in.mean()),
                     B_minus_C_signed=diff_signed(D[('B', lam)], D[('C', lam)], y, C, allj, sA, rng),
                     B_signed=diff_signed(D[('B', lam)], np.zeros(J), y, C, allj, sA, rng),
                     R_signed=diff_signed(D[('R', lam)], np.zeros(J), y, C, allj, sA, rng),
                     per_class={classes[k]: {kd: float(D[(kd, lam)][y == k].mean()) for kd in KINDS} for k in range(C)})
            res['R'][mn][str(lam)] = r
            if models[mn]['role'] == 'trained':
                for t in ('B_minus_C_signed', 'B_signed'):
                    keys.append(f'{mn}|{lam}|{t}'); pvals.append(r[t]['p'])
            log(f'{part} {mn} lam {lam}: R_B^w {r["R_B_w"]["value"]} {r["R_B_w"]["ci"]} | in-range ({r["inrange_frac"]:.2f}) {r["R_B_w_inrange"]["value"]} '
                f'{r["R_B_w_inrange"]["ci"]} | R_C^w {r["R_C_w"]["value"]} | B-C {r["B_minus_C_signed"]["mean"]:+.3f} {r["B_minus_C_signed"]["ci"]}')
    res['tests'] = dict(zip(keys, H.holm(pvals))) if pvals else {}
    return res


def save(res, part):
    part = part + ('_mfix' if MFIX else '') + ('' if LAMS == (0.8, 0.9, 1.1, 1.25) else '_smalllam')
    res['lams'] = list(LAMS); res['mfix'] = MFIX
    OUT.mkdir(parents=True, exist_ok=True); p = OUT / f'h64_verify_{part}.json'
    p.write_text(json.dumps(res, indent=1, default=float)); log(f'wrote {p}')


def mass_pt_from_ptetaphi(pt, eta, phi, m=None):
    real = pt > 0; m = np.zeros_like(pt) if m is None else m
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta); E = np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)
    S = [np.where(real, a, 0).sum(1) for a in (px, py, pz, E)]
    return np.sqrt(np.maximum(S[3] ** 2 - S[0] ** 2 - S[1] ** 2 - S[2] ** 2, 0)), np.hypot(S[0], S[1])


def part_hls():
    from interp.jedi_core import Jedi
    seed = SEEDS['hls']; T0, y, Tp, yp, _ = M.hls_tables(seed)
    nets = dict(jedi_n64=Jedi(64, '3-feature'))
    models = {k: dict(f=(lambda X, n=n: n.run(X)['logits']), src='prep', role='trained', known='0.8068 (E26)') for k, n in nets.items()}
    im = lambda X: mass_pt_from_ptetaphi(X[..., 0].astype(np.float64), X[..., 1].astype(np.float64), X[..., 2].astype(np.float64))
    res = vpanel('hls', T0, y, Tp, yp, H.HLS_CL, models, H.hls_prep, H.hls_window(T0), seed, im)
    save(res, 'hls')


def part_jetclass():
    from interp.h1_irc_panel import jc_sample, jc_table, pn_logits
    from interp.part_core import ParTNetwork, CL, inputs, build, sub, jmass
    from interp.p1v_verify_part import fwd
    from interp.pn_net import load_particlenet
    from jetdistill.config import MODELS
    dev = os.environ.get('JETDISTILL_DEVICE', 'cpu'); seed = SEEDS['jetclass']
    Jall = jc_sample(3000, seed); yall = Jall['label'].astype(int); r = np.random.default_rng(seed + 10); ev, rf = [], []
    for c in range(len(CL)):
        ii = r.permutation(np.flatnonzero(yall == c)); ev.append(ii[:1500]); rf.append(ii[1500:])
    ev, rf = np.sort(np.concatenate(ev)), np.sort(np.concatenate(rf))
    J0 = sub(Jall, ev); y = yall[ev]
    tab = lambda Jx: (lambda Tj: dict(pt=Tj['pt'], eta=Tj['eta'], phi=Tj['phi'], m=Tj['m'], ax_eta=Tj['ax_eta'], ax_phi=Tj['ax_phi'], ext=Tj['ext']))(jc_table(Jx))
    T0 = tab(J0); Jr = sub(Jall, rf); yp = yall[rf]; Tp = tab(Jr); Tp['ext'] = {}; del Jall, Jr
    log(f'JetClass: {len(ev)} evaluated + {len(rf)} reference jets')

    def prep(T):
        Jx = dict(J0); Jx['jet_eta'] = T['ax_eta'].astype(np.float32); Jx['jet_phi'] = T['ax_phi'].astype(np.float32)
        sg = np.where(Jx['jet_eta'] < 0, -1.0, 1.0)[:, None]; de, dp = H.rel(T)
        return build(Jx, T['pt'], sg * de, dp, T['m'], extra=T['ext'])
    pt_f = lambda net, n: (lambda J: fwd(net, *inputs(J, n))[1].astype(np.float64))
    known = dict(ParT_full='0.8575 (H12)', ParT_kin='0.7388 (H12)', PN_kin='0.7004 (H12)')
    models = {f'ParT_{n}': dict(f=pt_f(ParTNetwork(n), n), src='prep', role='trained', known=known[f'ParT_{n}']) for n in ('full', 'kin')}
    pn = load_particlenet(Path(MODELS).parent / 'particlenet' / 'ParticleNet_kin.pt', 7, dev)
    models['PN_kin'] = dict(f=(lambda J: pn_logits(pn, J, 'kin')), src='prep', role='trained', known=known['PN_kin'])
    im = lambda J: (jmass(J).astype(np.float64), J['jet_pt'].astype(np.float64))
    res = vpanel('jetclass', T0, y, Tp, yp, list(CL), models, prep, [500.0, 1000.0], seed, im)
    save(res, 'jetclass')


def part_top():
    import torch
    from interp import data_top as dt
    from interp.h4_top_wmass import WeaverNet, LNet
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; seed = SEEDS['top']
    d = dt.DST / 'test'; ylab = np.load(d / 'label.npy'); ev, rf = M.split_rows(ylab, 2000, 10000, seed)
    Fm = np.load(d / 'full.npy', mmap_mode='r'); jet = np.load(d / 'jet.npy')
    tab = lambda ii: H.table_from_full(np.asarray(Fm[ii]), np.minimum(jet[ii, 5].astype(np.int64), Fm.shape[1]))
    T0, Tp = tab(ev), tab(rf); y, yp = ylab[ev].astype(int), ylab[rf].astype(int); log(f'Top: {len(ev)} evaluated + {len(rf)} reference jets')

    def prep(T):
        pt, eta, phi, m = T['pt'], T['eta'], T['phi'], T['m']; real = pt > 0
        P4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)], -1) * real[..., None]
        return dict(P4=P4.astype(np.float32), nn=real.sum(1))
    two = lambda s: np.stack([np.zeros_like(s), s], 1); models = {}
    known = dict(ParT_kin='0.9398 (H12)', PN_kin='0.9402 (H12)', LorentzNet='0.9415 (H12)')
    for kind, dname in (('ParT', 'part_kin'), ('PN', 'pnet_kin')):
        net = WeaverNet(kind, M.V / 'models/top' / dname / 'net_best_epoch_state.pt', dev)
        models[f'{kind}_kin'] = dict(f=(lambda X, net=net: two(net.score(X['P4'], X['nn']))), src='prep', role='trained', known=known[f'{kind}_kin'])
    ln = LNet(dev, mass64=True); order = np.random.default_rng(seed + 5).permutation(len(ev))
    models['LorentzNet'] = dict(f=(lambda X: two(ln.score(X['P4'], X['nn'], order))), src='prep', role='trained', known=known['LorentzNet'])

    def im(X):
        P = X['P4'].astype(np.float64).sum(1); return np.sqrt(np.maximum(P[:, 3] ** 2 - (P[:, :3] ** 2).sum(1), 0)), np.hypot(P[:, 0], P[:, 1])
    res = vpanel('top', T0, y, Tp, yp, ['QCD', 'top'], models, prep, [550.0, 650.0], seed, im)
    save(res, 'top')


def part_qg():
    import torch
    from interp import data_qg as dq
    from interp.h5_qg_counting import Models
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; seed = SEEDS['qg']
    d = dq.DST / 'test'; ylab = np.load(d / 'label.npy'); ev, rf = M.split_rows(ylab, 2000, 10000, seed)
    Fm = np.load(d / 'full.npy', mmap_mode='r'); jet = np.load(d / 'jet.npy'); pidm = np.load(d / 'pid.npy', mmap_mode='r')

    def tab(ii, with_pid):
        T = H.table_from_full(np.asarray(Fm[ii]), np.minimum(jet[ii, 5].astype(np.int64), Fm.shape[1])); T['m'] = np.zeros_like(T['m'])
        if with_pid: T['ext'] = dict(pid=np.asarray(pidm[ii]).astype(np.int64) * (T['pt'] > 0))
        return T
    T0, Tp = tab(ev, True), tab(rf, False); y, yp = ylab[ev].astype(int), ylab[rf].astype(int); log(f'QG: {len(ev)} evaluated + {len(rf)} reference jets')

    def prep(T):
        real = T['pt'] > 0
        return dict(pt=T['pt'], eta=T['eta'], phi=T['phi'], pid=T['ext']['pid'].astype(np.int64) * real, n=real.sum(1))
    Mq = Models(dev)
    for name, arch in (('part_kin', 'part'), ('pnet_kin', 'pn')): Mq.add_weaver(name, arch, False)
    two = lambda s: np.stack([np.zeros_like(s), s], 1)
    known = dict(part_kin='0.8263 (H12)', pnet_kin='0.8269 (H12)')
    models = {k: dict(f=(lambda X, k=k: two(-Mq.logit(k, X))), src='prep', role='trained', known=known[k]) for k in ('part_kin', 'pnet_kin')}
    im = lambda X: mass_pt_from_ptetaphi(X['pt'], X['eta'], X['phi'])
    res = vpanel('qg', T0, y, Tp, yp, ['gluon', 'quark'], models, prep, [500.0, 550.0], seed, im)
    save(res, 'qg')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['hls', 'jetclass', 'top', 'qg'])
    ap.add_argument('--mfix', action='store_true'); ap.add_argument('--lams', type=float, nargs='+')
    a = ap.parse_args(); MFIX = a.mfix
    if a.lams: LAMS = tuple(a.lams)
    log(f'H64 verify part {a.part} mfix {MFIX} lams {LAMS}')
    dict(hls=part_hls, jetclass=part_jetclass, top=part_top, qg=part_qg)[a.part]()
    log('done')
