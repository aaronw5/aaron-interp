"""H13 verify (loop D): fresh-sample rerun of the H13 pileup panel's key numbers, two missing content controls, a per-jet
concordance test of "JEDI-linear's W call follows the ungroomed-mass window", and a working-point-matched comparison of the
Top Landscape networks with the cut-based top taggers.

Samples (rows of the H13 seed-302 samples excluded)
  hls  jedi_core n64 test jets, 2,000 per class, seed 3131 (10,000 jets)           JEDI-linear official n64 3-feature + redrawn null
  jc   JetClass, 500 random rows per class of each class's first test file, seed 3132 (5,000 jets)
                                                                                    ParT_full, ParT_kin, PN_full, PN_kin
Edit and the H13 conditions (same code path as interp/h13_pileup_panel.py: inject, finalize, eaxis)
  pu_rho     H13 edit (rho 5, 10, 20, 40 GeV per unit area), E-scheme axis recomputed from all particles
  pufix_rho  same injected particles, inputs about the stored axis
  c3_rho     original particles about the pu_rho axis (pure translation)
New controls (inputs about the stored axis, so they compare with pufix_rho and carry no axis offset)
  c4_rho     core: the same injected particles (same count, same q) with their displacement from the axis divided by 20
             (r <= 0.04): same multiplicity and added sum pT as pufix, mass change reduced ~400x
  c5_rho     one wide particle: one massless particle with pT = (sum pT kept by the model in pufix) - (sum pT of the unedited
             jet), at a random azimuth and at the radius that makes the invariant mass of the kept particles equal to pufix's
             (bisection): same kept sum pT and same kept mass as pufix, multiplicity +1
  c2k_rho    the c5 particle placed on the stored axis (r = 0): same kept sum pT, mass change ~0, multiplicity +1
Predictions: content = mass  -> c5 ~ pufix, c4 ~ c2k ~ small;  content = multiplicity / soft count -> c4 ~ pufix, c5 ~ c2k;
             content = sum pT only -> c2k ~ c4 ~ c5 ~ pufix.
Per-jet concordance (true W jets, pu_rho): network keeps the W call vs the 65-95 GeV window on the ungroomed mass of all
  particles (as H13) and of the kept particles (what the model sees); P(net W | window pass) - P(net W | window fail) with a
  bootstrap CI, Cohen's kappa, and where the lost W calls go (QCD classes vs heavier classes).
Classical: m, m_SD (C/A, zcut 0.1, beta 0) on rows [::5], all particles, for id, pu_rho, c4_rho, c5_rho.
--part top (GPU): Top Landscape, 2,500 top + 2,500 QCD test jets, seed 3133 (H13 rows excluded); ParT_kin, PN_kin, LorentzNet,
  PFN, EFN, permuted PFN; the same conditions (axis from the kept particles, 200 slots); top tag at the eps_S = 0.5 threshold and
  at the threshold matching top(mSD, tau32)'s starting efficiency, frozen on the unedited tops; retention on true tops.
--part topwp: from the H13 Top per-jet files (h13_top_perjet.npz, h13_obs_top_perjet.npz): retention of the top tag
  (tag rate after / before, on true tops) for each network at frozen thresholds set on the unedited tops at eps_S = 0.5 and at
  the starting efficiency of each cut-based tagger (top(mSD, tau32), top(m, tau32)), against that tagger's retention, paired on
  the same jets (rows [::5]); bootstrap over tops (2,000), Holm over all comparisons.
--part summary: Holm over every McNemar / bootstrap test of the three parts.
Outputs: $AI_RESULTS/physics/h13_verify_{hls,jc,topwp,summary}.json
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
from interp.h1_irc_panel import copyT, finalize, jet4, wrap, boot_binary, binom_two_sided, holm  # noqa: E402
from interp.h13_pileup_panel import (eaxis, dR, spectrum, inject, mass_all, moments, obs_of, RHOS, OUT)  # noqa: E402

T0_ = time.time()
SENT = 50.0          # sentinel eta marking the added particle through finalize
NB = 2000


def log(*a):
    print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


# ------------------------------------------------------------------------------------------------ samples (H13 rows excluded)
def hls_fresh(per, seed=3131, excl_seed=302, excl_per=2000):
    from jetdistill.config import RESULTS
    from interp.h1_irc_panel import hls_table
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); y = np.asarray(d['y'])
    r0 = np.random.default_rng(excl_seed); old = np.concatenate([r0.choice(np.flatnonzero(y == c), excl_per, replace=False) for c in range(5)])
    rng = np.random.default_rng(seed)
    idx = np.sort(np.concatenate([rng.choice(np.setdiff1d(np.flatnonzero(y == c), old), per, replace=False) for c in range(5)]))
    x = np.asarray(d['x'], np.float32)[idx]
    log(f'hls4ml fresh sample: {len(idx)} jets (seed {seed}; {len(np.intersect1d(idx, old))} overlap with H13)')
    return hls_table(x), y[idx].astype(int)


def jc_fresh(per, seed=3132, excl_seed=302, excl_per=1000):
    import uproot
    from interp.part_core import read_root, fof, cat, sub, CL
    r0 = np.random.default_rng(excl_seed); rng = np.random.default_rng(seed); Js = []
    for c in CL:
        f = fof(c); n = uproot.open(f)['tree'].num_entries
        old = r0.choice(n, excl_per, replace=False)
        idx = np.sort(rng.choice(np.setdiff1d(np.arange(n), old), per, replace=False)); parts = []
        for a in range(0, n, 20000):
            loc = idx[(idx >= a) & (idx < a + 20000)] - a
            if len(loc): parts.append(sub(read_root(f, start=a, stop=min(a + 20000, n)), loc))
        Js.append(cat(parts)); log(f'JetClass fresh {c}: {per} of {n} rows (H13 rows excluded)')
    return cat(Js)


# ------------------------------------------------------------------------------------------------ conditions
def add_one(T0, Q, r, th, cap):
    """T0 + one massless particle of pT Q at (stored axis + r (cos th, sin th)); returns finalized table"""
    U = copyT(T0, 1); c = U['pt'].shape[1] - 1; ok = Q > 0
    U['pt'][:, c] = np.where(ok, Q, 0.0)
    U['eta'][:, c] = np.where(ok, T0['ax_eta'] + r * np.cos(th), 0.0); U['phi'][:, c] = np.where(ok, wrap(T0['ax_phi'] + r * np.sin(th)), 0.0)
    if U['ext']:
        U['ext']['isNeutralHadron'][:, c] = ok; U['ext']['mask'][:, c] = ok
    return finalize(U, cap)[0], mass_all(U)


def match_radius(T0, Q, th, target, cap, rmax=2.0, it=50):
    """per-jet radius of the added particle so that the kept-particle mass equals target (bisection; kept set fixed by pT)"""
    U = copyT(T0, 1); c = U['pt'].shape[1] - 1; ok = Q > 0
    U['pt'][:, c] = np.where(ok, Q, 0.0); U['eta'][:, c] = np.where(ok, SENT, 0.0)
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
    return r, dict(abs_dm_median=float(np.median(np.abs(m_of(r) - target)[ok])), at_rmax=float((r > rmax * 0.999)[ok].mean()),
                   at_zero=float((r < 1e-6)[ok].mean()), r_median=float(np.median(r[ok])) if ok.any() else None)


def core(U, T0, W, f=20.0):
    V = copyT(U); on = V['pt'][:, W:] > 0
    de = V['eta'][:, W:] - T0['ax_eta'][:, None]; dp = wrap(V['phi'][:, W:] - T0['ax_phi'][:, None])
    V['eta'][:, W:] = np.where(on, T0['ax_eta'][:, None] + de / f, 0.0); V['phi'][:, W:] = np.where(on, wrap(T0['ax_phi'][:, None] + dp / f), 0.0)
    return V


def conditions(T0, spec, seed, cap, pre_cap=True):
    """yields (name, rho, Uf, ae, ap, m_all_pre_cut, U_pre_cut); pre_cap False (Top): E-scheme axis of the kept particles"""
    st = (T0['ax_eta'], T0['ax_phi']); W = T0['pt'].shape[1]
    Uf0, _ = finalize(T0, cap); yield ('id', 0, Uf0, *st, mass_all(T0), T0)
    for rho in RHOS:
        rng = np.random.default_rng([seed, rho, 0])
        U, inf = inject(T0, rho, spec, rng)
        Uf, _ = finalize(U, cap); ax = eaxis(U) if pre_cap else eaxis(Uf)
        yield (f'pu_{rho}', rho, Uf, ax[0], ax[1], mass_all(U), U)
        yield (f'pufix_{rho}', rho, Uf, *st, mass_all(U), U)
        yield (f'c3_{rho}', rho, Uf0, ax[0], ax[1], mass_all(T0), T0)
        C = core(U, T0, W); Cf, _ = finalize(C, cap); yield (f'c4_{rho}', rho, Cf, *st, mass_all(C), C)
        Q = np.maximum((Uf['pt']).sum(1) - Uf0['pt'].sum(1), 0.0); th = 2 * np.pi * np.random.default_rng([seed, rho, 7]).random(len(Q))
        target = mass_all(Uf); r, info = match_radius(T0, Q, th, target, cap)
        log(f'  c5_{rho}: added pT median {np.median(Q):.2f} GeV; radius match {info}')
        F5, m5 = add_one(T0, Q, r, th, cap); yield (f'c5_{rho}', rho, F5, *st, m5, None)
        F2, m2 = add_one(T0, Q, np.zeros(len(Q)), th, cap); yield (f'c2k_{rho}', rho, F2, *st, m2, None)


# ------------------------------------------------------------------------------------------------ statistics
def bdiff(a, b, rng, B=NB):
    """paired bootstrap of mean(a) - mean(b) (a, b per-jet arrays); two-sided bootstrap p for 0"""
    n = len(a); d = np.asarray(a, float) - np.asarray(b, float); idx = rng.integers(0, n, (B, n)); bs = d[idx].mean(1)
    return dict(diff=float(d.mean()), ci=np.percentile(bs, [2.5, 97.5]).tolist(), p=float(min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))))


def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool); n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return dict(edit=float(a.mean()), control=float(b.mean()), n_edit_only=n10, n_control_only=n01, p=binom_two_sided(n10, n10 + n01))


def concordance(net_keep, win_keep, dest, classes, rng, B=NB):
    n = len(net_keep); a = net_keep.astype(float); w = win_keep.astype(bool)
    pw, pf = (a[w].mean() if w.any() else np.nan), (a[~w].mean() if (~w).any() else np.nan)
    bs = []
    for _ in range(B):
        i = rng.integers(0, n, n); ww = w[i]; aa = a[i]
        if ww.any() and (~ww).any(): bs.append(aa[ww].mean() - aa[~ww].mean())
    po = (a == w).mean(); pe = a.mean() * w.mean() + (1 - a.mean()) * (1 - w.mean()); kappa = (po - pe) / max(1 - pe, 1e-12)
    lost = ~net_keep.astype(bool)
    return dict(n=int(n), window_pass=float(w.mean()), net_keep=float(a.mean()), p_net_given_pass=float(pw), p_net_given_fail=float(pf),
                diff=float(pw - pf) if np.isfinite(pw) and np.isfinite(pf) else None,
                diff_ci=np.percentile(bs, [2.5, 97.5]).tolist() if bs else None, kappa=float(kappa), agree=float(po),
                lost_to={classes[k]: int((dest[lost] == k).sum()) for k in range(len(classes))})


# ------------------------------------------------------------------------------------------------ multi-class parts
def run_mc(tag, T0, y, models, make_in, cap, classes, iW, heavier, smoke, obs_ok=True):
    rng = np.random.default_rng(31); spec = spectrum(T0)
    log(f'{tag}: N {len(y)}, spectrum q25 {spec["q25"]:.3f} mean {spec["mean"]:.3f}')
    C = {mn: {} for mn in models}; MOM = {}; MALL = {}; AX = {}; O = {}
    rows5 = np.arange(len(y))[::5]
    for name, rho, Uf, ae, ap, mall, Upre in conditions(T0, spec, 3131, cap):
        t = time.time(); X = make_in(Uf, ae, ap)
        for mn, f in models.items(): C[mn][name] = np.asarray(f(X), np.float64)
        MOM[name] = moments(Uf, ae, ap); MALL[name] = mall; AX[name] = (ae, ap)
        if obs_ok and name.split('_')[0] in ('id', 'pu', 'c4', 'c5'):
            src = Upre if Upre is not None else Uf
            Ua, _ = finalize({k: src[k][rows5] for k in ('pt', 'eta', 'phi', 'm')} | dict(ext={}, ax_eta=src['ax_eta'][rows5], ax_phi=src['ax_phi'][rows5], sgn=src['sgn'][rows5]), 10 ** 6)
            w = int((Ua['pt'] > 0).sum(1).max()); Us = {k: np.ascontiguousarray(Ua[k][:, :w]) for k in ('pt', 'eta', 'phi', 'm')}
            O[name] = obs_of(Us, ae[rows5], ap[rows5])
        log(f'{tag} {name}: dm_all {np.mean(mall - MALL["id"]):+.1f} dm_kept {np.mean(MOM[name]["m"] - MOM["id"]["m"]):+.1f} '
            f'dsumpt_kept {np.mean(MOM[name]["sumpt"] - MOM["id"]["sumpt"]):+.1f} | ' +
            ' '.join(f'{mn} {(C[mn][name].argmax(1) != C[mn]["id"].argmax(1)).mean():.3f}' for mn in models) + f' ({time.time() - t:.0f} s)')
    conds = list(MOM); res = dict(part=tag, N=int(len(y)), labels=np.bincount(y, minlength=len(classes)).tolist(), spectrum={k: v for k, v in spec.items() if k != 'pool'},
                                  conditions={}, models={}, classical={}, tests=[])
    st = AX['id']
    for name in conds:
        d = dict(dm_all=float(np.mean(MALL[name] - MALL['id'])), dm_kept=float(np.mean(MOM[name]['m'] - MOM['id']['m'])),
                 dsumpt_kept=float(np.mean(MOM[name]['sumpt'] - MOM['id']['sumpt'])))
        if name.startswith('pu_'):
            off = dR(*AX[name], *st); d['axis_offset_mrad'] = dict(p50=float(1e3 * np.median(off)), p90=float(1e3 * np.percentile(off, 90)))
        if name in O:
            d['dm_SD_rows5'] = float(np.mean(O[name]['mSD'] - O['id']['mSD'])); d['dm_rows5'] = float(np.mean(O[name]['m'] - O['id']['m']))
        res['conditions'][name] = d
    W = y == iW
    # classical W windows
    win = lambda m: (m > 65) & (m < 95)
    res['classical']['W_m_all'] = {n: float(win(MALL[n])[W].mean()) for n in conds}
    res['classical']['W_m_kept'] = {n: float(win(MOM[n]['m'])[W].mean()) for n in conds}
    if O:
        Wr = y[rows5] == iW
        res['classical']['W_mSD_rows5'] = {n: float(win(O[n]['mSD'])[Wr].mean()) for n in O}
        res['classical']['W_m_rows5'] = {n: float(win(O[n]['m'])[Wr].mean()) for n in O}
    for mn in models:
        c0 = C[mn]['id'].argmax(1); r = dict(acc=float((c0 == y).mean()), acc_ci=boot_binary(c0 == y, rng), conds={})
        CH = {}
        for name in conds:
            c1 = C[mn][name].argmax(1); ch = c1 != c0; CH[name] = ch
            r['conds'][name] = dict(changed=float(ch.mean()), ci=boot_binary(ch, rng), W_eff=float((c1[W] == iW).mean()),
                                    W_eff_ci=boot_binary(c1[W] == iW, rng))
        for rho in RHOS:
            k = lambda c: f'{c}_{rho}'
            tests = dict(pu_vs_pufix=mcnemar(CH[k('pu')], CH[k('pufix')]), pufix_vs_c4=mcnemar(CH[k('pufix')], CH[k('c4')]),
                         pufix_vs_c5=mcnemar(CH[k('pufix')], CH[k('c5')]), c5_vs_c2k=mcnemar(CH[k('c5')], CH[k('c2k')]),
                         c4_vs_c2k=mcnemar(CH[k('c4')], CH[k('c2k')]))
            for tn, tv in tests.items(): res['tests'].append([mn, rho, tn, tv['p']])
            r['conds'][k('pu')]['tests'] = tests
            # concordance on true W jets
            c1 = C[mn][k('pu')].argmax(1)
            r['conds'][k('pu')]['concordance_m_all'] = concordance((c1 == iW)[W], win(MALL[k('pu')])[W], c1[W], classes, rng)
            r['conds'][k('pu')]['concordance_m_kept'] = concordance((c1 == iW)[W], win(MOM[k('pu')]['m'])[W], c1[W], classes, rng)
            lost = W & (c0 == iW) & (c1 != iW)
            r['conds'][k('pu')]['lost_W_to_heavier'] = float(np.isin(c1[lost], heavier).mean()) if lost.any() else None
        res['models'][mn] = r
        log(f'{tag} {mn}: acc {r["acc"]:.4f}; W eff ' + ' '.join(f'{n} {r["conds"][n]["W_eff"]:.3f}' for n in conds if n == 'id' or n.endswith('_5') or n.endswith('_40')))
    h = holm([t[3] for t in res['tests']])
    for t, hh in zip(res['tests'], h): t.append(hh)
    return res


def save(res, name):
    OUT.mkdir(parents=True, exist_ok=True); p = OUT / f'h13_verify_{name}.json'; p.write_text(json.dumps(res, indent=1, default=float)); log(f'wrote {p}')


def part_hls(smoke, obs_ok=True):
    from interp.jedi_core import Jedi
    from interp.h13_pileup_panel import hls_in, HLS_CL
    T, y = hls_fresh(smoke or 2000)
    nets = dict(jedi=Jedi(64, '3-feature'), jedi_redrawn=Jedi(64, '3-feature', untrained=True, seed=0))
    models = {k: (lambda x, n=n: np.asarray(n.run(x)['logits'], np.float64)) for k, n in nets.items()}
    res = run_mc('hls', T, y, models, hls_in, 64, HLS_CL, 2, [3, 4], smoke, obs_ok)
    res['sample'] = f'n64 test jets, {smoke or 2000} per class, seed 3131, H13 seed-302 rows excluded'
    save(res, 'hls' + ('' if obs_ok else '_noobs') + ('_smoke' if smoke else ''))


def part_jc(smoke):
    from interp.part_core import ParTNetwork, CL, inputs
    from interp.p1v_verify_part import fwd
    from interp.pn_net import load_particlenet
    from interp.h1_irc_panel import pn_logits, jc_table
    from interp.h13_pileup_panel import jc_maker
    from jetdistill.config import MODELS
    dev = os.environ.get('JETDISTILL_DEVICE', 'cpu')
    J0 = jc_fresh(smoke or 500); y = J0['label'].astype(int); T = jc_table(J0); make = jc_maker(J0)
    models = {}
    for n in ('full', 'kin'):
        net = ParTNetwork(n); models[f'ParT_{n}'] = (lambda J, net=net, n=n: np.asarray(fwd(net, *inputs(J, n))[1], np.float64))
    for n, dims in (('full', 17), ('kin', 7)):
        pn = load_particlenet(Path(MODELS).parent / 'particlenet' / f'ParticleNet_{n}.pt', dims, dev)
        models[f'PN_{n}'] = (lambda J, pn=pn, n=n: pn_logits(pn, J, n))
    iW = CL.index('Wqq'); heavier = [CL.index(c) for c in ('Zqq', 'Tbqq', 'Tbl', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql')]
    res = run_mc('jc', T, y, models, make, 128, list(CL), iW, heavier, smoke)
    res['sample'] = f'JetClass first test file per class, {smoke or 500} random rows per class, seed 3132, H13 seed-302 rows excluded'
    save(res, 'jc' + ('_smoke' if smoke else ''))


# ------------------------------------------------------------------------------------------------ Top Landscape fresh sample
def top_fresh(per, seed=3133, excl_seed=302, excl_per=5000):
    from interp import data_top as dt
    from interp.h13_pileup_panel import top_table
    d = dt.DST / 'test'; y = np.load(d / 'label.npy')
    r0 = np.random.default_rng(excl_seed); old = np.concatenate([r0.choice(np.flatnonzero(y == c), excl_per, replace=False) for c in (0, 1)])
    rng = np.random.default_rng(seed); idx = np.sort(np.concatenate([rng.choice(np.setdiff1d(np.flatnonzero(y == c), old), per, replace=False) for c in (0, 1)]))
    full = np.asarray(np.load(d / 'full.npy', mmap_mode='r')[idx]); log(f'Top fresh sample: {len(idx)} jets (seed {seed}, H13 rows excluded)')
    return top_table(full), y[idx].astype(int)


def part_top(smoke):
    import torch
    from interp.h13_pileup_panel import TopModels, top_J, auc
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    T0, y = top_fresh(smoke or 2500); M = TopModels(dev); order = np.random.default_rng(305).permutation(len(y))
    rng = np.random.default_rng(34); spec = spectrum(T0); SC = {k: {} for k in M.names}; MOM = {}; MALL = {}; AX = {}; O = {}
    rows5 = np.arange(len(y))[::5]; top = y == 1
    for name, rho, Uf, ae, ap, mall, Upre in conditions(T0, spec, 3133, 200, pre_cap=False):
        t = time.time(); sc = M.score(top_J(Uf, ae, ap), order)
        for k, v in sc.items(): SC[k][name] = v
        MOM[name] = moments(Uf, ae, ap); MALL[name] = mall; AX[name] = (ae, ap)
        if name.split('_')[0] in ('id', 'pu', 'c4', 'c5'):
            src = Upre if Upre is not None else Uf
            Ua, _ = finalize({k: src[k][rows5] for k in ('pt', 'eta', 'phi', 'm')} | dict(ext={}, ax_eta=src['ax_eta'][rows5], ax_phi=src['ax_phi'][rows5], sgn=src['sgn'][rows5]), 10 ** 6)
            w = int((Ua['pt'] > 0).sum(1).max()); O[name] = obs_of({k: np.ascontiguousarray(Ua[k][:, :w]) for k in ('pt', 'eta', 'phi', 'm')}, ae[rows5], ap[rows5])
        log(f'top {name}: dm_all {np.mean(mall - MALL["id"]):+.1f} dsumpt_kept {np.mean(MOM[name]["sumpt"] - MOM["id"]["sumpt"]):+.1f} | ' +
            ' '.join(f'{k} {np.mean(SC[k][name] - SC[k]["id"]):+.3f}' for k in M.names) + f' ({time.time() - t:.0f} s)')
    conds = list(MOM); res = dict(part='top', N=int(len(y)), conditions={}, models={}, classical={}, tests=[])
    for name in conds:
        res['conditions'][name] = dict(dm_all=float(np.mean(MALL[name] - MALL['id'])), dsumpt_kept=float(np.mean(MOM[name]['sumpt'] - MOM['id']['sumpt'])))
        if name.startswith('pu_'):
            off = dR(*AX[name], *AX['id']); res['conditions'][name]['axis_offset_mrad'] = dict(p50=float(1e3 * np.median(off)), p90=float(1e3 * np.percentile(off, 90)))
    tr = y[rows5] == 1
    for tn, f in (('top_mSD_tau32', lambda o: (o['mSD'] > 105) & (o['mSD'] < 210) & (o['tau32'] < 0.65)), ('top_m_tau32', lambda o: (o['m'] > 105) & (o['m'] < 210) & (o['tau32'] < 0.65))):
        e0 = f(O['id'])[tr].mean(); res['classical'][tn] = dict(eff0=float(e0), retention={n: float(f(O[n])[tr].mean() / e0) for n in O})
    for k in M.names:
        s0 = SC[k]['id']; r = dict(auc=auc(s0, y), conds={})
        for wn, eps in (('eps50', 0.5), ('match_mSD', res['classical']['top_mSD_tau32']['eff0'])):
            t = float(np.quantile(s0[top], 1 - eps)); CH = {}
            for name in conds:
                c0, c1 = s0 > t, SC[k][name] > t; CH[name] = c0 != c1
                r['conds'].setdefault(name, {})[wn] = dict(changed=float(CH[name].mean()), ci=boot_binary(CH[name], rng), retention=float(c1[top].mean() / c0[top].mean()))
            if wn == 'eps50':
                for rho in RHOS:
                    kk = lambda c: f'{c}_{rho}'
                    tests = dict(pu_vs_pufix=mcnemar(CH[kk('pu')], CH[kk('pufix')]), pufix_vs_c4=mcnemar(CH[kk('pufix')], CH[kk('c4')]),
                                 pufix_vs_c5=mcnemar(CH[kk('pufix')], CH[kk('c5')]), c5_vs_c2k=mcnemar(CH[kk('c5')], CH[kk('c2k')]), c4_vs_c2k=mcnemar(CH[kk('c4')], CH[kk('c2k')]))
                    r['conds'][kk('pu')]['tests'] = tests
                    for tn, tv in tests.items(): res['tests'].append([k, rho, tn, tv['p']])
        res['models'][k] = r
        log(f'top {k}: AUC {r["auc"]:.4f}; retention eps50 ' + ' '.join(f'{n} {r["conds"][n]["eps50"]["retention"]:.2f}' for n in conds if n.endswith('_5') or n.endswith('_40')))
    for tn, v in res['classical'].items(): log(f'top {tn}: eff0 {v["eff0"]:.3f} retention ' + ' '.join(f'{n} {x:.2f}' for n, x in v['retention'].items()))
    h = holm([t[3] for t in res['tests']])
    for t, hh in zip(res['tests'], h): t.append(hh)
    save(res, 'top' + ('_smoke' if smoke else ''))


# ------------------------------------------------------------------------------------------------ Top working-point check
def part_topwp(smoke):
    Z = np.load(OUT / 'h13_top_perjet.npz'); O = np.load(OUT / 'h13_obs_top_perjet.npz'); y = Z['y']; rows = O['rows']
    assert np.array_equal(O['y'], y[rows]), 'obs rows do not match the model sample'
    rng = np.random.default_rng(33); top = y == 1; topr = y[rows] == 1
    tag = dict(top_mSD_tau32=lambda c: (O[f'{c}__mSD'] > 105) & (O[f'{c}__mSD'] < 210) & (O[f'{c}__tau32'] < 0.65),
               top_m_tau32=lambda c: (O[f'{c}__m'] > 105) & (O[f'{c}__m'] < 210) & (O[f'{c}__tau32'] < 0.65))
    res = dict(n_top=int(top.sum()), n_top_rows5=int(topr.sum()), classical={}, nets={}, tests=[])
    for tn, f in tag.items():
        c0 = f('id')[topr]; res['classical'][tn] = dict(eff0=float(c0.mean()), retention={str(r): float(f(f'pu_{r}')[topr].mean() / c0.mean()) for r in RHOS})
    nets = ('ParT', 'PN', 'LNet', 'PFN', 'EFN')
    for k in nets:
        s0 = Z[f'{k}__id']; r = {}
        wps = dict(eps50=0.5, **{f'match_{tn}': res['classical'][tn]['eff0'] for tn in tag})
        for wn, eps in wps.items():
            t = float(np.quantile(s0[top], 1 - eps)); d = dict(eps=eps, retention_all_tops={}, paired_rows5={})
            for rho in RHOS:
                s1 = Z[f'{k}__pu_{rho}']; a0, a1 = (s0 > t)[top], (s1 > t)[top]
                bs = []
                for _ in range(NB):
                    i = rng.integers(0, len(a0), len(a0)); bs.append(a1[i].mean() / max(a0[i].mean(), 1e-12))
                d['retention_all_tops'][str(rho)] = dict(value=float(a1.mean() / a0.mean()), ci=np.percentile(bs, [2.5, 97.5]).tolist())
                if wn.startswith('match_'):
                    tn = wn[6:]; n0, n1 = (s0[rows] > t)[topr], (s1[rows] > t)[topr]; k0, k1 = tag[tn]('id')[topr], tag[tn](f'pu_{rho}')[topr]
                    n = len(n0); bs = []
                    for _ in range(NB):
                        i = rng.integers(0, n, n); bs.append(n1[i].mean() / max(n0[i].mean(), 1e-12) - k1[i].mean() / max(k0[i].mean(), 1e-12))
                    bs = np.asarray(bs); p = float(min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
                    d['paired_rows5'][str(rho)] = dict(net=float(n1.mean() / n0.mean()), tagger=float(k1.mean() / k0.mean()),
                                                       diff=float(n1.mean() / n0.mean() - k1.mean() / k0.mean()), ci=np.percentile(bs, [2.5, 97.5]).tolist(), p=p)
                    res['tests'].append([k, rho, tn, p])
            r[wn] = d
        res['nets'][k] = r
        log(f'top {k}: ' + ' | '.join(f'{wn} eps {d["eps"]:.2f} ret ' + '/'.join(f'{d["retention_all_tops"][str(x)]["value"]:.2f}' for x in RHOS) for wn, d in r.items()))
    for tn in tag: log(f'top {tn}: eff0 {res["classical"][tn]["eff0"]:.3f} ret ' + '/'.join(f'{v:.2f}' for v in res['classical'][tn]['retention'].values()))
    h = holm([t[3] for t in res['tests']])
    for t, hh in zip(res['tests'], h): t.append(hh)
    save(res, 'topwp')


def part_summary():
    P = {p: json.loads((OUT / f'h13_verify_{p}.json').read_text()) for p in ('hls', 'jc', 'top', 'topwp') if (OUT / f'h13_verify_{p}.json').exists()}
    tests = [[p] + t[:4] for p, r in P.items() for t in r['tests']]
    h = holm([t[4] for t in tests])
    S = dict(n_tests=len(tests), tests=[t + [hh] for t, hh in zip(tests, h)], n_holm_lt_05=int(sum(hh < 0.05 for hh in h)))
    save(S, 'summary')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['hls', 'jc', 'top', 'topwp', 'summary'])
    ap.add_argument('--smoke', type=int, default=0); ap.add_argument('--no-obs', action='store_true'); a = ap.parse_args()
    log(f'H13 verify part {a.part} smoke {a.smoke}')
    if a.part == 'hls' and a.no_obs: part_hls(a.smoke, False); sys.exit(0)
    dict(hls=part_hls, jc=part_jc, top=part_top, topwp=part_topwp, summary=lambda s: part_summary())[a.part](a.smoke)
    log('done')
