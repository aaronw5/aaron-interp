"""H18 verify (loop D): fresh-seed rerun of the H18 soft-particle edits on QG Pythia, JetClass and Top Landscape, plus
three controls H18 does not have:

  e1p<x>   the e1 edit (one particle at a random existing particle's position + 0.05, that particle's fields) with its pT
           fixed at x = 0.13, 0.5, 1, 2 GeV instead of the jet's softest pT. H18's e1 adds +0.13 GeV (QG), +0.33 GeV (Top),
           +0.99 GeV (JetClass) on average, so its cross-dataset contrast mixes count with added-particle pT. Same rng as e1
           (same host particle and direction), so e1p<x> - e1 is paired.
  sp<k>    collinear split: k distinct random particles of the jet's softer half are each replaced by two halves
           (z = 0.5) at +-delta/2 about the parent (delta = 0.01; sp5d002: delta = 0.002). Count +k, sum pT and the
           pT-weighted centroid exact, IRC-safe observables changed by O(z delta). Identical edit on every dataset.
  c4m_20   the c4_20 particles (H13 disc at rho = 20 GeV / area, moved to r/20) merged into one particle carrying their summed
           four-momentum: same added four-momentum as c4_20, count +1 instead of +n. c4_20 - c4m_20 = count at fixed added
           four-momentum.
Also reruns e1, k1, k20, c4_20 for reproduction. LR(N) count-only reference built from H18's training dumps
(results/physics/h18_<ds>_train.npz on the volume).

Samples (seed 9311, random rows): QG Pythia test 4,000 q + 4,000 g; JetClass first test file, 2,500 random rows each of QCD,
Wqq, Tbqq (H18 seed-4181 rows excluded) + 300 of each other class (10-class accuracy only); Top Landscape test 3,000 + 3,000.

Run: ./part_job.sh d-h18-verify 'python -m interp.h18_verify --ds qg,jc,top'
  -> $AI_RESULTS/physics/h18_verify_<ds>.json, h18_verify.json
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from interp import h18_softcount_cross as X
from interp.h1_irc_panel import copyT, finalize, wrap, p3, from3, holm

SEED = 9311
OUT = X.OUT
T0_ = time.time()
EPT = (0.13, 0.5, 1.0, 2.0)
CONDS = ['base', 'e1'] + [f'e1p{x:g}' for x in EPT] + ['k1', 'k20', 'sp1', 'sp5', 'sp20', 'sp5d002', 'c4_20', 'c4m_20']
EDITS = CONDS[1:]


def log(*a):
    print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


def e_pt(T, ptval, rng):
    """X.e_onman with the added pT = ptval (None: the jet's softest pT); identical draws to X.e_onman"""
    n = X.nreal(T); J = len(n); rows = np.arange(J)
    i = np.minimum((rng.random(J) * n).astype(int), n - 1); th = 2 * np.pi * rng.random(J)
    U = copyT(T, 1); c = U['pt'].shape[1] - 1
    U['pt'][:, c] = T['pt'][rows, n - 1] if ptval is None else ptval
    U['eta'][:, c] = T['eta'][rows, i] + 0.05 * np.cos(th); U['phi'][:, c] = wrap(T['phi'][rows, i] + 0.05 * np.sin(th))
    X.copy_fields(U, c, T, i)
    return U


def splitk(T, k, delta, rng):
    n = X.nreal(T); J, W = T['pt'].shape; h = np.maximum(n // 2, 1); rows = np.arange(J)
    col = np.arange(W)[None, :]; soft = (col >= (n - h)[:, None]) & (col < n[:, None])
    order = np.argsort(np.where(soft, rng.random((J, W)), 2.0), 1)[:, :k]; valid = np.take_along_axis(soft, order, 1)
    U = copyT(T, k)
    for j in range(k):
        s = order[:, j]; ok = valid[:, j]; a = 2 * np.pi * rng.random(J)
        pt0, e0, f0, m0 = (T[q][rows, s] for q in ('pt', 'eta', 'phi', 'm'))
        de, dp = 0.5 * delta * np.cos(a), 0.5 * delta * np.sin(a)
        r, c = rows[ok], s[ok]
        U['pt'][r, c] = pt0[ok] / 2; U['eta'][r, c] = (e0 + de)[ok]; U['phi'][r, c] = wrap(f0 + dp)[ok]; U['m'][r, c] = m0[ok] / 2
        X.copy_fields(U, W + j, T, s)
        U['pt'][:, W + j] = np.where(ok, pt0 / 2, 0.0); U['eta'][:, W + j] = np.where(ok, e0 - de, 0.0)
        U['phi'][:, W + j] = np.where(ok, wrap(f0 - dp), 0.0); U['m'][:, W + j] = np.where(ok, m0 / 2, 0.0)
    return U, valid.sum(1)


def merge_added(V, T0, W):
    """the appended columns W.. of V (massless) merged into one particle with their summed four-momentum"""
    on = V['pt'][:, W:] > 0; pt, eta, phi = V['pt'][:, W:], V['eta'][:, W:], V['phi'][:, W:]
    px, py, pz = p3(pt, eta, phi); E = pt * np.cosh(eta)
    P = [np.where(on, a, 0).sum(1) for a in (px, py, pz, E)]
    ptm, etam, phim = from3(P[0], P[1], P[2]); mm = np.sqrt(np.maximum(P[3] ** 2 - P[0] ** 2 - P[1] ** 2 - P[2] ** 2, 0))
    ok = on.any(1)
    U = copyT(T0, 1); c = U['pt'].shape[1] - 1
    U['pt'][:, c] = np.where(ok, ptm, 0); U['eta'][:, c] = np.where(ok, etam, 0); U['phi'][:, c] = np.where(ok, phim, 0)
    U['m'][:, c] = np.where(ok, mm, 0); X.neutral(U, c, ok[:, None])
    return U


def conditions(T0, ds):
    spec = X.spectrum(T0); R = X.RDISC[ds]; n = X.nreal(T0); rows = np.arange(len(n))
    info = dict(softest_pt_median=float(np.median(T0['pt'][rows, n - 1])), softest_pt_mean=float(T0['pt'][rows, n - 1].mean()),
                soft_pool_mean=spec['mean'], N_mean=float(n.mean()))
    log(ds, info)
    yield 'base', T0, info
    yield 'e1', e_pt(T0, None, np.random.default_rng([SEED, 1])), {}
    for x in EPT: yield f'e1p{x:g}', e_pt(T0, x, np.random.default_rng([SEED, 1])), {}
    for k in (1, 20): yield f'k{k}', X.e_ksoft(T0, k, np.random.default_rng([SEED, 2, k])), {}
    for nm, k, d in (('sp1', 1, 0.01), ('sp5', 5, 0.01), ('sp20', 20, 0.01), ('sp5d002', 5, 0.002)):
        U, nv = splitk(T0, k, d, np.random.default_rng([SEED, 6, k, int(d * 1e4)]))
        yield nm, U, dict(n_split_mean=float(nv.mean()))
    U, W, nn = X.e_disc(T0, 20, R, spec, np.random.default_rng([SEED, 3, 20])); Vc = X.core(U, T0, W)
    yield 'c4_20', Vc, dict(n_injected_mean=float(nn.mean()))
    yield 'c4m_20', merge_added(Vc, T0, W), {}


def lrn_tables(ds):
    tds = 'qgpy' if ds.startswith('qg') else ds
    Tr = np.load(OUT / f'h18_{tds}_train.npz'); yt = Tr['y']; cap = X.NSEEN_CAP[ds]; Nt = np.minimum(np.diff(Tr['off']), cap).astype(int)
    tabs = {}
    for tn, sig, bkg in X.TASKS[ds]:
        rows = np.isin(yt, sig + bkg); g = np.isin(yt, sig)[rows]; Ns = Nt[rows]; M = cap + 1
        hs = np.bincount(Ns[g], minlength=M)[:M] + 0.5; hb = np.bincount(Ns[~g], minlength=M)[:M] + 0.5
        tabs[tn] = np.log(hs / hs.sum()) - np.log(hb / hb.sum())
    return tabs


def run(ds, T0, y, scorer, cap, extra):
    ae, ap = T0['ax_eta'], T0['ax_phi']; S = {}; NS = {}; info = dict(conditions={}, **extra)
    tabs = lrn_tables(ds); m0 = s0 = None
    for name, U, inf in conditions(T0, ds):
        t = time.time(); Uf, over = finalize(U, cap)
        ns = np.minimum((Uf['pt'] > 0).sum(1), X.NSEEN_CAP[ds]); NS[name] = ns
        sc = scorer(Uf, ae, ap)
        for mn, d in sc.items():
            for tn, s in d.items(): S[(mn, tn, name)] = np.asarray(s, np.float64)
        for tn, tab in tabs.items(): S[('LR_N', tn, name)] = tab[np.minimum(ns, X.NSEEN_CAP[ds])]
        m = X.mass_all(Uf); sp = Uf['pt'].sum(1)
        if name == 'base': m0, s0 = m, sp
        inf = dict(inf, dN_seen_mean=float((ns - NS['base']).mean()), dm_mean=float((m - m0).mean()), dsumpt_mean=float((sp - s0).mean()),
                   overflow_frac=float(over.mean()))
        info['conditions'][name] = inf
        tn0 = X.TASKS[ds][0][0]
        log(f'{ds} {name}: dN {inf["dN_seen_mean"]:+.2f} dm {inf["dm_mean"]:+.3f} dsumpt {inf["dsumpt_mean"]:+.3f} | ' +
            ' '.join(f'{mn} {np.mean(S[(mn, tn0, name)] - S[(mn, tn0, "base")]):+.4f}' for mn in list(sc) + ['LR_N']) + f' ({time.time() - t:.0f} s)')
    return S, NS, info


def stats(ds, y, S, NS, info, tests, nets, nulls):
    from scipy.stats import norm
    from interp.h18_softcount_cross import auc
    B = 1000; res = dict(ds=ds, info=info, tasks={})
    for tn, sig, bkg in X.TASKS[ds]:
        ix = np.flatnonzero(np.isin(y, sig + bkg)); sg = np.isin(y[ix], sig); n = len(ix)
        bidx = np.random.default_rng([SEED, 9]).integers(0, n, (B, n))
        R = dict(n=int(n), n_sig=int(sg.sum()), models={})
        for mn in sorted({k[0] for k in S}):
            s0 = S[(mn, tn, 'base')][ix]
            thr = float(np.quantile(s0[sg], 0.5)); c0 = s0 > thr
            M = dict(AUC=auc(s0, sg.astype(int)), edits={}, _boot={})
            for c in EDITS:
                d = S[(mn, tn, c)][ix] - s0; dN = (NS[c] - NS['base'])[ix].astype(float); bm = d[bidx].mean(1)
                sd = np.sqrt((d ** 2).sum()) / n; p = float(2 * norm.sf(abs(d.mean()) / sd)) if sd > 0 else 1.0
                c1 = S[(mn, tn, c)][ix] > thr
                M['edits'][c] = dict(mean=float(d.mean()), ci=X.ci(bm), dN=float(dN.mean()),
                                     per_particle=float(d.mean() / dN.mean()) if abs(dN.mean()) > 1e-9 else None,
                                     p_signflip=p, calls_changed_pct=float(100 * (c0 != c1).mean()), eff_sig_after=float(c1[sg].mean()))
                M['_boot'][c] = bm
                if mn in nets and mn not in nulls: tests.append([ds, tn, mn, c, p])
            # paired contrasts on the same jets
            for a, b in (('c4_20', 'c4m_20'), ('e1p1', 'e1'), ('sp5', 'sp5d002')):
                d = (S[(mn, tn, a)] - S[(mn, tn, b)])[ix]; bm = d[bidx].mean(1)
                dN = float((NS[a] - NS[b])[ix].mean())
                sd = np.sqrt((d ** 2).sum()) / n; p = float(2 * norm.sf(abs(d.mean()) / sd)) if sd > 0 else 1.0
                M['edits'][f'{a}-{b}'] = dict(mean=float(d.mean()), ci=X.ci(bm), dN=dN, per_particle=float(d.mean() / dN) if abs(dN) > 1e-9 else None, p_signflip=p)
                if mn in nets and mn not in nulls: tests.append([ds, tn, mn, f'{a}-{b}', p])
            # split slope per particle (sp1, sp5, sp20)
            xs = np.array([M['edits'][c]['dN'] for c in ('sp1', 'sp5', 'sp20')]); mus = np.array([M['edits'][c]['mean'] for c in ('sp1', 'sp5', 'sp20')])
            bmu = np.stack([M['_boot'][c] for c in ('sp1', 'sp5', 'sp20')], 1)
            M['split_slope'] = dict(slope=float(np.polyfit(xs, mus, 1)[0]), ci=X.ci(np.polyfit(xs, bmu.T, 1)[0]))
            R['models'][mn] = M
        res['tasks'][tn] = R
    return res


def strip(res):
    for T in res['tasks'].values():
        for M in T['models'].values(): M.pop('_boot', None)
    return res


# ----------------------------------------------------------------------------------------------------------------- datasets
def do_qg(dev):
    from interp import h5_qg_counting as H
    from interp import h14_qg_generator as G
    M = H.load_models(dev); G.add_keras_ef(M, 'efn'); names = [n for n in X.NETS['qgpy'] if n in M.m]
    L0, y, inf = G.load_sample('py', 4000, SEED); y = np.asarray(y).astype(int)
    T0 = finalize(X.qg_table(L0), X.CAP['qgpy'])[0]
    try:
        orig = set(json.loads((OUT / 'h18_qgpy_model.json').read_text())['idx']); ov = float(np.mean([i in orig for i in np.asarray(inf['idx']).tolist()]))
    except Exception as e:
        ov = repr(e)

    def scorer(Uf, ae, ap):
        Jt = dict(pt=Uf['pt'], eta=Uf['eta'], phi=Uf['phi'], pid=Uf['ext']['pid'].astype(np.int64), n=(Uf['pt'] > 0).sum(1))
        return {nm: {'q_vs_g': -M.logit(nm, Jt)} for nm in names}
    S, NS, info = run('qgpy', T0, y, scorer, X.CAP['qgpy'], dict(sample=f'QG Pythia test, 4000 per class, seed {SEED}', overlap_with_H18=ov))
    return y, S, NS, info, set(names), {'part_kin_perm'}


def jc_sample_excl(counts, seed, excl_counts, excl_seed):
    import uproot
    from interp.part_core import read_root, fof, cat, sub, CL
    rng = np.random.default_rng(seed); rx = np.random.default_rng(excl_seed); Js = []
    for c in CL:
        f = fof(c); n = uproot.open(f)['tree'].num_entries
        ex = np.sort(rx.choice(n, excl_counts[c], replace=False))
        pool = np.setdiff1d(np.arange(n), ex); idx = np.sort(rng.choice(pool, counts[c], replace=False)); parts = []
        for a in range(0, n, 20000):
            loc = idx[(idx >= a) & (idx < a + 20000)] - a
            if len(loc): parts.append(sub(read_root(f, start=a, stop=min(a + 20000, n)), loc))
        Js.append(cat(parts)); log(f'JetClass {c}: {counts[c]} random rows of {n}, H18 rows excluded')
    return cat(Js)


def do_jc(dev):
    from interp.part_core import ParTNetwork, CL, inputs, sub
    from interp.p1v_verify_part import fwd
    from interp.pn_net import load_particlenet
    from interp.h1_irc_panel import pn_logits, jc_table
    from interp.h13_pileup_panel import jc_maker
    from jetdistill.config import MODELS
    counts = {c: (2500 if c in ('QCD', 'Wqq', 'Tbqq') else 300) for c in CL}
    excl = {c: (3334 if c in ('QCD', 'Wqq', 'Tbqq') else 1000) for c in CL}
    Jall = jc_sample_excl(counts, SEED, excl, X.SEED); yall = Jall['label'].astype(int)
    nets = {}
    for n in ('full', 'kin'):
        net = ParTNetwork(n); nets[f'ParT_{n}'] = (lambda J, net=net, n=n: np.asarray(fwd(net, *inputs(J, n))[1], np.float64))
    net = ParTNetwork('full', untrained=True, seed=0); nets['ParT_full_redrawn'] = (lambda J, net=net: np.asarray(fwd(net, *inputs(J, 'full'))[1], np.float64))
    for n, dims in (('full', 17), ('kin', 7)):
        pn = load_particlenet(Path(MODELS).parent / 'particlenet' / f'ParticleNet_{n}.pt', dims, dev)
        nets[f'PN_{n}'] = (lambda J, pn=pn, n=n: pn_logits(pn, J, n))
    Tall = jc_table(Jall); mk = jc_maker(Jall); Jb = mk(finalize(Tall, 128)[0], Tall['ax_eta'], Tall['ax_phi']); acc10 = {}
    for mn, f in nets.items():
        c = f(Jb).argmax(1); acc10[mn] = float(np.mean([(c[yall == k] == k).mean() for k in range(len(CL))]))
    log('jc 10-class balanced accuracy', acc10)
    rows = np.flatnonzero(np.isin(yall, [0, 7, 8])); J0 = sub(Jall, rows); y = yall[rows]
    T0 = finalize(jc_table(J0), 128)[0]; make = jc_maker(J0)

    def scorer(Uf, ae, ap):
        J = make(Uf, ae, ap); out = {}
        for mn, f in nets.items():
            L = f(J); out[mn] = {'W_vs_QCD': L[:, 7] - L[:, 0], 'top_vs_QCD': L[:, 8] - L[:, 0]}
        return out
    S, NS, info = run('jc', T0, y, scorer, 128, dict(sample=f'JetClass first test file; QCD/Wqq/Tbqq 2500 each, others 300; seed {SEED}; H18 rows excluded',
                                                     balanced_acc_10class=acc10))
    return y, S, NS, info, set(nets), {'ParT_full_redrawn'}


def do_top(dev):
    from interp import data_top as dt
    from interp.h13_pileup_panel import TopModels, top_J, top_table
    Sm = dt.sample(3000, SEED); y = Sm['label'].astype(int); T0 = finalize(top_table(Sm['full']), 200)[0]
    try:
        orig = set(json.loads((OUT / 'h18_top_model.json').read_text())['idx']); ov = float(np.mean([i in orig for i in Sm['idx'].tolist()]))
    except Exception as e:
        ov = repr(e)
    Mo = TopModels(dev); order = np.random.default_rng(SEED + 7).permutation(len(y))

    def scorer(Uf, ae, ap):
        sc = Mo.score(top_J(Uf, ae, ap), order); return {k: {'t_vs_QCD': v} for k, v in sc.items()}
    S, NS, info = run('top', T0, y, scorer, 200, dict(sample=f'Top Landscape test, 3000 per class, seed {SEED}', overlap_with_H18=ov))
    return y, S, NS, info, set(Mo.names), {'PFN_permuted'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--ds', default='qg,jc,top'); a = ap.parse_args()
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; os.environ.setdefault('JETDISTILL_DEVICE', dev)
    tests = []; ALL = {}
    for ds in a.ds.split(','):
        y, S, NS, info, nets, nulls = dict(qg=do_qg, jc=do_jc, top=do_top)[ds](dev)
        key = 'qgpy' if ds == 'qg' else ds
        ALL[key] = stats(key, y, S, NS, info, tests, nets, nulls)
        (OUT / f'h18_verify_{key}.json').write_text(json.dumps(strip(ALL[key]), indent=1, default=float))
        log('wrote', OUT / f'h18_verify_{key}.json')
    h = holm([t[4] for t in tests])
    for t, hh in zip(tests, h): t.append(float(hh))
    (OUT / 'h18_verify.json').write_text(json.dumps(dict(n_tests=len(tests), n_holm_lt_05=int(sum(x < 0.05 for x in h)), tests=tests), indent=1))
    for ds, R in ALL.items():
        for tn, T in R['tasks'].items():
            for mn, M in T['models'].items():
                E = M['edits']
                log(f'{ds} {tn} {mn:18s} AUC {M["AUC"]:.4f} ' + ' '.join(f'{c} {E[c]["mean"]:+.3f}' for c in EDITS) +
                    f' | c4-c4m/pp {E["c4_20-c4m_20"]["per_particle"]} split slope {M["split_slope"]["slope"]:+.4f}')
    log('done')
