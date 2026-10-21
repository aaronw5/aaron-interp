"""H65 verification (loop D). Fresh-seed rerun of the H65 soft-contamination edit plus two controls H65 did not have.

Reuses interp.h65_pileup_dose (sample loaders, ghost draw, soft-drop core positions, slot cut, scorers) with new seeds.

Conditions (same ghosts as H65: pT uniform 0.5-1 GeV, Poisson count, one rho = 20 master set thinned to lower rho)
  base, pu2/5/10/20       ghosts uniform in the jet disc of radius R about the stored axis (as H65)
  c1_10                   one particle with the summed ghost pT, 0.002 from the axis (as H65)
  c2_10                   the rho = 10 ghosts all 0.002 from the axis (as H65; ~27 particles within 0.004 rad of each other)
  sh<f>_10                NEW shrink control: the rho = 10 ghosts at their pu10 positions scaled toward the axis by f
                          (Delta-eta, Delta-phi multiplied by f), f in {0.003, 0.03, 0.06, 0.125, 0.25, 0.5}; f = 1 is pu10.
                          Count, pT values and angular pattern fixed; added mass grows ~ f^2. If the network response is a
                          function of the count only, Delta s is flat in f. If c2's response comes from the stacked
                          (near-coincident) configuration, Delta s at f = 0.003 differs from Delta s at f = 0.06-0.25,
                          where the ghosts form a physical 0.05-0.2 rad soft cluster that still adds < 2 GeV of mass.
  in10, out10             ghosts inside / outside the soft-drop core (as H65)
  JetClass only, particle-ID arm (NEW): the pu10 / c2_10 / sh0.125_10 ghosts relabelled as photons (isPhoton) or as charged
                          hadrons (charge +-1 at random, d0 = dz = 0, d0err / dzerr = the jet's median over its real charged
                          particles), positions and pT unchanged ('_ph', '_ch' suffixes). Tests whether the full-input
                          models' response depends on the neutral-hadron label H65 used.
Metrics per model x signal class: mean Delta log-odds over signal + QCD rows with 95 % bootstrap CI (B = 1000); share =
  Delta s(cond) / Delta s(pu10) with bootstrap CI; signal efficiency eps_S at the unedited 50 % threshold; mean Delta m.
  Paired bootstrap p (normal approximation) for H0: Delta s(sh0.003) = Delta s(sh0.125) (stacking), and for H0: Delta s(_ph) =
  Delta s(neutral) (particle ID); Holm in the combine step.
Run
  ./part_job.sh d-h65-verify 'python -m interp.h65v_verify --ds jc && python -m interp.h65v_verify --ds top && python -m interp.h65v_verify --ds qg'
  ./job.sh d-h65-verify-hls 1 4 'python -m interp.h65v_verify --ds hls'
  ./job.sh d-h65-verify-sum 'python -m interp.h65v_verify --combine'
  -> $AI_RESULTS/physics/h65_verify_<ds>.json, h65_verify_summary.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h65_pileup_dose as H  # noqa: E402
from interp.h1_irc_panel import jet4, holm  # noqa: E402
from interp.h49_groomed_mass import softdrop, _m4  # noqa: E402

VSEEDS = dict(hls=8651, jc=8652, top=8653, qg=8654)
H.SEEDS.update(VSEEDS)
FS = (0.003, 0.03, 0.06, 0.125, 0.25, 0.5)
T0_ = time.time()


def log(*a):
    print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


def relabel(U, W, on, kind, T0):
    e = U['ext']; K = on.shape[1]; sl = slice(W, W + K)
    if kind == 'ph':
        e['isNeutralHadron'][:, sl] = 0.0; e['isPhoton'][:, sl] = on.astype(np.float64)
    elif kind == 'ch':
        real = T0['pt'] > 0; chg = real & (np.abs(T0['ext']['charge']) > 0)
        med = lambda a: np.array([np.median(np.abs(a[j][chg[j]])) if chg[j].any() else 0.0 for j in range(len(a))])
        d0e, dze = med(T0['ext']['d0err']), med(T0['ext']['dzerr'])
        rng = np.random.default_rng([VSEEDS['jc'], 5]); q = np.where(rng.random(on.shape) < 0.5, -1.0, 1.0)
        e['isNeutralHadron'][:, sl] = 0.0; e['isChargedHadron'][:, sl] = on.astype(np.float64)
        e['charge'][:, sl] = np.where(on, q, 0.0)
        e['d0err'][:, sl] = np.where(on, d0e[:, None], 0.0); e['dzerr'][:, sl] = np.where(on, dze[:, None], 0.0)
    return U


def conditions(T0, G, ds):
    J = len(T0['pt']); W = T0['pt'].shape[1]
    yield 'base', T0
    for r in H.RHOS:
        m = G['on'] & (G['v'] < r / H.RHOMAX)
        yield f'pu{r}', H.with_ghosts(T0, G['de'], G['dp'], G['q'], m)
    m = G['on'] & (G['v'] < 10 / H.RHOMAX); Q = (G['q'] * m).sum(1); a = G['thax'][:, :1]
    yield 'c1_10', H.with_ghosts(T0, H.R_AX * np.cos(a), H.R_AX * np.sin(a), Q[:, None], (Q > 0)[:, None])
    c2 = (H.R_AX * np.cos(G['thax']), H.R_AX * np.sin(G['thax']))
    yield 'c2_10', H.with_ghosts(T0, *c2, G['q'], m)
    for f in FS:
        yield f'sh{f}_10', H.with_ghosts(T0, f * G['de'], f * G['dp'], G['q'], m)
    yield 'in10', H.with_ghosts(T0, G['de_in'], G['dp_in'], G['q'], m)
    yield 'out10', H.with_ghosts(T0, G['de_out'], G['dp_out'], G['q'], m)
    if ds == 'jc':
        for kind in ('ph', 'ch'):
            yield f'pu10_{kind}', relabel(H.with_ghosts(T0, G['de'], G['dp'], G['q'], m), W, m, kind, T0)
            yield f'c2_10_{kind}', relabel(H.with_ghosts(T0, *c2, G['q'], m), W, m, kind, T0)
            yield f'sh0.125_10_{kind}', relabel(H.with_ghosts(T0, 0.125 * G['de'], 0.125 * G['dp'], G['q'], m), W, m, kind, T0)


def ci(v):
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def run(ds, smoke):
    H.OUT.mkdir(parents=True, exist_ok=True)
    T0, y, info, J0 = H.load(ds, smoke); T0 = H.finalize(T0, H.CAP[ds])[0]; W = T0['pt'].shape[1]
    log(f'{ds}: {len(y)} jets, seed {H.SEEDS[ds]}, labels {np.bincount(y).tolist()}')
    sd0 = softdrop(T0); G = H.ghosts(T0, ds, H.SEEDS[ds]); H.core_positions(G, T0, sd0, ds, H.SEEDS[ds])
    scorer, _ = H.make_scorer(ds, J0); ae, ap = T0['ax_eta'], T0['ax_phi']
    S, M, MSD, N = {}, {}, {}, {}
    for name, U in conditions(T0, G, ds):
        t = time.time(); Uf, gk, gpt = H.fin(U, W, H.CAP[ds]); N[name] = H.nreal(Uf)
        M[name] = _m4(jet4(Uf))
        if name == 'base' or name.startswith('pu') and '_' not in name: MSD[name] = softdrop(Uf)['msd']
        L = scorer(Uf, ae, ap)
        for mn, l in L.items():
            for cls, s in H.class_scores(np.asarray(l, np.float64), ds).items(): S[(mn, cls, name)] = s
            if name == 'base' and np.ndim(l) == 2:
                a = np.asarray(l).argmax(1)
                log(f'{ds} unedited {mn}: per-class acc ' + ' '.join(f'{k}:{(a[y == k] == k).mean():.3f}' for k in sorted(set(y.tolist())))
                    + (f' 5-class {(a == y).mean():.4f}' if ds == 'hls' else ''))
        log(f'{ds} {name}: dN {np.mean(N[name] - N["base"]):+.1f}, dm {np.mean(M[name] - M["base"]):+.2f} GeV ({time.time() - t:.0f} s)')
    models = sorted({k[0] for k in S}); conds = list(M)
    out = dict(ds=ds, seed=H.SEEDS[ds], n=int(len(y)), sample=info.get('sample'), c3_valid=float(G['c3ok'].mean()), classes={})
    for cls, k in H.SIG[ds].items():
        rows = np.isin(y, [k] + H.QCD[ds]); sig = (y == k)[rows]; nr = int(rows.sum()); c3 = G['c3ok'][rows]
        idx = np.random.default_rng([H.SEEDS[ds], 99, k]).integers(0, nr, (1000, nr))
        C = dict(n_rows=nr, n_sig=int(sig.sum()), observables={}, models={}, windows={})
        for c in conds:
            sel = c3 if c in ('in10', 'out10') else np.ones(nr, bool)
            C['observables'][c] = dict(dm=float((M[c] - M['base'])[rows][sel].mean()), dN=float((N[c] - N['base'])[rows][sel].mean()),
                                       dm_sig=float((M[c] - M['base'])[rows][sel & sig].mean()))
        # reference windows on the unedited signal (shortest 50 % interval; QG: m < median quark m)
        for ob, X in (('m', M), ('msd', MSD)):
            Xb = X['base'][rows]
            if ds == 'qg':
                t = float(np.median(Xb[sig])); call = lambda v: v < t
            else:
                lo, hi = H.shortest50(Xb[sig]); call = lambda v, lo=lo, hi=hi: (v >= lo) & (v <= hi)
            C['windows'][f'win_{ob}'] = {c: dict(effS=float(call(X[c][rows])[sig].mean()), effB=float(call(X[c][rows])[~sig].mean())) for c in X}
        for mn in models:
            sc = {c: S[(mn, cls, c)][rows] for c in conds}; thr = float(np.median(sc['base'][sig])); R = {}
            acc = dict(auc=H.auc(sc['base'], sig.astype(int)))
            dsb = {}
            for c in conds:
                if c == 'base': continue
                sel = c3 if c in ('in10', 'out10') else np.ones(nr, bool); d = sc[c] - sc['base']
                db = (d * sel)[idx].sum(1) / np.maximum(sel[idx].sum(1), 1); dsb[c] = db
                R[c] = dict(ds=float(d[sel].mean()), ds_ci=ci(db), effS=float((sc[c] > thr)[sig & sel].mean()),
                            effB=float((sc[c] > thr)[~sig & sel].mean()))
            for c in R:
                if c in ('in10', 'out10') or c.startswith('pu') and '_' not in c: continue
                ref = 'pu10' if not c.endswith(('_ph', '_ch')) else ('pu10_' + c[-2:])
                R[c]['share'] = R[c]['ds'] / R[ref]['ds'] if abs(R[ref]['ds']) > 1e-9 else None
                sb = dsb[c] / np.where(np.abs(dsb[ref]) > 1e-9, dsb[ref], np.nan); R[c]['share_ci'] = ci(sb)
            R['r_ds'] = R['out10']['ds'] / R['in10']['ds'] if abs(R['in10']['ds']) > 1e-9 else None
            R['r_ds_ci'] = ci(dsb['out10'] / np.where(np.abs(dsb['in10']) > 1e-9, dsb['in10'], np.nan))
            tests = {}
            pairs = [('sh0.003_10', 'sh0.125_10'), ('c2_10', 'sh0.125_10'), ('sh0.125_10', 'pu10')]
            if ds == 'jc':
                pairs += [(f'{b}_{kd}', b) for kd in ('ph', 'ch') for b in ('pu10', 'c2_10', 'sh0.125_10')]
            for a_, b_ in pairs:
                est = float((sc[a_] - sc[b_]).mean()); bb = dsb[a_] - dsb[b_]
                tests[f'{a_} - {b_}'] = dict(diff=est, ci=ci(bb), p=H.p_z(est, bb))
            C['models'][mn] = dict(unedited=acc, threshold=thr, cond=R, tests=tests)
        out['classes'][cls] = C
        log(f'{ds} {cls}: ' + '; '.join(f'{mn} pu10 {C["models"][mn]["cond"]["pu10"]["ds"]:+.2f} c2 {C["models"][mn]["cond"]["c2_10"]["ds"]:+.2f} '
                                         f'sh ' + '/'.join(f'{C["models"][mn]["cond"][f"sh{f}_10"]["ds"]:+.2f}' for f in FS) for mn in models))
    s = H.sfx(smoke)
    (H.OUT / f'h65_verify_{ds}{s}.json').write_text(json.dumps(out, indent=1, default=float)); log(f'wrote h65_verify_{ds}{s}.json')


def combine(smoke):
    s = H.sfx(smoke); res = dict(datasets={}); fam = []
    for ds in ('hls', 'jc', 'top', 'qg'):
        f = H.OUT / f'h65_verify_{ds}{s}.json'
        if not f.exists(): log('missing', f.name); continue
        D = json.loads(f.read_text()); res['datasets'][ds] = D
        for cls, C in D['classes'].items():
            for mn, R in C['models'].items():
                if mn in H.NULLS: continue
                for t, v in R['tests'].items(): fam.append((f'{ds}|{cls}|{mn}|{t}', v['p']))
    hp = holm([p for _, p in fam]); res['holm'] = {k: dict(p=p, p_holm=float(h)) for (k, p), h in zip(fam, hp)}
    log(f'Holm: {len(fam)} tests, {sum(h < 0.05 for h in hp)} with p_holm < 0.05')
    (H.OUT / f'h65_verify_summary{s}.json').write_text(json.dumps(res, indent=1, default=float)); log('wrote summary')


if __name__ == '__main__':
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--ds', choices=['hls', 'jc', 'top', 'qg'])
    ap_.add_argument('--combine', action='store_true')
    ap_.add_argument('--smoke', type=int, default=0)
    a = ap_.parse_args()
    if a.combine: combine(a.smoke)
    else: run(a.ds, a.smoke)
