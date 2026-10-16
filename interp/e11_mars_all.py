"""E11 (JEDI-linear 64 p, 3-feature; claims A5, B; user idea 2026-10-16): the ENTIRE network as hinge formulas.

Every live unit of P, g, S, h4, h5, z (live = changes >= 0.1 % of calls when set to its mean, E1 costs below) is fitted
with jetdistill.mars.fit_neuron (MARS-style forward selection: Q, max(0, Q - t), max(0, t - Q), products of two; knots at
the 5-95 % quantiles; here at most MAX_TERMS terms, stop after 5 steps without validation gain), on two input sets:
  (a) physics: the Q observable library of the test file (383) + hand-built energy-flow sums over pT-rank windows (H:...),
  (b) compositional, following the network's own graph: P <- physics; g <- P^ (the fitted P formulas); S <- g^ + physics
      (S reads the particles directly and the context g); h4 <- S^; h5 <- h4^; z <- h5^.  Each stage is fitted on the
      CHAIN values of the previous stage (formulas evaluated recursively from physics), so (b) is one nested if-statement
      program from the particles to the 16 neurons.
Target per unit: the post-ReLU value for P, g, S, h4, h5; the pre-ReLU value for z, clipped below as mars.py does.
R² is mars.py's: of max(0, formula) against max(0, unit).
Jets: all from the TEST file (the earlier MARS work fitted on the TRAIN archive): sample(per_class=6000, seed=0) ->
fixed permutation -> fit (first 2/3) / validation (last 1/3); held-out test sample(per_class=2000, seed=11) minus any jet
also in the fitting sample.
Evaluation (mode eval): per-unit held-out R²; stage substitution (stage k's live units replaced per jet by the formula
values, the real network finishes; jedi_core.Jedi.run_jetwise) for (a) and (b); cumulative substitution P..k; the last
row of (b) is the all-if-statement model; controls: the untrained network's per-unit R² (mode control untrained), a
shuffled-target fit of one stage (mode control shuffle); the per-unit "what it looks at" table; the feature-evolution
figure results/jedi/e11_evolution.png.

Usage (python -m interp.e11_mars_all ...; AI_RESULTS/jedi receives the files):
  fit a <stage>                 -> e11_fit_a_<stage>.json          (stage in P g S h4 h5 z)
  fit b                         -> e11_fit_a_P.json, e11_fit_b_<stage>.json for g S h4 h5 z (sequential chain)
  control untrained <st,st,..>  -> e11_control_untrained_<stages>.json
  control shuffle <stage>       -> e11_control_shuffle_<stage>.json
  eval                          -> e11_mars_all.json, e11_evolution.png
Options: --per-class 6000 --test-per-class 2000 --max-terms 30 --n-select 10000 --prefix e11
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.jedi_core import Jedi, sample
from jetdistill import mars

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
STG = ('P', 'g', 'S', 'h4', 'h5', 'z')
RELU = dict(P=True, g=True, S=True, h4=True, h5=True, z=False)
PREV = dict(P=None, g='P', S='g', h4='S', h5='h4', z='h5')
CL = ['g', 'q', 'W', 'Z', 't']
T0 = time.time()

# live units and their mean-ablation cost (% of calls changed; E1, results/jedi/e1_e3.json, 10,000 test jets)
LIVE = {
    'P': {3: 1.18, 6: 2.34, 15: 8.12, 29: 3.14, 30: 1.57, 32: 3.54, 36: 2.01, 40: 3.15, 43: 2.2, 46: 1.88, 56: 3.03, 57: 0.77},
    'g': {4: 4.28, 5: 1.78, 29: 1.07, 32: 0.96, 36: 0.64, 39: 2.09, 44: 0.78, 47: 0.6, 49: 10.43, 55: 2.41, 63: 1.37},
    'S': {0: 12.25, 1: 20.61, 13: 7.17, 17: 5.52, 19: 11.52, 20: 1.51, 29: 40.78, 30: 8.42, 31: 3.51, 40: 3.79, 43: 8.07,
          50: 12.02, 51: 2.76, 52: 2.27, 56: 5.1, 58: 1.97, 60: 7.21, 62: 2.24, 63: 2.29},
    'h4': {1: 2.68, 2: 2.06, 3: 3.27, 4: 1.86, 6: 4.16, 7: 0.98, 8: 0.19, 9: 1.1, 10: 1.22, 11: 1.1, 12: 1.08, 13: 2.62, 14: 1.8,
           16: 3.14, 17: 0.71, 18: 2.05, 21: 0.58, 22: 0.64, 24: 0.98, 25: 1.04, 26: 1.07, 27: 2.03, 28: 2.11, 29: 0.33, 30: 3.68,
           32: 1.38, 34: 2.77, 35: 1.23, 36: 1.06, 37: 0.94, 39: 4.58, 40: 1.37, 41: 7.0, 42: 1.4, 44: 1.75, 45: 1.85, 46: 1.8,
           47: 4.44, 48: 1.39, 49: 19.69, 50: 1.3, 51: 4.3, 52: 0.34, 53: 4.01, 54: 1.37, 55: 3.16, 56: 1.79, 57: 2.53, 58: 0.86,
           59: 0.83, 60: 2.78, 61: 0.52, 62: 1.2, 63: 1.14},
    'h5': {0: 2.57, 1: 3.31, 2: 4.43, 3: 5.29, 4: 0.87, 5: 1.2, 6: 1.21, 7: 0.7, 8: 6.39, 9: 1.48, 10: 1.6, 11: 6.29, 12: 1.23,
           13: 2.03, 14: 1.98, 15: 2.53, 16: 1.64, 17: 3.05, 18: 1.65, 19: 2.76, 20: 5.02, 21: 1.62, 22: 1.89, 23: 1.17, 24: 4.25,
           25: 5.87, 26: 16.12, 27: 1.24, 28: 1.19, 29: 2.77, 30: 1.1, 31: 2.65},
    'z': {0: 15.1, 1: 21.63, 2: 0.37, 3: 1.06, 4: 3.01, 5: 2.75, 6: 2.39, 7: 3.42, 8: 4.3, 9: 4.84, 10: 6.88, 11: 2.47, 12: 2.38,
          13: 4.06, 14: 0.67, 15: 2.01}}
UNITS = {s: sorted(LIVE[s]) for s in STG}
# pT-rank windows of the hand-built quantities (E3m / E5: the 14 hardest, all but the leading, ranks 13-63, ...)
WINDOWS = {'all': (0, 64), 'top14': (0, 14), 'r1-63': (1, 64), 'r13-63': (13, 64), 'top39': (0, 39), 'top26': (0, 26),
           'top3': (0, 3), 'top11': (0, 11), 'r8-63': (8, 64)}
FAMILIES = ['ΣpT / pT scale', 'multiplicity', 'radial moments (r, r²)', 'mass family', 'N-subjettiness, ECFs', 'subjet / prong', 'pT shares, other']


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def uid(stage, j):
    return f'{stage}:{j}'


# ------------------------------------------------------------------ inputs
def handbuilt(x, halfplanes=False):
    """energy-flow sums over pT-rank windows (pT in GeV, r = ΔR to the axis in rad, N = counts) -> {id: (J,)}
    halfplanes (supplement, --halfplanes): E2's first-round statements are half-planes through the axis, so also the
    one-sided angular sums Σ (n·r⃗)₊ and counts N(n·r⃗ > 0) for 8 directions n (every 45°) in 4 rank windows"""
    pt, eta, phi = (np.asarray(x[..., k], np.float64) for k in range(3)); real = pt > 0; r = np.hypot(eta, phi)
    H = {}
    if halfplanes:
        for name in ('all', 'top14', 'r1-63', 'r13-63'):
            lo, hi = WINDOWS[name]; w = real.copy(); w[:, :lo] = False; w[:, hi:] = False
            for deg in range(0, 360, 45):
                p = np.cos(np.radians(deg)) * eta + np.sin(np.radians(deg)) * phi
                H[f'H:Σ(n·r)₊[{deg}°][{name}]'] = (np.maximum(p, 0) * w).sum(1); H[f'H:N(n·r>0)[{deg}°][{name}]'] = (w & (p > 0)).sum(1)
    for name, (lo, hi) in WINDOWS.items():
        w = real.copy(); w[:, :lo] = False; w[:, hi:] = False; s = f'[{name}]'
        if name != 'all':
            H[f'H:N{s}'] = w.sum(1); H[f'H:ΣpT{s}'] = (pt * w).sum(1)
        H[f'H:ΣpT·r²{s}'] = (pt * r ** 2 * w).sum(1); H[f'H:Σr²{s}'] = (r ** 2 * w).sum(1); H[f'H:Σr{s}'] = (r * w).sum(1)
        H[f'H:ΣpT·r{s}'] = (pt * r * w).sum(1); H[f'H:Σ(r−0.2)₊{s}'] = (np.maximum(r - 0.2, 0) * w).sum(1)
        H[f'H:N(pT>1){s}'] = (w & (pt > 1)).sum(1); H[f'H:N(pT<2){s}'] = (w & (pt < 2)).sum(1)
        H[f'H:Σr[r<0.165]{s}'] = (r * (w & (r < 0.165))).sum(1); H[f'H:N(pT<2,r>0.1){s}'] = (w & (pt < 2) & (r > 0.1)).sum(1)
    M0 = pt.sum(1); M2 = (pt * r ** 2).sum(1)
    H['H:|M1|'] = np.hypot((pt * eta).sum(1), (pt * phi).sum(1)); H['H:√(M0·M2)'] = np.sqrt(M0 * M2)
    return {k: v.astype(np.float64) for k, v in H.items()}


def family(key):
    """observable id -> index into FAMILIES"""
    import re
    k = key[2:] if key.startswith('H:') else key
    if key.startswith(('P:', 'g:', 'S:', 'h4:', 'h5:')): return None
    if key.startswith('H:'):                      # ΣpT[window] | N..., | √(M0·M2) | the rest are radial sums (ΣpT·r², Σr², Σr, |M1|, ...)
        if k.startswith('ΣpT['): return 0
        if k.startswith('N'): return 1
        if k.startswith('√(M0'): return 3
        return 2
    if k in ('sum_pt', 'log_sum_pt') or re.fullmatch(r'sum_pt_top\d+|pt_\d+|soft\d+_pt', k): return 0
    if k in ('n_particles', 'sd_nremoved') or re.fullmatch(r'n_pt_above_\d+|n_dr_.*|n_real_top\d+|n_for_\d+pct', k): return 1
    if 'mass' in k or k in ('m01', 'm012') or k.startswith(('mratio_', 'sj3_pair')): return 3
    if re.fullmatch(r'tau\d+|tau\d+_b2|e[234]|e2_sq|[CDNM][23](_b2)?', k): return 4
    if k in ('sj2_zsoft', 'sj2_dr', 'sd_zg', 'sd_rg', 'dr01', 'dr02', 'dr12', 'dr_min_012', 'dr_max_012', 'pt1_over_pt0', 'pt2_over_pt0',
             'pt_balance01', 'pt1_dr01') or re.fullmatch(r'sj3_(z\d|dr.*)|dr[01]_\d+|ptdr0_\d+|soft\d+_dr0', k): return 5
    if k in ('girth', 'girth2', 'mean_eta', 'mean_eta2', 'mean_phi', 'mean_phi2', 'width', 'lam1', 'lam2', 'max_dr', 'LHA', 'centroid_offset') \
            or re.fullmatch(r'girth2_top\d+|dr_\d+|abseta_\d+|absphi_\d+|eta_\d+|phi_\d+|zdr_\d+|soft\d+_(dr|abseta|absphi)|psi_.*|z_dr_.*', k): return 2
    return 6


# ------------------------------------------------------------------ data
def indices(per_class, seed):
    """the test-file row indices sample(per_class, seed) draws (same rule as jedi_core.sample)"""
    from jetdistill.config import RESULTS
    y = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y']); rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in range(5)]))


class Data:
    """fit / dev / test jets, their physics inputs and the network's stage values (all splits from the TEST file)"""

    def __init__(self, a, untrained=False):
        self.a = a
        x, y, Q = sample(a.per_class, 0); xt, yt, Qt = sample(a.test_per_class, 11)
        i0, i1 = indices(a.per_class, 0), indices(a.test_per_class, 11); keep = ~np.isin(i1, i0)
        self.overlap = int((~keep).sum()); xt, yt = xt[keep], yt[keep]; Qt = {k: v[keep] for k, v in Qt.items()}
        J = len(x); perm = np.random.default_rng(0).permutation(J); nf = 2 * J // 3
        self.ix = dict(fit=perm[:nf], dev=perm[nf:])
        phys = {**{k: np.asarray(v, np.float64) for k, v in Q.items()}, **handbuilt(x, a.halfplanes)}
        physt = {**{k: np.asarray(v, np.float64) for k, v in Qt.items()}, **handbuilt(xt, a.halfplanes)}
        self.keys = [k for k in phys if np.isfinite(phys[k]).all() and np.isfinite(physt[k]).all() and np.ptp(phys[k][self.ix['fit']]) > 0]
        self.phys = {s: {k: phys[k][ix] for k in self.keys} for s, ix in self.ix.items()}; self.phys['test'] = {k: physt[k] for k in self.keys}
        self.y = {s: y[ix] for s, ix in self.ix.items()}; self.y['test'] = yt
        self.x = {s: x[ix] for s, ix in self.ix.items()}; self.x['test'] = xt
        self.net = Jedi(64, untrained=untrained)
        R = self.net.run(x, keep=STG + ('logits',)); Rt = self.net.run(xt, keep=STG + ('logits',))
        self.V = {s: {k: R[k][ix] for k in R} for s, ix in self.ix.items()}; self.V['test'] = Rt
        self.n = {s: len(self.y[s]) for s in ('fit', 'dev', 'test')}
        log(f'jets fit {self.n["fit"]} dev {self.n["dev"]} test {self.n["test"]} (overlap with the fitting sample removed: {self.overlap}); '
            f'physics inputs {len(self.keys)} ({sum(k.startswith("H:") for k in self.keys)} hand-built); '
            f'network acc fit {(R["logits"].argmax(1) == y).mean():.4f} test {(Rt["logits"].argmax(1) == yt).mean():.4f}; untrained {untrained}')


# ------------------------------------------------------------------ fitting
def knots_of(Qfit, keys):
    return {k: mars.thresholds(Qfit[k], False, False) for k in keys}


def evaluate(units, Qd, relu):
    """formula values (J, n_units) on inputs Qd; max(0, .) for ReLU stages"""
    J = len(next(iter(Qd.values()))); out = np.zeros((J, len(units)))
    for i, u in enumerate(units):
        v = np.full(J, float(u['intercept']))
        for t in u['terms']: v += t['coef'] * mars.term_value(t, Qd)
        out[:, i] = np.maximum(v, 0) if relu else v
    return out


def r2_of(pred, target, relu=True):
    h = np.maximum(target, 0) if relu else target; p = np.maximum(pred, 0) if relu else pred
    return float(1 - ((h - p) ** 2).mean() / max(h.var(), 1e-12))


def fit_stage(stage, Z, Qsplit, keys, label, units=None, max_terms=30):
    """Z = {split: (J, 64/32/16) stage values}; Qsplit = {split: {key: values}}. Returns the per-unit formula list."""
    mars.MAX_TERMS = max_terms
    knots = knots_of(Qsplit['fit'], keys); sel = np.random.default_rng(1).choice(len(Z['fit']), min(ARGS.n_select, len(Z['fit'])), replace=False)
    out = []
    for j in (units if units is not None else UNITS[stage]):
        t0 = time.time(); z = {s: Z[s][:, j] for s in ('fit', 'dev')}; zt = Z['test'][:, j]
        if np.ptp(z['fit']) < 1e-9:
            out.append(dict(unit=int(j), intercept=float(z['fit'].mean()), terms=[], constant=True, n_terms=0, dev_r2=0.0, test_r2=0.0, test_r2_raw=0.0, top3=[]))
            log(f'{label} {uid(stage, j)}: constant'); continue
        terms, coef, path = mars.fit_neuron(z, Qsplit, keys, knots, sel)
        Bt = np.stack([np.ones(len(zt))] + [mars.term_value(t, Qsplit['test']) for t in terms], 1); pt_ = Bt @ coef
        r2 = r2_of(pt_, zt, True); r2raw = r2_of(pt_, zt, False)
        ht = np.maximum(zt, 0); r2h = lambda p: float(1 - ((ht - np.maximum(p, 0)) ** 2).mean() / max(ht.var(), 1e-12))
        imp = [round(r2 - r2h(pt_ - c * Bt[:, i + 1]), 5) for i, c in enumerate(coef[1:])]
        spread = [float(mars.term_value(t, Qsplit['fit']).std()) for t in terms]
        w = np.array([abs(c) * s for c, s in zip(coef[1:], spread)]); ws = w / max(w.sum(), 1e-12)
        T = [dict(t, coef=float(c), spread=s, weight_share=float(sh), importance=m) for t, c, s, sh, m in zip(terms, coef[1:], spread, ws, imp)]
        top3 = [T[i] for i in np.argsort(-w)[:3]]
        out.append(dict(unit=int(j), intercept=float(coef[0]), terms=T, n_terms=len(T), dev_r2=float(path[len(T) - 1]) if T else 0.0, dev_r2_path=path,
                        test_r2=r2, test_r2_raw=r2raw, top3=[dict(q=t['q'], kind=t['kind'], t=t.get('t'), q2=t.get('q2'), kind2=t.get('kind2'), t2=t.get('t2'),
                                                              coef=t['coef'], weight_share=t['weight_share']) for t in top3], seconds=round(time.time() - t0, 1)))
        log(f'{label} {uid(stage, j)}: {len(T)} terms, dev R² {out[-1]["dev_r2"]:.4f}, test R² {r2:.4f}; top: ' +
            '; '.join(f"{t['q']} {t['kind']} {'' if t.get('t') is None else round(t['t'], 4)}" + (f" × {t['q2']} {t['kind2']} {round(t['t2'], 4)}" if t.get('q2') else '') + f" ({t['weight_share']:.2f})" for t in top3) +
            f'; {time.time() - t0:.0f} s')
    return out


def chain_inputs(stage, prev_vals, phys):
    """inputs of stage's (b) formulas: the previous stage's formula values (named 'P:15', ...), plus physics for S"""
    if stage == 'P': return dict(phys)
    d = {uid(PREV[stage], j): prev_vals[:, i] for i, j in enumerate(UNITS[PREV[stage]])}
    if stage == 'S': d.update(phys)
    return d


def save(name, obj):
    p = OUT / f'{ARGS.prefix}_{name}.json'; p.write_text(json.dumps(obj, indent=1)); log(f'wrote {p}')


def loadf(name):
    return json.loads((OUT / f'{ARGS.prefix}_{name}.json').read_text())


def meta(D):
    return dict(jets=D.n, overlap_removed=D.overlap, per_class=D.a.per_class, test_per_class=D.a.test_per_class, seeds=dict(fit=0, test=11), max_terms=ARGS.max_terms,
                n_select=min(ARGS.n_select, D.n['fit']), source='test file (full_test.npz) for every split', n_physics_inputs=len(D.keys), halfplanes=ARGS.halfplanes)


# ------------------------------------------------------------------ modes
def mode_fit_a(D, stage):
    units = fit_stage(stage, {s: D.V[s][stage] for s in D.V}, D.phys, D.keys, f'(a) {stage}', max_terms=ARGS.max_terms)
    save(f'fit_a_{stage}', dict(stage=stage, inputs='physics', keys=D.keys, units=units, **meta(D)))


def mode_fit_b(D):
    prev = None
    for stage in STG:
        Qs = {s: chain_inputs(stage, None if prev is None else prev[s], D.phys[s]) for s in ('fit', 'dev', 'test')}
        keys = [k for k in Qs['fit'] if np.ptp(Qs['fit'][k]) > 0 and all(np.isfinite(Qs[s][k]).all() for s in Qs)]
        units = fit_stage(stage, {s: D.V[s][stage] for s in D.V}, Qs, keys, f'(b) {stage}', max_terms=ARGS.max_terms)
        save('fit_a_P' if stage == 'P' else f'fit_b_{stage}', dict(stage=stage, inputs='physics' if stage == 'P' else f'chain values of {PREV[stage]}' + (' + physics' if stage == 'S' else ''),
                                                               keys=keys, units=units, **meta(D)))
        prev = {s: evaluate(units, Qs[s], RELU[stage]) for s in Qs}
        log(f'(b) {stage} done: chain values on test, mean R² {np.mean([u["test_r2"] for u in units]):.4f}')


def mode_control_untrained(D, stages):
    res = dict(control='untrained network (jedi_core.Jedi(untrained=True, seed=0))', stages={}, **meta(D))
    for stage in stages:
        Zf = D.V['fit'][stage]; sd = Zf.std(0); n = len(UNITS[stage])
        units = [int(j) for j in np.argsort(-sd)[:n] if sd[j] > 1e-9]
        res['stages'][stage] = dict(units_by_spread=units, spread=[float(sd[j]) for j in units],
                                    units=fit_stage(stage, {s: D.V[s][stage] for s in D.V}, D.phys, D.keys, f'(untrained a) {stage}', units=units, max_terms=ARGS.max_terms))
    save(f'control_untrained_{"_".join(stages)}', res)


def mode_control_shuffle(D, stage):
    rng = np.random.default_rng(2); Z = {s: D.V[s][stage][rng.permutation(len(D.y[s]))] for s in D.V}
    units = fit_stage(stage, Z, D.phys, D.keys, f'(shuffled {stage})', max_terms=ARGS.max_terms)
    save(f'control_shuffle_{stage}', dict(control=f'targets of stage {stage} permuted across jets in every split (rng 2)', stage=stage, units=units, **meta(D)))


def mode_eval(D):
    net, T = D.net, 'test'; Nt = D.V[T]; y = D.y[T]; call = Nt['logits'].argmax(1); J = len(y)
    acc_net = float((call == y).mean()); res = dict(network_acc_test=acc_net, network_per_class_acc=[float((call[y == k] == k).mean()) for k in range(5)],
                                                 **meta(D), units=UNITS, costs={s: LIVE[s] for s in STG})
    fa = {s: loadf(f'fit_a_{s}') for s in STG}; fb = {'P': fa['P'], **{s: loadf(f'fit_b_{s}') for s in STG[1:]}}
    # formula values on the test jets: (a) from physics; (b) chain (recursive) and (b) single (true previous-stage network values)
    Fa = {s: evaluate(fa[s]['units'], D.phys[T], RELU[s]) for s in STG}
    Vb, Fb1 = {}, {}
    for s in STG:
        Vb[s] = evaluate(fb[s]['units'], chain_inputs(s, None if s == 'P' else Vb[PREV[s]], D.phys[T]), RELU[s])
        Fb1[s] = Fa['P'] if s == 'P' else evaluate(fb[s]['units'], chain_inputs(s, Nt[PREV[s]][:, UNITS[PREV[s]]], D.phys[T]), RELU[s])
    # per-unit R² tables
    r2 = {}
    for s in STG:
        tgt = Nt[s][:, UNITS[s]]; cost = np.array([LIVE[s][j] for j in UNITS[s]]); cw = cost / cost.sum()
        rows = {}
        for name, F in (('a', Fa[s]), ('b_chain', Vb[s]), ('b_single', Fb1[s])):
            v = np.array([r2_of(F[:, i], tgt[:, i], True) for i in range(len(UNITS[s]))])
            rows[name] = dict(per_unit=dict(zip(map(str, UNITS[s]), map(float, v))), median=float(np.median(v)), min=float(v.min()), mean=float(v.mean()),
                              cost_weighted_mean=float((cw * v).sum()), n_ge_0p9=int((v >= .9).sum()), n_ge_0p5=int((v >= .5).sum()))
        rows['n_terms_a'] = [u['n_terms'] for u in fa[s]['units']]; rows['n_terms_b'] = [u['n_terms'] for u in fb[s]['units']]
        r2[s] = rows
        log(f'R² {s}: (a) median {rows["a"]["median"]:.3f} min {rows["a"]["min"]:.3f} cost-w {rows["a"]["cost_weighted_mean"]:.3f} | '
            f'(b chain) {rows["b_chain"]["median"]:.3f} / {rows["b_chain"]["min"]:.3f} / {rows["b_chain"]["cost_weighted_mean"]:.3f} | '
            f'(b single) {rows["b_single"]["median"]:.3f} / {rows["b_single"]["min"]:.3f} / {rows["b_single"]["cost_weighted_mean"]:.3f}; terms (a) {sum(rows["n_terms_a"])} (b) {sum(rows["n_terms_b"])}')
    res['r2'] = r2
    # substitution
    masks = {s: np.isin(np.arange(Nt[s].shape[1]), UNITS[s]).astype(np.float32) for s in STG}

    def full(s, F):
        v = Nt[s].copy(); v[:, UNITS[s]] = F; return v

    def sub(O):
        lo = net.run_jetwise(D.x[T], {s: (masks[s], full(s, F)) for s, F in O.items()})['logits']; c = lo.argmax(1)
        return dict(same_call=float((c == call).mean()), acc=float((c == y).mean()), per_class_acc=[float((c[y == k] == k).mean()) for k in range(5)],
                    confusion=np.array([[int(((y == a) & (c == b)).sum()) for b in range(5)] for a in range(5)]).tolist())
    S = dict(single_a={}, single_b={}, cumulative_a={}, cumulative_b={}, mean_ablation={})
    for i, s in enumerate(STG):
        S['single_a'][s] = sub({s: Fa[s]}); S['single_b'][s] = sub({s: Fb1[s]})
        S['cumulative_a'][s] = sub({k: Fa[k] for k in STG[:i + 1]}); S['cumulative_b'][s] = sub({k: Vb[k] for k in STG[:i + 1]})
        S['mean_ablation'][s] = sub({s: np.tile(Nt[s][:, UNITS[s]].mean(0), (J, 1))})
        log(f'substitution {s}: single (a) same {S["single_a"][s]["same_call"]:.4f} acc {S["single_a"][s]["acc"]:.4f} | single (b) {S["single_b"][s]["same_call"]:.4f} {S["single_b"][s]["acc"]:.4f} | '
            f'cumulative (a) {S["cumulative_a"][s]["same_call"]:.4f} {S["cumulative_a"][s]["acc"]:.4f} | cumulative (b) {S["cumulative_b"][s]["same_call"]:.4f} {S["cumulative_b"][s]["acc"]:.4f} | '
            f'all live units at their mean {S["mean_ablation"][s]["same_call"]:.4f} {S["mean_ablation"][s]["acc"]:.4f}')
    res['substitution'] = S
    res['all_if_statement_model'] = dict(acc=S['cumulative_b']['z']['acc'], same_call=S['cumulative_b']['z']['same_call'], network_acc=acc_net,
                                         n_terms=sum(u['n_terms'] for s in STG for u in fb[s]['units']), n_units=sum(len(UNITS[s]) for s in STG),
                                         z_on_physics_acc=S['single_a']['z']['acc'], z_on_physics_same_call=S['single_a']['z']['same_call'],
                                         reference_J12='16-neuron MARS formula fitted on the TRAIN archive: 81.31 % (network 80.94 %)')
    # controls
    ctrl = {}
    for p in sorted(OUT.glob(f'{ARGS.prefix}_control_untrained_*.json')):
        d = json.loads(p.read_text())
        for s, v in d['stages'].items():
            r = np.array([u['test_r2'] for u in v['units']]); tr = np.array(list(r2[s]['a']['per_unit'].values()))
            ctrl[s] = dict(untrained=dict(median=float(np.median(r)), min=float(r.min()), max=float(r.max()), per_unit=dict(zip(map(str, v['units_by_spread']), map(float, r))),
                                          n_terms=[u['n_terms'] for u in v['units']]), trained_median=float(np.median(tr)), trained_min=float(tr.min()))
            log(f'control untrained {s}: R² median {np.median(r):.3f} min {r.min():.3f} max {r.max():.3f} (trained median {np.median(tr):.3f} min {tr.min():.3f})')
    for p in sorted(OUT.glob(f'{ARGS.prefix}_control_shuffle_*.json')):
        d = json.loads(p.read_text()); r = np.array([u['test_r2'] for u in d['units']])
        ctrl.setdefault(d['stage'], {})['shuffled'] = dict(median=float(np.median(r)), min=float(r.min()), max=float(r.max()), n_terms=[u['n_terms'] for u in d['units']])
        log(f'control shuffled {d["stage"]}: R² median {np.median(r):.3f} min {r.min():.3f} max {r.max():.3f}; terms kept {sum(u["n_terms"] for u in d["units"])}')
    res['controls'] = ctrl
    # what each unit looks at
    res['lookup'] = {s: [dict(unit=u['unit'], cost=LIVE[s][u['unit']], n_terms=u['n_terms'], test_r2_a=u['test_r2'], top3_a=u['top3'],
                              test_r2_b=ub['test_r2'], top3_b=ub['top3']) for u, ub in zip(fa[s]['units'], fb[s]['units'])] for s in STG}
    # feature evolution: family shares per stage, (a) direct and (b) propagated through the chain
    def shares_a(u):
        v = np.zeros(len(FAMILIES))
        for t in u['terms']:
            if t.get('q2'): v[family(t['q'])] += t['weight_share'] / 2; v[family(t['q2'])] += t['weight_share'] / 2
            else: v[family(t['q'])] += t['weight_share']
        return v
    fam_b = {}
    def shares_b(s, u):
        v = np.zeros(len(FAMILIES))
        def one(q, w):
            f = family(q)
            if f is None: v[:] += w * fam_b[q]
            else: v[f] += w
        for t in u['terms']:
            if t.get('q2'): one(t['q'], t['weight_share'] / 2); one(t['q2'], t['weight_share'] / 2)
            else: one(t['q'], t['weight_share'])
        return v
    evo = {'a': {}, 'b': {}, 'a_equal': {}, 'b_equal': {}, 'families': FAMILIES}
    for s in STG:
        cost = np.array([LIVE[s][j] for j in UNITS[s]]); cw = cost / cost.sum(); eq = np.full(len(cost), 1 / len(cost))
        A = np.stack([shares_a(u) for u in fa[s]['units']]); B = []
        for u in fb[s]['units']:
            sb = shares_a(u) if s == 'P' else shares_b(s, u); fam_b[uid(s, u['unit'])] = sb; B.append(sb)
        B = np.stack(B)
        evo['a'][s] = (cw @ A).tolist(); evo['b'][s] = (cw @ B).tolist(); evo['a_equal'][s] = (eq @ A).tolist(); evo['b_equal'][s] = (eq @ B).tolist()
        log(f'families {s}: (a) ' + ', '.join(f'{f} {v:.2f}' for f, v in zip(FAMILIES, evo['a'][s])) + ' | (b) ' + ', '.join(f'{v:.2f}' for v in evo['b'][s]))
    res['evolution'] = evo
    save('mars_all', res)
    try:
        plot(res)
    except Exception as e:  # noqa: BLE001
        log(f'plot failed: {e!r}')


def plot(res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('blue', ['#ffffff', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
    evo = res['evolution']; fams = evo['families']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, key, title in zip(axes, ('a', 'b'), ('(a) each stage fitted on physics observables', '(b) nested program, physics content propagated through the chain')):
        M = np.array([evo[key][s] for s in STG]).T * 100
        ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect='auto')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', fontsize=9, color='white' if M[i, j] > 55 else '#222222')
        ax.set_xticks(range(len(STG))); ax.set_xticklabels(['P\nfirst objects', 'g\ncontext', 'S\nsecond objects', 'h4', 'h5', 'z\nneurons'], fontsize=9)
        ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams, fontsize=9); ax.set_title(title, fontsize=10, loc='left')
        ax.tick_params(length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
    fig.suptitle('What each stage\'s formulas are made of: % of formula weight |coef × spread| per observable family (units weighted by ablation cost)', fontsize=10.5, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94)); p = OUT / f'{ARGS.prefix}_evolution.png'; fig.savefig(p, dpi=150); log(f'wrote {p}')


# ------------------------------------------------------------------ main
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['fit', 'control', 'eval']); ap.add_argument('args', nargs='*')
    ap.add_argument('--per-class', type=int, default=6000); ap.add_argument('--test-per-class', type=int, default=2000)
    ap.add_argument('--max-terms', type=int, default=30); ap.add_argument('--n-select', type=int, default=10000); ap.add_argument('--prefix', default='e11')
    ap.add_argument('--halfplanes', action='store_true', help='supplement: add the one-sided angular sums of handbuilt() to the physics inputs')
    ARGS = ap.parse_args()
    log('args', vars(ARGS))
    if ARGS.mode == 'fit':
        D = Data(ARGS)
        if ARGS.args[0] == 'a': mode_fit_a(D, ARGS.args[1])
        else: mode_fit_b(D)
    elif ARGS.mode == 'control':
        if ARGS.args[0] == 'untrained': mode_control_untrained(Data(ARGS, untrained=True), ARGS.args[1].split(','))
        else: mode_control_shuffle(Data(ARGS), ARGS.args[1])
    else:
        mode_eval(Data(ARGS))
    log('done')
