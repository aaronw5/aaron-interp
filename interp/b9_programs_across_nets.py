"""B9 (Paper B claims B2, B4, B6t; Paper A claims A1x / A1r multi-seed stability; stage 2, exploratory / rigour): one program
extraction recipe applied to ten trained JEDI-linear networks (and 3 untrained redraws): fidelity vs description length,
recurrence of program statements across networks, cross-network transfer, and agreement with the networks' edit responses.

Question: does the rank-window MARS program (B6/B7) give programs of similar size and fidelity on every trained JEDI-linear
network, do the same statements recur across seeds and checkpoints, and does a program extracted from one network predict
another network's calls and edit responses as well as that network's own program?

Networks (no training; interp.e33_program_recurrence_jedi.Model): official 3-feature n64, n32, n128, perm-inv n64, 16-feature
n64, X12 HGQ-r150 s0 / s1 / s2, X12 float s0 / s1 (hgq150_s1, 57 % accuracy, reported as degenerate and left out of pooled
summaries); untrained floor: jedi_core.Jedi(64, untrained=True, seed 0 / 1 / 2) (weights redrawn).

PLAN (fixed before the first run)
Samples (jedi_core.sample rows; all networks see the same jets): fit = sample(1000, 4901); dev = sample(1000, 4902) minus fit
  rows (model selection only); test = sample(1000, 4903) minus fit and dev rows. n32 sees the first 32 slots; n128 the same rows
  of val_ptetaphi_f16.npy (128 slots); 16f the same rows of val_c16_f16.npy (e33 loader, copies checked equal to sample()).
Program inputs ('view'): the 64-slot (pT, deta, dphi) jet; for n32 slots >= 32 are zeroed (what the network sees).
Feature pool (identical for every network, as B6): the 97 rank-window sums of e11_mars_all.handbuilt (sum pT, N, sum pT r^2,
  sum r^2, sum r, sum pT r, sum (r-0.2)+, N(pT>1), N(pT<2), sum r[r<0.165], N(pT<2, r>0.1) over pT-rank windows all, top3,
  top11, top14, top26, top39, ranks 1-63, 8-63, 13-63) + M0 = sum pT (98 inputs).
(1) Program: MARS (jetdistill.mars.fit_neuron, B6 settings: target = logit + constant, stall rule off) per logit on the network's
  5 logits on the fit jets, forward path to 64 terms; nested programs K = 2, 4, 8, 16, 32, 64 terms per logit = path prefixes
  refitted by least squares; call = argmax. Description length DL (bits, B6): per logit 32 (intercept) + per term 32 (coef) +
  per factor [log2(98) + 2 + 32 (knot, unless linear)].
(2) Baselines fitted to the same network (calls or logits) on the fit jets: CART classifier (depth 2-12) on the pool;
  HistGradientBoosting classifier on the pool quantised to 2^q levels (q = 4, 6, 8; leaves 4 / 8 / 16; iterations 3 / 10 / 30 /
  100; lr 0.5; DL = B6w hgb_bits_q); Lasso per logit on the standardised pool (10 alphas; DL = non-zero weights x (32 +
  log2 98) + 5 x 32); MLP(m, sum pT, N) -> 5 logits (one hidden ReLU layer of 2-64 units; DL = 32 x parameters); MARS on the
  standard observables (B8 STD: m, m_SD, tau21, tau32, D2, N, sum pT, sum pT r^n n = 1, 2, 3; K = 2-32).
  Controls: shuffled-logit program (MARS K16 fitted on the fit rows' logits permuted, rng 4906); random-feature program (MARS
  K16 restricted to 6 pool inputs drawn at random, rng 4907); truth program ('class-score-free' baseline: MARS K16 fitted to
  the truth labels, target log(0.95 onehot + 0.01), never to the network).
  Metrics on test jets: same-call % with the network (Wilson 95 %), Cohen kappa (5 classes) between program and network calls,
  fidelity-vs-DL frontier (member chosen by dev same-call among those with DL <= b) and its mean over a common log2-DL grid.
(3) Edits on the test jets (64-slot jet; padding never moved; n32 sees the first 32 slots of the edited jet; n128 and 16f
  networks are not run on edits, as E39: their inputs carry slots / columns the edits do not define):
  shift  every real particle's deta + 0.06 rad (rigid translation; the W -> Z flow edit, E6 / E39 T).
  turn   E27w arm B, subset hard14 (pT ranks 0-13), dose 10 %: each of those particles turned about the jet axis toward the
         direction of the pT-weighted centroid M1 by a common fraction t of its angle gap, t solved per jet so that the jet
         mass drops by 10 %; pT and r of every particle kept (M0, M2, every rank-window sum unchanged); jets where it is
         feasible.
  split  E40 C1 split_r0_f0.5: exact collinear split of the hardest particle into two halves of equal pT at the same (deta,
         dphi), re-sorted (jet four-momentum, M0, M1, M2 exact, N + 1); jets with N <= 63.
  soft   E40 C2 emit_k1 at 0.5 GeV: one particle of 0.5 GeV at r = 0.3, uniform azimuth (rng 4905); all other particles get
         one common scale and boost so that the jet four-momentum is exact; jets with N <= 63.
  Per jet: network flip nf = call(edited) != call(unedited); surrogate flip sf likewise (surrogate's own calls);
  agreement = (no flip in both) or (both flip to the same new call); recall = share of network flips reproduced (flip to the
  same new call); precision = share of surrogate flips that are network flips to the same call; Cohen kappa of (nf, sf)
  (0 = never-flip rule, 1 = identical flip sets). Controls: shuffled-logit and random-feature programs.
(4) Recurrence: terms of each network's K = 16 program (5 logits; also K = 8). Two terms match if they use the same pool
  input(s) (same rank window and sum) and hinge type, and their knots differ by <= 0.05 in quantile units of the input on that
  network's fit jets (knots sit on the 5 %-spaced quantile grid, so this is the same or the neighbouring grid knot).
  Recurrence count of a term = number of the other 8 non-degenerate trained networks whose program contains a matching term
  (any logit). Null: key-relabel (each term of every other network's program gets its input(s) redrawn uniformly from the 98,
  hinge type and knot quantile kept; the match probability per other network is exact, the count is Poisson-binomial over the
  8 networks; one-sided tail p), Holm over each network's terms;
  floor: the share of the 3 untrained-network programs containing a match. Core families (any term using the input):
  S29-like radial = sum pT r^2, sum r^2, N(pT<2, r>0.1) in any window; S0-like hard = sum pT over top-k windows or M0;
  soft count = N(pT<2) in any window, N over ranks 1-63 / 8-63 / 13-63. Presence in the K = 8 programs per network vs the
  key-relabel null rate (Poisson-binomial tail), also for K = 2 (the first two terms per logit) and the share of program
  terms using the family vs the share expected under the key-relabel null (K = 2 and shares added after the smoke run, before
  the main aggregation: at K = 8 / 16 every family is present under the null too).
(5) Transfer: program K16 of network A scored against network B's calls on the same jets (same-call; edit agreement /
  recall on B's flips) vs B's own K16 program and vs the truth program; ordered pairs among the 9 non-degenerate trained
  networks (edits: among the 7 edit-capable ones, programs of n128 / 16f included as donors). Loss = own - transfer;
  D = loss(edit agreement) - loss(natural same-call) per pair; network-level bootstrap (2,000: networks resampled with
  replacement, mean over the pairs of distinct drawn networks).
Statistics: 1,000-draw jet bootstrap (multinomial counts over test rows; edited sets use the same draws of their base rows);
  exact McNemar for paired per-jet indicators (program vs matched-DL baseline: member of the baseline family with DL <= the
  program's DL and the highest dev same-call); Holm over the 9 non-degenerate trained networks per (K, family, metric).
Run: python -m interp.b9_programs_across_nets net --nets official_n64,official_n32   -> $AI_RESULTS/jedi/b9_net_<name>.npz/.json
     python -m interp.b9_programs_across_nets agg                                    -> b9.json, b9.png   (--smoke: tiny run)
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault('KERAS_BACKEND', 'jax')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
T0 = time.time()
CL = ['g', 'q', 'W', 'Z', 't']
TRAINED = ['official_n64', 'official_n32', 'official_n128', 'perminv_n64', 'f16_n64', 'hgq150_s0', 'hgq150_s1', 'hgq150_s2',
           'float_s0', 'float_s1']
UNTR = ['untrained_s0', 'untrained_s1', 'untrained_s2']
DEGEN = {'hgq150_s1'}
GOOD = [n for n in TRAINED if n not in DEGEN]
NOEDIT = {'official_n128', 'f16_n64'}
EDITS = ['shift', 'turn', 'split', 'soft']
EDIT_LABEL = dict(shift='shift: all particles deta + 0.06 rad', turn='turn: hardest 14 turned in azimuth, mass - 10 % (M2 kept)',
                  split='split: hardest particle split collinearly 50/50', soft='soft: one 0.5 GeV particle added at r = 0.3')
KS = [2, 4, 8, 16, 32, 64]
KSTD = [2, 4, 8, 16, 32]
SEEDS = dict(fit=4901, dev=4902, test=4903)


def log(*a): print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else round(float(o), 6)
    return o


def wilson(k, n, z=1.959964):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def holm(ps):
    ps = list(ps); idx = [i for i, p in enumerate(ps) if p is not None]; order = sorted(idx, key=lambda i: ps[i]); m = len(order)
    out = [None] * len(ps); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out


def mcnemar(a, b):
    from scipy.stats import binomtest
    a = np.asarray(a, bool); b = np.asarray(b, bool); n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return dict(a_only=n10, b_only=n01, p=1.0 if n10 + n01 == 0 else float(binomtest(n10, n10 + n01, 0.5).pvalue))


def ci(b):
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) else [None, None]


def bootp(b):
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    return float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()) + 1 / max(len(b), 1))) if len(b) else None


# ================================================================================================ data and edits
def load_sets(name, per):
    """-> dict(x3={fit, dev, test}: 64-slot jets, xin={...}: network inputs, y={...}, rows info)"""
    from interp import e33_program_recurrence_jedi as E33
    from jetdistill.config import RESULTS
    n, f16 = 64, False
    if name in E33.MODELS: _, fam, n = E33.MODELS[name]; f16 = fam.startswith('16-feature')
    yall = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y'])
    idx = {s: E33.rows(yall, per, sd) for s, sd in SEEDS.items()}
    keep = dict(fit=np.ones(len(idx['fit']), bool), dev=~np.isin(idx['dev'], idx['fit']),
                test=~np.isin(idx['test'], np.concatenate([idx['fit'], idx['dev']])))
    X3, XI, Y = {}, {}, {}
    for s, sd in SEEDS.items():
        xin, xk, x3, y, chk = E33.load(per, sd, n, f16)
        k = keep[s]; X3[s] = np.ascontiguousarray(x3[k], np.float32); XI[s] = np.ascontiguousarray(xin[k], np.float32); Y[s] = y[k]
    info = {s: dict(N=int(keep[s].sum()), removed=int((~keep[s]).sum())) for s in SEEDS}
    return X3, XI, Y, info, n, f16


def make_edits(xT, smoke=False):
    """edits on the 64-slot test jets -> {edit: (x_edited (J_e, 64, 3), base rows into test)}, info"""
    from interp.e27w_mass_vs_m2_jedi import solve_B, M1v, wrap, to_x, mass_p
    from interp.e40_qg_count_jedi import split_gen, emit
    from interp.e28w_split_route_jedi import nreal
    J = len(xT); real = xT[..., 0] > 0; N = real.sum(1); E, info = {}, {}
    s = xT.copy(); s[..., 1] += np.float32(0.06) * real; E['shift'] = (s, np.arange(J)); info['shift'] = dict(N=J)
    pt = xT[..., 0].astype(np.float64); r = np.hypot(xT[..., 1], xT[..., 2]).astype(np.float64)
    th = np.arctan2(xT[..., 2], xT[..., 1]).astype(np.float64)
    rank = np.argsort(np.argsort(-np.where(real, pt, -1), 1), 1); sub = real & (rank < 14)
    M10 = M1v(pt, r, th); phu = np.arctan2(M10[:, 1], M10[:, 0])
    d = np.where(sub, wrap(phu[:, None] - th), 0.0); tB, okB = solve_B(pt, r, th, d, 0.10); thB = th + tB[:, None] * d
    xB = to_x(xT, real, r, thB); m0, m1 = mass_p(pt, r, th), mass_p(pt, r, thB)
    okB = okB & (np.abs(m1 / np.maximum(m0, 1e-9) - 0.9) < 0.005); rows = np.flatnonzero(okB)
    E['turn'] = (xB[rows], rows)
    M2 = lambda pt_, r_: (pt_ * r_ ** 2).sum(1)
    info['turn'] = dict(N=int(len(rows)), median_dlogm=float(np.median(np.log(m1[rows] / m0[rows]))) if len(rows) else None,
                        max_rel_dM2=float(np.max(np.abs(M2(pt[rows], np.hypot(xB[rows][..., 1], xB[rows][..., 2])) / M2(pt[rows], r[rows]) - 1))) if len(rows) else None)
    slot = np.where(N <= 63, 0, -1); Ys, d1, d2 = split_gen(xT, slot, 0.5, 0.0, np.random.default_rng(4908)); rows = np.flatnonzero(slot >= 0)
    E['split'] = (np.ascontiguousarray(Ys[rows], np.float32), rows); info['split'] = dict(N=int(len(rows)), dN=float(np.mean(nreal(Ys[rows]) - N[rows])))
    Yo, ok, inf = emit(xT, 1, 0.5, 0.3, np.random.default_rng(4905)); rows = np.flatnonzero(ok)
    E['soft'] = (np.ascontiguousarray(Yo[rows], np.float32), rows); info['soft'] = dict(N=int(len(rows)), **inf)
    log('edits', json.dumps(info))
    return E, info


def view(x, n):
    if n >= 64: return x
    v = x.copy(); v[:, n:] = 0; return v


def feats(xv):
    """pool (98) and standard observables (10) of 64-slot jets"""
    from interp.b6_pareto import features
    from interp.b8_fidelity_dl_iia import std_obs
    X, keys = features(xv); return X.astype(np.float64), keys, std_obs(xv)


# ================================================================================================ fitting
def mars_path(Xf, Tf, Xd, Td, keys, kmax, nsel=5000):
    from jetdistill import mars
    from interp.e11_mars_all import knots_of
    Q = {'fit': {k: Xf[:, i] for i, k in enumerate(keys)}, 'dev': {k: Xd[:, i] for i, k in enumerate(keys)}}
    knots = knots_of(Q['fit'], keys); NF = len(Tf); sel = np.random.default_rng(1).choice(NF, min(nsel, NF), replace=False)
    mars.MAX_TERMS, mars.STALL = kmax, 10 ** 6; fits = []
    for k in range(Tf.shape[1]):
        c = 1.0 - min(Tf[:, k].min(), Td[:, k].min()) + 0.5 * max(Tf[:, k].std(), 1e-6)
        terms, _, path = mars.fit_neuron(dict(fit=Tf[:, k] + c, dev=Td[:, k] + c), Q, keys, knots, sel)
        fits.append((terms, c))
    return fits, Q


def mars_prefix(fits, Q, Tf, keys, K):
    from jetdistill import mars
    from interp.b6_pareto import term_bits
    NF = len(Tf); parts = []
    for k, (terms, c) in enumerate(fits):
        tr = terms[:K]; co = mars._lstsq(tr, Q['fit'], Tf[:, k] + c, NF); parts.append((tr, co, c))
    bits = sum(32 + sum(sum(term_bits(t, len(keys))) for t in tr) for tr, _, _ in parts)
    return parts, float(bits)


def mars_predict(parts, X, keys):
    from jetdistill import mars
    Qd = {k: X[:, i] for i, k in enumerate(keys)}; J = len(X)
    return np.stack([np.stack([np.ones(J)] + [mars.term_value(t, Qd) for t in tr], 1) @ co - c for tr, co, c in parts], 1)


def hgb_bits_q(m, nin, q):
    b, used = 0.0, set()
    for it in m._predictors:
        for p in it:
            il = p.nodes['is_leaf'].astype(bool); nl = int(il.sum())
            b += (len(p.nodes) - nl) * (np.log2(max(nin, 2)) + q) + nl * 32
            used |= set(p.nodes['feature_idx'][~il].tolist())
    return b + 64 * len(used)


def quantiser(Xf, q):
    lo, hi = np.percentile(Xf, 0.5, 0), np.percentile(Xf, 99.5, 0); hi = np.where(hi > lo, hi, lo + 1.0); L = min(2 ** q, 255) - 1
    return lambda Z: np.clip(np.round((Z - lo) / (hi - lo) * L), 0, L)


def term_record(t, Qfit):
    """JSON form of a MARS term with knot quantiles on the fit jets"""
    d = dict(q=t['q'], kind=t['kind'], t=t.get('t'), qt=None if t['kind'] == 'lin' else float((Qfit[t['q']] <= t['t']).mean()))
    if t.get('q2'):
        d.update(q2=t['q2'], kind2=t['kind2'], t2=t.get('t2'), qt2=None if t['kind2'] == 'lin' else float((Qfit[t['q2']] <= t['t2']).mean()))
    return d


class Runner:
    def __init__(self, name):
        self.name = name
        if name.startswith('untrained'):
            from interp.jedi_core import Jedi
            j = Jedi(64, '3-feature', untrained=True, seed=int(name[-1])); self.run = lambda x: j.run(x)['logits']
        else:
            from interp.e33_program_recurrence_jedi import Model
            m = Model(name); self.run = lambda x: m.run(x)['logits']


def run_net(name, smoke=False):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import Lasso
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import cohen_kappa_score
    import warnings; warnings.filterwarnings('ignore')
    tag = 'b9smoke' if smoke else 'b9'; per = 60 if smoke else 1000
    R = dict(node='B9', network=name, plan='module docstring (fixed before the first run)', smoke=smoke)
    X3, XI, Y, R['samples'], n, f16 = load_sets(name, per)
    log(f'{name}: n {n}, 16f {f16}; samples', json.dumps(R['samples']))
    net = Runner(name); edit_ok = name not in NOEDIT
    E, R['edits'] = make_edits(X3['test'], smoke)
    sets = dict(fit=X3['fit'], dev=X3['dev'], test=X3['test'], **{e: E[e][0] for e in EDITS})
    base = {e: E[e][1] for e in EDITS}
    # network logits / calls
    L, NC = {}, {}
    for s in ('fit', 'dev', 'test'):
        L[s] = net.run(XI[s]); NC[s] = L[s].argmax(1)
    R['accuracy'] = {s: dict(acc=float((NC[s] == Y[s]).mean()), ci=wilson(int((NC[s] == Y[s]).sum()), len(Y[s]))) for s in NC}
    R['call_frac_test'] = (np.bincount(NC['test'], minlength=5) / len(NC['test'])).tolist()
    log(f'{name}: accuracy', json.dumps({s: round(v["acc"] * 100, 2) for s, v in R['accuracy'].items()}))
    if edit_ok:
        for e in EDITS:
            xe = sets[e]; NC[e] = net.run(np.ascontiguousarray(xe[:, :n]) if n < 64 else xe).argmax(1)
        R['net_flip_rate'] = {e: float((NC[e] != NC['test'][base[e]]).mean()) for e in EDITS}
        log(f'{name}: network flip rates %', json.dumps({e: round(v * 100, 2) for e, v in R['net_flip_rate'].items()}))
    # features of the program's view
    F, S = {}, {}
    for s, xx in sets.items():
        F[s], keys, S[s] = feats(view(xx, n))
    log(f'{name}: features {len(keys)} pool + {S["test"].shape[1]} standard')
    calls, meta = {}, []
    J = len(NC['test'])

    def add(nm, fam, bits, pred_fn, extra=None):
        c = {s: np.asarray(pred_fn(s), np.int8) for s in sets}
        for s in sets: calls[f'{nm}|{s}'] = c[s]
        sd = float((c['dev'] == NC['dev']).mean()); st = float((c['test'] == NC['test']).mean())
        meta.append(dict(name=nm, family=fam, bits=float(bits), same_call_dev=sd, same_call_test=st,
                         kappa_test=float(cohen_kappa_score(c['test'], NC['test'])), **(extra or {})))
        log(f'{name}: {nm:18s} ({fam:9s}) bits {bits:9.0f}  same call dev {sd * 100:6.2f}  test {st * 100:6.2f}')
    # (1) program
    t = time.time(); kmax = 16 if smoke else 64
    fits, Q = mars_path(F['fit'], L['fit'], F['dev'], L['dev'], keys, kmax)
    log(f'{name}: program MARS path to {kmax} terms per logit in {time.time() - t:.0f} s')
    R['program_terms'] = [[term_record(tm, Q['fit']) for tm in terms] for terms, _ in fits]
    for K in KS:
        if K > kmax: continue
        parts, bits = mars_prefix(fits, Q, L['fit'], keys, K)
        add(f'prog_K{K}', 'prog', bits, lambda s, p=parts: mars_predict(p, F[s], keys).argmax(1))
    # controls
    perm = np.random.default_rng(4906).permutation(len(L['fit']))
    fs, Qs = mars_path(F['fit'], L['fit'][perm], F['dev'], L['dev'], keys, 16)
    p_, b_ = mars_prefix(fs, Qs, L['fit'][perm], keys, 16); add('shuf_K16', 'shuffled', b_, lambda s, p=p_: mars_predict(p, F[s], keys).argmax(1))
    rk = np.sort(np.random.default_rng(4907).choice(len(keys), 6, replace=False)); kr = [keys[i] for i in rk]
    fr, Qr = mars_path(F['fit'][:, rk], L['fit'], F['dev'][:, rk], L['dev'], kr, 16)
    p_, b_ = mars_prefix(fr, Qr, L['fit'], kr, 16)
    add('rfeat_K16', 'randfeat', b_, lambda s, p=p_: mars_predict(p, F[s][:, rk], kr).argmax(1), dict(keys=kr))
    Tt = lambda y: np.log(0.95 * np.eye(5)[y] + 0.01)
    ft, Qt = mars_path(F['fit'], Tt(Y['fit']), F['dev'], Tt(Y['dev']), keys, 16)
    p_, b_ = mars_prefix(ft, Qt, Tt(Y['fit']), keys, 16); add('truth_K16', 'truth', b_, lambda s, p=p_: mars_predict(p, F[s], keys).argmax(1))
    R['truth_terms'] = [[term_record(tm, Qt['fit']) for tm in terms] for terms, _ in ft]
    log(f'{name}: controls done')
    # (2) baselines: MARS on standard observables
    from interp.b8_fidelity_dl_iia import STD_NAMES
    fsd, Qsd = mars_path(S['fit'], L['fit'], S['dev'], L['dev'], STD_NAMES, 8 if smoke else 32)
    for K in KSTD:
        if K > (8 if smoke else 32): continue
        p_, b_ = mars_prefix(fsd, Qsd, L['fit'], STD_NAMES, K)
        add(f'std_K{K}', 'std', b_, lambda s, p=p_: mars_predict(p, S[s], STD_NAMES).argmax(1))
    nin = len(keys); lb = float(np.log2(nin))
    for dpt in range(2, 13):
        m = DecisionTreeClassifier(max_depth=dpt, random_state=0).fit(F['fit'], NC['fit']); tr = m.tree_
        bits = (tr.node_count - tr.n_leaves) * (lb + 32) + tr.n_leaves * np.log2(5)
        add(f'cart_d{dpt}', 'cart', bits, lambda s, m=m: m.predict(F[s]))
    for q in (4, 6, 8):
        qz = quantiser(F['fit'], q); Fq = {s: qz(F[s]) for s in sets}
        for lv in (4, 8, 16):
            for it in (3, 10, 30, 100):
                m = HistGradientBoostingClassifier(learning_rate=0.5, max_iter=it, max_leaf_nodes=lv, early_stopping=False, random_state=0)
                m.fit(Fq['fit'], NC['fit'])
                add(f'gbdt_q{q}_L{lv}_T{it}', 'gbdt', hgb_bits_q(m, nin, q), lambda s, m=m, Fq=Fq: m.predict(Fq[s]))
    mu, sdv = F['fit'].mean(0), F['fit'].std(0) + 1e-12; Z = {s: (F[s] - mu) / sdv for s in sets}
    for ia, al in enumerate(np.logspace(-3, 0, 10)):
        ms = [Lasso(alpha=al, max_iter=5000).fit(Z['fit'], L['fit'][:, k]) for k in range(5)]
        nz = sum(int((np.abs(m.coef_) > 0).sum()) for m in ms)
        add(f'lasso_a{ia}', 'lasso', nz * (32 + lb) + 5 * 32, lambda s, ms=ms: np.stack([m.predict(Z[s]) for m in ms], 1).argmax(1), dict(alpha=float(al), nonzero=nz))
    io = [STD_NAMES.index('m'), STD_NAMES.index('sum pT'), STD_NAMES.index('N')]
    mm, ss = S['fit'][:, io].mean(0), S['fit'][:, io].std(0) + 1e-12; M3 = {s: (S[s][:, io] - mm) / ss for s in sets}
    for h in (2, 4, 8, 16, 32, 64):
        m = MLPRegressor(hidden_layer_sizes=(h,), max_iter=200 if smoke else 1000, random_state=0).fit(M3['fit'], L['fit'])
        add(f'mlp_h{h}', 'mlp', 32 * (3 * h + h + 5 * h + 5), lambda s, m=m: m.predict(M3[s]).argmax(1))
    R['surrogates'] = meta; R['keys'] = keys; R['std_keys'] = STD_NAMES; R['edit_ok'] = edit_ok; R['n_slots_seen'] = n
    out = {f'netcall|{s}': np.asarray(v, np.int8) for s, v in NC.items()}
    out.update({f'base|{e}': base[e] for e in EDITS}); out['y|test'] = Y['test']; out['logits|test'] = L['test'].astype(np.float32)
    out.update(calls)
    np.savez_compressed(OUT / f'{tag}_net_{name}.npz', **out)
    (OUT / f'{tag}_net_{name}.json').write_text(json.dumps(clean(R), indent=1))
    log(f'{name}: wrote {tag}_net_{name}.npz/.json')


# ================================================================================================ aggregation
def edit_ind(nc0, nc1, sc0, sc1):
    nf = nc1 != nc0; sf = sc1 != sc0; same_new = sc1 == nc1
    agree = (~nf & ~sf) | (nf & sf & same_new)
    return nf, sf, agree, nf & sf & same_new


def kappa(nf, sf):
    nf = np.asarray(nf, float); sf = np.asarray(sf, float); po = (nf == sf).mean(); pn, ps = nf.mean(), sf.mean()
    pe = pn * ps + (1 - pn) * (1 - ps); return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def term_match(a, b, tol=0.05 + 1e-9):
    def f(q, k, qt, q_, k_, qt_):
        return q == q_ and k == k_ and (k == 'lin' or (qt is not None and qt_ is not None and abs(qt - qt_) <= tol))
    a2, b2 = a.get('q2'), b.get('q2')
    if (a2 is None) != (b2 is None): return False
    if a2 is None: return f(a['q'], a['kind'], a['qt'], b['q'], b['kind'], b['qt'])
    return (f(a['q'], a['kind'], a['qt'], b['q'], b['kind'], b['qt']) and f(a2, a['kind2'], a['qt2'], b2, b['kind2'], b['qt2'])) or \
           (f(a['q'], a['kind'], a['qt'], b2, b['kind2'], b['qt2']) and f(a2, a['kind2'], a['qt2'], b['q'], b['kind'], b['qt']))


def core_family(q):
    k = q[2:] if q.startswith('H:') else q
    fams = set()
    if k.startswith(('ΣpT·r²[', 'Σr²[', 'N(pT<2,r>0.1)[')): fams.add('S29-like radial')
    if (k.startswith('ΣpT[top')) or q.startswith('M0'): fams.add('S0-like hard pT sum')
    if k.startswith('N(pT<2)[') or k in ('N[r1-63]', 'N[r8-63]', 'N[r13-63]'): fams.add('soft count')
    return fams


def prog_terms(R, K):
    return [dict(t, logit=CL[i]) for i, terms in enumerate(R['program_terms']) for t in terms[:K]]


def agg(smoke=False, NB=1000):
    tag = 'b9smoke' if smoke else 'b9'
    names = [n for n in TRAINED + UNTR if (OUT / f'{tag}_net_{n}.json').exists()]
    log('networks found:', names)
    RJ = {n: json.loads((OUT / f'{tag}_net_{n}.json').read_text()) for n in names}
    names = [n for n in names if any(m['name'] == 'prog_K16' for m in RJ[n]['surrogates'])]; log('networks with a K16 program:', names)
    ZZ = {n: dict(np.load(OUT / f'{tag}_net_{n}.npz')) for n in names}
    ref = names[0]; Jt = len(ZZ[ref]['y|test'])
    for n in names: assert len(ZZ[n]['y|test']) == Jt and np.array_equal(ZZ[n]['y|test'], ZZ[ref]['y|test'])
    for e in EDITS:
        for n in names: assert np.array_equal(ZZ[n][f'base|{e}'], ZZ[ref][f'base|{e}']), (n, e)
    base = {e: ZZ[ref][f'base|{e}'] for e in EDITS}
    rng = np.random.default_rng(4909); C = np.stack([np.bincount(rng.integers(0, Jt, Jt), minlength=Jt) for _ in range(NB)]).astype(np.float32)
    Cs = {'test': C, **{e: C[:, base[e]] for e in EDITS}}
    bm = lambda s, v: (Cs[s] @ np.asarray(v, np.float32)) / np.maximum(Cs[s].sum(1), 1)
    RES = dict(node='B9', procedure=__doc__, networks=names, degenerate=sorted(DEGEN), edit_label=EDIT_LABEL,
               N_test=Jt, N_edit={e: int(len(base[e])) for e in EDITS}, per_network={}, matched=[], recurrence={}, transfer={})
    RES['edit_info'] = RJ[ref]['edits']
    edit_nets = [n for n in names if RJ[n]['edit_ok']]

    def sc(n, sname):   # natural same-call indicator of surrogate sname (program from any job) vs network n
        return ZZ[n]['netcall|test'] == sname

    # ---------------------------------------------------------------- (1)-(3) per network
    for n in names:
        R = RJ[n]; Z = ZZ[n]; nc = Z['netcall|test']; meta = R['surrogates']; o = dict(accuracy=R['accuracy'], samples=R['samples'],
                                                                                      net_flip_rate=R.get('net_flip_rate'), surrogates={})
        for m in meta:
            s = Z[f'{m["name"]}|test']; ind = s == nc; d = dict(bits=m['bits'], same_call_dev=m['same_call_dev'], same_call=float(ind.mean()),
                                                                  same_call_wilson=wilson(int(ind.sum()), Jt), same_call_ci=ci(bm('test', ind)), kappa=m['kappa_test'])
            if R['edit_ok']:
                d['edits'] = {}
                for e in EDITS:
                    b = base[e]; nf, sf, ag, hit = edit_ind(nc[b], Z[f'netcall|{e}'], s[b], Z[f'{m["name"]}|{e}'])
                    d['edits'][e] = dict(agree=float(ag.mean()), agree_ci=ci(bm(e, ag)), net_flips=int(nf.sum()), sur_flips=int(sf.sum()),
                                         recall=float(hit.sum() / max(nf.sum(), 1)), recall_ci=ci(bm(e, hit) / np.maximum(bm(e, nf), 1e-9)),
                                         precision=float(hit.sum() / max(sf.sum(), 1)) if sf.sum() else None, kappa=kappa(nf, sf),
                                         never_flip_agree=float(1 - nf.mean()))
            o['surrogates'][m['name']] = d
        # frontier AUC of same-call and of shift / turn agreement (members chosen by dev same-call at DL <= b)
        fams = sorted({m['family'] for m in meta if m['family'] in ('prog', 'std', 'cart', 'gbdt', 'lasso', 'mlp')})
        lbits = {m['name']: np.log2(max(m['bits'], 1)) for m in meta}
        def frontier(fam, grid):
            mem = [m for m in meta if m['family'] == fam]; sel = []
            for g in grid:
                ok = [m for m in mem if lbits[m['name']] <= g + 1e-9]
                sel.append(max(ok, key=lambda m: m['same_call_dev'])['name'] if ok else min(mem, key=lambda m: m['bits'])['name'])
            return sel
        pr = [lbits[m['name']] for m in meta if m['family'] == 'prog']; o['auc'] = {}
        metrics = ['nat'] + (['shift', 'turn', 'split', 'soft'] if R['edit_ok'] else [])
        def perjet(name, met):
            s = Z[f'{name}|test']
            if met == 'nat': return 'test', s == nc
            b = base[met]; return met, edit_ind(nc[b], Z[f'netcall|{met}'], s[b], Z[f'{name}|{met}'])[2]
        for fam in fams:
            if fam == 'prog': continue
            fe = [lbits[m['name']] for m in meta if m['family'] == fam]
            lo, hi = max(min(pr), min(fe)), min(max(pr), max(fe))
            if hi <= lo + 0.25: lo, hi = min(min(pr), min(fe)), max(max(pr), max(fe))
            grid = np.linspace(lo, hi, 100); sp, se = frontier('prog', grid), frontier(fam, grid); o['auc'][fam] = dict(range_log2_bits=[lo, hi])
            for met in metrics:
                cache = {}
                def val(nm):
                    if nm not in cache: st, ind = perjet(nm, met); cache[nm] = (float(ind.mean()), bm(st, ind))
                    return cache[nm]
                ap = np.mean([val(x)[0] for x in sp]); ae = np.mean([val(x)[0] for x in se])
                bp = np.mean([val(x)[1] for x in sp], 0); be = np.mean([val(x)[1] for x in se], 0)
                o['auc'][fam][met] = dict(prog=float(ap), base=float(ae), diff=float(ap - ae), diff_ci=ci(bp - be), p=bootp(bp - be))
        RES['per_network'][n] = o
        p16 = o['surrogates'].get('prog_K16', {})
        log(f'{n}: acc {R["accuracy"]["test"]["acc"] * 100:.2f} %; prog K16 same call {p16.get("same_call", 0) * 100:.2f} %, bits {p16.get("bits")}'
            + (f'; shift recall {p16["edits"]["shift"]["recall"] * 100:.1f} %, turn recall {p16["edits"]["turn"]["recall"] * 100:.1f} %' if 'edits' in p16 else ''))
    # ---------------------------------------------------------------- matched DL (program vs each baseline family), Holm over networks
    trained_here = [n for n in GOOD if n in names]
    for K in (4, 8, 16, 32):
        for fam in ('std', 'cart', 'gbdt', 'lasso', 'mlp'):
            for met in ['nat'] + EDITS:
                rows = []
                for n in trained_here:
                    R, Z = RJ[n], ZZ[n]
                    if met != 'nat' and not R['edit_ok']: continue
                    meta = {m['name']: m for m in R['surrogates']}; pn = f'prog_K{K}'
                    if pn not in meta: continue
                    mem = [m for m in R['surrogates'] if m['family'] == fam]; ok = [m for m in mem if m['bits'] <= meta[pn]['bits']]
                    bsel = max(ok, key=lambda m: m['same_call_dev']) if ok else min(mem, key=lambda m: m['bits'])
                    nc = Z['netcall|test']
                    if met == 'nat':
                        a, b_ = Z[f'{pn}|test'] == nc, Z[f'{bsel["name"]}|test'] == nc; st = 'test'
                    else:
                        bb = base[met]; st = met
                        a = edit_ind(nc[bb], Z[f'netcall|{met}'], Z[f'{pn}|test'][bb], Z[f'{pn}|{met}'])[2]
                        b_ = edit_ind(nc[bb], Z[f'netcall|{met}'], Z[f'{bsel["name"]}|test'][bb], Z[f'{bsel["name"]}|{met}'])[2]
                    dd = bm(st, a) - bm(st, b_); mc = mcnemar(a, b_)
                    rows.append(dict(network=n, K=K, family=fam, metric=met, program_bits=meta[pn]['bits'], baseline=bsel['name'],
                                     baseline_bits=bsel['bits'], baseline_fits=bool(ok), prog=float(a.mean()), base=float(b_.mean()),
                                     diff=float(a.mean() - b_.mean()), diff_ci=ci(dd), mcnemar=mc))
                hp = holm([r['mcnemar']['p'] for r in rows])
                for r, h in zip(rows, hp): r['p_holm_over_networks'] = h
                RES['matched'] += rows
    # summary of matched comparisons
    summ = {}
    for r in RES['matched']:
        k = f'K{r["K"]}|{r["family"]}|{r["metric"]}'; s = summ.setdefault(k, dict(n=0, prog_higher_sig=0, base_higher_sig=0, mean_diff=[]))
        s['n'] += 1; s['mean_diff'].append(r['diff'])
        if r['p_holm_over_networks'] is not None and r['p_holm_over_networks'] < 0.05:
            s['prog_higher_sig' if r['diff'] > 0 else 'base_higher_sig'] += 1
    for s in summ.values(): s['mean_diff'] = float(np.mean(s['mean_diff']))
    RES['matched_summary'] = summ
    # ---------------------------------------------------------------- (4) recurrence
    allkeys = RJ[ref]['keys']; NK = len(allkeys)
    def fams_of(t): return core_family(t['q']) | (core_family(t['q2']) if t.get('q2') else set())
    def p_key(t, u):
        """null probability that term u, with its input(s) redrawn uniformly from the pool (hinge type and knot quantile kept),
        matches term t"""
        if (t.get('q2') is None) != (u.get('q2') is None): return 0.0
        tt = dict(t); uu = dict(u)
        if t.get('q2') is None:
            uu['q'] = t['q']; return 1.0 / NK if term_match(tt, uu) else 0.0
        pr = 0.0
        uu['q'], uu['q2'] = t['q'], t['q2']
        if term_match(tt, uu): pr += 1.0 / NK ** 2
        uu['q'], uu['q2'] = t['q2'], t['q']
        if t['q'] != t['q2'] and term_match(tt, uu): pr += 1.0 / NK ** 2
        return pr
    def pbin_tail(qs, k):
        dist = np.array([1.0])
        for q_ in qs: dist = np.convolve(dist, [1 - q_, q_])
        return float(dist[k:].sum()), float(np.sum(qs))
    for K in (16, 8, 2):
        P = {n: prog_terms(RJ[n], K) for n in names}
        rec = dict(K=K, per_network={}, families={}, match_rule='same pool input(s), same hinge type, knot quantiles within 0.05')
        tn = [n for n in GOOD if n in names]; un = [n for n in UNTR if n in names]
        def contains(prog, t): return any(term_match(t, u) for u in prog)
        for n in tn:
            others = [m for m in tn if m != n]; rows = []
            truthp = [x for tt in RJ[n]['truth_terms'] for x in tt[:K]]
            for t in P[n]:
                cnt = sum(contains(P[m], t) for m in others)
                q0 = [1 - np.prod([1 - p_key(t, u) for u in P[m]]) for m in others]
                p, nm = pbin_tail(q0, cnt)
                rows.append(dict(term=t, count=int(cnt), of=len(others), null_mean=nm, p=p, untrained=int(sum(contains(P[m], t) for m in un)),
                                 of_untrained=len(un), in_truth_program=bool(contains(truthp, t)), families=sorted(fams_of(t))))
            hp = holm([r['p'] for r in rows])
            for r, h in zip(rows, hp): r['p_holm'] = h
            rec['per_network'][n] = dict(n_terms=len(rows), mean_count=float(np.mean([r['count'] for r in rows])),
                                         mean_null=float(np.mean([r['null_mean'] for r in rows])), n_sig=int(sum(r['p_holm'] < 0.05 for r in rows)),
                                         share_recur_ge7=float(np.mean([r['count'] >= 7 for r in rows])),
                                         share_recur_ge1=float(np.mean([r['count'] >= 1 for r in rows])),
                                         mean_untrained_rate=float(np.mean([r['untrained'] / max(r['of_untrained'], 1) for r in rows])),
                                         share_in_truth=float(np.mean([r['in_truth_program'] for r in rows])), terms=rows)
            log(f'recurrence K{K} {n}: mean count {rec["per_network"][n]["mean_count"]:.2f} / {len(others)} (null {rec["per_network"][n]["mean_null"]:.3f}); '
                f'{rec["per_network"][n]["n_sig"]} / {len(rows)} terms Holm p < 0.05; >= 7 / 8: {rec["per_network"][n]["share_recur_ge7"]:.2f}; untrained rate {rec["per_network"][n]["mean_untrained_rate"]:.2f}')
        # core families: presence per network vs the key-relabel null probability
        for fam in ('S29-like radial', 'S0-like hard pT sum', 'soft count'):
            f = np.mean([fam in core_family(k) for k in allkeys])
            has = lambda prog: any(fam in fams_of(t) for t in prog)
            pres = {n: bool(has(P[n])) for n in names}
            nullrate = {n: float(1 - np.prod([(1 - f) ** (2 if t.get('q2') else 1) for t in P[n]])) for n in tn}
            k = sum(pres[n] for n in tn); pt_, _ = pbin_tail(list(nullrate.values()), k)
            terms_in = {n: [f"{t['logit']}: {t['q']} {t['kind']} q{t['qt']}" + (f" x {t['q2']} {t['kind2']}" if t.get('q2') else '') for t in P[n] if fam in fams_of(t)] for n in names}
            share = {n: float(np.mean([fam in fams_of(t) for t in P[n]])) for n in names}
            null_share = {n: float(np.mean([1 - (1 - f) ** (2 if t.get('q2') else 1) for t in P[n]])) for n in names}
            rec['families'][fam] = dict(share_of_pool_inputs=float(f), share_of_terms=share, null_share_of_terms=null_share, present=pres, n_trained_present=int(k), of=len(tn), null_rate=nullrate, p_tail=pt_,
                                        untrained_present=int(sum(pres[n] for n in un)), of_untrained=len(un), terms=terms_in,
                                        in_truth_program={n: bool(has([x for tt in RJ[n]['truth_terms'] for x in tt[:K]])) for n in tn})
            log(f'family {fam} K{K}: present in {k} / {len(tn)} trained (null mean rate {np.mean(list(nullrate.values())):.2f}, p {pt_:.3g}); untrained {rec["families"][fam]["untrained_present"]} / {len(un)}')
        # pairwise term overlap (Jaccard of matched terms) between networks
        J_ = {}
        for a in names:
            for b in names:
                if a < b:
                    ma = sum(contains(P[b], t) for t in P[a]); mb = sum(contains(P[a], t) for t in P[b])
                    J_[f'{a}|{b}'] = float((ma + mb) / (len(P[a]) + len(P[b])))
        rec['pair_overlap'] = J_
        RES['recurrence'][f'K{K}'] = rec
    # ---------------------------------------------------------------- (5) transfer
    tn = [n for n in GOOD if n in names]; en = [n for n in tn if RJ[n]['edit_ok']]
    pairs = []
    for B_ in tn:
        nc = ZZ[B_]['netcall|test']; own = ZZ[B_]['prog_K16|test'] == nc; tru = ZZ[B_]['truth_K16|test'] == nc
        for A in tn:
            if A == B_: continue
            tr = ZZ[A]['prog_K16|test'] == nc; row = dict(A=A, B=B_, nat_own=float(own.mean()), nat_tr=float(tr.mean()), nat_truth=float(tru.mean()))
            if B_ in en:
                for e in EDITS:
                    bb = base[e]; nce = ZZ[B_][f'netcall|{e}']
                    _, _, ag_o, hit_o = edit_ind(nc[bb], nce, ZZ[B_]['prog_K16|test'][bb], ZZ[B_][f'prog_K16|{e}'])
                    nf, _, ag_t, hit_t = edit_ind(nc[bb], nce, ZZ[A]['prog_K16|test'][bb], ZZ[A][f'prog_K16|{e}'])
                    _, _, ag_u, hit_u = edit_ind(nc[bb], nce, ZZ[B_]['truth_K16|test'][bb], ZZ[B_][f'truth_K16|{e}'])
                    row[e] = dict(agree_own=float(ag_o.mean()), agree_tr=float(ag_t.mean()), agree_truth=float(ag_u.mean()),
                                  recall_own=float(hit_o.sum() / max(nf.sum(), 1)), recall_tr=float(hit_t.sum() / max(nf.sum(), 1)),
                                  recall_truth=float(hit_u.sum() / max(nf.sum(), 1)), net_flips=int(nf.sum()))
            pairs.append(row)
    def netboot(vals_fn, nets, B=2000, seed=4911):
        r_ = np.random.default_rng(seed); out = []
        for _ in range(B):
            dr = r_.choice(nets, len(nets), replace=True); vv = []
            for i, a in enumerate(dr):
                for j, b in enumerate(dr):
                    if i != j and a != b:
                        v = vals_fn(a, b)
                        if v is not None: vv.append(v)
            out.append(np.mean(vv) if vv else np.nan)
        return np.asarray(out)
    PI = {(p['A'], p['B']): p for p in pairs}
    tsum = {}
    def summarise(key, fn, nets):
        vals = [fn(a, b) for a in nets for b in nets if a != b and fn(a, b) is not None]
        bt = netboot(fn, nets); tsum[key] = dict(mean=float(np.mean(vals)), n_pairs=len(vals), ci_network_boot=ci(bt), p=bootp(bt))
    summarise('nat_loss_own_minus_transfer', lambda a, b: PI[(a, b)]['nat_own'] - PI[(a, b)]['nat_tr'], tn)
    summarise('nat_transfer_minus_truth', lambda a, b: PI[(a, b)]['nat_tr'] - PI[(a, b)]['nat_truth'], tn)
    for e in EDITS:
        g = lambda a, b, e=e: (PI[(a, b)][e]['agree_own'] - PI[(a, b)][e]['agree_tr']) if e in PI[(a, b)] else None
        summarise(f'{e}_agree_loss', g, tn)
        summarise(f'{e}_recall_loss', lambda a, b, e=e: (PI[(a, b)][e]['recall_own'] - PI[(a, b)][e]['recall_tr']) if e in PI[(a, b)] else None, tn)
        summarise(f'{e}_recall_transfer_minus_truth', lambda a, b, e=e: (PI[(a, b)][e]['recall_tr'] - PI[(a, b)][e]['recall_truth']) if e in PI[(a, b)] else None, tn)
        summarise(f'{e}_D_agree_loss_minus_nat_loss', lambda a, b, e=e: ((PI[(a, b)][e]['agree_own'] - PI[(a, b)][e]['agree_tr'])
                                                                        - (PI[(a, b)]['nat_own'] - PI[(a, b)]['nat_tr'])) if e in PI[(a, b)] else None, tn)
    RES['transfer'] = dict(pairs=pairs, summary=tsum, nets=tn, edit_nets=en)
    for k, v in tsum.items(): log(f'transfer {k}: {v["mean"] * 100:+.2f} pp, network-bootstrap CI [{v["ci_network_boot"][0] * 100:+.2f}, {v["ci_network_boot"][1] * 100:+.2f}], p {v["p"]:.3g} ({v["n_pairs"]} pairs)')
    (OUT / f'{tag}.json').write_text(json.dumps(clean(RES), indent=1))
    log(f'wrote {tag}.json')
    try:
        plot(RES, OUT / f'{tag}.png')
    except Exception as ex:
        log('plot failed', repr(ex))


def plot(RES, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    cmap = plt.get_cmap('tab10')
    for i, (n, o) in enumerate(RES['per_network'].items()):
        if n.startswith('untrained'): continue      # untrained redraws give one constant call: every program matches it (100 %)
        ks = [f'prog_K{K}' for K in KS if f'prog_K{K}' in o['surrogates']]
        b = [o['surrogates'][k]['bits'] for k in ks]; v = [o['surrogates'][k]['same_call'] * 100 for k in ks]
        st = dict(lw=2.4, color='k') if n == 'official_n64' else dict(lw=1.1, alpha=0.75, color=cmap(i % 10), ls='--' if n in DEGEN else '-')
        ax[0].plot(b, v, marker='o', ms=3, label=n.replace('_', ' ') + (' (57 % acc.)' if n in DEGEN else ''), **st)
    ax[0].set_xscale('log'); ax[0].set_xlabel('program description length (bits)'); ax[0].set_ylabel('same call as the network on test jets (%)')
    ax[0].set_title('Same recipe on 10 trained networks: similar fidelity-vs-size curves', fontsize=10); ax[0].set_ylim(58, 95); ax[0].legend(fontsize=6.5, ncol=2); ax[0].grid(alpha=0.3)
    T = RES['transfer']; en = T['edit_nets']
    if en:
        own = [np.mean([p['shift']['recall_own'] for p in T['pairs'] if p['B'] == b]) * 100 for b in en]
        tr = [np.mean([p['shift']['recall_tr'] for p in T['pairs'] if p['B'] == b]) * 100 for b in en]
        tu = [np.mean([p['shift']['recall_truth'] for p in T['pairs'] if p['B'] == b]) * 100 for b in en]
        x = np.arange(len(en)); w = 0.27
        ax[1].bar(x - w, own, w, label="network's own program (K16)"); ax[1].bar(x, tr, w, label="other networks' programs (mean)")
        ax[1].bar(x + w, tu, w, label='program fitted to truth labels', color='0.6')
        ax[1].set_xticks(x); ax[1].set_xticklabels([n.replace('_', '\n') for n in en], fontsize=7)
        ax[1].set_ylabel('network call changes reproduced (%)'); ax[1].set_title('Shift edit (deta + 0.06 rad): recall of the network\'s call changes', fontsize=10)
        ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3, axis='y')
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig); log('wrote', path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['net', 'agg']); ap.add_argument('--nets', default='')
    ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == 'net':
        for nm in a.nets.split(','):
            t = time.time(); run_net(nm, a.smoke); log(f'{nm} done in {time.time() - t:.0f} s')
    else:
        agg(a.smoke)
