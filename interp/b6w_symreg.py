"""B6w (Paper B, claims B4 / B6t; follows B6v): symbolic-regression and quantised-tree baselines for the blind rank-window
program, fidelity and edit agreement vs description length (DL), on four networks trained on the same 3-feature, 64-particle
hls4ml input: JEDI-linear n64 (jedi_core, fixed point), Deep Sets dsets-d64 s0 and transformer tf-2b-d32 s0 (torch, conda
env), PFN pfn_100-100-128_s0 (numpy forward).

Jets (interp.jedi_core.sample on the test file):
  F = sample(per_class 4000, seed 3701) = 20,000 jets (as B6v pass 1); permutation rng 0: first 2/3 fit (13,333), last 1/3 dev
      (6,667; MARS stopping and dev-same-call selection).
  T = sample(per_class 2000, seed 4701) minus every row of F (all surrogates are refitted here; nothing stored is reused).
Edits on T (padding never moved; observables recomputed from the edited particles):
  trans  Δη += 0.06 rad for every real particle
  m2h    trans, then all angles scaled per jet by sqrt(M2 / M2') about the stored axis (M0, M2 = ΣpT r² held)
  A13    swap of (Δη, Δφ) of pT ranks 13 and 14 (jets with >= 15 real particles)
  pt09   every pT x 0.9
  scr    per-particle azimuth scramble about the stored axis (r, pT kept)
  rcn    pT-weighted recentring: every real particle moved by -c, c = ΣpT (Δη, Δφ) / ΣpT (the pT centroid). ΣpT r² about the
         centroid is unchanged exactly; ΣpT r² about the stored axis drops by ΣpT |c|²; the invariant mass changes only through
         the small change of each particle's direction.
  controls  rdt  coherent 0.06 rad translation in a random direction per jet
            rpd  per-particle random 0.06 rad displacement
            rcr  coherent translation by |c| (the jet's own centroid offset) in a random direction (matched control for rcn)
            never-flip (agreement = 1 - network flip rate); shuffled-logit fits (prog K16, GBDT, gplearn).
Observables: RW = the 98 B6 observables (97 pT-rank-window sums of e11_mars_all.handbuilt + M0); HAND = 10 standard jet
  observables: mass, soft-drop mass, tau21, tau32, D2, N, ΣpT, ΣpT·r, ΣpT·r², ΣpT·r³ (r about the stored axis).
Surrogates (fitted to each network's logits on F-fit; call = argmax):
  prog      MARS on RW (B6 settings), up to 100 terms per logit; prefixes K 2-100 refitted by least squares (B6 K <= 30 ~ 13 kbit;
            K 45-100 extend the program to ~20-45 kbit).
  cart      CART on call / on logits, depth 2-10.
  gbdt32    HistGBDT on call / on logits, leaves {2,4,8,16} x lr {0.2,0.5} x iterations {5,10,25,60}, 32-bit thresholds.
  gbdtq     same grid on inputs quantised uniformly to 2^q levels (q = 4, 6, 8) over each observable's fit [0.5, 99.5] %
            range: a threshold costs q bits, plus 64 bits (range) per observable used.
  gpl_rw / gpl_hand   gplearn SymbolicRegressor per logit on the standardised RW / HAND observables, functions add sub mul div
            sqrt log abs neg max min, population 600, 15 generations, 4,000 fit rows, parsimony {0.0005, 0.002, 0.008, 0.03}.
  pysr_hand PySR 2.x on HAND (+ - * / sqrt log abs; 200 iterations, 15 populations, maxsize 40, 4,000 fit rows): hall of
            fame per logit, complexity caps 3-40 (one surrogate per cap: per logit the lowest-loss equation of complexity <= cap).
DL (bits; B6 rule: 32 bits per number + log2(#inputs) per structure element): MARS as B6; tree / GBDT node = log2(#inputs) +
  32 (q for gbdtq), leaf 32 x outputs (classifier tree leaf log2 5); symbolic expression: function node log2(#functions),
  variable leaf log2(#inputs), constant 32; + 64 per logit (target scale) + 64 per standardised observable used.
Metrics per (network, surrogate, edit): same call on unedited T; edit agreement per jet (surrogate flips iff the network flips,
  and to the same new call); Jaccard of the flipped sets. 95 % jet bootstrap (B 1000).
Tests: H0: at equal DL the program's edit agreement equals the best baseline's (all non-program families pooled; member with
  bits <= program bits; selected by dev same call ('devsc') or, as an optimistic bound for the baseline, by test agreement
  ('oracle')); H1: differs. Paired bootstrap CI and exact McNemar; Holm over networks x edits x programs (devsc, 6 edits).
  Secondary: per-family devsc comparison; lead specificity [lead on rcn] - [lead on rcr], [lead on trans] - [lead on rdt].
Modes (python -m interp.b6w_symreg <mode> [--model M]):
  logits --model jedi|pfn (jax env) / dsets|tf (AI_PY=conda)  -> $AI_RESULTS/jedi/b6w_logits_<m>.npz
  obs   (jax env)                                              -> b6w_obs.npz
  fit --model M  (venv /j-jepa-vol/aaron-interp/envs/symreg)   -> b6w_pred_<m>.npz / .json
  eval  (venv or jax env)                                      -> b6w.json, b6w_frontier.png
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
SMALL = Path(os.environ.get('AI_SMALL_MODELS', '/j-jepa-vol/aaron-interp/models/small'))
ENVS = Path('/j-jepa-vol/aaron-interp/envs')
T0 = time.time()
CL = ['g', 'q', 'W', 'Z', 't']
MODELS = dict(jedi='JEDI-linear n64', dsets='Deep Sets d64 s0', pfn='PFN 100-100-128 s0', tf='transformer 2b-d32 s0')
EDITS = ['trans', 'm2h', 'A13', 'pt09', 'scr', 'rcn']
CTRL = ['rdt', 'rpd', 'rcr']
SETS = ['test'] + EDITS + CTRL
ALLSETS = ['fit', 'dev'] + SETS
EDIT_LABEL = dict(trans='0.06 rad translation', m2h='translation, M0 and M2 held', A13='angle swap of pT ranks 13|14',
                  pt09='all pT x 0.9', scr='per-particle azimuth scramble', rcn='pT-weighted recentring',
                  rdt='control: random-direction 0.06 rad translation', rpd='control: per-particle random 0.06 rad',
                  rcr='control: random-direction shift by |centroid|')
PROG_K = (2, 4, 8, 16, 30, 45, 60, 80, 100)
MATCH_K = (4, 8, 16, 30, 60, 100)     # K 4, 8 (1.6-3.4 kbit) are the sizes where the symbolic-regression surrogates lie
BASEFAM = ['cart', 'gbdt32', 'gbdtq', 'gpl_rw', 'gpl_hand', 'pysr_hand']
HAND = ['mass', 'sd_mass', 'tau21', 'tau32', 'D2', 'n_particles', 'sum_pt']
HAND_KEYS = HAND + ['ΣpT·r', 'ΣpT·r²', 'ΣpT·r³']
GPL_FUN = ('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'max', 'min')


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


# ------------------------------------------------------------------ jets and edits (numpy only; identical in every env)
def indices(per_class, seed):
    from jetdistill.config import RESULTS
    y = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y']); rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in range(5)]))


def build():
    from interp.jedi_core import sample
    xF, yF, _ = sample(4000, 3701); xT, yT, _ = sample(2000, 4701)
    keep = ~np.isin(indices(2000, 4701), indices(4000, 3701)); xT, yT = np.asarray(xT[keep], np.float32), yT[keep]
    perm = np.random.default_rng(0).permutation(len(yF)); nf = 2 * len(yF) // 3
    x = dict(fit=np.asarray(xF[perm[:nf]]), dev=np.asarray(xF[perm[nf:]]), test=xT)
    y = dict(fit=yF[perm[:nf]], dev=yF[perm[nf:]], test=yT)
    real = xT[..., 0] > 0; nreal = real.sum(1); rf = real.astype(np.float32)
    M2 = lambda a, c=None: (a[..., 0] * ((a[..., 1] - (0 if c is None else c[:, 0:1])) ** 2 + (a[..., 2] - (0 if c is None else c[:, 1:2])) ** 2)).sum(1).astype(np.float64)
    e = xT.copy(); e[..., 1] += np.float32(0.06) * rf; x['trans'] = e
    s = np.sqrt(M2(xT) / np.maximum(M2(e), 1e-12)); m = e.copy(); m[..., 1:] = (e[..., 1:] * s[:, None, None]).astype(np.float32); x['m2h'] = m
    J13 = np.flatnonzero(nreal >= 15); a = xT[J13].copy(); a[:, [13, 14], 1:] = xT[J13][:, [14, 13], 1:]; x['A13'] = a
    p = xT.copy(); p[..., 0] *= np.float32(0.9); x['pt09'] = p
    rng = np.random.default_rng(4703); r = np.hypot(xT[..., 1], xT[..., 2]); th = rng.uniform(-np.pi, np.pi, r.shape)
    sc = xT.copy(); sc[..., 1] = np.where(real, r * np.cos(th), 0); sc[..., 2] = np.where(real, r * np.sin(th), 0); x['scr'] = sc
    pt = xT[..., 0].astype(np.float64); c = np.stack([(pt * xT[..., 1]).sum(1), (pt * xT[..., 2]).sum(1)], 1) / np.maximum(pt.sum(1), 1e-12)[:, None]
    q = xT.copy(); q[..., 1] -= (c[:, 0:1] * rf).astype(np.float32); q[..., 2] -= (c[:, 1:2] * rf).astype(np.float32); x['rcn'] = q
    ph = np.random.default_rng(4704).uniform(0, 2 * np.pi, len(xT)); d = xT.copy()
    d[..., 1] += (0.06 * np.cos(ph))[:, None].astype(np.float32) * rf; d[..., 2] += (0.06 * np.sin(ph))[:, None].astype(np.float32) * rf; x['rdt'] = d
    th2 = np.random.default_rng(4705).uniform(0, 2 * np.pi, real.shape); w = xT.copy()
    w[..., 1] += (0.06 * np.cos(th2)).astype(np.float32) * rf; w[..., 2] += (0.06 * np.sin(th2)).astype(np.float32) * rf; x['rpd'] = w
    cn = np.hypot(c[:, 0], c[:, 1]); ph3 = np.random.default_rng(4706).uniform(0, 2 * np.pi, len(xT)); v = xT.copy()
    v[..., 1] += (cn * np.cos(ph3))[:, None].astype(np.float32) * rf; v[..., 2] += (cn * np.sin(ph3))[:, None].astype(np.float32) * rf; x['rcr'] = v
    for k in list(x): x[k] = np.ascontiguousarray(x[k], dtype=np.float32)
    base = {k: np.arange(len(yT)) for k in SETS}; base['A13'] = J13
    for k in SETS: y[k] = yT[base[k]]
    cq = np.stack([(pt * x['rcn'][..., 1]).sum(1), (pt * x['rcn'][..., 2]).sum(1)], 1) / np.maximum(pt.sum(1), 1e-12)[:, None]
    info = dict(N_fit=int(nf), N_dev=int(len(yF) - nf), N_test=int(len(yT)), removed_from_test=int((~keep).sum()), N_A13=int(len(J13)),
                m2h_scale_mean=float(s.mean()), rcn_offset_rad=dict(mean=float(cn.mean()), median=float(np.median(cn)), p90=float(np.percentile(cn, 90))),
                rcn_residual_centroid_rad=float(np.abs(cq).max()),
                rcn_dM2_axis_rel=float(np.mean((M2(x['rcn']) - M2(xT)) / np.maximum(M2(xT), 1e-12))),
                rcn_dM2_centroid_rel_maxabs=float(np.max(np.abs(M2(x['rcn'], cq) - M2(xT, c)) / np.maximum(M2(xT, c), 1e-12))))
    log('jets', json.dumps(info))
    return dict(x=x, y=y, base=base, info=info)


# ------------------------------------------------------------------ mode logits
def mode_logits(model):
    B = build(); X = B['x']
    if model == 'jedi':
        from interp.jedi_core import Jedi
        net = Jedi(64); L = {s: net.run(X[s])['logits'] for s in ALLSETS}
    elif model == 'pfn':
        from interp.x9_efn_pfn import Flow
        f = Flow('pfn_100-100-128_s0'); L = {s: f.run(X[s])['logits'] for s in ALLSETS}
    else:
        import torch
        from interp.x2_small_models import load_net, normalise
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
        net, arch = load_net(SMALL / dict(dsets='dsets-d64-n64', tf='tf-2b-d32-n64')[model] / 'seed0'); L = {}
        for s in ALLSETS:
            xn, mask = normalise(X[s], arch); r = []
            with torch.no_grad():
                for i in range(0, len(xn), 2000):
                    r.append(net(torch.from_numpy(xn[i:i + 2000]), torch.from_numpy(mask[i:i + 2000])).double().numpy())
            L[s] = np.concatenate(r)
    acc = {s: float((L[s].argmax(1) == B['y'][s]).mean()) for s in ('fit', 'dev', 'test')}
    log(f'{model}: accuracy fit {acc["fit"] * 100:.2f} % dev {acc["dev"] * 100:.2f} % test {acc["test"] * 100:.2f} % (N {len(B["y"]["test"])})')
    flips = {}
    for s in EDITS + CTRL:
        cb = L['test'].argmax(1)[B['base'][s]]; ca = L[s].argmax(1); flips[s] = float((ca != cb).mean())
        log(f'{model}: edit {s} changes {flips[s] * 100:.2f} % of calls (N {len(ca)})')
    np.savez_compressed(OUT / f'b6w_logits_{model}.npz', **{s: L[s].astype(np.float32) for s in ALLSETS}, **{f'y_{s}': B['y'][s] for s in ALLSETS},
                        J13=B['base']['A13'])
    (OUT / f'b6w_logits_{model}.json').write_text(json.dumps(dict(model=model, acc=acc, flips=flips, info=B['info']), indent=1))
    log(f'wrote {OUT}/b6w_logits_{model}.npz')


# ------------------------------------------------------------------ mode obs
def mode_obs():
    from interp.b6_pareto import features
    from jetdistill.observables import compute
    B = build(); out = {}; keys = None
    for s in ALLSETS:
        t = time.time(); xx = B['x'][s]
        out[f'rw_{s}'], keys = features(xx); out[f'rw_{s}'] = out[f'rw_{s}'].astype(np.float64)
        q = compute(np.asarray(xx, np.float64), 64, ids=HAND)
        pt = np.asarray(xx[..., 0], np.float64); r = np.hypot(xx[..., 1], xx[..., 2]).astype(np.float64)
        H = [np.asarray(q[k], np.float64) for k in HAND] + [(pt * r ** n).sum(1) for n in (1, 2, 3)]
        out[f'hand_{s}'] = np.stack(H, 1); log(f'observables {s}: {len(xx)} jets, {time.time() - t:.0f} s')
    np.savez_compressed(OUT / 'b6w_obs.npz', **out, rw_keys=np.array(keys), hand_keys=np.array(HAND_KEYS))
    log(f'wrote {OUT}/b6w_obs.npz ({len(keys)} RW, {len(HAND_KEYS)} hand observables)')


# ------------------------------------------------------------------ mode fit
class Store:
    def __init__(self, model, nc):
        self.model, self.arr, self.meta, self.nc = model, {}, [], nc

    def add(self, name, family, method, bits, calls, extra=None):
        for s, c in calls.items(): self.arr[f'{name}|{s}'] = np.asarray(c, np.int8)
        sd = float((calls['dev'] == self.nc['dev']).mean()); sc = float((calls['test'] == self.nc['test']).mean())
        self.meta.append(dict(name=name, family=family, method=method, bits=float(bits), same_call_dev=sd, same_call_test=sc, **(extra or {})))
        log(f'{self.model}: {name:28s} ({family:9s}) bits {bits:9.0f}  same call dev {sd * 100:6.2f} test {sc * 100:6.2f}')


def fit_prog(X, keys, Lf, Ld, kmax, Ks, tag):
    from jetdistill import mars
    from interp.e11_mars_all import knots_of
    from interp.b6_pareto import term_bits
    Q = {s: {k: X[s][:, i] for i, k in enumerate(keys)} for s in ('fit', 'dev')}
    knots = knots_of(Q['fit'], keys); NF = len(Lf); sel = np.random.default_rng(1).choice(NF, min(6000, NF), replace=False)
    mars.MAX_TERMS, mars.STALL = kmax, 10 ** 6; fits = []
    for k in range(5):
        t = time.time(); c = 1.0 - min(Lf[:, k].min(), Ld[:, k].min()) + 0.5 * Lf[:, k].std()
        terms, _, path = mars.fit_neuron(dict(fit=Lf[:, k] + c, dev=Ld[:, k] + c), Q, keys, knots, sel)
        fits.append((terms, c)); log(f'{tag} MARS logit {CL[k]}: {len(terms)} terms kept (dev R² path end {path[-1] if path else None}); {time.time() - t:.0f} s')
    progs = {}
    for K in Ks:
        parts = []
        for k, (terms, c) in enumerate(fits):
            tr = terms[:K]; parts.append((tr, mars._lstsq(tr, Q['fit'], Lf[:, k] + c, NF), c))
        nterm = sum(len(p[0]) for p in parts)
        if any(nterm == v[2] for v in progs.values()): continue
        progs[K] = (parts, sum(32 + sum(sum(term_bits(t, len(keys))) for t in tr) for tr, _, _ in parts), nterm)

    def predict(parts, Xa):
        Qd = {k: Xa[:, i] for i, k in enumerate(keys)}; J = len(Xa)
        return np.stack([np.stack([np.ones(J)] + [mars.term_value(t, Qd) for t in tr], 1) @ co - c for tr, co, c in parts], 1)
    return progs, predict


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


def gpl_patch():
    from gplearn.genetic import SymbolicRegressor
    if not hasattr(SymbolicRegressor, '_validate_data'):      # removed in scikit-learn 1.6; gplearn 0.4.2 still calls it
        from sklearn.utils.validation import validate_data
        SymbolicRegressor._validate_data = lambda self, *a, **k: validate_data(self, *a, **k)
    return SymbolicRegressor


def gpl_bits(prog, nin):
    b, used = 0.0, set()
    for node in prog.program:
        if isinstance(node, (int, np.integer)): b += np.log2(max(nin, 2)); used.add(int(node))
        elif isinstance(node, (float, np.floating)): b += 32
        else: b += np.log2(len(GPL_FUN))
    return b, used


def fit_gplearn(Xs, Lf, sel, pars, seed, tag, pop=600, gens=15):
    """Xs standardised features per set; one SymbolicRegressor per logit on standardised targets -> (predict-fn, bits, info)"""
    SR = gpl_patch(); nin = Xs['fit'].shape[1]; mu, sd = Lf.mean(0), Lf.std(0); ms, bits, used, lens = [], 0.0, set(), []
    for k in range(5):
        t = time.time()
        m = SR(population_size=pop, generations=gens, tournament_size=20, function_set=GPL_FUN, parsimony_coefficient=pars,
               init_depth=(2, 5), max_samples=1.0, metric='mse', random_state=seed + k, n_jobs=1, verbose=0)
        m.fit(Xs['fit'][sel], (Lf[sel, k] - mu[k]) / sd[k]); b, u = gpl_bits(m._program, nin); bits += b + 64; used |= u; ms.append(m)
        m._programs = None      # every generation's population is kept by gplearn (pop 3000 x 30 gen per fit: OOM at 2 GiB); predict needs _program only
        lens.append(int(m._program.length_))
        log(f'{tag} parsimony {pars} logit {CL[k]}: length {m._program.length_}, fit R² {1 - m._program.raw_fitness_:.3f}, {time.time() - t:.0f} s: {str(m._program)[:160]}')
    bits += 64 * len(used)
    pred = lambda Z: np.stack([np.nan_to_num(m.predict(Z)) * sd[k] + mu[k] for k, m in enumerate(ms)], 1)
    return pred, bits, dict(lengths=lens, n_used=len(used), programs=[str(m._program) for m in ms])


def sympy_bits(expr, nin, nops):
    import sympy
    b = 0.0
    for e in sympy.preorder_traversal(expr):
        if e.is_Symbol: b += np.log2(max(nin, 2))
        elif e.is_Number: b += 32
        elif e.is_Add or e.is_Mul: b += (len(e.args) - 1) * np.log2(nops)
        else: b += np.log2(nops)
    return b


def fit_pysr(Xs, Lf, sel, log_):
    os.environ.setdefault('JULIA_DEPOT_PATH', str(ENVS / 'julia' / 'depot')); os.environ.setdefault('PYTHON_JULIAPKG_PROJECT', str(ENVS / 'julia' / 'project'))
    from pysr import PySRRegressor
    mu, sd = Lf.mean(0), Lf.std(0); per = []
    for k in range(5):
        t = time.time()
        m = PySRRegressor(niterations=200, populations=15, maxsize=40, binary_operators=['+', '-', '*', '/'], unary_operators=['sqrt', 'log', 'abs'],
                          parallelism='serial', deterministic=True, random_state=k, progress=False, verbosity=0, temp_equation_file=True,
                          timeout_in_seconds=900)
        m.fit(Xs['fit'][sel], (Lf[sel, k] - mu[k]) / sd[k]); eq = m.equations_
        per.append((m, eq)); log_(f'pysr logit {CL[k]}: {len(eq)} hall-of-fame equations, max complexity {int(eq["complexity"].max())}, {time.time() - t:.0f} s')
    out = []
    for cap in (3, 5, 8, 12, 17, 23, 30, 40):
        idx = []; bits = 0.0; nin = Xs['fit'].shape[1]
        for m, eq in per:
            ok = eq[eq['complexity'] <= cap]
            if not len(ok): idx = None; break
            i = int(ok['loss'].idxmin()); idx.append(i); bits += sympy_bits(m.sympy(i), nin, 7) + 64
        if idx is None: continue
        used = set()
        for (m, eq), i in zip(per, idx): used |= {int(str(s)[1:]) for s in m.sympy(i).free_symbols}
        bits += 64 * len(used)
        pred = lambda Z, idx=idx: np.stack([np.nan_to_num(m.predict(Z, index=i)) * sd[k] + mu[k] for k, ((m, eq), i) in enumerate(zip(per, idx))], 1)
        out.append((cap, pred, bits, [str(m.sympy(i)) for (m, eq), i in zip(per, idx)]))
    return out


def mode_fit(model, skip_pysr=False):
    import warnings
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from interp.b6_pareto import tree_bits, hgb_bits
    warnings.filterwarnings('ignore')
    Z = np.load(OUT / f'b6w_logits_{model}.npz'); O = np.load(OUT / 'b6w_obs.npz')
    L = {s: Z[s].astype(np.float64) for s in ALLSETS}; nc = {s: L[s].argmax(1) for s in ALLSETS}; st = Store(model, nc)
    RW = {s: O[f'rw_{s}'] for s in ALLSETS}; HD = {s: O[f'hand_{s}'] for s in ALLSETS}; keys = [str(k) for k in O['rw_keys']]
    F = len(keys); Lf, Ld, cf = L['fit'], L['dev'], nc['fit']
    have = {}
    for p in ('gplearn', 'pysr'):
        try:
            if p == 'pysr':
                if skip_pysr or 'pysr ok' not in (ENVS / 'symreg' / 'STATUS').read_text(): raise RuntimeError('setup reported no PySR')
            __import__(p); have[p] = True
        except Exception as ex:  # noqa: BLE001
            have[p] = f'missing: {ex}'[:200]
    log(f'{model}: packages {have}; {F} RW observables, {HD["fit"].shape[1]} hand observables; fit {len(Lf)} dev {len(Ld)} test {len(L["test"])}')
    t = time.time()
    progs, predict = fit_prog(RW, keys, Lf, Ld, max(PROG_K), PROG_K, 'prog')
    for K, (parts, bits, nterm) in progs.items():
        st.add(f'prog{K}', 'prog', 'MARS on rank-window observables -> logits', bits, {s: predict(parts, RW[s]).argmax(1) for s in ALLSETS}, dict(K=K, terms=nterm))
    pr = np.random.default_rng(4706).permutation(len(Lf)); pdv = np.random.default_rng(4707).permutation(len(Ld))
    sp, spred = fit_prog(RW, keys, Lf[pr], Ld[pdv], 16, (16,), 'shuffled')
    for K, (parts, bits, nterm) in sp.items():
        st.add('shuf_prog16', 'shuffled', 'MARS K16 on row-shuffled logits', bits, {s: spred(parts, RW[s]).argmax(1) for s in ALLSETS})
    log(f'programs {time.time() - t:.0f} s'); t = time.time()
    for d in range(2, 11):
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(RW['fit'], cf)
        st.add(f'tree_d{d}', 'cart', 'CART on call', tree_bits(m.tree_, F, 1, True), {s: m.predict(RW[s]) for s in ALLSETS}, dict(depth=d))
        r = DecisionTreeRegressor(max_depth=d, random_state=0).fit(RW['fit'], Lf)
        st.add(f'treereg_d{d}', 'cart', 'CART on logits', tree_bits(r.tree_, F, 5), {s: r.predict(RW[s]).argmax(1) for s in ALLSETS}, dict(depth=d))
    log(f'cart {time.time() - t:.0f} s'); t = time.time()
    for q in (32, 8, 6, 4):
        if q == 32: Xq = RW; fam = 'gbdt32'
        else:
            qf = quantiser(RW['fit'], q); Xq = {s: qf(RW[s]) for s in ALLSETS}; fam = 'gbdtq'
        bf = (lambda m: hgb_bits(m, F)) if q == 32 else (lambda m, q=q: hgb_bits_q(m, F, q))
        for lv in (2, 4, 8, 16):
            for lr in (0.2, 0.5):
                for it in (5, 10, 25, 60):
                    kw = dict(max_leaf_nodes=lv, max_iter=it, learning_rate=lr, early_stopping=False, random_state=0)
                    m = HistGradientBoostingClassifier(**kw).fit(Xq['fit'], cf)
                    st.add(f'gbdt{q}_L{lv}_T{it}_lr{lr}', fam, f'GBDT on call, {q}-bit thresholds', bf(m), {s: m.predict(Xq[s]) for s in ALLSETS}, dict(q=q, leaves=lv, iters=it, lr=lr))
                    rs = [HistGradientBoostingRegressor(**kw).fit(Xq['fit'], Lf[:, k]) for k in range(5)]
                    st.add(f'gbdtreg{q}_L{lv}_T{it}_lr{lr}', fam, f'GBDT on logits, {q}-bit thresholds', sum(bf(r) for r in rs),
                           {s: np.stack([r.predict(Xq[s]) for r in rs], 1).argmax(1) for s in ALLSETS}, dict(q=q, leaves=lv, iters=it, lr=lr))
        log(f'gbdt q{q}: {time.time() - t:.0f} s')
    rs = [HistGradientBoostingRegressor(max_leaf_nodes=8, max_iter=25, learning_rate=0.5, early_stopping=False, random_state=0).fit(RW['fit'], Lf[pr, k]) for k in range(5)]
    st.add('shuf_gbdtreg_L8_T25', 'shuffled', 'GBDT on row-shuffled logits', sum(hgb_bits(r, F) for r in rs), {s: np.stack([r.predict(RW[s]) for r in rs], 1).argmax(1) for s in ALLSETS})
    t = time.time()
    sel = np.random.default_rng(4708).choice(len(Lf), 4000, replace=False)
    std = lambda D: (lambda mu, sd: {s: np.nan_to_num((D[s] - mu) / sd) for s in ALLSETS})(D['fit'].mean(0), np.where(D['fit'].std(0) > 0, D['fit'].std(0), 1.0))
    RWs, HDs = std(RW), std(HD)
    if have['gplearn'] is True:
        for pool, Xs in (('rw', RWs), ('hand', HDs)):
            for pars in (0.0005, 0.002, 0.008, 0.03):
                pred, bits, inf = fit_gplearn(Xs, Lf, sel, pars, 4710, f'gplearn {pool}')
                st.add(f'gpl_{pool}_p{pars}', f'gpl_{pool}', f'gplearn on {pool} observables -> logits', bits, {s: pred(Xs[s]).argmax(1) for s in ALLSETS}, dict(parsimony=pars, **inf))
            log(f'gplearn {pool}: {time.time() - t:.0f} s')
        pred, bits, inf = fit_gplearn({'fit': RWs['fit'][pr]}, Lf, sel, 0.002, 4720, 'gplearn shuffled')
        # fitted on row-shuffled logits: features of row pr[i] paired with the logits of row i
        st.add('shuf_gpl_rw', 'shuffled', 'gplearn (rw, parsimony 0.002) on row-shuffled logits', bits, {s: pred(RWs[s]).argmax(1) for s in ALLSETS}, inf)
    if have['pysr'] is True:
        try:
            for cap, pred, bits, eqs in fit_pysr(HDs, Lf, sel, log):
                st.add(f'pysr_hand_c{cap}', 'pysr_hand', 'PySR on hand observables -> logits', bits, {s: pred(HDs[s]).argmax(1) for s in ALLSETS}, dict(cap=cap, equations=eqs))
        except Exception as ex:  # noqa: BLE001
            have['pysr'] = f'fit failed: {ex}'[:300]; log('PySR fit failed:', ex)
    np.savez_compressed(OUT / f'b6w_pred_{model}.npz', **st.arr)
    (OUT / f'b6w_pred_{model}.json').write_text(json.dumps(dict(model=model, packages=have, rw_observables=keys, hand_observables=HAND_KEYS, models=st.meta),
                                                          indent=1, ensure_ascii=False, default=float))
    log(f'wrote {OUT}/b6w_pred_{model}.npz/.json ({len(st.meta)} surrogates)')


# ------------------------------------------------------------------ mode eval
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def mode_eval(Bn=1000):
    from scipy.stats import binomtest
    res = dict(procedure=__doc__, models={}, matched=[], specificity=[])
    for model in MODELS:
        pz = OUT / f'b6w_pred_{model}.npz'
        if not pz.exists(): log(f'missing {pz}'); continue
        Z = np.load(OUT / f'b6w_logits_{model}.npz'); nc = {s: Z[s].argmax(1) for s in SETS}; J13 = Z['J13']
        base = {s: (J13 if s == 'A13' else np.arange(len(nc['test']))) for s in SETS}
        res['info'] = json.loads((OUT / f'b6w_logits_{model}.json').read_text())['info']
        P = dict(np.load(pz)); pj = json.loads((OUT / f'b6w_pred_{model}.json').read_text()); meta = pj['models']
        if (OUT / f'b6w_pysr_{model}.json').exists():
            ps = json.loads((OUT / f'b6w_pysr_{model}.json').read_text()); pj['packages']['pysr'] = ps['pysr']
            meta = [m for m in meta if m['family'] != 'pysr_hand'] + ps['models']; P.update(dict(np.load(OUT / f'b6w_pysr_{model}.npz')))
        gparts = [model] if (OUT / f'b6w_gpl_{model}.json').exists() else [f'{model}_{q}' for q in ('rw', 'hand', 'shuf')]
        if all((OUT / f'b6w_gpl_{g}.json').exists() for g in gparts):      # larger-budget gplearn runs replace the fit-mode ones
            gs = sum((json.loads((OUT / f'b6w_gpl_{g}.json').read_text())['models'] for g in gparts), [])
            meta = [m for m in meta if not (m['family'] in ('gpl_rw', 'gpl_hand') or m['name'] == 'shuf_gpl_rw')] + gs
            for g in gparts: P.update(dict(np.load(OUT / f'b6w_gpl_{g}.npz')))
            pj['packages']['gplearn_budget'] = 'population 3000, 30 generations'
        else:
            pj['packages']['gplearn_budget'] = 'population 600, 15 generations (larger-budget run missing)'
        N = len(nc['test']); rng = np.random.default_rng(7); cnt = {}
        for s in SETS:
            n = len(nc[s]); cnt.setdefault(n, np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(Bn)]).astype(np.float32))
        C = {s: cnt[len(nc[s])] for s in SETS}
        names = [m['name'] for m in meta]; rows = {m['name']: dict(m) for m in meta}
        yT = Z['y_test']; net = dict(acc_test=float((nc['test'] == yT).mean()), N_test=N, flip={}, N={})
        ba = C['test'] @ (nc['test'] == yT).astype(np.float32) / N; net['acc_test_ci'] = [float(np.percentile(ba, 2.5)), float(np.percentile(ba, 97.5))]
        SC = np.stack([(P[f'{n}|test'] == nc['test']) for n in names]).astype(np.float32); bsc = C['test'] @ SC.T / N
        for i, n in enumerate(names):
            rows[n]['same_call'] = [float(SC[i].mean()), float(np.percentile(bsc[:, i], 2.5)), float(np.percentile(bsc[:, i], 97.5))]
        AGJ = {}
        for e in EDITS + CTRL:
            bi = base[e]; nb, na = nc['test'][bi], nc[e]; nf = na != nb; ne = len(nf); bf = C[e] @ nf.astype(np.float32) / ne
            net['flip'][e] = [float(nf.mean()), float(np.percentile(bf, 2.5)), float(np.percentile(bf, 97.5))]; net['N'][e] = int(ne)
            A = np.zeros((len(names), ne), np.float32)
            for i, n in enumerate(names):
                sb, sa = P[f'{n}|test'][bi], P[f'{n}|{e}']; sf = sa != sb
                A[i] = (nf == sf) & (~nf | (sa == na))
                rows[n].setdefault('edits', {})[e] = dict(flip=float(sf.mean()), jaccard=float((nf & sf).sum() / max((nf | sf).sum(), 1)),
                                                          recall=float((sf & (sa == na))[nf].mean()) if nf.any() else None)
            bA = C[e] @ A.T / ne
            for i, n in enumerate(names):
                rows[n]['edits'][e]['agree'] = [float(A[i].mean()), float(np.percentile(bA[:, i], 2.5)), float(np.percentile(bA[:, i], 97.5))]
            AGJ[e] = A
        fam = np.array([rows[n]['family'] for n in names]); bits = np.array([rows[n]['bits'] for n in names])
        ipk = {rows[n]['K']: i for i, n in enumerate(names) if rows[n]['family'] == 'prog'}
        for K in MATCH_K:
            if K not in ipk: continue
            i = ipk[K]
            groups = {'pooled': [j for j in range(len(names)) if fam[j] in BASEFAM]}
            groups.update({f: [j for j in range(len(names)) if fam[j] == f] for f in BASEFAM})
            for g, cand in groups.items():
                el = [j for j in cand if bits[j] <= bits[i] * 1.0001]
                if not el: continue
                for rule in (('devsc', 'oracle') if g == 'pooled' else ('devsc',)):
                    for e in ['same_call'] + EDITS + CTRL:
                        if rule == 'devsc': j = max(el, key=lambda j: rows[names[j]]['same_call_dev'])
                        else: j = max(el, key=lambda j: SC[j].mean() if e == 'same_call' else AGJ[e][j].mean())
                        a_, b_ = (SC[i], SC[j]) if e == 'same_call' else (AGJ[e][i], AGJ[e][j]); Ce = C['test'] if e == 'same_call' else C[e]
                        dd = Ce @ (a_ - b_) / len(a_); n10, n01 = int(((a_ > 0) & (b_ == 0)).sum()), int(((a_ == 0) & (b_ > 0)).sum())
                        pm = float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
                        res['matched'].append(dict(model=model, program=names[i], K=K, group=g, rule=rule, baseline=names[j], base_family=fam[j], edit=e,
                                                   prog_bits=float(bits[i]), base_bits=float(bits[j]), prog=float(a_.mean()), base=float(b_.mean()),
                                                   diff=float((a_ - b_).mean()), ci=[float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                                                   prog_only=n10, base_only=n01, p_mcnemar=pm))
                        if g == 'pooled' and rule == 'devsc' and e in ('rcn', 'trans'):
                            ctl = dict(rcn='rcr', trans='rdt')[e]; c_, d_ = AGJ[ctl][i], AGJ[ctl][j]
                            sp = (C[e] @ (a_ - b_) / len(a_)) - (C[ctl] @ (c_ - d_) / len(c_)); obs = float((a_ - b_).mean() - (c_ - d_).mean())
                            res['specificity'].append(dict(model=model, program=names[i], edit=e, control=ctl, baseline=names[j], lead_minus_control=obs,
                                                           ci=[float(np.percentile(sp, 2.5)), float(np.percentile(sp, 97.5))],
                                                           p_boot=float(min(1.0, 2 * min((sp <= 0).mean(), (sp >= 0).mean()) + 1.0 / Bn))))
        res['models'][model] = dict(label=MODELS[model], network=net, packages=pj['packages'], surrogates=list(rows.values()))
        log(f'{model}: network acc test {net["acc_test"] * 100:.2f} % {np.round(np.array(net["acc_test_ci"]) * 100, 2)} (N {N}); flips ' +
            ', '.join(f'{e} {v[0] * 100:.1f} %' for e, v in net['flip'].items()))
        for n in names:
            r = rows[n]
            log(f"  {n:28s} {r['family']:9s} log2 bits {np.log2(r['bits']):5.1f} same call {r['same_call'][0] * 100:6.2f} | " +
                ' '.join(f"{e} {r['edits'][e]['agree'][0] * 100:5.1f}" for e in EDITS + CTRL))
    prim = [t for t in res['matched'] if t['group'] == 'pooled' and t['rule'] == 'devsc' and t['edit'] in EDITS]
    for t, q in zip(prim, holm([t['p_mcnemar'] for t in prim])): t['p_holm'] = float(q)
    res['n_holm_primary'] = len(prim)
    sec = [t for t in res['matched'] if t['group'] != 'pooled' and t['edit'] in EDITS]
    for t, q in zip(sec, holm([t['p_mcnemar'] for t in sec])): t['p_holm_family'] = float(q)
    res['n_holm_family'] = len(sec)
    for t in res['matched']:
        if t['rule'] == 'devsc' and t['group'] in ('pooled', 'gpl_rw', 'gpl_hand', 'gbdtq', 'pysr_hand'):
            log(f"matched {t['model']:5s} {t['program']:7s} vs {t['group']:9s} {t['baseline']:28s} {t['edit']:9s}: {t['prog'] * 100:5.1f} vs {t['base'] * 100:5.1f} = "
                f"{t['diff'] * 100:+6.2f} [{t['ci'][0] * 100:+.2f}, {t['ci'][1] * 100:+.2f}] McNemar {t['p_mcnemar']:.3g}" + (f" Holm {t['p_holm']:.3g}" if 'p_holm' in t else ''))
    for t in res['specificity']:
        log(f"specificity {t['model']:5s} {t['program']:7s} {t['edit']}-{t['control']}: {t['lead_minus_control'] * 100:+.2f} [{t['ci'][0] * 100:+.2f}, {t['ci'][1] * 100:+.2f}] p {t['p_boot']:.3g}")
    (OUT / 'b6w.json').write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float)); log(f'wrote {OUT}/b6w.json')
    try:
        plot(res)
    except Exception as ex:  # noqa: BLE001
        log('plot failed:', ex)


STYLE = dict(prog=('program (MARS on rank windows)', '#1c5cab', 'o'), cart=('CART', '#2a9d3c', '^'), gbdt32=('GBDT, 32-bit cuts', '#c0392b', 'D'),
             gbdtq=('GBDT, 4/6/8-bit cuts', '#e07b00', 'd'), gpl_rw=('gplearn on rank windows', '#7b4fa0', 's'),
             gpl_hand=('gplearn on standard observables', '#d35fb7', 'P'), pysr_hand=('PySR on standard observables', '#555555', 'X'))


def plot(res, path=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mods = [m for m in MODELS if m in res['models']]; rowsE = ['same_call', 'trans', 'm2h']
    fig, axes = plt.subplots(len(rowsE), len(mods), figsize=(3.9 * len(mods), 3.2 * len(rowsE)), squeeze=False)
    for c, mk in enumerate(mods):
        R = res['models'][mk]; S = R['surrogates']
        for r_, e in enumerate(rowsE):
            ax = axes[r_, c]
            for f, (lab, col, mkr) in STYLE.items():
                rr = sorted([r for r in S if r['family'] == f], key=lambda r: r['bits'])
                if not rr: continue
                v = np.array([r['same_call'] if e == 'same_call' else r['edits'][e]['agree'] for r in rr]) * 100; x = np.array([r['bits'] for r in rr]) / 1000
                ax.step(x, np.maximum.accumulate(v[:, 0]), where='post', color=col, lw=1.6 if f == 'prog' else 1.1, label=lab)
                ax.plot(x, v[:, 0], mkr, color=col, ms=3, alpha=0.6)
            if e != 'same_call':
                ch = (1 - R['network']['flip'][e][0]) * 100; ax.axhline(ch, color='#777777', lw=1, ls='--')
                ax.text(0.98, ch, 'never-flip', transform=ax.get_yaxis_transform(), fontsize=7, color='#555555', va='bottom', ha='right')
            ax.set_xscale('log'); ax.grid(alpha=0.3, lw=0.5); ax.tick_params(labelsize=7.5)
            if r_ == 0: ax.set_title(f"{R['label']}\nnetwork accuracy {R['network']['acc_test'] * 100:.1f} %", fontsize=9, loc='left')
            if c == 0: ax.set_ylabel({'same_call': 'same call as network (%)', 'trans': 'agreement under\n0.06 rad translation (%)',
                                      'm2h': 'agreement under translation\nwith M0, M2 held (%)'}[e], fontsize=8.5)
            if r_ == len(rowsE) - 1: ax.set_xlabel('description length (kbit)', fontsize=8.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=8, frameon=False)
    fig.suptitle('Surrogate fidelity vs size: lines = best surrogate of each family at or below each description length', fontsize=10, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.06, 1, 0.97)); p = path or OUT / 'b6w_frontier.png'; fig.savefig(p, dpi=130); plt.close(fig); log(f'wrote {p}')


def mode_pysr(model):
    """PySR surrogates on the hand observables (same standardisation and 4,000 fit rows as fit) -> b6w_pysr_<model>.npz/.json,
    merged into the surrogate list by eval"""
    Z = np.load(OUT / f'b6w_logits_{model}.npz'); O = np.load(OUT / 'b6w_obs.npz')
    L = {s: Z[s].astype(np.float64) for s in ALLSETS}; nc = {s: L[s].argmax(1) for s in ALLSETS}; st = Store(model, nc)
    HD = {s: O[f'hand_{s}'] for s in ALLSETS}; Lf = L['fit']
    sel = np.random.default_rng(4708).choice(len(Lf), 4000, replace=False)
    mu, sd = HD['fit'].mean(0), np.where(HD['fit'].std(0) > 0, HD['fit'].std(0), 1.0); HDs = {s: np.nan_to_num((HD[s] - mu) / sd) for s in ALLSETS}
    pk = True
    try:
        for cap, pred, bits, eqs in fit_pysr(HDs, Lf, sel, log):
            st.add(f'pysr_hand_c{cap}', 'pysr_hand', 'PySR on hand observables -> logits', bits, {s: pred(HDs[s]).argmax(1) for s in ALLSETS}, dict(cap=cap, equations=eqs))
    except Exception as ex:  # noqa: BLE001
        pk = f'fit failed: {ex}'[:300]; log('PySR fit failed:', ex)
    np.savez_compressed(OUT / f'b6w_pysr_{model}.npz', **st.arr)
    (OUT / f'b6w_pysr_{model}.json').write_text(json.dumps(dict(pysr=pk, models=st.meta), indent=1, ensure_ascii=False, default=float))
    log(f'wrote {len(st.meta)} PySR surrogates to b6w_pysr_{model}.npz/.json')


GPL_PARS = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03)


def mode_gpl(model, pop=3000, gens=30, part='all'):
    """gplearn with a larger search budget (population 3000, 30 generations, parsimony 1e-4 - 0.03) on RW and HAND observables
    -> b6w_gpl_<model>[_<part>].npz/.json; eval uses these in place of the fit-mode gplearn runs (population 600, 15 generations).
    part rw | hand | shuf runs one third (same seeds, identical fits) so the three parts can run as separate 1-CPU jobs."""
    Z = np.load(OUT / f'b6w_logits_{model}.npz'); O = np.load(OUT / 'b6w_obs.npz')
    L = {s: Z[s].astype(np.float64) for s in ALLSETS}; nc = {s: L[s].argmax(1) for s in ALLSETS}; st = Store(model, nc); Lf = L['fit']
    sel = np.random.default_rng(4708).choice(len(Lf), 4000, replace=False); pr = np.random.default_rng(4706).permutation(len(Lf))
    std = lambda D: (lambda mu, sd: {s: np.nan_to_num((D[s] - mu) / sd) for s in ALLSETS})(D['fit'].mean(0), np.where(D['fit'].std(0) > 0, D['fit'].std(0), 1.0))
    RWs, HDs = std({s: O[f'rw_{s}'] for s in ALLSETS}), std({s: O[f'hand_{s}'] for s in ALLSETS}); t = time.time()
    for pool, Xs in (('rw', RWs), ('hand', HDs)):
        if part not in ('all', pool): continue
        for pars in GPL_PARS:
            pred, bits, inf = fit_gplearn(Xs, Lf, sel, pars, 4710, f'gplearn {pool}', pop, gens)
            st.add(f'gpl_{pool}_p{pars}', f'gpl_{pool}', f'gplearn (pop {pop}, {gens} gen) on {pool} observables -> logits', bits,
                   {s: pred(Xs[s]).argmax(1) for s in ALLSETS}, dict(parsimony=pars, population=pop, generations=gens, **inf))
        log(f'gplearn {pool}: {time.time() - t:.0f} s')
    if part in ('all', 'shuf'):
        pred, bits, inf = fit_gplearn({'fit': RWs['fit'][pr]}, Lf, sel, 0.001, 4720, 'gplearn shuffled', pop, gens)
        st.add('shuf_gpl_rw', 'shuffled', f'gplearn (rw, parsimony 0.001, pop {pop}) on row-shuffled logits', bits, {s: pred(RWs[s]).argmax(1) for s in ALLSETS}, inf)
    tag = model if part == 'all' else f'{model}_{part}'
    np.savez_compressed(OUT / f'b6w_gpl_{tag}.npz', **st.arr)
    (OUT / f'b6w_gpl_{tag}.json').write_text(json.dumps(dict(models=st.meta), indent=1, ensure_ascii=False, default=float)); log(f'wrote b6w_gpl_{tag}')


def mode_smoke():
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: F401
    SR = gpl_patch(); X = np.random.default_rng(0).normal(size=(500, 3)); y = X[:, 0] * X[:, 1] + 2 * X[:, 2]
    m = SR(population_size=200, generations=5, random_state=0, function_set=GPL_FUN).fit(X, y)
    log('sklearn', sklearn.__version__, sklearn.__file__, 'program', m._program, 'pred', m.predict(X[:3]), 'bits', gpl_bits(m._program, 3))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['logits', 'obs', 'fit', 'pysr', 'gpl', 'eval', 'plot', 'smoke']); ap.add_argument('--model', default='jedi')
    ap.add_argument('--skip-pysr', action='store_true'); ap.add_argument('--part', default='all', choices=['all', 'rw', 'hand', 'shuf'])
    a = ap.parse_args(); log('args', vars(a))
    if a.mode == 'logits': mode_logits(a.model)
    elif a.mode == 'obs': mode_obs()
    elif a.mode == 'fit': mode_fit(a.model, a.skip_pysr)
    elif a.mode == 'eval': mode_eval()
    elif a.mode == 'smoke': mode_smoke()
    elif a.mode == 'pysr': mode_pysr(a.model)
    elif a.mode == 'gpl': mode_gpl(a.model, part=a.part)
    else: plot(json.loads((OUT / 'b6w.json').read_text()))
    log('done')
