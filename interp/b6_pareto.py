"""B6 (Paper B claims B2, B3, B4; review weaknesses 3 and 6): edit fidelity vs description length for program extraction on
four networks trained on the same 3-feature, 64-particle hls4ml input: official JEDI-linear n64 (fixed point, jedi_core), Deep
Sets dsets-d64 seed 0 and transformer tf-2b-d32 seed 0 (train/models.py, best.pt; X0/X2), PFN pfn_100-100-128_s0 (X9,
numpy forward).

Jets (all from the test file, interp.jedi_core.sample):
  F  = sample(per_class 4000, seed 2601) = 20,000 jets; permutation rng 0: first 2/3 fit (13,333), last 1/3 dev (6,667; MARS
       stopping, Lasso / MLP size path only).
  T  = sample(per_class 2000, seed 2602) minus every row of F and of sample(6000, seed 0) (the fit jets of the stored E11 / B4
       nested programs), so every surrogate is tested on jets it was not fitted on.
Observables of every blind surrogate (98): the 97 rank-window sums of interp.e11_mars_all.handbuilt (ΣpT, N, ΣpT·r², Σr², Σr,
  ΣpT·r, Σ(r−0.2)₊, N(pT>1), N(pT<2), Σr[r<0.165], N(pT<2, r>0.1) over pT-rank windows; M2 = ΣpT·r²[all] is one of them) + M0 =
  ΣpT.  No unit access: every blind surrogate maps these observables to the model's 5 logits (or its call).
Extractors (targets = the model's logits on F-fit; calls = argmax):
  prog      blind pipeline: MARS (jetdistill.mars.fit_neuron, B4 'direct' settings: target logit + a constant, 30-term forward
            path, stall rule off) per logit; programs = prefixes K in {2, 4, 8, 16, 30} terms per logit, refitted by LS.
  nested    JEDI-linear only: the stored unit-level nested programs (E11 K = 30, B4 refits K = 2, 4, 8, 16; fitted on
            sample(6000, 0)) on the E11 physics set recomputed from the (edited) particles, z -> the network's own d7.
  tree      CART: DecisionTreeClassifier on the call and DecisionTreeRegressor on the 5 logits, depth 2-10.
  gbdt      HistGradientBoosting (lr 0.2, no early stopping): classifier on the call and one regressor per logit,
            max_leaf_nodes 4 / 16 x max_iter 10 / 30 / 100 / 300.
  linear    Lasso path per logit on standardised observables (10 alphas) + Ridge (all 98).
  mlp       MLPRegressor on standardised observables -> 5 logits, one hidden ReLU layer of 2-64 units.
  efp       energy-flow polynomials, all 54 connected graphs d <= 5 (energyflow 1.4 EFPSet, measure 'hadr', beta 1, normed,
            from (pT, Δη, Δφ); venv-ef) + their logs, standardised; Lasso path per logit.
  PySR / gplearn: import attempted in the fit job (not installed -> reported missing).
Description length (bits; 32-bit numbers everywhere): MARS term = 32 (coef) + per factor [log2(#inputs) + 2 (kind) + 32 (knot,
  unless linear)], + 32 per intercept; tree / GBDT internal node = log2(#inputs) + 32, leaf = 32 x outputs (classifier tree
  leaf: log2 5); linear: non-zero weight = 32 + log2(#inputs), + 5 x 32 intercepts; MLP: all weights and biases x 32.
Edits on T (padding never moved; physics recomputed from the edited particles):
  trans   Δη += 0.06 rad for every real particle
  m2h     trans, then all angles scaled per jet by sqrt(M2 / M2') about the stored axis (M0 and M2 = ΣpT r² as before; X9v)
  A13     swap of (Δη, Δφ) of pT ranks 13 and 14 (jets with >= 15 real particles)
  pt09    every particle's pT x 0.9
  scr     per-particle azimuth scramble: each real particle rotated to a uniformly random azimuth about the axis, r and pT
          kept (E5v / E24; rng 2603)
  controls  rdt  coherent translation of 0.06 rad in a uniformly random direction per jet (rng 2604)
            rpd  each real particle moved 0.06 rad in an independent random direction (B4 trans_c; rng 2605)
            never-flip reference: agreement 1 - (network flip rate); shuffled: prog K16 and gbdt-reg L16 T100 fitted on the
            model's logits with the F-fit rows permuted (rng 2606).
Metrics per (model, surrogate, edit): same call on unedited T; edit agreement per jet (B4: surrogate flips iff the network
flips, and to the same new call); Jaccard |S ∩ N| / |S ∪ N| of the flipped-jet sets; recall.  95 % jet bootstrap (B 1000).
Area under the frontier: F_e(b) = max agreement over family e's surrogates with log2 bits <= b; mean over a 200-point grid of
log2 bits from max(min_prog, min_e) to min(max_prog, max_e); paired
bootstrap of AUC(prog) - AUC(e) (same jet resamples); two-sided bootstrap-z p (d / bootstrap s.e., normal); grid = the
overlap of the two families' log2-bits ranges (union, frontiers held flat, when they overlap by < 0.25); Holm over models x
edits x baseline families (5 edits; programs: blind program on 4 models + nested on JEDI-linear).
Matched DL (B4 rule): each program vs, per baseline family, the member with bits <= the program's bits and the highest dev
same call; paired bootstrap CI of the per-jet agreement difference, exact McNemar, Holm over models x programs x families x
edits.
Modes (python -m interp.b6_pareto <mode> [--model M]):
  logits --model jedi|pfn   (jax env)     -> $AI_RESULTS/jedi/b6_logits_<model>.npz (+ b6_nested_jedi.npz for jedi)
  logits --model dsets|tf   (AI_PY=conda) -> same
  efp                       (venv-ef)     -> b6_efp.npz
  fit --model M             (jax env)     -> b6_pred_<model>.npz / .json
  eval                      (jax env)     -> b6.json, b6_<edit>.png
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
SMALL = Path(os.environ.get('AI_SMALL_MODELS', '/j-jepa-vol/aaron-interp/models/small'))
T0 = time.time()
CL = ['g', 'q', 'W', 'Z', 't']
MODELS = dict(jedi='JEDI-linear n64', dsets='Deep Sets d64 s0', pfn='PFN 100-100-128 s0', tf='transformer 2b-d32 s0')
EDITS = ['trans', 'm2h', 'A13', 'pt09', 'scr']
CTRL = ['rdt', 'rpd']
SETS = ['test'] + EDITS + CTRL
EDIT_LABEL = dict(trans='0.06 rad translation', m2h='translation with M0, M2 held', A13='angle swap of pT ranks 13|14',
                  pt09='all pT x 0.9', scr='per-particle azimuth scramble', rdt='control: random-direction translation 0.06 rad',
                  rpd='control: per-particle random 0.06 rad displacement')
FAMILIES = ['prog', 'nested', 'tree', 'gbdt', 'linear', 'mlp', 'efp']
BASELINES = ['tree', 'gbdt', 'linear', 'mlp', 'efp']


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


# ------------------------------------------------------------------ jets and edits (numpy only; identical in every env)
def indices(per_class, seed):
    from jetdistill.config import RESULTS
    y = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y']); rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in range(5)]))


def build():
    """-> dict(x={set: (J, 64, 3)}, y={set: labels}, J13=indices into T, nfit, ...).  Sets: fit, dev, test, edits, controls."""
    from interp.jedi_core import sample
    xF, yF, _ = sample(4000, 2601); xT, yT, _ = sample(2000, 2602)
    iF, iT, i0 = indices(4000, 2601), indices(2000, 2602), indices(6000, 0)
    keep = ~np.isin(iT, np.concatenate([iF, i0])); xT, yT = np.asarray(xT[keep]), yT[keep]
    perm = np.random.default_rng(0).permutation(len(yF)); nf = 2 * len(yF) // 3
    x = dict(fit=np.asarray(xF[perm[:nf]]), dev=np.asarray(xF[perm[nf:]]), test=xT)
    y = dict(fit=yF[perm[:nf]], dev=yF[perm[nf:]], test=yT)
    real = xT[..., 0] > 0; nreal = real.sum(1)
    e = xT.copy(); e[..., 1] += np.float32(0.06) * real; x['trans'] = e
    M2 = lambda a: (a[..., 0] * (a[..., 1] ** 2 + a[..., 2] ** 2)).sum(1).astype(np.float64)
    s = np.sqrt(M2(xT) / np.maximum(M2(e), 1e-12)); m = e.copy(); m[..., 1:] = (e[..., 1:] * s[:, None, None]).astype(np.float32); x['m2h'] = m
    J13 = np.flatnonzero(nreal >= 15); a = xT[J13].copy(); a[:, [13, 14], 1:] = xT[J13][:, [14, 13], 1:]; x['A13'] = a
    p = xT.copy(); p[..., 0] *= np.float32(0.9); x['pt09'] = p
    rng = np.random.default_rng(2603); r = np.hypot(xT[..., 1], xT[..., 2]); th = rng.uniform(-np.pi, np.pi, r.shape)
    sc = xT.copy(); sc[..., 1] = np.where(real, r * np.cos(th), 0); sc[..., 2] = np.where(real, r * np.sin(th), 0); x['scr'] = sc.astype(np.float32)
    ph = np.random.default_rng(2604).uniform(0, 2 * np.pi, len(xT)); d = xT.copy()
    d[..., 1] += (0.06 * np.cos(ph))[:, None].astype(np.float32) * real; d[..., 2] += (0.06 * np.sin(ph))[:, None].astype(np.float32) * real; x['rdt'] = d
    th2 = np.random.default_rng(2605).uniform(0, 2 * np.pi, real.shape); q = xT.copy()
    q[..., 1] += (0.06 * np.cos(th2)).astype(np.float32) * real; q[..., 2] += (0.06 * np.sin(th2)).astype(np.float32) * real; x['rpd'] = q
    for k in list(x): x[k] = np.ascontiguousarray(x[k], dtype=np.float32)
    base = {k: np.arange(len(yT)) for k in SETS}; base['A13'] = J13          # rows of T each edited set corresponds to
    for k in SETS: y[k] = yT[base[k]]
    info = dict(N_fit=int(nf), N_dev=int(len(yF) - nf), N_test=int(len(yT)), removed_from_test=int((~keep).sum()), N_A13=int(len(J13)),
                m2h_scale_mean=float(s.mean()))
    log(f"jets: fit {info['N_fit']}, dev {info['N_dev']}, test {info['N_test']} ({info['removed_from_test']} test rows shared with F or "
        f"sample(6000, 0) removed); A13 jets {len(J13)}; m2h angle scale mean {s.mean():.4f}")
    return dict(x=x, y=y, base=base, info=info)


ALLSETS = ['fit', 'dev'] + SETS


# ------------------------------------------------------------------ mode logits
def logits_jedi(B):
    from interp.jedi_core import Jedi
    net = Jedi(64); out = {s: net.run(B['x'][s])['logits'] for s in ALLSETS}
    return out, net


def logits_pfn(B):
    from interp.x9_efn_pfn import Flow
    f = Flow('pfn_100-100-128_s0'); return {s: f.run(B['x'][s])['logits'] for s in ALLSETS}


def logits_torch(B, name):
    import torch
    from interp.x2_small_models import load_net, normalise
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    run = dict(dsets='dsets-d64-n64', tf='tf-2b-d32-n64')[name]
    net, arch = load_net(SMALL / run / 'seed0'); out = {}
    for s in ALLSETS:
        xn, mask = normalise(B['x'][s], arch); r = []
        with torch.no_grad():
            for i in range(0, len(xn), 2000):
                r.append(net(torch.from_numpy(xn[i:i + 2000]), torch.from_numpy(mask[i:i + 2000])).double().numpy())
        out[s] = np.concatenate(r)
    return out


def nested_jedi(B, net):
    """stored E11 / B4 nested programs on the E11 physics set recomputed from each set's particles"""
    from interp import e11_mars_all as E
    from interp.e11_verify import Last, chain, count_terms
    from jetdistill.observables import compute
    last = Last(net); progs = {}
    for K in (2, 4, 8, 16, 30):
        pre = 'e11' if K == 30 else f'b4m{K}'; rd = lambda n: json.loads((OUT / f'{pre}_{n}.json').read_text())
        fb = {'P': rd('fit_a_P'), **{s: rd(f'fit_b_{s}') for s in E.STG[1:]}}
        for s in E.STG: assert [u['unit'] for u in fb[s]['units']] == E.UNITS[s], (K, s)
        bits = sum(32 * len(fb[s]['units']) + sum(sum(term_bits(t, len(fb[s]['keys']))) for u in fb[s]['units'] for t in u['terms']) for s in E.STG)
        progs[K] = (fb, count_terms(fb), float(bits))
    calls, meta = {}, []
    for s in ['test'] + EDITS + CTRL:
        t = time.time(); xx = B['x'][s]; q = compute(np.asarray(xx, np.float64), 64)
        ph = {k: np.asarray(v, np.float32).astype(np.float64) for k, v in q.items()}; ph.update(E.handbuilt(xx, False)); del q
        for K, (fb, ct, bits) in progs.items():
            V = chain(fb, ph); calls[f'nested{K}|{s}'] = np.asarray(last.jax_logits(V['z'])).argmax(1).astype(np.int8)
        del ph; log(f'nested programs on {s} ({len(xx)} jets): {time.time() - t:.0f} s')
    for K, (fb, ct, bits) in progs.items():
        meta.append(dict(name=f'nested{K}', family='nested', method='unit-level nested MARS program (E11/B4, stored)', bits=bits, dl=ct['terms'],
                         n_inputs=None, K=K, counts=ct))
    return calls, meta


def term_bits(t, nin):
    """per term: [32 coef] + per factor log2(#inputs) + 2 + 32 (knot unless linear); returned as a list (sum it)"""
    lb = float(np.log2(max(nin, 2)))
    f = lambda kind: lb + 2 + (0 if kind == 'lin' else 32)
    return [32.0, f(t['kind'])] + ([f(t['kind2'])] if t.get('q2') else [])


def mode_logits(model):
    B = build()
    if model == 'jedi': L, net = logits_jedi(B)
    elif model == 'pfn': L = logits_pfn(B)
    else: L = logits_torch(B, model)
    acc = {s: float((L[s].argmax(1) == B['y'][s]).mean()) for s in ('fit', 'dev', 'test')}
    log(f'{model}: accuracy fit {acc["fit"] * 100:.2f} % dev {acc["dev"] * 100:.2f} % test {acc["test"] * 100:.2f} % (N {len(B["y"]["test"])})')
    for s in EDITS + CTRL:
        cb = L['test'].argmax(1)[B['base'][s]]; ca = L[s].argmax(1); log(f'{model}: edit {s} changes {(ca != cb).mean() * 100:.2f} % of calls (N {len(ca)})')
    np.savez_compressed(OUT / f'b6_logits_{model}.npz', **{s: L[s].astype(np.float32) for s in ALLSETS}, **{f'y_{s}': B['y'][s] for s in ALLSETS},
                        J13=B['base']['A13'])
    (OUT / f'b6_logits_{model}.json').write_text(json.dumps(dict(model=model, acc=acc, info=B['info']), indent=1))
    log(f'wrote {OUT}/b6_logits_{model}.npz')
    if model == 'jedi':
        calls, meta = nested_jedi(B, net)
        np.savez_compressed(OUT / 'b6_nested_jedi.npz', **calls); (OUT / 'b6_nested_jedi.json').write_text(json.dumps(meta, indent=1, default=float))
        log(f'wrote {OUT}/b6_nested_jedi.npz')


# ------------------------------------------------------------------ mode efp (venv-ef)
def mode_efp():
    import energyflow as ef
    B = build(); es = ef.EFPSet('d<=5', 'p==1', measure='hadr', beta=1, normed=True, coords='ptyphim')
    log(f'EFPSet d<=5 connected: {len(es.graphs())} graphs')
    out = {}
    for s in ALLSETS:
        t = time.time(); xx = B['x'][s]
        jets = [j[j[:, 0] > 0][:, [0, 1, 2]].astype(np.float64) for j in xx]
        out[s] = np.asarray(es.batch_compute(jets, n_jobs=1), np.float32); log(f'EFPs {s}: {out[s].shape} in {time.time() - t:.0f} s')
    np.savez_compressed(OUT / 'b6_efp.npz', **out); log(f'wrote {OUT}/b6_efp.npz')


# ------------------------------------------------------------------ mode fit
def features(xx):
    from interp import e11_mars_all as E
    H = E.handbuilt(xx, False); H.pop('H:|M1|'); H.pop('H:√(M0·M2)')
    keys = sorted(H); X = np.stack([H[k] for k in keys], 1); M0 = np.asarray(xx[..., 0], np.float64).sum(1)
    return np.concatenate([X, M0[:, None]], 1), keys + ['M0=ΣpT']


class Store:
    def __init__(self, model, net_call):
        self.model, self.arr, self.meta, self.nc = model, {}, [], net_call

    def add(self, name, family, method, bits, calls, extra=None):
        for s, c in calls.items(): self.arr[f'{name}|{s}'] = np.asarray(c, np.int8)
        sc = float((calls['test'] == self.nc['test']).mean()); sd = float((calls['dev'] == self.nc['dev']).mean())
        self.meta.append(dict(name=name, family=family, method=method, bits=float(bits), same_call_dev=sd, same_call_test=sc, **(extra or {})))
        log(f'{self.model}: {name:22s} ({family}) bits {bits:9.0f}  same call dev {sd * 100:6.2f} test {sc * 100:6.2f}')


def fit_mars(Xs, keys, Lf, Ld, kmax=30, tag='prog'):
    from jetdistill import mars
    Q = {s: {k: Xs[s][:, i].astype(np.float64) for i, k in enumerate(keys)} for s in Xs}
    from interp.e11_mars_all import knots_of
    knots = knots_of(Q['fit'], keys); NF = len(Lf); sel = np.random.default_rng(1).choice(NF, min(6000, NF), replace=False)
    mars.MAX_TERMS, mars.STALL = kmax, 10 ** 6; fits = []
    for k in range(5):
        t = time.time(); c = 1.0 - min(Lf[:, k].min(), Ld[:, k].min()) + 0.5 * Lf[:, k].std()
        terms, _, path = mars.fit_neuron(dict(fit=Lf[:, k] + c, dev=Ld[:, k] + c), {'fit': Q['fit'], 'dev': Q['dev']}, keys, knots, sel)
        fits.append((terms, c)); log(f'{tag} MARS logit {CL[k]}: {len(terms)} terms (dev R² path end {path[-1] if path else None}); {time.time() - t:.0f} s')
    progs = {}
    for K in (2, 4, 8, 16, 30):
        parts = []
        for k, (terms, c) in enumerate(fits):
            tr = terms[:K]; co = mars._lstsq(tr, Q['fit'], Lf[:, k] + c, NF); parts.append((tr, co, c))
        bits = sum(32 + sum(sum(term_bits(t, len(keys))) for t in tr) for tr, _, _ in parts)
        progs[K] = (parts, bits)

    def predict(parts, X):
        Qd = {k: X[:, i].astype(np.float64) for i, k in enumerate(keys)}; J = len(X)
        return np.stack([np.stack([np.ones(J)] + [mars.term_value(t, Qd) for t in tr], 1) @ co - c for tr, co, c in parts], 1)
    return progs, predict, fits


def tree_bits(tr, nin, nout, cls=False):
    n_int = int(tr.node_count - tr.n_leaves)
    return n_int * (np.log2(max(nin, 2)) + 32) + tr.n_leaves * (np.log2(5) if cls else 32 * nout)


def hgb_bits(m, nin):
    b = 0.0
    for it in m._predictors:
        for p in it:
            nl = int(p.nodes['is_leaf'].sum()); b += (len(p.nodes) - nl) * (np.log2(max(nin, 2)) + 32) + nl * 32
    return b


def mode_fit(model):
    import warnings
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import lasso_path, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    warnings.filterwarnings('ignore')
    B = build(); Z = np.load(OUT / f'b6_logits_{model}.npz'); L = {s: Z[s].astype(np.float64) for s in ALLSETS}
    assert np.array_equal(Z['y_test'], B['y']['test']) and np.array_equal(Z['J13'], B['base']['A13'])
    nc = {s: L[s].argmax(1) for s in ALLSETS}; st = Store(model, nc)
    have = {}
    for m in ('pysr', 'gplearn'):
        try: __import__(m); have[m] = True
        except Exception: have[m] = False  # noqa: BLE001
    log(f'optional packages: {have}')
    X = {}; keys = None
    for s in ALLSETS: X[s], keys = features(B['x'][s])
    F = len(keys); log(f'{F} observables: {keys[:4]} ... {keys[-3:]}')
    Lf, Ld, cf = L['fit'], L['dev'], nc['fit']

    # (1) blind program
    progs, predict, fits = fit_mars(X, keys, Lf, Ld)
    for K, (parts, bits) in progs.items():
        st.add(f'prog{K}', 'prog', 'MARS on rank-window observables -> logits', bits, {s: predict(parts, X[s]).argmax(1) for s in ALLSETS},
               dict(K=K, terms=sum(len(p[0]) for p in parts)))
    prog_txt = [[dict(q=t['q'], kind=t['kind'], t=t.get('t'), q2=t.get('q2'), kind2=t.get('kind2'), t2=t.get('t2')) for t in tr] for tr, _ in fits]
    # shuffled-logit control
    pr = np.random.default_rng(2606).permutation(len(Lf)); pd_ = np.random.default_rng(2607).permutation(len(Ld))
    sprogs, spredict, _ = fit_mars(X, keys, Lf[pr], Ld[pd_], kmax=16, tag='shuffled')
    st.add('shuf_prog16', 'shuffled', 'MARS K16 on row-shuffled logits', sprogs[16][1], {s: spredict(sprogs[16][0], X[s]).argmax(1) for s in ALLSETS})
    rs = [HistGradientBoostingRegressor(max_leaf_nodes=16, max_iter=100, learning_rate=0.2, early_stopping=False, random_state=0).fit(X['fit'], Lf[pr, k]) for k in range(5)]
    st.add('shuf_gbdtreg_L16_T100', 'shuffled', 'GBDT on row-shuffled logits', sum(hgb_bits(r, F) for r in rs),
           {s: np.stack([r.predict(X[s]) for r in rs], 1).argmax(1) for s in ALLSETS})
    # (2) trees
    t = time.time()
    for d in range(2, 11):
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X['fit'], cf)
        st.add(f'tree_d{d}', 'tree', 'CART on call', tree_bits(m.tree_, F, 1, True), {s: m.predict(X[s]) for s in ALLSETS}, dict(depth=d))
        r = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X['fit'], Lf)
        st.add(f'treereg_d{d}', 'tree', 'CART on logits', tree_bits(r.tree_, F, 5), {s: r.predict(X[s]).argmax(1) for s in ALLSETS}, dict(depth=d))
    log(f'trees {time.time() - t:.0f} s'); t = time.time()
    # (3) GBDT
    for lv in (4, 16):
        for it in (10, 30, 100, 300):
            m = HistGradientBoostingClassifier(max_leaf_nodes=lv, max_iter=it, learning_rate=0.2, early_stopping=False, random_state=0).fit(X['fit'], cf)
            st.add(f'gbdt_L{lv}_T{it}', 'gbdt', 'GBDT on call', hgb_bits(m, F), {s: m.predict(X[s]) for s in ALLSETS}, dict(leaves=lv, iters=it))
            rs = [HistGradientBoostingRegressor(max_leaf_nodes=lv, max_iter=it, learning_rate=0.2, early_stopping=False, random_state=0).fit(X['fit'], Lf[:, k]) for k in range(5)]
            st.add(f'gbdtreg_L{lv}_T{it}', 'gbdt', 'GBDT on logits', sum(hgb_bits(r, F) for r in rs),
                   {s: np.stack([r.predict(X[s]) for r in rs], 1).argmax(1) for s in ALLSETS}, dict(leaves=lv, iters=it))
        log(f'gbdt leaves {lv}: {time.time() - t:.0f} s')
    # (5) linear and MLP probes
    sc = StandardScaler().fit(X['fit']); Zs = {s: np.nan_to_num(sc.transform(X[s])) for s in ALLSETS}; mu = Lf.mean(0)
    def lasso_family(Zf, Zall, nin, fam, tag):
        alphas = np.logspace(0, -3.5, 10); paths = [lasso_path(Zf, Lf[:, k] - mu[k], alphas=alphas, max_iter=3000, tol=1e-4)[1] for k in range(5)]
        for i, al in enumerate(alphas):
            W = np.stack([p[:, i] for p in paths], 1); nz = int((np.abs(W) > 0).sum())
            st.add(f'{tag}_a{al:.3g}', fam, f'Lasso on logits ({tag})', nz * (32 + np.log2(nin)) + 5 * 32, {s: (Zall[s] @ W + mu).argmax(1) for s in ALLSETS}, dict(alpha=float(al), nonzero=nz))
    lasso_family(Zs['fit'], Zs, F, 'linear', 'lasso')
    rd = Ridge(alpha=1.0).fit(Zs['fit'], Lf)
    st.add('ridge', 'linear', 'Ridge on logits (all observables)', (F * 5) * 32 + 5 * 32, {s: rd.predict(Zs[s]).argmax(1) for s in ALLSETS})
    log(f'linear {time.time() - t:.0f} s'); t = time.time()
    for h in (2, 4, 8, 16, 32, 64):
        m = MLPRegressor(hidden_layer_sizes=(h,), max_iter=400, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=0).fit(Zs['fit'], Lf)
        st.add(f'mlp_h{h}', 'mlp', '1-hidden-layer MLP on logits', (F * h + h + h * 5 + 5) * 32, {s: m.predict(Zs[s]).argmax(1) for s in ALLSETS}, dict(hidden=h))
    log(f'mlp {time.time() - t:.0f} s'); t = time.time()
    # (6) EFP-linear
    pe = OUT / 'b6_efp.npz'
    if pe.exists():
        G = np.load(pe); Ge = {s: np.concatenate([G[s], np.log(np.maximum(G[s], 1e-12))], 1).astype(np.float64) for s in ALLSETS}
        sg = StandardScaler().fit(Ge['fit']); Gs = {s: np.nan_to_num(sg.transform(Ge[s])) for s in ALLSETS}
        lasso_family(Gs['fit'], Gs, Gs['fit'].shape[1], 'efp', 'efp')
        log(f'efp {time.time() - t:.0f} s')
    else:
        log('b6_efp.npz missing: EFP family skipped')
    if model == 'jedi' and (OUT / 'b6_nested_jedi.npz').exists():
        N = np.load(OUT / 'b6_nested_jedi.npz'); meta = json.loads((OUT / 'b6_nested_jedi.json').read_text())
        for m in meta:
            calls = {s: N[f'{m["name"]}|{s}'] for s in ['test'] + EDITS + CTRL}
            for k, v in calls.items(): st.arr[f'{m["name"]}|{k}'] = v
            st.meta.append(dict(m, same_call_dev=None, same_call_test=float((calls['test'] == nc['test']).mean())))
            log(f'{model}: {m["name"]} (nested) bits {m["bits"]:.0f} same call test {st.meta[-1]["same_call_test"] * 100:.2f}')
    np.savez_compressed(OUT / f'b6_pred_{model}.npz', **st.arr)
    (OUT / f'b6_pred_{model}.json').write_text(json.dumps(dict(model=model, packages=have, observables=keys, prog_terms=prog_txt, models=st.meta), indent=1, ensure_ascii=False, default=float))
    log(f'wrote {OUT}/b6_pred_{model}.npz/.json ({len(st.meta)} surrogates)')


# ------------------------------------------------------------------ mode eval
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def frontier_auc(lb, A, fam_idx_p, fam_idx_e, grid=200):
    """lb (M,) log2 bits; A (..., M) agreement values; mean of the step frontier difference prog - e on a common grid"""
    lo = max(lb[fam_idx_p].min(), lb[fam_idx_e].min()); hi = min(lb[fam_idx_p].max(), lb[fam_idx_e].max())
    if hi <= lo + 0.25: hi = max(lb[fam_idx_p].max(), lb[fam_idx_e].max())      # no overlap: union range, frontiers held flat
    g = np.linspace(lo, hi, grid)
    def fr(ix):
        ok = lb[ix][None, :] <= g[:, None] + 1e-9                     # (grid, m)
        v = np.where(ok[None], A[..., ix][..., None, :], -np.inf).max(-1)  # (..., grid)
        return v
    return (fr(fam_idx_p) - fr(fam_idx_e)).mean(-1), (float(lo), float(hi))


def mode_eval(Bn=1000):
    B = build(); res = dict(procedure=__doc__, info=B['info'], models={}, tests=[], matched=[])
    for model in MODELS:
        pz, pj = OUT / f'b6_pred_{model}.npz', OUT / f'b6_pred_{model}.json'
        if not pz.exists(): log(f'missing {pz}'); continue
        Z = np.load(OUT / f'b6_logits_{model}.npz'); nc = {s: Z[s].argmax(1) for s in SETS}
        P = np.load(pz); meta = [m for m in json.loads(pj.read_text())['models']]
        pk = json.loads(pj.read_text())['packages']
        N = len(nc['test']); rng = np.random.default_rng(7)
        cnt = {}
        for s in ['test'] + EDITS + CTRL:
            n = len(nc[s]); cnt.setdefault(n, np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(Bn)]).astype(np.float32))
        C = {s: cnt[len(nc[s])] for s in ['test'] + EDITS + CTRL}
        names = [m['name'] for m in meta]; M = len(names)
        net = dict(acc_test=float((nc['test'] == B['y']['test']).mean()), N_test=N, flip={}, N={})
        acc_ind = (nc['test'] == B['y']['test']).astype(np.float32); bacc = C['test'] @ acc_ind / N
        net['acc_test_ci'] = [float(np.percentile(bacc, 2.5)), float(np.percentile(bacc, 97.5))]
        rows = {n: dict(m) for n, m in zip(names, meta)}
        SC = np.stack([(P[f'{n}|test'] == nc['test']) for n in names]).astype(np.float32)        # (M, N)
        bsc = C['test'] @ SC.T / N
        for i, n in enumerate(names):
            rows[n]['same_call'] = [float(SC[i].mean()), float(np.percentile(bsc[:, i], 2.5)), float(np.percentile(bsc[:, i], 97.5))]
        AG = {}; BAG = {}; AGJ = {}
        for e in EDITS + CTRL:
            bi = B['base'][e]; nb, na = nc['test'][bi], nc[e]; nf = na != nb; ne = len(nf)
            bf = C[e] @ nf.astype(np.float32) / ne
            net['flip'][e] = [float(nf.mean()), float(np.percentile(bf, 2.5)), float(np.percentile(bf, 97.5))]; net['N'][e] = int(ne)
            A = np.zeros((M, ne), np.float32); I = np.zeros((M, ne), np.float32); U = np.zeros((M, ne), np.float32)
            for i, n in enumerate(names):
                sb, sa = P[f'{n}|test'][bi], P[f'{n}|{e}']; sf = sa != sb
                A[i] = (nf == sf) & (~nf | (sa == na)); I[i] = nf & sf; U[i] = nf | sf
                r = rows[n].setdefault('edits', {})
                r[e] = dict(flip=float(sf.mean()), recall=float((sf & (sa == na))[nf].mean()) if nf.any() else None,
                            jaccard=float(I[i].sum() / max(U[i].sum(), 1)))
                if e in ('trans', 'rdt'):
                    w = B['y'][e] == 2; r[e]['W_to_Z'] = float((sa[w] == 3).mean()); r[e]['W_to_Z_before'] = float((sb[w] == 3).mean())
            bA = C[e] @ A.T / ne; bI = C[e] @ I.T; bU = C[e] @ U.T; bJ = bI / np.maximum(bU, 1)
            for i, n in enumerate(names):
                rows[n]['edits'][e]['agree'] = [float(A[i].mean()), float(np.percentile(bA[:, i], 2.5)), float(np.percentile(bA[:, i], 97.5))]
                rows[n]['edits'][e]['jaccard_ci'] = [float(np.percentile(bJ[:, i], 2.5)), float(np.percentile(bJ[:, i], 97.5))]
            AG[e], BAG[e] = A.mean(1), bA; AGJ[e] = A
            if e in ('trans', 'rdt'):
                w = B['y'][e] == 2; net.setdefault('W_to_Z', {})[e] = [float((nb[w] == 3).mean()), float((na[w] == 3).mean())]
        lb = np.log2(np.array([rows[n]['bits'] for n in names], float))
        fam = np.array([rows[n]['family'] for n in names])
        progs = ['prog'] + (['nested'] if model == 'jedi' else [])
        for pf in progs:
            ip = np.flatnonzero(fam == pf)
            for e in ['same_call'] + EDITS + CTRL:
                obs = SC.mean(1) if e == 'same_call' else AG[e]; boot = bsc if e == 'same_call' else BAG[e]
                for bf_ in BASELINES:
                    ie = np.flatnonzero(fam == bf_)
                    if not len(ie): continue
                    d, rngb = frontier_auc(lb, obs, ip, ie); db, _ = frontier_auc(lb, boot, ip, ie); d = float(np.ravel(d)[0])
                    from scipy.stats import norm
                    se = float(db.std()); p = float(2 * norm.sf(abs(d) / se)) if se > 0 else (1.0 if d == 0 else 0.0)
                    pe = float(min(1.0, 2 * min((db <= 0).mean(), (db >= 0).mean()) + 1.0 / Bn))
                    overlap = bool(min(lb[ip].max(), lb[ie].max()) > max(lb[ip].min(), lb[ie].min()) + 0.25)
                    res['tests'].append(dict(model=model, program=pf, edit=e, baseline=bf_, auc_diff=d, ci=[float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))],
                                             se=se, p=p, p_boot_empirical=pe, log2_bits_range=rngb, overlap_range=overlap))
        # matched-DL paired comparisons (B4 rule): each program vs the baseline of each family with bits <= program bits and the
        # highest same call on the dev jets (selected without the test jets); paired bootstrap CI and exact McNemar per edit
        from scipy.stats import binomtest
        for pf in progs:
            for i in np.flatnonzero(fam == pf):
                for bf_ in BASELINES:
                    el = [j for j in np.flatnonzero(fam == bf_) if rows[names[j]]['bits'] <= rows[names[i]]['bits'] and rows[names[j]].get('same_call_dev') is not None]
                    if not el: continue
                    j = max(el, key=lambda j: rows[names[j]]['same_call_dev'])
                    for e in ['same_call'] + EDITS + CTRL:
                        a_, b_ = (SC[i], SC[j]) if e == 'same_call' else (AGJ[e][i], AGJ[e][j]); Ce = C['test'] if e == 'same_call' else C[e]
                        dd = Ce @ (a_ - b_) / len(a_); n10, n01 = int(((a_ > 0) & (b_ == 0)).sum()), int(((a_ == 0) & (b_ > 0)).sum())
                        pm = float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
                        res['matched'].append(dict(model=model, program=names[i], baseline=names[j], family=bf_, edit=e, prog_bits=rows[names[i]]['bits'],
                                                   base_bits=rows[names[j]]['bits'], diff=float((a_ - b_).mean()), ci=[float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                                                   prog_only=n10, base_only=n01, p_mcnemar=pm))
        res['models'][model] = dict(label=MODELS[model], network=net, packages=pk, surrogates=list(rows.values()))
        log(f'{model}: network acc test {net["acc_test"] * 100:.2f} % {np.round(np.array(net["acc_test_ci"]) * 100, 2)} (N {N}); flips ' +
            ', '.join(f'{e} {v[0] * 100:.1f} %' for e, v in net['flip'].items()))
        for n in names:
            r = rows[n]
            log(f"  {n:22s} {r['family']:8s} log2 bits {np.log2(r['bits']):5.1f} same call {r['same_call'][0] * 100:6.2f} | " +
                ' '.join(f"{e} {r['edits'][e]['agree'][0] * 100:5.1f}/J{r['edits'][e]['jaccard']:.2f}" for e in EDITS + CTRL))
    hold = [t for t in res['tests'] if t['edit'] in EDITS]
    q = holm([t['p'] for t in hold])
    for t, qq in zip(hold, q): t['p_holm'] = float(qq)
    res['n_holm'] = len(hold)
    mh = [t for t in res['matched'] if t['edit'] in EDITS]; q = holm([t['p_mcnemar'] for t in mh])
    for t, qq in zip(mh, q): t['p_holm'] = float(qq)
    res['n_holm_matched'] = len(mh)
    for t in res['matched']:
        log(f"matched {t['model']:5s} {t['program']:9s} vs {t['baseline']:18s} ({t['family']:6s}) {t['edit']:9s}: {t['diff'] * 100:+6.2f} "
            f"[{t['ci'][0] * 100:+.2f}, {t['ci'][1] * 100:+.2f}] McNemar p {t['p_mcnemar']:.3g}" + (f" Holm {t['p_holm']:.3g}" if 'p_holm' in t else ''))
    for t in res['tests']:
        log(f"AUC {t['model']:5s} {t['program']:6s} vs {t['baseline']:6s} {t['edit']:9s}: {t['auc_diff'] * 100:+6.2f} "
            f"[{t['ci'][0] * 100:+.2f}, {t['ci'][1] * 100:+.2f}] p {t['p']:.3g}" + (f" Holm {t['p_holm']:.3g}" if 'p_holm' in t else ''))
    (OUT / 'b6.json').write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float)); log(f'wrote {OUT}/b6.json')
    plot(res)


STYLE = dict(prog=('blind program (MARS on rank windows)', '#1c5cab', 'o', '-'), nested=('unit-level nested program', '#17becf', 'h', '-'),
             tree=('CART', '#2a9d3c', '^', '-'), gbdt=('GBDT', '#c0392b', 'D', '-'), linear=('linear (Lasso / ridge)', '#7b4fa0', 'v', '-'),
             mlp=('1-hidden-layer MLP', '#e07b00', 's', '-'), efp=('EFP d ≤ 5 + Lasso', '#555555', 'X', ':'), shuffled=('fitted on shuffled logits', '#999999', '*', 'none'))


def plot(res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for e in ['same_call'] + EDITS + CTRL:
        mods = [m for m in MODELS if m in res['models']]
        fig, axes = plt.subplots(1, len(mods), figsize=(4.3 * len(mods), 4.2), sharey=False)
        axes = np.atleast_1d(axes)
        for ax, mk in zip(axes, mods):
            R = res['models'][mk]; S = R['surrogates']
            for fam, (lab, col, mkr, ls) in STYLE.items():
                rr = sorted([r for r in S if r['family'] == fam], key=lambda r: r['bits'])
                if not rr: continue
                v = np.array([r['same_call'] if e == 'same_call' else r['edits'][e]['agree'] for r in rr]) * 100; x = np.log2([r['bits'] for r in rr])
                # frontier line (best value at <= bits), raw points as markers
                fr = np.maximum.accumulate(v[:, 0])
                if ls != 'none': ax.step(x, fr, where='post', color=col, lw=1.5, ls=ls, label=lab)
                ax.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=mkr, color=col, ms=3.5, lw=0.8, alpha=0.75,
                            label=None if ls != 'none' else lab)
            if e != 'same_call':
                ch = (1 - R['network']['flip'][e][0]) * 100
                ax.axhline(ch, color='#777777', lw=1, ls='--'); ax.text(0.02, ch, f'never-flip {ch:.1f} %', transform=ax.get_yaxis_transform(), fontsize=7.5, color='#555555', va='bottom')
                ax.set_title(f"{R['label']}\nnetwork changes {R['network']['flip'][e][0] * 100:.1f} % of calls (N {R['network']['N'][e]:,})", fontsize=9, loc='left')
            else:
                ax.set_title(f"{R['label']}\nnetwork accuracy {R['network']['acc_test'] * 100:.1f} % (N {R['network']['N_test']:,})", fontsize=9, loc='left')
            ax.set_xlabel('description length, log2 bits', fontsize=8.5); ax.grid(alpha=0.3, lw=0.5); ax.tick_params(labelsize=8)
        axes[0].set_ylabel('same call with the network (%)' if e == 'same_call' else 'edit agreement with the network (%)', fontsize=8.5)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc='lower center', ncol=len(l), fontsize=8, frameon=False)
        ttl = 'same call on unedited test jets' if e == 'same_call' else f'agreement with the network\'s call changes under: {EDIT_LABEL[e]}'
        fig.suptitle(f'{ttl}, vs surrogate size (lines: best surrogate at or below each size; markers: 95 % jet-bootstrap intervals)', fontsize=10, x=0.01, ha='left')
        fig.tight_layout(rect=(0, 0.07, 1, 0.95)); p = OUT / f'b6_{e}.png'; fig.savefig(p, dpi=130); plt.close(fig); log(f'wrote {p}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['logits', 'efp', 'fit', 'eval', 'plot']); ap.add_argument('--model', default='jedi')
    a = ap.parse_args(); log('args', vars(a))
    if a.mode == 'logits': mode_logits(a.model)
    elif a.mode == 'efp': mode_efp()
    elif a.mode == 'fit': mode_fit(a.model)
    elif a.mode == 'eval': mode_eval()
    else: plot(json.loads((OUT / 'b6.json').read_text()))
    log('done')
