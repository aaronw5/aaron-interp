"""B6 verification (b6-verify): rerun the matched-DL comparison of interp.b6_pareto on fresh jets and test whether the
program's lead over trees / GBDT at <= its description length depends on (i) the baseline selection rule and (ii) the
single GBDT hyper-parameter setting (lr 0.2, 4 leaves, 10 iterations) fitted in B6 at that size.

Models: JEDI-linear n64 (jedi_core, fixed point) and PFN 100-100-128 s0 (x9 numpy forward); both run in the jax env.
Jets: F = sample(4000, 3701) -> permutation rng 0, 2/3 fit / 1/3 dev; T = sample(2000, 3702) minus every row of F,
  sample(6000, 0) and sample(4000, 2601) (B6's F).  Same 98 observables, MARS program and bit costs as b6_pareto.
Edits (on T and on dev, same definitions as B6): trans (Δη += 0.06), m2h (trans with M0, M2 held), pt09 (pT x 0.9),
  rdt (random-direction coherent 0.06 rad translation), rpd (per-particle random 0.06 rad displacement).
Baselines with bits <= prog30's bits: CART d2-10 (call / logits); B6's GBDT L4 T10 (lr 0.2); a GBDT grid at <= 14k bits:
  leaves {2, 3, 4, 8} x lr {0.1, 0.2, 0.5, 1.0} x iterations up to the budget, on the call and on the logits; MLP h2, h4.
Selection of the matched baseline among all of these with bits <= the program's bits:
  devsc   highest dev same call (B6 rule)
  devedit highest dev agreement under the same edit (edit applied to the dev jets; no test jets used)
  oracle  highest test agreement under the edit (optimistic for the baseline; upper bound)
Per (model, program K 16 / 30, rule, edit): agreement difference prog - baseline, 95 % jet bootstrap (B 1000), exact McNemar.
Specificity: paired bootstrap of [lead on trans] - [lead on rdt] for each rule.
Output: $AI_RESULTS/jedi/b6_verify.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp.b6_pareto import indices, features, fit_mars, tree_bits, hgb_bits, holm  # noqa: E402

OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
EDITS = ['trans', 'm2h', 'pt09', 'rdt', 'rpd']


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def edit(x, e, seed):
    real = x[..., 0] > 0; y = x.copy()
    if e == 'trans':
        y[..., 1] += np.float32(0.06) * real
    elif e == 'm2h':
        y[..., 1] += np.float32(0.06) * real
        M2 = lambda a: (a[..., 0] * (a[..., 1] ** 2 + a[..., 2] ** 2)).sum(1).astype(np.float64)
        s = np.sqrt(M2(x) / np.maximum(M2(y), 1e-12)); y[..., 1:] = (y[..., 1:] * s[:, None, None]).astype(np.float32)
    elif e == 'pt09':
        y[..., 0] *= np.float32(0.9)
    elif e == 'rdt':
        ph = np.random.default_rng(seed).uniform(0, 2 * np.pi, len(x))
        y[..., 1] += (0.06 * np.cos(ph))[:, None].astype(np.float32) * real; y[..., 2] += (0.06 * np.sin(ph))[:, None].astype(np.float32) * real
    elif e == 'rpd':
        th = np.random.default_rng(seed).uniform(0, 2 * np.pi, real.shape)
        y[..., 1] += (0.06 * np.cos(th)).astype(np.float32) * real; y[..., 2] += (0.06 * np.sin(th)).astype(np.float32) * real
    return np.ascontiguousarray(y, np.float32)


def build():
    from interp.jedi_core import sample
    xF, yF, _ = sample(4000, 3701); xT, yT, _ = sample(2000, 3702)
    iT = indices(2000, 3702); excl = np.concatenate([indices(4000, 3701), indices(6000, 0), indices(4000, 2601)])
    keep = ~np.isin(iT, excl); xT, yT = np.asarray(xT[keep]), yT[keep]
    perm = np.random.default_rng(0).permutation(len(yF)); nf = 2 * len(yF) // 3
    x = dict(fit=np.asarray(xF[perm[:nf]]), dev=np.asarray(xF[perm[nf:]]), test=np.asarray(xT)); del xF
    y = dict(fit=yF[perm[:nf]], dev=yF[perm[nf:]], test=yT)
    for i, e in enumerate(EDITS):
        x[e] = edit(x['test'], e, 3703 + i); x['dev_' + e] = edit(x['dev'], e, 3713 + i)
    info = dict(N_fit=int(nf), N_dev=int(len(yF) - nf), N_test=int(len(yT)), removed_from_test=int((~keep).sum()))
    log('jets', info); return x, y, info


def logits(model, x):
    if model == 'jedi':
        from interp.jedi_core import Jedi
        net = Jedi(64); return {s: net.run(v)['logits'] for s, v in x.items()}
    from interp.x9_efn_pfn import Flow
    f = Flow('pfn_100-100-128_s0'); return {s: f.run(v)['logits'] for s, v in x.items()}


def run_model(model, x, y, Bn=1000):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import binomtest
    L = logits(model, x); nc = {s: v.argmax(1) for s, v in L.items()}
    acc = float((nc['test'] == y['test']).mean()); log(f'{model}: accuracy test {acc * 100:.2f} % (N {len(y["test"])})')
    flips = {e: float((nc[e] != nc['test']).mean()) for e in EDITS}; log(f'{model}: flips', {e: round(v * 100, 2) for e, v in flips.items()})
    X = {}; keys = None
    for s in x: X[s], keys = features(x[s])
    F = len(keys); Lf, Ld = L['fit'], L['dev']
    S = {}  # name -> dict(family, bits, calls={set: calls})

    def add(name, fam, bits, pred):
        S[name] = dict(family=fam, bits=float(bits), calls={s: np.asarray(pred(X[s]), np.int16) for s in X})
    progs, predict, _ = fit_mars(X, keys, Lf, Ld)
    for K in (8, 16, 30):
        parts, bits = progs[K]; add(f'prog{K}', 'prog', bits, lambda Z, p=parts: predict(p, Z).argmax(1))
    budget = progs[30][1]
    for d in range(2, 11):
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X['fit'], nc['fit']); add(f'tree_d{d}', 'tree', tree_bits(m.tree_, F, 1, True), m.predict)
        r = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X['fit'], Lf); add(f'treereg_d{d}', 'tree', tree_bits(r.tree_, F, 5), lambda Z, r=r: r.predict(Z).argmax(1))
    log(f'{model}: trees done')
    grid = {2: (5, 10, 20, 26), 3: (5, 10, 15), 4: (5, 10), 8: (2, 5)}
    for lv, its in grid.items():
        for lr in (0.1, 0.2, 0.5, 1.0):
            for it in its:
                m = HistGradientBoostingClassifier(max_leaf_nodes=lv, max_iter=it, learning_rate=lr, early_stopping=False, random_state=0).fit(X['fit'], nc['fit'])
                add(f'gbdt_L{lv}_T{it}_lr{lr}', 'gbdt', hgb_bits(m, F), m.predict)
                rs = [HistGradientBoostingRegressor(max_leaf_nodes=lv, max_iter=it, learning_rate=lr, early_stopping=False, random_state=0).fit(X['fit'], Lf[:, k]) for k in range(5)]
                add(f'gbdtreg_L{lv}_T{it}_lr{lr}', 'gbdt', sum(hgb_bits(r, F) for r in rs), lambda Z, rs=rs: np.stack([r.predict(Z) for r in rs], 1).argmax(1))
        log(f'{model}: gbdt leaves {lv} done')
    sc = StandardScaler().fit(X['fit'])
    for h in (2, 4):
        m = MLPRegressor(hidden_layer_sizes=(h,), max_iter=400, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=0).fit(sc.transform(X['fit']), Lf)
        add(f'mlp_h{h}', 'mlp', (F * h + h + h * 5 + 5) * 32, lambda Z, m=m: m.predict(np.nan_to_num(sc.transform(Z))).argmax(1))
    for n, v in S.items():
        v['same_call_dev'] = float((v['calls']['dev'] == nc['dev']).mean()); v['same_call_test'] = float((v['calls']['test'] == nc['test']).mean())

    def agree(v, e, pre=''):
        base = 'dev' if pre else 'test'; nb, na = nc[base], nc[pre + e]; sb, sa = v['calls'][base], v['calls'][pre + e]
        nf, sf = na != nb, sa != sb; return ((nf == sf) & (~nf | (sa == na))).astype(np.float32)
    for n, v in S.items():
        v['agree_test'] = {e: agree(v, e) for e in EDITS}; v['agree_dev'] = {e: float(agree(v, e, 'dev_').mean()) for e in EDITS}
    N = len(nc['test'])
    rng = np.random.default_rng(11); Cb = np.stack([np.bincount(rng.integers(0, N, N), minlength=N) for _ in range(Bn)]).astype(np.float32)
    table = []; cand = lambda pb: [n for n, v in S.items() if v['family'] in ('tree', 'gbdt', 'mlp') and v['bits'] <= pb]
    for pn in ('prog16', 'prog30'):
        P = S[pn]; el = cand(P['bits'])
        for e in EDITS:
            for rule in ('devsc', 'devedit', 'oracle', 'b6gbdt'):
                if rule == 'devsc': j = max(el, key=lambda n: S[n]['same_call_dev'])
                elif rule == 'devedit': j = max(el, key=lambda n: S[n]['agree_dev'][e])
                elif rule == 'oracle': j = max(el, key=lambda n: S[n]['agree_test'][e].mean())
                else:
                    j = max([n for n in el if n in ('gbdt_L4_T10_lr0.2', 'gbdtreg_L4_T10_lr0.2')] or [None], key=lambda n: S[n]['same_call_dev'] if n else 0)
                    if j is None: continue
                a, b = P['agree_test'][e], S[j]['agree_test'][e]; dd = Cb @ (a - b) / N
                n10, n01 = int(((a > 0) & (b == 0)).sum()), int(((a == 0) & (b > 0)).sum())
                pm = float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
                table.append(dict(model=model, program=pn, prog_bits=P['bits'], edit=e, rule=rule, baseline=j, base_bits=S[j]['bits'],
                                  prog_agree=float(a.mean()), base_agree=float(b.mean()), diff=float((a - b).mean()),
                                  ci=[float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))], p_mcnemar=pm, _boot=dd))
                log(f"{model} {pn} {e:5s} {rule:7s} vs {j:24s} ({S[j]['bits']:7.0f} b): prog {a.mean() * 100:5.2f} base {b.mean() * 100:5.2f} "
                    f"diff {(a - b).mean() * 100:+5.2f} [{np.percentile(dd, 2.5) * 100:+.2f}, {np.percentile(dd, 97.5) * 100:+.2f}] McNemar {pm:.3g}")
    spec = []
    for pn in ('prog16', 'prog30'):
        for rule in ('devsc', 'devedit', 'oracle', 'b6gbdt'):
            t = [r for r in table if r['program'] == pn and r['rule'] == rule and r['edit'] == 'trans']
            c = [r for r in table if r['program'] == pn and r['rule'] == rule and r['edit'] == 'rdt']
            if t and c:
                dd = t[0]['_boot'] - c[0]['_boot']; d = t[0]['diff'] - c[0]['diff']; se = float(dd.std())
                from scipy.stats import norm
                spec.append(dict(model=model, program=pn, rule=rule, trans_minus_rdt=d, ci=[float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                                 p=float(2 * norm.sf(abs(d) / se)) if se > 0 else 1.0))
    for r in table: r.pop('_boot')
    surr = [dict(name=n, family=v['family'], bits=v['bits'], same_call_dev=v['same_call_dev'], same_call_test=v['same_call_test'],
                 agree_dev=v['agree_dev'], agree_test={e: float(a.mean()) for e, a in v['agree_test'].items()}) for n, v in S.items()]
    return dict(acc_test=acc, flips=flips, prog30_bits=budget, surrogates=surr, matched=table, specificity=spec)


def main():
    x, y, info = build(); res = dict(procedure=__doc__, info=info, models={})
    for model in ('jedi', 'pfn'):
        res['models'][model] = run_model(model, x, y)
        (OUT / 'b6_verify.json').write_text(json.dumps(res, indent=1, default=float)); log(f'wrote {OUT}/b6_verify.json ({model})')
    allm = [r for m in res['models'].values() for r in m['matched'] if r['edit'] in ('trans', 'm2h', 'pt09')]
    for r, q in zip(allm, holm([r['p_mcnemar'] for r in allm])): r['p_holm'] = float(q)
    res['n_holm'] = len(allm)
    (OUT / 'b6_verify.json').write_text(json.dumps(res, indent=1, default=float)); log('done')


if __name__ == '__main__':
    main()
