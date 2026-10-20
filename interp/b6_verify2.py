"""B6 second verification (b6-verify2 / b6-verify3).

Question: does the corrected B6 reading (prog30 at ~13 kbit matches, but does not exceed, a dev-selected tree / GBDT of equal
bits on the 0.06 rad translation; keeps a ~+2 point lead on the M2-held translation in JEDI-linear) hold (a) on another fresh
jet sample and (b) for Deep Sets and the transformer, which the first verification (b6_verify) did not rerun?

Modes (python -m interp.b6_verify2 <mode>):
  fresh   JEDI-linear n64 + PFN 100-100-128 s0 (jax env). F = sample(4000, 4701) -> perm rng 0, 2/3 fit / 1/3 dev;
          T = sample(2000, 4702) minus every row of F, sample(6000, 0), sample(4000, 2601), sample(4000, 3701) (all earlier fit
          sets) and of sample(2000, 2602), sample(2000, 3702) (earlier test sets), so T is disjoint from every earlier B6 set.
          Same 98 observables, MARS program, bit costs, edits and baseline grid as b6_verify (imported).
          -> $AI_RESULTS/jedi/b6_verify2.json
  stored  all four networks on B6's stored jets and logits (b6_logits_<m>.npz, b6_pred_<m>.npz: prog16 / prog30 calls as fitted
          in B6). Fits the b6_verify baseline grid (CART d2-10, GBDT leaves {2,3,4,8} x lr {0.1,0.2,0.5,1.0} x iterations,
          MLP h2/h4) on B6's F-fit rows with B6's observables; selection on B6's dev rows by same call (devsc; dev edits
          were not stored, so no devedit rule) and the oracle rule.  -> $AI_RESULTS/jedi/b6_verify3.json
Per (model, program, rule, edit): agreement difference prog - baseline (per-jet: surrogate flips iff the network flips, to the
same call), 95 % jet bootstrap (B 1000), exact McNemar; Holm over all (trans, m2h, pt09) rows of the mode.  Specificity:
paired bootstrap of [lead on edit] - [lead on rdt] for edit in trans, m2h.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp.b6_pareto import indices, features, tree_bits, hgb_bits, holm, fit_mars  # noqa: E402
from interp import b6_verify as V  # noqa: E402

OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
ED = ['trans', 'm2h', 'pt09', 'rdt', 'rpd']


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def baselines(X, F, ncfit, Lf, budget):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    S = {}

    def add(name, fam, bits, pred):
        if bits <= budget * 1.0001: S[name] = dict(family=fam, bits=float(bits), calls={s: np.asarray(pred(X[s]), np.int16) for s in X})
    for d in range(2, 11):
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X['fit'], ncfit); add(f'tree_d{d}', 'tree', tree_bits(m.tree_, F, 1, True), m.predict)
        r = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X['fit'], Lf); add(f'treereg_d{d}', 'tree', tree_bits(r.tree_, F, 5), lambda Z, r=r: r.predict(Z).argmax(1))
    grid = {2: (5, 10, 20, 26), 3: (5, 10, 15), 4: (5, 10), 8: (2, 5)}
    for lv, its in grid.items():
        for lr in (0.1, 0.2, 0.5, 1.0):
            for it in its:
                m = HistGradientBoostingClassifier(max_leaf_nodes=lv, max_iter=it, learning_rate=lr, early_stopping=False, random_state=0).fit(X['fit'], ncfit)
                add(f'gbdt_L{lv}_T{it}_lr{lr}', 'gbdt', hgb_bits(m, F), m.predict)
                rs = [HistGradientBoostingRegressor(max_leaf_nodes=lv, max_iter=it, learning_rate=lr, early_stopping=False, random_state=0).fit(X['fit'], Lf[:, k]) for k in range(5)]
                add(f'gbdtreg_L{lv}_T{it}_lr{lr}', 'gbdt', sum(hgb_bits(r, F) for r in rs), lambda Z, rs=rs: np.stack([r.predict(Z) for r in rs], 1).argmax(1))
    sc = StandardScaler().fit(X['fit'])
    for h in (2, 4):
        m = MLPRegressor(hidden_layer_sizes=(h,), max_iter=400, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=0).fit(sc.transform(X['fit']), Lf)
        add(f'mlp_h{h}', 'mlp', (F * h + h + h * 5 + 5) * 32, lambda Z, m=m: m.predict(np.nan_to_num(sc.transform(Z))).argmax(1))
    return S


def compare(model, S, nc, progs, edits, rules, devedit, Bn=1000, seed=11):
    """S: name -> dict(family, bits, calls{set}); nc: network calls per set. Sets 'test', edits, 'dev' (+ 'dev_'+e if devedit)."""
    from scipy.stats import binomtest, norm
    def agree(c, e, base='test'):
        nb, na = nc[base], nc[e]; sb, sa = c[base], c[e]; nf, sf = na != nb, sa != sb
        return ((nf == sf) & (~nf | (sa == na))).astype(np.float32)
    for n, v in S.items():
        v['same_call_dev'] = float((v['calls']['dev'] == nc['dev']).mean()); v['same_call_test'] = float((v['calls']['test'] == nc['test']).mean())
        v['agree_test'] = {e: agree(v['calls'], e) for e in edits}
        if devedit: v['agree_dev'] = {e: float(agree(v['calls'], 'dev_' + e, 'dev').mean()) for e in edits}
    N = len(nc['test']); rng = np.random.default_rng(seed)
    Cb = np.stack([np.bincount(rng.integers(0, N, N), minlength=N) for _ in range(Bn)]).astype(np.float32)
    table = []
    for pn in progs:
        P = S[pn]; el = [n for n, v in S.items() if v['family'] in ('tree', 'gbdt', 'mlp') and v['bits'] <= P['bits']]
        for e in edits:
            for rule in rules:
                if rule == 'devsc': j = max(el, key=lambda n: S[n]['same_call_dev'])
                elif rule == 'devedit': j = max(el, key=lambda n: S[n]['agree_dev'][e])
                elif rule == 'oracle': j = max(el, key=lambda n: S[n]['agree_test'][e].mean())
                else:
                    c = [n for n in el if n in ('gbdt_L4_T10_lr0.2', 'gbdtreg_L4_T10_lr0.2')]
                    if not c: continue
                    j = max(c, key=lambda n: S[n]['same_call_dev'])
                a, b = P['agree_test'][e], S[j]['agree_test'][e]; dd = Cb @ (a - b) / N
                n10, n01 = int(((a > 0) & (b == 0)).sum()), int(((a == 0) & (b > 0)).sum())
                pm = float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
                table.append(dict(model=model, program=pn, prog_bits=P['bits'], edit=e, rule=rule, baseline=j, base_bits=S[j]['bits'],
                                  prog_agree=float(a.mean()), base_agree=float(b.mean()), never_flip=float(1 - (nc[e] != nc['test']).mean()),
                                  prog_same_call=P['same_call_test'], base_same_call=S[j]['same_call_test'],
                                  diff=float((a - b).mean()), ci=[float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                                  prog_only=n10, base_only=n01, p_mcnemar=pm, _boot=dd))
                log(f"{model} {pn} {e:5s} {rule:7s} vs {j:24s} ({S[j]['bits']:7.0f} b): prog {a.mean() * 100:5.2f} base {b.mean() * 100:5.2f} "
                    f"diff {(a - b).mean() * 100:+5.2f} [{np.percentile(dd, 2.5) * 100:+.2f}, {np.percentile(dd, 97.5) * 100:+.2f}] McNemar {pm:.3g}")
    spec = []
    for pn in progs:
        for rule in rules:
            c = [r for r in table if r['program'] == pn and r['rule'] == rule and r['edit'] == 'rdt']
            for e in ('trans', 'm2h'):
                t = [r for r in table if r['program'] == pn and r['rule'] == rule and r['edit'] == e]
                if t and c:
                    dd = t[0]['_boot'] - c[0]['_boot']; d = t[0]['diff'] - c[0]['diff']; se = float(dd.std())
                    spec.append(dict(model=model, program=pn, rule=rule, edit=e, edit_minus_rdt=d, ci=[float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                                     p=float(2 * norm.sf(abs(d) / se)) if se > 0 else 1.0))
    for r in table: r.pop('_boot')
    surr = [dict(name=n, family=v['family'], bits=v['bits'], same_call_dev=v['same_call_dev'], same_call_test=v['same_call_test'],
                 agree_test={e: float(a.mean()) for e, a in v['agree_test'].items()}) for n, v in S.items()]
    return table, spec, surr


def finish(res, name):
    allm = [r for m in res['models'].values() for r in m['matched'] if r['edit'] in ('trans', 'm2h', 'pt09')]
    for r, q in zip(allm, holm([r['p_mcnemar'] for r in allm])): r['p_holm'] = float(q)
    res['n_holm'] = len(allm)
    (OUT / name).write_text(json.dumps(res, indent=1, default=float)); log(f'wrote {OUT / name}')


def mode_fresh():
    from interp.jedi_core import sample
    xF, yF, _ = sample(4000, 4701); xT, yT, _ = sample(2000, 4702)
    iT = indices(2000, 4702)
    excl = np.concatenate([indices(4000, 4701), indices(6000, 0), indices(4000, 2601), indices(4000, 3701), indices(2000, 2602), indices(2000, 3702)])
    keep = ~np.isin(iT, excl); xT, yT = np.asarray(xT[keep]), yT[keep]
    perm = np.random.default_rng(0).permutation(len(yF)); nf = 2 * len(yF) // 3
    x = dict(fit=np.asarray(xF[perm[:nf]]), dev=np.asarray(xF[perm[nf:]]), test=np.asarray(xT)); del xF
    y = dict(fit=yF[perm[:nf]], dev=yF[perm[nf:]], test=yT)
    for i, e in enumerate(ED):
        x[e] = V.edit(x['test'], e, 4703 + i); x['dev_' + e] = V.edit(x['dev'], e, 4713 + i)
    info = dict(N_fit=int(nf), N_dev=int(len(yF) - nf), N_test=int(len(yT)), removed_from_test=int((~keep).sum()), seeds=[4701, 4702])
    log('jets', info); res = dict(procedure=__doc__, mode='fresh', info=info, models={})
    for model in ('jedi', 'pfn'):
        L = V.logits(model, x); nc = {s: v.argmax(1) for s, v in L.items()}
        acc = float((nc['test'] == y['test']).mean()); flips = {e: float((nc[e] != nc['test']).mean()) for e in ED}
        log(f'{model}: acc {acc * 100:.2f} % flips', {e: round(v * 100, 2) for e, v in flips.items()})
        X = {}; keys = None
        for s in x: X[s], keys = features(x[s])
        F = len(keys); progs, predict, _ = fit_mars(X, keys, L['fit'], L['dev'])
        S = {}
        for K in (16, 30):
            parts, bits = progs[K]; S[f'prog{K}'] = dict(family='prog', bits=float(bits), calls={s: np.asarray(predict(parts, X[s]).argmax(1), np.int16) for s in X})
        S.update(baselines(X, F, nc['fit'], L['fit'], progs[30][1])); log(f'{model}: {len(S)} surrogates')
        table, spec, surr = compare(model, S, nc, ['prog16', 'prog30'], ED, ('devsc', 'devedit', 'oracle', 'b6gbdt'), True)
        res['models'][model] = dict(acc_test=acc, flips=flips, prog30_bits=progs[30][1], matched=table, specificity=spec, surrogates=surr)
        finish(res, 'b6_verify2.json')


def mode_stored():
    from interp.b6_pareto import build
    B = build(); X = {}; keys = None
    for s in ['fit', 'dev', 'test'] + ED: X[s], keys = features(B['x'][s])
    F = len(keys); res = dict(procedure=__doc__, mode='stored', info=B['info'], models={})
    for model in ('jedi', 'dsets', 'pfn', 'tf'):
        Z = np.load(OUT / f'b6_logits_{model}.npz'); P = np.load(OUT / f'b6_pred_{model}.npz')
        assert np.array_equal(Z['y_test'], B['y']['test'])
        nc = {s: Z[s].argmax(1) for s in ['fit', 'dev', 'test'] + ED}
        meta = {m['name']: m for m in json.loads((OUT / f'b6_pred_{model}.json').read_text())['models']}
        S = {pn: dict(family='prog', bits=float(meta[pn]['bits']), calls={s: np.asarray(P[f'{pn}|{s}'], np.int16) for s in ['fit', 'dev', 'test'] + ED})
             for pn in ('prog16', 'prog30')}
        Xs = {s: X[s] for s in ['fit', 'dev', 'test'] + ED}
        S.update(baselines(Xs, F, nc['fit'], Z['fit'].astype(np.float64), S['prog30']['bits'])); log(f'{model}: {len(S)} surrogates')
        table, spec, surr = compare(model, S, nc, ['prog16', 'prog30'], ED, ('devsc', 'oracle', 'b6gbdt'), False)
        res['models'][model] = dict(acc_test=float((nc['test'] == B['y']['test']).mean()), prog30_bits=S['prog30']['bits'], matched=table, specificity=spec, surrogates=surr)
        finish(res, 'b6_verify3.json')


if __name__ == '__main__':
    m = sys.argv[1]; log('mode', m)
    mode_fresh() if m == 'fresh' else mode_stored()
    log('done')
