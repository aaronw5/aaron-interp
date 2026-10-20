"""B6w verify (skeptic): rerun of the key B6w comparisons on fresh jet samples, plus a pool-vs-method control.

Jets: F = sample(per_class 4000, seed 5301) split by rng 0 into fit 2/3 / dev 1/3; T = sample(2000, 5401) minus every row of F.
Edits on T as in interp.b6w_symreg.build (same code, new seeds): trans, m2h, A13, pt09, rcn; controls rdt, rcr.
Edit sanity checks on the edited arrays: padded slots unchanged, trans moves only Δη of real particles by 0.06, m2h holds
ΣpT and ΣpT r² about the stored axis, A13 changes only slots 13 and 14, rcn residual centroid.
Surrogates (fitted to the logits on F-fit):
  prog      MARS on the 98 rank-window (RW) observables, K 4, 8, 16, 30 (fit_prog of b6w_symreg)
  cart, gbdt32, gbdtq   on RW (as b6w; GBDT grid leaves 2-16 x lr 0.2/0.5 x iters 5/10/25; q 32, 8, 6)
  gpl_hand  gplearn on the 10 standard observables (pop 2000, 20 generations, parsimony 3e-4, 1e-3, 3e-3; 4,000 fit rows)
  pysr_hand PySR on the 10 standard observables (fit_pysr of b6w_symreg; --pysr 1)
  control (new): the same fitting methods on the standard-observable pool
      prog_hand  MARS on HAND, K 4, 8, 16, 30;  gbdt_hand  GBDT (32-bit and 8-bit) on HAND
  -> separates "observable pool" from "fitting method" as the cause of the symbolic-regression deficit.
  shuffled  GBDT on row-shuffled logits.
Metrics and tests as b6w eval: same call on T; edit agreement; never-flip = 1 - network flip rate; matched comparison at
K 4/8/16/30 against the dev-same-call-selected member of each family with <= program bits; paired jet bootstrap (B 1000),
exact McNemar, Holm over the comparisons run; specificity [trans lead] - [rdt lead].
Modes: prep --model M (jax env; writes /tmp/b6wv_<M>.npz inside the pod), fit --model M [--pysr 1] (symreg venv)
-> $AI_RESULTS/jedi/b6w_verify_<M>.json
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
EDITS = ['trans', 'm2h', 'A13', 'pt09', 'rcn']
CTRL = ['rdt', 'rcr']
SETS = ['test'] + EDITS + CTRL
ALLSETS = ['fit', 'dev'] + SETS
SF, ST = 5301, 5401
HAND = ['mass', 'sd_mass', 'tau21', 'tau32', 'D2', 'n_particles', 'sum_pt']


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def build():
    from interp.jedi_core import sample
    from interp.b6w_symreg import indices
    xF, yF, _ = sample(4000, SF); xT, yT, _ = sample(2000, ST)
    iF, iT = indices(4000, SF), indices(2000, ST)
    keep = ~np.isin(iT, iF); xT, yT = np.asarray(xT[keep], np.float32), yT[keep]
    perm = np.random.default_rng(0).permutation(len(yF)); nf = 2 * len(yF) // 3
    x = dict(fit=np.asarray(xF[perm[:nf]]), dev=np.asarray(xF[perm[nf:]]), test=xT)
    y = dict(fit=yF[perm[:nf]], dev=yF[perm[nf:]], test=yT)
    real = xT[..., 0] > 0; nreal = real.sum(1); rf = real.astype(np.float32)
    M2 = lambda a: (a[..., 0].astype(np.float64) * (a[..., 1].astype(np.float64) ** 2 + a[..., 2].astype(np.float64) ** 2)).sum(1)
    e = xT.copy(); e[..., 1] += np.float32(0.06) * rf; x['trans'] = e
    s = np.sqrt(M2(xT) / np.maximum(M2(e), 1e-12)); m = e.copy(); m[..., 1:] = (e[..., 1:] * s[:, None, None]).astype(np.float32); x['m2h'] = m
    J13 = np.flatnonzero(nreal >= 15); a = xT[J13].copy(); a[:, [13, 14], 1:] = xT[J13][:, [14, 13], 1:]; x['A13'] = a
    p = xT.copy(); p[..., 0] *= np.float32(0.9); x['pt09'] = p
    pt = xT[..., 0].astype(np.float64); c = np.stack([(pt * xT[..., 1]).sum(1), (pt * xT[..., 2]).sum(1)], 1) / np.maximum(pt.sum(1), 1e-12)[:, None]
    q = xT.copy(); q[..., 1] -= (c[:, 0:1] * rf).astype(np.float32); q[..., 2] -= (c[:, 1:2] * rf).astype(np.float32); x['rcn'] = q
    ph = np.random.default_rng(5404).uniform(0, 2 * np.pi, len(xT)); d = xT.copy()
    d[..., 1] += (0.06 * np.cos(ph))[:, None].astype(np.float32) * rf; d[..., 2] += (0.06 * np.sin(ph))[:, None].astype(np.float32) * rf; x['rdt'] = d
    cn = np.hypot(c[:, 0], c[:, 1]); ph3 = np.random.default_rng(5406).uniform(0, 2 * np.pi, len(xT)); v = xT.copy()
    v[..., 1] += (cn * np.cos(ph3))[:, None].astype(np.float32) * rf; v[..., 2] += (cn * np.sin(ph3))[:, None].astype(np.float32) * rf; x['rcr'] = v
    for k in list(x): x[k] = np.ascontiguousarray(x[k], dtype=np.float32)
    base = {k: np.arange(len(yT)) for k in SETS}; base['A13'] = J13
    for k in SETS: y[k] = yT[base[k]]
    # sanity checks of the edits
    pad = ~real; chk = {}
    for k in ('trans', 'm2h', 'pt09', 'rcn', 'rdt', 'rcr'):
        chk[f'{k}_padding_maxabs_change'] = float(np.abs(x[k][pad] - xT[pad]).max()) if pad.any() else 0.0
    dtr = x['trans'] - xT
    chk['trans_deta_real'] = [float(dtr[..., 1][real].min()), float(dtr[..., 1][real].max())]
    chk['trans_other_maxabs'] = float(max(np.abs(dtr[..., 0]).max(), np.abs(dtr[..., 2]).max()))
    chk['m2h_dM2_rel_maxabs'] = float(np.max(np.abs(M2(x['m2h']) - M2(xT)) / np.maximum(M2(xT), 1e-12)))
    chk['m2h_dM0_maxabs'] = float(np.abs(x['m2h'][..., 0].sum(1) - xT[..., 0].sum(1)).max())
    dA = np.abs(x['A13'] - xT[J13]); dA[:, [13, 14]] = 0; chk['A13_other_slots_maxabs'] = float(dA.max())
    chk['A13_slots_13_14_both_real'] = bool(real[J13][:, [13, 14]].all())
    chk['pt_sorted_desc_frac'] = float(np.mean(np.all(np.diff(xT[..., 0], axis=1) <= 1e-6, axis=1)))
    info = dict(seeds=dict(F=SF, T=ST), N_fit=int(nf), N_dev=int(len(yF) - nf), N_test=int(len(yT)), removed_from_test=int((~keep).sum()),
                N_A13=int(len(J13)), test_rows_in_F=int(np.isin(iT[keep], iF).sum()), rcn_offset_mean_rad=float(cn.mean()), checks=chk)
    log('jets', json.dumps(info))
    return dict(x=x, y=y, base=base, info=info)


def mode_prep(model):
    from interp.b6_pareto import features
    from jetdistill.observables import compute
    B = build(); X = B['x']
    if model == 'jedi':
        from interp.jedi_core import Jedi
        net = Jedi(64); L = {s: net.run(X[s])['logits'] for s in ALLSETS}
    else:
        from interp.x9_efn_pfn import Flow
        f = Flow('pfn_100-100-128_s0'); L = {s: f.run(X[s])['logits'] for s in ALLSETS}
    acc = {s: float((np.asarray(L[s]).argmax(1) == B['y'][s]).mean()) for s in ('fit', 'dev', 'test')}
    log(model, 'accuracy', acc)
    out = {}
    for s in ALLSETS:
        xx = X[s]; out[f'L_{s}'] = np.asarray(L[s], np.float64); out[f'y_{s}'] = B['y'][s]
        rw, keys = features(xx); out[f'rw_{s}'] = rw.astype(np.float64)
        q = compute(np.asarray(xx, np.float64), 64, ids=HAND)
        pt = np.asarray(xx[..., 0], np.float64); r = np.hypot(xx[..., 1], xx[..., 2]).astype(np.float64)
        out[f'hand_{s}'] = np.stack([np.asarray(q[k], np.float64) for k in HAND] + [(pt * r ** n).sum(1) for n in (1, 2, 3)], 1)
        log(f'observables {s}: {len(xx)} jets')
    np.savez(f'/tmp/b6wv_{model}.npz', **out, J13=B['base']['A13'], info=json.dumps(dict(B['info'], acc=acc)))
    log('prep done')


def mode_fit(model, use_pysr):
    import warnings
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from scipy.stats import binomtest
    from interp.b6_pareto import tree_bits, hgb_bits
    from interp.b6w_symreg import fit_prog, quantiser, hgb_bits_q, fit_gplearn, fit_pysr, holm
    warnings.filterwarnings('ignore')
    Z = np.load(f'/tmp/b6wv_{model}.npz'); info = json.loads(str(Z['info']))
    L = {s: Z[f'L_{s}'] for s in ALLSETS}; nc = {s: L[s].argmax(1) for s in ALLSETS}
    RW = {s: Z[f'rw_{s}'] for s in ALLSETS}; HD = {s: Z[f'hand_{s}'] for s in ALLSETS}
    F, FH = RW['fit'].shape[1], HD['fit'].shape[1]; Lf, Ld, cf = L['fit'], L['dev'], nc['fit']
    keysR = [f'r{i}' for i in range(F)]; keysH = [f'h{i}' for i in range(FH)]
    S = []    # (name, family, bits, calls per set)

    def add(name, fam, bits, calls):
        S.append(dict(name=name, family=fam, bits=float(bits), calls={s: np.asarray(c, np.int8) for s, c in calls.items()}))
        log(f'{name:30s} {fam:10s} bits {bits:8.0f} same call dev {(calls["dev"] == nc["dev"]).mean() * 100:5.1f} test {(calls["test"] == nc["test"]).mean() * 100:5.1f}')

    for tag, D, keys, fam in (('prog', RW, keysR, 'prog'), ('proghand', HD, keysH, 'prog_hand')):
        progs, pred = fit_prog(D, keys, Lf, Ld, 30, (4, 8, 16, 30), tag)
        for K, (parts, bits, nterm) in progs.items():
            add(f'{tag}{K}', fam, bits, {s: pred(parts, D[s]).argmax(1) for s in ALLSETS}); S[-1]['K'] = K
    for d in range(2, 11):
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(RW['fit'], cf)
        add(f'tree_d{d}', 'cart', tree_bits(m.tree_, F, 1, True), {s: m.predict(RW[s]) for s in ALLSETS})
        r = DecisionTreeRegressor(max_depth=d, random_state=0).fit(RW['fit'], Lf)
        add(f'treereg_d{d}', 'cart', tree_bits(r.tree_, F, 5), {s: r.predict(RW[s]).argmax(1) for s in ALLSETS})
    for pool, D, nin in (('rw', RW, F), ('hand', HD, FH)):
        for q in ((32, 8, 6) if pool == 'rw' else (32, 8)):
            Xq = D if q == 32 else (lambda qf: {s: qf(D[s]) for s in ALLSETS})(quantiser(D['fit'], q))
            fam = ('gbdt32' if q == 32 else 'gbdtq') if pool == 'rw' else 'gbdt_hand'
            bf = (lambda m: hgb_bits(m, nin)) if q == 32 else (lambda m, q=q: hgb_bits_q(m, nin, q))
            for lv in (2, 4, 8, 16):
                for lr in (0.2, 0.5):
                    for it in (5, 10, 25):
                        kw = dict(max_leaf_nodes=lv, max_iter=it, learning_rate=lr, early_stopping=False, random_state=0)
                        m = HistGradientBoostingClassifier(**kw).fit(Xq['fit'], cf)
                        add(f'gbdt{pool}{q}_L{lv}_T{it}_lr{lr}', fam, bf(m), {s: m.predict(Xq[s]) for s in ALLSETS})
                        rs = [HistGradientBoostingRegressor(**kw).fit(Xq['fit'], Lf[:, k]) for k in range(5)]
                        add(f'gbdtreg{pool}{q}_L{lv}_T{it}_lr{lr}', fam, sum(bf(r) for r in rs), {s: np.stack([r.predict(Xq[s]) for r in rs], 1).argmax(1) for s in ALLSETS})
    pr = np.random.default_rng(5407).permutation(len(Lf))
    rs = [HistGradientBoostingRegressor(max_leaf_nodes=8, max_iter=25, learning_rate=0.5, early_stopping=False, random_state=0).fit(RW['fit'], Lf[pr, k]) for k in range(5)]
    add('shuf_gbdtreg', 'shuffled', sum(hgb_bits(r, F) for r in rs), {s: np.stack([r.predict(RW[s]) for r in rs], 1).argmax(1) for s in ALLSETS})
    sel = np.random.default_rng(5408).choice(len(Lf), 4000, replace=False)
    mu, sd = HD['fit'].mean(0), np.where(HD['fit'].std(0) > 0, HD['fit'].std(0), 1.0); HDs = {s: np.nan_to_num((HD[s] - mu) / sd) for s in ALLSETS}
    for pars in (0.0003, 0.001, 0.003):
        pred, bits, inf = fit_gplearn(HDs, Lf, sel, pars, 5410, 'gplearn hand', 2000, 20)
        add(f'gpl_hand_p{pars}', 'gpl_hand', bits, {s: pred(HDs[s]).argmax(1) for s in ALLSETS})
    pysr_status = 'not run'
    if use_pysr:
        try:
            for cap, pred, bits, eqs in fit_pysr(HDs, Lf, sel, log):
                add(f'pysr_hand_c{cap}', 'pysr_hand', bits, {s: pred(HDs[s]).argmax(1) for s in ALLSETS})
            pysr_status = 'ok'
        except Exception as ex:  # noqa: BLE001
            pysr_status = f'failed: {ex}'[:300]; log('PySR failed', ex)
    # ---------------------------------------------------------------- evaluation
    J13 = Z['J13']; base = {s: (J13 if s == 'A13' else np.arange(len(nc['test']))) for s in SETS}
    Bn = 1000; rng = np.random.default_rng(11); cnt = {}
    for s in SETS:
        n = len(nc[s]); cnt.setdefault(n, np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(Bn)]).astype(np.float32))
    C = {s: cnt[len(nc[s])] for s in SETS}
    net = dict(acc=info['acc'], flip={}, never_flip={})
    AG = {}
    for e in EDITS + CTRL:
        bi = base[e]; nb, na = nc['test'][bi], nc[e]; nf_ = na != nb
        bf_ = C[e] @ nf_.astype(np.float32) / len(nf_)
        net['flip'][e] = [float(nf_.mean()), float(np.percentile(bf_, 2.5)), float(np.percentile(bf_, 97.5))]; net['never_flip'][e] = float(1 - nf_.mean())
        AG[e] = np.stack([((nf_ == (r['calls'][e] != r['calls']['test'][bi])) & (~nf_ | (r['calls'][e] == na))) for r in S]).astype(np.float32)
    SC = np.stack([(r['calls']['test'] == nc['test']) for r in S]).astype(np.float32)
    rows = []
    for i, r in enumerate(S):
        rows.append(dict(name=r['name'], family=r['family'], bits=r['bits'], K=r.get('K'), same_call_dev=float((r['calls']['dev'] == nc['dev']).mean()),
                         same_call=float(SC[i].mean()), agree={e: float(AG[e][i].mean()) for e in EDITS + CTRL},
                         flip={e: float((r['calls'][e] != r['calls']['test'][base[e]]).mean()) for e in EDITS + CTRL}))
    bits = np.array([r['bits'] for r in rows]); fam = np.array([r['family'] for r in rows])
    groups = dict(pooled_rw=('cart', 'gbdt32', 'gbdtq'), gpl_hand=('gpl_hand',), pysr_hand=('pysr_hand',), gbdt_hand=('gbdt_hand',),
                  prog_hand=('prog_hand',))
    matched, spec = [], []
    for i, r in enumerate(rows):
        if r['family'] != 'prog': continue
        for g, fams in groups.items():
            el = [j for j in range(len(rows)) if fam[j] in fams and bits[j] <= bits[i] * 1.0001]
            if not el: continue
            j = max(el, key=lambda j: rows[j]['same_call_dev'])
            for e in ['same_call'] + EDITS + CTRL:
                a_, b_ = (SC[i], SC[j]) if e == 'same_call' else (AG[e][i], AG[e][j]); Ce = C['test'] if e == 'same_call' else C[e]
                dd = Ce @ (a_ - b_) / len(a_); n10, n01 = int(((a_ > 0) & (b_ == 0)).sum()), int(((a_ == 0) & (b_ > 0)).sum())
                pm = float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
                matched.append(dict(program=r['name'], K=r['K'], prog_bits=r['bits'], group=g, baseline=rows[j]['name'], base_bits=rows[j]['bits'], edit=e,
                                    prog=float(a_.mean()), base=float(b_.mean()), diff=float((a_ - b_).mean()),
                                    ci=[float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))], p_mcnemar=pm,
                                    never_flip=net['never_flip'].get(e)))
            a_, b_, c_, d_ = AG['trans'][i], AG['trans'][j], AG['rdt'][i], AG['rdt'][j]
            sp = C['trans'] @ (a_ - b_) / len(a_) - C['rdt'] @ (c_ - d_) / len(c_)
            spec.append(dict(program=r['name'], group=g, baseline=rows[j]['name'], trans_lead=float((a_ - b_).mean()), rdt_lead=float((c_ - d_).mean()),
                             lead_minus_control=float((a_ - b_).mean() - (c_ - d_).mean()), ci=[float(np.percentile(sp, 2.5)), float(np.percentile(sp, 97.5))],
                             p_boot=float(min(1.0, 2 * min((sp <= 0).mean(), (sp >= 0).mean()) + 1.0 / Bn))))
    prim = [t for t in matched if t['edit'] in EDITS]
    for t, q in zip(prim, holm([t['p_mcnemar'] for t in prim])): t['p_holm'] = float(q)
    for t in matched:
        if t['edit'] in ('same_call', 'trans', 'm2h', 'rdt', 'A13', 'rcn'):
            log(f"{t['program']:8s} vs {t['group']:10s} {t['baseline']:30s} ({t['base_bits']:6.0f} b) {t['edit']:9s} {t['prog'] * 100:5.1f} vs {t['base'] * 100:5.1f} "
                f"= {t['diff'] * 100:+6.2f} [{t['ci'][0] * 100:+.2f}, {t['ci'][1] * 100:+.2f}] nf {'' if t['never_flip'] is None else round(t['never_flip'] * 100, 1)} Holm {t.get('p_holm', float('nan')):.2g}")
    for t in spec:
        log(f"spec {t['program']:8s} vs {t['group']:10s}: trans lead {t['trans_lead'] * 100:+.2f} rdt lead {t['rdt_lead'] * 100:+.2f} diff {t['lead_minus_control'] * 100:+.2f} "
            f"[{t['ci'][0] * 100:+.2f}, {t['ci'][1] * 100:+.2f}] p {t['p_boot']:.3g}")
    log('network', json.dumps(net))
    res = dict(procedure=__doc__, model=model, info=info, pysr=pysr_status, network=net, surrogates=rows, matched=matched, specificity=spec,
               n_holm=len(prim))
    (OUT / f'b6w_verify_{model}.json').write_text(json.dumps(res, indent=1, default=float)); log(f'wrote {OUT}/b6w_verify_{model}.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['prep', 'fit']); ap.add_argument('--model', default='jedi', choices=['jedi', 'pfn'])
    ap.add_argument('--pysr', type=int, default=0)
    a = ap.parse_args(); log('args', vars(a))
    if a.mode == 'prep': mode_prep(a.model)
    else: mode_fit(a.model, bool(a.pysr))
    log('done')
