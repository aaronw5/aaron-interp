"""B7 verification (skeptical rerun on fresh jets + leave-one-in decomposition of the D(dc) gain).

Same pool construction as interp.b7_counterfactual_programs.mode_pool, but every jedi_core.sample / b6w_symreg.indices seed is
offset by +1000 (natural fit 4901, natural test 4902, base C 4903); the pool is written to a pod-local temp directory, never to
$AI_RESULTS/jedi/b7_pool.*.  Fit conditions: nat, all, lofo_cf, lofo_D, lofo_trans, shuf, rnd (as B7) plus leave-one-in
only_trans / only_cf / only_rot (natural + that single family) to locate which added family moves held-out D(dc) agreement.
Surrogates: prog K 8/16/30/50 (MARS on the 98 rank-window inputs, B7 settings), the B7 quantised-GBDT grid (for gbdt@K) and mlp32.
Pre-specified tests (cluster bootstrap over base jets, B = 5000, Holm over the list): agreement and kappa differences.
Extra checks: padded slots have zero pT and zero angles (so dil / rot leave padding untouched); per-set flip rates.
Output: $AI_RESULTS/jedi/b7_verify.json, b7_verify_pool.json.
"""
import argparse, json, os, sys, time, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('KERAS_BACKEND', 'jax')
from interp import b7_counterfactual_programs as B7  # noqa: E402

OUTV = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'; OUTV.mkdir(parents=True, exist_ok=True)
OFF = 1000
PADCHK = {}
log = B7.log


def patch_seeds():
    import interp.jedi_core as JC, interp.b6w_symreg as SR
    s0, i0 = JC.sample, SR.indices

    def samp(per_class=5000, seed=0, path=None):
        x, y, Q = s0(per_class, seed + OFF, path)
        pad = x[..., 0] <= 0
        PADCHK[seed + OFF] = dict(n_jets=int(len(x)), n_pad_slots=int(pad.sum()),
                                  max_abs_angle_on_padding=float(np.abs(x[..., 1:][pad]).max()) if pad.any() else 0.0)
        log('padding check', seed + OFF, json.dumps(PADCHK[seed + OFF]))
        return x, y, Q
    JC.sample = samp
    SR.indices = lambda per_class, seed: i0(per_class, seed + OFF)


def build(D, cond, rng_seed=11):
    """as B7.build_condition, plus 'only_<f>' = natural + family f only"""
    if not cond.startswith('only_'): return B7.build_condition(D, cond, rng_seed)
    rng = np.random.default_rng(rng_seed); f = cond[5:]
    parts = [('nat_fit', np.arange(len(D.L['nat_fit'])))]
    rows = [(s, np.flatnonzero(D.fit_half[D.base[s]])) for s in B7.FAMS[f] if s in D.base]
    n = sum(len(r) for _, r in rows); keep = min(n, B7.CAP)
    allr = np.concatenate([np.stack([np.full(len(r), i), r], 1) for i, (_, r) in enumerate(rows)]); pick = allr[rng.choice(n, keep, replace=False)]
    for i, (s, _) in enumerate(rows): parts.append((s, pick[pick[:, 0] == i, 1]))
    RW = np.concatenate([D.rw[s][r] for s, r in parts]); L = np.concatenate([D.L[s][r] for s, r in parts])
    return RW, None, L, dict(N_fit=int(len(L)), N_added=int(keep), by_set={s: int(len(r)) for s, r in parts})


def fit_lean(D, cond, a):
    import warnings
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from interp.b6w_symreg import quantiser, hgb_bits_q
    warnings.filterwarnings('ignore')
    RW, _, L, cinfo = build(D, cond); log(f'condition {cond}: {json.dumps(cinfo)}'); S = {}; F = RW.shape[1]
    progs, pred = B7.mars_fit(RW, L, D.rw['nat_dev'], D.L['nat_dev'], D.keys, max(a.prog_k), a.prog_k, f'{cond} RW', a.nsel)
    for K, (parts, bits, nt) in progs.items():
        S[f'prog{K}'] = (lambda s, parts=parts: pred(parts, D.rw[s]), bits)
    for q in (4, 8):
        qf = quantiser(RW, q)
        for lv in (4, 8):
            for it in (3, 5, 10, 30):
                rs = [HistGradientBoostingRegressor(max_leaf_nodes=lv, max_iter=it, learning_rate=0.5, early_stopping=False, random_state=0).fit(qf(RW), L[:, k]) for k in range(5)]
                S[f'gbdt{q}_L{lv}_T{it}'] = (lambda s, rs=rs, qf=qf: np.stack([r.predict(qf(D.rw[s])) for r in rs], 1), sum(hgb_bits_q(r, F, q) for r in rs))
    mu, sd = RW.mean(0), np.where(RW.std(0) > 0, RW.std(0), 1.0)
    m = MLPRegressor(hidden_layer_sizes=(32,), max_iter=300, early_stopping=True, random_state=0).fit((RW - mu) / sd, L)
    S['mlp32'] = (lambda s, m=m: m.predict((D.rw[s] - mu) / sd), 0.0)
    return S, cinfo


def kappa_vec(C, inv, ng, nf, sf): return B7.kappa_boot(C, inv, ng, nf, sf)


def main(a):
    a.prog_k = (8, 16, 30, 50) if not a.smoke else (2, 4)
    tmp = Path(tempfile.mkdtemp(prefix='b7v_'))
    patch_seeds(); B7.OUT = tmp
    B7.mode_pool(a)
    pool = tmp / f'b7{"smoke" if a.smoke else ""}_pool.npz'
    D = B7.Data(pool); pinfo = json.loads(Path(str(pool).replace('.npz', '.json')).read_text())
    pinfo['padding_check'] = {str(k): v for k, v in PADCHK.items()}
    (OUTV / 'b7_verify_pool.json').write_text(json.dumps(pinfo, indent=1, default=float))
    log('pool loaded; sets', D.sets)
    FR = {f: B7.family_rows(D, f) for f in B7.FAMS}
    FR['cfmin'] = [(s, r) for s, r in FR['cf'] if s == 'cf']; FR['Dmin'] = [(s, r) for s, r in FR['D'] if s.startswith('D')]
    views = ['cfmin', 'Dmin', 'D', 'trans', 'rot']
    NET = {}
    for f in views:
        rows = FR[f]
        nb = np.concatenate([D.call[D.bset[s]][D.base[s][r]] for s, r in rows]); ne = np.concatenate([D.call[s][r] for s, r in rows])
        grp = np.concatenate([D.base[s][r] for s, r in rows]); sub = np.concatenate([np.full(len(r), s) for s, r in rows])
        C, inv, ng = B7.boot_counts(grp, a.B, 27)
        NET[f] = dict(nb=nb, ne=ne, nf=ne != nb, C=C, inv=inv, ng=ng, sub=sub)
        log(f'view {f}: {len(nb)} rows / {ng} base jets; network flip {np.mean(ne != nb) * 100:.2f} %; '
            + ', '.join(f'{s} {np.mean((ne != nb)[sub == s]) * 100:.1f}' for s, _ in rows))
    nt = D.call['nat_test']
    conds = ['nat', 'lofo_D', 'rnd', 'only_trans', 'only_cf', 'only_rot', 'lofo_trans', 'all', 'shuf', 'lofo_cf']
    if a.smoke: conds = conds[:4]
    RES = dict(doc=__doc__, offset=OFF, pool=pinfo, network={f: dict(N=int(len(NET[f]['nf'])), N_base=int(NET[f]['ng']), flip=float(NET[f]['nf'].mean()),
                                                                     by_set={s: float(NET[f]['nf'][NET[f]['sub'] == s].mean()) for s, _ in FR[f]}) for f in views},
               rows=[], tests=[], matched={}, conditions={})
    PER = {}
    for cond in conds:
        t = time.time(); S, cinfo = fit_lean(D, cond, a); RES['conditions'][cond] = cinfo
        sc_dev = {}
        for name, (fn, bits) in S.items():
            cache = {}
            def P(s):
                if s not in cache: cache[s] = np.asarray(fn(s), np.float64)
                return cache[s]
            sc_dev[name] = float((P('nat_dev').argmax(1) == D.call['nat_dev']).mean())
            row = dict(cond=cond, model=name, bits=float(bits), same_call=float((P('nat_test').argmax(1) == nt).mean()), same_call_dev=sc_dev[name], fam={})
            PER[(cond, name)] = dict(nat=P('nat_test').argmax(1) == nt)
            for f in views:
                n = NET[f]; rows = FR[f]
                sb = np.concatenate([P(D.bset[s])[D.base[s][r]] for s, r in rows]).argmax(1)
                se = np.concatenate([P(s)[r] for s, r in rows]).argmax(1); sf = sb != se; ag = sf == n['nf']
                bk = kappa_vec(n['C'], n['inv'], n['ng'], n['nf'], sf)
                row['fam'][f] = dict(agree=float(ag.mean()), kappa=[B7.kappa_of(n['nf'], sf)] + B7.ci(bk), flip=float(sf.mean()))
                PER[(cond, name)][f] = (ag, sf)
            RES['rows'].append(row)
        RES['matched'][cond] = {}
        for K in a.prog_k:
            pb = S[f'prog{K}'][1]; ok = [n_ for n_ in S if n_.startswith('gbdt') and S[n_][1] <= pb]
            if ok: RES['matched'][cond][K] = max(ok, key=lambda n_: sc_dev[n_])
        log(f'condition {cond} done {time.time() - t:.0f} s: ' + '; '.join(
            f'{r["model"]} sc {r["same_call"] * 100:.1f} ' + ' '.join(f'{f}:k{r["fam"][f]["kappa"][0]:.3f}' for f in views)
            for r in RES['rows'] if r['cond'] == cond and r['model'] in ('prog30', 'prog50', 'mlp32')))
        (OUTV / 'b7_verify.json').write_text(json.dumps(RES, indent=1, default=float))

    def test(label, f, c1, m1, c2, m2):
        if (c1, m1) not in PER or (c2, m2) not in PER: return
        n = NET[f]; a1, s1 = PER[(c1, m1)][f]; a2, s2 = PER[(c2, m2)][f]
        d = B7.paired_boot(n['C'], n['inv'], n['ng'], a1.astype(float) - a2.astype(float))
        k1, k2 = kappa_vec(n['C'], n['inv'], n['ng'], n['nf'], s1), kappa_vec(n['C'], n['inv'], n['ng'], n['nf'], s2); dk = k1 - k2
        pk = float(min(1.0, 2 * min((dk <= 0).mean(), (dk >= 0).mean()) + 1 / len(dk)))
        RES['tests'].append(dict(label=label, view=f, a=[c1, m1], b=[c2, m2], d_agree=d, kappa=[B7.kappa_of(n['nf'], s1), B7.kappa_of(n['nf'], s2)],
                                 d_kappa=dict(mean=float(B7.kappa_of(n['nf'], s1) - B7.kappa_of(n['nf'], s2)), ci=B7.ci(dk), p=pk), mcnemar=B7.mcnemar(a1, a2)))

    for m in ('prog30', 'prog50', 'mlp32'):
        test('Dmin lofo_D vs nat', 'Dmin', 'lofo_D', m, 'nat', m)
        test('Dmin lofo_D vs rnd', 'Dmin', 'lofo_D', m, 'rnd', m)
        test('Dmin rnd vs nat', 'Dmin', 'rnd', m, 'nat', m)
        for g in ('trans', 'cf', 'rot'): test(f'Dmin only_{g} vs nat', 'Dmin', f'only_{g}', m, 'nat', m)
        test('Dmin all vs lofo_D', 'Dmin', 'all', m, 'lofo_D', m)
        test('trans lofo_trans vs nat', 'trans', 'lofo_trans', m, 'nat', m)
        test('trans all vs nat', 'trans', 'all', m, 'nat', m)
        test('trans all vs shuf', 'trans', 'all', m, 'shuf', m)
        test('cfmin lofo_cf vs nat', 'cfmin', 'lofo_cf', m, 'nat', m)
        test('cfmin all vs nat', 'cfmin', 'all', m, 'nat', m)
    for K in (30, 50):
        for f, c1 in (('Dmin', 'lofo_D'), ('trans', 'lofo_trans')):
            g1, g0 = RES['matched'].get(c1, {}).get(K), RES['matched'].get('nat', {}).get(K)
            if g1 and g0 and (c1, f'prog{K}') in PER:
                n = NET[f]
                d = (PER[(c1, f'prog{K}')][f][0].astype(float) - PER[('nat', f'prog{K}')][f][0]) - (PER[(c1, g1)][f][0].astype(float) - PER[('nat', g0)][f][0])
                RES['tests'].append(dict(label=f'{f} prog{K} gain minus matched gbdt gain ({c1} - nat)', view=f, gbdt=[g0, g1], d_agree=B7.paired_boot(n['C'], n['inv'], n['ng'], d)))
    h = B7.holm([t['d_agree']['p'] for t in RES['tests']]); hk = B7.holm([t['d_kappa']['p'] if 'd_kappa' in t else 1.0 for t in RES['tests']])
    for t, q, qk in zip(RES['tests'], h, hk): t['holm_agree'] = float(q); t['holm_kappa'] = float(qk)
    for t in RES['tests']:
        s = f"{t['label']:45s} {t.get('a', [''])[-1]:7s} agree {t['d_agree']['mean'] * 100:+.2f} {[round(x * 100, 2) for x in t['d_agree']['ci']]} p {t['d_agree']['p']:.4f} Holm {t['holm_agree']:.3g}"
        if 'd_kappa' in t: s += f"; kappa {t['kappa'][1]:.3f} -> {t['kappa'][0]:.3f} d {t['d_kappa']['mean']:+.3f} {[round(x, 3) for x in t['d_kappa']['ci']]} Holm {t['holm_kappa']:.3g}"
        log(s)
    RES['n_tests'] = len(RES['tests'])
    (OUTV / 'b7_verify.json').write_text(json.dumps(RES, indent=1, default=float))
    log('wrote', OUTV / 'b7_verify.json')



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--chunk', type=int, default=250)
    ap.add_argument('--B', type=int, default=5000); ap.add_argument('--nsel', type=int, default=8000)
    a = ap.parse_args()
    if a.smoke: a.B = 200
    log('args', vars(a))
    main(a)
    log('done')
