"""B4 verification (skeptic, 2026-10-17): rerun the key B4 comparisons on a FRESH test sample.

Fresh test: sample(per_class=2000, seed=23) minus every jet of seeds 0 (fit/dev), 11 (B4 test) and 91 (B4 confirmation).
Fit split identical to B4 / E11: sample(6000, seed 0), permutation rng 0, first 2/3 (20,000 jets).  Nothing is selected on
the fresh jets: every baseline is the one B4 named (dev-selected or B4's test-selected "oracle"), refitted with the same
settings and random_state (deterministic), or read from the stored formulas (nested MARS, zphys30, direct MARS terms).
Check 0: the refitted / reloaded models reproduce B4's same-call on the B4 test split (seed 11) to the stored value.
Edits on the fresh jets (physics recomputed from the edited particles, as B4):
  trans     Δη += 0.06 for every real particle (B4's edit)
  transphi  Δφ += 0.06 for every real particle (second coherent-shift direction; replicate)
  trans_c   every real particle moved 0.06 rad in an independent random direction (B4's control, rng 101)
Tests (paired over the same jets): agreement difference nested - baseline per edit (bootstrap 95 % CI over jets, McNemar);
difference of leads (trans lead - trans_c lead, and transphi lead - trans_c lead) with a paired bootstrap CI and two-sided
bootstrap p (H0: lead equal on the coherent shift and on the random-direction control).  BH over all McNemar cells.
Writes AI_RESULTS/jedi/b4_verify.json.        python -m interp.b4_verify [seed]
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp import e11_mars_all as E
from interp.jedi_core import Jedi, sample
from interp.b4_baselines_jedi import edit_shift, randdisp, mcnemar, bh

OUT = E.OUT; T0 = time.time(); B = 1000


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def shift_phi(xx, d):
    e = xx.copy(); rl = e[..., 0] > 0; e[..., 2] += np.float32(d) * rl; return e


def main(seed=23):
    from jetdistill.observables import compute
    from jetdistill import mars
    from interp.e11_verify import Last, chain
    keys = json.loads((OUT / 'e11_fit_a_z.json').read_text())['keys']; F = len(keys)
    i0, i11, i91, iF = E.indices(6000, 0), E.indices(2000, 11), E.indices(2000, 91), E.indices(2000, seed)
    keepF = ~np.isin(iF, np.concatenate([i0, i11, i91])); keep11 = ~np.isin(i11, i0)
    assert not np.isin(iF[keepF], np.concatenate([i0, i11, i91])).any()
    net = Jedi(64); last = Last(net)
    # ---- fit split (as E11 Data)
    x0, y0, Q0 = sample(6000, 0); perm = np.random.default_rng(0).permutation(len(y0)); fit = perm[:2 * len(y0) // 3]
    xf = x0[fit]; hf = E.handbuilt(xf, False)
    physf = {k: (np.asarray(Q0[k][fit], np.float64) if k in Q0 else hf[k]) for k in keys}; del Q0, x0, hf
    Xf = np.stack([physf[k].astype(np.float32) for k in keys], 1); Lf = net.run(xf)['logits']; cf = Lf.argmax(1); NF = len(cf)
    log(f'fit jets {NF}; network acc on fit {(cf == y0[fit]).mean():.4f} (E11: 0.8058 on fit+dev)'); del xf
    # ---- evaluation sets
    def physd(x, Q=None):
        if Q is None:
            q = compute(np.asarray(x, np.float64), 64); Q = {k: np.asarray(v, np.float32).astype(np.float64) for k, v in q.items()}
        h = E.handbuilt(x, False); return {k: (np.asarray(Q[k], np.float64) if k in Q else h[k]) for k in keys}
    x11, y11, Q11 = sample(2000, 11); x11, y11 = x11[keep11], y11[keep11]; Q11 = {k: v[keep11] for k, v in Q11.items()}
    S = {'b4test': physd(x11, Q11)}; Y = {'b4test': y11}; XS = {'b4test': x11}; del Q11
    xF, yF, QF = sample(2000, seed); xF, yF = xF[keepF], yF[keepF]; QF = {k: v[keepF] for k, v in QF.items()}
    S['test'] = physd(xF, QF); del QF
    real = xF[..., 0] > 0
    XS.update(test=xF, trans=edit_shift(xF, 0.06), transphi=shift_phi(xF, 0.06), trans_c=randdisp(xF, 0.06 * real, 101))
    for s in ('test_rc', 'trans', 'transphi', 'trans_c'):
        t = time.time(); S[s] = physd(XS['test' if s == 'test_rc' else s]); log(f'{s}: physics recomputed ({time.time() - t:.0f} s)')
    XS['test_rc'] = xF
    for s in S: Y[s] = yF if s != 'b4test' else y11
    rel = max(float(np.max(np.abs(S['test_rc'][k] - S['test'][k])) / max(float(np.max(np.abs(S['test'][k]))), 1e-12)) for k in keys)
    nonfin = {s: int(sum((~np.isfinite(v)).sum() for v in S[s].values())) for s in S}
    log(f'fresh: {len(yF)} jets ({int((~keepF).sum())} shared with seeds 0/11/91 removed); recompute vs stored max rel {rel:.2e}; non-finite {nonfin}')
    NL = {s: net.run(XS[s])['logits'] for s in S}; NC = {s: v.argmax(1) for s, v in NL.items()}
    Xm = {s: np.nan_to_num(np.stack([S[s][k].astype(np.float32) for k in keys], 1)) for s in S}
    # ---- models: calls per set
    C = {}
    for K in (2, 4, 8, 16, 30):
        pre = 'e11' if K == 30 else f'b4m{K}'; rd = lambda n: json.loads((OUT / f'{pre}_{n}.json').read_text())
        fb = {'P': rd('fit_a_P'), **{s: rd(f'fit_b_{s}') for s in E.STG[1:]}}
        C[f'mars{K}'] = {s: last.jax_logits(chain(fb, S[s])['z']).argmax(1) for s in S}
    fa = json.loads((OUT / 'e11_fit_a_z.json').read_text())
    C['zphys30'] = {s: last.jax_logits(E.evaluate(fa['units'], S[s], False)).argmax(1) for s in S}
    fits = json.loads((OUT / 'b4_direct_terms.json').read_text())
    for kb in (8, 16, 32, 100):
        parts = []
        for f in fits:
            tr = f['terms'][:kb]; co = mars._lstsq(tr, physf, Lf[:, f['cls']] + f['shift'], NF); parts.append((tr, co, f['shift']))
        C[f'direct{kb}'] = {s: np.stack([np.stack([np.ones(len(Y[s]))] + [mars.term_value(t, S[s]) for t in tr], 1) @ co - sh for tr, co, sh in parts], 1).argmax(1) for s in S}
    log('MARS models evaluated')
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    for d in (4, 6, 8):
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xf, cf); C[f'tree_d{d}'] = {s: m.predict(Xm[s]) for s in S}
    for L_, T_ in ((4, 4), (4, 8), (4, 16), (8, 32), (16, 16), (16, 32), (4, 128), (32, 128)):
        t = time.time(); m = HistGradientBoostingClassifier(max_leaf_nodes=L_, max_iter=T_, learning_rate=0.2, early_stopping=False, random_state=0).fit(Xf, cf)
        C[f'gbdt_L{L_}_T{T_}'] = {s: m.predict(Xm[s]) for s in S}; log(f'gbdt_L{L_}_T{T_} ({time.time() - t:.0f} s)')
    for L_, T_ in ((32, 8), (32, 16)):
        t = time.time()
        rs = [HistGradientBoostingRegressor(max_leaf_nodes=L_, max_iter=T_, learning_rate=0.2, early_stopping=False, random_state=0).fit(Xf, Lf[:, k]) for k in range(5)]
        C[f'gbdtreg_L{L_}_T{T_}'] = {s: np.stack([r.predict(Xm[s]) for r in rs], 1).argmax(1) for s in S}; log(f'gbdtreg_L{L_}_T{T_} ({time.time() - t:.0f} s)')
    # ---- metrics
    stored = {m['name']: m for m in json.loads((OUT / 'b4.json').read_text())['models']}
    J = len(yF); rng = np.random.default_rng(7); IX = rng.integers(0, J, (B, J))
    def ci(v): v = np.asarray(v, float); b = v[IX].mean(1) if len(v) == J else None; return [float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    def ci_sub(v, mask):                          # mean over a subset of jets, bootstrap over all jets
        v = np.asarray(v, float); m = mask.astype(float); b = (v * m)[IX].sum(1) / np.maximum(m[IX].sum(1), 1)
        return [float(v[mask].mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    EDITS = ('trans', 'transphi', 'trans_c'); nb = NC['test_rc']
    net_res = dict(acc_fresh=ci(NC['test'] == yF), acc_b4test=float((NC['b4test'] == y11).mean()), N_fresh=J,
                   flip={e: ci(NC[e] != nb) for e in EDITS},
                   W_to_Z={e: float((NC[e][yF == 2] == 3).mean()) for e in EDITS}, W_to_Z_before=float((nb[yF == 2] == 3).mean()))
    log(f"network: acc fresh {net_res['acc_fresh']}; flips {[(e, round(v[0], 4)) for e, v in net_res['flip'].items()]}; W->Z {net_res['W_to_Z']}")
    AG, rows = {}, {}
    for name, c in C.items():
        sc = c['test'] == NC['test']; r = dict(same_call=ci(sc), same_call_b4test=float((c['b4test'] == NC['b4test']).mean()),
                                              same_call_b4test_stored=stored.get(name, {}).get('same_call_test'), edits={})
        sb = c['test_rc']; AG[name] = {}
        for e in EDITS:
            nf, sf = NC[e] != nb, c[e] != sb; ag = (nf == sf) & (~nf | (c[e] == NC[e])); AG[name][e] = ag
            hit = sf & (c[e] == NC[e])
            r['edits'][e] = dict(agree=ci(ag), recall=ci_sub(hit, nf), flip_rate=float(sf.mean()), W_to_Z=float((c[e][yF == 2] == 3).mean()))
        rows[name] = r
        log(f"{name:18s} same call fresh {sc.mean() * 100:6.2f} | B4 test {r['same_call_b4test'] * 100:6.2f} (stored {r['same_call_b4test_stored'] if r['same_call_b4test_stored'] is None else round(r['same_call_b4test_stored'][0] * 100, 2)}) | " +
            ' '.join(f"{e} agree {r['edits'][e]['agree'][0] * 100:.2f} rec {r['edits'][e]['recall'][0] * 100:.1f} W>Z {r['edits'][e]['W_to_Z'] * 100:.1f}" for e in EDITS))
    # ---- paired comparisons (baselines pre-specified by B4)
    PAIRS = [('mars16', 'gbdt_L16_T16', 'B4 dev-selected (also B4 trans_c oracle)'), ('mars16', 'gbdtreg_L32_T8', 'B4 trans oracle'),
             ('mars30', 'gbdt_L4_T128', 'B4 dev-selected'), ('mars30', 'gbdtreg_L32_T16', 'B4 trans oracle'), ('mars30', 'gbdt_L16_T32', 'B4 trans_c oracle')]
    comp, pv, cell = [], [], []
    for a, b, why in PAIRS:
        e_ = dict(nested=a, baseline=b, why=why, edits={}, lead_diff={})
        sa, sb_ = (C[a]['test'] == NC['test']).astype(float), (C[b]['test'] == NC['test']).astype(float)
        e_['same_call_diff'] = ci(sa - sb_)
        d = {e: AG[a][e].astype(float) - AG[b][e] for e in EDITS}
        for e in EDITS:
            mc = mcnemar(AG[a][e], AG[b][e]); e_['edits'][e] = dict(diff=ci(d[e]), mcnemar=mc); pv.append(mc['p']); cell.append((len(comp), e))
        for e in ('trans', 'transphi'):
            v = d[e] - d['trans_c']; bs = v[IX].mean(1); p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
            e_['lead_diff'][f'{e} - trans_c'] = dict(diff=ci(v), p_boot_two_sided=p)
        comp.append(e_)
        log(f"{a} vs {b} ({why}): same call {np.round(np.array(e_['same_call_diff']) * 100, 2)} | " +
            ' '.join(f"{e} {np.round(np.array(v['diff']) * 100, 2)} p {v['mcnemar']['p']:.2g}" for e, v in e_['edits'].items()) +
            ' | lead diffs ' + ' '.join(f"{k} {np.round(np.array(v['diff']) * 100, 2)} p {v['p_boot_two_sided']:.3g}" for k, v in e_['lead_diff'].items()))
    q = bh(pv)
    for (i, e), qq in zip(cell, q): comp[i]['edits'][e]['mcnemar']['q_bh'] = float(qq)
    # ---- reach checks (B4's "lowest DL reaching nested same call", re-measured on fresh jets)
    REACH = [('mars2', 'tree_d4'), ('mars2', 'direct8'), ('mars2', 'gbdt_L4_T4'), ('mars4', 'direct8'), ('mars4', 'tree_d6'), ('mars4', 'gbdt_L4_T8'),
             ('mars8', 'direct16'), ('mars8', 'gbdt_L4_T16'), ('mars16', 'gbdt_L8_T32'), ('mars16', 'direct100'), ('mars30', 'gbdt_L32_T128')]
    reach = []
    for a, b in REACH:
        v = (C[b]['test'] == NC['test']).astype(float) - (C[a]['test'] == NC['test']); reach.append(dict(nested=a, baseline=b, dl_nested=stored[a]['dl'], dl_baseline=stored[b]['dl'],
                                                                                                         diff_baseline_minus_nested=ci(v)))
        log(f"reach: {b} (DL {stored[b]['dl']}) - {a} (DL {stored[a]['dl']}) same call {np.round(np.array(reach[-1]['diff_baseline_minus_nested']) * 100, 2)}")
    res = dict(procedure=__doc__, seed=seed, n_fresh=J, n_removed=int((~keepF).sum()), recompute_max_rel=rel, nonfinite=nonfin, network=net_res,
               models=rows, comparisons=comp, n_bh=len(pv), reach=reach, bootstrap=B,
               dl={n: stored[n]['dl'] for n in C if n in stored})
    p = OUT / 'b4_verify.json'; p.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float)); log(f'wrote {p}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 23)
