"""E25w verify: fresh-sample rerun of the E25w DAS (16f n64) with two added controls, the same DAS on the 3-feature n64 network
(matched comparison for "no late formation step as in the 3-feature net"), and a dose arm with ΔR-only dilations matched on
other radial moments (does the translation response single out M2?).
   python -m interp.e25w_verify [--smoke]   -> $AI_RESULTS/jedi/e25w_verify[_smoke].json

Sample: rows of interp.jedi_core.sample(1500, 2541) minus the rows of sample(3000, 2501), sample(1500, 2502), sample(1500, 2507)
(E25 main / fresh / E25v) and sample(1500, 2521) (E25w). 16-feature columns from val_c16_f16.npy, checked against sample() x.
Fit / test halves: rng(2542).permutation. Pairs: 3,000 train (fit half, rng 2543), 2,000 test (test half, rng 2544).
Part form (k = 2; S, h4, h5, z; M0, M2, D; edits and DAS exactly as interp.e25w_jedi16_mass):
  models 16f n64 and 3f n64 (3f input = the edited 16-column jets' pt / etarel / phirel columns).
  Controls: 500 random 2-subspaces per cell (Holm over 12 cells), full-stage swap (source call), and "call-DAS": a 2-dim DAS
  trained on the source call as label (the best 2-dim subspace for moving the call itself), scored against each variable's
  counterfactuals; paired bootstrap DAS_V - callDAS. On-manifold subset: M2 pairs with angle scale in [1/2, 2].
Part dose (16f n64, all jets): T_d (Δη += d), dilR_M2 (ΔR column x s, s = sqrt(M2_T / M2), as E25w), dilR_M1 (ΔR x s1,
  s1 = Σ pT r_T / Σ pT r), dilR_Mr (ΔR x sr, sr = Σ r_T / Σ r, unweighted), R_d (rotation, as E25w). Metrics on true W: Z fraction,
  W->Z flow with bootstrap CI, ratio flow(T)/flow(arm) with CI, per-jet call agreement with T on all jets, McNemar (Z on true W) T vs arm.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true')
args = ap.parse_args()
SM = args.smoke
import interp.e25w_jedi16_mass as E  # noqa: E402  (its own argparse runs in smoke mode on import; only functions are used)
from interp.e19_common import wilson, mcnemar, holm, boot_mean, boot_ratio  # noqa: E402

T0 = time.time()
def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
PER = 150 if SM else 1500
NTR, NTE = (400, 300) if SM else (3000, 2000)
NRAND = 20 if SM else 500
STEPS = 60 if SM else 600
NB = 200 if SM else 1000
VARS = E.VARS; STAGES = ('S', 'h4', 'h5', 'z')
C = E.C; W_, Z_ = 2, 3
JS = OUT / f'e25w_verify{"_smoke" if SM else ""}.json'


def load():
    from jetdistill.config import RESULTS, DATA
    from interp.jedi_core import sample
    y = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y'])
    idx = E.rows(y, PER, 2541); x3, ys, _ = sample(PER, 2541); assert np.array_equal(ys, y[idx])
    X = np.asarray(np.load(Path(DATA) / 'val_c16_f16.npy', mmap_mode='r')[idx, :64]).astype(np.float32)
    d = float(np.abs(X[..., [5, 8, 11]] - x3).max()); log(f'max |16f cols - sample x| {d:.3g}')
    if d > 1e-3: raise SystemExit('ABORT: 16-feature columns differ from sample() x')
    excl = np.concatenate([E.rows(y, 3000, 2501), E.rows(y, 1500, 2502), E.rows(y, 1500, 2507), E.rows(y, 1500, 2521)])
    keep = ~np.isin(idx, excl)
    return X[keep], ys[keep], idx[keep], d


def clean(o): return E.clean(o)


def main():
    from interp.jedi_core import Jedi
    X, y, idx, dmax = load(); N = len(X)
    p = np.random.default_rng(2542).permutation(N); fit, te = np.sort(p[:N // 2]), np.sort(p[N // 2:]); I = E.jinfo(X)
    rl = I['rl']; dr_chk = float(np.abs(np.where(rl, X[..., C['deltaR']] - np.hypot(X[..., C['etarel']], X[..., C['phirel']]), 0)).max())
    R = dict(sample=dict(per_class=PER, seed=2541, N=N, N_fit=len(fit), N_test=len(te), class_counts=np.bincount(y, minlength=5),
                         excluded='sample(3000,2501), sample(1500,2502), sample(1500,2507), sample(1500,2521)', max_col_diff=dmax,
                         max_deltaR_minus_hypot=dr_chk, split='rng(2542).permutation halves'), procedure=__doc__)
    log(f'N {N}; classes {np.bincount(y, minlength=5).tolist()}; max |deltaR - hypot(etarel, phirel)| {dr_chk:.3g}')
    def save(): JS.write_text(json.dumps(clean(R), indent=1))  # noqa: E306
    pairs = dict(train=E.make_pairs(fit, NTR, 2543, I), test=E.make_pairs(te, NTE, 2544, I))
    PE, meta = {}, {}
    for arm, (b, s) in pairs.items(): PE[arm], meta[arm] = E.pair_edits(X, I, b, s)
    R['pair_meta'] = meta
    bt, st_ = pairs['test']; a2 = np.sqrt(I['M2'][st_] / I['M2'][bt]); onm = (a2 >= 0.5) & (a2 <= 2.0)
    R['onmanifold_M2_frac'] = float(onm.mean())
    R['models'] = {}
    for name, fam in (('16f', '16-feature'), ('3f', '3-feature')):
        j = Jedi(64, fam); net = E.Net(j, name=name)
        sel = (lambda A: A) if name == '16f' else (lambda A: A[..., [5, 8, 11]])
        Xin = sel(X); PEin = {arm: {V: sel(v) for V, v in d.items()} for arm, d in PE.items()}
        reps = net.run(Xin, ('S', 'h4', 'h5', 'z', 'logits')); c0 = reps['logits'].argmax(1)
        ref = j.run(Xin[:1000])['logits']
        M = dict(acc=float((c0 == y).mean()), acc_ci=wilson(int((c0 == y).sum()), N), own_vs_jedi_run=float(np.abs(reps['logits'][:1000] - ref).max()))
        log(f'{name}: acc {M["acc"]:.4f} {M["acc_ci"]}; own forward vs Jedi.run {M["own_vs_jedi_run"]:.2e}')
        lab = E.labels(net, PEin, None, None, reps, pairs)
        M['cf_change_frac_test'] = {V: float((lab['test'][V] != lab['test']['base']).mean()) for V in VARS}
        log(f'{name}: cf change frac {M["cf_change_frac_test"]}')
        M['das'] = E.das_block(net, reps, pairs, lab, (2,), NRAND, STEPS, STAGES, nb=NB); save()
        # call-DAS control and on-manifold subset
        bte = lab['test']['base']; M['callDAS'] = {}; M['onmanifold_M2'] = {}
        for stg in STAGES:
            hb_tr, hs_tr = reps[stg][pairs['train'][0]], reps[stg][pairs['train'][1]]
            hb_te, hs_te = reps[stg][pairs['test'][0]], reps[stg][pairs['test'][1]]
            Q, info = E.das_fit(net, stg, hb_tr, hs_tr, lab['train']['source'], lab['train']['base'], 2, seed=12, steps=STEPS)
            cc = E.eval_Q(net, stg, hb_te, hs_te, Q); d = dict(train=info, eff_source=E.eff(cc, lab['test']['source'], bte))
            for V in VARS:
                QV, _ = E.das_fit(net, stg, hb_tr, hs_tr, lab['train'][V], lab['train']['base'], 2, seed=12, steps=STEPS)
                cV = E.eval_Q(net, stg, hb_te, hs_te, QV)
                d[V] = dict(callDAS_eff=E.eff(cc, lab['test'][V], bte), callDAS_ci=E.boot_eff(cc, lab['test'][V], bte, NB),
                            das_eff_refit=E.eff(cV, lab['test'][V], bte), das_minus_callDAS=E.boot_diff(cV, cc, lab['test'][V], bte, NB))
                if V == 'M2':
                    M['onmanifold_M2'][stg] = dict(n_eff=int((lab['test']['M2'][onm] != bte[onm]).sum()),
                                                   das=E.eff(cV[onm], lab['test']['M2'][onm], bte[onm]),
                                                   das_ci=E.boot_eff(cV[onm], lab['test']['M2'][onm], bte[onm], NB),
                                                   full_swap=E.eff(lab['test']['source'][onm], lab['test']['M2'][onm], bte[onm]))
            M['callDAS'][stg] = d
            log(f'{name} {stg} callDAS: ' + '; '.join(f'{V} callDAS {d[V]["callDAS_eff"]} DAS {d[V]["das_eff_refit"]} diff {d[V]["das_minus_callDAS"]}' for V in VARS)
                + f'; on-manifold M2 {M["onmanifold_M2"][stg]}')
            save()
        R['models'][name] = M; save(); del reps
    # ---------------------------------------------------------------- dose with alternative matched dilations (16f n64)
    j = Jedi(64, '16-feature'); net = E.Net(j, name='16f'); c0 = net.calls(X); Wm = y == W_; rng = np.random.default_rng(2546)
    Xd = X.astype(np.float64); pt = np.where(rl, Xd[..., C['pt']], 0); er, pr = Xd[..., C['etarel']], Xd[..., C['phirel']]
    r0 = np.where(rl, np.hypot(er, pr), 0); D = R['dose'] = {}; tests = []
    for d in ((0.03, 0.045, 0.06) if SM else (0.015, 0.03, 0.045, 0.06, 0.09)):
        rT = np.where(rl, np.hypot(er + d, pr), 0)
        sM2 = np.sqrt((pt * rT ** 2).sum(1) / np.maximum((pt * r0 ** 2).sum(1), 1e-30))
        sM1 = (pt * rT).sum(1) / np.maximum((pt * r0).sum(1), 1e-30); sMr = rT.sum(1) / np.maximum(r0.sum(1), 1e-30)
        ang = 2 * np.arcsin(np.minimum(1.0, d / np.maximum(2 * I['rbar'], 1e-12))) * np.random.default_rng(int(d * 1e4)).choice([-1, 1], N)
        arms = dict(T=E.edit16(X, I, der=d), dilR_M2=E.edit16(X, I, dR=sM2), dilR_M1=E.edit16(X, I, dR=sM1), dilR_Mr=E.edit16(X, I, dR=sMr),
                    R=E.edit16(X, I, rot=ang))
        calls = {k: net.calls(v) for k, v in arms.items()}; res = dict(s_median=dict(M2=float(np.median(sM2)), M1=float(np.median(sM1)), Mr=float(np.median(sMr))))
        fl = {}
        for k, ce in calls.items():
            fl[k] = (ce[Wm] == Z_).astype(float) - (c0[Wm] == Z_)
            res[k] = dict(W_Z=float((ce[Wm] == Z_).mean()), W_t=float((ce[Wm] == 4).mean()), W_qg=float((ce[Wm] <= 1).mean()),
                          flow=float(fl[k].mean()), flow_ci=boot_mean(fl[k], rng), calls_changed=float((ce != c0).mean()),
                          agree_with_T_all=float((ce == calls['T']).mean()), agree_with_T_W=float((ce[Wm] == calls['T'][Wm]).mean()))
        for k in ('dilR_M2', 'dilR_M1', 'dilR_Mr'):
            if fl[k].mean() > 0: res[k]['ratio_T_over'] = float(fl['T'].mean() / fl[k].mean()); res[k]['ratio_ci'] = boot_ratio(fl['T'], fl[k], rng)
            res[k]['mcnemar_vs_T'] = mcnemar(calls['T'][Wm] == Z_, calls[k][Wm] == Z_); tests.append((str(d), k, res[k]['mcnemar_vs_T']['p']))
        D[str(d)] = res
        log(f'dose {d}: s med {res["s_median"]}; ' + '; '.join(f'{k} Z {v["W_Z"]:.3f} t {v["W_t"]:.3f} qg {v["W_qg"]:.3f} flow {v["flow"]:+.3f} agreeT {v["agree_with_T_all"]:.3f} ratio {v.get("ratio_T_over")}'
                                                          for k, v in res.items() if k != 's_median'))
        save()
    ph = holm([t[2] for t in tests])
    for (d, k, _), h in zip(tests, ph): D[d][k]['holm_p_vs_T'] = h
    R['acc_16f_dose'] = float((c0 == y).mean())
    import resource; R['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024; R['wall_s'] = time.time() - T0
    save(); log('done')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True); main()
