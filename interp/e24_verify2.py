"""E24 second verification (fresh sample; positive control for the scramble).

Question: is a 7 % call change under the moment-held azimuth scramble A' small compared with what a classifier that does use
prong shape (tau21, D2) would show? E24 compared it with a plan prediction (20-40 %) that was not calibrated.

Sample: interp.jedi_core.sample(per_class=2000, seed=2601), N = 10,000 test jets (E24 used 2401, E24-verify 2501).
Edits (functions from interp.e24_verify): rot, rot2 (two rigid rotations, rot2 = null), A' (theta_i ~ U(0, 2 pi), then the
affine moment hold: M0, vector M1, M2 exact).
Positive / negative controls (feature classifiers, sklearn HistGradientBoosting, fit on pool sample(4000, 2602) minus main jets):
  F0  log m, log pT_jet, log M0, log(M2/M0), multiplicity, girth = sum pT r / M0      (radial / mass only)
  F1  F0 + tau21                                                                        (uses 2-prong shape)
  F2  F0 + tau21 + log D2
Each feature model is evaluated on the unedited, rot and A' jets with features recomputed on the edited particles.
Read-outs: accuracy (all, per class), calls changed, A' - rot with 95 % jet bootstrap and exact McNemar, heavy -> light flow,
for JEDI-linear official, Deep Sets s0, PFN s0 and F0/F1/F2. Paired McNemar JEDI-change vs F-change on the same jets
(H0: equal change rates under A').
  ./job.sh e24-verify2 'python -m interp.e24_verify2'  ->  $AI_RESULTS/jedi/e24_verify2.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
T0 = time.time()
CL = ['g', 'q', 'W', 'Z', 't']


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def feats(x, t21, d2, which):
    import interp.e24_azimuth_prong as E24
    mo = E24.moments(x); xx = np.asarray(x, np.float64); r = np.hypot(xx[..., 1], xx[..., 2])
    cols = [np.log(np.maximum(mo['m'], 1e-6)), np.log(mo['ptj']), np.log(mo['M0']), np.log(mo['M2'] / mo['M0']),
            (xx[..., 0] > 0).sum(1).astype(float), (xx[..., 0] * r).sum(1) / mo['M0']]
    if which in ('F1', 'F2'): cols.append(t21)
    if which == 'F2': cols.append(np.log(np.maximum(d2, 1e-6)))
    return np.stack(cols, 1)


def main():
    import interp.e24_azimuth_prong as E24
    import interp.e24_verify as V
    from interp.jedi_core import sample, Jedi
    from jetdistill.config import RESULTS
    from sklearn.ensemble import HistGradientBoostingClassifier
    ci, mcnemar, holm = V.ci, V.mcnemar, V.holm
    R = dict(node='E24-verify2', sample='interp.jedi_core.sample(per_class=2000, seed=2601)',
             pool='interp.jedi_core.sample(per_class=4000, seed=2602) minus main jets', models={})
    save = lambda: (OUT / 'e24_verify2.json').write_text(json.dumps(V.clean(R), indent=1))
    x, y, Q = sample(2000, 2601); N = len(x); R['N'] = N
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); y_all = np.asarray(d['y'])
    im = E24.sample_idx(y_all, 2000, 2601)
    R['overlap'] = {s: int(np.isin(im, E24.sample_idx(y_all, 2000, s)).sum()) for s in (2401, 2501)}
    Ed = {'rot': V.rotate(x, 2610), 'rot2': V.rotate(x, 2611)}
    Ed["A'"] = V.moment_hold(V.scramble(x, 2612), x)
    Ed = {k: v.astype(np.float32) for k, v in Ed.items()}
    R['edit_checks'] = V.checks(x, Ed)
    m0 = E24.moments(x)['m']
    R['edit_checks_mass'] = {k: dict(med_rel_dm=float(np.median(np.abs(E24.moments(e)['m'] / np.maximum(m0, 1e-9) - 1))),
                                     p90_rel_dm=float(np.quantile(np.abs(E24.moments(e)['m'] / np.maximum(m0, 1e-9) - 1), .9)))
                             for k, e in Ed.items()}
    log('edit checks', json.dumps(V.clean(R['edit_checks'])), json.dumps(R['edit_checks_mass']))
    t0, d0 = E24.substructure(x); R['tau21_check_vs_Q'] = float(np.max(np.abs(t0 - Q['tau21']))) if 'tau21' in Q else None
    SUB = {'none': (t0, d0)}
    for k, e in Ed.items(): SUB[k] = E24.substructure(e)
    R['substructure'] = {k: {CL[c]: dict(tau21_median=float(np.median(t[y == c])), D2_median=float(np.median(dd[y == c])))
                             for c in range(5)} for k, (t, dd) in SUB.items()}
    log('substructure', json.dumps({k: {c: round(v['tau21_median'], 3) for c, v in s.items()} for k, s in R['substructure'].items()}))
    save()
    # feature classifiers (pool excludes the main jets and both earlier E24 main samples)
    xp, yp, _ = sample(4000, 2602); ip = E24.sample_idx(y_all, 4000, 2602)
    excl = np.concatenate([im, E24.sample_idx(y_all, 2000, 2401), E24.sample_idx(y_all, 2000, 2501)])
    keep = ~np.isin(ip, excl); xp, yp = xp[keep], yp[keep]; R['pool_n'] = int(len(xp))
    tp, dp = E24.substructure(xp); log('pool', len(xp))
    calls = {}
    for F in ('F0', 'F1', 'F2'):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0)
        clf.fit(feats(xp, tp, dp, F), yp)
        calls[F] = {k: clf.predict(feats(Ed[k] if k != 'none' else x, *SUB[k], F)) for k in ('none', 'rot', 'rot2', "A'")}
        log(F, 'acc', float((calls[F]['none'] == y).mean()))
    # networks
    for name in ('jedi', 'dsets_s0', 'pfn_s0'):
        run, acc_ref, kind, net = E24.get_model(name)
        calls[name] = {'none': run(x)['logits'].argmax(1)}
        for k in ('rot', 'rot2', "A'"): calls[name][k] = run(Ed[k])['logits'].argmax(1)
        log(name, 'acc', float((calls[name]['none'] == y).mean()))
    ps = {}
    for name, C in calls.items():
        c0 = C['none']; heavy = np.isin(c0, [2, 3, 4]); ch = {k: C[k] != c0 for k in ('rot', 'rot2', "A'")}
        M = dict(acc={k: ci(C[k] == y) for k in C}, acc_per_class={k: {CL[c]: float((C[k][y == c] == c).mean()) for c in range(5)} for k in C},
                 calls_changed={k: ci(v) for k, v in ch.items()},
                 heavy_to_light={k: ci(np.isin(C[k][heavy], [0, 1])) for k in ('rot', "A'")},
                 Aprime_minus_rot=ci(ch["A'"].astype(float) - ch['rot']), mcnemar_Aprime_rot=mcnemar(ch["A'"], ch['rot']),
                 rot2_minus_rot=ci(ch['rot2'].astype(float) - ch['rot']), mcnemar_rot2_rot=mcnemar(ch['rot2'], ch['rot']),
                 acc_drop_Aprime_minus_rot=ci((C['rot'] == y).astype(float) - (C["A'"] == y)))
        R['models'][name] = M; ps[name] = M['mcnemar_Aprime_rot']['p']
        log(name, 'acc', round(M['acc']['none']['mean'], 4), 'rot', round(M['calls_changed']['rot']['mean'], 4),
            "A'", round(M['calls_changed']["A'"]['mean'], 4), 'excess', M['Aprime_minus_rot'], 'acc drop', M['acc_drop_Aprime_minus_rot'])
    h = holm(list(ps.values()))
    for (name, _), hv in zip(ps.items(), h): R['models'][name]['mcnemar_Aprime_rot']['p_holm'] = float(hv)
    # JEDI vs feature models: paired change indicators under A' (and the excess over rot)
    cj = calls['jedi']; chj = cj["A'"] != cj['none']; exj = chj.astype(float) - (cj['rot'] != cj['none'])
    R['jedi_vs_features'] = {}
    for F in ('F0', 'F1', 'F2'):
        cf = calls[F]; chf = cf["A'"] != cf['none']; exf = chf.astype(float) - (cf['rot'] != cf['none'])
        R['jedi_vs_features'][F] = dict(change_diff=ci(chf.astype(float) - chj), mcnemar=mcnemar(chf, chj),
                                        excess_diff=ci(exf - exj))
    log('jedi vs features', json.dumps(V.clean(R['jedi_vs_features'])))
    save(); log('wrote', OUT / 'e24_verify2.json')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True); main()
