"""H35 verify, part 3 (loop D): does the reverse-move result depend on the quality of the kT-3 mass assignment?

Sample: the same top jets as h35_verify (JetClass test TTBar_101, 550 < jet_pt < 650 GeV, |jet_eta| < 2, >= 3 particles,
  2,000 random rows, rng 3551). Roles and edits: h35_top_bprong (M, C1) and h35_verify.new_edits (Mk, Mpt, RV, RVpt, CW),
  unchanged.
Mass-assignment quality per jet: good = |m_Wpair - 80.4| < 10 GeV and 150 < m_jet < 200 GeV (m_Wpair = mass of the subjet pair
  closest to 80.4 GeV; m_jet = invariant jet mass from h4_top_wmass._obs); bad = the rest.
Tests (ParT_full, PN_full; ParT_perm as null), per-jet change in top log-odds, 95 % bootstrap CI (B = 2000), sign-flip p (5,000),
  Holm over all cells:
  RVpt - CW on disagree jets (the W prong holds at least as many displaced tracks as c), good vs bad mass assignment
  Mpt - C1 on agree jets, good vs bad
  Mpt - C1 vs residual pT mismatch: per-jet least-squares line of (Mpt - C1) on mean |d ln pT| of the receiving tracks, pooling
    Mk - C1 and Mpt - C1 rows (two mismatch levels per jet); intercept = contrast at zero pT mismatch, bootstrap over jets.
Predictions: if the reverse-move drop in disagree jets comes from wrong mass assignments (true b on the W-pair prong), RVpt - CW
  shrinks toward 0 or turns positive in good-mass jets; if the networks follow the subjet holding most displaced tracks, RVpt - CW
  stays negative in good-mass jets.
Run: ./part_job.sh d-h35-verify3 'python -m interp.h35_verify3'   Output: $AI_RESULTS/physics/h35_verify3.json
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h4_top_wmass as H          # noqa: E402
from interp import h19_wpair_cross as X       # noqa: E402
from interp import h35_top_bprong as T        # noqa: E402
from interp import h35_verify as V            # noqa: E402

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
B, NPERM = 2000, 5000
NETS = ('ParT_full', 'PN_full', 'ParT_perm')


def main():
    RES.mkdir(parents=True, exist_ok=True); out = dict(node='H35_verify3', B=B, n_perm=NPERM)
    J, it = X.jc_read('Tbqq', 101, 2000, 3551)[::2]; y = (J['label'] == T.TBQQ).astype(int); assert y.all()
    out['sample'] = dict(Tbqq=it, n=int(len(y)))
    P4, nn = X.jc_P4(J); LAB = H.kt3_batch(np.ascontiguousarray(P4.astype(np.float64)), nn); R = T.roles(P4, nn, LAB)
    O = H._obs(P4, nn); nd0, _ = T.counts(J, LAB); r = np.arange(len(y))
    ndc = nd0[r, R['c']]; agree = ndc > np.maximum(nd0[r, R['a']], nd0[r, R['b']])
    good = (np.abs(R['mW'] - H.MWREF) < 10) & (O['m'] > 150) & (O['m'] < 200)
    out['roles'] = dict(agree_frac=float(agree.mean()), good_frac=float(good.mean()), good_frac_agree=float(good[agree].mean()),
                        good_frac_disagree=float(good[~agree].mean()), m_median=float(np.median(O['m'])))
    log('roles', out['roles'])
    E0, info0 = T.make_edits(J, LAB, R, y); E1, info1 = V.new_edits(J, LAB, R, y)
    E = {v: E0[v] for v in ('M', 'C1')}; E.update(E1)
    for v, Ev in E.items():
        Jv = dict(J); Jv.update(Ev); ndv, _ = T.counts(Jv, LAB)
        assert np.array_equal(ndv.sum(1), nd0.sum(1)), v
    nets = T.Nets(); Jid = T.variant_dict(J, {k: J[k] for k in T.IPK}); Jvar = {v: T.variant_dict(J, Ev) for v, Ev in E.items()}
    LT = {}; out['unedited'] = {}
    for name in NETS:
        Lid = nets.logits(name, Jid); LT[(name, 'id')] = T.top_lo(Lid)
        out['unedited'][name] = dict(top_call_rate=float((Lid.argmax(1) == T.TBQQ).mean()))
        for v in E: LT[(name, v)] = T.top_lo(nets.logits(name, Jvar[v]))
        log(name, out['unedited'][name])
    d = lambda m, v: LT[(m, v)] - LT[(m, 'id')]
    k1, kr = info1['k1'], info1['kr']
    sets = {'disagree_kr_good': ~agree & (kr > 0) & good, 'disagree_kr_bad': ~agree & (kr > 0) & ~good,
            'agree_k1_good': agree & (k1 > 0) & good, 'agree_k1_bad': agree & (k1 > 0) & ~good,
            'agree_kr_good': agree & (kr > 0) & good}
    out['sets'] = {k: int(v.sum()) for k, v in sets.items()}; log('sets', out['sets'])
    CON = {'RVpt-CW': ('disagree_kr_good', 'disagree_kr_bad', 'agree_kr_good'), 'Mpt-C1': ('agree_k1_good', 'agree_k1_bad')}
    rng = np.random.default_rng(3598); res, cells = {}, []
    for m in NETS:
        for cn, sns in CON.items():
            va, vb = cn.split('-')
            for sn in sns:
                x = (d(m, va) - d(m, vb))[sets[sn]]; mu, ci = T.boot(x, rng, B); p = T.perm2(x, rng, NPERM)
                res[f'{sn}|{m}|{cn}'] = dict(mean=mu, ci=ci, p_perm=p, n=int(len(x))); cells.append(f'{sn}|{m}|{cn}')
        # zero-mismatch extrapolation on agree jets with k1 > 0
        sm = agree & (k1 > 0) & np.isfinite(info1['dlogpt_Mk']) & np.isfinite(info1['dlogpt_Mpt'])
        ya = np.concatenate([(d(m, 'Mk') - d(m, 'C1'))[sm], (d(m, 'Mpt') - d(m, 'C1'))[sm]])
        xa = np.concatenate([info1['dlogpt_Mk'][sm], info1['dlogpt_Mpt'][sm]]); nj = int(sm.sum())

        def fit(idx):
            ii = np.concatenate([idx, idx + nj]); A = np.column_stack([np.ones(len(ii)), xa[ii]])
            return np.linalg.lstsq(A, ya[ii], rcond=None)[0]
        b0 = fit(np.arange(nj)); bs = np.array([fit(rng.integers(0, nj, nj)) for _ in range(B)])
        p0 = float(min(1.0, 2 * min((bs[:, 0] >= 0).mean(), (bs[:, 0] <= 0).mean()) + 1 / B))
        res[f'agree_k1|{m}|intercept_zero_dlogpt'] = dict(intercept=float(b0[0]), slope=float(b0[1]),
                                                           ci_intercept=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
                                                           ci_slope=np.percentile(bs[:, 1], [2.5, 97.5]).tolist(), p_boot=p0, n=nj)
        cells.append(f'agree_k1|{m}|intercept_zero_dlogpt')
    pk = lambda c: res[c]['p_perm'] if 'p_perm' in res[c] else res[c]['p_boot']
    adj = T.holm([pk(c) for c in cells])
    for c, a in zip(cells, adj): res[c]['p_holm'] = float(a)
    out['holm_family'] = dict(n_tests=len(cells), cells=cells); out['results'] = res
    for c in cells: log(c, json.dumps(res[c]))
    (RES / 'h35_verify3.json').write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o)))
    log('saved')


if __name__ == '__main__':
    main()
