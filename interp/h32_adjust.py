"""H32 follow-up: residual side-effect adjustment of D - cside. The c-side control reproduces only part of D's tau32 and prong
pT-fraction change, so per (sample, model) the per-jet contrast y_j = score_D - score_cside of top jets is regressed, pooled
over the four dc cells (both signs), on the per-jet differences (D minus cside, re-clustered observables)
    y = b0 + b_cos * d|cos theta*| + b_tau * d tau32 + b_hi * d f_hi + b_lo * d f_lo
(f_hi, f_lo = largest / smallest sorted kT-3 subjet pT fraction). b_cos is the score change per unit |cos theta*| at zero
residual tau32 / fraction mismatch. Built-in check: MLPx and MLP3 have no access to |cos theta*| or the pair masses, so their
b_cos measures leakage of unmodelled side effects. 95 % bootstrap CI over jets (B = 2000), two-sided bootstrap p, Holm over
networks x samples. Run: python -m interp.h32_adjust  (reads h32_perjet.npz; writes h32_adjust.json)"""
import json
import os
from pathlib import Path
import numpy as np

RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
NETS = ['ParT_top', 'PN_top', 'LNet_top', 'PFN_top', 'EFN_top', 'ParT_kin', 'PN_kin', 'ParT_full', 'PN_full']
B = 2000


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1.0, ps[i] * (m - r))); adj[i] = run
    return adj


def main(tag='h32'):
    Z = np.load(RES / f'{tag}_perjet.npz'); R = json.loads((RES / f'{tag}_top_dalitz_full.json').read_text())
    dcs = [str(d) for d in R['dc_values']]; out = dict(sets={}, B=B, design='y = b0 + b_cos dcos + b_tau dtau32 + b_hi df_hi + b_lo df_lo, '
                                                             'D minus cside per jet, top jets, pooled over dc')
    rng = np.random.default_rng(32)
    for ds in R['sets']:
        y = Z[f'{ds}|y']; Xs, rows_dc, cell = [], [], []
        models = sorted({k.split('|')[-1] for k in Z.files if k.startswith(f'{ds}|id|')})
        Y = {m: [] for m in models}
        for dc in dcs:
            ok = Z[f'{ds}|acc|{dc}|D'] & Z[f'{ds}|acc|{dc}|cphi'] & Z[f'{ds}|acc|{dc}|cside'] & (y == 1)
            r = np.flatnonzero(ok)
            d = lambda q: (Z[f'{ds}|{q}|{dc}|D'] - Z[f'{ds}|{q}|{dc}|cside'])[r]
            fD, fS = Z[f'{ds}|sortedfrac|{dc}|D'][r], Z[f'{ds}|sortedfrac|{dc}|cside'][r]
            Xs.append(np.column_stack([np.ones(len(r)), d('cos'), d('tau32'), fD[:, 2] - fS[:, 2], fD[:, 0] - fS[:, 0]]))
            cell += [dc] * len(r)
            for m in models: Y[m].append((Z[f'{ds}|{dc}|D|{m}'] - Z[f'{ds}|{dc}|cside|{m}'])[r])
        Xm = np.concatenate(Xs).astype(np.float64); cell = np.array(cell); n = len(Xm)
        bi = rng.integers(0, n, (B, n)); S = dict(n=int(n), n_per_dc={dc: int((cell == dc).sum()) for dc in dcs},
                                                 mean_regressors={dc: Xm[cell == dc, 1:].mean(0).tolist() for dc in dcs},
                                                 regressor_corr=np.corrcoef(Xm[:, 1:].T).tolist(), models={})
        for m in models:
            yy = np.concatenate(Y[m]).astype(np.float64)
            coef = np.linalg.lstsq(Xm, yy, rcond=None)[0]
            bc = np.array([np.linalg.lstsq(Xm[i], yy[i], rcond=None)[0] for i in bi])
            ci = np.percentile(bc, [2.5, 97.5], 0); p = 2 * np.minimum((bc[:, 1] <= 0).mean(), (bc[:, 1] >= 0).mean())
            S['models'][m] = dict(b0=float(coef[0]), b_cos=float(coef[1]), b_tau=float(coef[2]), b_hi=float(coef[3]), b_lo=float(coef[4]),
                                  b_cos_ci95=ci[:, 1].tolist(), b_tau_ci95=ci[:, 2].tolist(), b_cos_p_boot=float(max(p, 1 / B)),
                                  angle_part_per_dc={dc: float(coef[1] * Xm[cell == dc, 1].mean()) for dc in dcs},
                                  side_part_per_dc={dc: float((Xm[cell == dc, 2:] @ coef[2:]).mean()) for dc in dcs})
        out['sets'][ds] = S
        for m in models:
            if '@' in m: continue
            M = S['models'][m]
            print(f"[{ds}] {m:9s} b_cos {M['b_cos']:+.3f} {np.round(M['b_cos_ci95'], 3)} p {M['b_cos_p_boot']:.3g} b_tau {M['b_tau']:+.2f} "
                  f"angle part +0.3/+0.6 {M['angle_part_per_dc'].get('0.3', 0):+.3f}/{M['angle_part_per_dc'].get('0.6', 0):+.3f}", flush=True)
        for base in ('MLP3', 'MLP5', 'MLP5c', 'MLPx'):
            v = [S['models'][m]['b_cos'] for m in models if m.split('@')[0] == base]
            S[f'{base}_b_cos_range'] = [float(min(v)), float(max(v))]; print(f'[{ds}] {base} b_cos range over {len(v)} retrains {np.round(S[f"{base}_b_cos_range"], 3)}')
    tests = [(ds, m) for ds in out['sets'] for m in NETS if m in out['sets'][ds]['models']]
    for (ds, m), pa in zip(tests, holm([out['sets'][ds]['models'][m]['b_cos_p_boot'] for ds, m in tests])):
        out['sets'][ds]['models'][m]['b_cos_p_holm'] = float(pa)
    out['holm_family'] = f'{len(tests)} tests (networks x samples), bootstrap p floor 1/{B}'
    (RES / f'{tag}_adjust.json').write_text(json.dumps(out, indent=1)); print('wrote', RES / f'{tag}_adjust.json')


if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else 'h32')
