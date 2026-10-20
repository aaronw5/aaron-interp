"""H32 verification, part 2 (loop D): on the ORIGINAL H32 sample (h32_perjet.npz, seed 3201), refit the H32 residual
regression y = s_D - s_cside = b0 + b_cos dcos + b_tau dtau32 + b_f13 df(col 13) + b_f11 df(col 11) separately on +dc cells
(raise |cos theta*|) and -dc cells (lower it), and the non-parametric matched subset (top jets whose D - cside residual
|dtau32|, |df_11|, |df_13| are all < 0.01), to test whether the pooled b_cos of H32 is the same in both directions.
95 % bootstrap CI over jets (B = 2000, rng 3321); b_cos minus each MLPx retrain (paired bootstrap).
Run: ./job.sh d-h32-vsplit 'python -m interp.h32_verify_split'   Output: $AI_RESULTS/physics/h32_verify_split.json"""
import json
import os
from pathlib import Path
import numpy as np

RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
NETS = ['ParT_top', 'PN_top', 'LNet_top', 'PFN_top', 'EFN_top', 'ParT_kin', 'PN_kin', 'ParT_full', 'PN_full']
B = 2000


def main():
    Z = np.load(RES / 'h32_perjet.npz'); R = json.loads((RES / 'h32_top_dalitz_full.json').read_text())
    dcs = [str(d) for d in R['dc_values']]; rng = np.random.default_rng(3321); out = dict(sets={}, B=B)
    for ds in R['sets']:
        y = Z[f'{ds}|y']; models = sorted({k.split('|')[-1] for k in Z.files if k.startswith(f'{ds}|id|')})
        nets = [m for m in NETS if m in models]; mlpx = [m for m in models if m.split('@')[0] == 'MLPx']
        mlp5 = [m for m in models if m.split('@')[0] == 'MLP5']
        cells = {}
        for dc in dcs:
            r = np.flatnonzero(Z[f'{ds}|acc|{dc}|D'] & Z[f'{ds}|acc|{dc}|cphi'] & Z[f'{ds}|acc|{dc}|cside'] & (y == 1))
            d = lambda q: (Z[f'{ds}|{q}|{dc}|D'] - Z[f'{ds}|{q}|{dc}|cside'])[r].astype(np.float64)
            fD, fS = Z[f'{ds}|sortedfrac|{dc}|D'][r], Z[f'{ds}|sortedfrac|{dc}|cside'][r]
            Xc = np.column_stack([np.ones(len(r)), d('cos'), d('tau32'), fD[:, 2] - fS[:, 2], fD[:, 0] - fS[:, 0]])
            Yc = np.column_stack([(Z[f'{ds}|{dc}|D|{m}'] - Z[f'{ds}|{dc}|cside|{m}'])[r] for m in models]).astype(np.float64)
            cells[dc] = (Xc, Yc)
        S = {}
        for vname, sel in (('all', dcs), ('pos', [d for d in dcs if float(d) > 0]), ('neg', [d for d in dcs if float(d) < 0])):
            Xm = np.concatenate([cells[d][0] for d in sel]); Ym = np.concatenate([cells[d][1] for d in sel]); n = len(Xm)
            coef = np.linalg.lstsq(Xm, Ym, rcond=None)[0][1]
            bi = rng.integers(0, n, (B, n)); bc = np.stack([np.linalg.lstsq(Xm[i], Ym[i], rcond=None)[0][1] for i in bi])
            V = dict(n=int(n), models={})
            for j, m in enumerate(models):
                v = dict(b_cos=float(coef[j]), ci95=np.percentile(bc[:, j], [2.5, 97.5]).tolist())
                if m in nets:
                    dd = [bc[:, j] - bc[:, models.index(r_)] for r_ in mlpx]
                    v['minus_MLPx_ci95'] = [np.percentile(x, [2.5, 97.5]).tolist() for x in dd]
                    v['minus_MLPx_pmax'] = float(max(max(2 * min((x <= 0).mean(), (x >= 0).mean()), 1 / B) for x in dd))
                V['models'][m] = v
            V['MLPx_range'] = [min(V['models'][m]['b_cos'] for m in mlpx), max(V['models'][m]['b_cos'] for m in mlpx)]
            V['MLP5_range'] = [min(V['models'][m]['b_cos'] for m in mlp5), max(V['models'][m]['b_cos'] for m in mlp5)]
            S[f'regression|{vname}'] = V
            print(f'[{ds}] {vname} n {n}: ' + ' '.join(f"{m} {V['models'][m]['b_cos']:+.2f}{np.round(V['models'][m]['ci95'], 2)}" for m in nets)
                  + f" | MLPx {np.round(V['MLPx_range'], 2)} MLP5 {np.round(V['MLP5_range'], 2)}", flush=True)
        for sgn, sel in (('pos', [d for d in dcs if float(d) > 0]), ('neg', [d for d in dcs if float(d) < 0])):
            Xm = np.concatenate([cells[d][0] for d in sel]); Ym = np.concatenate([cells[d][1] for d in sel])
            k = (np.abs(Xm[:, 2]) < 0.01) & (np.abs(Xm[:, 3]) < 0.01) & (np.abs(Xm[:, 4]) < 0.01); n = int(k.sum())
            V = dict(n=n, mean_dcos=float(Xm[k, 1].mean()) if n else None, models={})
            if n >= 15:
                bi = rng.integers(0, n, (B, n)); Yk = Ym[k]
                for j, m in enumerate(models):
                    V['models'][m] = dict(mean=float(Yk[:, j].mean()), ci95=np.percentile(Yk[bi, j].mean(1), [2.5, 97.5]).tolist())
                    if m in nets:
                        dd = [Yk[:, j] - Yk[:, models.index(r_)] for r_ in mlpx]
                        V['models'][m]['minus_MLPx_ci95'] = [np.percentile(x[bi].mean(1), [2.5, 97.5]).tolist() for x in dd]
                V['MLPx_range'] = [min(V['models'][m]['mean'] for m in mlpx), max(V['models'][m]['mean'] for m in mlpx)]
                print(f'[{ds}] matched {sgn} n {n} dcos {V["mean_dcos"]:+.3f}: ' + ' '.join(
                    f"{m} {V['models'][m]['mean']:+.2f}{np.round(V['models'][m]['ci95'], 2)}" for m in nets) + f" | MLPx {np.round(V['MLPx_range'], 2)}", flush=True)
            S[f'matched|{sgn}'] = V
        out['sets'][ds] = S
    (RES / 'h32_verify_split.json').write_text(json.dumps(out, indent=1)); print('wrote h32_verify_split.json')


if __name__ == '__main__':
    main()
