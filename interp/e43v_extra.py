"""E43 verifier, extra controls (fresh sample, true W and Z jets only).
(1) Q2 / count-quadrupole edits vs their matched controls restricted to true W and Z jets (the E43 main run keeps only
    3-4 % of W/Z jets for Q2p, because raising |C2| by 20 % at fixed M2 is often infeasible for two-prong jets with Q2 near 1),
    with a paired bootstrap over jets and the real 10 % mass change (dil10) on the same jets as the calibration.
(2) C2ST with a selection-matched natural set: natural jets of one half restricted to jets the edit kept vs edited jets of the
    other half (the main script compares edited kept jets against ALL natural jets, so its AUC also measures the selection).
  python -m interp.e43v_extra [--seed 2343] [--pc 3000]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'


def qci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]; return np.quantile(v, [.025, .975]).tolist() if len(v) else [None, None]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=2343); ap.add_argument('--pc', type=int, default=3000)
    a = ap.parse_args()
    import interp.e43_azimuth_route_jedi as E
    import interp.e41_true_mass_layers_jedi as E41
    from interp.jedi_core import Jedi, sample
    from interp.x10_das_jedi import clean
    from interp.h12_boost_panel import mcnemar
    OUT.mkdir(parents=True, exist_ok=True); R = dict(node='E43 verify extra', seed=a.seed, per_class=a.pc)
    x, y, _ = sample(a.pc, a.seed); keep = (y == E.W_) | (y == E.Z_); x, y = x[keep], y[keep]; N = len(x)
    R['sample'] = dict(call=f'interp.jedi_core.sample(per_class={a.pc}, seed={a.seed}), true W and Z only', N=N)
    net = Jedi(64); F = E.Fwd(net)
    Ed, info, sub = E.make_edits(x, a.seed + 7); E.harm_edits(x, [('Xq2', 'r', 2)], np.random.default_rng(a.seed + 8), Ed, info)
    R['edit_info'] = info; E.log(f'edits: {json.dumps(info)}')
    lo0 = F.run(x)['logits']; m0 = E.margin(lo0); c0 = lo0.argmax(1); LO = {k: F.run(v[0])['logits'] for k, v in Ed.items()}
    R['accuracy_WZ_5class'] = float((c0 == y).mean())
    rng = np.random.default_rng(a.seed + 21); out = {}
    for fam in ('Q2', 'Xq2'):
        for sg in ('p', 'm'):
            ke, kc = f'{fam}{sg}', f'{fam}c{sg}'; ok = Ed[kc][1] & Ed['dil10'][1]; n = int(ok.sum())
            if n < 20: out[ke] = dict(n=n); continue
            de = E.margin(LO[ke][ok]) - m0[ok]; dc = E.margin(LO[kc][ok]) - m0[ok]; dd = E.margin(LO['dil10'][ok]) - m0[ok]
            Wb = rng.poisson(1.0, (2000, n)).astype(float); bm = lambda v: (Wb @ v) / Wb.sum(1)  # noqa: E731
            spec = np.abs(de) - np.abs(dc); bs = bm(spec); ce, cc = LO[ke][ok].argmax(1) != c0[ok], LO[kc][ok].argmax(1) != c0[ok]
            out[ke] = dict(n=n, n_trueW=int((y[ok] == E.W_).sum()), n_trueZ=int((y[ok] == E.Z_).sum()),
                           mean_abs_dmargin_edit=float(np.abs(de).mean()), ci_edit=qci(bm(np.abs(de))),
                           mean_abs_dmargin_ctrl=float(np.abs(dc).mean()), ci_ctrl=qci(bm(np.abs(dc))),
                           specific=float(spec.mean()), specific_ci=qci(bs), p_two_sided=float(min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))),
                           mean_dmargin_edit=float(de.mean()), mean_dmargin_edit_ci=qci(bm(de)), mean_dmargin_ctrl=float(dc.mean()), mean_dmargin_ctrl_ci=qci(bm(dc)),
                           mean_abs_dmargin_dil10=float(np.abs(dd).mean()), specific_over_dil10=float(spec.mean() / np.abs(dd).mean()),
                           specific_over_dil10_ci=qci(bs / bm(np.abs(dd))),
                           calls_changed_edit=float(ce.mean()), calls_changed_ctrl=float(cc.mean()), calls_changed_dil10=float((LO['dil10'][ok].argmax(1) != c0[ok]).mean()),
                           mcnemar=mcnemar(ce, cc))
            E.log(f'{ke}: {json.dumps(clean(out[ke]))}')
        okb = Ed[f'{fam}cp'][1] & Ed[f'{fam}cm'][1]; n = int(okb.sum())
        if n >= 20:
            da = (E.margin(LO[f'{fam}p'][okb]) - E.margin(LO[f'{fam}m'][okb])) / 2; dca = (E.margin(LO[f'{fam}cp'][okb]) - E.margin(LO[f'{fam}cm'][okb])) / 2
            Wb = rng.poisson(1.0, (2000, n)).astype(float); bm = lambda v: (Wb @ v) / Wb.sum(1)  # noqa: E731
            out[f'{fam}_antisym'] = dict(n=n, edit=float(da.mean()), edit_ci=qci(bm(da)), ctrl=float(dca.mean()), ctrl_ci=qci(bm(dca)))
            E.log(f'{fam} antisym: {json.dumps(clean(out[f"{fam}_antisym"]))}')
    R['WZ_tests'] = out
    # selection-matched C2ST
    half = np.random.default_rng(a.seed + 22).permutation(N); h0, h1 = np.zeros(N, bool), np.zeros(N, bool); h0[half[:N // 2]] = True; h1[half[N // 2:]] = True
    FA, FX = E41.c2st_feats(x); CC = {}
    for k in ('turn10', 'ctrl10', 'Q2p', 'Q2cp', 'Q2m', 'Q2cm', 'Xq2p', 'Xq2cp'):
        ok = Ed[k][1]; i0, i1 = np.flatnonzero(h0 & ok), np.flatnonzero(h1 & ok)
        if min(len(i0), len(i1)) < 100: CC[k] = dict(n0=len(i0), n1=len(i1)); continue
        Fa, Fx = E41.c2st_feats(Ed[k][0][i1])
        CC[k] = dict(matched_axis_free=E41.c2st_auc(FA[i0], Fa, 4401), matched_axis_relative=E41.c2st_auc(FX[i0], Fx, 4402),
                     unmatched_axis_free=E41.c2st_auc(FA[h0], Fa, 4403),        # as the main script: all natural vs edited kept
                     selection_only_axis_free=E41.c2st_auc(FA[h0], FA[i1], 4404))  # all natural vs the same kept jets unedited
        E.log(f'C2ST {k}: ' + json.dumps(clean({q: v['auc'] for q, v in CC[k].items()})))
    CC['null_nat_vs_nat'] = E41.c2st_auc(FA[h0], FA[h1], 4405); R['c2st'] = CC
    (OUT / 'e43_verify_extra.json').write_text(json.dumps(clean(R), indent=1)); E.log('wrote e43_verify_extra.json')


if __name__ == '__main__':
    main()
