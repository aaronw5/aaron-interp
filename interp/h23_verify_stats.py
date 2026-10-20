"""H23 verification statistics: one family over the fresh-row rerun (h23_verify_jc_perjet.npz), Holm and BH.
Tests (per-jet paired t-tests unless noted; n = analysis jets):
  H4 rc: (W - N)_net - (W - N)_MLPx (one-sided, down); (W - N)_net - (W - N)_MLPxW (two-sided);
  H4 residual (dN_st - dN_rc) - dTN_st (two-sided);
  H3v rc: (E - B)_net - (E - B)_MLPx (one-sided, down); vs MLPxW (two-sided);
  dose: |W - N| rc at T60 vs T70 and T100 vs T90 (Welch, one-sided, different jet sets).
Run: ./job.sh d-h23-vstat 'python -m interp.h23_verify_stats'   Output: $AI_RESULTS/physics/h23_verify_stats.json
"""
import json
import os
from pathlib import Path
import numpy as np
from scipy import stats

RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
NETS = ('ParT_full', 'ParT_kin', 'PN_full', 'PN_kin')


def tp(x, sign):
    t, p2 = stats.ttest_1samp(np.asarray(x, float), 0.0)
    return float(p2) if sign == 0 else float(p2 / 2 if sign * t > 0 else 1 - p2 / 2)


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); a = np.empty(m); run = 0.0
    for i, k in enumerate(o): run = max(run, min(1.0, (m - i) * p[k])); a[k] = run
    return a


def bh(p):
    p = np.asarray(p); m = len(p); o = np.argsort(p); q = np.empty(m); run = 1.0
    for i in range(m - 1, -1, -1): k = o[i]; run = min(run, p[k] * m / (i + 1)); q[k] = run
    return q


def main():
    Z = np.load(RES / 'h23_verify_jc_perjet.npz'); g = lambda *k: Z['__'.join(('jc',) + k)].astype(np.float64)
    fam = []; WN = {}
    for s in ('T60', 'T70', 'T90', 'T100'):
        d = lambda v, p, m: g('H4', s, v, p, m) - g('H4', s, 'id', p, m)
        rx = d('W', 'st', 'MLPx_jc') - d('N', 'st', 'MLPx_jc'); rw = d('W', 'st', 'MLPxW_jc') - d('N', 'st', 'MLPxW_jc')
        for m in NETS:
            x = d('W', 'rc', m) - d('N', 'rc', m); WN[(s, m)] = x
            fam.append(dict(arm='H4', set=s, model=m, test='rc W-N minus MLPx (one-sided down)', effect=float((x - rx).mean()), n=len(x), p=tp(x - rx, -1)))
            fam.append(dict(arm='H4', set=s, model=m, test='rc W-N minus MLPxW (two-sided)', effect=float((x - rw).mean()), n=len(x), p=tp(x - rw, 0)))
            z = (d('N', 'st', m) - d('N', 'rc', m)) - d('TN', 'st', m)
            fam.append(dict(arm='H4', set=s, model=m, test='residual (dN_st - dN_rc) - dTN_st (two-sided)', effect=float(z.mean()), n=len(z), p=tp(z, 0)))
    for s in ('H3v_60', 'H3v_100'):
        d = lambda v, p, m: g(s, v, p, m) - g(s, 'id', p, m)
        rx = d('E', 'st', 'MLPx_jc') - d('B', 'st', 'MLPx_jc'); rw = d('E', 'st', 'MLPxW_jc') - d('B', 'st', 'MLPxW_jc')
        for m in NETS:
            x = d('E', 'rc', m) - d('B', 'rc', m)
            fam.append(dict(arm='H3v', set=s, model=m, test='rc E-B minus MLPx (one-sided down)', effect=float((x - rx).mean()), n=len(x), p=tp(x - rx, -1),
                            ref_MLPx=float(rx.mean()), ref_MLPxW=float(rw.mean()), net=float(x.mean())))
            fam.append(dict(arm='H3v', set=s, model=m, test='rc E-B minus MLPxW (two-sided)', effect=float((x - rw).mean()), n=len(x), p=tp(x - rw, 0)))
    for m in NETS:
        for a, b in (('T60', 'T70'), ('T100', 'T90')):
            xa, xb = WN[(a, m)], WN[(b, m)]; t, p2 = stats.ttest_ind(xa, xb, equal_var=False)
            fam.append(dict(arm='dose', set=f'{a}_vs_{b}', model=m, test='rc W-N more negative at the larger |mW - 80| (Welch, one-sided)',
                            effect=float(xa.mean() - xb.mean()), n=[len(xa), len(xb)], p=float(p2 / 2 if t < 0 else 1 - p2 / 2)))
    ps = [t['p'] for t in fam]; H = holm(ps); Q = bh(ps)
    for t, h, q in zip(fam, H, Q): t['holm'] = float(h); t['bh'] = float(q)
    out = dict(node='H23_verify', family=f'{len(fam)} tests, Holm and BH over all', n_holm_lt_05=int((H < 0.05).sum()), tests=fam)
    (RES / 'h23_verify_stats.json').write_text(json.dumps(out, indent=1))
    for t in fam: print(t['arm'], t['set'], t['model'], t['test'], f"{t['effect']:+.3f}", f"p {t['p']:.2g} holm {t['holm']:.2g}")


if __name__ == '__main__':
    main()
