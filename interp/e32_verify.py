"""E32 verification (fresh sample) + missing controls for the reading "on trigger-style axes JEDI-linear's W-Z margin moves ~0.9 of a
matched dilation; no direction term at 18 mrad; M2-responsive h4/h5 units carry the response".

Network: official JEDI-linear 3-feature n64, exact forward pass (interp.jedi_core).  Sample: sample(5000, 3233) minus jets also in
sample(5000, 3201) or sample(5000, 3202) (byte-identical particle arrays), so it is disjoint from E32's main and replicate samples.
Edits T1 (E-scheme axis of particles quantised to 0.0873 towers), T2 (0.0436 towers), T3 (hardest-particle axis) and controls
C1 (matched dilation), C2 (same |c|, random azimuth), C3 (random rotation about the stored axis) are built with E32's own code
(interp.e32_seeded_axis_jedi.build, rng seed+1), so this part is a straight rerun of E32 --part main on a new sample.
New controls (not in E32):
  Tm  reflected offset: the same per-jet axis offset with the opposite sign (axis at -c).  Odd part e_odd = (e(+c) - e(-c)) / 2,
      even part e_even = (e(+c) + e(-c)) / 2 of the W-Z margin change.  A pure function of r^2 about the axis (A4: M2 about the
      given axis) has an odd part only through the small c.M1 term, which the matched dilations of +c and -c reproduce
      (d_odd = (d(+c) - d(-c)) / 2).  E6v measured an odd term of 12-20 % of the even term at 30-60 mrad along fixed directions.
      Reported: rms(e_odd)/rms(e_even), the same for the dilation prediction (d), and the linear fit e_odd ~ a_eta c_eta + a_phi c_phi.
  TC  translation + compensating dilation: T, then all real particles' angles about the NEW axis x sqrt(M2/M2'), so M2 about the
      axis the network sees is restored to its unedited value while the centroid stays displaced and the true (recentred) jet mass
      drops.  Predictions for R_TC = projection of the TC margin change on the C1 margin change:
        A4 (network reads M2 about the given axis only) -> 0;  recentred/true mass only -> -1;  mixture w*A4 + (1-w)*recentred ->
        R_T = w and R_TC = -(1 - w), i.e. R_T - R_TC = 1.
  Explained share of the T margin change by the dilation direction: 1 - sum (e - R d)^2 / sum e^2 over W/Z-called jets (how much
      of the response is the mass-like part, not only its projection), for T, C2, C3.
  Freeze under T1 without selection leakage: the 4 h4 + 4 h5 live units with the largest standardised response to a 5 % M2
      dilation ranked on half A; evaluated only on half B (T1-changed jets of half B + 2,000 random half-B jets); 100 random holds
      of 8 live h4/h5 units; rank p for calls removed AND for margin removed.
Writes $AI_RESULTS/jedi/e32_verify.json.
  ./job.sh e32-verify 'python -m interp.e32_verify'       (--smoke N: N jets per class)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0_ = time.time(); W_, Z_ = 2, 3


def log(*a): print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


def qci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return np.quantile(v, [.025, .975]).tolist() if len(v) else [None, None]


def main(a):
    from interp.jedi_core import Jedi, sample
    from interp.x10_das_jedi import clean, RidgeR, mass
    from interp.e32_seeded_axis_jedi import build, reaxis, dilate, m2, mcn, proj
    sm = a.smoke; pc = sm or 5000; seed = 3233; NB = 500; NNULL = 10 if sm else 100
    OUT.mkdir(parents=True, exist_ok=True); JS = OUT / ('e32_verify' + ('_smoke' if sm else '') + '.json')
    R = dict(sample=f'sample(per_class={pc}, seed={seed}) minus jets in sample({pc}, 3201) and sample({pc}, 3202)',
             network='official JEDI-linear 3-feature n64 (jedi_core.Jedi(64))')
    save = lambda: JS.write_text(json.dumps(clean(R), indent=1))
    net = Jedi(64)
    x, y, _ = sample(pc, seed); x = np.ascontiguousarray(x, np.float32)
    hs = set()
    for s in (3201, 3202):
        xo, _, _ = sample(pc, s); hs |= {xo[j].tobytes() for j in range(len(xo))}; del xo
    keep = np.array([x[j].tobytes() not in hs for j in range(len(x))]); R['overlap_removed'] = int((~keep).sum()); del hs
    x, y = x[keep], y[keep]; J = len(y); R['N'] = J; R['N_by_class'] = np.bincount(y, minlength=5).tolist()
    B0 = net.run(x, keep=('logits', 'h4', 'h5'), batch=1000); c0 = B0['logits'].argmax(1); R['acc'] = float((c0 == y).mean())
    log(f'N {J} acc {R["acc"]:.4f} overlap removed {R["overlap_removed"]}')
    logm0 = np.log(np.maximum(mass(x), 1e-3)); pr = np.random.default_rng(seed + 10).permutation(J); hA = np.sort(pr[:J // 2])
    hB = np.sort(pr[J // 2:]); nf = 2 * len(hA) // 3; RO = RidgeR(B0['h4'], logm0, hA[:nf], hA[nf:])
    mg = lambda L: L[:, W_] - L[:, Z_]
    base = dict(call=c0, margin=mg(B0['logits']), ro=RO(B0['h4']))
    E, R['axis_info'] = build(x, np.random.default_rng(seed + 1))
    Wb = np.random.default_rng(seed + 2).poisson(1.0, (NB, J)).astype(np.float32)
    wz = (c0 == W_) | (c0 == Z_); M2 = m2(x)

    def resp(xx):
        P = net.run(xx, keep=('logits',), batch=1000)['logits']; return dict(call=P.argmax(1), margin=mg(P))

    def expl(e, d, sel):
        """share of sum e^2 explained by R d (R = projection), with bootstrap CI"""
        e, d = e[sel], d[sel]; wb = Wb[:, sel]
        Rb = (wb @ (e * d)) / np.maximum(wb @ (d * d), 1e-30)
        R0 = (e * d).sum() / (d * d).sum(); v = 1 - ((e - R0 * d) ** 2).sum() / (e * e).sum()
        vb = []
        for i in range(len(Rb)):
            w = wb[i]; vb.append(1 - (w * (e - Rb[i] * d) ** 2).sum() / max((w * e * e).sum(), 1e-30))
        return dict(explained=float(v), ci=qci(vb), corr=float(np.corrcoef(e, d)[0, 1]))

    R['edits'] = {}; RS_T1 = None
    for t in ('T1', 'T2', 'T3'):
        Et = E[t]; ce, cp = Et['ce'], Et['cp']; c = Et['c']
        xm = reaxis(x, -ce, -cp); M2m = m2(xm); C1m = dilate(x, np.sqrt(M2m / M2))
        M2t = m2(Et['x']); TC = dilate(Et['x'], np.sqrt(M2 / M2t))
        chk = float(np.max(np.abs(np.log(m2(TC) / M2))))
        RS = dict(T=resp(Et['x']), C1=resp(Et['C1']), C2=resp(Et['C2']), C3=resp(Et['C3']), Tm=resp(xm), C1m=resp(C1m), TC=resp(TC))
        o = dict(offset_mrad_q50=float(1e3 * np.median(c)), offset_mrad_q10_q90=(1e3 * np.quantile(c, [.1, .9])).tolist(),
                 dlog_mean=float(Et['dlog'].mean()), dlog_frac_pos=float((Et['dlog'] > 0).mean()), TC_M2_restored_max_abs_dlog=chk,
                 TC_true_mass_dlog_mean=float(np.mean(np.log(np.maximum(mass(TC), 1e-3)) - logm0)),
                 T_true_mass_dlog_max_abs=float(np.max(np.abs(np.log(np.maximum(mass(Et['x']), 1e-3)) - logm0))))
        d = RS['C1']['margin'] - base['margin']; dm = RS['C1m']['margin'] - base['margin']
        o['by_edit'] = {}
        for k, r in RS.items():
            ch = r['call'] != c0; wzf = (c0 == W_) & (r['call'] == Z_); zwf = (c0 == Z_) & (r['call'] == W_)
            oo = dict(calls_changed=float(ch.mean()), calls_changed_ci=qci((Wb @ ch.astype(np.float32)) / Wb.sum(1)),
                      n_W_to_Z=int(wzf.sum()), n_Z_to_W=int(zwf.sum()),
                      flow_table=np.array([[int(((c0 == i) & (r['call'] == j)).sum()) for j in range(5)] for i in range(5)]).tolist())
            if k != 'C1':
                e = r['margin'] - base['margin']; dd = dm if k == 'Tm' else d
                oo['R_margin'] = proj(Wb, e, dd, wz)
                wb = Wb[:, wz]; Rb = (wb @ (e * dd)[wz]) / np.maximum(wb @ (dd * dd)[wz], 1e-30)
                oo['R_margin']['p_R_eq_0_boot'] = float(min(1.0, 2 * max(min((Rb <= 0).mean(), (Rb >= 0).mean()), 1 / len(Rb))))
                if k in ('T', 'C2', 'C3', 'Tm'): oo['explained_by_dilation'] = expl(e, dd, wz)
            if k != 'C3': oo['mcnemar_vs_C3'] = mcn(ch, RS['C3']['call'] != c0)
            if k == 'T':
                oo['mcnemar_vs_C1'] = mcn(ch, RS['C1']['call'] != c0); oo['mcnemar_vs_C2'] = mcn(ch, RS['C2']['call'] != c0)
                oo['mcnemar_WZ_vs_C1'] = mcn(wzf, (c0 == W_) & (RS['C1']['call'] == Z_))
                oo['mcnemar_vs_Tm'] = mcn(ch, RS['Tm']['call'] != c0)
                oo['mcnemar_WZ_vs_Tm'] = mcn(wzf, (c0 == W_) & (RS['Tm']['call'] == Z_))
                oo['WZ_flow_dlog_pos'] = float((Et['dlog'][wzf] > 0).mean()) if wzf.any() else None
                oo['W_called_dlog_pos_base_rate'] = float((Et['dlog'][c0 == W_] > 0).mean())
            o['by_edit'][k] = oo
        # R_T - R_TC consistency (mixture prediction = 1)
        eT = RS['T']['margin'] - base['margin']; eTC = RS['TC']['margin'] - base['margin']; wb = Wb[:, wz]
        dd_ = wb @ (d * d)[wz]; Rb_T = (wb @ (eT * d)[wz]) / dd_; Rb_TC = (wb @ (eTC * d)[wz]) / dd_
        diff = Rb_T - Rb_TC
        o['R_T_minus_R_TC'] = dict(value=float(o['by_edit']['T']['R_margin']['R'] - o['by_edit']['TC']['R_margin']['R']), ci=qci(diff),
                                   p_eq_1=float(min(1.0, 2 * max(min((diff <= 1).mean(), (diff >= 1).mean()), 1 / len(diff)))))
        # odd / even decomposition
        eTm = RS['Tm']['margin'] - base['margin']; eo, ee = (eT - eTm) / 2, (eT + eTm) / 2; do, de_ = (d - dm) / 2, (d + dm) / 2
        def rms_ratio(a_, b_):
            v = float(np.sqrt((a_[wz] ** 2).sum() / max((b_[wz] ** 2).sum(), 1e-30)))
            vb = np.sqrt((wb @ (a_[wz] ** 2)) / np.maximum(wb @ (b_[wz] ** 2), 1e-30)); return dict(value=v, ci=qci(vb))
        # odd part after removing the dilation-predicted odd part (scaled by R_even)
        Re = float((ee * de_)[wz].sum() / max((de_ * de_)[wz].sum(), 1e-30)); res_o = eo - Re * do
        A = np.stack([ce, cp], 1)[wz]; yo = eo[wz]; coef = np.linalg.lstsq(A, yo, rcond=None)[0]
        cb = []
        for i in range(min(200, NB)):
            w = np.sqrt(wb[i]); cb.append(np.linalg.lstsq(A * w[:, None], yo * w, rcond=None)[0])
        cb = np.array(cb)
        o['odd_even'] = dict(rms_odd_over_even_net=rms_ratio(eo, ee), rms_odd_over_even_dilation_prediction=rms_ratio(do, de_),
                             R_even=Re, rms_odd_residual_over_even=rms_ratio(res_o, ee),
                             mean_abs_margin_change_even=float(np.abs(ee[wz]).mean()), mean_abs_margin_change_odd=float(np.abs(eo[wz]).mean()),
                             fit_odd_on_c=dict(a_eta=float(coef[0]), a_phi=float(coef[1]), a_eta_ci=qci(cb[:, 0]), a_phi_ci=qci(cb[:, 1]),
                                               units='margin change per rad of axis offset (odd part)'),
                             fit_odd_R2=float(1 - ((yo - A @ coef) ** 2).sum() / max((yo ** 2).sum(), 1e-30)))
        R['edits'][t] = o
        b = o['by_edit']
        log(f'{t}: |c| {o["offset_mrad_q50"]:.1f} mrad; calls T {b["T"]["calls_changed"]:.4f} C1 {b["C1"]["calls_changed"]:.4f} '
            f'C2 {b["C2"]["calls_changed"]:.4f} C3 {b["C3"]["calls_changed"]:.4f} Tm {b["Tm"]["calls_changed"]:.4f} TC {b["TC"]["calls_changed"]:.4f}; '
            f'R T {b["T"]["R_margin"]["R"]:.3f} {b["T"]["R_margin"]["ci"]} Tm {b["Tm"]["R_margin"]["R"]:.3f} C2 {b["C2"]["R_margin"]["R"]:.3f} '
            f'C3 {b["C3"]["R_margin"]["R"]:.3f} TC {b["TC"]["R_margin"]["R"]:.3f} {b["TC"]["R_margin"]["ci"]}; '
            f'explained T {b["T"]["explained_by_dilation"]}; odd/even {o["odd_even"]["rms_odd_over_even_net"]} pred {o["odd_even"]["rms_odd_over_even_dilation_prediction"]}; '
            f'WZ T {b["T"]["n_W_to_Z"]} C1 {b["C1"]["n_W_to_Z"]} Tm {b["Tm"]["n_W_to_Z"]}')
        if t == 'T1': RS_T1 = RS; xT1 = Et['x']
        save()

    # ---- freeze under T1 without selection leakage: rank on half A, evaluate on half B
    live = {L: B0[L].std(0) > 1e-9 for L in ('h4', 'h5')}
    xd = dilate(x[hA], np.full(len(hA), np.sqrt(1.05))); Pd = net.run(xd, keep=('h4', 'h5'), batch=1000); top = {}
    for L in ('h4', 'h5'):
        sd = B0[L][hA].std(0) + 1e-12; sc = np.where(live[L], np.abs((Pd[L] - B0[L][hA]).mean(0)) / sd, -1); top[L] = np.argsort(-sc)[:4].tolist()
    R['top_units_halfA'] = top; del xd, Pd
    chT1 = RS_T1['T']['call'] != c0
    ev = np.union1d(hB[chT1[hB]], np.random.default_rng(seed + 3).choice(hB, min(len(hB), 2000), replace=False))
    n_ch = int(chT1[ev].sum()); eT = RS_T1['T']['margin'][ev] - base['margin'][ev]; wzv = wz[ev]

    def held(units):
        ov = {}
        for L, us in units.items():
            m = np.zeros(B0[L].shape[1]); m[us] = 1; ov[L] = (m, B0[L][ev])
        P = net.run_jetwise(xT1[ev], ov, keep=('logits',), batch=1000)['logits']
        ch = P.argmax(1) != c0[ev]; em = mg(P) - base['margin'][ev]
        return float(1 - ch.sum() / max(n_ch, 1)), float(1 - (em * eT)[wzv].sum() / max((eT * eT)[wzv].sum(), 1e-30))
    rc, rm = held(top)
    pool = [('h4', u) for u in np.flatnonzero(live['h4'])] + [('h5', u) for u in np.flatnonzero(live['h5'])]
    pool = [p for p in pool if p[1] not in top[p[0]]]; rs = np.random.default_rng(seed + 4); nc, nm = [], []
    for _ in range(NNULL):
        U = {}
        for i in rs.choice(len(pool), 8, replace=False): U.setdefault(pool[i][0], []).append(int(pool[i][1]))
        a_, b_ = held(U); nc.append(a_); nm.append(b_)
    nc, nm = np.array(nc), np.array(nm)
    R['freeze_h45_top_halfB'] = dict(units=top, n_eval=int(len(ev)), n_T1_changed_eval=n_ch, removed_calls=rc, removed_margin_proj=rm,
                                     null_calls_q50=float(np.median(nc)), null_calls_q975=float(np.quantile(nc, .975)),
                                     null_margin_q50=float(np.median(nm)), null_margin_q975=float(np.quantile(nm, .975)),
                                     p_rank_calls=float((1 + (nc >= rc).sum()) / (1 + len(nc))),
                                     p_rank_margin=float((1 + (nm >= rm).sum()) / (1 + len(nm))), n_null=len(nc))
    log('freeze h45_top (half B):', R['freeze_h45_top_halfB'])
    R['seconds'] = time.time() - T0_; save(); log(f'done, wrote {JS}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', type=int, default=0)
    main(ap.parse_args())
