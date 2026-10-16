"""E5 verification: direct tests of the E4/E5 readings (R1 radial symmetry, R2 S29 naming, R3 context through four objects).

Fresh jets: sample(2000 per class, seed 7). Nothing is refit; the stored E4/E5 coefficients (results/jedi/e4_e5.json) are applied.
  1. Rigid rotation of every jet about its axis by 45/90/180 deg in (Δη, Δφ) (r of every particle unchanged), plus a per-particle
     azimuth scramble (r unchanged, pairwise distances not): ΔS_c in units of SD(S_c) for every live S unit (fixed-point network and
     the masked-float network without input rounding), ΔP / SD(P) for the live first-round units, Δg / SD(g), calls changed overall
     and per class, and which S units carry the call change (one rotated S_c patched into the unrotated network at a time).
  2. Reflections Δφ -> -Δφ and Δη -> -Δη, same measurements.
  3. Stored naming formulas (3-term hand-built fit, single term, hinge sum) applied to the fresh sample: R² vs the fixed-point S_c,
     for all 17 particle units (also on the jets not in the E5 fit sample). Causal edit: r x 1.1 for particles with r > 0.15
     (Σr² and ΣpT r² change by known amounts, N(pT<2, r>0.1) does not): measured ΔS_c vs the formula's prediction.
  4. g -> mean (fresh mean and the E4 sample's mean), S_c -> mean-context value for S0/S30/S31/S58 together and one at a time,
     agreement of the four-object edit's calls with g -> mean's calls, and controls: four random live S units (5 draws from the 19
     live units, 5 draws from the 15 other units).
  5. Code checks: the zero-bit masks from k+i+f vs the quantizer's b; pool scale; overlap of the E5 fit sample (seed 0) with the
     causal sample (seed 1) and with this sample; per-unit agreement of the masked-float response with the fixed-point response.
Output: $AI_RESULTS/jedi/e5_verify.json; the log is stdout.  Run: python -m interp.e5_verify
"""
import json, os, sys, time, traceback
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.jedi_core import Jedi, sample
from interp.e4_e5_jedi import handbuilt, hinge_sum, run_pj, run_ov, masks, make_masked, bits, r2, OUT
from interp.e5_causal_jedi import in_range, seen, lin_formula

SEED = 7
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
CLASSES = ['g', 'q', 'W', 'Z', 't']
TOP4 = [0, 30, 31, 58]


def sample_idx(per, seed):
    """the indices sample() draws (same code path), to measure overlaps between samples"""
    from jetdistill.config import RESULTS
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')
    y = np.asarray(d['y']); rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per, replace=False) for c in range(5)]))


def rotate(x, deg):
    th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th)
    xe = x.copy()
    xe[..., 1] = c * x[..., 1] - s * x[..., 2]
    xe[..., 2] = s * x[..., 1] + c * x[..., 2]
    return xe.astype(np.float32)


def reflect(x, axis):
    xe = x.copy(); xe[..., axis] = -xe[..., axis]
    return xe


def scramble(x, rng):
    xe = x.copy(); real = x[..., 0] > 0
    r = np.hypot(x[..., 1], x[..., 2]); th = rng.uniform(-np.pi, np.pi, r.shape)
    xe[..., 1] = np.where(real, r * np.cos(th), 0); xe[..., 2] = np.where(real, r * np.sin(th), 0)
    return xe.astype(np.float32)


def stats(v):
    v = np.asarray(v, np.float64)
    return dict(mean=float(v.mean()), sem=float(v.std() / np.sqrt(len(v))), sd=float(v.std()))


def main():
    t0 = time.time(); np.seterr(all='ignore'); OUT.mkdir(parents=True, exist_ok=True)
    R5 = json.loads((OUT / 'e4_e5.json').read_text())
    UN = {u['S']: u for u in R5['E5']['units']}
    live_S = np.array(R5['E4']['live_u']); live_g = np.array(R5['E4']['live_g'])
    part_units = [c for c in UN if not UN[c].get('context_only')]
    x, y, _ = sample(PER, seed=SEED); J = len(x)
    real = x[..., 0] > 0; pt = x[..., 0].astype(np.float64); r = np.hypot(x[..., 1], x[..., 2]).astype(np.float64)
    net = Jedi(64); M3, KEEP = masks(net)
    rng = np.random.default_rng(SEED)
    res = dict(N=J, per_class=PER, seed=SEED, checks={})
    out_path = OUT / 'e5_verify.json'

    def save():
        out_path.write_text(json.dumps(res, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))

    B = net.run(x, keep=('P', 'g', 'S', 'logits'))
    P, g, S, call = B['P'], B['g'], B['S'], B['logits'].argmax(1)
    acc = float((call == y).mean())
    live_P = np.flatnonzero(P.std(0) > 1e-9)
    sdS = S.std(0); sdP = P.std(0); sdg = g.std(0)
    MF = make_masked(net, live_S); Mb = MF(x); cmf = Mb['logits'].argmax(1)
    res['baseline'] = dict(acc=acc, acc_masked_float=float((cmf == y).mean()), masked_float_calls_differ=float((cmf != call).mean()),
                           real_per_jet=float(real.sum(1).mean()), live_P=live_P.tolist(), live_S=live_S.tolist())
    print(f'{J} jets ({PER}/class, seed {SEED}); fixed-point accuracy {acc:.4f}; masked float {res["baseline"]["acc_masked_float"]:.4f} '
          f'({res["baseline"]["masked_float_calls_differ"]*100:.2f}% calls differ); live P {len(live_P)}, live S {len(live_S)}', flush=True)

    # ================================================================ 5a. code checks that need the network
    try:
        C5 = res['checks']['5_code'] = {}
        for k in ('l3', 'qs1', 'qs', 'add'):
            L = net.L[k]; qq = L.iq.quantizers if hasattr(L.iq, 'quantizers') else [L.iq]
            for qi, q in enumerate(qq):
                Q = q.quantizer
                kif = (np.asarray(Q.k) + np.asarray(Q.i) + np.asarray(Q.f)) > 0; bb = np.asarray(Q.b) > 0
                C5[f'mask_{k}{qi}_kif_vs_b_disagree'] = int((kif != bb).sum()); C5[f'mask_{k}{qi}_kept'] = int(bb.sum())
        C5['qs1_scale'] = net.pool_scale[1]; C5['qs_scale'] = net.pool_scale[0]
        C5['KEEP_matches_json_pooled_slots'] = all(sorted(np.flatnonzero(KEEP[:, c]).tolist()) == sorted(UN[c]['pooled_slots']) for c in UN)
        i0, i1, i7 = sample_idx(4000, 0), sample_idx(2000, 1), sample_idx(PER, SEED)
        C5['overlap_seed1_in_seed0'] = int(len(np.intersect1d(i0, i1))); C5['overlap_seed7_in_seed0'] = int(len(np.intersect1d(i0, i7)))
        C5['overlap_seed7_in_seed1'] = int(len(np.intersect1d(i1, i7)))
        fresh = ~np.isin(i7, i0)                                         # jets of this sample not in the E5 fit sample
        C5['f_maskedfloat_vs_fixedpoint_R2'] = {c: UN[c].get('f_clean_vs_fixedpoint_R2') for c in part_units}
        C5['hinge_R2_vs_fixedpoint_f'] = {c: UN[c].get('hinge_R2_vs_fixedpoint_f') for c in part_units}
        print('5 code: mask disagreements (k+i+f>0 vs b>0):', {k: v for k, v in C5.items() if k.endswith('disagree')},
              f"| pool scales qs {C5['qs_scale']} qs1 {C5['qs1_scale']} | KEEP == json slots {C5['KEEP_matches_json_pooled_slots']} | "
              f"overlap seed1 in seed0 {C5['overlap_seed1_in_seed0']}/10000, seed7 in seed0 {C5['overlap_seed7_in_seed0']}/{J}", flush=True)
        print('   masked-float f vs fixed-point f R² per unit:', {c: round(v, 3) for c, v in C5['f_maskedfloat_vs_fixedpoint_R2'].items() if v is not None}, flush=True)
        save()
    except Exception:
        traceback.print_exc(); fresh = np.ones(J, bool)

    # ================================================================ 1–2. symmetry edits
    def compare(name, xe):
        inr = in_range(net, xe)
        E = net.run(xe, keep=('P', 'g', 'S', 'logits')); Me = MF(xe)
        ce = E['logits'].argmax(1); ch = ce != call
        rec = dict(n_out_of_range=int((~inr).sum()), acc=float((ce == y).mean()), calls_changed=float(ch.mean()),
                   calls_changed_in_range=float(ch[inr].mean()) if inr.any() else None,
                   calls_changed_per_class={CLASSES[k]: float(ch[y == k].mean()) for k in range(5)},
                   calls_changed_masked_float=float((Me['logits'].argmax(1) != cmf).mean()))
        dS = E['S'] - S; dSm = Me['S'] - Mb['S']
        rec['S_rms_over_sd'] = {int(c): float(np.sqrt((dS[:, c] ** 2).mean()) / sdS[c]) for c in live_S}
        rec['S_meanabs_over_sd'] = {int(c): float(np.abs(dS[:, c]).mean() / sdS[c]) for c in live_S}
        rec['S_mean_shift_over_sd'] = {int(c): float(dS[:, c].mean() / sdS[c]) for c in live_S}
        rec['S_rms_over_sd_masked_float'] = {int(c): float(np.sqrt((dSm[:, c] ** 2).mean()) / max(Mb['S'][:, c].std(), 1e-9)) for c in live_S}
        dP = E['P'] - P; dg = E['g'] - g
        rec['P_rms_over_sd'] = {int(k): float(np.sqrt((dP[:, k] ** 2).mean()) / sdP[k]) for k in live_P}
        rec['g_rms_over_sd'] = {int(k): float(np.sqrt((dg[:, k] ** 2).mean()) / max(sdg[k], 1e-9)) for k in live_g}
        # the quantized ("seen") r: how much the input grid moves r under this edit
        rs0 = np.hypot(*np.moveaxis(seen(net, x)[..., 1:], -1, 0)); rs1 = np.hypot(*np.moveaxis(seen(net, xe)[..., 1:], -1, 0))
        dr = np.abs(rs1 - rs0)
        rec['seen_r_meanabs_change'] = dict(all=float(dr[real].mean()), **{f'slots_{a}_{b - 1}': float(dr[:, a:b][real[:, a:b]].mean())
                                                                            for a, b in ((0, 16), (16, 32), (32, 48), (48, 64))})
        rec['float_r_maxabs_change'] = float(np.abs(np.hypot(xe[..., 1], xe[..., 2]) - r).max())
        # which S units carry the call change: one edited S_c patched into the unedited network
        carry = {}
        for c in live_S:
            m = np.zeros(64); m[c] = 1
            c2 = run_ov(net, x, {'S': (m, E['S'])}, keep=('logits',))['logits'].argmax(1)
            carry[int(c)] = float((c2 != call).mean())
        rec['calls_changed_by_single_S_patch'] = dict(sorted(carry.items(), key=lambda kv: -kv[1]))
        m = np.zeros(64); m[live_S] = 1
        c2 = run_ov(net, x, {'S': (m, E['S'])}, keep=('logits',))['logits'].argmax(1)
        rec['calls_changed_all_live_S_patch'] = float((c2 != call).mean())
        top = sorted(rec['S_rms_over_sd'].items(), key=lambda kv: -kv[1])[:6]
        topm = sorted(rec['S_rms_over_sd_masked_float'].items(), key=lambda kv: -kv[1])[:4]
        topP = sorted(rec['P_rms_over_sd'].items(), key=lambda kv: -kv[1])
        print(f"\n{name}: calls changed {rec['calls_changed']*100:.2f}% (in-range jets {rec['calls_changed_in_range']*100:.2f}%, "
              f"{rec['n_out_of_range']} out of range; masked float {rec['calls_changed_masked_float']*100:.2f}%); acc {rec['acc']:.4f}; "
              f"per class {' '.join(f'{k} {v*100:.1f}%' for k, v in rec['calls_changed_per_class'].items())}\n"
              f"   ΔS rms/SD (fixed point) top: {' '.join(f'S{c} {v:.3f}' for c, v in top)} | masked float top: {' '.join(f'S{c} {v:.3f}' for c, v in topm)}\n"
              f"   ΔS rms/SD all: {' '.join(f'S{c} {v:.2f}' for c, v in rec['S_rms_over_sd'].items())}\n"
              f"   ΔP rms/SD: median {np.median([v for _, v in topP]):.2f}, max {topP[0][1]:.2f} (P{topP[0][0]}), "
              f"n(P with rms/SD > 0.5) {sum(v > 0.5 for _, v in topP)} of {len(topP)}; Δg rms/SD: "
              f"{' '.join(f'g{k} {v:.2f}' for k, v in rec['g_rms_over_sd'].items())}\n"
              f"   seen-r mean |Δ| {rec['seen_r_meanabs_change']['all']:.4f} (slots 0-15 {rec['seen_r_meanabs_change']['slots_0_15']:.4f}, "
              f"48-63 {rec['seen_r_meanabs_change']['slots_48_63']:.4f}); float r max |Δ| {rec['float_r_maxabs_change']:.1e}\n"
              f"   calls changed by one patched S_c: {' '.join(f'S{c} {v*100:.2f}%' for c, v in list(rec['calls_changed_by_single_S_patch'].items())[:8])}; "
              f"all live S patched {rec['calls_changed_all_live_S_patch']*100:.2f}%", flush=True)
        return rec

    try:
        C1 = res['checks']['1_rotation'] = {}
        for deg in (45, 90, 180):
            C1[f'rot{deg}'] = compare(f'1 rotation {deg} deg', rotate(x, deg)); save()
        C1['azimuth_scramble'] = compare('1 per-particle azimuth scramble', scramble(x, rng)); save()
    except Exception:
        traceback.print_exc()
    try:
        C2 = res['checks']['2_reflection'] = {}
        C2['phi_flip'] = compare('2 reflection Δφ -> -Δφ', reflect(x, 2)); save()
        C2['eta_flip'] = compare('2 reflection Δη -> -Δη', reflect(x, 1)); save()
    except Exception:
        traceback.print_exc()

    # ================================================================ 3. stored formulas on the fresh sample + r-scale edit
    try:
        C3 = res['checks']['3_naming'] = dict(formula_R2={}, r_scale_edit={})
        keeps = {c: np.isin(np.arange(64), UN[c]['pooled_slots']) for c in part_units}
        for c in part_units:
            u = UN[c]; t = S[:, c]; k = keeps[c]
            pH3 = lin_formula(x, u['H_greedy3'], k); pH1 = lin_formula(x, u['H_single'], k)
            ph = hinge_sum(u['hinge'], u['f_pad'], k, x); a, b = u['hinge_sum_rescale']
            C3['formula_R2'][c] = dict(H3=r2(pH3, t), H3_fresh_only=r2(pH3[fresh], t[fresh]), H1=r2(pH1, t), hinge_sum_raw=r2(ph, t),
                                       hinge_sum_rescaled=r2(a * ph + b, t), H3_terms=u['H_greedy3']['terms'],
                                       stored=dict(H3=u['H_greedy3']['r2_test'], H1=u['H_single']['r2_test'], hinge_sum_raw=u['hinge_sum_R2_raw']))
        f3 = C3['formula_R2']
        print('\n3 stored formulas on the fresh sample, R² vs fixed-point S (H3 / H1 / hinge sum raw; stored held-out in brackets):')
        for c in part_units:
            d = f3[c]; s = d['stored']
            print(f"   S{c:<2d} H3 {d['H3']:.3f} [{s['H3']:.3f}] (jets outside the fit sample {d['H3_fresh_only']:.3f}) | H1 {d['H1']:.3f} [{s['H1']:.3f}] | "
                  f"hinge sum {d['hinge_sum_raw']:.3f} [{s['hinge_sum_raw']:.3f}] rescaled {d['hinge_sum_rescaled']:.3f}", flush=True)
        save()
        # causal edit: r x 1.1 for particles with r > 0.15 (azimuth unchanged)
        sel = real & (r > 0.15)
        xe = x.copy(); xe[..., 1] = np.where(sel, 1.1 * x[..., 1], x[..., 1]); xe[..., 2] = np.where(sel, 1.1 * x[..., 2], x[..., 2])
        xe = xe.astype(np.float32)
        valid = in_range(net, xe)
        E = net.run(xe[valid], keep=('S', 'logits')); Fz = run_pj(net, xe[valid], g[valid], keep=('S',))
        ce = E['logits'].argmax(1)
        C3['r_scale_edit']['_n'] = int(valid.sum()); C3['r_scale_edit']['_calls_changed'] = float((ce != call[valid]).mean())
        C3['r_scale_edit']['_jets_with_edited_particles'] = float(sel.any(1)[valid].mean())
        C3['r_scale_edit']['_edited_particles_per_jet'] = float(sel.sum(1)[valid].mean())
        for c in part_units:
            u = UN[c]; k = keeps[c]
            dm = E['S'][:, c] - S[valid, c]; dz = Fz['S'][:, c] - S[valid, c]
            p3 = lin_formula(xe[valid], u['H_greedy3'], k) - lin_formula(x[valid], u['H_greedy3'], k)
            p1 = lin_formula(xe[valid], u['H_single'], k) - lin_formula(x[valid], u['H_single'], k)
            ph = hinge_sum(u['hinge'], u['f_pad'], k, xe[valid]) - hinge_sum(u['hinge'], u['f_pad'], k, x[valid])
            d = dict(measured=stats(dm), measured_g_frozen=stats(dz), sd_S=float(sdS[c]),
                     measured_per_class={CLASSES[q]: float(dm[y[valid] == q].mean()) for q in range(5)})
            for nm, p in (('H3', p3), ('H1', p1), ('hinge', ph)):
                ok = np.std(p) > 1e-9 and np.std(dm) > 1e-9
                d[nm] = dict(mean=float(p.mean()), corr=float(np.corrcoef(p, dm)[0, 1]) if ok else None,
                             slope=float(np.polyfit(p, dm, 1)[0]) if ok else None, R2_of_delta=r2(p, dm) if ok else None,
                             ratio_of_means=float(dm.mean() / p.mean()) if abs(p.mean()) > 1e-9 else None)
            if c == 29:                                                   # the analytic prediction for S29's 3-term formula
                cf = dict(zip(u['H_greedy3']['terms'], u['H_greedy3']['coef']))
                w = sel[valid]
                ana = 0.21 * (cf['ΣpT·r^2'] * (pt[valid] * r[valid] ** 2 * w).sum(1) + cf['Σr^2'] * (r[valid] ** 2 * w).sum(1))
                d['H3_analytic_check_maxdiff'] = float(np.abs(ana - p3).max())
                d['H3_analytic_terms'] = dict(dSpT_r2_mean=float((0.21 * pt[valid] * r[valid] ** 2 * w).sum(1).mean()),
                                              dSr2_mean=float((0.21 * r[valid] ** 2 * w).sum(1).mean()))
            C3['r_scale_edit'][c] = d
        e = C3['r_scale_edit']
        print(f"\n3 edit r x 1.1 for r > 0.15: {e['_n']} in-range jets, {e['_edited_particles_per_jet']:.1f} edited particles/jet, calls changed {e['_calls_changed']*100:.2f}%")
        for c in part_units:
            d = e[c]; f = lambda v: 'n/a' if v is None else f'{v:+.3f}'
            print(f"   S{c:<2d} measured ΔS {d['measured']['mean']:+.3f}±{d['measured']['sem']:.3f} (g frozen {d['measured_g_frozen']['mean']:+.3f}; SD(S) {d['sd_S']:.2f}) | "
                  f"H3 {d['H3']['mean']:+.3f} corr {f(d['H3']['corr'])} slope {f(d['H3']['slope'])} R²(Δ) {f(d['H3']['R2_of_delta'])} | "
                  f"hinge {d['hinge']['mean']:+.3f} corr {f(d['hinge']['corr'])} slope {f(d['hinge']['slope'])} | H1 {d['H1']['mean']:+.3f}"
                  + (f" | analytic check max diff {d['H3_analytic_check_maxdiff']:.1e}" if c == 29 else ''), flush=True)
        save()
    except Exception:
        traceback.print_exc()

    # ================================================================ 4. context through four objects
    try:
        C4 = res['checks']['4_context'] = {}
        gm = g.mean(0)
        x0, y0, _ = sample(4000, seed=0); gm0 = net.run(x0, keep=('g',))['g'].mean(0); del x0, y0
        C4['g_mean_fresh_vs_E4_maxabsdiff_over_sd'] = float((np.abs(gm - gm0) / np.maximum(sdg, 1e-9))[live_g].max())
        one = np.ones(64)
        res_edit = {}
        for nm, gv in (('g_to_mean_fresh', gm), ('g_to_mean_E4sample', gm0), ('g_to_zero', np.zeros(64))):
            c2 = net.run(x, override={'g': (one, gv)})['logits'].argmax(1)
            res_edit[nm] = c2
            C4[nm] = dict(acc=float((c2 == y).mean()), calls_changed=float((c2 != call).mean()),
                          per_class={CLASSES[k]: float((c2 != call)[y == k].mean()) for k in range(5)})
        cg = res_edit['g_to_mean_fresh']
        Sm = net.run(x, override={'g': (one, gm)}, keep=('S',))['S']
        C4['S_meanctx_differs_from_S'] = {int(c): float((np.abs(Sm[:, c] - S[:, c]) > 1e-6).mean()) for c in live_S}

        def patch(units):
            m = np.zeros(64); m[list(units)] = 1
            c2 = run_ov(net, x, {'S': (m, Sm)}, keep=('logits',))['logits'].argmax(1)
            chg = c2 != call
            return dict(units=[int(u) for u in units], calls_changed=float(chg.mean()), acc=float((c2 == y).mean()),
                        agree_with_g_mean_all=float((c2 == cg).mean()),
                        agree_with_g_mean_on_changed=float((c2 == cg)[cg != call].mean()),
                        changed_where_g_mean_unchanged=float((chg & (cg == call)).mean()))
        C4['single'] = {int(c): patch([c]) for c in live_S}
        C4['top4'] = patch(TOP4)
        C4['all_live'] = patch(live_S)
        others = [int(c) for c in live_S if c not in TOP4]
        rc = np.random.default_rng(SEED)
        C4['control_random4_of_19'] = [patch(sorted(rc.choice(live_S, 4, replace=False))) for _ in range(5)]
        C4['control_random4_of_15_others'] = [patch(sorted(rc.choice(others, 4, replace=False))) for _ in range(5)]
        gf, g0, t4, al = C4['g_to_mean_fresh'], C4['g_to_mean_E4sample'], C4['top4'], C4['all_live']
        per_class = ' '.join('%s %.1f%%' % (k, v * 100) for k, v in gf['per_class'].items())
        single = ' '.join('S%d %.2f%%' % (c, d['calls_changed'] * 100) for c, d in sorted(C4['single'].items(), key=lambda kv: -kv[1]['calls_changed'])[:8])
        ctl19 = ' '.join('%s %.2f%%' % (d['units'], d['calls_changed'] * 100) for d in C4['control_random4_of_19'])
        ctl15 = ' '.join('%s %.2f%%' % (d['units'], d['calls_changed'] * 100) for d in C4['control_random4_of_15_others'])
        differs = {c: round(v, 3) for c, v in C4['S_meanctx_differs_from_S'].items() if v > 0}
        print(f"\n4 context: g -> fresh mean: acc {gf['acc']:.4f}, calls changed {gf['calls_changed']*100:.2f}% (per class {per_class}); "
              f"g -> E4-sample mean {g0['calls_changed']*100:.2f}%; g -> 0 {C4['g_to_zero']['calls_changed']*100:.2f}%\n"
              f"   S_c -> mean-context S_c alone: {single}\n"
              f"   four together {TOP4}: calls changed {t4['calls_changed']*100:.2f}%, acc {t4['acc']:.4f}; "
              f"same call as g -> mean on {t4['agree_with_g_mean_all']*100:.2f}% of jets, on {t4['agree_with_g_mean_on_changed']*100:.1f}% of the jets g -> mean changes; "
              f"changed where g -> mean did not {t4['changed_where_g_mean_unchanged']*100:.2f}%\n"
              f"   all 19 live S together {al['calls_changed']*100:.2f}% (agree with g -> mean {al['agree_with_g_mean_all']*100:.2f}%)\n"
              f"   control, 4 random of 19: {ctl19}\n   control, 4 random of the 15 others: {ctl15}\n"
              f"   units whose mean-context S differs from S on > 0 jets: {differs}", flush=True)
        save()
    except Exception:
        traceback.print_exc()

    save()
    print(f'\ndone {time.time() - t0:.0f}s -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
