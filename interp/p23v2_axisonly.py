"""P23v2 — added control for P23 (verifier): does the odd (c -> -c antisymmetric) part of the axis response, and P23's pooled
'centroid' gain g_odd, come from the axis-relative inputs, or from the particle-energy change that an eta shift at fixed pT makes?

Why. With deta stored with the jet's eta sign factored out (part_core.build), +c_eta moves every particle to larger |eta|, so
E_i = pT_i cosh(eta_i) changes by about c_eta tanh|eta_i| (odd in c). The pT-weighted centroid about the stored axis also lies
along eta with one sign for almost all jets (P23v: M1_eta < 0 in 98 % of jets), with size set by the jet's |eta| and width, so
P23's centroid predictor Po ~ c_eta M1_eta / M2 can be collinear with the energy change.

Sample: JetClass test_20M WToQQ_108, ZToQQ_108 (not used by P23 or P23v), PER random rows each (numpy default_rng(2311); never
first-k). Models: ParT_full, ParT_kin, ParticleNet_full, ParticleNet_kin, untrained ParT_full (init1), loaded as in P23.
Edits (all from interp/p23_axis_surface.py):
  raw   P23's 9 x 9 rigid translation grid (80 points), inputs fully rebuilt (as P23).
  ax    the same translations, but only the axis-relative per-particle inputs (deta, dphi, deltaR) are taken from the edited jet;
        every other per-particle feature (log pT, log E, log pT/pT_jet, log E/E_jet, PID, IP) and the four-vectors (hence ParT's
        pair features) are those of the unedited jet. Particle order is unchanged by a translation (pT kept); checked by mask.
  pv    the same translations with every per-particle feature taken from the edited jet but the four-vectors (ParT's pair
        features) from the unedited jet. For a phi-only shift the pair features are rotation-invariant in exact arithmetic, so
        raw - pv on phi-only shifts measures float32 rounding of the pair features (ParT recomputes them from px, py, pz, E).
  dil   P23's 4 matched dilations (raw), for kappa.
Read-out d = change of logit_Z - logit_W on W/Z jets. P23's analyse() on raw and on ax (grid replaced; kappa from raw dilations).
Added per model and mode: odd share on the phi-axis pairs and on the eta-axis pairs separately; per-jet cos of the odd gradient with
the pT-weighted centroid, unweighted centroid and leading particle vs random-other-jet null; raw odd fit with an extra
energy-change regressor c_eta tanh|eta_jet|. Paired bootstrap over jets for raw - ax differences; Holm over the 5 models.
Also: ParT_kin accuracy on the P1 sample with multiplicity-sorted batches (ParTNetwork.run) vs P23's unsorted chunks.

  ./sync.sh && ./part_job.sh c-p23-verify2 'python -m interp.p23v2_axisonly'   (--smoke: 40 jets per class)
  -> $AI_RESULTS/part/p23_verify2[_smoke].json
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SMOKE = '--smoke' in sys.argv
_argv = sys.argv; sys.argv = [_argv[0]]
from interp import p23_axis_surface as P  # noqa: E402
sys.argv = _argv

P.SMOKE = SMOKE
P.SEED = 2311
P.PER = 40 if SMOKE else 2000
P.NB = 200 if SMOKE else 1000
P.NPERM = 200 if SMOKE else 2000
P.FILES = dict(Wqq='WToQQ_108.root', Zqq='ZToQQ_108.root')
KEEP = ['ParT_full', 'ParT_kin', 'PN_full', 'PN_kin', 'ParT_full_untr1']
FN = P.OUTD / f'p23_verify2{"_smoke" if SMOKE else ""}.json'
AXF = ('deta', 'dphi', 'deltaR')
log = P.log


def main():
    from interp.part_core import CL, sub, cat, kin, sample, ParTNetwork
    from jetdistill.part.network import FEATURES
    iW, iZ = CL.index('Wqq'), CL.index('Zqq')
    OUT = dict(models={})
    rng = np.random.default_rng(P.SEED); Js = []; rowinfo = {}
    for c, f in P.FILES.items():
        J, rows = P.load_rows(P.TEST / f, P.PER, rng); Js.append(J); rowinfo[c] = dict(file=f, first_rows=rows[:5].tolist(), n=len(rows))
        log(f'loaded {P.PER} random rows of {f}')
    J = cat(Js); del Js
    keep = J['mask'].sum(1) >= 2; J = sub(J, np.where(keep)[0]); y = J['label'].astype(int); N = len(y)
    OUT['sample'] = dict(files=rowinfo, seed=P.SEED, N=N, dropped=int((~keep).sum()), classes=np.bincount(y, minlength=10).tolist())
    M, torch = P.load_models()
    for k in list(M):
        if k not in KEEP: del M[k]
    names = P.edit_names(); Ei = {nm: i for i, nm in enumerate(names)}
    grid = [nm for nm in names if nm.startswith('tr_')]; run_raw = ['base'] + grid + list(P.REF)
    rd = np.zeros((N, 4, 2), np.float32)                                     # rd / rot not run here (analyse fields ignored)
    pt, de, dp, ms = kin(J); Mo = P.moments(pt, de, dp)
    Lr = {k: np.zeros((N, len(names), 10), np.float32) for k in M}; La = {k: np.zeros((N, len(names), 10), np.float32) for k in M}
    Lv = {k: np.zeros((N, len(names), 10), np.float32) for k in M}
    axi = {f: [i for i, ft in enumerate(FEATURES[f]) if ft[0] in AXF] for f in ('full', 'kin')}
    log(f'axis-only feature rows: {axi}')
    order = np.argsort(J['mask'].sum(1), kind='stable'); t1 = time.time(); maxdiff_phi = 0.0
    for ci_, i in enumerate(range(0, N, P.CH)):
        idx = np.sort(order[i:i + P.CH]); S = P.prep_chunk(sub(J, idx), rd[idx])
        Xb = P.all_inputs(P.make_edit(S, 'base'))
        for nm in run_raw:
            X = Xb if nm == 'base' else P.all_inputs(P.make_edit(S, nm))
            if nm in grid:
                Xa = {}
                for f in X:
                    assert np.array_equal(X[f][2], Xb[f][2])
                    xa = Xb[f][0].copy(); xa[:, axi[f]] = X[f][0][:, axi[f]]; Xa[f] = (xa, Xb[f][1], Xb[f][2])
                Xv = {f: (X[f][0], Xb[f][1], X[f][2]) for f in X}
            for k, mm in M.items():
                Lr[k][idx, Ei[nm]] = P.run_model(mm, X, torch)
                if nm in grid:
                    La[k][idx, Ei[nm]] = P.run_model(mm, Xa, torch)
                    Lv[k][idx, Ei[nm]] = Lr[k][idx, Ei[nm]] if mm['kind'] == 'pn' else P.run_model(mm, Xv, torch)
                    if nm.startswith('tr_+0.00') or nm.startswith('tr_-0.00'):
                        maxdiff_phi = max(maxdiff_phi, float(np.abs(La[k][idx, Ei[nm]] - Lr[k][idx, Ei[nm]]).max()))
                else:
                    La[k][idx, Ei[nm]] = Lr[k][idx, Ei[nm]]; Lv[k][idx, Ei[nm]] = Lr[k][idx, Ei[nm]]
        if ci_ % 5 == 0: log(f'  chunk {ci_ + 1} / {int(np.ceil(N / P.CH))} ({time.time() - t1:.0f} s)')
    log(f'edits done ({time.time() - t1:.0f} s); max |logit raw - ax| on phi-only shifts {maxdiff_phi:.2e}')
    OUT['sanity_phi_only_raw_vs_ax_maxabs_logit'] = maxdiff_phi

    sel = np.isin(y, [iW, iZ]); idx = np.where(sel)[0]; n = len(idx)
    pairs = [(ce, cp) for ce in P.GRID for cp in P.GRID if (ce, cp) != (0, 0) and (ce > 0 or (ce == 0 and cp > 0))]
    a = np.array(pairs); ip = [Ei[P.K(*c)] for c in pairs]; im = [Ei[P.K(-c[0], -c[1])] for c in pairs]
    iphi = [j for j, (ce, cp) in enumerate(pairs) if ce == 0]; ieta = [j for j, (ce, cp) in enumerate(pairs) if cp == 0]
    Mi = {k: v[idx].astype(np.float64) for k, v in Mo.items()}
    teta = np.tanh(np.abs(J['jet_eta'][idx].astype(np.float64)))
    OUT['M1eta_neg_frac'] = float((Mi['M1e'] < 0).mean())
    OUT['corr_M1eta_over_M2_vs_tanh_abs_eta'] = float(np.corrcoef(Mi['M1e'] / Mi['M2'], teta)[0, 1])
    Pf = lambda ce, cp: 0.5 * np.log(np.maximum(P.M2_after(Mi, ce, cp), 1e-30) / Mi['M2'])
    Po06 = (Pf(0.06, 0) - Pf(-0.06, 0)) / 2
    OUT['corr_Po_eta006_vs_tanh_abs_eta'] = float(np.corrcoef(Po06, teta)[0, 1])
    pt_, de_, dp_ = (x[idx].astype(np.float64) for x in (pt, de, dp)); real = pt_ > 0
    V = dict(pt_centroid=np.stack([(pt_ * de_).sum(1), (pt_ * dp_).sum(1)], 1),
             unweighted_centroid=np.stack([(de_ * real).sum(1), (dp_ * real).sum(1)], 1),
             leading_particle=np.stack([de_[:, 0], dp_[:, 0]], 1))
    Wb = P.boot_W(n, P.NB, np.random.default_rng(11)); perm = np.random.default_rng(12).permutation(n)
    cosf = lambda u, v: (u * v).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-30)
    pdiff = []
    for k in M:
        r = {}; B = Lr[k][:, 0]; r['acc_WZ_sample'] = float((B[sel].argmax(1) == y[sel]).mean())
        GB = {}
        DZ = {}
        for mode, L in dict(raw=Lr[k], pv=Lv[k], ax=La[k]).items():
            dzw = (L[..., iZ] - L[..., iW]) - (B[:, None, iZ] - B[:, None, iW]); DZ[mode] = dzw[idx].astype(np.float64)
            A = P.analyse(dzw, Mo, names, sel, rd, np.random.default_rng(P.SEED + 1), y, iW, iZ, None, None)
            GB[mode] = (A['_go_b'], A['_ge_b'])
            o = {q: A[q] for q in ('kappa', 'odd_var_share', 'g_odd', 'g_even', 'g_odd_minus_even', 'perm_p_b_odd', 'a_eta', 'a_phi',
                                   'perm_p_a_eta', 'R2_odd_centroid_only', 'R2_odd_fixed_only', 'R2_even')}
            o['phi006'] = A['phi006']
            d = dzw[idx].astype(np.float64); odd = (d[:, ip] - d[:, im]) / 2; even = (d[:, ip] + d[:, im]) / 2
            for ax_, ii in dict(phi_axis=iphi, eta_axis=ieta).items():
                so, se = (odd[:, ii] ** 2).sum(1), (even[:, ii] ** 2).sum(1)
                o[f'odd_share_{ax_}'] = dict(v=float(so.sum() / (so.sum() + se.sum())), ci=P.ci((Wb @ so) / (Wb @ (so + se))))
                o[f'_so_{ax_}'], o[f'_st_{ax_}'] = so, so + se
            g = np.linalg.solve(a.T @ a, a.T @ odd.T).T
            o['cos_oddgrad'] = {}
            for vn, v in V.items():
                c = cosf(g, v); c0 = cosf(g, v[perm])
                o['cos_oddgrad'][vn] = dict(mean=float(c.mean()), null=float(c0.mean()), diff_ci=P.ci((Wb @ (c - c0)) / n), p=P.p_two((Wb @ (c - c0)) / n))
            # odd fit with the energy-change regressor
            Pp = np.stack([Pf(ce, cp) for ce, cp in pairs], 1); Pm = np.stack([Pf(-ce, -cp) for ce, cp in pairs], 1); Po = (Pp - Pm) / 2
            ce_ = np.broadcast_to(a[:, 0], (n, len(pairs))); cp_ = np.broadcast_to(a[:, 1], (n, len(pairs)))
            Xo = np.stack([Po, ce_, cp_, ce_ * teta[:, None]], 2); bo, bo_b, _, _ = P.fit_boot(Xo, odd, Wb)
            kap, kap_b = A['kappa']['v'], None
            o['odd_fit_with_energy_proxy'] = dict(b_Po=P.st(bo[0], bo_b[:, 0]), g_Po=float(bo[0] / kap), b_ceta_tanh=P.st(bo[3], bo_b[:, 3]),
                                                  a_eta=P.st(bo[1], bo_b[:, 1]))
            r[mode] = o
        # rounding / pair-feature and energy-feature contributions on axis-only shifts (W/Z jets)
        cols = dict(phi_only=[Ei[nm] for nm in grid if nm.startswith('tr_+0.00')], eta_only=[Ei[nm] for nm in grid if nm.endswith('_+0.00')])
        r['route_split'] = {}
        for cn, cc in cols.items():
            dr, dv, da = DZ['raw'][:, cc], DZ['pv'][:, cc], DZ['ax'][:, cc]
            ms_ = lambda z: float(np.mean(z ** 2))
            r['route_split'][cn] = dict(ms_raw=ms_(dr), ms_raw_minus_pv=ms_(dr - dv), ms_pv_minus_ax=ms_(dv - da), ms_ax=ms_(da),
                                        mean_abs_raw=float(np.abs(dr).mean()), mean_abs_raw_minus_pv=float(np.abs(dr - dv).mean()),
                                        mean_abs_pv_minus_ax=float(np.abs(dv - da).mean()),
                                        frac_jets_raw_minus_pv_gt_0p1=float((np.abs(dr - dv) > 0.1).mean()))
        dg = GB['raw'][0] - GB['ax'][0]
        r['g_odd_raw_minus_ax'] = dict(v=r['raw']['g_odd']['v'] - r['ax']['g_odd']['v'], ci=P.ci(dg), p=P.p_two(dg))
        de_b = GB['raw'][1] - GB['ax'][1]
        r['g_even_raw_minus_ax'] = dict(v=r['raw']['g_even']['v'] - r['ax']['g_even']['v'], ci=P.ci(de_b), p=P.p_two(de_b))
        for ax_ in ('phi_axis', 'eta_axis'):
            sr = (Wb @ r['raw'][f'_so_{ax_}']) / (Wb @ r['raw'][f'_st_{ax_}']); sa = (Wb @ r['ax'][f'_so_{ax_}']) / (Wb @ r['ax'][f'_st_{ax_}'])
            r[f'odd_share_{ax_}_raw_minus_ax'] = dict(v=r['raw'][f'odd_share_{ax_}']['v'] - r['ax'][f'odd_share_{ax_}']['v'], ci=P.ci(sr - sa), p=P.p_two(sr - sa))
        pdiff.append(r['g_odd_raw_minus_ax']['p'] or 1.0)
        r['calls_raw'] = P.calls(B, {nm: Lr[k][:, Ei[nm]] for nm in ('tr_+0.00_+0.06', 'tr_+0.06_+0.00', 'tr_-0.06_+0.00')}, y, iW, iZ)
        r['calls_ax'] = P.calls(B, {nm: La[k][:, Ei[nm]] for nm in ('tr_+0.06_+0.00', 'tr_-0.06_+0.00')}, y, iW, iZ)
        OUT['models'][k] = r
        log(f'{k}: acc {r["acc_WZ_sample"]:.4f}; raw odd share {r["raw"]["odd_var_share"]["v"]:.3f} (phi {r["raw"]["odd_share_phi_axis"]["v"]:.3f}, '
            f'eta {r["raw"]["odd_share_eta_axis"]["v"]:.3f}) g_o {r["raw"]["g_odd"]["v"]:.2f} g_e {r["raw"]["g_even"]["v"]:.2f} a_eta {r["raw"]["a_eta"]["v"]:.2f}; '
            f'ax odd share {r["ax"]["odd_var_share"]["v"]:.3f} (phi {r["ax"]["odd_share_phi_axis"]["v"]:.3f}, eta {r["ax"]["odd_share_eta_axis"]["v"]:.3f}) '
            f'g_o {r["ax"]["g_odd"]["v"]:.2f} g_e {r["ax"]["g_even"]["v"]:.2f} a_eta {r["ax"]["a_eta"]["v"]:.2f}; '
            f'raw with energy proxy g_Po {r["raw"]["odd_fit_with_energy_proxy"]["g_Po"]:.2f}; pv odd share {r["pv"]["odd_var_share"]["v"]:.3f} g_o {r["pv"]["g_odd"]["v"]:.2f}; '
            f'route split {r["route_split"]}')
        with open(FN, 'w') as f: json.dump(P.strip(OUT), f, indent=1)
    for k, h in zip(M, P.holm(pdiff)): OUT['models'][k]['g_odd_raw_minus_ax']['p_holm'] = h
    # ParT_kin P1-sample accuracy: sorted-batch ParTNetwork.run vs P23's unsorted chunks
    try:
        J1 = sample(per_class=100 if SMOKE else 1000, start=3000); y1 = J1['label'].astype(int); X1 = P.all_inputs(J1); acc = {}
        for k in ('ParT_full', 'ParT_kin'):
            Lp = np.concatenate([P.run_model(M[k], {f: tuple(a_[s] for a_ in X1[f]) for f in X1}, torch) for s in [slice(j, j + P.CH) for j in range(0, len(y1), P.CH)]])
            net = ParTNetwork('full' if k == 'ParT_full' else 'kin'); Ls = net.run(J1)['logits']
            acc[k] = dict(unsorted_chunks=float((Lp.argmax(1) == y1).mean()), sorted_batches=float((Ls.argmax(1) == y1).mean()),
                          max_abs_logit_diff=float(np.abs(Lp - Ls).max()))
        OUT['acc_P1_sample'] = acc; log(f'P1 sample acc {acc}')
    except Exception as ex:                                            # noqa: BLE001
        log(f'P1 sample failed {ex!r}')
    with open(FN, 'w') as f: json.dump(P.strip(OUT), f, indent=1)
    log(f'wrote {FN}'); log('P23v2 DONE')


if __name__ == '__main__':
    main()
