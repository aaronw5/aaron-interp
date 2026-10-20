"""P23b — per-jet follow-up of P23 (CPU, from the P23 npz): which per-jet vector the odd (direction-dependent) part of the
axis response follows, whether the even part's anisotropy follows the jet's own orientation, and per-jet calibration.

Input: $AI_RESULTS/part/p23_axis_surface.npz (per-jet change of logit_Z - logit_W for the 92 P23 edits, 8 models) and the same
jets re-read with P23's sampler (JetClass test_20M *_105, default_rng(2305), 3,000 random rows per class; checked equal labels).
W and Z jets only (N = 6,000).
Per jet (from the 40 mirror pairs of the 9 x 9 grid): odd(c) = (d(c) - d(-c))/2, even(c) = (d(c) + d(-c))/2;
  gradient g_j = least-squares fit odd(c) ~ g_j . c; quadratic form H_j: even(c) ~ c^T H_j c (3 parameters).
Candidate jet vectors for the odd gradient (all in the stored (deta, dphi) frame): pT-weighted centroid M1 = sum pT r;
  unweighted centroid sum r / N; energy-weighted centroid; leading-particle position r_1; log(pT)-weighted centroid.
  Metric: mean cos(g_j, v_j) with a 95 % jet bootstrap; null = the same with v taken from a random other jet (permutation).
Even anisotropy vs the jet's pT-weighted second-moment tensor T = sum pT r r^T / M0: cos between the doubled-angle vectors
  (H_ee - H_pp, 2 H_ep) and (T_ee - T_pp, 2 T_ep) (cos = 1: the response grows fastest along the jet's long axis; -1: across it);
  relative anisotropy |(H_ee - H_pp, 2 H_ep)| / (H_ee + H_pp).
Per-jet calibration: kappa_j = slope through 0 of the 4 matched-dilation responses on log k_j; R^2 of the even part under
  g kappa_j Pe (one pooled g) vs g Pe (P23's pooled form).

  ./sync.sh && AI_PY=conda ./job.sh c-p23b-perjet 'JETDISTILL_TAGGER=part python -m interp.p23b_perjet'
  -> $AI_RESULTS/part/p23b_perjet.json
"""
import json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_argv = sys.argv; sys.argv = [_argv[0]]
from interp import p23_axis_surface as P  # noqa: E402
sys.argv = _argv

NB = 1000


def main():
    from interp.part_core import cat, sub, kin, CL
    iW, iZ = CL.index('Wqq'), CL.index('Zqq')
    Z = np.load(P.OUTD / 'p23_axis_surface.npz'); y0 = Z['y']
    rng = np.random.default_rng(P.SEED); Js = []
    for c, f in P.FILES.items():
        J, _ = P.load_rows(P.TEST / f, P.PER, rng); Js.append(J)
    J = cat(Js); keep = J['mask'].sum(1) >= 2; J = sub(J, np.where(keep)[0])
    assert np.array_equal(J['label'].astype(int), y0), 'sample mismatch'
    pt, de, dp, ms = kin(J); pt, de, dp = (a.astype(np.float64) for a in (pt, de, dp)); real = pt > 0
    E = J['energy'].astype(np.float64) * real
    sel = np.isin(y0, [iW, iZ]); idx = np.where(sel)[0]; n = len(idx)
    pt, de, dp, real, E = pt[idx], de[idx], dp[idx], real[idx], E[idx]
    M0 = pt.sum(1); N = real.sum(1)
    lp = np.where(real, np.maximum(np.log(np.maximum(pt, 1e-9)) + 1.0, 0.0), 0)       # log pT + 1 (>= 0) weights
    V = dict(pt_centroid=np.stack([(pt * de).sum(1), (pt * dp).sum(1)], 1) / M0[:, None],
             unweighted_centroid=np.stack([(de * real).sum(1), (dp * real).sum(1)], 1) / N[:, None],
             energy_centroid=np.stack([(E * de).sum(1), (E * dp).sum(1)], 1) / E.sum(1)[:, None],
             logpt_centroid=np.stack([(lp * de).sum(1), (lp * dp).sum(1)], 1) / np.maximum(lp.sum(1), 1e-9)[:, None],
             leading_particle=np.stack([de[:, 0], dp[:, 0]], 1))
    Tee, Tpp, Tep = (pt * de * de).sum(1) / M0, (pt * dp * dp).sum(1) / M0, (pt * de * dp).sum(1) / M0
    tvec = np.stack([Tee - Tpp, 2 * Tep], 1)
    Mo = dict(M0=M0, M1e=(pt * de).sum(1), M1p=(pt * dp).sum(1), M2=(pt * (de ** 2 + dp ** 2)).sum(1))
    names = P.edit_names(); Ei = {k: i for i, k in enumerate(names)}
    pairs = [(ce, cp) for ce in P.GRID for cp in P.GRID if (ce, cp) != (0, 0) and (ce > 0 or (ce == 0 and cp > 0))]
    a = np.array(pairs); ip = [Ei[P.K(*c)] for c in pairs]; im = [Ei[P.K(-c[0], -c[1])] for c in pairs]
    Pf = lambda ce, cp: 0.5 * np.log(np.maximum(P.M2_after(Mo, ce, cp), 1e-30) / Mo['M2'])
    Pe = np.stack([(Pf(ce, cp) + Pf(-ce, -cp)) / 2 for ce, cp in pairs], 1)
    kk = np.stack([Pf(*P.REF[k]) for k in P.REF], 1)
    Q = np.stack([a[:, 0] ** 2, a[:, 1] ** 2, 2 * a[:, 0] * a[:, 1]], 1)
    Wb = P.boot_W(n, NB, np.random.default_rng(5)); perm = np.random.default_rng(6).permutation(n)
    cosf = lambda u, v: (u * v).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-30)
    res = dict(n=int(n), centroid_rad=dict(M1e_over_M0_median_abs=float(np.median(np.abs(V['pt_centroid'][:, 0]))),
                                           M1p_over_M0_median_abs=float(np.median(np.abs(V['pt_centroid'][:, 1]))),
                                           unweighted_eta_median_abs=float(np.median(np.abs(V['unweighted_centroid'][:, 0]))),
                                           unweighted_phi_median_abs=float(np.median(np.abs(V['unweighted_centroid'][:, 1])))), models={})
    for key in [k for k in Z.files if k.startswith('dzw_')]:
        d = Z[key][idx].astype(np.float64); m = key[4:]
        odd = (d[:, ip] - d[:, im]) / 2; even = (d[:, ip] + d[:, im]) / 2
        g = np.linalg.solve(a.T @ a, a.T @ odd.T).T; h = np.linalg.solve(Q.T @ Q, Q.T @ even.T).T
        o = dict(grad_norm_median=float(np.median(np.linalg.norm(g, axis=1))), cos={})
        for vn, v in V.items():
            c = cosf(g, v); c0 = cosf(g, v[perm])
            o['cos'][vn] = dict(mean=float(c.mean()), ci=P.ci((Wb @ c) / n), mean_abs=float(np.abs(c).mean()),
                                null_mean=float(c0.mean()), null_ci=P.ci((Wb @ c0) / n), diff_ci=P.ci((Wb @ (c - c0)) / n),
                                p_vs_null=P.p_two((Wb @ (c - c0)) / n))
        # grad projected on candidate vectors: share of per-jet odd energy along v
        hv = np.stack([h[:, 0] - h[:, 1], 2 * h[:, 2]], 1); ct = cosf(hv, tvec); ct0 = cosf(hv, tvec[perm])
        o['even_aniso'] = dict(rel_median=float(np.median(np.linalg.norm(hv, axis=1) / np.maximum(np.abs(h[:, 0] + h[:, 1]), 1e-12))),
                               cos_with_T=dict(mean=float(ct.mean()), ci=P.ci((Wb @ ct) / n), null_mean=float(ct0.mean()),
                                               null_ci=P.ci((Wb @ ct0) / n), p_vs_null=P.p_two((Wb @ (ct - ct0)) / n)))
        # even response along the jet's long vs short axis: H projected on T's principal directions
        th = 0.5 * np.arctan2(2 * Tep, Tee - Tpp); u = np.stack([np.cos(th), np.sin(th)], 1); w = np.stack([-np.sin(th), np.cos(th)], 1)
        quad = lambda z: h[:, 0] * z[:, 0] ** 2 + h[:, 1] * z[:, 1] ** 2 + 2 * h[:, 2] * z[:, 0] * z[:, 1]
        hl, hs = quad(u), quad(w); r_ = (hl.sum() - hs.sum()) / (hl.sum() + hs.sum())
        rb = ((Wb @ hl) - (Wb @ hs)) / ((Wb @ hl) + (Wb @ hs))
        o['even_long_vs_short'] = dict(H_long_mean=float(hl.mean()), H_short_mean=float(hs.mean()), rel_diff=float(r_), ci=P.ci(rb), p0=P.p_two(rb))
        # per-jet calibration
        yd = np.stack([d[:, Ei[k]] for k in P.REF], 1); kap_j = (yd * kk).sum(1) / (kk * kk).sum(1)
        X1, X2 = Pe, kap_j[:, None] * Pe
        b1 = (X1 * even).sum() / (X1 * X1).sum(); b2 = (X2 * even).sum() / (X2 * X2).sum()
        o['even_R2_pooled'] = P.r2(even, b1 * X1); o['even_R2_perjet_kappa'] = P.r2(even, b2 * X2); o['g_perjet_kappa'] = float(b2)
        o['even_R2_perjet_quadratic'] = P.r2(even, h @ Q.T)
        o['kappa_j'] = dict(median=float(np.median(kap_j)), q10=float(np.quantile(kap_j, .1)), q90=float(np.quantile(kap_j, .9)))
        res['models'][m] = o
        P.log(f'{m}: cos(g, M1) {o["cos"]["pt_centroid"]["mean"]:.3f} (null {o["cos"]["pt_centroid"]["null_mean"]:.3f}); '
              f'unweighted {o["cos"]["unweighted_centroid"]["mean"]:.3f}; lead {o["cos"]["leading_particle"]["mean"]:.3f}; '
              f'aniso cos T {ct.mean():.3f}; long-short {r_:.3f}; even R2 pooled {o["even_R2_pooled"]:.3f} per-jet kappa {o["even_R2_perjet_kappa"]:.3f}')
    fn = P.OUTD / 'p23b_perjet.json'; json.dump(P.strip(res), open(fn, 'w'), indent=1); P.log(f'wrote {fn}')


if __name__ == '__main__':
    main()
