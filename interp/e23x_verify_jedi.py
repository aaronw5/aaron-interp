"""E23x verify: rerun E23x on fresh samples and add the controls the node lacked.

--part rerun : interp.e23x_centre_jedi.main on test sample(1500, 5311), fit sample(1500, 5310); output redirected to
               $AI_RESULTS/jedi/e23x_verify_rerun.json (same arms, same statistics).
--part extra : -> $AI_RESULTS/jedi/e23x_verify_extra.json; on test sample(1500, 5311) (N 7,500), rigid shifts 30 / 60 mrad x 8 azimuths as in E23x:
  (1) decomposition of the g freeze: g held (all 64), S31 + S58 held, and g held with S31 / S58 restored to their shifted
      values (= g's contribution through the other S units only); path difference P(O_g - O_S31S58) / P(O).
  (2) size-matched null: remove the shift-induced change of S along a random 2-D subspace V of the 19 live S units
      (sd-standardised): S_f = S_c - sd * P_V[(S_c - S_0) / sd]. Freezing S31 + S58 is the coordinate case V = (e31, e58)
      (checked to reproduce the freeze). V = orth(C^(alpha/2) G), G Gaussian 19 x 2, C = covariance of the odd ΔS / sd,
      alpha in {0, 0.5, 1}, so the S-level odd power removed covers S31 + S58's value. Matched band: draws whose S-level
      share is within +-0.10 of S31 + S58's; also the subset whose largest principal cosine with span(e31, e58) < 0.8.
      H0: S31 + S58's downstream share equals that of S-level-matched random subspaces (two-sided rank p).
  (2b) on-manifold null: freeze random subsets (k = 2..17) of the 17 other live S units; downstream share vs S-level share.
  (3) 5a (soft-only move) / 5b (hard-14 move) at fixed M2 on 3,000 test jets: S31 + S58 freeze share and g freeze share on
      the jets feasible for both edits (E23x computed 5a and 5b shares on different jet sets).
  ./job.sh e23x-verify2 'python -m interp.e23x_verify_jedi --part extra && python -m interp.e23x_verify_jedi --part rerun'
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import e23x_centre_jedi as E  # noqa: E402
from interp.e23w_gains_jedi import LIVE_S, ci, clean, shift  # noqa: E402

OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
T0 = time.time(); NB = 2000


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


class _Redirect:
    """stands in for e23x_centre_jedi.OUT so that main() writes e23x_verify_rerun.json"""
    def mkdir(self, **k): OUT.mkdir(parents=True, exist_ok=True)
    def __truediv__(self, name): return OUT / 'e23x_verify_rerun.json'


def boot_ratio(num, den, WB):
    return float(num.sum() / den.sum()), ci((WB @ num) / (WB @ den))


def extra(a):
    from interp.jedi_core import Jedi, sample
    OUT.mkdir(parents=True, exist_ok=True); JS = OUT / 'e23x_verify_extra.json'
    R = dict(node='E23x verify', seed_test=a.seed_test, per=a.per); save = lambda: JS.write_text(json.dumps(clean(R)))
    xt, yt, _ = sample(a.per, a.seed_test); xt = np.ascontiguousarray(xt, np.float32); J = len(xt)
    net = Jedi(64); head = E.Head(net)
    B0 = net.run(xt, keep=('g', 'S', 'logits'), batch=4000); S0 = B0['S'].astype(np.float32); g0 = B0['g']; call0 = B0['logits'].argmax(1)
    h4, h5, z, lo = head(S0)
    R.update(N=J, acc=float((call0 == yt).mean()), head_exact_max_abs_logit=float(np.abs(lo - B0['logits']).max()),
             n_real_mean=float((xt[..., 0] > 0).sum(1).mean()))
    l4 = [int(c) for c in np.flatnonzero(h4.std(0) > 1e-9)]; l5 = [int(c) for c in np.flatnonzero(h5.std(0) > 1e-9)]
    UN = E.Units(head, LIVE_S, l4, l5); U0, _ = UN(S0); sd = U0.std(0); sd[sd < 1e-9] = 1.0; sl = UN.sl
    WB = np.random.default_rng(11).poisson(1.0, (NB, J)).astype(np.float64)
    log(f'N {J} acc {R["acc"]:.4f} head max |dlogit| {R["head_exact_max_abs_logit"]:.1e}')
    Sc = np.zeros((len(E.C), J, 64), np.float32); Sg = np.zeros_like(Sc)
    for i, (r, k, ce, cp) in enumerate(E.C):
        xs = shift(xt, ce, cp)
        # padding check: the shift leaves padded slots untouched
        if i == 0: R['shift_padding_untouched'] = bool(np.array_equal(xs[xt[..., 0] <= 0], xt[xt[..., 0] <= 0]))
        Sc[i] = net.run(xs, keep=('S',), batch=4000)['S']
        Sg[i] = net.run_jetwise(xs, {'g': (np.ones(64), g0)}, keep=('S',), batch=4000)['S']
    log('shift S done')
    O, _ = E.odd_even(UN(Sc)[0], U0, sd)
    odd = lambda S_: E.odd_even(UN(S_)[0], U0, sd)[0]
    O31 = odd(E.freeze(Sc, S0, [31, 58])); Og = odd(Sg)
    Sgx = Sg.copy(); Sgx[..., [31, 58]] = Sc[..., [31, 58]]; Ogx = odd(Sgx); del Sgx
    R['g_freeze_S31_S58_max_abs_change'] = float(np.abs(Sg[..., [31, 58]] - S0[None][..., [31, 58]]).max())
    R['g_freeze_other_S_mean_abs_change_over_sd'] = {f'S{u}': float(np.abs(Sg[..., u] - Sc[..., u]).mean() / max(S0[:, u].std(), 1e-9)) for u in LIVE_S}
    den = E.perjet(O, sl); dec = {}
    for nm, Of in (('S31_S58', O31), ('g_all', Og), ('g_except_S31_S58', Ogx)):
        num = E.perjet(Of, sl); pth = E.perjet(O - Of, sl)
        dec[nm] = {st: dict(odd_removed=E.share(num[st], den[st], WB)[0], path_share=boot_ratio(pth[st], den[st], WB)) for st in sl}
    d_ = E.perjet(Og - O31, sl)
    dec['path_diff_g_vs_S31_S58'] = {st: boot_ratio(d_[st], den[st], WB) for st in sl}
    R['g_decomposition'] = dec; del Og, Ogx, Sg
    log('decomposition', json.dumps(clean(dec))[:3000]); save()

    # ---------------- size-matched random-subspace null
    li = np.array(LIVE_S); i31, i58 = LIVE_S.index(31), LIVE_S.index(58); sdS = S0[:, li].astype(np.float64).std(0)
    D = ((Sc[..., li].astype(np.float64) - S0[None][..., li]) / sdS)                      # (C, J, 19)
    Od = np.stack([0.5 * (D[i] - D[j]) for i, j in E.PAIRS]).reshape(-1, len(li))
    Cod = Od.T @ Od / len(Od); ev, U = np.linalg.eigh(Cod); ev = np.clip(ev, 0, None); del Od
    R['odd_dS_cov_top_eigvec'] = {f'S{u}': float(U[k, -1]) for k, u in enumerate(LIVE_S)}; R['odd_dS_cov_eig_frac'] = (ev[::-1] / ev.sum()).tolist()

    def remove(V):
        Sf = Sc.copy(); Sf[..., li] = (Sc[..., li] - ((D @ V) @ V.T) * sdS).astype(np.float32); return Sf

    def shares(Of): return E.share_point(Of, O, sl)
    Vc = np.zeros((len(li), 2)); Vc[i31, 0] = 1; Vc[i58, 1] = 1
    s_coord = shares(odd(remove(Vc))); s_frz = shares(O31)
    R['coord_projection_vs_freeze'] = dict(projection=s_coord, freeze=s_frz, max_abs_diff=float(max(abs(s_coord[k] - s_frz[k]) for k in sl)))
    log('coordinate projection vs freeze', R['coord_projection_vs_freeze'])
    # (2b) on-manifold coordinate null: freeze random subsets of the 17 other live S units (k = 2..17), so the S-level odd
    # power removed reaches beyond the 2-unit pairs; downstream share vs S-level share, S31 + S58 against that relation
    others = [u for u in LIVE_S if u not in (31, 58)]; rc = np.random.default_rng(91); cdraw = []
    for kk in (2, 3, 4, 6, 8, 10, 12, 14, 16, 17):
        seen = set()
        for _ in range(a.ncoord if kk < 17 else 1):
            sub = tuple(sorted(rc.choice(others, kk, replace=False).tolist()))
            if sub in seen: continue
            seen.add(sub); cdraw.append(dict(k=kk, units=list(sub), share=shares(odd(E.freeze(Sc, S0, list(sub))))))
    R['coord_null'] = cdraw
    cS = np.array([d['share']['S'] for d in cdraw]); cres = dict(n=len(cdraw), S_share_range=[float(cS.min()), float(cS.max())])
    for st in E.DOWN:
        v = np.array([d['share'][st] for d in cdraw]); A = np.c_[np.ones_like(cS), cS]; b, *_ = np.linalg.lstsq(A, v, rcond=None); r_ = v - A @ b
        pred = float(b[0] + b[1] * s_frz['S']); eff = v / np.maximum(cS, 1e-9); mk = cS >= 0.2
        cres[st] = dict(slope=float(b[1]), intercept=float(b[0]), pred_at_obs_S=pred, obs=float(s_frz[st]), resid_sd=float(r_.std(ddof=2)),
                        z=float((s_frz[st] - pred) / r_.std(ddof=2)), p_two=float(math.erfc(abs(s_frz[st] - pred) / r_.std(ddof=2) / math.sqrt(2))),
                        ratio_obs=float(s_frz[st] / s_frz['S']), ratio_null_Sge0p2=dict(n=int(mk.sum()), lo=float(np.percentile(eff[mk], 2.5)) if mk.sum() > 2 else None,
                                                                                   hi=float(np.percentile(eff[mk], 97.5)) if mk.sum() > 2 else None,
                                                                                   mean=float(eff[mk].mean()) if mk.any() else None,
                                                                                   frac_ge_obs=float((eff[mk] >= s_frz[st] / s_frz['S']).mean()) if mk.any() else None),
                        all17=float(v[np.array([d['k'] for d in cdraw]) == 17][0]))
    cres['all17_S'] = float(cS[np.array([d['k'] for d in cdraw]) == 17][0])
    R['coord_null_summary'] = cres; log('coordinate-subset null', json.dumps(clean(cres))); save()

    rng = np.random.default_rng(97); draws = []
    i0, i19 = LIVE_S.index(0), LIVE_S.index(19)       # S0, S19 have zero odd response; a projection injects change into them
    for al in (0.0, 0.5, 1.0):
        M = (U * ev ** (al / 2)) @ U.T
        for _ in range(a.ndraw):
            V, _ = np.linalg.qr(M @ rng.normal(size=(len(li), 2)))
            cos = float(np.linalg.svd(V[[i31, i58], :], compute_uv=False).max()); cos31 = float(np.linalg.norm(V[i31]))
            inj = ((D @ V) @ V.T)[..., [i0, i19]]
            draws.append(dict(alpha=al, cos_e31_e58=cos, cos_e31=cos31, injected_S0_S19_rms_sd=float(np.sqrt((inj ** 2).mean())), share=shares(odd(remove(V)))))
        log(f'alpha {al}: {len(draws)} draws')
    R['null_draws'] = draws
    obs = s_frz; sS = np.array([d['share']['S'] for d in draws]); cosv = np.array([d['cos_e31_e58'] for d in draws])
    R['null_S_share_range'] = [float(sS.min()), float(sS.max())]
    res = {}
    for nmb, msk in (('matched', np.abs(sS - obs['S']) <= 0.10), ('matched_cos_lt_0.8', (np.abs(sS - obs['S']) <= 0.10) & (cosv < 0.8)), ('all', np.ones(len(draws), bool))):
        res[nmb] = dict(n=int(msk.sum()))
        if msk.sum() < 3: continue
        for st in E.STAGES:
            v = np.array([d['share'][st] for d in draws])[msk]
            p2 = float(min(1, 2 * min((1 + (v >= obs[st]).sum()) / (len(v) + 1), (1 + (v <= obs[st]).sum()) / (len(v) + 1))))
            res[nmb][st] = dict(mean=float(v.mean()), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), obs=float(obs[st]), p_rank_two_sided=p2,
                                frac_null_ge_obs=float((v >= obs[st]).mean()))
    # downstream share vs S-level share across all draws: linear fit, S31 + S58 residual
    fit = {}
    for st in E.DOWN:
        v = np.array([d['share'][st] for d in draws]); A = np.c_[np.ones_like(sS), sS]; b, *_ = np.linalg.lstsq(A, v, rcond=None); r_ = v - A @ b
        pred = float(b[0] + b[1] * obs['S']); fit[st] = dict(intercept=float(b[0]), slope=float(b[1]), pred_at_obs_S=pred, obs=float(obs[st]),
                                                            resid_z=float((obs[st] - pred) / r_.std()), resid_sd=float(r_.std()))
    res['linear_fit_vs_S_share'] = fit
    R['matched_null'] = res; log('matched null', json.dumps(clean(res))[:3000]); save()
    del D

    # ---------------- 5a / 5b on jointly feasible jets
    i5 = np.sort(np.random.default_rng(23).choice(J, min(a.n5, J), replace=False)); xx = xt[i5]; S0s = S0[i5]; U0s = U0[i5]; g0s = g0[i5]
    Ed = {}
    for ed in ('soft', 'hard'):
        Se = np.zeros((len(E.C), len(xx), 64), np.float32); Sge = np.zeros_like(Se); okj = np.ones(len(xx), bool)
        for i, (r, k, ce, cp) in enumerate(E.C):
            xe, ok = E.centre_edit(xx, ed, ce, cp); okj &= ok
            if i == 0: R[f'{ed}_padding_untouched'] = bool(np.array_equal(xe[xx[..., 0] <= 0], xx[xx[..., 0] <= 0]))
            Se[i] = net.run(xe, keep=('S',), batch=4000)['S']; Sge[i] = net.run_jetwise(xe, {'g': (np.ones(64), g0s)}, keep=('S',), batch=4000)['S']
        Oe = E.odd_even(UN(Se)[0], U0s, sd)[0]; Of = E.odd_even(UN(E.freeze(Se, S0s, [31, 58]))[0], U0s, sd)[0]; Ofg = E.odd_even(UN(Sge)[0], U0s, sd)[0]
        Ed[ed] = dict(O=Oe, Of=Of, Ofg=Ofg, ok=okj)
    okb = Ed['soft']['ok'] & Ed['hard']['ok']; Wj = np.random.default_rng(37).poisson(1.0, (NB, int(okb.sum()))).astype(np.float64)
    c06p = np.flatnonzero(E.PRING == 1); r5 = dict(n_jets=len(xx), n_soft_ok=int(Ed['soft']['ok'].sum()), n_hard_ok=int(Ed['hard']['ok'].sum()), n_both=int(okb.sum()))
    for ed in ('soft', 'hard'):
        d = Ed[ed]; den_ = E.perjet(d['O'][:, okb], sl)
        r5[ed] = {st: dict(S31_S58=E.share(E.perjet(d['Of'][:, okb], sl)[st], den_[st], Wj)[0], g_all=E.share(E.perjet(d['Ofg'][:, okb], sl)[st], den_[st], Wj)[0]) for st in sl}
    r5['soft_over_hard_rms_06'] = {}
    for st, s in sl.items():
        pa = (Ed['soft']['O'][c06p][:, okb][..., s].astype(np.float64) ** 2).sum((0, 2)); pb = (Ed['hard']['O'][c06p][:, okb][..., s].astype(np.float64) ** 2).sum((0, 2))
        r5['soft_over_hard_rms_06'][st] = dict(ratio=float(np.sqrt(pa.sum() / pb.sum())), ci=ci(np.sqrt((Wj @ pa) / (Wj @ pb))))
    # are the jointly feasible jets typical? class mix and multiplicity
    yb = yt[i5][okb]; r5['both_class_frac'] = [float((yb == c).mean()) for c in range(5)]; r5['all_class_frac'] = [float((yt[i5] == c).mean()) for c in range(5)]
    r5['n_real_both_mean'] = float((xx[okb][..., 0] > 0).sum(1).mean()); r5['n_real_all_mean'] = float((xx[..., 0] > 0).sum(1).mean())
    R['arm5_joint'] = r5; log('arm5 joint', json.dumps(clean(r5))[:3000])
    R['wall_s'] = time.time() - T0; save(); log('done ->', JS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=('extra', 'rerun'), required=True)
    ap.add_argument('--per', type=int, default=1500); ap.add_argument('--seed_test', type=int, default=5311); ap.add_argument('--seed_fit', type=int, default=5310)
    ap.add_argument('--ndraw', type=int, default=40); ap.add_argument('--ncoord', type=int, default=25); ap.add_argument('--n5', type=int, default=3000)
    a = ap.parse_args()
    if a.part == 'extra':
        extra(a)
    else:
        E.OUT = _Redirect()
        sys.argv = [sys.argv[0], '--per', str(a.per), '--seed_test', str(a.seed_test), '--seed_fit', str(a.seed_fit)]
        E.main()


if __name__ == '__main__':
    main()
