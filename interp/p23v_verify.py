"""P23v — verification of P23 (shape of the axis response) on a fresh sample, with two added controls.

Fresh sample: JetClass test_20M WToQQ_107, ZToQQ_107, ZJetsToNuNu_107 (files not used by P23, which used *_105), PER random rows
each (numpy default_rng(2399); never first-k); jets with < 2 particles dropped. Models: ParT_full, ParT_kin, ParticleNet_full,
ParticleNet_kin (released), untrained ParT_full (weaver init, torch seed 1). Edits and read-out exactly as P23 (imported from
interp/p23_axis_surface.py: 9 x 9 rigid-translation grid |c_i| <= 0.08 rad, rd0-3 random directions |c| = 0.06, rot90/180/270,
4 matched dilations; d = change of logit_Z - logit_W vs the rebuilt unedited jet).

A. Reproduction: P23's analyse() on W/Z jets (odd share, g_odd, g_even, kappa, sign-flip p, phi+0.06 decomposition, held-out),
   W -> Z calls on true W.
B. Which angular moment the response follows (competing hypotheses for the even part and the odd gradient).
   S_b(c) = sum_i pT_i |r_i + c|^b about the stored axis, b in {0 (log), 0.5, 1, 2, 3}; P_b(c) = (1/b) log(S_b(c)/S_b(0))
   (b = 0: sum pT [log|r+c| - log|r|] / M0, |r| clipped at 1e-3). A real dilation r -> k r gives P_b = log k for every b, so the
   matched-dilation kappa calibrates every b alike. b = 2 is P23's predictor (P = 1/2 log M2(c)/M2).
   - even part: pooled gain, R2 pooled, R2 with per-jet kappa_j, R2 with a free per-jet scale (shape only); paired bootstrap
     of the R2 difference vs b = 2 (Holm over the 4 alternatives).
   - long-vs-short-axis anisotropy of the even part predicted by each b (per-jet quadratic fit of P_b's even part, same metric as
     P23b) vs the measured one.
   - odd part: the first-order odd vector of S_b is V_b = sum pT |r|^(b-2) r (b = 2: the centroid M1); cos(g_j, V_b) per jet
     vs a random-other-jet null; R2 of odd with a free per-jet scale on Po_b.
   - total grid response with one free per-jet scale on P_b(c) (one quantity, both parts).
C. Is the per-jet kappa_j improvement specific to the matched real change? R2 of the even part (b = 2) with g kappa_j Pe vs
   with per-jet scales from (i) kappa of a random other jet (null), (ii) the rms response to the three rotations at fixed M2,
   (iii) the rms response to the 4 held-out random-direction translations (upper reference; same edit family).

  ./sync.sh && ./part_job.sh c-p23-verify 'python -m interp.p23v_verify'   (--smoke: 60 jets per class)
  -> $AI_RESULTS/part/p23_verify[_smoke].json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SMOKE = '--smoke' in sys.argv
_argv = sys.argv; sys.argv = [_argv[0]]
from interp import p23_axis_surface as P  # noqa: E402
sys.argv = _argv

P.SMOKE = SMOKE
P.SEED = 2399
P.PER = 60 if SMOKE else 2000
P.NB = 200 if SMOKE else 1000
P.NPERM = 200 if SMOKE else 2000
P.FILES = dict(Wqq='WToQQ_107.root', Zqq='ZToQQ_107.root', QCD='ZJetsToNuNu_107.root')
KEEP = ['ParT_full', 'ParT_kin', 'PN_full', 'PN_kin', 'ParT_full_untr1']
BETAS = ['log', 0.5, 1.0, 2.0, 3.0]
FN = P.OUTD / f'p23_verify{"_smoke" if SMOKE else ""}.json'
log = P.log


def S_beta(pt, de, dp, ce, cp, b):
    """pT-weighted moment after a shift; ce, cp: (n,) per jet. Returns (n,)"""
    r = np.sqrt((de + ce[:, None]) ** 2 + (dp + cp[:, None]) ** 2); real = pt > 0
    if b == 'log': return (pt * np.where(real, np.log(np.maximum(r, 1e-3)), 0)).sum(1)
    return (pt * np.where(real, np.maximum(r, 1e-3) ** b, 0)).sum(1)


def P_beta(pt, de, dp, ce, cp, b, S0, M0):
    S = S_beta(pt, de, dp, ce, cp, b)
    return (S - S0) / M0 if b == 'log' else np.log(np.maximum(S, 1e-30) / S0) / b


def main():
    from interp.part_core import CL, sub, cat, kin
    iW, iZ = CL.index('Wqq'), CL.index('Zqq')
    OUT = dict(sample=None, models={})
    rng = np.random.default_rng(P.SEED); Js = []; rowinfo = {}
    for c, f in P.FILES.items():
        J, rows = P.load_rows(P.TEST / f, P.PER, rng); Js.append(J); rowinfo[c] = dict(file=f, first_rows=rows[:5].tolist(), n=len(rows))
        log(f'loaded {P.PER} random rows of {f}')
    J = cat(Js); del Js
    keep = J['mask'].sum(1) >= 2; J = sub(J, np.where(keep)[0]); y = J['label'].astype(int); N = len(y)
    OUT['sample'] = dict(files=rowinfo, seed=P.SEED, N=N, dropped=int((~keep).sum()), classes=np.bincount(y, minlength=10).tolist())
    log(f'N = {N}')
    M, torch = P.load_models()
    for k in list(M):
        if k not in KEEP: del M[k]
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    names = P.edit_names(); E = len(names)
    rd = np.zeros((N, 4, 2), np.float32); th = rng.uniform(0, 2 * np.pi, (N, 4)); rd[..., 0] = 0.06 * np.cos(th); rd[..., 1] = 0.06 * np.sin(th)
    pt, de, dp, ms = kin(J); Mo = P.moments(pt, de, dp)
    L = {k: np.zeros((N, E, 10), np.float32) for k in M}
    order = np.argsort(J['mask'].sum(1), kind='stable'); t1 = time.time()
    for ci_, i in enumerate(range(0, N, P.CH)):
        idx = np.sort(order[i:i + P.CH]); S = P.prep_chunk(sub(J, idx), rd[idx])
        for e, nm in enumerate(names):
            X = P.all_inputs(P.make_edit(S, nm))
            for k, mm in M.items(): L[k][idx, e] = P.run_model(mm, X, torch)
        if ci_ % 5 == 0: log(f'  chunk {ci_ + 1} / {int(np.ceil(N / P.CH))} ({time.time() - t1:.0f} s)')
    log(f'edits done ({time.time() - t1:.0f} s)')
    Ei = {nm: i for i, nm in enumerate(names)}

    # ---------------------------------------------------------------- B: moment family predictors (W/Z jets)
    sel = np.isin(y, [iW, iZ]); idx = np.where(sel)[0]; n = len(idx)
    pt_, de_, dp_ = (a[idx].astype(np.float64) for a in (pt, de, dp)); M0 = pt_.sum(1)
    pairs = [(ce, cp) for ce in P.GRID for cp in P.GRID if (ce, cp) != (0, 0) and (ce > 0 or (ce == 0 and cp > 0))]
    a = np.array(pairs); nK = len(pairs)
    ip = [Ei[P.K(*c)] for c in pairs]; im = [Ei[P.K(-c[0], -c[1])] for c in pairs]
    one = np.ones(n)
    PB = {}
    for b in BETAS:
        S0 = S_beta(pt_, de_, dp_, 0 * one, 0 * one, b)
        Pp = np.stack([P_beta(pt_, de_, dp_, ce * one, cp * one, b, S0, M0) for ce, cp in pairs], 1)
        Pm = np.stack([P_beta(pt_, de_, dp_, -ce * one, -cp * one, b, S0, M0) for ce, cp in pairs], 1)
        rv = np.sqrt(de_ ** 2 + dp_ ** 2); w = np.where(pt_ > 0, pt_ * np.maximum(rv, 1e-3) ** ((0.0 if b == 'log' else b) - 2), 0)
        V = np.stack([(w * de_).sum(1), (w * dp_).sum(1)], 1)
        PB[str(b)] = dict(Po=(Pp - Pm) / 2, Pe=(Pp + Pm) / 2, Pp=Pp, Pm=Pm, V=V)
    log('moment predictors done')
    Tee, Tpp, Tep = (pt_ * de_ * de_).sum(1) / M0, (pt_ * dp_ * dp_).sum(1) / M0, (pt_ * de_ * dp_).sum(1) / M0
    thT = 0.5 * np.arctan2(2 * Tep, Tee - Tpp); u = np.stack([np.cos(thT), np.sin(thT)], 1); wv = np.stack([-np.sin(thT), np.cos(thT)], 1)
    Q = np.stack([a[:, 0] ** 2, a[:, 1] ** 2, 2 * a[:, 0] * a[:, 1]], 1)

    def long_short(even, Wb):
        h = np.linalg.solve(Q.T @ Q, Q.T @ even.T).T
        quad = lambda z: h[:, 0] * z[:, 0] ** 2 + h[:, 1] * z[:, 1] ** 2 + 2 * h[:, 2] * z[:, 0] * z[:, 1]
        hl, hs = quad(u), quad(wv); rb = ((Wb @ hl) - (Wb @ hs)) / ((Wb @ hl) + (Wb @ hs))
        return dict(v=float((hl.sum() - hs.sum()) / (hl.sum() + hs.sum())), ci=P.ci(rb))

    Wb = P.boot_W(n, P.NB, np.random.default_rng(7)); perm = np.random.default_rng(8).permutation(n)
    cosf = lambda g, v: (g * v).sum(1) / np.maximum(np.linalg.norm(g, axis=1) * np.linalg.norm(v, axis=1), 1e-30)
    OUT['moment_family_pred_long_short'] = {k: long_short(v['Pe'], Wb) for k, v in PB.items()}
    OUT['M1e_sign_frac_pos'] = float((Mo['M1e'][idx] > 0).mean())
    OUT['acc'] = {}; OUT['calls_phi006_W'] = {}
    for k in M:
        Lk = L[k]; B = Lk[:, 0]; OUT['acc'][k] = float((B.argmax(1) == y).mean())
        dzw = (Lk[..., iZ] - Lk[..., iW]) - (B[:, None, iZ] - B[:, None, iW])
        r = dict()
        try:
            A = P.analyse(dzw, Mo, names, sel, rd, np.random.default_rng(P.SEED + 1), y, iW, iZ, None, None)
            r['A'] = {kk: A[kk] for kk in ('n', 'kappa', 'odd_var_share', 'g_odd', 'g_even', 'g_odd_minus_even', 'perm_p_b_odd', 'R2_even',
                                           'R2_odd_centroid_only', 'phi006', 'heldout', 'R_matched', 'perjet', 'a_eta', 'a_phi')}
            r['A']['even_eta_minus_phi'] = A['split']['g_even_eta_minus_phi']
        except Exception as ex:                                        # noqa: BLE001
            r['A_error'] = repr(ex); log(f'{k} analyse failed {ex!r}')
        r['calls'] = P.calls(B, {nm: Lk[:, Ei[nm]] for nm in ('tr_+0.00_+0.06', 'tr_+0.00_-0.06', 'rot90')}, y, iW, iZ)
        d = dzw[idx].astype(np.float64); odd = (d[:, ip] - d[:, im]) / 2; even = (d[:, ip] + d[:, im]) / 2
        tot = np.concatenate([d[:, ip], d[:, im]], 1)
        # calibration
        Mi = {kk: v[idx] for kk, v in Mo.items()}
        kk_ = np.stack([0.5 * np.log(np.maximum(P.M2_after(Mi, *P.REF[q]), 1e-30) / Mi['M2']) for q in P.REF], 1)
        yd = np.stack([d[:, Ei[q]] for q in P.REF], 1); kap_j = (yd * kk_).sum(1) / (kk_ * kk_).sum(1); kap = (yd * kk_).sum() / (kk_ * kk_).sum()
        r['kappa'] = float(kap)
        r['observed_long_short'] = long_short(even, Wb)
        g_j = np.linalg.solve(a.T @ a, a.T @ odd.T).T
        fam = {}
        sse2 = None
        for bk, pb in PB.items():
            Pe, Po, V = pb['Pe'], pb['Po'], pb['V']
            be = (Pe * even).sum() / (Pe * Pe).sum()
            X2 = kap_j[:, None] * Pe; b2 = (X2 * even).sum() / (X2 * X2).sum()
            sj = (Pe * even).sum(1) / np.maximum((Pe * Pe).sum(1), 1e-30); res_e = ((even - sj[:, None] * Pe) ** 2).sum(1); sst_e = (even ** 2).sum(1)
            so = (Po * odd).sum(1) / np.maximum((Po * Po).sum(1), 1e-30); res_o = ((odd - so[:, None] * Po) ** 2).sum(1); sst_o = (odd ** 2).sum(1)
            PP = np.concatenate([pb['Pp'], pb['Pm']], 1); st_ = (PP * tot).sum(1) / np.maximum((PP * PP).sum(1), 1e-30)
            res_t = ((tot - st_[:, None] * PP) ** 2).sum(1); sst_t = (tot ** 2).sum(1)
            c = cosf(g_j, V); c0 = cosf(g_j, V[perm])
            fam[bk] = dict(g_even_pooled=float(be / kap), R2_even_pooled=P.r2(even, be * Pe), R2_even_kappa_j=P.r2(even, b2 * X2),
                           R2_even_perjet_scale=dict(v=float(1 - res_e.sum() / sst_e.sum()), ci=P.ci(1 - (Wb @ res_e) / (Wb @ sst_e))),
                           R2_odd_perjet_scale=dict(v=float(1 - res_o.sum() / sst_o.sum()), ci=P.ci(1 - (Wb @ res_o) / (Wb @ sst_o))),
                           R2_total_perjet_scale=dict(v=float(1 - res_t.sum() / sst_t.sum()), ci=P.ci(1 - (Wb @ res_t) / (Wb @ sst_t))),
                           cos_oddgrad_Vb=dict(v=float(c.mean()), ci=P.ci((Wb @ c) / n), null=float(c0.mean()), diff_ci=P.ci((Wb @ (c - c0)) / n),
                                               p_vs_null=P.p_two((Wb @ (c - c0)) / n)),
                           _res=(res_e, sst_e, res_t, sst_t))
        base = fam['2.0']['_res']; ps = {}
        for bk in fam:
            if bk == '2.0': continue
            re_, se_, rt_, stt_ = fam[bk]['_res']
            de_b = (1 - (Wb @ re_) / (Wb @ se_)) - (1 - (Wb @ base[0]) / (Wb @ base[1]))
            dt_b = (1 - (Wb @ rt_) / (Wb @ stt_)) - (1 - (Wb @ base[2]) / (Wb @ base[3]))
            fam[bk]['R2_even_perjet_minus_b2'] = dict(v=fam[bk]['R2_even_perjet_scale']['v'] - fam['2.0']['R2_even_perjet_scale']['v'], ci=P.ci(de_b), p=P.p_two(de_b))
            fam[bk]['R2_total_perjet_minus_b2'] = dict(v=fam[bk]['R2_total_perjet_scale']['v'] - fam['2.0']['R2_total_perjet_scale']['v'], ci=P.ci(dt_b), p=P.p_two(dt_b))
            ps[bk] = fam[bk]['R2_even_perjet_minus_b2']['p']
        hp = P.holm(list(ps.values()))
        for bk, h in zip(ps, hp): fam[bk]['R2_even_perjet_minus_b2']['p_holm'] = h
        r['moment_family'] = fam
        # C: per-jet scale controls for the even part (b = 2)
        Pe2 = PB['2.0']['Pe']
        s_rot = np.sqrt(np.mean(np.stack([d[:, Ei[q]] for q in ('rot90', 'rot180', 'rot270')], 1) ** 2, 1))
        s_rd = np.sqrt(np.mean(np.stack([d[:, Ei[f'rd{q}']] for q in range(4)], 1) ** 2, 1))
        ctrl = {}
        for cn, s in dict(kappa_j=kap_j, kappa_other_jet=kap_j[perm], rms_rotation=s_rot, rms_random_translation=s_rd, none=one).items():
            X = s[:, None] * Pe2; bb = (X * even).sum() / (X * X).sum(); res = ((even - bb * X) ** 2).sum(1); sst = (even ** 2).sum(1)
            ctrl[cn] = dict(R2=float(1 - res.sum() / sst.sum()), ci=P.ci(1 - (Wb @ res) / (Wb @ sst)))
        r['scale_controls_even_b2'] = ctrl
        r['corr_kappa_j_vs_rms_rotation'] = float(np.corrcoef(kap_j, s_rot)[0, 1])
        OUT['models'][k] = r
        log(f'{k}: acc {OUT["acc"][k]:.4f}; ' + (f'odd share {r["A"]["odd_var_share"]["v"]:.3f} g_o {r["A"]["g_odd"]["v"]:.2f} g_e {r["A"]["g_even"]["v"]:.2f}; ' if 'A' in r else '')
            + 'even R2 per-jet scale by b: ' + ', '.join(f'{bk} {fam[bk]["R2_even_perjet_scale"]["v"]:.3f}' for bk in fam)
            + '; long-short obs ' + f'{r["observed_long_short"]["v"]:.3f}; ctrl ' + ', '.join(f'{c_} {v["R2"]:.2f}' for c_, v in ctrl.items()))
        with open(FN, 'w') as f: json.dump(P.strip(OUT), f, indent=1)
    # P1-sample accuracy for the two ParT models (unedited reference)
    try:
        from interp.part_core import sample
        J1 = sample(per_class=100 if SMOKE else 1000, start=3000); X1 = P.all_inputs(J1); y1 = J1['label'].astype(int); OUT['acc_P1_sample'] = {}
        for k in ('ParT_full', 'ParT_kin'):
            Lp = np.concatenate([P.run_model(M[k], {f: tuple(a_[s] for a_ in X1[f]) for f in X1}, torch) for s in [slice(j, j + P.CH) for j in range(0, len(y1), P.CH)]])
            OUT['acc_P1_sample'][k] = float((Lp.argmax(1) == y1).mean())
        log(f'P1 sample acc {OUT["acc_P1_sample"]}')
    except Exception as ex:                                            # noqa: BLE001
        log(f'P1 sample failed {ex!r}')
    with open(FN, 'w') as f: json.dump(P.strip(OUT), f, indent=1)
    log(f'wrote {FN}'); log('P23v DONE')


if __name__ == '__main__':
    main()
