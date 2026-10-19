"""P3v — verification of P3 (ParT_full W/Z mass circuit: b2h8 threshold, head routes into n98, literal circuit).

Fresh samples (JetClass val_5M, random rows, numpy default_rng(97); files not used by P3):
  T': 5,000 random rows each of WToQQ_122.root and ZToQQ_122.root (10,000 W/Z jets).
  S': 1,000 random rows of each class's <sample>_123.root (10,000 jets).
Model access: P3's Stage (exact re-implementation of the two class-attention blocks, checked against the model logits).

(1) Threshold on T': b2h8 class-token share vs jet mass, logistic fit (NFIT bootstrap refits), span per head over 50-130 GeV.
(2) Missing direct test (not in P3): per-head activation patching under the real edits (angles x0.9, x1.1, phi+0.06) on T'.
    removal  : edited jet, head h's context vector ctx_h (16-dim, before out_proj) set to its unedited value;
               removed share = 1 - mean(dz97_patched) / mean(dz97_edit).
    sufficiency: unedited jet, head h's ctx set to its edited value; share = mean(dz97_suff) / mean(dz97_edit).
    Sets: each of the 16 heads, block 1, block 2, all 16 (= 100 % by construction: the class token only reads the particles
    through ctx), {b2h8, b1h3}, all heads but b2h8.
    Control: the unedited jet with a random 16-dim direction of the same per-jet norm as the head's edit change added to ctx_h
    (NDIR draws per head), mean |dz97| vs the sufficiency patch's mean |dz97|.
    Calls: W/Z binary calls changed by the edit and the fraction restored by each removal patch (McNemar on correctness).
(3) On S': resample cal (P3's rule: ctx from a same-class jet with |dm| >= 10 GeV; slope of dz97 per 10 GeV / real-edit slope)
    for all 16, b1h3, b2h8; mean ablation calls changed (McNemar).
(4) Literal circuits on S' (uniform donor over S' without self): V_a {b2h8, b1h3} + n98 (MLP resampled), all heads + n98 (MLP
    resampled), V_e n98 alone; V_d heads only (MLP + read-out intact) for every one of the 120 head pairs and 16 single heads.

  ./part_job.sh c-p3-verify 'python -m interp.p3v_verify_part'   -> results/part/p3_verify.json
"""
import itertools, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.p3_wz_circuit_part import (Stage, load_rows, base_pass, eval_S, batches, boot_W, ci, p_two, holm, mcnemar, slope_stats,
                                        logistic, fit_logistic, clean, OUT, VAL, H, HD, C, B2H8, HEADS, lab)

SMOKE = '--smoke' in sys.argv
SEED = 97
PER_S, PER_T = (60, 300) if SMOKE else (1000, 5000)
NB = 100 if SMOKE else 500
NFIT = 20 if SMOKE else 100
NDIR = 2 if SMOKE else 5
NRAND = 5 if SMOKE else 30
B1H3 = (0, 2)
FN = OUT / ('p3_verify_smoke.json' if SMOKE else 'p3_verify.json')
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


def sets_():
    S = {lab(bh): [bh] for bh in HEADS}
    S.update(all16=list(HEADS), block1=[(0, h) for h in range(H)], block2=[(1, h) for h in range(H)], b2h8_b1h3=[B2H8, B1H3],
             all_but_b2h8=[bh for bh in HEADS if bh != B2H8])
    return S


def ctl_from(t, dev, ctx_src, hs):
    ctl = {}
    for b in range(2):
        sel = t.tensor([(b, h) in hs for h in range(H)], device=dev)
        if bool(sel.any()): ctl[('ctx', b)] = (ctx_src[b], sel)
    return ctl


def t_loop(st, JT, JE, N98, iW, iZ):
    from interp.part_core import sub
    t = st.t; N = len(JT['label']); SETS = sets_()
    share = np.zeros((N, 2, H), np.float32); shareE = {e: np.zeros((N, 2, H), np.float32) for e in JE}
    base = dict(z=np.zeros(N, np.float32), wz=np.zeros(N, np.float32))
    keys = ['edit'] + [f'{m}_{s}' for m in ('rm', 'suf') for s in SETS]
    R = {e: {k: dict(z=np.zeros(N, np.float32), wz=np.zeros(N, np.float32)) for k in keys} for e in JE}
    RD = {e: {lab(bh): np.zeros((NDIR, N), np.float32) for bh in HEADS} for e in JE}
    dn = {e: dict(ctx=np.zeros((N, 2, H), np.float32), out=np.zeros((N, 2, H), np.float32)) for e in JE}
    md = 0.0
    for bi, idx in enumerate(batches(JT, 256)):
        Xb, pb, Lm, _ = st.run_X(sub(JT, idx))
        with t.no_grad():
            zb, Lb, Rb = st.cls(Xb, pb, rec=True); md = max(md, float((Lb - Lm).abs().max()))
            share[idx] = t.stack(Rb['A'], 1)[..., 0].cpu().numpy()
            base['z'][idx] = zb[:, N98].cpu().numpy(); base['wz'][idx] = (Lb[:, iW] - Lb[:, iZ]).cpu().numpy()
            for e, J in JE.items():
                Xe, pe, Lme, _ = st.run_X(sub(J, idx)); assert bool((pe == pb).all())
                ze, Le, Re = st.cls(Xe, pe, rec=True); md = max(md, float((Le - Lme).abs().max()))
                shareE[e][idx] = t.stack(Re['A'], 1)[..., 0].cpu().numpy()

                def put(k, z, L):
                    R[e][k]['z'][idx] = z[:, N98].cpu().numpy(); R[e][k]['wz'][idx] = (L[:, iW] - L[:, iZ]).cpu().numpy()
                put('edit', ze, Le)
                for s, hs in SETS.items():
                    z, L, _ = st.cls(Xe, pe, ctl_from(t, st.dev, Rb['ctx'], hs)); put(f'rm_{s}', z, L)
                    z, L, _ = st.cls(Xb, pb, ctl_from(t, st.dev, Re['ctx'], hs)); put(f'suf_{s}', z, L)
                for b in range(2):
                    d = Re['ctx'][b] - Rb['ctx'][b]
                    dn[e]['ctx'][idx, b] = d.norm(dim=-1).cpu().numpy()
                    dn[e]['out'][idx, b] = t.einsum('nhd,chd->nhc', d, st.B[b]['Wo'].view(C, H, HD)).norm(dim=-1).cpu().numpy()
                    for h in range(H):
                        nrm = d[:, h].norm(dim=-1)
                        for k in range(NDIR):
                            g = t.Generator(device=st.dev).manual_seed(5000 + 101 * k + 7 * (b * H + h) + 100003 * bi + 7919 * list(JE).index(e))
                            r = t.randn(len(idx), HD, generator=g, device=st.dev); r = r / r.norm(dim=-1, keepdim=True) * nrm[:, None]
                            add = t.zeros(len(idx), H, HD, device=st.dev); add[:, h] = r
                            z, L, _ = st.cls(Xb, pb, {('ctx_add', b): add})
                            RD[e][lab((b, h))][k, idx] = (z[:, N98] - zb[:, N98]).cpu().numpy()
    log(f'T loop done; re-implementation max |dlogit| {md:.2e}')
    return dict(share=share, shareE=shareE, base=base, R=R, RD=RD, dn=dn, md=md)


def patch_analysis(TL, yT, mT, iW, iZ, sd97, rng):
    SETS = sets_(); base = TL['base']; out = {}
    sels = dict(WZ_70_110=(mT >= 70) & (mT <= 110), W=yT == iW, Z=yT == iZ)
    for e, Re in TL['R'].items():
        out[e] = {}
        for sn, sel in sels.items():
            n = int(sel.sum()); Wb = boot_W(n, NB, rng); row = dict(n=n)
            for key, k2, sc in (('z97', 'z', sd97), ('wz', 'wz', 1.0)):
                ed = ((Re['edit'][k2] - base[k2])[sel] / sc).astype(np.float64); bt = (Wb @ ed) / Wb.sum(1)
                r = dict(edit=float(ed.mean()), edit_ci=ci(bt), sets={}); BS = {}
                for s in SETS:
                    rm = ((Re[f'rm_{s}'][k2] - base[k2])[sel] / sc).astype(np.float64); su = ((Re[f'suf_{s}'][k2] - base[k2])[sel] / sc).astype(np.float64)
                    brm = 1 - ((Wb @ rm) / Wb.sum(1)) / bt; bsu = ((Wb @ su) / Wb.sum(1)) / bt; BS[s] = (brm, bsu)
                    r['sets'][s] = dict(removal=float(1 - rm.mean() / ed.mean()), removal_ci=ci(brm), removal_p0=p_two(brm),
                                        suff=float(su.mean() / ed.mean()), suff_ci=ci(bsu), suff_p0=p_two(bsu),
                                        mean_abs_suff=float(np.abs(su).mean()))
                for kk in ('removal_p0', 'suff_p0'):
                    for bh, pa in zip(HEADS, holm([r['sets'][lab(bh)][kk] for bh in HEADS])): r['sets'][lab(bh)][kk + '_holm'] = pa
                b8, b3 = BS['b2h8'], BS['b1h3']
                r['b2h8_p_removal_ge_0.5'] = float(max(1 / NB, (b8[0] >= 0.5).mean())); r['b2h8_p_suff_ge_0.5'] = float(max(1 / NB, (b8[1] >= 0.5).mean()))
                r['b1h3_minus_b2h8_removal'] = dict(value=r['sets']['b1h3']['removal'] - r['sets']['b2h8']['removal'], ci=ci(b3[0] - b8[0]), p0=p_two(b3[0] - b8[0]))
                if key == 'z97':
                    rd = {}
                    for bh in HEADS:
                        act = r['sets'][lab(bh)]['mean_abs_suff']; rv = np.abs(TL['RD'][e][lab(bh)][:, sel] / sd97).mean(1)
                        rd[lab(bh)] = dict(actual_mean_abs=act, random_mean_abs=rv.tolist(), p=float((1 + (rv >= act).sum()) / (1 + NDIR)))
                    r['random_direction_control'] = rd
                row[key] = r
            out[e][sn] = row
            z = row['z97']['sets']
            log(f'  patch {e} {sn}: dz97 {row["z97"]["edit"]:+.3f}; removal b2h8 {z["b2h8"]["removal"]:+.3f} {np.round(z["b2h8"]["removal_ci"], 3).tolist()} '
                f'b1h3 {z["b1h3"]["removal"]:+.3f} block1 {z["block1"]["removal"]:+.3f} block2 {z["block2"]["removal"]:+.3f} all16 {z["all16"]["removal"]:+.3f} '
                f'all_but_b2h8 {z["all_but_b2h8"]["removal"]:+.3f}; suff b2h8 {z["b2h8"]["suff"]:+.3f} b1h3 {z["b1h3"]["suff"]:+.3f}')
        # calls (all W/Z jets of T')
        cb = base['wz'] > 0; ce = Re['edit']['wz'] > 0; truthW = yT == iW; chg = cb != ce
        cr = dict(edit_changed=float(chg.mean()), n_changed=int(chg.sum()), sets={})
        for s in SETS:
            crm = Re[f'rm_{s}']['wz'] > 0
            cr['sets'][s] = dict(restored=float((crm[chg] == cb[chg]).mean()) if chg.any() else None,
                                 mcnemar_vs_edit=mcnemar(ce == truthW, crm == truthW))
        for bh, pa in zip(HEADS, holm([cr['sets'][lab(bh)]['mcnemar_vs_edit']['p'] for bh in HEADS])): cr['sets'][lab(bh)]['mcnemar_p_holm'] = pa
        out[e]['calls'] = cr
        out[e]['ctx_change_norm'] = {lab(bh): dict(ctx=float(TL['dn'][e]['ctx'][:, bh[0], bh[1]].mean()), after_out_proj=float(TL['dn'][e]['out'][:, bh[0], bh[1]].mean())) for bh in HEADS}
        log(f'  calls {e}: changed {cr["edit_changed"]:.4f}; restored by b2h8 {cr["sets"]["b2h8"]["restored"]}, b1h3 {cr["sets"]["b1h3"]["restored"]}, '
            f'block1 {cr["sets"]["block1"]["restored"]}, block2 {cr["sets"]["block2"]["restored"]}')
    return out


def threshold(share, shareE, mT, yT, iW, iZ, mE, rng):
    N = len(mT); res = dict(heads={}); edges = np.arange(40, 205, 5.0); cen = 0.5 * (edges[1:] + edges[:-1])
    inb = [(mT >= lo) & (mT < hi) for lo, hi in zip(edges[:-1], edges[1:])]; cnt = np.array([b.sum() for b in inb]); ok = (cen > 50) & (cen < 130) & (cnt >= 50)
    Wb = boot_W(N, NB, rng)
    for bh in HEADS:
        s = share[:, bh[0], bh[1]].astype(np.float64); mean = np.array([s[b].mean() if c >= 20 else np.nan for b, c in zip(inb, cnt)])
        bm = np.stack([(Wb @ (s * b)) / np.maximum(Wb @ b.astype(float), 1) for b in inb], 1)
        bspan = np.nanmax(bm[:, ok], 1) - np.nanmin(bm[:, ok], 1)
        res['heads'][lab(bh)] = dict(span_50_130=float(np.nanmax(mean[ok]) - np.nanmin(mean[ok])), span_ci=ci(bspan),
                                     p_span_le_20pt=float(max(1 / NB, (bspan <= 0.20).mean())), mean=mean.tolist(),
                                     dshare={e: float((shareE[e][:, bh[0], bh[1]] - s).mean()) for e in shareE})
    for bh, pa in zip(HEADS, holm([res['heads'][lab(bh)]['p_span_le_20pt'] for bh in HEADS])): res['heads'][lab(bh)]['p_span_le_20pt_holm'] = pa
    res['bins'] = cen.tolist(); res['counts'] = cnt.tolist(); fits = {}
    for nm, sel0 in (('WZ', np.ones(N, bool)), ('W', yT == iW), ('Z', yT == iZ)):
        sel = sel0 & (mT >= 50) & (mT <= 140); m = mT[sel].astype(np.float64); s = share[sel, 1, 7].astype(np.float64)
        p = fit_logistic(m, s); bp = []
        for _ in range(NFIT):
            ii = rng.integers(0, len(m), len(m))
            try: bp.append(fit_logistic(m[ii], s[ii]))
            except Exception: pass
        bp = np.array(bp); fits[nm] = dict(n=int(sel.sum()), lo=float(p[0]), hi=float(p[1]), m50=float(p[2]), width=float(p[3]), m50_ci=ci(bp[:, 2]), width_ci=ci(bp[:, 3]))
        log(f'  b2h8 logistic {nm}: {fits[nm]}')
    res['logistic_b2h8'] = fits
    f = fits['WZ']; res['curve_prediction'] = {e: float((logistic(me, f['lo'], f['hi'], f['m50'], f['width']) - logistic(mT, f['lo'], f['hi'], f['m50'], f['width'])).mean()) for e, me in mE.items()}
    return res


def s_part(st, D, y, mS, iW, iZ, N98, ref, rng):
    t = st.t; N = len(y); wz = np.isin(y, [iW, iZ]); nwz = int(wz.sum()); sd97 = float(D['z'][:, N98].std())
    out = dict(sd97=sd97)
    don = np.zeros(N, int)
    for c in np.unique(y):
        ix = np.where(y == c)[0]
        for i in ix:
            cand = ix[np.abs(mS[ix] - mS[i]) >= 10.0]; don[i] = rng.choice(cand) if len(cand) else rng.choice(ix[ix != i])
    dT = t.from_numpy(don).to(st.dev); dm = (mS[don] - mS) / 10.0; Wb = boot_W(nwz, NB, rng)
    ctx_mean = D['T']['ctx'].mean(0); b0 = np.where(D['L'][:, iW] > D['L'][:, iZ], iW, iZ)
    selm = lambda hs, b: t.tensor([(b, h) in hs for h in range(H)], device=st.dev)
    for g, hs in (('all16', HEADS), ('b1h3', [B1H3]), ('b2h8', [B2H8])):
        Z, L = eval_S(st, D, lambda it: {('ctx', b): (D['T']['ctx'][dT[it]][:, b], selm(hs, b)) for b in range(2) if any(bb == b for bb, _ in hs)})
        dz = (Z[:, N98] - D['z'][:, N98]) / sd97; s, bs = slope_stats(Wb, dz[wz].astype(np.float64), dm[wz])
        Zm, Lm = eval_S(st, D, lambda it: {('ctx', b): (ctx_mean[b][None].expand(len(it), H, HD), selm(hs, b)) for b in range(2) if any(bb == b for bb, _ in hs)})
        b1 = np.where(Lm[:, iW] > Lm[:, iZ], iW, iZ)
        out[g] = dict(resample_slope_z97=s, slope_ci=ci(bs), cal_z97=s / ref, cal_ci=ci(bs / ref),
                      mean_abl_wz_calls_changed=float((b0[wz] != b1[wz]).mean()), mean_abl_mcnemar=mcnemar(b0[wz] == y[wz], b1[wz] == y[wz]))
        log(f'  S\' {g}: resample cal {s / ref:+.4f} {np.round(np.array(ci(bs / ref)), 4).tolist()}; mean-abl W/Z calls changed {out[g]["mean_abl_wz_calls_changed"]:.4f}')
    return out


def circuits(st, D, y, iW, iZ, N98, rng):
    from interp.part_core import ACTIVE
    t = st.t; N = len(y); wz = np.isin(y, [iW, iZ])
    don = rng.integers(0, N - 1, N); don = don + (don >= np.arange(N)); dT = t.from_numpy(don).to(st.dev)

    def run(heads, mlp_keep, z_keep):
        hm = [t.tensor([(b, h) not in heads for h in range(H)], device=st.dev) for b in range(2)]
        mm = [t.from_numpy(~mlp_keep[b]).to(st.dev) for b in range(2)]; zm = t.from_numpy(~z_keep).to(st.dev)

        def mk(it):
            d = dT[it]; ctl = {}
            for b in range(2):
                if bool(hm[b].any()): ctl[('ctx', b)] = (D['T']['ctx'][d][:, b], hm[b])
                if bool(mm[b].any()): ctl[('mlp', b)] = (D['T']['h'][d][:, b], mm[b])
            if bool(zm.any()): ctl['z'] = (D['T']['z'][d], zm)
            return ctl
        _, L = eval_S(st, D, mk); bw = np.where(L[:, iW] > L[:, iZ], iW, iZ)
        return dict(acc10=float((L.argmax(1) == y).mean()), wz_binary_acc=float((bw[wz] == y[wz]).mean()))
    allm = np.ones((2, 512), bool); nom = np.zeros((2, 512), bool); allz = np.ones(C, bool); z97 = np.zeros(C, bool); z97[N98] = True
    Hc = [B2H8, B1H3]; others = [a for a in ACTIVE if a != N98]
    res = dict(unedited=dict(acc10=float((D['L'].argmax(1) == y).mean()), wz_binary_acc=float((np.where(D['L'][:, iW] > D['L'][:, iZ], iW, iZ)[wz] == y[wz]).mean())))
    res['V_a_b2h8_b1h3_n98'] = run(Hc, nom, z97); res['allheads_mlpres_n98'] = run(HEADS, nom, z97); res['V_e_n98'] = run(HEADS, allm, z97)
    res['V_d_b2h8_b1h3'] = run(Hc, allm, allz); res['unedited_through_code'] = run(HEADS, allm, allz)
    rnd = []
    for k in range(NRAND):
        sel = rng.choice(16, 2, replace=False); rz = np.zeros(C, bool); rz[rng.choice(others)] = True
        rnd.append(run([HEADS[i] for i in sel], nom, rz)['wz_binary_acc'])
    rnd = np.array(rnd); res['V_a_random'] = dict(mean=float(rnd.mean()), max=float(rnd.max()), p=float((1 + (rnd >= res['V_a_b2h8_b1h3_n98']['wz_binary_acc']).sum()) / (1 + len(rnd))))
    pairs = {}
    for a, b in itertools.combinations(range(16), 2):
        pairs[f'{lab(HEADS[a])}+{lab(HEADS[b])}'] = run([HEADS[a], HEADS[b]], allm, allz)['wz_binary_acc']
    singles = {lab(bh): run([bh], allm, allz)['wz_binary_acc'] for bh in HEADS}
    pv = np.array(list(pairs.values())); me = res['V_d_b2h8_b1h3']['wz_binary_acc']
    res['V_d_pair_scan'] = dict(rank_of_b2h8_b1h3=int((pv > me).sum() + 1), n_pairs=len(pv), median=float(np.median(pv)),
                                top8=sorted(pairs.items(), key=lambda kv: -kv[1])[:8], all=pairs)
    res['V_d_single_scan'] = dict(sorted(singles.items(), key=lambda kv: -kv[1]))
    log(f'  circuits: {json.dumps({k: v for k, v in res.items() if k not in ("V_d_pair_scan", "V_d_single_scan")})}')
    log(f'  V_d pair scan: rank of b2h8+b1h3 {res["V_d_pair_scan"]["rank_of_b2h8_b1h3"]}/120; top {res["V_d_pair_scan"]["top8"][:5]}; singles {list(res["V_d_single_scan"].items())[:5]}')
    return res


def main():
    from interp.part_core import kin, build, jmass, cat, sample, CL, N98, PRE
    OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED)
    iW, iZ = CL.index('Wqq'), CL.index('Zqq'); res = dict(seed=SEED, smoke=SMOKE, NB=NB, NFIT=NFIT, NDIR=NDIR, NRAND=NRAND, samples={})
    parts = []
    for c in CL:
        J, rows, Nf = load_rows(VAL / f'{PRE[c]}_123.root', PER_S, rng); parts.append(J); res['samples'][c] = dict(file=f'{PRE[c]}_123.root', n=len(rows), first_rows=rows[:5].tolist())
        assert np.all(J['label'] == CL.index(c))
    J0 = cat(parts); y = J0['label'].astype(int); JS = build(J0, *kin(J0)); mS = jmass(JS); del parts, J0
    tp = []
    for c in ('Wqq', 'Zqq'):
        J, rows, Nf = load_rows(VAL / f'{PRE[c]}_122.root', PER_T, rng); tp.append(J); res['samples'][f'T_{c}'] = dict(file=f'{PRE[c]}_122.root', n=len(rows), first_rows=rows[:5].tolist())
    T0j = cat(tp); yT = T0j['label'].astype(int); pt, de, dp, ms = kin(T0j); real = pt > 0
    JT = build(T0j, pt, de, dp, ms); mT = jmass(JT)
    JE = {'ang0.9': build(T0j, pt, de * 0.9, dp * 0.9, ms), 'ang1.1': build(T0j, pt, de * 1.1, dp * 1.1, ms), 'phi0.06': build(T0j, pt, de, dp + 0.06 * real, ms)}
    mE = {e: jmass(J) for e, J in JE.items()}
    res['edit_truth'] = {e: dict(dm_mean=float((m - mT).mean()), max_abs_dm=float(np.abs(m - mT).max()), mask_equal=bool(np.array_equal(JE[e]['mask'], JT['mask']))) for e, m in mE.items()}
    log(f'S\' {len(y)} jets, T\' {len(yT)} jets; edit truth {res["edit_truth"]}')
    st = Stage('full')
    JP = sample(100 if SMOKE else 1000, 3000); JPb = build(JP, *kin(JP)); RP = st.net.run(JPb)
    res['P1_sample_acc'] = float((RP['logits'].argmax(1) == JP['label']).mean()); del JP, JPb, RP
    D = base_pass(st, JS, mS); sd97 = float(D['z'][:, N98].std())
    res['S_acc10'] = float((D['L'].argmax(1) == y).mean()); res['reimpl_S'] = dict(max_dlogit=D['reimpl_max_dlogit'], calls_changed=D['reimpl_calls_changed'])
    log(f'P1-sample acc {res["P1_sample_acc"]:.4f}; S\' acc {res["S_acc10"]:.4f}; reimpl {res["reimpl_S"]}')
    TL = t_loop(st, JT, JE, N98, iW, iZ); res['reimpl_T_max_dlogit'] = TL['md']
    res['threshold'] = threshold(TL['share'], TL['shareE'], mT, yT, iW, iZ, mE, np.random.default_rng(SEED + 1))
    th = res['threshold']['heads']; log('  spans: ' + ', '.join(f'{k} {v["span_50_130"]:.3f}' for k, v in sorted(th.items(), key=lambda kv: -kv[1]['span_50_130'])[:4]))
    log('  b2h8 dshare: ' + str(th['b2h8']['dshare']) + ' curve prediction ' + str(res['threshold']['curve_prediction']))
    dm = (mE['ang1.1'] - mT) / 10.0; dz = (TL['R']['ang1.1']['edit']['z'] - TL['base']['z']) / sd97
    ref = float((dz * dm).sum() / (dm * dm).sum()); res['ref_slope_z97_ang1.1_per10GeV'] = ref; log(f'  real-edit slope dz97 per 10 GeV {ref:+.4f}')
    res['patch'] = patch_analysis(TL, yT, mT, iW, iZ, sd97, np.random.default_rng(SEED + 2)); del TL
    res['S_resample'] = s_part(st, D, y, mS, iW, iZ, N98, ref, np.random.default_rng(SEED + 3))
    res['circuits'] = circuits(st, D, y, iW, iZ, N98, np.random.default_rng(SEED + 4))
    json.dump(clean(res), open(FN, 'w'), indent=1); log(f'wrote {FN}'); log('P3v DONE')


if __name__ == '__main__':
    main()
