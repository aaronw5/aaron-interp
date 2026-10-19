"""P3v2 — supplementary verification of P3: the particle-block far-side head patch with a stratified null, and the G21b
attention/value decomposition, on a fresh sample.

P3 compared the far-side set (heads 4 and 8 of each of the 8 particle blocks, 16 heads, exactly 2 per block) against 20 random
16-of-64 sets (not stratified by block), p = 0.048 at the 20-draw floor; P3v did not re-run it, nor G21b.

Sample: JetClass val_5M, WToQQ_124.root and ZToQQ_124.root (files not used by P3/P3v), NPER random rows each,
numpy default_rng(113); no first-k. Build part_core.build (sign(jet_eta) convention). Edits: angles x 1.1 (deta, dphi x 1.1; mass
x ~1.1) and phi + 0.06 rad on real particles (invariant mass unchanged).
Patch (P3's G103-style hook, P3's Stage._wrap): run the edited jet with the outputs (A_ih V_ih) W_o,ih of the heads in a set
replaced by the unedited jet's values; removed share = 1 - mean(dz97_patched) / mean(dz97_edit), per class (W, Z).
Sets: far side {(i,3),(i,7)}; all 64; per head-index columns {(i,h) : i = 0..7} for h = 0..7; NR stratified random sets (2 random
heads per block, != far side); NR unstratified random 16-of-64 sets. p = (1 + #null >= far) / (1 + NR). Bootstrap over jets (NB).
G21b (class stage, P3's Stage.cls): dz97 = attention part (edited class-attention weights, base embeddings) + value part (base
weights, edited embeddings) + interaction; per head: that head's weights only (b2h8, b1h3); on jets with base mass 70-110 GeV.

  ./part_job.sh c-p3-verify2 'python -m interp.p3v2_farside_part'   -> results/part/p3_verify2_farside.json
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.p3_wz_circuit_part import (Stage, load_rows, batches, boot_W, ci, p_two, clean, OUT, VAL, H, HEADS, B2H8, lab)

SMOKE = '--smoke' in sys.argv
SEED = 113
NPER = 150 if SMOKE else 2000
NB = 100 if SMOKE else 500
NR = 4 if SMOKE else 60
FN = OUT / ('p3_verify2_farside_smoke.json' if SMOKE else 'p3_verify2_farside.json')
FAR = [(i, h) for i in range(8) for h in (3, 7)]
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


def main():
    from interp.part_core import kin, build, jmass, sub, cat, CL, N98, PRE
    rng = np.random.default_rng(SEED); iW, iZ = CL.index('Wqq'), CL.index('Zqq'); OUT.mkdir(parents=True, exist_ok=True)
    res = dict(seed=SEED, smoke=SMOKE, NPER=NPER, NB=NB, NR=NR, samples={})
    tp = []
    for c in ('Wqq', 'Zqq'):
        J, rows, Nf = load_rows(VAL / f'{PRE[c]}_124.root', NPER, rng); tp.append(J)
        res['samples'][c] = dict(file=f'{PRE[c]}_124.root', entries=Nf, n=len(rows), first_rows=rows[:5].tolist())
    J0 = cat(tp); y = J0['label'].astype(int); pt, de, dp, ms = kin(J0); real = pt > 0
    JT = build(J0, pt, de, dp, ms); mT = jmass(JT)
    JE = {'ang1.1': build(J0, pt, de * 1.1, dp * 1.1, ms), 'phi0.06': build(J0, pt, de, dp + 0.06 * real, ms)}
    res['edit_truth'] = {e: dict(dm_mean=float((jmass(J) - mT).mean()), max_abs_dm=float(np.abs(jmass(J) - mT).max()),
                                 mask_equal=bool(np.array_equal(J['mask'], JT['mask']))) for e, J in JE.items()}
    log('edit truth', res['edit_truth'])
    allp = [(i, h) for i in range(8) for h in range(H)]
    sets = dict(far_side_h4_h8=FAR, all64=allp)
    for h in range(H): sets[f'column_h{h + 1}'] = [(i, h) for i in range(8)]
    for k in range(NR):
        while True:
            s = [(i, int(h)) for i in range(8) for h in rng.choice(H, 2, replace=False)]
            if set(s) != set(FAR): break
        sets[f'strat_{k}'] = s
    for k in range(NR):
        sets[f'unstrat_{k}'] = [allp[i] for i in rng.choice(64, 16, replace=False)]
    st = Stage('full'); t = st.t; N = len(y); ED = list(JE)
    out = {e: {s: np.zeros(N, np.float32) for s in ['edit'] + list(sets)} for e in ED}
    outwz = {e: {s: np.zeros(N, np.float32) for s in ['edit'] + list(sets)} for e in ED}
    dparts = ['full', 'attn', 'value', 'attn_b2h8', 'attn_b1h3']
    dec = {e: {p: np.zeros(N, np.float32) for p in dparts} for e in ED}
    z0 = np.zeros(N, np.float32); L0 = np.zeros((N, 10), np.float32); md = 0.0; st.recon = 0.0
    for bi, idx in enumerate(batches(JT, 96)):
        st.pm = 'record'; Xb, pb, Lb, zb = st.run_X(sub(JT, idx))
        with t.no_grad():
            zc, Lc, Rb = st.cls(Xb, pb, rec=True)
        md = max(md, float((Lc - Lb).abs().max())); z0[idx] = zb[:, N98].cpu().numpy(); L0[idx] = Lb.cpu().numpy()
        for e in ED:
            st.pm = None; Xe, pe, Le, ze = st.run_X(sub(JE[e], idx)); assert bool((pe == pb).all())
            out[e]['edit'][idx] = (ze[:, N98] - zb[:, N98]).cpu().numpy(); outwz[e]['edit'][idx] = ((Le[:, iW] - Le[:, iZ]) - (Lb[:, iW] - Lb[:, iZ])).cpu().numpy()
            with t.no_grad():
                _, _, Re = st.cls(Xe, pe, rec=True)
                allh = t.ones(H, dtype=t.bool, device=st.dev)
                runs = dict(full=(Xe, pe, {}), attn=(Xb, pb, {('attn', b): (Re['A'][b], allh) for b in range(2)}),
                            value=(Xe, pe, {('attn', b): (Rb['A'][b], allh) for b in range(2)}))
                for nm, (b, h) in (('attn_b2h8', B2H8), ('attn_b1h3', (0, 2))):
                    sel = t.tensor([hh == h for hh in range(H)], device=st.dev); runs[nm] = (Xb, pb, {('attn', b): (Re['A'][b], sel)})
                for p, (X, pd, ctl) in runs.items():
                    z, _, _ = st.cls(X, pd, ctl); dec[e][p][idx] = (z[:, N98] - zc[:, N98]).cpu().numpy()
            for s, hs in sets.items():
                st.pm = hs; _, _, Lp, zp = st.run_X(sub(JE[e], idx))
                out[e][s][idx] = (zp[:, N98] - zb[:, N98]).cpu().numpy(); outwz[e][s][idx] = ((Lp[:, iW] - Lp[:, iZ]) - (Lb[:, iW] - Lb[:, iZ])).cpu().numpy()
        st.pm = None; st.rec = {}
        if bi % 10 == 0: log(f'batch {bi}: {len(idx)} jets, recon err {st.recon:.2e}, reimpl {md:.2e}')
    sd97 = float(z0.std()); bw = np.where(L0[:, iW] > L0[:, iZ], iW, iZ)
    res.update(recon_max_err=st.recon, reimpl_max_dlogit=md, sd97_on_sample=sd97, wz_binary_acc=float((bw == y).mean()),
               acc10=float((L0.argmax(1) == y).mean()), sets={s: [f'p{i + 1}h{h + 1}' for i, h in hs] for s, hs in sets.items()})
    log(f'W/Z binary acc {res["wz_binary_acc"]:.4f}; reimpl {md:.2e}; recon {st.recon:.2e}')
    brng = np.random.default_rng(SEED + 1); Wb = boot_W(N, NB, brng); res['patch'] = {}
    for e in ED:
        for key, O in (('z97', out[e]), ('wz', outwz[e])):
            for cn, c in (('W', iW), ('Z', iZ), ('WZ', None)):
                s = np.ones(N, bool) if c is None else (y == c); Wc = Wb[:, s]; ed = O['edit'][s].astype(np.float64)
                bt = (Wc @ ed) / Wc.sum(1); r = dict(n=int(s.sum()), edit=float(ed.mean()) / (sd97 if key == 'z97' else 1), edit_ci=ci(bt / (sd97 if key == 'z97' else 1)))
                rem = {}
                for sn in sets:
                    x = O[sn][s].astype(np.float64); rem[sn] = 1 - x.mean() / ed.mean()
                    if not sn.startswith(('strat_', 'unstrat_')):
                        brem = 1 - ((Wc @ x) / Wc.sum(1)) / bt; r[sn] = dict(removed=float(rem[sn]), ci=ci(brem))
                far = rem['far_side_h4_h8']; xf = O['far_side_h4_h8'][s].astype(np.float64); bfar = 1 - ((Wc @ xf) / Wc.sum(1)) / bt
                for nul in ('strat', 'unstrat'):
                    v = np.array([rem[f'{nul}_{k}'] for k in range(NR)])
                    r[f'null_{nul}'] = dict(mean=float(v.mean()), sd=float(v.std()), p95=float(np.percentile(v, 95)), max=float(v.max()), min=float(v.min()),
                                            p_far_ge=float((1 + (v >= far).sum()) / (1 + NR)), ratio_far_to_mean=float(far / v.mean()) if v.mean() != 0 else None,
                                            far_minus_mean_ci=ci(bfar - v.mean()))
                r['columns_ranked'] = sorted([[f'h{h + 1}', float(rem[f'column_h{h + 1}'])] for h in range(H)], key=lambda a: -a[1])
                res['patch'][f'{e}_{key}_{cn}'] = r
                if key == 'z97':
                    log(f'{e} {key} {cn}: edit {r["edit"]:+.4f}; far {far:.3f} {np.round(r["far_side_h4_h8"]["ci"], 3).tolist()}; all64 {rem["all64"]:.3f}; '
                        f'strat null mean {r["null_strat"]["mean"]:.3f} max {r["null_strat"]["max"]:.3f} p {r["null_strat"]["p_far_ge"]:.3f}; '
                        f'unstrat mean {r["null_unstrat"]["mean"]:.3f} p {r["null_unstrat"]["p_far_ge"]:.3f}; columns {r["columns_ranked"]}')
    sel = (mT >= 70) & (mT <= 110); n = int(sel.sum()); Wd = boot_W(n, NB, brng); res['g21b'] = dict(n=n, window='base mass 70-110 GeV')
    for e in ED:
        f = dec[e]['full'][sel] / sd97; a = dec[e]['attn'][sel] / sd97; v = dec[e]['value'][sel] / sd97; it = f - a - v
        bt = (Wd @ f) / Wd.sum(1); r = {}
        for nm, x in (('total', f), ('attention', a), ('value', v), ('interaction', it), ('attn_b2h8', dec[e]['attn_b2h8'][sel] / sd97),
                      ('attn_b1h3', dec[e]['attn_b1h3'][sel] / sd97)):
            bx = (Wd @ x) / Wd.sum(1); r[nm] = dict(mean=float(x.mean()), ci=ci(bx), share=float(x.mean() / f.mean()), share_ci=ci(bx / bt))
        res['g21b'][e] = r
        log(f'G21b {e}: total {r["total"]["mean"]:+.4f}; attention share {r["attention"]["share"]:.3f} {np.round(r["attention"]["share_ci"], 3).tolist()}; '
            f'value {r["value"]["share"]:.3f}; interaction {r["interaction"]["share"]:.3f}; b2h8 attn {r["attn_b2h8"]["share"]:.3f}; b1h3 attn {r["attn_b1h3"]["share"]:.3f}')
    json.dump(clean(res), open(FN, 'w'), indent=1); log(f'wrote {FN}'); log('P3v2 DONE')


if __name__ == '__main__':
    main()
